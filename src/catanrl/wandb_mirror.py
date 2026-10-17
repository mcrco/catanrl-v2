"""A Weights & Biases mirror of a run (DESIGN 6.10: W&B mirrors and never
decides).

The run directory is the record; this module copies it to wandb, live from
the :class:`~catanrl.runs.RunWriter` and after the fact through
``tools/wandb_backfill.py``, in one layout under one run id, so a run
mirrored either way (or both, in turn) looks the same:

* one history row per ``metrics.jsonl`` row, its numeric keys as logged
  (``launch``, ``step`` and strings dropped) at ``_step = agent_steps``;
* the select curve, from the run's select eval records, as
  ``select/<suite> vs <opponents>/{score_rate, wilson_low, wilson_high,
  seat<k>, games}`` at the checkpoint's ``agent_steps`` (the row of that
  epoch; ``e0`` gets a row of its own);
* the finals, every other record of the run, as summary keys
  ``final/<suite>/<lineup>`` (the score rate) and a ``final_evals`` table;
* ``config``: the run's config plus ``experiment``, ``variant`` (the ref),
  ``config_hash``, ``steps_per_epoch`` and a compact ``_provenance`` per
  launch; ``group`` the experiment, ``job_type`` the kind, and tags derived
  from the lineage (DESIGN 6.4: never read back);
* ``summary[backfilled_through_agent_steps]``: the last mirrored step, so a
  backfill or a relaunch continues where the mirror stopped.

The mirror is on when the experiment's ``meta.wandb.project`` (``meta`` is
unhashed) or ``CATANRL_WANDB_PROJECT`` is set, and off under
``WANDB_MODE=disabled`` or without the package. The wandb run id derives from
the run's ref and config hash, so a relaunch resumes the same wandb run.
Every wandb call runs on one worker thread fed by a queue, and every failure
there only warns: the mirror never changes a run's outcome, timing, hashes or
determinism, and nothing in ``catanrl.train`` knows it exists.

Known limit: wandb keeps the first row logged at a step and drops rows below
its current step. After a crash the trainer redoes the epochs since its last
commit, whose rows (from the crashed launch) stay in wandb; ``Run.metrics()``
hides them in the record. A select result of the last mirrored step that the
mirror did not live to log is skipped by a later sync for the same reason.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import re
import threading
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from catanrl import canon

#: Rows through this ``agent_steps`` are in wandb (kept by the live mirror and the backfill).
THROUGH_KEY = "backfilled_through_agent_steps"
FINAL_TABLE = "final_evals"
FINAL_COLUMNS = ["suite", "lineup", "games", "score_rate", "wilson_low", "wilson_high", "expected", "truncated",
                 "fair", "eval_id"]
LIVE_NOTES = "Mirrored live by the trainer from runs/<experiment>/<variant> (metrics.jsonl, eval records)."
BACKFILL_NOTES = "Backfilled once from runs/<experiment>/<variant> (metrics.jsonl, eval records)."
#: How long ``finish`` waits for the worker to drain its queue and close the wandb run.
FINISH_TIMEOUT_S = 120.0
#: metrics.jsonl bookkeeping, not metrics.
_ROW_SKIP = {"launch", "step"}
_ID_BAD = re.compile(r"[^A-Za-z0-9_.-]")
_EPOCH_RE = re.compile(r"e(\d+)")


def _warn(message: str) -> None:
    """A warning that cannot raise, whatever the warning filters (the worker thread relies on it)."""
    try:
        warnings.warn(f"wandb mirror: {message}", RuntimeWarning, stacklevel=2)
    except Exception:  # noqa: BLE001  (-W error: the warning must still reach the terminal)
        import sys

        print(f"wandb mirror: {message}", file=sys.stderr, flush=True)


def import_wandb() -> ModuleType | None:
    """The wandb package, or None when it is missing or broken (the mirror is then off)."""
    try:
        import wandb
    except Exception:  # noqa: BLE001  (an import error of any kind means no mirror, not a failed run)
        return None
    return wandb


# ---------------------------------------------------------------- settings and identity


@dataclass(frozen=True)
class Settings:
    project: str
    entity: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()


def settings_for(config: Mapping[str, Any], environ: Mapping[str, str] | None = None) -> Settings | None:
    """The mirror's settings from the config's ``meta.wandb`` (a project name, or
    ``{project, entity, group, tags}``) and the environment, or None when it is off.

    ``CATANRL_WANDB_PROJECT`` and ``CATANRL_WANDB_ENTITY`` override the YAML (a
    smoke run into a scratch project); ``WANDB_MODE=disabled`` turns it off."""
    env = os.environ if environ is None else environ
    if env.get("WANDB_MODE", "").strip().lower() == "disabled":
        return None
    meta = (config.get("meta") or {}).get("wandb") or {}
    if isinstance(meta, str):
        meta = {"project": meta}
    project = env.get("CATANRL_WANDB_PROJECT") or meta.get("project")
    if not project:
        return None
    return Settings(project=str(project), entity=env.get("CATANRL_WANDB_ENTITY") or meta.get("entity") or None,
                    group=meta.get("group") or None, tags=tuple(str(t) for t in meta.get("tags") or ()))


def run_id(ref: str, config_hash: str) -> str:
    """``<experiment>-<variant>-<hash8>`` in wandb's id charset: the same for every launch
    and every backfill of a run. ``config_hash`` is the manifest's (the hash at creation):
    ``Run.config_hash`` is recomputed after migrations and moves when the schema does."""
    return _ID_BAD.sub("-", f"{ref.replace('/', '-')}-{config_hash[:8]}")


def stored_hash(run: Any) -> str:
    return str(run.manifest["config_hash"])


def lineage_tags(run: Any) -> list[str]:
    """The experiment, then ``from:<run>`` for every ancestor the bound ``init_from`` refs
    name (breadth first, without loading a checkpoint)."""
    from catanrl.runs import RunError, load_experiment, parse_ref

    tags = [str(run.manifest["experiment"])]
    pending, seen = [run], {run.ref}
    while pending:
        current = pending.pop(0)
        for text in ((current.manifest.get("resolved") or {}).get("init_from") or {}).values():
            try:
                parent, _, _ = parse_ref(str(text))
            except RunError:
                continue
            if parent in seen:
                continue
            seen.add(parent)
            tags.append(f"from:{parent}")
            with contextlib.suppress(Exception):  # a parent that is gone still names itself in the tag
                pending.append(load_experiment(parent))
    return [tag[:64] for tag in tags]  # wandb's tag length


# ---------------------------------------------------------------- layout


def history_row(row: Mapping[str, Any]) -> dict:
    """A metrics.jsonl row as wandb logs it: numbers only, bookkeeping keys out."""
    return {k: v for k, v in row.items() if k not in _ROW_SKIP and isinstance(v, (bool, int, float))}


def steps_per_epoch(rows: Iterable[Mapping[str, Any]]) -> int | None:
    for row in rows:
        epoch, steps = row.get("epoch"), row.get("agent_steps")
        if epoch and steps:
            return int(steps) // int(epoch)
    return None


def is_select(record: Any) -> bool:
    return ((record.key.get("suite") or {}).get("purpose")) == "select"


def record_epoch(run: Any, record: Any) -> int | None:
    """The epoch of ``run``'s checkpoint in the record's slot 0, or None when slot 0 is
    not an epoch ref of this run."""
    from catanrl.runs import RunError, parse_ref

    refs = record.data.get("refs") or [None]
    if not refs[0]:
        return None
    try:
        run_ref, which, _ = parse_ref(str(refs[0]))
    except RunError:
        return None
    m = _EPOCH_RE.fullmatch(which or "")
    return int(m[1]) if run_ref == run.ref and m else None


def select_label(record: Any) -> str:
    """``select-4p vs F F F``: the suite and the opponents' labels."""
    return f"{record.data['suite']} vs {' '.join(record.data['labels'][1:])}"


def select_row(record: Any) -> dict:
    s = record.summary
    prefix = f"select/{select_label(record)}"
    out = {f"{prefix}/score_rate": s["score_rate"], f"{prefix}/wilson_low": s["wilson"][0],
           f"{prefix}/wilson_high": s["wilson"][1], f"{prefix}/games": s["games"]}
    for seat, rate in (s["slots"][0].get("per_seat") or {}).items():
        out[f"{prefix}/seat{seat}"] = rate
    return out


def final_summary(records: Iterable[Any]) -> dict:
    """``final/<suite>/<lineup>`` -> score rate (records in order: a repeated label keeps the last)."""
    return {f"final/{r.data['suite']}/{' vs '.join(r.data['labels'])}": r.summary["score_rate"] for r in records}


def final_table_rows(records: Iterable[Any]) -> list[list]:
    rows = []
    for r in records:
        s, slot = r.summary, r.summary["slots"][0]
        rows.append([r.data["suite"], " vs ".join(r.data["labels"]), s["games"], s["score_rate"], s["wilson"][0],
                     s["wilson"][1], slot.get("expected"), s.get("truncated", 0), json.dumps(r.data.get("fair")),
                     r.eval_id])
    return rows


def provenance_summary(prov: Mapping[str, Any]) -> dict:
    """What a reader of the wandb config wants of a launch's provenance capture."""
    repos = (prov.get("git") or {}).get("repos") or {}
    build = (prov.get("engine") or {}).get("build") or {}
    torch_ = prov.get("torch") or {}
    return {"git": {name: {k: repo.get(k) for k in ("head", "dirty", "branch")} for name, repo in repos.items()},
            "argv": (prov.get("run") or {}).get("argv"), "time": prov.get("time"),
            "torch": {k: torch_.get(k) for k in ("cuda", "gpus", "torch")},
            "engine_build": {k: build.get(k) for k in ("git_sha", "compiler", "git_dirty", "compiler_version")},
            "uv_lock_sha256": (prov.get("python") or {}).get("uv_lock_sha256")}


def provenance_config(run: Any) -> dict:
    out = {}
    paths = sorted((run.path / "provenance").glob("launch*.json"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
    for path in paths:
        try:
            out[path.stem] = provenance_summary(canon.read_json(path))
        except (OSError, ValueError):  # being written: the next sync has it
            continue
    return out


def wandb_config(run: Any, *, steps_per_epoch: int | None = None) -> dict:
    """The config as launched (the manifest's, not the migrated one) plus the run's identity."""
    return {**run.manifest["config"], "experiment": run.manifest["experiment"], "variant": run.ref,
            "config_hash": stored_hash(run), "steps_per_epoch": steps_per_epoch, "_provenance": provenance_config(run)}


def init_kwargs(run: Any, settings: Settings, *, tags: Iterable[str] = (), notes: str) -> dict:
    """``wandb.init``'s arguments for ``run``: the stable id, resumable."""
    kwargs = {"project": settings.project, "id": run_id(run.ref, stored_hash(run)), "name": run.ref,
              "group": settings.group or run.manifest["experiment"], "job_type": run.config.get("kind", "train"),
              "tags": list(dict.fromkeys([*tags, *lineage_tags(run), *settings.tags])), "notes": notes,
              "config": wandb_config(run, steps_per_epoch=steps_per_epoch(run.metrics())), "resume": "allow"}
    if settings.entity:
        kwargs["entity"] = settings.entity
    if not os.environ.get("WANDB_DIR"):  # wandb's own files next to the runs, not in whatever cwd
        from catanrl.runs import runs_root

        kwargs["dir"] = str(runs_root() / "_wandb")
        Path(kwargs["dir"]).mkdir(parents=True, exist_ok=True)
    return kwargs


# ---------------------------------------------------------------- the run's records


def load_records(run: Any, *, skip: Iterable[Path] = ()) -> list[Any]:
    """Every eval record under the run, in path order, except those at ``skip``
    (a file being written is left for the next call)."""
    from catanrl.eval.results import EvalRecord

    skip = set(skip)
    out = []
    for path in sorted((run.path / "evals").glob("*/*.json")):
        if path in skip:
            continue
        try:
            out.append(EvalRecord.load(path))
        except (OSError, ValueError, KeyError):
            continue
    return out


def select_entries(run: Any, records: Iterable[Any]) -> list[tuple[int, int, Any]]:
    """``(agent_steps, epoch, record)`` of the select records of committed checkpoints,
    in step order."""
    by_epoch = {c.epoch: c.agent_steps for c in run.checkpoints()}
    out = [(int(by_epoch[epoch]), epoch, r) for r in records
           if is_select(r) and (epoch := record_epoch(run, r)) in by_epoch]
    return sorted(out, key=lambda t: (t[0], str(t[2].path)))


def final_records(run: Any) -> list[Any]:
    """Every record of the run's checkpoints that is not a select one, in path order."""
    return [r for r in load_records(run) if not is_select(r) and record_epoch(run, r) is not None]


def timeline(run: Any, *, after: int = -1, skip: Iterable[Path] = ()) -> list[tuple[int, dict]]:
    """``(agent_steps, row)`` in step order: the kept metrics rows after ``after`` merged
    with the select rows of checkpoints after it (records at ``skip`` left out)."""
    rows: dict[int, dict] = {}
    for row in run.metrics():
        steps = int(row.get("agent_steps", row.get("step", -1)))
        if steps > after:
            rows.setdefault(steps, {}).update(history_row(row))
    for steps, epoch, record in select_entries(run, load_records(run, skip=skip)):
        if steps > after:
            rows.setdefault(steps, {"epoch": epoch, "agent_steps": steps}).update(select_row(record))
    return sorted(rows.items())


def sync_history(wb: Any, run: Any, *, through: int, skip: Iterable[Path] = ()) -> int:
    """Log the run's timeline after ``through``; returns the last step logged."""
    for steps, row in timeline(run, after=through, skip=skip):
        wb.log(row, step=steps)
        through = max(through, steps)
    wb.summary[THROUGH_KEY] = through
    return through


def sync_finals(wandb: ModuleType, wb: Any, run: Any) -> int:
    """The finals as summary keys and the table; returns how many records there are."""
    records = final_records(run)
    if records:
        wb.summary.update(final_summary(records))
        wb.summary[FINAL_TABLE] = wandb.Table(columns=list(FINAL_COLUMNS), data=final_table_rows(records))
    return len(records)


def backfill(run: Any, settings: Settings, *, tags: Iterable[str] = ("backfill",), dry_run: bool = False) -> dict:
    """Mirror (or re-sync) a run directory into its wandb run, idempotent through
    ``summary[THROUGH_KEY]``. Returns what was (or, dry, would be) logged."""
    if dry_run:  # reads only: no wandb, no directory of its own
        return {"id": run_id(run.ref, stored_hash(run)), "rows": len(timeline(run)), "finals": len(final_records(run)),
                "dry_run": True}
    wandb = import_wandb()
    if wandb is None:
        raise RuntimeError("wandb is not installed")
    kwargs = init_kwargs(run, settings, tags=tags, notes=BACKFILL_NOTES)
    wb = wandb.init(**kwargs)
    before = wb.summary.get(THROUGH_KEY)
    through = -1 if before is None else int(before)
    rows = len(timeline(run, after=through))
    after = sync_history(wb, run, through=through)
    finals = sync_finals(wandb, wb, run)
    status = run.status
    wb.config.update({"status_at_backfill": status, "steps_per_epoch": kwargs["config"]["steps_per_epoch"],
                      "_provenance": provenance_config(run)}, allow_val_change=True)
    wb.summary["status"] = status
    wb.finish()
    return {"id": kwargs["id"], "rows": rows, "finals": finals, "through": after, "resumed": before is not None,
            "dry_run": False}


# ---------------------------------------------------------------- the live mirror


class Mirror:
    """One launch's live mirror: the writer posts to a queue; a worker thread owns the
    wandb run and makes every wandb call, warning on any failure."""

    def __init__(self, run: Any, settings: Settings, wandb: ModuleType) -> None:
        self.run = run
        self.settings = settings
        self._wandb = wandb
        self._wb: Any = None
        self._through = -1  # the step wandb holds a row at; rows below it are gone for good
        self._seen: set[Path] = set()  # record files read (each once)
        self._pending: list[tuple[int, int, Any]] = []  # select rows waiting for their epoch's row
        self._steps_per_epoch: int | None = None
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._main, name="wandb-mirror", daemon=True)
        self._thread.start()

    @classmethod
    def start(cls, run: Any) -> Mirror | None:
        """The mirror of ``run``'s launch, or None when it is off or cannot start (warned)."""
        try:
            settings = settings_for(run.config)
            if settings is None:
                return None
            wandb = import_wandb()
            if wandb is None:
                _warn(f"{run.ref}: wandb is not importable; not mirroring")
                return None
            return cls(run, settings, wandb)
        except Exception as error:  # noqa: BLE001  (the run must not fail for its mirror)
            _warn(f"{run.ref}: not mirroring ({error!r})")
            return None

    # ------------------------------------------------------------ the writer's side

    def log(self, row: Mapping[str, Any], step: int) -> None:
        self._queue.put(("log", dict(row), int(step)))

    def alias(self, name: str, ref: Any, *, metric: float | None = None) -> None:
        self._queue.put(("alias", name, ref, metric))

    def status(self, status: str) -> None:
        self._queue.put(("status", status))

    def finish(self, timeout: float | None = None) -> None:
        """Drain the queue, add the finals and close the wandb run; never waits longer than
        ``timeout`` (the daemon thread then keeps syncing until the process exits)."""
        timeout = FINISH_TIMEOUT_S if timeout is None else timeout
        self._queue.put(None)
        self._thread.join(timeout)
        if self._thread.is_alive():
            _warn(f"{self.run.ref}: wandb is still syncing after {timeout:.0f} s; not waiting")

    # ------------------------------------------------------------ the worker

    def _main(self) -> None:
        try:
            self._open()
        except Exception as error:  # noqa: BLE001
            _warn(f"{self.run.ref}: wandb.init failed ({error!r}); not mirroring this launch")
            self._wb = None
            while self._queue.get() is not None:  # let finish() return promptly
                pass
            return
        while True:
            item = self._queue.get()
            if item is None:
                break
            try:
                self._handle(item)
            except Exception as error:  # noqa: BLE001
                _warn(f"{self.run.ref}: wandb {item[0]} failed ({error!r})")
        try:
            self._close()
        except Exception as error:  # noqa: BLE001
            _warn(f"{self.run.ref}: closing the wandb run failed ({error!r})")

    def _open(self) -> None:
        kwargs = init_kwargs(self.run, self.settings, notes=LIVE_NOTES)
        self._steps_per_epoch = kwargs["config"]["steps_per_epoch"]
        self._wb = self._wandb.init(**kwargs)
        through = self._wb.summary.get(THROUGH_KEY)
        self._through = -1 if through is None else int(through)
        # Rows an earlier launch made without a mirror (or a reopened finished run) replay
        # through the live path, so their select rows fall into place the same way.
        for row in self.run.metrics():
            self._handle(("log", row, int(row.get("step", -1))))

    def _handle(self, item: tuple) -> None:
        kind = item[0]
        if kind == "log":
            _, row, step = item
            steps = int(row.get("agent_steps", step))
            self._sync_selects(limit=steps)
            if steps <= self._through:
                return
            if self._steps_per_epoch is None and (per_epoch := steps_per_epoch([row])) is not None:
                self._steps_per_epoch = per_epoch
                self._wb.config.update({"steps_per_epoch": per_epoch}, allow_val_change=True)
            self._wb.log(history_row(row), step=steps)
            self._through = steps
            self._wb.summary[THROUGH_KEY] = steps
        elif kind == "alias":
            _, name, ref, metric = item
            self._wb.summary.update({f"{name}/ref": str(ref), f"{name}/epoch": int(ref.epoch),
                                     **({f"{name}/metric": metric} if metric is not None else {})})
            self._sync_selects(limit=int(ref.agent_steps))
        elif kind == "status":
            self._wb.summary["status"] = item[1]

    def _sync_selects(self, *, limit: int) -> None:
        """Log the select rows of checkpoints up to ``limit`` (newer ones wait for their
        epoch's row), in step order. One at the step wandb holds the last row at (the
        last epoch's off-interval checkpoint) joins that row; below it, wandb would drop it."""
        new = load_records(self.run, skip=self._seen)
        self._seen.update(r.path for r in new)
        self._pending = sorted(self._pending + select_entries(self.run, new), key=lambda t: (t[0], str(t[2].path)))
        while self._pending and self._pending[0][0] <= limit:
            steps, epoch, record = self._pending.pop(0)
            if steps >= self._through:
                self._wb.log({"epoch": epoch, "agent_steps": steps, **select_row(record)}, step=steps)

    def _close(self) -> None:
        if self._wb is None:
            return
        self._sync_selects(limit=1 << 62)  # the last epoch's off-interval checkpoint joins its row
        sync_finals(self._wandb, self._wb, self.run)
        self._wb.summary["status"] = self.run._run_json()["status"]
        self._wb.config.update({"_provenance": provenance_config(self.run)}, allow_val_change=True)
        self._wb.finish()
