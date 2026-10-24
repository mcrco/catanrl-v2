"""The runs directory (DESIGN 6.1, 6.4): one place for every run, read by
name from any cwd or worktree.

::

    runs/<experiment>/<variant>/
      config.json      write-once: resolved config, its hash, bound refs
      run.json         trainer-owned, atomic: status, launches, committed checkpoints
      aliases.jsonl    append-only: best (cites its select eval), pins
      bindings.jsonl   append-only: refs to other runs pinned after creation
      metrics.jsonl    one row per epoch, one per monitor eval (select/<name>/...);
                       rows a resume rolled back are hidden
      provenance/      launch<n>.json and a patch per dirty repo
      checkpoints/     e000600.pt ...; orphaned/launch<n>/ after a crash
      trainer_state.pt the commit point
      evals/<suite>/   <eval_id>.json + .games.jsonl

At a checkpoint the trainer writes the ``.pt``, then ``trainer_state.pt``
(the commit), then ``run.json``; then the monitor evals (``eval.monitor``) run,
each logging a metrics row of its own at the checkpoint's step, and the select
eval runs and ``best`` moves. A resume moves any ``.pt`` newer than the trainer state to
``orphaned/``, so it is never ``latest`` nor an epoch ref.

When the experiment's ``meta.wandb.project`` (or ``CATANRL_WANDB_PROJECT``)
is set, the writer also drives :mod:`catanrl.wandb_mirror`: a copy of the
rows, select results and finals that never decides anything (DESIGN 6.10).
"""

from __future__ import annotations

import dataclasses
import fcntl
import fnmatch
import os
import re
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Self

import torch

from catanrl import canon
from catanrl.checkpoint import Checkpoint, CheckpointRef
from catanrl.config import (
    NAME_RE,
    SCHEMA,
    ConfigConflict,
    ConfigError,
    Variant,
    config_diff,
    config_hash,
    init_parts,
    merge,
    migrate,
    parse_overrides,
    resolve,
)

Status = Literal["running", "finished", "time_limit", "interrupted", "failed", "dead"]
RESERVED = {"best", "latest", "init"}
REF_RE = re.compile(r"^(?P<run>[a-z0-9][a-z0-9._+-]*(?:/[a-z0-9][a-z0-9._+-]*)?)"
                    r"(?:@(?P<which>[A-Za-z0-9._+-]+))?(?:#(?P<sha>[0-9a-f]{6,64}))?$")
#: How long a writer waits for a lock that a status probe may hold for an instant.
LOCK_WAIT_S = 2.0
#: Statuses another run may be used from: a dependent stage starts, a final's ref to a run
#: of another experiment is pinned (a time_limit run pins its epoch).
USABLE = ("finished", "time_limit")
#: Keys of a monitor's metrics row (catanrl.train.select): ``select/<name>/<field>``, at the
#: step of the checkpoint it scored, beside the bookkeeping ones.
MONITOR_PREFIX = "select/"
ROW_KEYS = ("launch", "step", "epoch", "agent_steps")


class RunError(RuntimeError):
    pass


class RunLocked(RunError):
    pass


class Unbound(RunError):
    """A final's ref to a run of another experiment that cannot be pinned yet: that run is
    missing or not usable (:data:`USABLE`). The final waits for it (:func:`bound_lineup`)."""


def runs_root() -> Path:
    """``$CATANRL_RUNS_DIR``, else ``runs/`` in the main worktree (shared by every worktree)."""
    env = os.environ.get("CATANRL_RUNS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve().parent
    common = subprocess.run(["git", "-C", str(here), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            capture_output=True, text=True, check=False).stdout.strip()
    if not common:
        raise RunError("cannot find the main worktree; set CATANRL_RUNS_DIR")
    return Path(common).parent / "runs"


def parse_ref(ref: str) -> tuple[str, str | None, str | None]:
    """``"exp/var@which#sha"`` -> (run, which, sha)."""
    m = REF_RE.match(str(ref))
    if not m:
        raise RunError(f"bad run ref {ref!r}: expected experiment[/variant][@which][#sha]")
    return m["run"], m["which"], m["sha"]


def _ckpt_name(epoch: int) -> str:
    return f"e{int(epoch):06d}.pt"


def _lock(path: Path, what: str, wait: float = LOCK_WAIT_S) -> int:
    """The run's exclusive flock (retried briefly: readers probe it for an instant)."""
    path.mkdir(parents=True, exist_ok=True)
    fd = os.open(path / ".lock", os.O_RDWR | os.O_CREAT, 0o644)
    deadline = time.time() + wait
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            if time.time() >= deadline:
                os.close(fd)
                raise RunLocked(f"{what} is locked by a live process") from None
            time.sleep(0.05)


def _unlock(fd: int) -> None:
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def _lock_free(path: Path) -> bool:
    lock = path / ".lock"
    if not lock.exists():
        return True
    fd = os.open(lock, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    finally:
        os.close(fd)


def _rebase(value: Any, old: str, new: str) -> Any:
    """Replace refs to run ``old`` by refs to ``new`` in every string of a config."""
    pattern = re.compile(rf"(?<![\w/.+-]){re.escape(old)}(?=[@#:]|$)")
    if isinstance(value, Mapping):
        return {k: _rebase(v, old, new) for k, v in value.items()}
    if isinstance(value, list):
        return [_rebase(v, old, new) for v in value]
    return pattern.sub(new, value) if isinstance(value, str) else value


# ---------------------------------------------------------------- reading


class Run:
    """A run directory, read-only."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        if not (self.path / "config.json").exists():
            raise RunError(f"{self.path} is not a run (no config.json)")
        self.manifest: dict = canon.read_json(self.path / "config.json")
        self.ref = f"{self.manifest['experiment']}/{self.manifest['variant']}"
        self.config: dict = migrate(self.manifest["config"], from_schema=self.manifest["schema"])
        self.config_hash: str = config_hash(self.config)  # migrated, like every plan it is compared with

    def __repr__(self) -> str:
        return f"Run({self.ref!r}, {self.status})"

    @property
    def experiment(self) -> str:
        return self.manifest["experiment"]

    def _run_json(self) -> dict:
        path = self.path / "run.json"
        return canon.read_json(path) if path.exists() else {"status": None, "launches": [], "checkpoints": []}

    @property
    def status(self) -> Status | None:
        status = self._run_json()["status"]
        if status == "running" and _lock_free(self.path):
            return "dead"
        return status

    def variant(self) -> Variant:
        m = self.manifest
        return Variant(experiment=m["experiment"], name=m["variant"], stage=m.get("stage"), labels=m.get("labels", {}),
                       config=self.config, config_hash=self.config_hash, depends_on=tuple(m.get("depends_on", ())))

    def checkpoints(self) -> list[CheckpointRef]:
        """Committed checkpoints only (never orphaned ones), by epoch."""
        return [CheckpointRef(run=self.ref, which=f"e{c['epoch']}", epoch=c["epoch"], agent_steps=c["agent_steps"],
                              path=self.path / "checkpoints" / c["file"], weights_sha256=c["weights_sha256"])
                for c in sorted(self._run_json()["checkpoints"], key=lambda c: c["epoch"])]

    def aliases(self) -> dict[str, dict]:
        return {row["name"]: row for row in canon.read_jsonl(self.path / "aliases.jsonl")}

    def resolve(self, which: str | int = "best") -> CheckpointRef:
        """A selector (best, latest, init, an alias, N or eN, optionally #sha) as a pinned ref."""
        which, _, sha = str(which).partition("#")
        ckpts = self.checkpoints()
        by_epoch = {c.epoch: c for c in ckpts}
        aliases = self.aliases()
        found: CheckpointRef | None = None
        if which in ("latest", "init"):
            if ckpts:
                found = ckpts[-1] if which == "latest" else by_epoch.get(0)
        elif re.fullmatch(r"e?\d+", which):
            found = by_epoch.get(int(which.lstrip("e")))
        elif which in aliases:
            row = aliases[which]
            if which == "best" and not row.get("eval_id"):
                raise RunError(f"{self.ref}@best does not cite a select eval")
            found = by_epoch.get(row["epoch"])
            if found is None or found.weights_sha256 != row["weights_sha256"]:
                raise RunError(f"{self.ref}@{which} points at e{row['epoch']}, which is not a committed checkpoint")
        if found is None:
            known = sorted({"latest", "init", *aliases}) + [f"e{c.epoch}" for c in ckpts]
            raise RunError(f"{self.ref} has no checkpoint {which!r}; selectors: {', '.join(known) or 'none'}")
        if sha and not found.weights_sha256.startswith(sha):
            raise RunError(f"{self.ref}@{which} is #{found.weights_sha256[:12]}, not #{sha}")
        return dataclasses.replace(found, which=which)

    def checkpoint(self, which: str | int = "best") -> Checkpoint:
        ref = self.resolve(which)
        return Checkpoint.load(ref.path, ref=ref)

    def build_policy(self, which: str | int = "best", device: str = "cpu", *, allow_drift: bool = False):
        return self.checkpoint(which).model("actor", device, allow_drift=allow_drift)

    def build_critic(self, which: str | int = "best", device: str = "cpu", *, allow_drift: bool = False):
        return self.checkpoint(which).model("critic", device, allow_drift=allow_drift)

    def agent(self, key: str = "POLICY", which: str | int = "best", **params: Any):
        from catanrl.agents import resolve_agent

        return resolve_agent({"key": key, "params": {"ckpt": f"{self.ref}@{which}", **params}})

    def evals(self, suite: str | None = None) -> list:
        from catanrl.eval.results import EvalRecord

        return [EvalRecord.load(p) for p in sorted((self.path / "evals").glob(f"{suite or '*'}/*.json"))]

    @property
    def init_ref(self) -> str | None:
        """The checkpoint ``init_from`` loaded the actor from, as pinned at creation: ``${init}``."""
        return ((self.manifest.get("resolved") or {}).get("init_from") or {}).get("actor")

    @property
    def champion_ref(self) -> str | None:
        """``eval.champion`` as pinned at creation: ``${champion}`` (None when the config sets
        none: its ``${champion}`` was resolved to ``${init}``)."""
        return (self.manifest.get("resolved") or {}).get("champion")

    def bindings(self) -> dict[str, str]:
        """Specs naming other runs, as pinned (at creation, or when a later final eval first ran)."""
        out = dict((self.manifest.get("resolved") or {}).get("specs") or {})
        out.update({row["spec"]: row["bound"] for row in canon.read_jsonl(self.path / "bindings.jsonl")})
        return out

    def metrics(self, *, lineage: bool = False, until: int | None = None) -> list[dict]:
        """Epoch rows; rows of epochs a later launch rolled back (it resumed from an
        earlier commit) are dropped. ``lineage`` prepends every ancestor's rows up to
        the epoch this run started from."""
        resumed = {l["launch"]: l["resumed_from"] for l in self._run_json()["launches"] if "resumed_from" in l}

        def kept(row: dict) -> bool:
            epoch = row.get("epoch")
            if epoch is None:
                return True
            later = [e for launch, e in resumed.items() if launch > row.get("launch", 0)]
            return all(epoch <= e for e in later) and (until is None or epoch <= until)

        rows = [r for r in canon.read_jsonl(self.path / "metrics.jsonl") if kept(r)]
        if not lineage:
            return rows
        out, seen = [], set()
        for ref in reversed(self.lineage()):
            if ref.run and ref.run not in seen and ref.run != self.ref:
                seen.add(ref.run)
                out += load_experiment(ref.run).metrics(until=ref.epoch)
        return out + rows

    def lineage(self) -> list[CheckpointRef]:
        """Every bound ``init_from`` checkpoint of this run and its ancestors (breadth first)."""
        out: list[CheckpointRef] = []
        queue, seen = [self], {self.ref}
        while queue:
            run = queue.pop(0)
            for text in ((run.manifest.get("resolved") or {}).get("init_from") or {}).values():
                ckpt = load_checkpoint(text)
                if str(ckpt.ref) not in {str(r) for r in out}:
                    out.append(ckpt.ref)
                if ckpt.ref.run and ckpt.ref.run not in seen:
                    seen.add(ckpt.ref.run)
                    queue.append(load_experiment(ckpt.ref.run))
        return out

    def provenance(self) -> list[dict]:
        return [canon.read_json(p) for p in sorted((self.path / "provenance").glob("launch*.json"),
                                                   key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))]


def load_experiment(ref: str | Path) -> Run:
    """``"exp/variant"``, ``"exp"`` (one variant), or an explicit run directory
    (a Path, or a string starting with ``/``, ``./`` or ``../``)."""
    text = str(ref)
    if isinstance(ref, Path) or text.startswith(("/", "./", "../")):
        return Run(ref)
    run, which, sha = parse_ref(text)
    if which or sha:
        raise RunError(f"{ref}: load_experiment takes a run, not a checkpoint (use load_checkpoint)")
    root = runs_root()
    if "/" in run:
        if not (root / run / "config.json").exists():
            raise RunError(f"no run {run} under {root}")
        return Run(root / run)
    variants = sorted(p.parent for p in (root / run).glob("*/config.json"))
    if len(variants) != 1:
        names = [p.name for p in variants]
        raise RunError(f"{run} has {len(variants)} variants{': ' + ', '.join(names) if names else ''}; "
                       "name one as experiment/variant")
    return Run(variants[0])


def list_runs(pattern: str = "*", *, status: Status | None = None) -> list[Run]:
    """Runs whose ``experiment/variant`` matches ``pattern`` (``*`` matches one name)."""
    pattern = pattern if "/" in pattern else f"{pattern}/*"
    runs = [Run(p.parent) for p in sorted(runs_root().glob("*/*/config.json"))
            if not p.parent.parent.name.startswith("_")]
    return [r for r in runs if fnmatch.fnmatchcase(r.ref, pattern) and (status is None or r.status == status)]


def load_checkpoint(ref: str | Path) -> Checkpoint:
    """``"exp/var@which[#sha]"`` (default ``best``) or a ``.pt`` path (``path.pt#sha`` pins its weights)."""
    text = str(ref)
    path, _, sha = text.partition("#")
    if path.endswith(".pt"):
        ckpt = Checkpoint.load(path)
        if sha and not ckpt.ref.weights_sha256.startswith(sha):
            raise RunError(f"{path} holds #{ckpt.ref.weights_sha256[:12]}, not #{sha}")
        return ckpt
    run, which, sha = parse_ref(text)
    return load_experiment(run).checkpoint((which or "best") + (f"#{sha}" if sha else ""))


def pin(ref: str, name: str, note: str = "") -> CheckpointRef:
    """Name a checkpoint (appends to aliases.jsonl; safe while it trains)."""
    if not NAME_RE.match(name) or name in RESERVED or re.fullmatch(r"e?\d+", name):
        raise RunError(f"bad alias {name!r}: names match {NAME_RE.pattern} and are not best, latest, init or epochs")
    run_ref, which, sha = parse_ref(ref)
    run = load_experiment(run_ref)
    target = run.resolve((which or "latest") + (f"#{sha}" if sha else ""))
    _append_alias(run.path, name, target, by="cli", note=note)
    return target


def _append_alias(path: Path, name: str, ref: CheckpointRef, *, by: str, note: str = "",
                  eval_id: str | None = None, metric: float | None = None) -> None:
    canon.append_jsonl(path / "aliases.jsonl", {
        "name": name, "ref": str(ref), "epoch": ref.epoch, "weights_sha256": ref.weights_sha256,
        "eval_id": eval_id, "metric": metric, "note": note, "by": by, "time": time.time()})


def check_stored(variant: Variant) -> Run | None:
    """The existing run of ``variant``, after checking it was made from the same
    config (``ConfigConflict`` with the diff otherwise); None if there is none.
    Every launch path calls this, whatever the run's status."""
    path = runs_root() / variant.experiment / variant.name
    if not (path / "config.json").exists():
        return None
    run = Run(path)
    if run.config_hash != variant.config_hash:
        diff = "\n  ".join(config_diff(run.config, variant.config)) or "(hash differs)"
        raise ConfigConflict(f"{variant.ref} exists with another config:\n  {diff}\n"
                             "use --variant NAME or catanrl fork")
    return run


def fork(ref: str, *, as_: str, overrides: Sequence[str] = ()) -> Variant:
    """A new variant continuing a stopped run's latest commit (weights and optimizer).

    The parent's trainer state is copied into the fork now, under the parent's
    lock, so a later relaunch of the parent cannot invalidate the fork. Refs to
    the parent in the parent's config (``${run}`` in eval.final) become refs to
    the fork; ``${init}`` stays the parent's start (pinned), not the fork's."""
    run_ref, which, _ = parse_ref(ref)
    parent = load_experiment(run_ref)
    experiment, _, name = as_.rpartition("/")
    experiment = experiment or parent.manifest["experiment"]
    if not NAME_RE.match(name):
        raise RunError(f"bad variant name {name!r}")
    path = runs_root() / experiment / name
    if (path / "config.json").exists():
        raise RunError(f"{experiment}/{name} exists")
    fd = _lock(parent.path, parent.ref, wait=0)
    try:
        target = parent.resolve(which or "latest")
        state_path = parent.path / "trainer_state.pt"
        state = torch.load(state_path, map_location="cpu", weights_only=False) if state_path.exists() else None
        if state is None or int(state["epoch"]) != target.epoch:
            raise RunError(f"{parent.ref}'s committed trainer state is at "
                           f"{'nothing' if state is None else 'e' + str(state['epoch'])}, not e{target.epoch}: "
                           "a fork continues the latest commit")
        raw = merge(_rebase(parent.config, parent.ref, f"{experiment}/{name}"), parse_overrides(overrides))
        # A continuation plays the parent's start and champion, as the parent pinned them.
        raw["eval"] = expand_init(raw["eval"], parent.init_ref, parent.champion_ref)
        if parent.champion_ref is not None:
            raw["eval"]["champion"] = parent.champion_ref
        raw["init_from"] = {"ckpt": str(target), "parts": ["actor", "critic"], "restore": "weights+optimizer"}
        cfg = resolve(raw, experiment=experiment, variant=name)
        variant = Variant(experiment=experiment, name=name, stage=None, labels={"fork": parent.ref}, config=cfg,
                          config_hash=config_hash(cfg), depends_on=())
        _write_manifest(path, variant, source=None)
        shutil.copyfile(state_path, path / "fork_state.pt")
    finally:
        _unlock(fd)
    return variant


def finish(ref: str) -> None:
    """End a stopped run (time_limit, interrupted, dead, failed) so dependents may use it.

    Under the run's lock, the commit is reconciled as a resume would (orphans
    moved, committed checkpoints listed); a run whose committed checkpoints
    still lack select evals must be resumed first."""
    run = load_experiment(ref)
    if run.status not in ("time_limit", "interrupted", "dead", "failed"):
        raise RunError(f"{run.ref} is {run.status}; only a stopped run can be finished")
    fd = _lock(run.path, run.ref, wait=0)
    try:
        state = _reconcile(run.path, launch=len(run._run_json()["launches"]))
        missing = [str(r) for r in _missing_selects(run)]
        if missing:
            raise RunError(f"{run.ref}: committed checkpoints lack select evals ({', '.join(missing)}); "
                           "resume it (catanrl run) before finishing")
        state["status"] = "finished"
        state["finished_by"] = "catanrl finish"
        canon.atomic_write_json(run.path / "run.json", state)
    finally:
        _unlock(fd)


def _missing_selects(run: Run) -> list[CheckpointRef]:
    select = (run.config.get("eval") or {}).get("select")
    if not select:
        return []
    from catanrl.eval.results import find

    every = int(select.get("every", 1))
    lineup = expand_init(select["lineup"], run.init_ref, run.champion_ref)
    return [ref for ref in run.checkpoints() if ref.epoch % every == 0
            and find([str(s).replace("${self}", str(ref)) for s in lineup], select["suite"],
                     fill=select.get("fill")) is None]


def check_runs(pattern: str = "*") -> list[dict]:
    """Load, verify and self-test every alias and the latest checkpoint of every run."""
    rows = []
    for run in list_runs(pattern):
        for which in sorted(set(run.aliases()) | ({"latest"} if run.checkpoints() else set())):
            row = {"run": run.ref, "which": which, "ok": True, "error": None}
            try:
                ckpt = run.checkpoint(which)
                ckpt.self_test()
                row["ref"] = str(ckpt.ref)
            except Exception as error:  # noqa: BLE001  (report every failure, do not stop at the first)
                row.update(ok=False, error=f"{type(error).__name__}: {error}")
            rows.append(row)
    return rows


# ---------------------------------------------------------------- binding refs


def expand_init(value: Any, init: str | None, champion: str | None = None) -> Any:
    """``value`` (a spec, a lineup, an eval section) with ``${init}`` as ``init``, the pinned
    checkpoint a run's actor started from (:attr:`Run.init_ref`), and ``${champion}`` as
    ``champion``, the pinned ``eval.champion`` (:attr:`Run.champion_ref`)."""
    if isinstance(value, Mapping):
        return {k: expand_init(v, init, champion) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_init(v, init, champion) for v in value]
    if not isinstance(value, str):
        return value
    if "${champion}" in value:
        if champion is None:
            raise RunError(f"{value}: ${{champion}} is eval.champion as pinned when the run was created, "
                           "and there is none")
        value = value.replace("${champion}", champion)
    if "${init}" not in value:
        return value
    if init is None:
        raise RunError(f"{value}: ${{init}} is the checkpoint init_from loads the actor from, and there is none")
    return value.replace("${init}", init)


def monitor_evals(row: Mapping[str, Any]) -> dict[str, str]:
    """``{monitor name: eval_id}`` of the monitor results a metrics row holds."""
    head, tail = MONITOR_PREFIX, "/eval_id"
    return {key[len(head):-len(tail)]: value for key, value in row.items()
            if key.startswith(head) and key.endswith(tail) and isinstance(value, str)}


def is_monitor_row(row: Mapping[str, Any]) -> bool:
    """Whether a metrics row is a monitor's (its own row at a checkpoint's step), not an epoch's."""
    return bool(monitor_evals(row)) and all(k in ROW_KEYS or k.startswith(MONITOR_PREFIX) for k in row)


def _model_spec(spec: str) -> bool:
    """Whether ``spec`` names a model key (POLICY, MCTS, BELIEF, PREDICT) with no ``${...}`` left."""
    from catanrl.agents.spec import MODEL_KEYS

    return "${" not in spec and spec.startswith(MODEL_KEYS)


def bind_spec(spec: str, own: str | None) -> str | None:
    """The canonical form of a lineup spec naming another run's checkpoint, or None
    when it names no run (a bot, ``${...}``) or the run ``own`` (bound at eval time)."""
    if not _model_spec(spec):
        return None
    if own and re.search(rf"(^|[:=]){re.escape(own)}(@|#|:|$)", spec):
        return None
    from catanrl.agents import resolve_agent

    return resolve_agent(spec).canonical


def spec_runs(spec: str) -> list[str]:
    """The runs (``experiment/variant``, or an experiment with one variant) whose checkpoints
    a lineup spec names: one for POLICY, MCTS and BELIEF, the policy's then the predictor's
    for PREDICT; none for a bot, a ``.pt`` path or an unexpanded ``${...}``."""
    if not _model_spec(spec):
        return []
    from catanrl.agents.spec import spec_ckpts

    out = []
    for ckpt in spec_ckpts(spec):
        if "${" not in ckpt and not ckpt.partition("#")[0].endswith(".pt"):
            out.append(parse_ref(ckpt)[0])
    return out


def spec_run(spec: str) -> str | None:
    """The run whose checkpoint a lineup spec names (PREDICT: its policy's), or None."""
    runs = spec_runs(spec)
    return runs[0] if runs else None


def foreign_runs(spec: str, experiment: str) -> list[str]:
    """The runs of experiments other than ``experiment`` that ``spec`` names."""
    return [run for run in spec_runs(spec) if run.partition("/")[0] != experiment]


def foreign_run(spec: str, experiment: str) -> str | None:
    """The first run of another experiment that ``spec`` names, or None."""
    runs = foreign_runs(spec, experiment)
    return runs[0] if runs else None


def run_status(ref: str) -> Status | None:
    """The status of the run ``ref`` names, None when there is no such run."""
    try:
        return load_experiment(ref).status
    except RunError:
        return None


def bind_refs(cfg: Mapping[str, Any], own: str | None = None) -> dict:
    """Refs to other runs, pinned once at creation: ``init_from`` per part and eval.final
    specs (``${init}`` as the actor's ``init_from``, keyed as written). A final's ref to a
    run of another experiment that is not usable yet (missing, or still training: the runner
    cannot order it) is left to :func:`bound_lineup`, which pins it when the final is first
    played."""
    out: dict[str, Any] = {}
    init = cfg.get("init_from")
    if init:
        ckpt = init["ckpt"]
        out["init_from"] = {part: str(load_checkpoint(ckpt[part] if isinstance(ckpt, Mapping) else ckpt).ref)
                            for part in init_parts(init)}
    champion = (cfg.get("eval") or {}).get("champion")
    if champion is not None:  # absent before eval.champion existed: older manifests stay as they are
        out["champion"] = str(load_checkpoint(champion).ref)
    start = (out.get("init_from") or {}).get("actor")
    experiment = own.partition("/")[0] if own else None
    out["specs"] = {}
    for entry in (cfg.get("eval") or {}).get("final") or []:
        for spec in entry["lineup"]:
            others = foreign_runs(spec, experiment) if experiment is not None else []
            if any(run_status(other) not in USABLE for other in others):
                continue
            bound = bind_spec(expand_init(spec, start, out.get("champion")), own)
            if bound is not None:
                out["specs"][spec] = bound
    return out


def bound_lineup(run: Run, lineup: Sequence[str], *, pin: bool = True) -> list[str]:
    """``lineup`` with refs to other runs as first pinned for ``run``, pinning new ones now
    (``pin=False`` only resolves them: nothing is written). A ref to a run of another
    experiment waits for that run to be usable (:data:`USABLE`): :class:`Unbound` before."""
    known = run.bindings()
    out = []
    for spec in lineup:
        bound = known.get(spec)
        if bound is None:
            for other in foreign_runs(spec, run.experiment):
                status = run_status(other)
                if status not in USABLE:
                    raise Unbound(f"{spec} waits for {other}, which is {status or 'not there yet'}; "
                                  "the final is played once it is finished")
            expanded = expand_init(spec, run.init_ref, run.champion_ref)
            bound = bind_spec(expanded, run.ref)
            if bound is None:  # a bot, ``${...}`` or the run itself: resolved when the eval runs
                bound = expanded
            elif pin:
                canon.append_jsonl(run.path / "bindings.jsonl", {"spec": spec, "bound": bound, "time": time.time()})
        out.append(bound)
    return out


def _write_manifest(path: Path, variant: Variant, *, source: Mapping[str, Any] | None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    canon.write_once_json(path / "config.json", {
        "schema": SCHEMA, "experiment": variant.experiment, "variant": variant.name, "stage": variant.stage,
        "labels": variant.labels, "depends_on": list(variant.depends_on), "config": variant.config,
        "config_hash": variant.config_hash, "resolved": bind_refs(variant.config, variant.ref), "source": source,
    })


def _reconcile(path: Path, *, launch: int) -> dict:
    """run.json after a crash: checkpoints newer than the trainer state move to
    orphaned/launch<n>/, committed ones missing from the index are listed.
    Returns run.json (written)."""
    state_path = path / "trainer_state.pt"
    committed = -1
    if state_path.exists():
        committed = int(torch.load(state_path, map_location="cpu", weights_only=False)["epoch"])
    run = canon.read_json(path / "run.json") if (path / "run.json").exists() else {
        "status": None, "launches": [], "checkpoints": []}
    listed = {c["epoch"]: c for c in run["checkpoints"]}
    for file in sorted((path / "checkpoints").glob("e*.pt")):
        epoch = int(file.stem[1:])
        if epoch > committed:
            dest = path / "checkpoints" / "orphaned" / f"launch{launch}"
            dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(file), dest / file.name)
            listed.pop(epoch, None)
        elif epoch not in listed:
            header = Checkpoint.load(file).header
            listed[epoch] = {"epoch": epoch, "agent_steps": header["agent_steps"], "file": file.name,
                             "weights_sha256": header["weights_sha256"], "launch": None}
    run["checkpoints"] = [listed[e] for e in sorted(listed)]
    if committed >= 0:
        run["trainer_state"] = {"epoch": committed, "time": time.time()}
    canon.atomic_write_json(path / "run.json", run)
    return run


# ---------------------------------------------------------------- writing


class RunWriter(Run):
    """The trainer's side of a run: lock, launches, checkpoints, commits, aliases.

    A context manager; leaving it by an exception records ``failed`` (or
    ``interrupted`` for KeyboardInterrupt) and releases the lock. A run that
    had already finished stays finished: only its launch row records that.
    The wandb mirror, when on, sees every logged row, alias and the end of
    the launch; it can only warn, never fail the run.
    """

    mode: Literal["new", "resume", "done"]

    def __init__(self, path: Path, variant: Variant, mode: str, launch: int, lock_fd: int) -> None:
        super().__init__(path)
        self.bound_variant = variant
        self.mode = mode  # type: ignore[assignment]
        self.launch = launch
        self.run_id = f"{self.ref}#launch{launch}"
        self.stop_reason: Literal["time_limit", "interrupted"] | None = None
        self._lock_fd: int | None = lock_fd
        self._pending: list[CheckpointRef] = []
        hours = (variant.config.get("runtime") or {}).get("time_limit_hours")
        self._deadline = None if hours is None else time.time() + float(hours) * 3600
        self._stop_requested = False
        self.mirror: Any = None  # catanrl.wandb_mirror.Mirror, once the launch is recorded

    @classmethod
    def open(cls, variant: Variant, *, argv: Sequence[str], allow_code_change: bool = False,
             source: Mapping[str, Any] | None = None) -> RunWriter:
        """Create, resume or reopen (``done``) the run of ``variant`` (DESIGN 6.4's table)."""
        path = runs_root() / variant.experiment / variant.name
        fd = _lock(path, variant.ref)
        try:
            return cls._open_locked(path, variant, argv, allow_code_change, source, fd)
        except BaseException:
            _unlock(fd)
            raise

    @classmethod
    def _open_locked(cls, path, variant, argv, allow_code_change, source, fd) -> RunWriter:
        from catanrl import provenance

        if check_stored(variant) is None:
            _write_manifest(path, variant, source=source)
        state = canon.read_json(path / "run.json") if (path / "run.json").exists() else None
        resolved = canon.read_json(path / "config.json").get("resolved") or {}
        prov = provenance.capture(argv=argv, kind="train" if variant.kind == "train" else "eval")
        mixed = False
        if state is None:
            mode = "new"
            if (variant.config.get("init_from") or {}).get("restore") == "weights+optimizer":
                if not (path / "fork_state.pt").exists():
                    raise ConfigError("restore: weights+optimizer is made by catanrl fork, which copies the state")
                shutil.copyfile(path / "fork_state.pt", path / "trainer_state.pt")
                mode = "resume"
            state = {"status": None, "launches": [], "checkpoints": []}
        else:
            mode = "done" if state["status"] == "finished" else "resume"
            if mode == "resume" and state["launches"]:
                first = canon.read_json(path / "provenance" / f"launch{state['launches'][0]['launch']}.json")
                diff = provenance.resume_diff({"code": first["code"], "python": first["python"]}, prov)
                if diff and not allow_code_change:
                    raise provenance.ProvenanceError(
                        f"{variant.ref}: the training code or uv.lock changed since its first launch:\n  "
                        + "\n  ".join(diff) + "\nrefusing to resume (--allow-code-change records a mixed run)")
                mixed = bool(diff)
        provenance.check_launch(prov, variant.config.get("runtime") or {})
        # A launch number no earlier attempt used, even one that died before its run.json row.
        used = [l["launch"] for l in state["launches"]] + [
            int(p.stem[6:]) for p in (path / "provenance").glob("launch*.json") if p.stem[6:].isdigit()]
        launch = max(used, default=-1) + 1
        prov["run"]["launch_name"] = f"launch{launch}"
        provenance.save_patches(prov, path / "provenance")
        provenance.keep_commits(prov)
        canon.write_once_json(path / "provenance" / f"launch{launch}.json", prov)
        writer = cls(path, _bound_variant(variant, resolved), mode, launch, fd)
        state["launches"].append({"launch": launch, "mode": mode, "started": time.time(), "status": "running",
                                  "argv": list(argv), "mixed_code": mixed,
                                  "runtime": dict(variant.config.get("runtime") or {})})
        if mode != "done":
            state["status"] = "running"
        canon.atomic_write_json(path / "run.json", state)
        try:
            from catanrl._puffer import install_signal_handlers

            install_signal_handlers(writer.request_stop)
        except ValueError:  # not the main thread
            pass
        from catanrl.wandb_mirror import Mirror

        writer.mirror = Mirror.start(writer)  # None unless configured; warns rather than raises
        return writer

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if exc_type is None:
                if self.mode == "done":
                    self._end_launch("finished")
            elif self._run_json()["status"] == "finished":  # e.g. a final eval failed: the training stands
                self._end_launch("interrupted" if issubclass(exc_type, KeyboardInterrupt) else "failed")
            else:
                self.finish("interrupted" if issubclass(exc_type, KeyboardInterrupt) else "failed")
        finally:
            self.release()

    def release(self) -> None:
        if self.mirror is not None:
            self.mirror.finish()  # the finals and the status, read from the run under its lock
            self.mirror = None
        if self._lock_fd is not None:
            _unlock(self._lock_fd)
            self._lock_fd = None

    @property
    def variant_for_training(self) -> Variant:
        """The variant with ``init_from`` refs as bound at creation (and ``${init}`` as the actor's)."""
        return self.bound_variant

    def record_runtime(self, info: Mapping[str, Any]) -> None:
        """What the trainer actually ran with (torch flags after PuffeRL's init, ...), on this launch's row."""
        run = self._run_json()
        run["launches"][-1].setdefault("effective", {}).update(dict(info))
        canon.atomic_write_json(self.path / "run.json", run)

    # ---------------------------------------------------------- PufferLib logger

    def log(self, logs: Mapping[str, Any], step: int) -> None:
        row = {k: v for k, v in logs.items() if isinstance(v, (int, float, str, bool)) or v is None}
        row = {"launch": self.launch, "step": int(step), **row}
        canon.append_jsonl(self.path / "metrics.jsonl", row)
        if self.mirror is not None:
            self.mirror.log(row, step)

    def close(self, model_path: str | None = None, early_stop: bool | None = None) -> None:
        pass

    # ---------------------------------------------------------- stopping

    def request_stop(self, reason: str = "interrupted") -> None:
        self._stop_requested = True

    def should_stop(self) -> bool:
        if self._stop_requested:
            self.stop_reason = "interrupted"
            return True
        if self._deadline is not None and time.time() >= self._deadline:
            self.stop_reason = "time_limit"
            return True
        return False

    # ---------------------------------------------------------- checkpoints and commits

    def save_checkpoint(self, model, *, epoch: int, agent_steps: int) -> CheckpointRef:
        from catanrl import checkpoint, provenance

        prov = canon.read_json(self.path / "provenance" / f"launch{self.launch}.json")
        code = {"id": provenance.code_id("train"), "git": prov["git"]["repos"]["catanrl-v2"]["head"],
                "dirty": provenance.any_dirty(prov), "launch": self.launch}
        ref = checkpoint.save(self.path / "checkpoints" / _ckpt_name(epoch), model, run=self.ref, epoch=epoch,
                              agent_steps=agent_steps, config=self.config, code=code)
        self._pending.append(ref)
        return ref

    def save_trainer_state(self, state: Mapping[str, Any]) -> None:
        """The commit: trainer_state.pt, then run.json lists the new checkpoints."""
        tmp = self.path / f".trainer_state.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            torch.save({**state, "launch": self.launch}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path / "trainer_state.pt")
        run = self._run_json()
        known = {c["epoch"] for c in run["checkpoints"]}
        for ref in self._pending:
            if ref.epoch <= int(state["epoch"]) and ref.epoch not in known:
                run["checkpoints"].append({"epoch": ref.epoch, "agent_steps": ref.agent_steps,
                                           "file": ref.path.name, "weights_sha256": ref.weights_sha256,
                                           "launch": self.launch})
        self._pending = []
        run["trainer_state"] = {"epoch": int(state["epoch"]), "launch": self.launch, "time": time.time()}
        canon.atomic_write_json(self.path / "run.json", run)

    def load_trainer_state(self) -> dict | None:
        """The committed state; moves newer checkpoints to orphaned/ and reconciles run.json."""
        run = _reconcile(self.path, launch=self.launch)
        path = self.path / "trainer_state.pt"
        state = torch.load(path, map_location="cpu", weights_only=False) if path.exists() else None
        run["launches"][-1]["resumed_from"] = -1 if state is None else int(state["epoch"])
        canon.atomic_write_json(self.path / "run.json", run)
        return state

    def pending_monitors(self, monitors: Sequence[Mapping[str, Any]]) -> list[tuple[CheckpointRef, list[str]]]:
        """Committed checkpoints at a monitor's cadence whose row for it is not in metrics.jsonl
        (a crash cut its eval or its row off; a new run's e0 before its monitors ran), with the
        names of the monitors each still owes."""
        ckpts = self.checkpoints() if monitors else []
        if not ckpts:
            return []
        logged = {(row.get("epoch"), name) for row in self.metrics() for name in monitor_evals(row)}
        out = []
        for ref in ckpts:
            names = [str(m["name"]) for m in monitors if ref.epoch % int(m.get("every", 1)) == 0
                     and (ref.epoch, str(m["name"])) not in logged]
            if names:
                out.append((ref, names))
        return out

    def pending_selects(self) -> list[CheckpointRef]:
        """Committed checkpoints whose select eval is missing, plus the last one
        (its select ran after the commit, so its alias decision may be lost)."""
        select = (self.config.get("eval") or {}).get("select")
        ckpts = self.checkpoints()
        if not select or not ckpts:
            return []
        every = int(select.get("every", 1))
        missing = {r.epoch for r in _missing_selects(self)}
        return [r for r in ckpts if r.epoch % every == 0 and (r.epoch in missing or r.epoch == ckpts[-1].epoch)]

    def set_alias(self, name: str, ref: CheckpointRef, *, record: Any = None, note: str = "") -> None:
        current = self.aliases().get(name)
        if current and current["weights_sha256"] == ref.weights_sha256 and current["epoch"] == ref.epoch:
            return
        metric = None
        if record is not None:
            select = (self.config.get("eval") or {}).get("select") or {}
            metric = record.summary.get(select.get("metric", "score_rate"))
        _append_alias(self.path, name, ref, by="trainer", note=note,
                      eval_id=getattr(record, "eval_id", None), metric=metric)
        if self.mirror is not None:
            self.mirror.alias(name, ref, metric=metric)

    def finish(self, status: Status) -> None:
        run = self._run_json()
        run["status"] = status
        run["launches"][-1].update(status=status, ended=time.time())
        canon.atomic_write_json(self.path / "run.json", run)
        if self.mirror is not None:
            self.mirror.status(status)

    def _end_launch(self, status: str) -> None:
        run = self._run_json()
        run["launches"][-1].update(status=status, ended=time.time())
        canon.atomic_write_json(self.path / "run.json", run)


def _bound_variant(variant: Variant, resolved: Mapping[str, Any]) -> Variant:
    init = variant.config.get("init_from")
    bound_init = bool(init) and "init_from" in resolved
    champion = resolved.get("champion")
    if not bound_init and champion is None:
        return variant
    cfg = dict(variant.config)
    actor = None
    if bound_init:
        cfg["init_from"] = {**init, "ckpt": dict(resolved["init_from"]), "parts": list(resolved["init_from"])}
        actor = resolved["init_from"].get("actor")
    cfg["eval"] = expand_init(cfg["eval"], actor, champion)  # the trainer's select and monitor lineups
    return dataclasses.replace(variant, config=cfg)
