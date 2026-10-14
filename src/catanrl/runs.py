"""The runs directory (DESIGN 6.1, 6.4): one place for every run, read by
name from any cwd or worktree.

::

    runs/<experiment>/<variant>/
      config.json      write-once: resolved config, its hash, bound refs
      run.json         trainer-owned, atomic: status, launches, committed checkpoints
      aliases.jsonl    append-only: best (cites its select eval), pins
      metrics.jsonl    one row per epoch
      provenance/      launch<n>.json and a patch per dirty repo
      checkpoints/     e000600.pt ...; orphaned/launch<n>/ after a crash
      trainer_state.pt the commit point
      evals/<suite>/   <eval_id>.json + .games.jsonl

At a checkpoint the trainer writes the ``.pt``, then ``trainer_state.pt``
(the commit), then ``run.json``; then the select eval runs and ``best``
moves. A resume moves any ``.pt`` newer than the trainer state to
``orphaned/``, so it is never ``latest`` nor an epoch ref.

Readers::

    run = load_experiment("arch-lineage/graph-s-s42.g1")
    net = run.build_policy("best")            # net.catanrl_ref == "...g1@e600#41f2a9c07d11"
    ckpt = load_checkpoint("arch-lineage/graph-s-s42.g1@e600")
"""

from __future__ import annotations

import dataclasses
import fcntl
import os
import re
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

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
    merge,
    migrate,
    parse_overrides,
    resolve,
)

Status = Literal["running", "finished", "time_limit", "interrupted", "failed", "dead"]
RESERVED = {"best", "latest", "init"}
REF_RE = re.compile(r"^(?P<run>[a-z0-9][a-z0-9._+-]*(?:/[a-z0-9][a-z0-9._+-]*)?)"
                    r"(?:@(?P<which>[A-Za-z0-9._+-]+))?(?:#(?P<sha>[0-9a-f]{6,64}))?$")


class RunError(RuntimeError):
    pass


class RunLocked(RunError):
    pass


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
        self.config_hash: str = self.manifest["config_hash"]

    def __repr__(self) -> str:
        return f"Run({self.ref!r}, {self.status})"

    # ---------------------------------------------------------- state

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
        out: dict[str, dict] = {}
        for row in canon.read_jsonl(self.path / "aliases.jsonl"):
            out[row["name"]] = row
        return out

    def resolve(self, which: str | int = "best") -> CheckpointRef:
        """A selector (best, latest, init, an alias, N or eN, optionally #sha) as a pinned ref."""
        text = str(which)
        which, _, sha = text.partition("#")
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

        root = self.path / "evals"
        pattern = f"{suite}/*.json" if suite else "*/*.json"
        return [EvalRecord.load(p) for p in sorted(root.glob(pattern))]

    def metrics(self, *, lineage: bool = False) -> list[dict]:
        rows = canon.read_jsonl(self.path / "metrics.jsonl")
        if not lineage:
            return rows
        parents = [load_experiment(ref.run) for ref in self.lineage() if ref.run and ref.run != self.ref]
        return [r for parent in reversed(parents) for r in parent.metrics()] + rows

    def lineage(self) -> list[CheckpointRef]:
        """The bound ``init_from`` checkpoints back to the root (nearest first)."""
        out: list[CheckpointRef] = []
        run: Run | None = self
        seen = set()
        while run is not None and run.ref not in seen:
            seen.add(run.ref)
            bound = (run.manifest.get("resolved") or {}).get("init_from") or {}
            refs = list(bound.values())
            if not refs:
                break
            parents = []
            for text in refs:
                ckpt = load_checkpoint(text)
                out.append(ckpt.ref)
                if ckpt.ref.run:
                    parents.append(ckpt.ref.run)
            run = load_experiment(parents[0]) if parents else None
        return out

    def provenance(self) -> list[dict]:
        return [canon.read_json(p) for p in sorted((self.path / "provenance").glob("launch*.json"),
                                                   key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))]


def load_experiment(ref: str | Path) -> Run:
    """``"exp/variant"``, ``"exp"`` (one variant), or a run directory."""
    path = Path(ref)
    if path.is_dir() and (path / "config.json").exists():
        return Run(path)
    run, which, sha = parse_ref(str(ref))
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
    import fnmatch

    root = runs_root()
    if "/" not in pattern:
        pattern = f"{pattern}/*"
    runs = [Run(p.parent) for p in sorted(root.glob("*/*/config.json"))
            if not p.parent.parent.name.startswith("_")]
    runs = [r for r in runs if fnmatch.fnmatchcase(r.ref, pattern)]
    return [r for r in runs if status is None or r.status == status]


def load_checkpoint(ref: str | Path) -> Checkpoint:
    """``"exp/var@which[#sha]"`` (default ``best``) or a ``.pt`` path."""
    text = str(ref)
    if text.endswith(".pt"):
        return Checkpoint.load(text)
    run, which, sha = parse_ref(text)
    selector = (which or "best") + (f"#{sha}" if sha else "")
    return load_experiment(run).checkpoint(selector)


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


def fork(ref: str, *, as_: str, overrides: Sequence[str] = ()) -> Variant:
    """A new variant continuing ``ref``'s latest checkpoint (weights and optimizer)."""
    run_ref, which, _ = parse_ref(ref)
    parent = load_experiment(run_ref)
    experiment, _, name = as_.rpartition("/")
    experiment = experiment or parent.manifest["experiment"]
    if not NAME_RE.match(name):
        raise RunError(f"bad variant name {name!r}")
    raw = merge(parent.config, parse_overrides(overrides))
    raw["init_from"] = {"ckpt": f"{parent.ref}@{which or 'latest'}", "parts": ["actor", "critic"],
                        "restore": "weights+optimizer"}
    cfg = resolve(raw, experiment=experiment, variant=name)
    variant = Variant(experiment=experiment, name=name, stage=None, labels={"fork": parent.ref}, config=cfg,
                      config_hash=config_hash(cfg), depends_on=())
    path = runs_root() / experiment / name
    if (path / "config.json").exists():
        raise RunError(f"{variant.ref} exists")
    _write_manifest(path, variant, source=None)
    return variant


def finish(ref: str) -> None:
    """End a stopped run (time_limit, interrupted, dead): its dependents may use it."""
    run = load_experiment(ref)
    if run.status not in ("time_limit", "interrupted", "dead", "failed"):
        raise RunError(f"{run.ref} is {run.status}; only a stopped run can be finished")
    if not _lock_free(run.path):
        raise RunLocked(f"{run.ref} is locked by a live process")
    state = run._run_json()
    state["status"] = "finished"
    state["finished_by"] = "catanrl finish"
    canon.atomic_write_json(run.path / "run.json", state)


def check_runs(pattern: str = "*") -> list[dict]:
    """Load, verify and self-test every alias and the latest checkpoint of every run."""
    rows = []
    for run in list_runs(pattern):
        selectors = sorted(set(run.aliases()) | ({"latest"} if run.checkpoints() else set()))
        for which in selectors:
            row = {"run": run.ref, "which": which, "ok": True, "error": None}
            try:
                ckpt = run.checkpoint(which)
                ckpt.self_test()
                row["ref"] = str(ckpt.ref)
            except Exception as error:  # report every failure, do not stop at the first
                row.update(ok=False, error=f"{type(error).__name__}: {error}")
            rows.append(row)
    return rows


def bind_refs(cfg: Mapping[str, Any], own: str | None = None) -> dict:
    """Refs to other runs, pinned once at creation: ``init_from`` per part and eval
    lineups (refs to ``own``, the run being created, resolve when its evals run)."""
    out: dict[str, Any] = {}
    init = cfg.get("init_from")
    if init:
        ckpt = init["ckpt"]
        parts = list(init.get("parts") or (ckpt.keys() if isinstance(ckpt, Mapping) else ("actor", "critic")))
        out["init_from"] = {part: str(load_checkpoint(ckpt[part] if isinstance(ckpt, Mapping) else ckpt).ref)
                            for part in parts}
    lineups = []
    for entry in (cfg.get("eval") or {}).get("final") or []:
        lineups.append([_bind_spec(spec, own) for spec in entry["lineup"]])
    if lineups:
        out["lineups"] = lineups
    return out


def _bind_spec(spec: str, own: str | None) -> str:
    """A lineup spec with another run's checkpoint pinned; own-run refs stay for eval time."""
    if "${" in spec or not spec.startswith(("POLICY", "MCTS", "BELIEF")):
        return spec
    if own and re.search(rf"(^|[:=]){re.escape(own)}(@|#|:|$)", spec):
        return spec
    from catanrl.agents import resolve_agent

    return resolve_agent(spec).canonical


def _write_manifest(path: Path, variant: Variant, *, source: Mapping[str, Any] | None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    canon.write_once_json(path / "config.json", {
        "schema": SCHEMA, "experiment": variant.experiment, "variant": variant.name, "stage": variant.stage,
        "labels": variant.labels, "depends_on": list(variant.depends_on), "config": variant.config,
        "config_hash": variant.config_hash, "resolved": bind_refs(variant.config, variant.ref), "source": source,
    })


# ---------------------------------------------------------------- writing


class RunWriter(Run):
    """The trainer's side of a run: lock, launches, checkpoints, commits, aliases.

    A context manager; leaving it by an exception records ``failed`` (or
    ``interrupted`` for KeyboardInterrupt) and releases the lock.
    """

    mode: Literal["new", "resume", "done"]

    def __init__(self, path: Path, variant: Variant, mode: str, launch: int, lock_fd: int) -> None:
        super().__init__(path)
        self.bound_variant = variant
        self.mode = mode  # type: ignore[assignment]
        self.launch = launch
        self.run_id = f"{self.ref}#launch{launch}"
        self.stop_reason: Literal["time_limit", "interrupted"] | None = None
        self._lock_fd = lock_fd
        self._pending: list[CheckpointRef] = []
        self._started = time.time()
        hours = (variant.config.get("runtime") or {}).get("time_limit_hours")
        self._deadline = None if hours is None else self._started + float(hours) * 3600
        self._stop_requested = False

    # ---------------------------------------------------------- opening

    @classmethod
    def open(cls, variant: Variant, *, argv: Sequence[str], allow_code_change: bool = False,
             source: Mapping[str, Any] | None = None) -> RunWriter:
        """Create, resume or reopen (``done``) the run of ``variant`` (DESIGN 6.4's table)."""
        from catanrl import provenance

        path = runs_root() / variant.experiment / variant.name
        path.mkdir(parents=True, exist_ok=True)
        fd = os.open(path / ".lock", os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise RunLocked(f"{variant.ref} is locked by a live process") from None
        try:
            return cls._open_locked(path, variant, argv, allow_code_change, source, fd, provenance)
        except BaseException:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            raise

    @classmethod
    def _open_locked(cls, path, variant, argv, allow_code_change, source, fd, provenance) -> RunWriter:
        kind = "train" if variant.kind == "train" else "eval"
        if (path / "config.json").exists():
            stored = canon.read_json(path / "config.json")
            old = migrate(stored["config"], from_schema=stored["schema"])
            if config_hash(old) != variant.config_hash:
                diff = "\n  ".join(config_diff(old, variant.config)) or "(hash differs)"
                raise ConfigConflict(f"{variant.ref} exists with another config:\n  {diff}\n"
                                     "use --variant NAME or catanrl fork")
        else:
            _write_manifest(path, variant, source=source)
        state = canon.read_json(path / "run.json") if (path / "run.json").exists() else None
        resolved = canon.read_json(path / "config.json").get("resolved") or {}
        bound = _bound_variant(variant, resolved)
        prov = provenance.capture(argv=argv, kind=kind)
        mixed = False
        if state is None:
            mode, launch = "new", 0
            if (variant.config.get("init_from") or {}).get("restore") == "weights+optimizer":
                _seed_fork_state(path, resolved)
                mode = "resume"
            state = {"status": None, "launches": [], "checkpoints": []}
        else:
            launch = len(state["launches"])
            mode = "done" if state["status"] == "finished" else "resume"
            if mode == "resume":
                first = canon.read_json(path / "provenance" / "launch0.json")
                diff = provenance.resume_diff({"code": first["code"], "python": first["python"]}, prov)
                if diff and not allow_code_change:
                    raise provenance.ProvenanceError(
                        f"{variant.ref}: the training code or uv.lock changed since launch 0:\n  " + "\n  ".join(diff)
                        + "\nrefusing to resume (--allow-code-change records a mixed run)")
                mixed = bool(diff)
        for warning in provenance.check(prov):
            print(f"warning: {warning}", flush=True)
        prov["run"]["launch_name"] = f"launch{launch}"
        provenance.save_patches(prov, path / "provenance")
        provenance.keep_commits(prov)
        canon.write_once_json(path / "provenance" / f"launch{launch}.json", prov)
        writer = cls(path, bound, mode, launch, fd)
        state["launches"].append({"launch": launch, "mode": mode, "started": time.time(), "status": "running",
                                  "argv": list(argv), "mixed_code": mixed})
        if mode != "done":
            state["status"] = "running"
        canon.atomic_write_json(path / "run.json", state)
        try:
            from catanrl._puffer import install_signal_handlers

            install_signal_handlers(writer.request_stop)
        except ValueError:  # not the main thread
            pass
        return writer

    def __enter__(self) -> RunWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if exc_type is not None:
                self.finish("interrupted" if issubclass(exc_type, KeyboardInterrupt) else "failed")
            elif self.mode == "done":
                self._end_launch("finished")
        finally:
            self.release()

    def release(self) -> None:
        if self._lock_fd is not None:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = None

    @property
    def variant_for_training(self) -> Variant:
        """The variant with ``init_from`` refs as bound at creation."""
        return self.bound_variant

    # ---------------------------------------------------------- PufferLib logger

    def log(self, logs: Mapping[str, Any], step: int) -> None:
        row = {k: v for k, v in logs.items() if isinstance(v, (int, float, str, bool)) or v is None}
        canon.append_jsonl(self.path / "metrics.jsonl", {"launch": self.launch, "step": int(step), **row})

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
        repo = prov["git"]["repos"]["catanrl-v2"]
        code = {"id": provenance.code_id("train"), "git": repo["head"], "dirty": repo["dirty"],
                "launch": self.launch}
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
        path = self.path / "trainer_state.pt"
        state = torch.load(path, map_location="cpu", weights_only=False) if path.exists() else None
        committed = -1 if state is None else int(state["epoch"])
        run = self._run_json()
        listed = {c["epoch"]: c for c in run["checkpoints"]}
        for file in sorted((self.path / "checkpoints").glob("e*.pt")):
            epoch = int(file.stem[1:])
            if epoch > committed:
                dest = self.path / "checkpoints" / "orphaned" / f"launch{self.launch}"
                dest.mkdir(parents=True, exist_ok=True)
                shutil.move(str(file), dest / file.name)
                listed.pop(epoch, None)
            elif epoch not in listed:
                ckpt = Checkpoint.load(file)
                listed[epoch] = {"epoch": epoch, "agent_steps": ckpt.header["agent_steps"], "file": file.name,
                                 "weights_sha256": ckpt.header["weights_sha256"], "launch": None}
        run["checkpoints"] = [listed[e] for e in sorted(listed)]
        if state is not None:
            run["trainer_state"] = {"epoch": committed, "launch": state.get("launch"), "time": time.time()}
        canon.atomic_write_json(self.path / "run.json", run)
        return state

    def pending_selects(self) -> list[CheckpointRef]:
        """Committed checkpoints whose select eval is missing, plus the last one
        (its select ran after the commit, so its alias decision may be lost)."""
        select = (self.config.get("eval") or {}).get("select")
        ckpts = self.checkpoints()
        if not select or not ckpts:
            return []
        from catanrl.eval.results import find

        every = int(select.get("every", 1))
        last = ckpts[-1].epoch
        out = []
        for ref in ckpts:
            if ref.epoch % every:
                continue
            lineup = [str(s).replace("${self}", str(ref)) for s in select["lineup"]]
            if ref.epoch == last or find(lineup, select["suite"], fill=select.get("fill")) is None:
                out.append(ref)
        return out

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

    def finish(self, status: Status) -> None:
        run = self._run_json()
        run["status"] = status
        if run["launches"]:
            run["launches"][-1].update(status=status, ended=time.time())
        canon.atomic_write_json(self.path / "run.json", run)

    def _end_launch(self, status: str) -> None:
        run = self._run_json()
        if run["launches"]:
            run["launches"][-1].update(status=status, ended=time.time())
        canon.atomic_write_json(self.path / "run.json", run)


def _bound_variant(variant: Variant, resolved: Mapping[str, Any]) -> Variant:
    init = variant.config.get("init_from")
    if not init or "init_from" not in resolved:
        return variant
    cfg = dict(variant.config)
    cfg["init_from"] = {**init, "ckpt": dict(resolved["init_from"]), "parts": list(resolved["init_from"])}
    return dataclasses.replace(variant, config=cfg)


def _seed_fork_state(path: Path, resolved: Mapping[str, Any]) -> None:
    """A fork restores weights and optimizer: start from the parent's committed trainer state."""
    refs = set((resolved.get("init_from") or {}).values())
    if len(refs) != 1:
        raise ConfigError("restore: weights+optimizer needs one checkpoint for every part")
    ckpt = load_checkpoint(refs.pop())
    if ckpt.ref.run is None:
        raise ConfigError("restore: weights+optimizer needs a run checkpoint, not a .pt file")
    parent = load_experiment(ckpt.ref.run)
    state_path = parent.path / "trainer_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if int(state["epoch"]) != ckpt.ref.epoch:
        raise ConfigError(f"{parent.ref}'s trainer state is at epoch {state['epoch']}, not e{ckpt.ref.epoch}: "
                          "fork @latest")
    shutil.copyfile(state_path, path / "trainer_state.pt")
