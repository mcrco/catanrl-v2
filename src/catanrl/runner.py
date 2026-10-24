"""Running plans: one subprocess per variant, stages in order, ``--next``.

``run_plan(plan)`` launches ``catanrl run ... --variant V --inline`` once per
variant whose run is not finished, in dependency order, each in a fresh
process (so a long runner never trains with code it imported hours ago).
``--next`` claims the first free variant whose dependencies are done, so
several terminals can share one plan. ``run_variant`` is what the child
does: open the run, train (``train.algo``), then play ``eval.final``.
"""

from __future__ import annotations

import contextlib
import os
import signal
import statistics
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from catanrl import canon
from catanrl.config import ConfigError, Plan, Variant
from catanrl.runs import (
    USABLE,
    RunLocked,
    RunWriter,
    Unbound,
    bound_lineup,
    check_stored,
    expand_init,
    list_runs,
    load_experiment,
    runs_root,
)


def _trainer(algo: str):
    if algo == "ppo":
        from catanrl.train.puffer import train_ppo

        return train_ppo
    if algo == "imitation":
        from catanrl.train.imitation import train_imitation

        return train_imitation
    if algo == "alphazero":
        from catanrl.train.alphazero import train_alphazero

        return train_alphazero
    raise NotImplementedError(f"train.algo {algo!r} has no trainer yet")


def _status(experiment: str, name: str) -> str | None:
    path = runs_root() / experiment / name
    return load_experiment(path).status if (path / "config.json").exists() else None


def run_variant(v: Variant, *, argv: Sequence[str], allow_code_change: bool = False,
                source: dict | None = None) -> str:
    """Create or resume ``v``'s run, train it, then play its final evals. Returns the status."""
    with RunWriter.open(v, argv=argv, allow_code_change=allow_code_change, source=source) as writer:
        if v.kind == "train" and writer.mode != "done":
            _trainer(v.config["train"]["algo"])(writer.variant_for_training, writer)
        if v.kind == "eval" or writer.status == "finished":
            final_evals(writer, v)
            if v.kind == "eval" and writer.status != "finished":
                writer.finish("finished")
        return writer.status


def final_evals(run: Any, v: Variant) -> list:
    """The plan's ``eval.final`` entries for ``run`` (refs to other runs as first pinned), cached.

    ``eval.final`` and ``runtime`` are outside the training config hash, so they
    come from the plan, not from the run's stored config."""
    from catanrl.eval.results import evaluate

    runtime = v.config.get("runtime") or {}
    options = {k: runtime[k] for k in ("device", "num_envs", "threads", "infer_rows") if k in runtime}
    records = []
    with canopy_release(runtime):
        for entry in v.config["eval"]["final"]:
            try:
                lineup = bound_lineup(run, entry["lineup"])
            except Unbound as error:  # a control of another experiment, not finished: a later run plays it
                print(f"{run.ref}: final deferred: {error}", flush=True)
                continue
            record = evaluate(lineup, entry["suite"], fill=entry.get("fill"), harness=entry.get("harness", "native"),
                              restart_partial=True, **options)
            print(f"{run.ref}: {record.data['specs'][0]} on {record.data['suite']}: "
                  f"{record.summary['score_rate']:.3f} (n={record.summary['games']}) {record.eval_id}", flush=True)
            records.append(record)
    return records


@contextlib.contextmanager
def canopy_release(runtime: Any):
    """``runtime.canopy_dir``: the Canopy build (``catanrl.canopy.release``) this plan's CANOPY
    agents play, as ``$CATANRL_CANOPY_DIR`` while its evals resolve and play; a relative path is
    under the runs root, like the default ``_canopy``. The build is in every CANOPY record's key
    (binary sha256 and backend), so naming it in the plan keeps a launch that forgot the variable
    from playing (and keying) another build. Refused when ``$CATANRL_CANOPY_DIR`` names another
    directory. Without it nothing changes."""
    value = (runtime or {}).get("canopy_dir")
    if not value:
        yield None
        return
    path = Path(os.path.expanduser(str(value)))
    path = (path if path.is_absolute() else runs_root() / path).resolve()
    before = os.environ.get("CATANRL_CANOPY_DIR")
    if before and Path(before).expanduser().resolve() != path:
        raise ConfigError(f"runtime.canopy_dir is {path}, but CATANRL_CANOPY_DIR is {before}: unset one")
    os.environ["CATANRL_CANOPY_DIR"] = str(path)
    try:
        yield path
    finally:
        if before is None:
            os.environ.pop("CATANRL_CANOPY_DIR", None)
        else:
            os.environ["CATANRL_CANOPY_DIR"] = before


def _missing_finals(run: Any, v: Variant) -> bool:
    """Whether a final of the plan has no record yet. One waiting for a run of another
    experiment (:class:`Unbound`) is not missing: nothing can play it now."""
    from catanrl.eval.results import find

    with canopy_release(v.config.get("runtime")):
        for entry in v.config["eval"]["final"]:
            try:
                lineup = bound_lineup(run, entry["lineup"], pin=False)
                if find(lineup, entry["suite"], fill=entry.get("fill"),
                        harness=entry.get("harness", "native")) is None:
                    return True
            except Unbound:
                continue
            except (RuntimeError, ValueError):  # unresolvable now: the child run reports why
                return True
    return False


def _check_source(plan: Plan) -> None:
    """Launches come from versioned YAML: the package's repo (DESIGN 6.7)."""
    from catanrl.provenance import repo_root

    if not plan.source.is_relative_to(repo_root()):
        raise ConfigError(f"{plan.source} is outside {repo_root()}; experiments launch from the repo's YAML")


def run_child(argv: Sequence[str]) -> tuple[int, bool]:
    """Run a variant's process; SIGINT/SIGTERM go to it, once each, and we wait.

    The child has its own process group, so a terminal Ctrl-C reaches it only
    through this forward: the first signal asks the trainer to stop at its next
    epoch (off-interval checkpoint, status interrupted), a second one interrupts
    it. Returns (exit code, whether a stop was asked)."""
    child = subprocess.Popen(list(argv), process_group=0)
    stopped = False

    def forward(signum, frame):
        nonlocal stopped
        stopped = True
        with contextlib.suppress(ProcessLookupError):
            child.send_signal(signum)

    previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        while True:
            try:
                return child.wait(), stopped
            except InterruptedError:
                continue
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _child_argv(plan: Plan, v: Variant, overrides: Sequence[str], allow_code_change: bool) -> list[str]:
    argv = [sys.executable, "-P", "-m", "catanrl.cli", "run", str(plan.source), "--variant", v.name, "--inline"]
    for item in overrides:
        argv += ["--set", item]
    if plan.smoke:
        argv.append("--smoke")
    if allow_code_change:
        argv.append("--allow-code-change")
    return argv


def run_plan(plan: Plan, *, next_: bool = False, allow_code_change: bool = False,
             overrides: Sequence[str] = ()) -> None:
    """Run every unfinished variant in order, resuming stopped ones. With ``next_``,
    claim only new or dead variants (and finished ones missing final evals), one
    at a time, until none is left: a time-limited, interrupted or failed run waits
    for an explicit ``catanrl run``."""
    _check_source(plan)
    while True:
        ran = False
        for v in plan.variants:
            run = check_stored(v)  # a changed config is a conflict, whatever the status
            status = None if run is None else run.status
            if status == "finished" and not _missing_finals(run, v):
                continue
            if next_ and status not in (None, "dead", "finished"):
                continue
            blocked = [d for d in v.depends_on if _status(v.experiment, d) not in USABLE]
            if blocked:
                if not next_:
                    raise RuntimeError(f"{v.ref} waits for {blocked}; run them first (or use --next)")
                continue
            print(f"== {v.ref}", flush=True)
            code, stopped = run_child(_child_argv(plan, v, overrides, allow_code_change))
            if stopped:
                raise SystemExit(130)  # the child stopped gracefully; do not start the next variant
            if code == 3 and next_:
                continue  # locked: someone else claimed it
            if code != 0:
                raise SystemExit(code)
            ran = True
            if next_:
                break
        if not next_ or not ran:
            return


def run_inline(plan: Plan, name: str, *, argv: Sequence[str], allow_code_change: bool = False) -> int:
    """The child's side: run one variant; exit 3 when it is locked."""
    _check_source(plan)
    try:
        run_variant(plan.variant(name), argv=argv, allow_code_change=allow_code_change,
                    source={"path": str(plan.source), "sha256": plan.source_sha256})
    except RunLocked as error:
        print(f"skip: {error}", flush=True)
        return 3
    return 0


# ---------------------------------------------------------------- summaries


def summarize(experiment: str) -> Path:
    """``runs/<experiment>/summary.md``: one row per variant, then each final eval
    entry's mean ± sd over seeds (only the runs' own final records, this code's)."""
    from catanrl.eval.results import find

    runs = list_runs(f"{experiment}/*")
    if not runs:
        raise RuntimeError(f"no runs of {experiment}")
    lines = [f"# {experiment}", "", "| variant | status | epoch | best | final |", "|---|---|---|---|---|"]
    groups: dict[tuple, list[float]] = {}
    for run in runs:
        ckpts, best, known = run.checkpoints(), run.aliases().get("best"), run.bindings()
        finals = []
        for i, entry in enumerate(run.config["eval"]["final"]):
            try:
                with canopy_release(run.config.get("runtime")):
                    record = find([known.get(s) or expand_init(s, run.init_ref) for s in entry["lineup"]],
                                  entry["suite"], fill=entry.get("fill"), harness=entry.get("harness", "native"))
            except (RuntimeError, ValueError):
                record = None
            if record is not None:
                finals.append((i, _lineup_label(entry["lineup"]), record))
        final = "; ".join(f"{key} {r.data['suite']} {r.summary['score_rate']:.3f}" for _, key, r in finals)
        best_text = f"e{best['epoch']} {best['metric'] or 0:.3f}" if best else "-"
        lines.append(f"| {run.manifest['variant']} | {run.status} | {ckpts[-1].epoch if ckpts else '-'} | "
                     f"{best_text} | {final or '-'} |")
        labels = {k: v for k, v in run.manifest.get("labels", {}).items() if k != "seed"}
        cell = ", ".join(f"{k}={v}" for k, v in labels.items()) + (f" .{run.manifest['stage']}"
                                                                  if run.manifest.get("stage") else "")
        for i, key, r in finals:
            groups.setdefault((cell or "-", f"{i}:{key}", r.data["suite"]), []).append(r.summary["score_rate"])
    if groups:
        lines += ["", "| cell (over seeds) | final | suite | mean | sd | n |", "|---|---|---|---|---|---|"]
        for (cell, entry, suite), rates in sorted(groups.items()):
            sd = statistics.stdev(rates) if len(rates) > 1 else float("nan")
            lines.append(f"| {cell} | {entry} | {suite} | {statistics.mean(rates):.3f} | {sd:.3f} | {len(rates)} |")
    path = runs_root() / experiment / "summary.md"
    canon.atomic_write_bytes(path, ("\n".join(lines) + "\n").encode())
    return path


def _lineup_label(lineup: list[str]) -> str:
    """``MCTS vs POLICY(init)``: each slot's key, marking the run's start."""
    return " vs ".join(str(s).split(":")[0] + ("(init)" if "${init}" in str(s) else "") for s in lineup)
