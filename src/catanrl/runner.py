"""Running plans: one subprocess per variant, stages in order, ``--next``.

``run_plan(plan)`` launches ``catanrl run ... --variant V --inline`` once per
variant whose run is not finished, in dependency order, each in a fresh
process (so a long runner never trains with code it imported hours ago).
``--next`` claims the first free variant whose dependencies are done, so
several terminals can share one plan. ``run_variant`` is what the child
does: open the run, train (``train.algo``), then play ``eval.final``.
"""

from __future__ import annotations

import statistics
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from catanrl import canon
from catanrl.config import Plan, Variant
from catanrl.runs import RunLocked, RunWriter, list_runs, load_experiment, runs_root

#: Statuses a dependent stage may start from (a time_limit run pins its epoch).
USABLE = ("finished", "time_limit")


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


def _status(v: Variant) -> str | None:
    path = runs_root() / v.experiment / v.name
    return load_experiment(path).status if (path / "config.json").exists() else None


def run_variant(v: Variant, *, argv: Sequence[str], allow_code_change: bool = False,
                source: dict | None = None) -> str:
    """Create or resume ``v``'s run, train it, then play its final evals. Returns the status."""
    with RunWriter.open(v, argv=argv, allow_code_change=allow_code_change, source=source) as writer:
        cfg = writer.variant_for_training.config
        if v.kind == "train" and writer.mode != "done":
            _trainer(cfg["train"]["algo"])(writer.variant_for_training, writer)
        status = writer.status
        if v.kind == "eval" or status == "finished":
            final_evals(writer)
            if v.kind == "eval" and status != "finished":
                writer.finish("finished")
        return writer.status


def final_evals(run: Any) -> list:
    """Every ``eval.final`` entry (bound lineups for refs to other runs), cached."""
    from catanrl.eval.results import evaluate

    cfg = run.config
    runtime = cfg.get("runtime") or {}
    options = {k: runtime[k] for k in ("device", "num_envs", "threads", "infer_rows") if k in runtime}
    bound = (run.manifest.get("resolved") or {}).get("lineups") or []
    records = []
    for i, entry in enumerate(cfg["eval"]["final"]):
        lineup = bound[i] if i < len(bound) else entry["lineup"]
        record = evaluate(lineup, entry["suite"], fill=entry.get("fill"), **options)
        print(f"{run.ref}: {record.data['specs'][0]} on {record.data['suite']}: "
              f"{record.summary['score_rate']:.3f} (n={record.summary['games']}) {record.eval_id}", flush=True)
        records.append(record)
    return records


def _child_argv(plan: Plan, v: Variant, overrides: Sequence[str], allow_code_change: bool) -> list[str]:
    argv = [sys.executable, "-m", "catanrl.cli", "run", str(plan.source), "--variant", v.name, "--inline"]
    for item in overrides:
        argv += ["--set", item]
    if plan.smoke:
        argv.append("--smoke")
    if allow_code_change:
        argv.append("--allow-code-change")
    return argv


def run_plan(plan: Plan, *, next_: bool = False, allow_code_change: bool = False,
             overrides: Sequence[str] = ()) -> None:
    """Run every unfinished variant (``next_``: claim free ones until none is left)."""
    while True:
        ran = False
        for v in plan.variants:
            status = _status(v)
            if status == "finished" and not _missing_finals(v):
                continue
            blocked = [d for d in v.depends_on if _status(plan.variant(d)) not in USABLE]
            if blocked:
                if not next_:
                    raise RuntimeError(f"{v.ref} waits for {blocked}; run them first (or use --next)")
                continue
            if next_ and status == "running":
                continue  # another terminal has it
            print(f"== {v.ref}", flush=True)
            result = subprocess.run(_child_argv(plan, v, overrides, allow_code_change), check=False)
            if result.returncode == 3 and next_:
                continue  # locked: someone else claimed it
            if result.returncode != 0:
                raise SystemExit(result.returncode)
            ran = True
            if next_:
                break
        if not next_ or not ran:
            return


def _missing_finals(v: Variant) -> bool:
    from catanrl.eval.results import find

    run = load_experiment(runs_root() / v.experiment / v.name)
    bound = (run.manifest.get("resolved") or {}).get("lineups") or []
    for i, entry in enumerate(run.config["eval"]["final"]):
        lineup = bound[i] if i < len(bound) else entry["lineup"]
        try:
            if find(lineup, entry["suite"], fill=entry.get("fill")) is None:
                return True
        except Exception:
            return True
    return False


def run_inline(plan: Plan, name: str, *, argv: Sequence[str], allow_code_change: bool = False) -> int:
    """The child's side: run one variant; exit 3 when it is locked."""
    v = plan.variant(name)
    try:
        run_variant(v, argv=argv, allow_code_change=allow_code_change,
                    source={"path": str(plan.source), "sha256": plan.source_sha256})
    except RunLocked as error:
        print(f"skip: {error}", flush=True)
        return 3
    return 0


# ---------------------------------------------------------------- summaries


def summarize(experiment: str) -> Path:
    """``runs/<experiment>/summary.md``: one row per variant, then mean ± sd over seeds."""
    runs = list_runs(f"{experiment}/*")
    if not runs:
        raise RuntimeError(f"no runs of {experiment}")
    lines = [f"# {experiment}", "", "| variant | status | epoch | best | final |", "|---|---|---|---|---|"]
    groups: dict[str, list[float]] = {}
    for run in runs:
        ckpts = run.checkpoints()
        best = run.aliases().get("best")
        finals = [r for r in run.evals() if r.data["specs"][0].startswith(("POLICY", "MCTS", "BELIEF"))
                  and load_suite_purpose(r) == "heldout"]
        final = "; ".join(f"{r.data['suite']} {r.summary['score_rate']:.3f}" for r in finals)
        lines.append(f"| {run.manifest['variant']} | {run.status} | {ckpts[-1].epoch if ckpts else '-'} | "
                     f"{'e%d %.3f' % (best['epoch'], best['metric'] or 0) if best else '-'} | {final or '-'} |")
        labels = {k: v for k, v in run.manifest.get("labels", {}).items() if k != "seed"}
        key = ", ".join(f"{k}={v}" for k, v in labels.items()) + (f" .{run.manifest['stage']}"
                                                                 if run.manifest.get("stage") else "")
        for r in finals:
            groups.setdefault(f"{key} | {r.data['suite']}", []).append(r.summary["score_rate"])
    if groups:
        lines += ["", "| cell (over seeds) | suite | mean | sd | n |", "|---|---|---|---|---|"]
        for key, rates in sorted(groups.items()):
            cell, suite = key.split(" | ")
            sd = statistics.stdev(rates) if len(rates) > 1 else float("nan")
            lines.append(f"| {cell or '-'} | {suite} | {statistics.mean(rates):.3f} | {sd:.3f} | {len(rates)} |")
    path = runs_root() / experiment / "summary.md"
    canon.atomic_write_bytes(path, ("\n".join(lines) + "\n").encode())
    return path


def load_suite_purpose(record: Any) -> str:
    return record.key["suite"]["purpose"]
