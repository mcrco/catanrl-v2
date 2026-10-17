"""Backfill (or re-sync) run directories into Weights & Biases.

Gives a run launched without the live mirror the same wandb run the mirror
would have made (one layout, one id: ``catanrl.wandb_mirror``), and brings a
mirrored run up to date. Idempotent: rows the wandb run already holds
(``summary[backfilled_through_agent_steps]``) are skipped, finals and the
status are rewritten.

::

    uv run python tools/wandb_backfill.py ppo-mixed/main --project catanrl-v2
    uv run python tools/wandb_backfill.py 'ppo-4p/*' --dry-run      # what would be logged
    WANDB_MODE=offline uv run python tools/wandb_backfill.py exp/v   # to runs/_wandb, `wandb sync` later

``--project`` defaults to ``CATANRL_WANDB_PROJECT``, then the run's
``meta.wandb.project``; ``--entity`` to ``CATANRL_WANDB_ENTITY``, then
``meta.wandb.entity``, then wandb's default. The runs come from
``$CATANRL_RUNS_DIR`` (the main worktree's ``runs/`` by default), by
``experiment/variant``, a glob over them, or a directory.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from catanrl import wandb_mirror
from catanrl.runs import Run, list_runs, load_experiment


def _runs(targets: list[str]) -> list[Run]:
    out: list[Run] = []
    for target in targets:
        if Path(target).is_dir():
            out.append(Run(target))
        elif any(ch in target for ch in "*?["):
            out += list_runs(target)
        else:
            out.append(load_experiment(target))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="experiment/variant, a glob over them, or a run directory")
    p.add_argument("--project", help="wandb project (default: CATANRL_WANDB_PROJECT, then meta.wandb.project)")
    p.add_argument("--entity", help="wandb entity (default: CATANRL_WANDB_ENTITY, then meta.wandb.entity)")
    p.add_argument("--group", help="wandb group (default: meta.wandb.group, then the experiment)")
    p.add_argument("--tag", action="append", default=[], help="an extra tag (repeatable); 'backfill' is always set")
    p.add_argument("--dry-run", action="store_true", help="only count what would be logged")
    args = p.parse_args(argv)

    failures = 0
    for run in _runs(args.runs):
        meta = (run.config.get("meta") or {}).get("wandb") or {}
        if isinstance(meta, str):
            meta = {"project": meta}
        project = args.project or os.environ.get("CATANRL_WANDB_PROJECT") or meta.get("project")
        if not project and args.dry_run:
            project = "?"
        if not project:
            print(f"{run.ref}: no project (--project, CATANRL_WANDB_PROJECT or meta.wandb.project)", file=sys.stderr)
            failures += 1
            continue
        settings = wandb_mirror.Settings(
            project=str(project), entity=args.entity or os.environ.get("CATANRL_WANDB_ENTITY") or meta.get("entity"),
            group=args.group or meta.get("group"), tags=tuple(str(t) for t in meta.get("tags") or ()))
        try:
            result = wandb_mirror.backfill(run, settings, tags=["backfill", *args.tag], dry_run=args.dry_run)
        except Exception as error:  # noqa: BLE001  (report every run; the rest still sync)
            print(f"{run.ref}: failed: {error!r}", file=sys.stderr)
            failures += 1
            continue
        verb = "would log" if result["dry_run"] else ("added" if result.get("resumed") else "logged")
        print(f"{run.ref} -> {project}/{result['id']}: {verb} {result['rows']} rows, {result['finals']} finals"
              + ("" if result["dry_run"] else f", through agent_steps {result['through']} ({run.status})"), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
