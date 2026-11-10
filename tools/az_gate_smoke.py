"""Smoke az-gate-2p (configs/experiments/az-gate-2p.yaml) end to end in a scratch runs directory.

Every variant of the real config runs in order (positions, then the three fits) through
``catanrl.runner.run_variant``, the child process's own path, with ``--set`` overrides for size,
and with the finals cut to their suite's first ``--boards`` boards (``--boards 0``: none). Per-arm
finals live in the sweep, so ``catanrl run --set`` cannot shrink them; this does, and nothing else.

The runs directory must not be the real one: the start's run (``ppo-2p-fr3/full``: its config,
run.json, aliases and the e4350 checkpoint, read-only) is copied into it so the config's pinned
refs resolve, and wandb is off (``WANDB_MODE=disabled``; a smoke must never reach the real runs).

    # CPU (2 threads, no finals): ~1 min
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 uv run python tools/az_gate_smoke.py --runs /tmp/az-gate-cpu \\
        --boards 0 --set runtime.device=cpu --set runtime.threads=2 --set train.kernels=torch ...
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import shutil
import sys
from pathlib import Path

START = ("ppo-2p-fr3", "full", "e004350.pt")
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "az-gate-2p.yaml"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", required=True, help="a scratch runs directory (created; never the real one)")
    parser.add_argument("--real-runs", default=None, help="where to copy the start's run from (default: the "
                        "main worktree's runs/)")
    parser.add_argument("--boards", type=int, default=0, help="finals on their suite's first N boards (0: none)")
    parser.add_argument("--only", default=None, help="run only these variants (comma-separated), in plan order")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    os.environ["WANDB_MODE"] = "disabled"
    os.environ.pop("WANDB_API_KEY", None)
    os.environ.pop("CATANRL_WANDB_PROJECT", None)
    os.environ.pop("CATANRL_RUNS_DIR", None)
    from catanrl.runs import runs_root

    real = Path(args.real_runs).resolve() if args.real_runs else runs_root()
    scratch = Path(args.runs).resolve()
    if scratch == real or real in scratch.parents or scratch in real.parents:
        raise SystemExit(f"--runs {scratch} overlaps the real runs directory {real}")
    os.environ["CATANRL_RUNS_DIR"] = str(scratch)

    experiment, variant, ckpt = START
    source, target = real / experiment / variant, scratch / experiment / variant
    if not (target / "checkpoints" / ckpt).exists():
        (target / "checkpoints").mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "run.json", "aliases.jsonl"):
            shutil.copyfile(source / name, target / name)
        shutil.copyfile(source / "checkpoints" / ckpt, target / "checkpoints" / ckpt)

    from catanrl.config import load_plan
    from catanrl.runner import run_variant

    plan = load_plan(CONFIG, overrides=args.overrides)
    only = None if args.only is None else set(args.only.split(","))
    for v in plan.variants:
        if only is not None and v.name not in only:
            continue
        finals = []
        if args.boards > 0:
            for entry in v.config["eval"]["final"]:
                suite = entry["suite"] if isinstance(entry["suite"], dict) else {"name": entry["suite"]}
                finals.append({**entry, "suite": {**suite, "boards": min(int(suite.get("boards", args.boards)),
                                                                          args.boards)}})
        v = dataclasses.replace(v, config={**v.config, "eval": {**v.config["eval"], "final": finals}})
        status = run_variant(v, argv=["tools/az_gate_smoke.py", *(argv if argv is not None else sys.argv[1:])],
                             source={"path": str(CONFIG), "sha256": plan.source_sha256})
        print(f"== {v.ref}: {status}", flush=True)
        if status != "finished":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
