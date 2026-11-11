"""Smoke fp16-2p (configs/experiments/fp16-2p.yaml) on the GPU in a scratch runs directory.

Both variants of the real config (graph-m, 1,032 envs, Muon, torch.compile, max_minibatch_size
2048) train for ``--epochs`` epochs each (``train.stop_epoch``: the cosine is still the 3B
budget's), fp32 then fp16, each in its own process as the runner does (so torch.compile, the
CUDA caches and the peak-memory counter start fresh), through ``catanrl.runner.run_variant``. The
champion monitor plays its suite's first ``--boards`` boards every ``--monitor-every`` epochs (0
boards: no monitor, no select), so the evals run beside a float16 trainer; the finals play only
with ``--finals N`` (their first N boards). Then a summary per arm from its metrics.jsonl
(``summary.json`` in the runs directory too):

* ``steps_per_s``: median throughput/agent_steps_per_s after the first ``--warmup`` epochs
  (torch.compile), and fp16's ratio to fp32 (the decision rule wants 1.15x over epochs 50-300);
* ``peak_GiB``: memory/peak_GiB (fp32 graph-m-2p-lr: 4.3 GiB at e300);
* ``finite``: every losses/* finite in every epoch;
* ``approx_kl_e1``: losses/approx_kl at the first update; fp16 adds ``probe_kl_e1`` and
  ``probe_gap_max``: the training forward against the rollout's stored log-probs on the same
  rows before any update (float16's two forwards round differently; expect well under the
  approx_kl of an update, ~0.02);
* fp16: ``skipped`` (all, and after the warm-up; a few at the start while the scale finds its
  level from 65536 are normal), ``optimizer_steps`` (16 an epoch) and the scale at the end.

The runs directory must not be the real one: fr3's run (``ppo-2p-fr3/full``: its config,
run.json, aliases and the e4350 checkpoint, read-only), which the config pins as its champion, is
copied into it. W&B is off (``WANDB_MODE=disabled``, ``meta.wandb`` dropped).

    # GPU (free card; ~15 min: 2 x 20 epochs at ~22 s (fp32) / less (fp16), plus compile)
    env -u WANDB_API_KEY WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        uv run python tools/fp16_smoke.py --runs /tmp/fp16-smoke
    # CPU check of the tool itself (tiny, a minute)
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 uv run python tools/fp16_smoke.py --runs /tmp/fp16-cpu \\
        --epochs 3 --warmup 1 --boards 2 --monitor-every 2 --set runtime.device=cpu --set runtime.threads=2 \\
        --set runtime.compile=false --set env.num_envs=8 --set train.bptt_horizon=8 \\
        --set train.minibatch_size=64 --set train.max_minibatch_size=32 --set train.total_timesteps=512
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

START = ("ppo-2p-fr3", "full", "e004350.pt")
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "fp16-2p.yaml"
ARMS = ("fp32", "fp16")


def _variant(name: str, args: argparse.Namespace):
    """The plan's variant with the smoke's epochs, monitor and finals (its hash recomputed)."""
    from catanrl.config import config_hash, load_plan

    plan = load_plan(CONFIG, overrides=[f"train.stop_epoch={args.epochs}", *args.overrides])
    v = plan.variant(name)
    cfg = json.loads(json.dumps(v.config))
    cfg["meta"].pop("wandb", None)
    ev = cfg["eval"]
    if args.boards > 0:
        (monitor,) = ev["monitor"]
        monitor.update(suite={"name": monitor["suite"], "boards": args.boards}, every=args.monitor_every)
        ev["select"] = {**{k: x for k, x in monitor.items() if k != "name"}, "from": monitor["name"]}
    else:
        ev.pop("monitor", None)
        ev["select"] = None
    ev["final"] = [{**e, "suite": {"name": e["suite"], "boards": args.finals}} for e in ev["final"]] \
        if args.finals > 0 else []
    return plan, dataclasses.replace(v, config=cfg, config_hash=config_hash(cfg))


def _child(name: str, args: argparse.Namespace) -> None:
    from catanrl.runner import run_variant

    plan, v = _variant(name, args)
    status = run_variant(v, argv=["tools/fp16_smoke.py", *sys.argv[1:]],
                         source={"path": str(CONFIG), "sha256": plan.source_sha256})
    print(f"== {v.ref}: {status}", flush=True)
    if status != "finished":
        raise SystemExit(1)


def _summary(name: str, warmup: int) -> dict:
    from catanrl.runs import load_experiment

    rows = [r for r in load_experiment(f"fp16-2p/{name}").metrics() if "throughput/agent_steps_per_s" in r]
    late = [r for r in rows if r["epoch"] > warmup] or rows
    losses = [v for r in rows for k, v in r.items() if k.startswith("losses/") and k != "losses/explained_variance"]
    out = {"epochs": len(rows),
           "steps_per_s": statistics.median(r["throughput/agent_steps_per_s"] for r in late),
           "peak_GiB": max(r.get("memory/peak_GiB", float("nan")) for r in rows),
           "finite": all(isinstance(v, (int, float)) and math.isfinite(v) for v in losses),
           "approx_kl_e1": rows[0].get("losses/approx_kl"),
           "approx_kl_median": statistics.median(r["losses/approx_kl"] for r in rows)}
    if "amp/scale" in rows[0]:
        out.update(skipped=rows[-1]["amp/skipped_total"],
                   skipped_after_warmup=sum(r["amp/skipped_steps"] for r in late),
                   optimizer_steps=sum(r["amp/optimizer_steps"] for r in rows),
                   scale_first=rows[0]["amp/scale"], scale_last=rows[-1]["amp/scale"],
                   probe_kl_e1=rows[0].get("amp/probe_kl"),
                   probe_kl_max=max(r.get("amp/probe_kl", 0.0) for r in rows),
                   probe_gap_max=max(r.get("amp/probe_logprob_gap_max", 0.0) for r in rows))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", required=True, help="a scratch runs directory (created; never the real one)")
    parser.add_argument("--real-runs", default=None, help="where to copy fr3's run from (default: the main "
                        "worktree's runs/)")
    parser.add_argument("--epochs", type=int, default=20, help="epochs per arm (train.stop_epoch)")
    parser.add_argument("--warmup", type=int, default=3, help="first epochs left out of the steps/s median")
    parser.add_argument("--boards", type=int, default=20, help="champion monitor on its suite's first N boards "
                        "(0: no monitor or select)")
    parser.add_argument("--monitor-every", type=int, default=10, help="epochs between monitors")
    parser.add_argument("--finals", type=int, default=0, help="finals on their suite's first N boards (0: none)")
    parser.add_argument("--only", default=None, help="arms to run (comma-separated: fp32,fp16)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--child", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    os.environ["WANDB_MODE"] = "disabled"
    for key in ("WANDB_API_KEY", "CATANRL_WANDB_PROJECT", "CATANRL_WANDB_ENTITY"):
        os.environ.pop(key, None)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    scratch = Path(args.runs).resolve()
    if args.child is not None:
        os.environ["CATANRL_RUNS_DIR"] = str(scratch)
        _child(args.child, args)
        return

    os.environ.pop("CATANRL_RUNS_DIR", None)
    from catanrl.runs import runs_root

    real = Path(args.real_runs).resolve() if args.real_runs else runs_root()
    if scratch == real or real in scratch.parents or scratch in real.parents:
        raise SystemExit(f"--runs {scratch} overlaps the real runs directory {real}")
    experiment, variant, ckpt = START
    source, target = real / experiment / variant, scratch / experiment / variant
    if not (target / "checkpoints" / ckpt).exists():
        (target / "checkpoints").mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "run.json", "aliases.jsonl"):
            shutil.copyfile(source / name, target / name)
        shutil.copyfile(source / "checkpoints" / ckpt, target / "checkpoints" / ckpt)
    os.environ["CATANRL_RUNS_DIR"] = str(scratch)

    arms = [a for a in ARMS if args.only is None or a in args.only.split(",")]
    for arm in arms:
        child = [sys.executable, "-P", __file__, *(argv if argv is not None else sys.argv[1:]), "--child", arm]
        if subprocess.run(child, env=os.environ.copy(), check=False).returncode != 0:
            raise SystemExit(f"{arm} failed")
    summary = {arm: _summary(arm, args.warmup) for arm in arms}
    if set(ARMS) <= set(summary):
        summary["fp16_over_fp32_steps_per_s"] = summary["fp16"]["steps_per_s"] / summary["fp32"]["steps_per_s"]
    (scratch / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
