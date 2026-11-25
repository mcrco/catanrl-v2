"""AlphaZero self-play with and without the evaluation cache (``runtime.eval_cache_mb``), side by side.

One network (``--ckpt``: a ``.pt`` path, read-only) plays ``--games`` games against itself with
a run's stored settings (``--config``: its ``config.json``, read-only; the train section's
searches, the env's scenarios, the seed's map pool), once per cache size of ``--cache-mb`` (0: no
cache), each in a SelfPlay of its own on the same seeds, one after another in this process:

1. ``--warm`` steps in which every decision is a ``--warm-sims`` fast search, without a cache:
   the games leave their openings (a run's games are at every stage; new ones all place
   settlements). Not timed. With the same seeds every size starts from the same positions.
2. The trees are dropped (an iteration's start: the cache is empty, as after a learning step)
   and ``--steps`` steps are timed with the cache attached: one iteration of a run at the
   default 64. A progress line per ``--every`` steps (rate and ETA).

Printed per size and written to ``--out`` (JSON): the leaves searched a second and a step, the
share of them that took a row in a batch (``forwarded``; the rest took their outputs from the
table or from another leaf's row of their round), seconds a step, decisions a second, network
rounds a step, the table's entries, the share of the rows it stored that displaced a row still
in use, the device's peak memory (allocated and reserved, per size) and the host's resident
memory (after the size's run, and the process's peak so far).

With ``--mode fast`` (a run's) a row's outputs depend on its batch's size in their last bits, so
the games of two sizes part ways after a while and the leaves a step differ a little: compare
the seconds a step and the decisions a second. With ``--mode strict --kernels fused`` (or
``torch``) a row's outputs are its own, the games must be the same, and ``same games`` says
whether every size played the moves of the first to the last one (the exit code is 1 if not).

Nothing is written but ``--out``; W&B and the runs directory are not touched. The card should
be free (not beside a training run), and the fused kernels build with nvcc and ninja on PATH.

    # the live run's settings, no cache against 256 MB: about 10 minutes on this box (an estimate: per size some
    # 3 minutes of timed steps at 2.7 s a step, a minute of warm-up and the kernels' build)
    env -u WANDB_API_KEY WANDB_MODE=disabled PATH=<a directory with ninja>:/opt/cuda/bin:$PATH \\
        .venv/bin/python tools/eval_cache_check.py \\
        --config runs/az-2p-graph-m-uniform/main+300/config.json \\
        --ckpt runs/az-2p-graph-m-uniform/main/checkpoints/e000900.pt --cache-mb 0 256
    # the same searches, to the bit (slower: every forward is 512 rows)
    ... --mode strict --kernels fused --steps 8
    # CPU check of the tool itself (a few minutes)
    CUDA_VISIBLE_DEVICES= .venv/bin/python tools/eval_cache_check.py --config ... --ckpt ... --device cpu \\
        --kernels torch --games 8 --threads 4 --warm 4 --steps 2 --cache-mb 0 16
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from catanrl.envs.puffer import plan_env
from catanrl.runs import load_checkpoint
from catanrl.seeding import derive_seed
from catanrl.train.alphazero import SEARCH_GROUPS, ReplayBuffer, SelfPlay, check_train


def resident() -> tuple[float, float]:
    """The process's resident memory now and at its peak, in MB."""
    fields = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
    return int(fields["VmRSS"].split()[0]) / 1024, int(fields["VmHWM"].split()[0]) / 1024


def counters() -> dict[str, float]:
    return {"leaves": 0.0, "rounds": 0.0, "full": 0.0, "depth": 0.0, "max_depth": 0.0}


def measure(args: argparse.Namespace, cfg: dict, megabytes: float) -> tuple[dict, np.ndarray]:
    """One size's run: its numbers and the moves of its timed steps."""
    name = f"{megabytes:g} MB" if megabytes else "no cache"
    cuda = torch.device(args.device).type == "cuda"
    train = check_train(cfg["train"])
    train["kernels"] = args.kernels or train["kernels"]
    runtime = cfg.get("runtime") or {}
    threads = args.threads or int(runtime.get("threads", 1))
    games = args.games or int(cfg["env"]["num_envs"])
    plan = plan_env(dict(cfg["env"], num_envs=games), pool_seed=derive_seed(int(cfg["seed"]), "map_pool"))
    model = load_checkpoint(args.ckpt).model("all", args.device).eval()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    play = SelfPlay(plan, train, model=model, num_envs=games, threads=threads,
                    seed=derive_seed(args.seed, "rollout", 0), device=args.device,
                    mode=args.mode or runtime.get("determinism", "fast"),
                    groups=args.groups or int(runtime.get("search_groups") or SEARCH_GROUPS), cache_mb=megabytes)
    replay = ReplayBuffer(4096, play.vec.observation_size, play.vec.action_size, play.players,
                          outcome=play.outcome, search=play.search)
    play.check_kernels()

    # 1. out of the openings: cheap fast searches for every decision, no cache
    for pool in play.pools:
        pool.cache = None
    saved = (play.train["full_search_prob"], play.train.get("decided_full_search_prob", 0.0),
             play.options[1]["simulations"], play.decided_options["simulations"])
    play.train["full_search_prob"], play.train["decided_full_search_prob"] = 1e-12, 0.0
    play.options[1]["simulations"] = play.decided_options["simulations"] = args.warm_sims
    rng, start = np.random.default_rng(derive_seed(args.seed, "warm")), time.time()
    for step in range(args.warm):
        play.step(-1, step, rng, replay, counters())
        if (step + 1) % (5 * args.every) == 0 or step + 1 == args.warm:
            took = time.time() - start
            print(f"  {name}: warm-up {step + 1}/{args.warm}, {took:.0f} s, "
                  f"ETA {took / (step + 1) * (args.warm - step - 1):.0f} s", flush=True)
    (play.train["full_search_prob"], play.train["decided_full_search_prob"],
     play.options[1]["simulations"], play.decided_options["simulations"]) = saved
    play.drop_trees()
    if play.cache is not None:
        for pool in play.pools:
            pool.cache = play.cache.table

    # 2. the timed steps: an iteration's start
    rng, stats, moves = np.random.default_rng(derive_seed(args.seed, "selfplay", 0)), counters(), []
    decisions, finished, before = play.decisions, play.games, play.evaluator.exact_forwards
    start = time.time()
    for step in range(args.steps):
        play.step(0, step, rng, replay, stats)
        moves.append(play.buffers["actions"].copy())
        if (step + 1) % args.every == 0 or step + 1 == args.steps:
            took = time.time() - start
            print(f"  {name}: step {step + 1}/{args.steps}, {stats['leaves'] / took:,.0f} leaves/s, "
                  f"{100 * (1 - stats.get('cached', 0.0) / max(stats['leaves'], 1)):.1f}% forwarded, "
                  f"{took / (step + 1):.2f} s/step, ETA {took / (step + 1) * (args.steps - step - 1):.0f} s",
                  flush=True)
    took = time.time() - start
    table = play.cache.table.stats if play.cache is not None else None
    now, peak = resident()
    row = {
        "cache_mb": megabytes, "steps": args.steps, "games": games, "seconds": took,
        "seconds_per_step": took / args.steps, "leaves": stats["leaves"], "leaves_per_s": stats["leaves"] / took,
        "leaves_per_step": stats["leaves"] / args.steps,
        "forwarded": 1.0 - stats.get("cached", 0.0) / max(stats["leaves"], 1.0),
        "forwarded_per_s": (stats["leaves"] - stats.get("cached", 0.0)) / took,
        "decisions_per_s": (play.decisions - decisions) / took, "rounds_per_step": stats["rounds"] / args.steps,
        "games_finished": play.games - finished, "full_searches": stats["full"],
        "float32_chunks": play.evaluator.exact_forwards - before,  # fused16: forwards run again in float32
        "table_entries": int(table.entries) if table else 0,
        "table_capacity": int(play.cache.table.capacity) if table else 0,
        "table_mb": play.cache.table.bytes / 2**20 if table else 0.0,
        "in_round_repeats": table.repeats / max(table.leaves, 1) if table else 0.0,
        "replaced": table.replaced / max(table.stored, 1) if table else 0.0,
        "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20 if cuda else 0.0,
        "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20 if cuda else 0.0,
        "host_rss_mb": now, "host_peak_rss_mb": peak,
    }
    play.drop_trees()
    del play, model, replay
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    return row, np.stack(moves)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="a run's config.json (its train, env and seed)")
    parser.add_argument("--ckpt", required=True, help="the network: a .pt path")
    parser.add_argument("--cache-mb", type=float, nargs="+", default=[0.0, 256.0], help="cache sizes; 0: none")
    parser.add_argument("--steps", type=int, default=64, help="timed self-play steps per size")
    parser.add_argument("--warm", type=int, default=150, help="untimed fast-search steps before them")
    parser.add_argument("--warm-sims", type=int, default=16)
    parser.add_argument("--every", type=int, default=4, help="steps per progress line")
    parser.add_argument("--games", type=int, default=0, help="games side by side (0: the config's env.num_envs)")
    parser.add_argument("--threads", type=int, default=0, help="engine threads (0: the config's runtime.threads)")
    parser.add_argument("--groups", type=int, default=0, help="search groups (0: the config's, or the default)")
    parser.add_argument("--mode", choices=["fast", "strict"], default=None, help="default: runtime.determinism")
    parser.add_argument("--kernels", choices=["torch", "fused", "fused16"], default=None,
                        help="default: train.kernels")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="a JSON file for the rows")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())["config"]
    rows, moves = [], []
    for megabytes in args.cache_mb:
        row, played = measure(args, cfg, float(megabytes))
        rows.append(row)
        moves.append(played)
    print(f"\n{'cache':>9} {'leaves/s':>9} {'forwarded':>9} {'rows/s':>8} {'s/step':>7} {'decisions/s':>11} "
          f"{'rounds/step':>11} {'entries':>9} {'replaced':>8} {'GPU peak MB':>11} {'RSS MB':>7}")
    for row in rows:
        name = f"{row['cache_mb']:g} MB" if row["cache_mb"] else "none"
        print(f"{name:>9} {row['leaves_per_s']:>9,.0f} {100 * row['forwarded']:>8.1f}% "
              f"{row['forwarded_per_s']:>8,.0f} {row['seconds_per_step']:>7.2f} {row['decisions_per_s']:>11.1f} "
              f"{row['rounds_per_step']:>11.1f} {row['table_entries']:>9,} {100 * row['replaced']:>7.1f}% "
              f"{row['gpu_peak_allocated_mb']:>11,.0f} {row['host_rss_mb']:>7,.0f}")
    first = rows[0]
    for row in rows[1:]:
        print(f"{row['cache_mb']:g} MB against {first['cache_mb']:g}: seconds a step "
              f"x{row['seconds_per_step'] / first['seconds_per_step']:.3f}, decisions a second "
              f"x{row['decisions_per_s'] / first['decisions_per_s']:.3f}")
    same = all(np.array_equal(played, moves[0]) for played in moves[1:])
    mode = args.mode or (cfg.get("runtime") or {}).get("determinism", "fast")
    print(f"same games: {'yes' if same else 'no'} (mode {mode}"
          f"{'' if mode == 'strict' else ': the sizes may part ways, see the module docstring'})")
    if args.out:
        Path(args.out).write_text(json.dumps({"args": vars(args), "rows": rows, "same_games": same}, indent=2))
    return 0 if same or mode != "strict" else 1


if __name__ == "__main__":
    sys.exit(main())
