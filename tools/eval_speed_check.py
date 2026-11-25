"""Throughput of an MCTS eval row on the two speed levers of its search agent, alone and together.

Plays the first ``--games`` games of a suite with the row's lineup through the native harness,
once per arm, each in a process of its own so the peaks are that arm's:

* ``python``: the path before the levers. ``kernels=fused``, repeated leaf rows merged by the
  Python ``DedupEvaluator`` (``CATANRL_EVAL_CACHE_MB=0``);
* ``cache``: ``kernels=fused``, repeated rows skipped by the engine's evaluation cache
  (``CATANRL_EVAL_CACHE_MB=--cache-mb``, 256);
* ``fused16``: the half-precision kernels (``kernels=fused16`` in the spec, another agent with a
  key of its own), the Python dedup;
* ``fused16+cache``: both.

Per arm it prints games a minute, decisions a second, leaf rows a second (searched by the
engine, and forwarded to the network), the share forwarded, where the wall time went (the
search's rounds: waiting for the device, hashing in Python, the engine's collect and apply,
which hashes for the engine's cache; the rest of the search agent's ``act``; the opponent's
``act``; the harness and the environment), peak GPU memory, peak host RSS, the search's score,
a digest over every game's row and one over the searches (the rows' digest with the search
agent's summed search statistics and counters: two networks a rounding apart can play the same
moves over a few games from searches that differ). A progress line with a rate and an ETA is
printed every ``--every`` seconds. Nothing is recorded and nothing is written under the runs
directory.

The games are the same in every arm. In strict mode (the suite's determinism) the cache must
not change a row or a search: ``python`` and ``cache`` must share both digests, and ``fused16``
and ``fused16+cache`` theirs (the half-precision network is within rounding of the float32 one,
not its bits, so its searches are its own). The check says which arms played identical games,
which searched identically, and fails (exit 1) if a pair that must agree does not.
The defaults are the standard row (the eval of
``configs/experiments/az-2p-graph-m-uniform.yaml``) on 64 games in flight, to be run on a GPU
nothing else is using::

    .venv/bin/python tools/eval_speed_check.py --json eval_speed.json

Memory: a search tree kept between a seat's decisions (``reuse``) is about 11 MB of host RAM
at 1,024 simulations, one per game in flight, and it grows with the simulations (90-100 MB at
4,096); the engine's cache is at most ``--cache-mb``. The check estimates both before it
starts and refuses an arm that would leave less than ``--min-free-gb`` available; it stops at
once (checked twice a second) if the machine falls under it while playing.

``--device cpu`` is for a smoke of the tool itself (small ``--games``, a spec with few
simulations and no ``kernels``, ``--arms python cache``): the fused kernels are CUDA's.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path

SPEC = ("MCTS:ckpt=az-2p-graph-m-uniform/main@e900#73249d10e959:sims=1024:dets=1:pending=64:reuse=true"
        ":transpositions=true:kernels=fused")
OPPONENT = "BELIEF:ckpt=ppo-2p-fr3/full@e4350#43eca7ffdd41:worlds=16"
#: arm -> (the spec's kernels, None: the spec's own; whether the engine's cache is on).
ARMS = {"python": (None, False), "cache": (None, True), "fused16": ("fused16", False), "fused16+cache": ("fused16", True)}
#: Arms that must play identical games in strict mode: the cache changes no row.
PAIRS = (("python", "cache"), ("fused16", "fused16+cache"))
CACHE_ENV = "CATANRL_EVAL_CACHE_MB"
TREE_MB = 11.0  # a kept tree at 1,024 simulations (measured on the standard row's evals)
PROCESS_GB = 2.5  # the process without trees (torch, CUDA's host side, the networks): an allowance
#: The search seat's decisions in a game of the standard row, for the ETA before games finish.
DECISIONS_PER_GAME = 95.0
LOW_MEMORY = 3  # the exit code of an arm stopped under --min-free-gb


def available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    raise RuntimeError("no MemAvailable in /proc/meminfo")


def rss_gb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 2**20
    return 0.0


def tree_gb(spec: str, in_flight: int) -> float:
    """The kept trees of ``in_flight`` games (one search seat each), by the simulations."""
    sims = next((int(part[5:]) for part in spec.split(":") if part.startswith("sims=")), 1024)
    return in_flight * TREE_MB * max(1.0, (sims / 1024) ** 1.6) / 1024  # 11 MB at 1,024, ~100 MB at 4,096


def arm_spec(spec: str, arm: str) -> str:
    """``spec`` with the arm's kernels in place of its own (a spec without any gains the part)."""
    kernels = ARMS[arm][0]
    if kernels is None:
        return spec
    parts = [part for part in spec.split(":") if not part.startswith("kernels=")]
    return ":".join([*parts, f"kernels={kernels}"])


def arm_cache_mb(args, arm: str) -> float:
    return float(args.cache_mb) if ARMS[arm][1] else 0.0


class Clock:
    """Seconds and counts by name; ``slot`` is the lineup slot acting now."""

    def __init__(self) -> None:
        self.seconds: defaultdict[str, float] = defaultdict(float)
        self.counts: defaultdict[str, int] = defaultdict(int)
        self.slot = "harness"

    def timed(self, name, fn, count=None):
        """``fn`` with its wall time under ``name`` (a callable: under ``name(self)``) and
        ``count(result)`` added to the name's count (1 a call without it)."""
        def wrapped(*args, **kwargs):
            key = name(self) if callable(name) else name
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
            finally:
                self.seconds[key] += time.perf_counter() - start
            self.counts[key] += 1 if count is None else int(count(result))
            return result
        return wrapped


def measure(args, arm: str) -> dict:
    """Play the games on ``arm``; the numbers of the run."""
    megabytes = arm_cache_mb(args, arm)
    os.environ[CACHE_ENV] = repr(megabytes)  # read when the agent is built
    import torch

    from catanrl import canon
    from catanrl.agents import native
    from catanrl.agents.spec import resolve_agent
    from catanrl.eval.harness import play_schedule
    from catanrl.eval.suites import Suite, load_suite
    from catanrl.search import BatchEvaluator, DedupEvaluator

    base = load_suite(args.suite)
    players = max(game.num_players for game in base.games())
    suite = Suite(**{**base.__dict__, "boards": min(base.boards, -(-args.games // players))})
    spec = arm_spec(args.spec, arm)
    resolved = [resolve_agent(spec), resolve_agent(args.opponent)]
    games = suite.schedule(len(resolved))[: args.games]
    in_flight = min(args.envs, len(games))
    cuda = args.device.startswith("cuda")
    # What is still to come: the trees, the cache, and the process's allowance less what it holds already.
    need = tree_gb(spec, in_flight) + megabytes / 1024 + max(0.0, (PROCESS_GB if cuda else 1.0) - rss_gb())
    free = available_gb()
    print(f"{arm}: {len(games)} games of {suite.name} ({suite.determinism}), {in_flight} in flight, {args.device}; "
          f"about {need:.1f} GB more of host memory ({tree_gb(spec, in_flight):.1f} GB of kept trees, "
          f"{megabytes / 1024:.2f} GB of cache), {free:.1f} GB available", flush=True)
    if free - need < args.min_free_gb:
        raise SystemExit(f"{arm}: that would leave {free - need:.1f} GB available, under --min-free-gb "
                         f"{args.min_free_gb}: lower --envs or free memory")

    # The clocks go on the classes before the agents are built.
    clock = Clock()
    BatchEvaluator.__call__ = clock.timed(lambda c: f"forward/{c.slot}", BatchEvaluator.__call__)
    DedupEvaluator.__call__ = _hashing(clock, DedupEvaluator.__call__)
    native.run_searches = clock.timed("rounds", native.run_searches, count=lambda rows: rows)

    agents = [agent.native(device=args.device, infer_rows=args.infer_rows, mode=suite.determinism)
              for agent in resolved]
    search, opponent = agents
    if not hasattr(search, "search_stats"):
        raise SystemExit("--spec must be an MCTS spec")
    if (search.cache is not None) != ARMS[arm][1]:
        raise SystemExit(f"{arm}: the agent's cache is {'on' if search.cache is not None else 'off'} "
                         f"(dedup=false in the spec, or --cache-mb 0?)")
    python = isinstance(search.evaluator, DedupEvaluator)
    inner = search.evaluator.evaluator if python else search.evaluator
    search.act = clock.timed("act/search", _in_slot(clock, "search", search.act))
    opponent.act = clock.timed("act/opponent", _in_slot(clock, "opponent", opponent.act))

    def leaves() -> int:
        """Leaf rows the engine asked for: the cache's count, or what the rounds handed the evaluator."""
        return int(search.cache_stats["leaves"]) if search.cache is not None else clock.counts["rounds"]

    rows: list[dict] = []
    stop = threading.Event()
    start = time.time()

    def line() -> str:
        elapsed = time.time() - start
        decisions = int(search.counters["decisions"])
        mine = [row["decisions"][row["seats"].index(0)] for row in rows]
        per_game = sum(mine) / len(mine) if len(mine) >= 8 else args.decisions_per_game
        left = max(0.0, len(games) * per_game - decisions)
        eta = f"{left * elapsed / decisions / 60:.1f} min" if decisions else "unknown"
        return (f"{arm}: {len(rows)}/{len(games)} games, {decisions} decisions "
                f"({decisions / max(elapsed, 1e-9):.1f}/s), {leaves() / max(elapsed, 1e-9):,.0f} leaf rows/s "
                f"({inner.rows / max(elapsed, 1e-9):,.0f} forwarded), RSS {rss_gb():.1f} GB, "
                f"{available_gb():.1f} GB available, elapsed {elapsed / 60:.1f} min, ETA {eta}")

    def tick() -> None:
        last = time.time()
        while not stop.wait(0.5):
            if available_gb() < args.min_free_gb:
                # At once, not at the next finished game (minutes away at the start): nothing is
                # recorded, so nothing is lost, and the memory is back in the time of an exit.
                print(f"{arm}: under {args.min_free_gb} GB available, stopped ({line()})", flush=True)
                os._exit(LOW_MEMORY)
            if time.time() - last >= args.every:
                last = time.time()
                print(line(), flush=True)

    if cuda:
        torch.cuda.reset_peak_memory_stats()
    ticker = threading.Thread(target=tick, daemon=True)
    ticker.start()
    try:
        play_schedule(agents, games, suite=suite, device=args.device, num_envs=args.envs, threads=args.threads,
                      infer_rows=args.infer_rows, on_game=rows.append)
    finally:
        stop.set()
        ticker.join()
    wall = time.time() - start
    print(line(), flush=True)

    seconds = clock.seconds
    waiting = seconds["forward/search"]
    hashing = seconds["hashing/search"]
    cache = search.cache_stats or None
    counters = {name: int(value) for name, value in sorted(search.counters.items())}
    stats = {name: int(value) for name, value in sorted(search.search_stats.items())}
    rows_digest = canon.digest(sorted(rows, key=lambda row: row["game_id"]))
    return {
        "arm": arm, "spec": spec, "canonical": resolved[0].canonical, "opponent": args.opponent,
        "cache_mb": megabytes, "kernels": inner.kernels, "suite": suite.name, "mode": suite.determinism,
        "device": args.device, "games": len(rows), "in_flight": in_flight, "wall_s": wall,
        "games_per_min": len(rows) * 60 / wall,
        "decisions": int(search.counters["decisions"]), "decisions_per_s": search.counters["decisions"] / wall,
        "leaf_rows": leaves(), "leaf_rows_per_s": leaves() / wall,
        "forwarded_rows": int(inner.rows), "forwarded_rows_per_s": inner.rows / wall, "forwards": int(inner.forwards),
        "forwarded_share": inner.rows / max(leaves(), 1),
        "counters": counters,
        "search_stats": stats,
        "dedup": dict(search.evaluator.counters) if python else None,
        "cache": cache,
        # fused16: chunks with a half past its range, and the rows that took float32 outputs.
        "float32_chunks": int(inner.exact_forwards), "float32_rows": int(getattr(inner, "exact_rows", 0)),
        "seconds": {
            "search_act": seconds["act/search"],
            "search_rounds": seconds["rounds"],
            "search_rounds_device_wait": waiting,
            "search_rounds_hashing": hashing,
            "search_rounds_engine": seconds["rounds"] - waiting - hashing,
            "search_act_rest": seconds["act/search"] - seconds["rounds"],
            "opponent_act": seconds["act/opponent"],
            "opponent_forwards": seconds["forward/opponent"],
            "harness_and_env": wall - seconds["act/search"] - seconds["act/opponent"],
        },
        "gpu_peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30 if cuda else None,
        "gpu_peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30 if cuda else None,
        "rss_peak_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "rows_digest": rows_digest,
        "searches_digest": canon.digest({"rows": rows_digest, "search_stats": stats, "counters": counters}),
        "score": sum(row["scores"][0] for row in rows) / max(len(rows), 1),
    }


def _hashing(clock: Clock, call):
    """``DedupEvaluator.__call__`` with its time outside the wrapped evaluator's forwards (the
    hashing, the merge and the scatter) under ``hashing/<slot>``."""
    def wrapped(*args, **kwargs):
        slot = clock.slot
        start, inside = time.perf_counter(), clock.seconds[f"forward/{slot}"]
        try:
            return call(*args, **kwargs)
        finally:
            forwards = clock.seconds[f"forward/{slot}"] - inside
            clock.seconds[f"hashing/{slot}"] += time.perf_counter() - start - forwards
    return wrapped


def _in_slot(clock: Clock, slot: str, act):
    def acting(*args, **kwargs):
        clock.slot = slot
        try:
            return act(*args, **kwargs)
        finally:
            clock.slot = "harness"
    return acting


def identical(results: list[dict], digest: str = "rows_digest") -> list[list[str]]:
    """The arms grouped by the games they played (their rows' digest; ``searches_digest``: by
    their searches too), in the order played."""
    groups: dict[str, list[str]] = {}
    for result in results:
        groups.setdefault(result[digest], []).append(result["arm"])
    return list(groups.values())


def disagreeing(results: list[dict]) -> list[tuple[str, str]]:
    """Strict pairs of :data:`PAIRS` that were both played and did not play the same rows from
    the same searches."""
    by_arm = {result["arm"]: result for result in results}
    return [(a, b) for a, b in PAIRS if a in by_arm and b in by_arm
            and by_arm[a]["mode"] == by_arm[b]["mode"] == "strict"
            and by_arm[a]["searches_digest"] != by_arm[b]["searches_digest"]]


def report(results: list[dict]) -> str:
    """The arms side by side."""
    def column(result: dict) -> list[str]:
        s, wall = result["seconds"], result["wall_s"]

        def share(name: str) -> str:
            return f"{s[name]:8.1f} s {100 * s[name] / wall:5.1f}%"

        gpu, cache = result["gpu_peak_allocated_gb"], result["cache"]
        return [
            result["arm"],
            result["kernels"],
            f"engine, {result['cache_mb']:g} MB" if cache else "Python" if result["dedup"] else "none",
            f"{result['games']} in {wall / 60:.1f} min",
            f"{result['games_per_min']:.2f}",
            f"{result['decisions_per_s']:.1f}",
            f"{result['leaf_rows_per_s']:,.0f}",
            f"{result['forwarded_rows_per_s']:,.0f}",
            f"{result['forwarded_share']:.3f}",
            f"{cache['entries']:,} of {cache['capacity']:,} ({cache['replaced']:,} replaced)" if cache else "n/a",
            f"{result['float32_rows']:,} rows in {result['float32_chunks']:,} chunks"
            if result["kernels"] == "fused16" else "n/a",
            share("search_act"), share("search_rounds"), share("search_rounds_device_wait"),
            share("search_rounds_hashing"), share("search_rounds_engine"),
            share("search_act_rest"), share("opponent_act"), share("opponent_forwards"), share("harness_and_env"),
            "n/a" if gpu is None else f"{gpu:.2f} GB ({result['gpu_peak_reserved_gb']:.2f} reserved)",
            f"{result['rss_peak_gb']:.2f} GB",
            f"{result['score']:.3f}",
            result["rows_digest"],
            result["searches_digest"],
        ]

    names = ["", "kernels", "repeated rows", "games", "games/min", "decisions/s", "leaf rows/s (searched)",
             "leaf rows/s (forwarded)", "share forwarded", "cache entries", "float32 again (half overflow)",
             "search act", "  rounds", "    waiting for the device", "    hashing (Python dedup)",
             "    engine (collect, apply, its cache)", "  rest (add, re-root, select)", "opponent act",
             "  its forwards", "harness and environment", "peak GPU memory", "peak host RSS", "search's score",
             "rows digest", "searches digest"]
    columns = [names, *(column(result) for result in results)]
    widths = [max(len(cell) for cell in col) for col in columns]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in zip(*columns)]
    lines.append("")
    for group in identical(results):
        lines.append(f"played identical games: {', '.join(group)}" if len(group) > 1 else
                     f"played games of its own: {group[0]}")
    for group in identical(results, "searches_digest"):
        lines.append(f"searched identically (the same games from the same search statistics): {', '.join(group)}"
                     if len(group) > 1 else f"searched on its own: {group[0]}")
    for a, b in disagreeing(results):
        lines.append(f"{a.upper()} AND {b.upper()} DISAGREE: strict rows and searches must not depend on the cache")
    if any(result["mode"] != "strict" for result in results):
        lines.append("(fast mode: rows may differ between arms, forwards of other sizes)")
    first = results[0]
    for other in results[1:]:
        lines.append(f"{other['arm']} against {first['arm']}: "
                     f"{other['games_per_min'] / first['games_per_min']:.3f}x the games a minute, "
                     f"{other['leaf_rows_per_s'] / max(first['leaf_rows_per_s'], 1e-9):.3f}x the leaf rows a second, "
                     f"{other['forwarded_rows'] / max(first['forwarded_rows'], 1):.3f}x the rows forwarded")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--spec", default=SPEC, help="the MCTS spec (the fused16 arms replace its kernels)")
    parser.add_argument("--opponent", default=OPPONENT)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS), metavar="ARM",
                        help=f"the arms, in the order played (default: {' '.join(ARMS)})")
    parser.add_argument("--suite", default="heldout-2p-fr", help="its determinism is the evaluators' mode "
                        "(dev-2p-fr plays fast)")
    parser.add_argument("--games", type=int, default=64, help="the suite's first games")
    parser.add_argument("--envs", type=int, default=64, help="games in flight (the standard row: 256)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=12, help="the environment's threads (the row's runtime.threads)")
    parser.add_argument("--infer-rows", type=int, default=512)
    parser.add_argument("--cache-mb", type=float, default=256.0,
                        help="the engine cache's megabytes in the cache arms (CATANRL_EVAL_CACHE_MB there)")
    parser.add_argument("--min-free-gb", type=float, default=4.0,
                        help="MemAvailable to keep: refuse to start under it, stop under it")
    parser.add_argument("--decisions-per-game", type=float, default=DECISIONS_PER_GAME,
                        help="the search seat's decisions a game, for the ETA until 8 games finished")
    parser.add_argument("--every", type=float, default=30.0, help="seconds between progress lines")
    parser.add_argument("--json", type=Path, default=None, help="also write the results here")
    parser.add_argument("--one", default=None, choices=list(ARMS), help=argparse.SUPPRESS)  # an arm, in this process
    parser.add_argument("--out", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.games < 1 or args.envs < 1:
        parser.error("--games and --envs are positive")
    if not args.cache_mb > 0:
        parser.error("--cache-mb is positive (the arms without the cache run at 0)")
    if len(set(args.arms)) != len(args.arms):
        parser.error("--arms names an arm once")
    if args.one is not None:
        args.out.write_text(json.dumps(measure(args, args.one)))
        return
    free = available_gb()
    if free < args.min_free_gb:
        raise SystemExit(f"{free:.1f} GB of host memory available, under --min-free-gb {args.min_free_gb}: not started")
    passed = sys.argv[1:] if argv is None else list(argv)
    results = []
    began = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        for index, arm in enumerate(args.arms):
            print(f"arm {index + 1}/{len(args.arms)}: {arm} (elapsed {(time.time() - began) / 60:.1f} min)", flush=True)
            out = Path(tmp) / f"arm{index}.json"
            # A process per arm: its peak RSS and GPU memory are its own, its trees and its cache are freed.
            done = subprocess.run([sys.executable, str(Path(__file__).resolve()), *passed, "--one", arm,
                                   "--out", str(out)], check=False,
                                  env={**os.environ, "WANDB_MODE": "disabled", CACHE_ENV: repr(arm_cache_mb(args, arm))})
            if done.returncode == LOW_MEMORY:
                raise SystemExit(f"{arm} stopped for memory (--min-free-gb {args.min_free_gb})")
            if done.returncode:
                raise SystemExit(f"{arm} failed (exit {done.returncode})")
            results.append(json.loads(out.read_text()))
    print()
    print(report(results))
    if args.json is not None:
        args.json.write_text(json.dumps(results, indent=1) + "\n")
    if disagreeing(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
