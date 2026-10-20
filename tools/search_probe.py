"""Reuse, transposition and dedup counters of MCTS agents over a schedule.

Plays a suite's first ``--boards`` boards through the native harness with the
lineup's agents and prints, per MCTS slot: the agent's counters (``decisions``,
``fresh``, ``reused``, ``dropped_first``, ``refused_diverged``,
``refused_engine``; the reuse rate is ``reused / decisions``), its search
statistics per decision (nodes, evaluations, simulations, coalesced outcomes,
transpositions, max depth), the dedup evaluator's ``rows``, ``forwarded``,
``batch_hits`` and ``cache_hits`` as counts and fractions of the rows, the
slot's score rate, and the wall time. Nothing is recorded. The probe sets
``CATANRL_DEDUP_FORCE=1`` before building the agents, so the dedup counters
count on a GPU too (where the wrapper otherwise passes a strict batch below
``infer_rows`` through unhashed).

    uv run python tools/search_probe.py \\
        'MCTS:ckpt=ppo-4p-critic-fix/lambda-critic-g1@e100:sims=64:dets=1:reuse=true' \\
        POLICY:ckpt=ppo-4p-critic-fix/lambda-critic-g1@e100 \\
        POLICY:ckpt=ppo-4p-critic-fix/lambda-critic-g1@e100 \\
        POLICY:ckpt=ppo-4p-critic-fix/lambda-critic-g1@e100 \\
        --suite heldout-4p --boards 3 --device cpu
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path


def probe(agent) -> dict:
    """An MCTS agent's counters, per-decision search statistics and dedup fractions."""
    from catanrl.search import DedupEvaluator

    counters, stats = agent.counters, agent.search_stats
    decisions = max(int(counters["decisions"]), 1)
    out = {"counters": dict(counters), "reuse_rate": counters["reused"] / decisions,
           "per_decision": {name: value / decisions for name, value in sorted(stats.items())}}
    if isinstance(agent.evaluator, DedupEvaluator):
        dedup = agent.evaluator.counters
        rows = max(int(dedup["rows"]), 1)
        out["dedup"] = {**dict(dedup), "batch_hit_fraction": dedup["batch_hits"] / rows,
                        "cache_hit_fraction": dedup["cache_hits"] / rows,
                        "forwarded_fraction": dedup["forwarded"] / rows}
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("lineup", nargs="+", help="agent specs, in slot order")
    parser.add_argument("--suite", default="heldout-2p")
    parser.add_argument("--boards", type=int, default=2)
    parser.add_argument("--fill", default=None, help="the fill slot's spec (seats past the lineup)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--infer-rows", type=int, default=256)
    parser.add_argument("--json", type=Path, default=None, help="also write the report here")
    args = parser.parse_args(argv)
    os.environ.setdefault("CATANRL_DEDUP_FORCE", "1")  # before any agent builds its evaluator

    from catanrl.agents import resolve_agent
    from catanrl.agents.native import MctsAgent
    from catanrl.eval.harness import play_schedule
    from catanrl.eval.results import summarize
    from catanrl.eval.suites import load_suite

    suite = dataclasses.replace(load_suite(args.suite), boards=args.boards)
    resolved = [resolve_agent(spec, games=suite.games()) for spec in args.lineup]
    fill = resolve_agent(args.fill, games=suite.games()) if args.fill else None
    agents = [r.native(device=args.device, infer_rows=args.infer_rows) for r in resolved]
    fill_agent = fill.native(device=args.device, infer_rows=args.infer_rows) if fill is not None else None
    games = suite.schedule(len(agents), fill=fill_agent is not None)
    rows: list[dict] = []
    start = time.perf_counter()
    play_schedule(agents, games, fill=fill_agent, suite=suite, on_game=rows.append, device=args.device,
                  num_envs=args.num_envs, threads=args.threads, infer_rows=args.infer_rows)
    wall = time.perf_counter() - start
    summary = summarize(rows, [r.label for r in resolved])
    report = {"suite": suite.name, "boards": args.boards, "games": len(rows), "wall_s": wall,
              "truncated": summary["truncated"], "slots": []}
    for slot, (agent, r) in enumerate(zip(agents, resolved)):
        entry = {"slot": slot, "spec": r.canonical, "score_rate": summary["slots"][slot]["score_rate"],
                 "wilson": summary["slots"][slot]["wilson"]}
        if isinstance(agent, MctsAgent):
            entry.update(probe(agent))
        report["slots"].append(entry)
    text = json.dumps(report, indent=1)
    print(text)
    if args.json is not None:
        args.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
