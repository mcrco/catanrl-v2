"""Reuse, transposition and dedup counters of MCTS agents over a schedule, and the
shared-game pairing of two records.

Plays a suite's first ``--boards`` boards through the native harness with the
lineup's agents and prints, per MCTS slot: the agent's counters (``decisions``,
``fresh``, ``reused``, ``dropped_first``, ``refused_diverged``,
``refused_engine``; the reuse rate is ``reused / decisions``), its search
statistics per decision (nodes, evaluations, simulations, coalesced outcomes,
transpositions, max depth), the dedup evaluator's ``rows``, ``forwarded``,
``batch_hits`` and ``cache_hits`` as counts and fractions of the rows, the
slot's score rate, and the wall time. Every counter is printed, zero or not,
so the report's schema does not vary by run. Nothing is recorded. The probe
sets ``CATANRL_DEDUP_FORCE=1`` before building the agents, so the dedup
counters count on a GPU too (where the wrapper otherwise passes a strict
batch below ``infer_rows`` through unhashed).

    uv run python tools/search_probe.py \\
        'MCTS:ckpt=ppo-4p-critic-fix/lambda-critic-g1@e100:sims=64:dets=1:reuse=true' \\
        POLICY:ckpt=ppo-4p-critic-fix/lambda-critic-g1@e100 \\
        POLICY:ckpt=ppo-4p-critic-fix/lambda-critic-g1@e100 \\
        POLICY:ckpt=ppo-4p-critic-fix/lambda-critic-g1@e100 \\
        --suite heldout-4p --boards 3 --device cpu

``--pair A B`` (eval ids or record paths; ``--slot`` picks the slot) instead
prints the slot's score on the games both records played: the rates, the
delta and McNemar's p over the shared game ids, so a shorter schedule of the
same suite (a suite plays its first ``boards`` seeds: the smaller is a prefix)
compares with a longer one. ``catanrl.eval.results.paired`` refuses records
whose code or schedule differ, since a code change can move every seat's
rows; this pairing reports ``code_differs`` and ``schedule_differs`` instead
and leaves the reading to the user. The sound comparison with a stored record
is still a replay of its row at the current code, paired through
``results.paired``.

    uv run python tools/search_probe.py --pair 90a1f06c <the reuse=true row's eval id>

A training run's monitor (``eval.monitor``, catanrl.train.select) against a fixed
opponent, such as ``${champion}``, plays the same suite schedule at every
checkpoint, so two checkpoints' records of it pair game for game. Their eval ids
are in the run's metrics rows, ``select/<name>/eval_id`` at the two epochs:

    uv run python tools/search_probe.py --pair <eval id at eA> <eval id at eB>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

COUNTERS = ("decisions", "fresh", "reused", "dropped_first", "refused_diverged", "refused_engine")
DEDUP = ("rows", "forwarded", "batch_hits", "cache_hits")


def probe(agent) -> dict:
    """An MCTS agent's counters, per-decision search statistics and dedup fractions."""
    from catanrl.search import DedupEvaluator

    counters, stats = agent.counters, agent.search_stats
    decisions = max(int(counters["decisions"]), 1)
    # By name: a Counter holds only the names it incremented.
    out = {"counters": {name: int(counters[name]) for name in COUNTERS}, "reuse_rate": counters["reused"] / decisions,
           "per_decision": {name: value / decisions for name, value in sorted(stats.items())}}
    if isinstance(agent.evaluator, DedupEvaluator):
        dedup = agent.evaluator.counters
        rows = max(int(dedup["rows"]), 1)
        out["dedup"] = {**{name: int(dedup[name]) for name in DEDUP},
                        "batch_hit_fraction": dedup["batch_hits"] / rows,
                        "cache_hit_fraction": dedup["cache_hits"] / rows,
                        "forwarded_fraction": dedup["forwarded"] / rows}
    return out


def pair(a, b, *, slot: int = 0) -> dict:
    """``slot``'s score in records ``a`` and ``b`` over the games both played: the same
    suite (a prefix of its boards at most) and the same opponents in the same seats, the
    code and the schedule free to differ (the report says when they do)."""
    from catanrl.eval.stats import _paired_comparison

    for record in (a, b):
        if not 0 <= slot < len(record.key["lineup"]):
            raise ValueError(f"{record.eval_id}: no slot {slot}")
    suites = [{k: v for k, v in r.key["suite"].items() if k not in ("boards", "schedule_hash")} for r in (a, b)]
    if suites[0] != suites[1]:
        raise ValueError("paired records must play the same suite (a prefix of its boards at most)")
    others = [[d for i, d in enumerate(r.key["lineup"]) if i != slot] + [r.key["fill"]] for r in (a, b)]
    if others[0] != others[1]:
        raise ValueError("paired records must face the same opponents in the same seats")
    rows_a = {r["game_id"]: r for r in a.games()}
    rows_b = {r["game_id"]: r for r in b.games()}
    shared = sorted(set(rows_a) & set(rows_b))
    if not shared:
        raise ValueError("paired records share no game")
    result = _paired_comparison([rows_a[i].get("winner") == slot for i in shared],
                                [rows_b[i].get("winner") == slot for i in shared])
    return {**result, "a": a.eval_id, "b": b.eval_id, "slot": slot, "of": [len(rows_a), len(rows_b)],
            "code_differs": a.key["code"] != b.key["code"],
            "schedule_differs": a.key["suite"]["schedule_hash"] != b.key["suite"]["schedule_hash"]}


def record(ref: str):
    """A record by path, or by (a prefix of) its eval id under the runs directory. One eval
    stored by several runs (a select of weights another run already played in the same
    search) resolves when every copy holds the same games (``games_sha256``)."""
    from catanrl.eval.results import EvalRecord, all_records

    path = Path(ref)
    if path.suffix == ".json" and path.is_file():
        return EvalRecord.load(path)
    found = [EvalRecord.load(p) for p in all_records() if p.stem.startswith(ref)]
    if len(found) > 1 and len({r.eval_id for r in found}) == 1:
        if len({r.data.get("games_sha256") for r in found}) == 1 and found[0].data.get("games_sha256"):
            return found[0]
        raise SystemExit(f"{ref}: {len(found)} records of eval {found[0].eval_id} hold different games "
                         f"({', '.join(str(r.path) for r in found)}); give a path")
    if len(found) != 1:
        raise SystemExit(f"{ref}: {len(found)} records match (give a path or a longer id)")
    return found[0]


def play(args) -> dict:
    """The lineup's probe report over the suite's first ``boards`` boards."""
    os.environ.setdefault("CATANRL_DEDUP_FORCE", "1")  # before any agent builds its evaluator

    from catanrl.agents.native import MctsAgent
    from catanrl.canopy import resolve as resolve_agent  # every key, CANOPY too
    from catanrl.eval.harness import play_schedule
    from catanrl.eval.results import _with_canopy, summarize
    from catanrl.eval.suites import load_suite

    suite = dataclasses.replace(load_suite(args.suite), boards=args.boards)
    resolved = [resolve_agent(spec, games=suite.games()) for spec in args.lineup]
    fill = resolve_agent(args.fill, games=suite.games()) if args.fill else None
    agents = [r.native(device=args.device, infer_rows=args.infer_rows) for r in resolved]
    fill_agent = fill.native(device=args.device, infer_rows=args.infer_rows) if fill is not None else None
    games = suite.schedule(len(agents), fill=fill_agent is not None)
    rows: list[dict] = []
    start = time.perf_counter()
    slots = [*agents, *([fill_agent] if fill_agent is not None else [])]
    try:
        # A CANOPY slot's per-game stats (fallbacks, refreshes, refused moves) go into the rows, as in
        # a record, and into the report's slot entry.
        play_schedule(agents, games, fill=fill_agent, suite=suite, on_game=_with_canopy(slots, games, rows.append),
                      device=args.device, num_envs=args.num_envs, threads=args.threads, infer_rows=args.infer_rows)
    finally:
        for agent in [*agents, fill_agent]:
            if hasattr(agent, "close"):
                agent.close()
    wall = time.perf_counter() - start
    summary = summarize(rows, [r.label for r in resolved])
    report = {"suite": suite.name, "boards": args.boards, "games": len(rows), "wall_s": wall,
              "truncated": summary["truncated"], "slots": []}
    for slot, (agent, r) in enumerate(zip(agents, resolved)):
        entry = {"slot": slot, "spec": r.canonical, "score_rate": summary["slots"][slot]["score_rate"],
                 "wilson": summary["slots"][slot]["wilson"]}
        if isinstance(agent, MctsAgent):
            entry.update(probe(agent))
        if "canopy" in summary["slots"][slot]:
            entry["canopy"] = summary["slots"][slot]["canopy"]
        report["slots"].append(entry)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("lineup", nargs="*", help="agent specs, in slot order")
    parser.add_argument("--suite", default="heldout-2p")
    parser.add_argument("--boards", type=int, default=2)
    parser.add_argument("--fill", default=None, help="the fill slot's spec (seats past the lineup)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--infer-rows", type=int, default=256)
    parser.add_argument("--pair", nargs=2, metavar=("A", "B"), default=None,
                        help="two eval ids or record paths: the slot's score on the games both played")
    parser.add_argument("--slot", type=int, default=0, help="the slot --pair compares")
    parser.add_argument("--json", type=Path, default=None, help="also write the report here")
    args = parser.parse_args(argv)
    if bool(args.lineup) == bool(args.pair):
        parser.error("give a lineup to probe, or --pair A B")
    report = pair(record(args.pair[0]), record(args.pair[1]), slot=args.slot) if args.pair else play(args)
    text = json.dumps(report, indent=1)
    print(text)
    if args.json is not None:
        args.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
