"""Read a TRADE lineup's trading in process: the pilot readings a record cannot hold.

A record's rows count each seat's offers, accepts, confirms and cancels, and
``results.trade_totals`` sums them per slot with the cancel rate (``catanrl
eval`` prints them). Which bundles were offered and confirmed, the gains the
critic claimed for them, the fronts it read and the critic rows a read costs
live only on the agents (``TradeAgent.decisions`` after each call,
``TradeAgent.stats``). This tool plays a lineup through the native harness
holding the agents in process, logs every trade decision as it is made and
prints those readings. It writes no record: the games are the suite's
schedule under the suite's determinism flags, so the rows are the ones
``catanrl eval`` records for the lineup and the log reads that record's games.

    python tools/trade_pilot.py --config trade-harness-pilot --row 0 --boards 4 --device cpu
    python tools/trade_pilot.py --lineup TRADE:ckpt=aux-hidden-4p/plain@best \\
        TRADE:ckpt=aux-hidden-4p/plain@best:propose=false --suite heldout-4p --boards 2
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from catanrl.agents.trade import TradeAgent
from catanrl.eval.harness import play_schedule
from catanrl.eval.results import trade_totals
from catanrl.eval.suites import load_suite
from catanrl.seeding import determinism


@dataclass
class Readings:
    """What one lineup did over a schedule: the harness's rows, every trade decision of
    the TRADE seats (slot, kind, bundle, gains in pp), one entry per critic-reading call
    (the rows it read, the decisions that read) and the agents' counters per slot."""

    rows: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    reads: list[dict] = field(default_factory=list)
    stats: dict[int, dict] = field(default_factory=dict)
    wall_s: float = 0.0


def _watch(agent: TradeAgent, slot: int, readings: Readings) -> None:
    """Log the agent's decisions after every ``act`` / ``respond_trade`` call: what it did,
    read off the agent (no second evaluation; the actions it plays are untouched)."""
    for name in ("act", "respond_trade"):
        inner = getattr(agent, name)

        def wrapped(*args, _inner=inner, _name=name, **kwargs):
            before = agent.stats["critic_rows"]
            out = _inner(*args, **kwargs)
            reads = 0
            for d in agent.decisions:
                # A play row reads the critic when it builds a front (an offer from the turn's
                # cached front does not); a trade prompt when it judged the offer (a REJECT
                # with a note did not).
                read = d.front is not None if _name == "act" else d.kind in ("confirm", "cancel") or d.gain is not None
                reads += read
                if d.kind == "policy" and not read:
                    continue
                readings.decisions.append({
                    "slot": slot, "kind": d.kind, "seat": d.seat, "read": read,
                    "bundle": None if d.bundle is None else d.bundle.label,
                    "shape": None if d.bundle is None else d.bundle.shape, "partner": d.partner,
                    "gain_pp": None if d.gain is None else 100.0 * d.gain,
                    "partner_gain_pp": None if d.partner_gain is None else 100.0 * d.partner_gain,
                    "front": None if d.front is None else len(d.front), "note": d.note,
                })
            readings.reads.append({"call": _name, "rows": agent.stats["critic_rows"] - before, "reads": reads})
            return out

        setattr(agent, name, wrapped)


def read_lineup(agents: Sequence[Any], games: Sequence[Any], *, suite: Any, fill: Any = None,
                num_envs: int = 256, threads: int = 1) -> Readings:
    """Play ``games`` (the suite's schedule) with native ``agents`` (slot order, ``fill`` for
    the seats past them) exactly as the harness does, every TRADE seat watched."""
    readings = Readings()
    for slot, agent in enumerate([*agents, *([fill] if fill is not None else [])]):
        if isinstance(agent, TradeAgent):
            _watch(agent, slot, readings)
            readings.stats[slot] = agent.stats
    started = time.time()
    with determinism(suite.determinism):  # the record's flags: these rows are the record's
        play_schedule(agents, games, fill=fill, suite=suite, num_envs=num_envs, threads=threads,
                      on_game=readings.rows.append)
    readings.wall_s = time.time() - started
    return readings


# ---------------------------------------------------------------- the readings


def _quantiles(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    x = np.asarray(values, dtype=np.float64)
    return {"n": int(x.size), "mean": float(x.mean()), "p50": float(np.quantile(x, 0.5)),
            "p95": float(np.quantile(x, 0.95)), "max": float(x.max())}


def _bundles(decisions: Sequence[dict]) -> list[dict]:
    """Bundles by count (then label), with the mean claimed gains of the decisions that
    chose them: the proposer's own and, for offers, the partner's it collapsed on."""
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for d in decisions:
        groups[d["bundle"]].append(d)
    out = []
    for bundle, ds in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        own = [d["gain_pp"] for d in ds if d["gain_pp"] is not None]
        partner = [d["partner_gain_pp"] for d in ds if d["partner_gain_pp"] is not None]
        out.append({"bundle": bundle, "shape": ds[0]["shape"], "count": len(ds),
                    "own_gain_pp": float(np.mean(own)) if own else None,
                    "partner_gain_pp": float(np.mean(partner)) if partner else None})
    return out


def _rows_per_read(reads: Sequence[dict]) -> float:
    count = sum(r["reads"] for r in reads)
    return round(sum(r["rows"] for r in reads) / count, 2) if count else 0.0


def summarize(readings: Readings) -> dict:
    """The pilot's readings: the record's per-slot trade totals (``trade_totals``, as the
    summary holds them), the decisions by kind with the cancel rate, the proposal and
    trade-prompt reads with their critic rows, the bundles offered and confirmed with the
    claimed gains, the gain distributions, and the agents' counters."""
    rows, log = readings.rows, readings.decisions
    by_kind = {k: [d for d in log if d["kind"] == k] for k in ("offer", "accept", "reject", "confirm", "cancel")}
    fronts = [d for d in log if d["front"] is not None]
    answered = len(by_kind["confirm"]) + len(by_kind["cancel"])
    slots = {}
    for slot in sorted(readings.stats):
        mine = [r for r in rows if slot in r["seats"]]
        traded = [r for r in mine if r.get("trades")]
        slots[slot] = trade_totals(traded, slot, len(mine)) if traded else None
    return {
        "games": len(rows), "wall_s": round(readings.wall_s, 1),
        "wall_per_game_s": round(readings.wall_s / len(rows), 2) if rows else None,
        "slots": slots,
        "decisions": {k: len(v) for k, v in by_kind.items()},
        "cancel_rate": len(by_kind["cancel"]) / answered if answered else 0.0,
        "rejects_by_note": dict(collections.Counter(d["note"] or "read" for d in by_kind["reject"])),
        "proposal_reads": {"count": len(fronts), "empty_fronts": sum(d["front"] == 0 for d in fronts),
                           "front_size_mean": round(float(np.mean([d["front"] for d in fronts])), 2) if fronts else None,
                           "critic_rows_per_read": _rows_per_read([r for r in readings.reads if r["call"] == "act"])},
        "prompt_reads": {"count": sum(r["reads"] for r in readings.reads if r["call"] == "respond_trade"),
                         "critic_rows_per_read": _rows_per_read([r for r in readings.reads
                                                                 if r["call"] == "respond_trade"])},
        "critic_rows": sum(r["rows"] for r in readings.reads),
        "offers_by_shape": dict(collections.Counter(d["shape"] for d in by_kind["offer"])),
        "offered_bundles": _bundles(by_kind["offer"]),
        "confirmed_bundles": _bundles(by_kind["confirm"]),
        "gains_pp": {
            "offer_own": _quantiles([d["gain_pp"] for d in by_kind["offer"]]),
            "offer_partner": _quantiles([d["partner_gain_pp"] for d in by_kind["offer"]]),
            "accept_own": _quantiles([d["gain_pp"] for d in by_kind["accept"]]),
            "reject_own": _quantiles([d["gain_pp"] for d in by_kind["reject"] if d["gain_pp"] is not None]),
            "confirm_own": _quantiles([d["gain_pp"] for d in by_kind["confirm"]]),
        },
        "stats": {slot: dict(stats) for slot, stats in sorted(readings.stats.items())},
    }


def print_summary(summary: dict, file=None, top: int = 12) -> None:
    file = file or sys.stdout  # the stream of the call, not of the import (a test's capture)
    n = summary["games"]
    print(f"{n} games, {summary['wall_s']:.1f} s ({summary['wall_per_game_s']} s per game), "
          f"{summary['critic_rows']} critic rows", file=file)
    for slot, totals in summary["slots"].items():
        if totals is None:
            print(f"  slot {slot}: no trade step at its seats", file=file)
            continue
        per_game = ", ".join(f"{v:.2f} {k}" for k, v in totals["per_game"].items())
        print(f"  slot {slot} per game: {per_game}; cancel rate {totals['cancel_rate']:.2f}", file=file)
    d = summary["decisions"]
    print(f"  decisions: {d['offer']} offers, {d['accept']} accepts, {d['reject']} rejects {summary['rejects_by_note']}, "
          f"{d['confirm']} confirms, {d['cancel']} cancels (cancel rate {summary['cancel_rate']:.2f})", file=file)
    p, q = summary["proposal_reads"], summary["prompt_reads"]
    print(f"  proposal reads: {p['count']} ({p['empty_fronts']} empty fronts, front size mean {p['front_size_mean']}), "
          f"{p['critic_rows_per_read']} critic rows per read; trade-prompt reads: {q['count']}, "
          f"{q['critic_rows_per_read']} rows per read", file=file)
    print(f"  offers by shape: {summary['offers_by_shape']}", file=file)
    for name in ("offered_bundles", "confirmed_bundles"):
        print(f"  {name.replace('_', ' ')} (count, mean claimed own / partner gain pp):", file=file)
        for b in summary[name][:top]:
            partner = "-" if b["partner_gain_pp"] is None else f"{b['partner_gain_pp']:+.2f}"
            print(f"    {b['bundle']:>24} x{b['count']:<4} {b['own_gain_pp']:+.2f} / {partner}", file=file)
    for name, q in summary["gains_pp"].items():
        if q:
            print(f"  {name:<14} n {q['n']:<5} mean {q['mean']:+.2f}  p50 {q['p50']:+.2f}  p95 {q['p95']:+.2f}  "
                  f"max {q['max']:+.2f} pp", file=file)
    for slot, stats in summary["stats"].items():
        print(f"  slot {slot} counters: {stats}", file=file)


# ---------------------------------------------------------------- CLI


def plan_row(config: str, row: int, variant: str | None = None) -> tuple[list[str], str | None, Any]:
    """The lineup, fill and suite of one ``eval.final`` entry of an experiment YAML, as
    written: a spec with a ``${...}`` ref is bound by the runner for its run, not here."""
    from catanrl.config import load_plan

    plan = load_plan(config)
    chosen = plan.variant(variant) if variant else plan.variants[0]
    entry = chosen.config["eval"]["final"][row]
    specs = [*entry["lineup"], *([entry["fill"]] if entry.get("fill") else [])]
    if any("${" in spec for spec in specs):
        raise ValueError(f"{config} row {row} names a run-bound ref; pass the bound specs with --lineup")
    return list(entry["lineup"]), entry.get("fill"), entry["suite"]


def main(argv: Sequence[str] | None = None) -> None:
    from catanrl.agents import resolve_agent

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="an experiment YAML (configs/experiments) whose eval.final row to read")
    parser.add_argument("--variant", default=None, help="the plan's variant (default: its first)")
    parser.add_argument("--row", type=int, default=0, help="the eval.final entry (default 0)")
    parser.add_argument("--lineup", nargs="+", help="agent specs, slot order (instead of --config)")
    parser.add_argument("--fill", default=None, help="the spec for seats past the lineup")
    parser.add_argument("--suite", default=None, help="suite name (with --lineup)")
    parser.add_argument("--boards", type=int, default=None, help="override the suite's boards")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4, help="VecEnv threads")
    parser.add_argument("--infer-rows", type=int, default=256)
    parser.add_argument("--top", type=int, default=12, help="bundles printed per list")
    parser.add_argument("--out", type=Path, default=None, help="JSON path for the summary and the decision log")
    args = parser.parse_args(argv)
    if bool(args.config) == bool(args.lineup):
        parser.error("pass --config (with --row) or --lineup (with --suite)")
    if args.config:
        lineup, fill, suite = plan_row(args.config, args.row, args.variant)
    else:
        if not args.suite:
            parser.error("--lineup needs --suite")
        lineup, fill, suite = args.lineup, args.fill, args.suite
    if args.boards is not None:
        suite = {**(suite if isinstance(suite, dict) else {"name": suite}), "boards": args.boards}
    suite = load_suite(suite)
    resolved = [resolve_agent(spec, harness="native", games=suite.games()) for spec in lineup]
    fill_resolved = resolve_agent(fill, harness="native", games=suite.games()) if fill else None
    build = lambda a: a.native(device=args.device, infer_rows=args.infer_rows, mode=suite.determinism)
    games = suite.schedule(len(lineup), fill=fill is not None)
    print(f"{suite.name}: {len(games)} games, " + ", ".join(a.label for a in resolved)
          + (f" (fill {fill_resolved.label})" if fill_resolved else ""), flush=True)
    readings = read_lineup([build(a) for a in resolved], games, suite=suite,
                           fill=build(fill_resolved) if fill_resolved else None,
                           num_envs=args.num_envs, threads=args.threads)
    summary = {"suite": suite.name, "lineup": [a.canonical for a in resolved],
               "fill": fill_resolved.canonical if fill_resolved else None, **summarize(readings)}
    print_summary(summary, top=args.top)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "decisions": readings.decisions}, indent=1,
                                       sort_keys=True) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
