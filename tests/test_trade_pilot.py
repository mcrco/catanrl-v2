"""tools/trade_pilot.py: the in-process readings of a TRADE lineup (bundles, claimed gains,
critic rows per read, the cancel rate) agree with the harness's rows and the agents' own
counters, and holding the agents changes no row."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from eval_helpers import agent, make_suite, schedule, tiny_model

from catanrl.agents.trade import SHAPE_NAMES
from catanrl.eval.harness import play_schedule


def _load_pilot():
    path = Path(__file__).resolve().parents[1] / "tools" / "trade_pilot.py"
    spec = importlib.util.spec_from_file_location("trade_pilot", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve the module's annotations by name
    spec.loader.exec_module(module)
    return module


pilot = _load_pilot()
SUITE = make_suite(({"game": "base-3p"}, {"game": "base-4p"}), turn_limit=60)
MODEL = tiny_model(["base-3p", "base-4p"], seed=1)
CRITIC = tiny_model(["base-3p", "base-4p"], seed=2)


def lineup():
    """The trade-harness tests' table: a proposer, a responder-only TRADE and R (F fills)."""
    trade = lambda **p: agent("TRADE", MODEL, critic=CRITIC, infer_rows=8, dets=3, **p)
    return [trade(), trade(propose=False), agent("R")]


def test_readings_agree_with_the_rows_and_the_agents(capsys):
    games = schedule(SUITE, 3, boards=1)
    agents = lineup()
    readings = pilot.read_lineup(agents, games, suite=SUITE, fill=agent("F"), num_envs=4, threads=2)
    plain = []
    play_schedule(lineup(), games, suite=SUITE, fill=agent("F"), on_game=plain.append, num_envs=1)
    by_id = lambda rows: sorted(rows, key=lambda r: r["game_id"])
    assert by_id(readings.rows) == by_id(plain)  # watching the agents changes nothing they play
    summary = pilot.summarize(readings)
    proposer, responder = agents[0], agents[1]
    assert summary["games"] == len(games) and summary["critic_rows"] == sum(s["critic_rows"] for s in readings.stats.values())
    # The record's totals (trade_totals, per slot) and the decision log count the same steps.
    assert summary["slots"][0]["offers"] == summary["decisions"]["offer"] == proposer.stats["offers"] > 0
    assert summary["slots"][0]["confirms"] == summary["decisions"]["confirm"] == proposer.stats["confirms"] > 0
    assert summary["slots"][0]["cancels"] == summary["decisions"]["cancel"] == proposer.stats["cancels"]
    assert summary["slots"][1]["accepts"] == summary["decisions"]["accept"] == responder.stats["accepts"] > 0
    assert summary["cancel_rate"] == summary["slots"][0]["cancel_rate"]
    assert summary["proposal_reads"]["count"] == proposer.stats["evaluations"] > 0
    assert summary["proposal_reads"]["critic_rows_per_read"] > 1  # the pre row and a post row per candidate
    assert summary["prompt_reads"]["count"] == proposer.stats["confirms"] + proposer.stats["cancels"] + sum(
        1 for d in readings.decisions if d["slot"] == 1 and d["kind"] in ("accept", "reject") and d["read"])
    assert summary["prompt_reads"]["critic_rows_per_read"] >= 2  # pre and post per world, or pre plus an acceptee
    offered = summary["offered_bundles"]
    assert sum(b["count"] for b in offered) == proposer.stats["offers"]
    assert all(b["shape"] in SHAPE_NAMES and b["own_gain_pp"] > 0 and b["partner_gain_pp"] > 0 for b in offered)
    assert offered == sorted(offered, key=lambda b: (-b["count"], b["bundle"]))
    assert sum(summary["offers_by_shape"].values()) == proposer.stats["offers"]
    confirmed = {b["bundle"] for b in summary["confirmed_bundles"]}
    assert confirmed and confirmed <= {b["bundle"] for b in offered}
    assert summary["gains_pp"]["offer_own"]["n"] == proposer.stats["offers"] and summary["gains_pp"]["confirm_own"]["p50"] > 0
    assert summary["gains_pp"]["accept_own"]["n"] == responder.stats["accepts"] and summary["gains_pp"]["accept_own"]["max"] > 0
    assert summary["rejects_by_note"].get("own offer", 0) == sum(
        1 for d in readings.decisions if d["slot"] == 0 and d["kind"] == "reject")
    pilot.print_summary(summary)
    out = capsys.readouterr().out
    assert "slot 0 per game:" in out and "cancel rate" in out and "offered bundles" in out and "confirm_own" in out


def test_plan_row_reads_the_pilot_config():
    """The yaml's rows read as written (the design's `oracle=true` notation included)."""
    lineup, fill, suite = pilot.plan_row("trade-harness-pilot", 3)
    assert suite == {"name": "heldout-4p", "boards": 50} and fill is None and len(lineup) == 4
    assert all(":oracle=true" in spec and ":dets=" not in spec for spec in lineup)
    assert lineup[0].endswith(":oracle=true") and all(spec.endswith(":propose=false") for spec in lineup[1:])
