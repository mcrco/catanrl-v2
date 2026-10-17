"""The trade oracle (tools/trade_oracle.py): its trade enumeration, the engine
route that applies a trade, and the gain bookkeeping against a stub critic."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cppanatron as cpp
import numpy as np
import pytest

from catanrl import checkpoint as ck
from catanrl.envs.encoding import dims_for, make_encoder


def _load_oracle():
    path = Path(__file__).resolve().parents[1] / "tools" / "trade_oracle.py"
    spec = importlib.util.spec_from_file_location("trade_oracle", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve the module's annotations by name
    spec.loader.exec_module(module)
    return module


oracle = _load_oracle()
MOVER_HAND = [2, 1, 0, 0, 0]  # wood, brick, sheep, wheat, ore
PARTNER_HAND = [0, 0, 1, 2, 0]


@pytest.fixture
def position():
    """A packaged 4-player position at the mover's PLAY_TURN after the roll, with the
    mover and its next seat given known hands."""
    pos = next(p for p in ck.packaged_positions() if p["game"]["num_players"] == 4
               and p["options"]["prompt"] == "PLAY_TURN" and p["options"]["has_rolled"])
    game = ck._position_game(pos)
    mover = game.current_player_index
    partner = (mover + 1) % 4
    game.set_resources(mover, MOVER_HAND)
    game.set_resources(partner, PARTNER_HAND)
    return pos, game, mover, partner


def test_enumerate_trades_lists_every_payable_shape():
    trades = oracle.enumerate_trades(MOVER_HAND, PARTNER_HAND)
    # 1:1 wood|brick > sheep|wheat; 2:1 {wood,wood}|{wood,brick} > sheep|wheat; 1:2 wood|brick > {wheat,wheat}|{sheep,wheat}
    assert sorted(t.shape for t in trades) == ["1:1"] * 4 + ["1:2"] * 4 + ["2:1"] * 4
    assert {t.label for t in trades if t.shape == "1:1"} == {"wood>sheep", "wood>wheat", "brick>sheep", "brick>wheat"}
    assert "2wood>sheep" in {t.label for t in trades} and "wood+brick>wheat" in {t.label for t in trades}
    for t in trades:
        assert all(h >= g for h, g in zip(MOVER_HAND, t.give)) and all(h >= g for h, g in zip(PARTNER_HAND, t.get))
        assert not any(a and b for a, b in zip(t.give, t.get))
    assert len(oracle.enumerate_trades([2] * 5, [2] * 5)) == 120  # 20 + 50 + 50
    assert oracle.enumerate_trades([0] * 5, [3] * 5) == [] and oracle.enumerate_trades([3] * 5, [0] * 5) == []


def test_apply_trade_changes_only_the_two_hands(position):
    _, game, mover, partner = position
    before = {seat: list(p.resources) for seat, p in enumerate(game.players)}
    for trade in (oracle.Trade((1, 0, 0, 0, 0), (0, 0, 1, 0, 0)), oracle.Trade((1, 1, 0, 0, 0), (0, 0, 0, 1, 0))):
        post = oracle.apply_trade(game, mover, partner, trade)
        assert list(post.players[mover].resources) == [h - g + r for h, g, r in zip(MOVER_HAND, trade.give, trade.get)]
        assert list(post.players[partner].resources) == [h - r + g for h, g, r in zip(PARTNER_HAND, trade.give, trade.get)]
        for seat, player in enumerate(post.players):
            for attr in [a for a in dir(player) if not a.startswith("_") and a != "resources"]:
                assert getattr(player, attr) == getattr(game.players[seat], attr), (seat, attr)
            if seat not in (mover, partner):
                assert list(player.resources) == before[seat]
        assert list(post.resource_bank) == list(game.resource_bank)
        assert (post.current_prompt, post.current_player_index, post.completed_turns) == (
            cpp.ActionPrompt.PLAY_TURN, mover, game.completed_turns)
        # The engine route and two hand overwrites are the same game, RNG and all.
        assert post.search_equivalent(oracle.trade_by_hands(game, mover, partner, trade))
    assert {seat: list(p.resources) for seat, p in enumerate(game.players)} == before  # the original is untouched
    with pytest.raises(ValueError):
        oracle.apply_trade(game, mover, partner, oracle.Trade((0, 0, 0, 0, 1), (0, 0, 1, 0, 0)))  # the mover has no ore
    with pytest.raises(ValueError):
        oracle.apply_trade(game, mover, partner, oracle.Trade((1, 0, 0, 0, 0), (0, 1, 0, 0, 0)))  # the partner has no brick


def test_gains_are_read_from_the_right_rows_and_slots(position):
    pos, game, mover, partner = position
    n = game.num_players
    encoder = make_encoder(dims_for([pos["game"]]), "private")
    far = (mover + 2) % n  # a second partner, two seats after the mover; the last seat has nothing to trade
    game.set_resources(far, [1, 0, 0, 0, 1])
    game.set_resources((mover + 3) % n, [0] * 5)
    chosen = oracle.Trade((1, 0, 0, 0, 0), (0, 0, 1, 0, 0))  # wood > sheep with the next seat
    post = oracle.apply_trade(game, mover, partner, chosen)
    pre_values = np.array([0.40, 0.20, 0.25, 0.15])  # every row the table does not name: the gains read 0
    table = {
        encoder.encode(post, mover).tobytes(): np.array([0.43, 0.24, 0.20, 0.13]),
        encoder.encode(game, partner).tobytes(): np.array([0.40, 0.10, 0.25, 0.25]),
        encoder.encode(post, partner).tobytes(): np.array([0.42, 0.10, 0.25, 0.23]),
    }

    def critic(rows: np.ndarray) -> np.ndarray:
        return np.stack([table.get(row.tobytes(), pre_values) for row in rows])

    p = oracle.Position(game=game, mover=mover, completed_turns=game.completed_turns, turn=game.num_turns,
                        vps=[0] * n, hand_sizes=[int(sum(x.resources)) for x in game.players], game_id="0.0")
    trades, meta = oracle.evaluate_positions([p], encoder, critic, flush_rows=1)
    expected = len(oracle.enumerate_trades(MOVER_HAND, PARTNER_HAND)) + len(oracle.enumerate_trades(MOVER_HAND, [1, 0, 0, 0, 1]))
    assert len(trades["pos"]) == expected == meta[0]["n_trades"]
    assert set(trades["partner_slot"]) == {1, 2}
    hit = np.flatnonzero((trades["label"] == chosen.label) & (trades["partner_slot"] == 1))
    assert len(hit) == 1
    i = hit[0]
    assert trades["mover_gain"][i] == pytest.approx(0.03)
    assert trades["partner_gain_mover_row"][i] == pytest.approx(0.04)  # slot 1 of the mover's row
    assert trades["partner_gain_own_row"][i] == pytest.approx(0.02)  # slot 0 of the partner's row
    assert trades["others_gain"][i] == pytest.approx(-0.07)
    rest = np.ones(len(trades["pos"]), dtype=bool)
    rest[i] = False
    for name in ("mover_gain", "partner_gain_mover_row", "partner_gain_own_row", "others_gain"):
        assert np.all(trades[name][rest] == 0.0)
    assert meta[0]["mover_value"] == pytest.approx(0.40) and meta[0]["value_sum"] == pytest.approx(1.0)


def test_summarize_counts_mutual_and_selfish_trades():
    trades = {
        "pos": np.array([0, 0, 1]), "partner_slot": np.array([1, 2, 1]), "shape": np.array([0, 1, 0]),
        "label": np.array(["wood>ore", "2wood>ore", "wood>ore"], dtype=object),
        "mover_gain": np.array([0.02, 0.03, 0.006]), "partner_gain_mover_row": np.array([0.02, -0.03, 0.006]),
        "partner_gain_own_row": np.array([0.0, -0.03, 0.006]), "others_gain": np.array([-0.04, 0.0, -0.012]),
    }
    meta = [{"mover": 0, "completed_turns": t, "turn": t, "vps": [0] * 4, "hand_sizes": [2] * 4, "game_id": "0.0",
             "n_trades": k, "mover_value": 0.25, "value_sum": 1.0} for t, k in ((5, 2), (20, 1), (40, 0))]
    s = oracle.summarize(trades, meta, num_players=4)
    assert s["positions"] == 3 and s["trades"] == 3 and s["positions_with_a_trade"] == 2
    m = s["measures"]["mover_row"]
    assert m["positions_with_mutual_trade"] == {"0.5": pytest.approx(2 / 3, abs=1e-4), "1": pytest.approx(1 / 3, abs=1e-4),
                                                "2": pytest.approx(1 / 3, abs=1e-4), "5": 0.0}
    assert m["best_trade"]["mover_gain_pp"]["p50"] == pytest.approx((2.0 + 0.6) / 2)
    assert m["selfish"]["trade_rate_good_for_mover_bad_for_partner"]["1"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["selfish"]["best_for_mover"]["partner_hurt"]["1"] == pytest.approx(0.5)  # 2wood>ore wins position 0
    assert m["shapes_of_mutual_trades"] == {"1:1": 1.0, "2:1": 0.0, "1:2": 0.0}
    own = s["measures"]["own_row"]
    assert own["positions_with_mutual_trade"]["1"] == 0.0 and own["positions_with_mutual_trade"]["0.5"] == pytest.approx(1 / 3, abs=1e-4)
    assert [p["positions"] for p in m["by_phase"]] == [1, 1, 1]
