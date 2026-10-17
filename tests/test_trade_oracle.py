"""The trade oracle (tools/trade_oracle.py): its trade enumeration, the engine
route that applies a trade (the partner at every place of the decide order, and
the 2-player offer), and the gain bookkeeping and noise floors against a stub
critic."""

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
WOOD_FOR_SHEEP = oracle.Trade((1, 0, 0, 0, 0), (0, 0, 1, 0, 0))
WOOD_BRICK_FOR_WHEAT = oracle.Trade((1, 1, 0, 0, 0), (0, 0, 0, 1, 0))


def _position(num_players: int, rank: int):
    """A packaged position at the mover's PLAY_TURN after the roll. The partner is
    the seat at ``rank`` in the engine's decide order (the other seats in seat
    order); the mover and the partner get known hands."""
    pos = next(p for p in ck.packaged_positions() if p["game"]["num_players"] == num_players
               and p["options"]["prompt"] == "PLAY_TURN" and p["options"]["has_rolled"])
    game = ck._position_game(pos)
    mover = game.current_player_index
    partner = [seat for seat in range(num_players) if seat != mover][rank]
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


@pytest.mark.parametrize("num_players, rank", [(4, 0), (4, 1), (4, 2), (2, 0)],
                         ids=["4p-partner-first", "4p-partner-middle", "4p-partner-last", "2p"])
def test_apply_trade_changes_only_the_two_hands(num_players, rank):
    """Every place of the partner in the decide order: accept then rejects (the
    rejects end in DECIDE_ACCEPTEES), reject-accept-reject, rejects then accept
    (the accept ends it), and the 2-player offer where the partner is the only
    decider."""
    _, game, mover, partner = _position(num_players, rank)
    before = {seat: list(p.resources) for seat, p in enumerate(game.players)}
    for trade in (WOOD_FOR_SHEEP, WOOD_BRICK_FOR_WHEAT):
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
        assert all(a.action_type != cpp.ActionType.CONFIRM_TRADE for a in post.playable_actions)  # the offer is over
        # The engine route and two hand overwrites are the same game, RNG and all.
        assert post.search_equivalent(oracle.trade_by_hands(game, mover, partner, trade))
    assert {seat: list(p.resources) for seat, p in enumerate(game.players)} == before  # the original is untouched
    with pytest.raises(ValueError):
        oracle.apply_trade(game, mover, partner, oracle.Trade((0, 0, 0, 0, 1), (0, 0, 1, 0, 0)))  # the mover has no ore
    with pytest.raises(ValueError):
        oracle.apply_trade(game, mover, partner, oracle.Trade((1, 0, 0, 0, 0), (0, 1, 0, 0, 0)))  # the partner has no brick
    with pytest.raises(ValueError):
        oracle.apply_trade(game, mover, mover, WOOD_FOR_SHEEP)


def test_apply_trade_only_during_the_movers_rolled_play_turn():
    pos = next(p for p in ck.packaged_positions() if p["game"]["num_players"] == 4
               and p["options"]["prompt"] == "PLAY_TURN" and not p["options"]["has_rolled"])
    game = ck._position_game(pos)
    mover = game.current_player_index
    game.set_resources(mover, MOVER_HAND)
    game.set_resources((mover + 1) % 4, PARTNER_HAND)
    assert not oracle._tradeable(game, mover)
    with pytest.raises(ValueError):
        oracle.apply_trade(game, mover, (mover + 1) % 4, WOOD_FOR_SHEEP)


@pytest.mark.parametrize("num_players, level", [(4, "private"), (2, "full")])
def test_gains_are_read_from_the_mover_row_and_its_slots(num_players, level):
    pos, game, mover, partner = _position(num_players, 0)
    n = game.num_players
    encoder = make_encoder(dims_for([pos["game"]]), level)
    if n > 2:
        game.set_resources((mover + 2) % n, [1, 0, 0, 0, 1])  # a second partner, two seats after the mover
        game.set_resources((mover + 3) % n, [0] * 5)  # the last seat has nothing to trade
    slot = (partner - mover) % n
    post = oracle.apply_trade(game, mover, partner, WOOD_FOR_SHEEP)
    pre_values = np.linspace(0.40, 0.10, n)  # every row the table does not name: the gains read 0
    post_values = pre_values.copy()
    post_values[0] += 0.03
    post_values[slot] += 0.04
    post_values[1:] -= 0.07 / (n - 1)  # the rest comes out of every opponent, the partner included
    table = {encoder.encode(post, mover).tobytes(): post_values}
    rows_seen = []

    def critic(rows: np.ndarray) -> np.ndarray:
        rows_seen.append(len(rows))
        return np.stack([table.get(row.tobytes(), pre_values) for row in rows])

    p = oracle.Position(game=game, mover=mover, completed_turns=game.completed_turns, turn=game.num_turns,
                        vps=[0] * n, hand_sizes=[int(sum(x.resources)) for x in game.players], game_id="0.0")
    trades, meta = oracle.evaluate_positions([p], encoder, critic, flush_rows=1)
    expected = len(oracle.enumerate_trades(MOVER_HAND, PARTNER_HAND))
    if n > 2:
        expected += len(oracle.enumerate_trades(MOVER_HAND, [1, 0, 0, 0, 1]))
    assert len(trades["pos"]) == expected == meta[0]["n_trades"]
    assert sum(rows_seen) == 1 + expected  # the mover's pre-trade row and one post-trade row per trade; no waiting seat's row
    assert set(trades["partner_slot"]) == ({1, 2} if n > 2 else {1})
    hit = np.flatnonzero((trades["label"] == WOOD_FOR_SHEEP.label) & (trades["partner_slot"] == slot))
    assert len(hit) == 1
    i = hit[0]
    assert trades["mover_gain"][i] == pytest.approx(0.03)
    assert trades["partner_gain"][i] == pytest.approx(0.04 - 0.07 / (n - 1))  # the partner's slot of the mover's row
    assert trades["others_gain"][i] == pytest.approx(-0.07 * (n - 2) / (n - 1))
    rest = np.ones(len(trades["pos"]), dtype=bool)
    rest[i] = False
    for name in oracle.GAIN_FIELDS:
        assert np.all(trades[name][rest] == 0.0)
    assert meta[0]["mover_value"] == pytest.approx(0.40) and meta[0]["value_sum"] == pytest.approx(pre_values.sum())


def test_summarize_counts_mutual_and_selfish_trades_with_noise_floors():
    trades = {
        "pos": np.array([0, 0, 1]), "partner_slot": np.array([1, 2, 1]), "shape": np.array([0, 1, 0]),
        "label": np.array(["wood>ore", "2wood>ore", "wood>ore"], dtype=object),
        "mover_gain": np.array([0.02, 0.03, 0.006]), "partner_gain": np.array([0.02, -0.03, 0.006]),
        "others_gain": np.array([-0.04, 0.005, -0.012]),  # the second trade leaves 0.5 pp unaccounted for
    }
    meta = [{"mover": 0, "completed_turns": t, "turn": t, "vps": [0] * 4, "hand_sizes": [2] * 4, "game_id": g,
             "n_trades": k, "mover_value": 0.25, "value_sum": 1.0} for t, k, g in ((5, 2, "0.0"), (20, 1, "0.0"), (40, 0, "1.0"))]
    s = oracle.summarize(trades, meta, num_players=4)
    assert s["positions"] == 3 and s["trades"] == 3 and s["positions_with_a_trade"] == 2
    assert s["games"] == {"distinct": 2, "positions_per_game": {"mean": 1.5, "max": 2}}
    assert s["positions_with_mutual_trade"] == {"0.5": pytest.approx(2 / 3, abs=1e-4), "1": pytest.approx(1 / 3, abs=1e-4),
                                                "2": pytest.approx(1 / 3, abs=1e-4), "5": 0.0}
    assert s["best_trade"]["mover_gain_pp"]["p50"] == pytest.approx((2.0 + 0.6) / 2)
    assert s["selfish"]["trade_rate_good_for_mover_bad_for_partner"]["1"] == pytest.approx(1 / 3, abs=1e-4)
    assert s["selfish"]["best_for_mover"]["partner_hurt"]["1"] == pytest.approx(0.5)  # 2wood>ore wins position 0
    assert s["shapes_of_mutual_trades"] == {"1:1": 1.0, "2:1": 0.0, "1:2": 0.0}
    assert [p["positions"] for p in s["by_phase"]] == [1, 1, 1]
    noise = s["noise"]
    # Position 0's trades pair as (2, 2), (3, -3) or (2, -3), (3, 2): a mutual trade at 2 pp either way.
    assert noise["independence_reference_positions_with_mutual_trade"] == {
        "0.5": pytest.approx(2 / 3, abs=1e-4), "1": pytest.approx(1 / 3, abs=1e-4), "2": pytest.approx(1 / 3, abs=1e-4),
        "5": 0.0}
    assert noise["conservation_residual_pp"]["mean"] == pytest.approx(0.5 / 3, abs=1e-3)
    assert noise["positions_with_residual_at_margin"] == {"0.5": pytest.approx(1 / 3, abs=1e-4), "1": 0.0, "2": 0.0, "5": 0.0}
    # Both of game 0's positions qualify at 0.5 pp, one at 1 and 2 pp, none of game 1's: resampling the
    # two games gives 1 or 1/2 (game 0 twice), 2/3 or 1/3 (one each) and 0 (game 1 twice).
    ci = s["positions_with_mutual_trade_ci95"]
    assert ci == {"0.5": [0.0, 1.0], "1": [0.0, 0.5], "2": [0.0, 0.5], "5": [0.0, 0.0]}
    assert ci == oracle.summarize(trades, meta, num_players=4)["positions_with_mutual_trade_ci95"]  # seeded
    assert "others_gain_pp_of_mutual_trades" in s and "others_gain_pp_of_mutual_trades" not in oracle.summarize(
        trades, meta, num_players=2)


def test_default_out_names_a_zero_shot_game():
    assert oracle.default_out("ppo-4p/main").name == "trade-oracle-ppo-4p-main.json"
    assert oracle.default_out("ppo-4p/main", "base-3p").name == "trade-oracle-ppo-4p-main-base-3p.json"
