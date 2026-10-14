"""A seat's view as a cppanatron game: rules, maps, the apply_setup snapshot,
and the key equivalence: the bridged game of a seat encodes, at PRIVATE
level, exactly as the native game does for that seat, at every step of
random 2-6 player games."""

from __future__ import annotations

import os

import numpy as np
import pytest
from bridge_games import GAMES, lockstep

from catanrl.bridge.position import (
    PROMPT_COVERAGE,
    Inexpressible,
    SeatView,
    ViewError,
    game_rules,
    public_mismatches,
    shared_map,
    snapshot,
    transport,
)
from catanrl.bridge.tracking import SeatTracker
from catanrl.envs.encoding import make_encoder

#: Random games per config (CATANRL_BRIDGE_GAMES fuzzes more) and their length cap.
GAMES_PER_CONFIG = int(os.environ.get("CATANRL_BRIDGE_GAMES", "1"))
MAX_STEPS = int(os.environ.get("CATANRL_BRIDGE_STEPS", "600"))


# ---------------------------------------------------------------- dealing


def test_transport_meets_bounds_and_keeps_preferred_cards():
    deal = transport([3, 2, 0, 1, 0], {"a": 3, "b": 3}, lower={"a": [0, 2, 0, 0, 0]},
                     prefer={"a": [3, 0, 0, 0, 0], "b": [0, 2, 0, 1, 0]})
    assert deal["a"][1] == 2 and sum(deal["a"]) == 3 and sum(deal["b"]) == 3
    assert [deal["a"][k] + deal["b"][k] for k in range(5)] == [3, 2, 0, 1, 0]
    assert deal["a"][0] == 1  # keeps one of its wood where the bound leaves room


def test_transport_reassigns_along_paths_and_refuses_the_impossible():
    # b may hold no wood, so the wood must all go to a, which must give up its brick.
    deal = transport([2, 1, 0, 0, 0], {"a": 2, "b": 1}, upper={"b": [0, 1, 1, 1, 1]},
                     prefer={"a": [1, 1, 0, 0, 0], "b": [1, 0, 0, 0, 0]})
    assert deal == {"a": [2, 0, 0, 0, 0], "b": [0, 1, 0, 0, 0]}
    with pytest.raises(ValueError):
        transport([2, 0, 0, 0, 0], {"a": 1, "b": 1}, upper={"a": [0] * 5})
    with pytest.raises(ValueError):
        transport([1, 0, 0, 0, 0], {"a": 2}, lower={})


# ---------------------------------------------------------------- rules and maps


@pytest.mark.parametrize("name", GAMES)
def test_rules_and_map_come_from_the_view(name):
    ls = lockstep(name, 0)
    for _ in range(120):
        ls.step(ls.choose())
    view = ls.view(1)
    assert game_rules(view) == ls.native.rules
    assert game_rules(view, ls.config) == ls.native.rules
    catan_map = shared_map(view["map"])
    assert catan_map is shared_map(dict(view["map"]))  # one shared map per document
    assert catan_map.spec().to_dict() == ls.native.board.map.spec().to_dict()


def test_rules_refuse_a_supply_the_view_contradicts():
    view = lockstep("base-3p", 0).view(0)
    with pytest.raises(ViewError):
        game_rules(view, "extension")
    assert game_rules(view, "base") == game_rules(view)


def test_seat_view_reads_one_seat():
    ls = lockstep("base-4p", 1)
    for _ in range(200):
        ls.step(ls.choose())
    view = ls.view(2)
    sv = SeatView.of(view)
    assert sv.seat == 2 and sv.visible(2) and not sv.visible(0)
    assert sv.resources(2) == tuple(ls.native.player(2).resources)
    assert sv.hand_sizes() == tuple(sum(p.resources) for p in ls.native.players)
    with pytest.raises(ViewError):
        sv.resources(0)


# ---------------------------------------------------------------- the snapshot


def test_snapshot_expresses_turn_positions_and_refuses_the_rest():
    assert "DISCARD" in PROMPT_COVERAGE and PROMPT_COVERAGE["DISCARD"][0] == "replay"
    exact = refused = 0
    for index in range(3):
        ls = lockstep("base-3p", index)
        encoder = make_encoder(ls.dims, "private")
        for _ in range(MAX_STEPS):
            if ls.over:
                break
            ls.step(ls.choose())
            seat = ls.current_seat
            view = ls.view(seat)
            try:
                position = snapshot(view)
            except Inexpressible:
                assert (view["current_prompt"] != "PLAY_TURN" or view["is_road_building"]
                        or view["is_initial_build_phase"])
                refused += 1
                continue
            assert position.method == "snapshot" and position.seat == seat
            assert position.mismatches == tuple(public_mismatches(position.game, view, seat))
            if position.exact:
                exact += 1
                assert np.array_equal(encoder.encode(position.game, seat), encoder.encode(ls.native, seat))
            else:  # only what PROMPT_COVERAGE says apply_setup cannot express
                for mismatch in position.mismatches:
                    assert any(what in mismatch for what in ("road length", "longest road", "awards",
                                                            "playable cards", "points")), mismatch
    assert exact > 100 and refused > 20


def test_two_player_placeholders_are_the_truth():
    """With one other seat its hand is the bank's complement: the snapshot's
    and the replay's placeholders are its true hand (what makes a PUBLIC
    POLICY fair in two-player games)."""
    checked = 0
    for index in range(2):
        ls = lockstep("base-2p", index)
        tracker = SeatTracker(ls.view(0))
        for _ in range(400):
            if ls.over:
                break
            ls.step(ls.choose())
            view = ls.view(0)
            position = tracker.sync(view)
            assert position.game.player(1).resources == ls.native.player(1).resources
            if view["current_prompt"] == "PLAY_TURN" and not view["is_road_building"]:
                assert snapshot(view).game.player(1).resources == ls.native.player(1).resources
                checked += 1
    assert checked > 100


# ---------------------------------------------------------------- the key equivalence


def _equivalence(name: str, index: int, every: int) -> int:
    """Play a random game; every ``every`` steps (and at the seat's own
    decisions) sync each seat's tracker from its view and compare its game
    with the native one. Returns the number of comparisons."""
    ls = lockstep(name, index)
    n = len(ls.colors)
    trackers = [SeatTracker(view) for view in ls.views()]
    encoder = make_encoder(ls.dims, "private")
    compared = 0
    for step in range(MAX_STEPS):
        if ls.over:
            break
        ls.step(ls.choose())
        over = ls.truth.winning_color() is not None
        syncing = [seat for seat in range(n) if not step % every or over or ls.current_seat == seat]
        views = ls.views() if len(syncing) > 1 else {seat: ls.view(seat) for seat in syncing}
        for seat in syncing:
            tracker = trackers[seat]
            position = tracker.sync(views[seat], game_over=over)
            if position.mismatches:
                # Only a decide-only stream can lose a monopoly's takes (tracking module doc).
                assert every > 1 and tracker.stats["broken"] + tracker.stats["mismatched_syncs"], position
                continue
            assert np.array_equal(encoder.encode(position.game, seat), encoder.encode(ls.native, seat)), (
                name, index, step, seat)
            compared += 1
    return compared


@pytest.mark.parametrize("name", GAMES)
def test_bridged_games_encode_like_native_ones_every_step(name):
    """Synced after every step (the server path, which sees every step)."""
    total = sum(_equivalence(name, index, every=1) for index in range(GAMES_PER_CONFIG))
    assert total > 200


@pytest.mark.parametrize("name", ("base-2p", "base-4p", "extension-6p"))
def test_bridged_games_encode_like_native_ones_from_sparse_views(name):
    """Synced every 7 steps and at the seat's turns (the TUI path, which sees
    views only when the seat decides)."""
    total = sum(_equivalence(name, index, every=7) for index in range(GAMES_PER_CONFIG))
    assert total > 100


def test_a_seat_not_in_the_game_is_refused():
    view = lockstep("base-2p", 0).view(0)
    with pytest.raises(ViewError):
        SeatView.of(view, "WHITE")
