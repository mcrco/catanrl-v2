"""Test 10: schedules are golden, disjoint, exact, extendable and fill correctly."""

from __future__ import annotations

from collections import Counter

import pytest

from catanrl.eval.suites import load_suite

SUITE = {"name": "t", "purpose": "heldout", "seed": 67, "boards": 3, "scenarios": [{"game": "base-2p"}],
         "turn_limit": 500}

#: Pinned: a change here means every eval record's schedule changed.
GOLDEN = [
    ("s0b0r0", 1708996947412614815, 4585279141350608056, (0, 1)),
    ("s0b0r1", 1708996947412614815, 4585279141350608056, (1, 0)),
]


def test_first_games_are_golden():
    games = load_suite(SUITE).schedule(2)
    found = [(g.game_id, g.map_seed, g.game_seed, g.seats) for g in games[:8]]
    assert found[:2] == GOLDEN
    assert [g.game_id for g in games] == [f"s0b{b}r{r}" for b in range(3) for r in range(2)]
    assert games[0].bot_seed != games[1].bot_seed and games[0].map_hash == games[1].map_hash


def test_select_and_heldout_are_disjoint():
    select, heldout = load_suite({"name": "select-2p", "boards": 100}), load_suite({"name": "heldout-2p", "boards": 100})
    a, b = select.schedule(2), heldout.schedule(2)
    assert not {g.game_seed for g in a} & {g.game_seed for g in b}
    assert not {g.map_hash for g in a} & {g.map_hash for g in b}


def test_rotations_and_quota():
    suite = load_suite({**SUITE, "scenarios": [{"game": "base-2p"}, {"game": "base-3p"}], "boards": 4})
    games = suite.schedule(2, fill=True)
    assert len(games) == 4 * 2 + 4 * 3
    for scenario, n in ((0, 2), (1, 3)):
        mine = [g for g in games if g.scenario == scenario]
        for board in range(4):
            rotations = [g for g in mine if g.board == board]
            for seat in range(n):
                assert Counter(g.seats[seat] for g in rotations) == Counter(min(2, (seat + r) % n) for r in range(n))
    with pytest.raises(ValueError):
        load_suite({**SUITE, "scenarios": [{"game": "base-3p"}]}).schedule(2)  # needs fill


def test_larger_suites_extend_and_names_do_not_matter():
    small, large = load_suite(SUITE).schedule(2), load_suite({**SUITE, "boards": 6}).schedule(2)
    assert large[: len(small)] == small
    renamed = {**SUITE, "scenarios": [{"game": {"name": "other-name", "num_players": 2, "map": {"type": "BASE"},
                                                "victory_points_to_win": 15, "discard_limit": 9}}]}
    assert load_suite(renamed).schedule(2) == small
    assert load_suite(renamed).schedule_hash(2) == load_suite(SUITE).schedule_hash(2)


@pytest.mark.parametrize(("game", "n"), [("base-4p", 4), ("extension-6p", 6)])
def test_fill_expands(game, n):
    games = load_suite({**SUITE, "scenarios": [{"game": game}], "boards": 1}).schedule(2, fill=True)
    assert len(games) == n
    for g in games:
        assert sorted(g.seats) == [0, 1] + [2] * (n - 2)
    assert [g.seats.index(0) for g in games] == [(-r) % n for r in range(n)]
