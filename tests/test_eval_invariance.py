"""Test 21: native eval rows do not depend on num_envs, threads, workers,
interrupts or reruns, and play exactly the schedule's games."""

from __future__ import annotations

import pytest
from eval_helpers import agent, make_suite, schedule, tiny_model
from reference_stepper import play_reference

from catanrl.eval.harness import play_schedule

SUITE = make_suite(({"game": "base-2p"}, {"game": "base-3p"}), turn_limit=160)
GAMES = schedule(SUITE, 3, boards=4)  # 16 boards: 4 x 2p rotations + 4 x 3p rotations = 20 games
MODEL = tiny_model(["base-2p", "base-3p"], level="full", seed=11)


def lineup(key: str, **params):
    first = agent(key, MODEL, infer_rows=8, **params)
    return [first, agent("F"), agent("R")]


def play(players, games=GAMES, **kwargs):
    rows = []
    play_schedule(players, games, suite=SUITE, on_game=rows.append, **kwargs)
    return {row["game_id"]: row for row in rows}


CASES = {
    "policy": ("POLICY", {}),
    "sampled": ("POLICY", {"sample": True}),
    "mcts-oracle": ("MCTS", {"sims": 6, "oracle": True}),
    "mcts-dets": ("MCTS", {"sims": 6, "dets": 2}),
    "mcts-reuse": ("MCTS", {"sims": 6, "oracle": True, "reuse": True}),
    "mcts-reuse-dets": ("MCTS", {"sims": 6, "dets": 2, "reuse": True}),
    # Enough simulations for the transposition table to link nodes in these games.
    "mcts-reuse-tt": ("MCTS", {"sims": 16, "oracle": True, "reuse": True, "transpositions": True}),
    "mcts-reuse-tt-dets": ("MCTS", {"sims": 16, "dets": 2, "reuse": True, "transpositions": True}),
    "belief": ("BELIEF", {"worlds": 3}),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_rows_are_invariant(case):
    key, params = CASES[case]
    # Searches are slow on CPU: a 2p and a 3p board are enough for them.
    games = [g for g in GAMES if g.board == 0] if key == "MCTS" else GAMES
    reference = play(lineup(key, **params), games, num_envs=16, threads=1)
    assert sorted(reference) == sorted(g.game_id for g in games)
    for game in games:
        assert reference[game.game_id]["seats"] == list(game.seats)
    for num_envs, threads in ((1, 1), (5, 4)):
        assert play(lineup(key, **params), games, num_envs=num_envs, threads=threads) == reference
    # Two workers (halves of the schedule) and an interrupted run resumed on the rest.
    half = len(games) // 2
    split = {**play(lineup(key, **params), games[:half], num_envs=3),
             **play(lineup(key, **params), games[half:], num_envs=7)}
    assert split == reference
    # A rerun in another order.
    assert play(lineup(key, **params), list(reversed(games)), num_envs=4) == reference


def test_policy_agrees_with_the_reference_stepper():
    reference = {row["game_id"]: row for row in play_reference(lineup("POLICY"), GAMES, suite=SUITE)}
    assert play(lineup("POLICY"), num_envs=6, threads=4) == reference
