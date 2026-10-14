"""Test 21: native eval rows do not depend on num_envs, threads, workers,
interrupts or reruns, and play exactly the schedule's games."""

from __future__ import annotations

import cppanatron as cpp
import pytest
from eval_standins import Suite, agent, schedule, tiny_model

from catanrl.eval.harness import play_schedule

pytestmark = pytest.mark.skipif(not hasattr(cpp.VecEnv, "start_games"), reason="needs engine E1/E2")

SUITE = Suite(({"game": "base-2p"}, {"game": "base-3p"}), turn_limit=160)
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
    "belief": ("BELIEF", {"worlds": 3}),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_rows_are_invariant(case):
    key, params = CASES[case]
    reference = play(lineup(key, **params), num_envs=16, threads=1)
    assert sorted(reference) == sorted(g.game_id for g in GAMES)
    for game in GAMES:
        assert reference[game.game_id]["seats"] == list(game.seats)
    for num_envs, threads in ((1, 1), (5, 4)):
        assert play(lineup(key, **params), num_envs=num_envs, threads=threads) == reference
    # Two workers (halves of the schedule) and an interrupted run resumed on the rest.
    half = len(GAMES) // 2
    split = {**play(lineup(key, **params), GAMES[:half], num_envs=3),
             **play(lineup(key, **params), GAMES[half:], num_envs=7)}
    assert split == reference
    # A rerun in another order.
    assert play(lineup(key, **params), list(reversed(GAMES)), num_envs=4) == reference


def test_policy_agrees_across_backends():
    vectorized = play(lineup("POLICY"), backend="vecenv", num_envs=6, threads=4)
    stepped = play(lineup("POLICY"), backend="stepper", num_envs=6)
    assert vectorized == stepped
