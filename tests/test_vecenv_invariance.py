"""Test 20: VecEnv buffers do not depend on threads; env i does not depend on
num_envs; schedule mode reproduces a scheduled game exactly."""

from __future__ import annotations

import cppanatron as cpp
import numpy as np
import pytest
from eval_standins import Suite, agent, schedule

from catanrl.envs.puffer import plan_env
from catanrl.eval.harness import play_schedule

PLAN = plan_env({"scenarios": [
    {"game": "base-2p", "maps": 4},
    {"game": "base-3p", "maps": 4, "seats": ["learner", "F", "R"]},
]}, pool_seed=9)


def roll(num_envs: int, threads: int, steps: int = 120, seed: int = 21):
    vec = cpp.VecEnv(PLAN.scenarios, num_envs=num_envs, dims=cpp.EncoderDims(**PLAN.dims), seed=seed,
                     num_threads=threads)
    buffers = vec.allocate()
    vec.reset()
    size = int(vec.action_size)
    players = int(vec.max_players)
    history = []
    for step in range(steps):
        acting = np.flatnonzero(buffers["acting"])
        for row in acting:
            env = row // players
            legal = np.flatnonzero(buffers["observations"][row, -size:])
            # Each env's actions come from its own stream, so env i acts alike in any VecEnv.
            choice = np.random.default_rng([seed, int(env), step]).integers(len(legal))
            buffers["actions"][row] = legal[choice]
        vec.step()
        history.append({key: np.array(value, copy=True) for key, value in buffers.items()})
    return history


def test_buffers_do_not_depend_on_threads():
    one, four = roll(8, 1), roll(8, 4)
    for a, b in zip(one, four):
        for key in a:
            assert np.array_equal(a[key], b[key]), key


def test_env_does_not_depend_on_num_envs():
    small, large = roll(8, 2), roll(16, 3)
    players = PLAN.dims["max_players"]
    rows = slice(0, 8 * players)
    for a, b in zip(small, large):
        for key in ("observations", "rewards", "terminals", "acting", "learner", "seat"):
            assert np.array_equal(a[key], b[key][rows]), key
        assert np.array_equal(a["num_players"], b["num_players"][:8])


@pytest.mark.skipif(not hasattr(cpp.VecEnv, "start_games"), reason="needs engine E1")
def test_schedule_mode_reproduces_a_scheduled_game():
    """A bots-only game started in schedule mode ends exactly like the same
    ScheduledGame stepped in Python (engine F, engine R stream)."""
    suite = Suite(({"game": "base-3p"},), turn_limit=300)
    games = schedule(suite, 3, boards=2)
    lineup = [agent("F"), agent("R"), agent("R")]
    rows = {}
    for backend in ("vecenv", "stepper"):
        found = []
        play_schedule(lineup, games, suite=suite, on_game=found.append, backend=backend, num_envs=3, threads=2)
        rows[backend] = sorted(found, key=lambda r: r["game_id"])
    assert rows["vecenv"] == rows["stepper"]
    assert any(not r["truncated"] for r in rows["stepper"])
