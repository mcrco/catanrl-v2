"""The native evaluation harness: rows, bots, agents and backends."""

from __future__ import annotations

import cppanatron as cpp
import numpy as np
import pytest
from eval_standins import Suite, agent, schedule, tiny_model

from catanrl.eval.harness import MT19937_64, below, play_schedule

HAS_SCHEDULE_MODE = hasattr(cpp.VecEnv, "start_games")


def test_mt19937_64_matches_the_standard():
    rng = MT19937_64(5489)  # std::mt19937_64's default seed; [rand.predef] pins the 10000th draw
    for _ in range(9999):
        rng()
    assert rng() == 9981545732273789042


def test_below_is_uniform_and_in_range():
    rng = MT19937_64(3)
    draws = [below(rng, 7) for _ in range(7000)]
    assert set(draws) == set(range(7))
    assert max(draws.count(i) for i in range(7)) < 1150


def play(lineup, suite, games, **kwargs):
    rows = []
    play_schedule(lineup, games, suite=suite, on_game=rows.append, **{"backend": "stepper", **kwargs})
    return sorted(rows, key=lambda r: r["game_id"])


def test_rows_score_every_slot():
    suite = Suite(({"game": "base-2p"},), turn_limit=200)
    games = schedule(suite, 2, boards=3)
    rows = play([agent("POLICY", tiny_model()), agent("R")], suite, games)
    assert [r["game_id"] for r in rows] == sorted(g.game_id for g in games)
    for row, scheduled in zip(rows, sorted(games, key=lambda g: g.game_id)):
        assert row["seats"] == list(scheduled.seats)
        assert sum(row["scores"]) == pytest.approx(1.0)
        if row["truncated"]:
            assert row["winner"] is None and row["scores"] == [0.5, 0.5]
        else:
            assert row["scores"][row["winner"]] == 1.0 and row["seats"][row["winner_seat"]] == row["winner"]


def test_fill_plays_the_empty_seats():
    suite = Suite(({"game": "base-4p"},), turn_limit=150)
    games = schedule(suite, 2, boards=1)
    assert any(2 in g.seats for g in games)
    rows = play([agent("POLICY", tiny_model(["base-4p"])), agent("R")], suite, games, fill=agent("F"))
    assert len(rows) == 4 and all(len(r["scores"]) == 3 for r in rows)


def test_sampled_policy_reproduces_from_decision_seeds():
    suite = Suite(({"game": "base-2p"},), turn_limit=120)
    games = schedule(suite, 2, boards=2)
    lineup = lambda: [agent("POLICY", tiny_model(), sample=True, temperature=1.0), agent("F")]
    assert play(lineup(), suite, games) == play(lineup(), suite, games, num_envs=1)


def test_unknown_backend_and_missing_fill_raise():
    suite = Suite(({"game": "base-4p"},), turn_limit=150)
    games = schedule(suite, 2, boards=1)
    with pytest.raises(ValueError):
        play([agent("POLICY", tiny_model(["base-4p"])), agent("R")], suite, games)
    with pytest.raises(ValueError):
        play([agent("R"), agent("R")], Suite(({"game": "base-2p"},)), schedule(Suite(({"game": "base-2p"},)), 2, 1),
             backend="mystery")


@pytest.mark.skipif(not HAS_SCHEDULE_MODE, reason="needs VecEnv schedule mode (engine E1)")
@pytest.mark.parametrize("bot", ["F", "R"])
def test_stepper_and_vecenv_play_the_same_games(bot):
    suite = Suite(({"game": "base-2p"}, {"game": "base-3p"}), turn_limit=200)
    games = schedule(suite, 2, boards=3)
    lineup = lambda: [agent("POLICY", tiny_model(["base-2p", "base-3p"])), agent(bot)]
    stepped = play(lineup(), suite, games, fill=agent("F"))
    vectorized = play(lineup(), suite, games, fill=agent("F"), backend="vecenv", num_envs=4, threads=2)
    assert stepped == vectorized


def test_policy_of_an_asymmetric_critic_model_plays_from_actor_rows():
    from catanrl.envs.encoding import action_size, dims_for, row_layout
    from catanrl.models import build_model

    dims = dims_for(["base-2p"])
    model = build_model({"preset": "graph-tiny"}, dims=dims, levels={"level": "private", "critic_level": "full"},
                        num_actions=action_size(dims), layout=row_layout(dims, "private", "full"))
    suite = Suite(({"game": "base-2p"},), turn_limit=100)
    rows = play([agent("POLICY", model), agent("R")], suite, schedule(suite, 2, boards=1))
    assert len(rows) == 2
    with pytest.raises(NotImplementedError):
        agent("MCTS", model, sims=4, oracle=True)


class _FailingEvaluator:
    """Fails on its n-th call with a recognizable error."""

    def __init__(self, inner, fail_on):
        self.inner, self.calls, self.fail_on = inner, 0, fail_on
        self.obs_size, self.num_actions, self.num_players = inner.obs_size, inner.num_actions, inner.num_players

    def __call__(self, rows):
        self.calls += 1
        if self.calls == self.fail_on:
            raise MemoryError("evaluator failed")
        return self.inner(rows)


def _decisions(count=3):
    from catanrl import games as game_configs

    out = []
    for seed in range(count):
        game = game_configs.load_game("base-2p").cppanatron_game(seed, seed)
        while len(game.playable_actions) < 2:
            game.execute(game.playable_actions[0])
        out.append(game)
    return out


def test_mcts_errors_propagate_and_leave_the_pool_usable():
    mcts = agent("MCTS", tiny_model(), sims=8, oracle=True)
    healthy = mcts.evaluator
    games = _decisions()
    rows = np.stack([mcts.encode(g, g.current_player_index) for g in games])
    masks, seeds = rows[:, -mcts.action_space.size:], np.arange(len(games), dtype=np.uint64)
    mcts.evaluator = _FailingEvaluator(healthy, fail_on=3)
    with pytest.raises(MemoryError, match="evaluator failed"):  # the real error, not the pool's
        mcts.act(rows, masks, seeds, games=games)
    assert len(mcts.pool) == 0 and mcts.pool.pending == 0
    mcts.evaluator = healthy
    assert len(mcts.act(rows, masks, seeds, games=games)) == len(games)
    assert len(mcts.pool) == 0


def test_mcts_add_failure_leaks_no_search():
    import cppanatron as cpp

    mcts = agent("MCTS", tiny_model(), sims=4, dets=2)
    games = _decisions(2)
    rows = np.stack([mcts.encode(g, g.current_player_index) for g in games])
    good = cpp.HandTracker(games[0], games[0].current_player_index)
    wrong = cpp.HandTracker(games[1], 1 - games[1].current_player_index)  # follows the wrong seat
    with pytest.raises(ValueError):
        mcts.act(rows, rows[:, -mcts.action_space.size:], np.arange(2, dtype=np.uint64), games=games,
                 trackers=[good, wrong])
    assert len(mcts.pool) == 0


@pytest.mark.parametrize("backend", ["stepper", "vecenv"])
@pytest.mark.parametrize("num_envs, threads", [(0, 1), (-2, 1), (4, 0)])
def test_non_positive_envs_or_threads_raise(backend, num_envs, threads):
    """The stepper used to spin forever with num_envs <= 0 while the VecEnv clamped it."""
    suite = Suite(({"game": "base-2p"},), turn_limit=50)
    with pytest.raises(ValueError, match="positive"):
        play([agent("R"), agent("F")], suite, schedule(suite, 2, boards=1), backend=backend,
             num_envs=num_envs, threads=threads)
