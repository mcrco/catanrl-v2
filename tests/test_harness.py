"""The native evaluation harness: rows, bots, trackers and agents."""

from __future__ import annotations

import cppanatron as cpp
import numpy as np
import pytest
from eval_helpers import agent, make_suite, schedule, tiny_model
from reference_stepper import MT19937_64, below, play_reference

from catanrl.eval.harness import play_schedule


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
    play_schedule(lineup, games, suite=suite, on_game=rows.append, **kwargs)
    return sorted(rows, key=lambda r: r["game_id"])


def test_rows_score_every_slot():
    suite = make_suite(({"game": "base-2p"},), turn_limit=200)
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
    suite = make_suite(({"game": "base-4p"},), turn_limit=150)
    games = schedule(suite, 2, boards=1)
    assert any(2 in g.seats for g in games)
    rows = play([agent("POLICY", tiny_model(["base-4p"])), agent("R")], suite, games, fill=agent("F"))
    assert len(rows) == 4 and all(len(r["scores"]) == 3 for r in rows)


def test_sampled_policy_reproduces_from_decision_seeds():
    suite = make_suite(({"game": "base-2p"},), turn_limit=120)
    games = schedule(suite, 2, boards=2)
    lineup = lambda: [agent("POLICY", tiny_model(), sample=True, temperature=1.0), agent("F")]
    assert play(lineup(), suite, games) == play(lineup(), suite, games, num_envs=1)


def test_missing_fill_raises():
    suite = make_suite(({"game": "base-4p"},), turn_limit=150)
    games = schedule(suite, 2, boards=1)
    with pytest.raises(ValueError):
        play([agent("POLICY", tiny_model(["base-4p"])), agent("R")], suite, games)


# Agents whose decisions read games and VecEnv's hand trackers (E6), and bots.
REFERENCE_CASES = {
    "policy-F": (("POLICY", {}), "F"),
    "policy-R": (("POLICY", {}), "R"),
    "mcts-dets": (("MCTS", {"sims": 6, "dets": 2}), "R"),
    "belief": (("BELIEF", {"worlds": 3}), "F"),
}


@pytest.mark.parametrize("case", sorted(REFERENCE_CASES))
def test_vecenv_plays_exactly_the_reference_games(case):
    """play_schedule (VecEnv schedule mode, engine bots, engine trackers) gives exactly the rows of
    the games stepped one action at a time in Python with HandTracker.observe on every transition."""
    (key, params), bot = REFERENCE_CASES[case]
    suite = make_suite(({"game": "base-2p"}, {"game": "base-3p"}), turn_limit=120)
    games = schedule(suite, 2, boards=2 if key == "MCTS" else 3)
    lineup = lambda: [agent(key, tiny_model(["base-2p", "base-3p"], level="full"), infer_rows=8, **params),
                      agent(bot)]
    reference = play_reference(lineup(), games, suite=suite, fill=agent("F"))
    assert play(lineup(), suite, games, fill=agent("F"), num_envs=4, threads=2) == reference


def test_policy_of_an_asymmetric_critic_model_plays_from_actor_rows():
    from catanrl.envs.encoding import action_size, dims_for, row_layout
    from catanrl.models import build_model

    dims = dims_for(["base-2p"])
    model = build_model({"preset": "graph-tiny"}, dims=dims, levels={"level": "private", "critic_level": "full"},
                        num_actions=action_size(dims), layout=row_layout(dims, "private", "full"))
    suite = make_suite(({"game": "base-2p"},), turn_limit=100)
    rows = play([agent("POLICY", model), agent("R")], suite, schedule(suite, 2, boards=1))
    assert len(rows) == 2


@pytest.mark.parametrize("params", [{"dets": 2}, {"oracle": True}])
def test_mcts_searches_with_a_critic_at_another_level(params):
    """Leaves carry the critic block (ActorCriticEncoder, engine E8), as in training rows."""
    from catanrl.envs.encoding import action_size, dims_for, make_encoder, row_layout
    from catanrl.models import build_model

    dims = dims_for(["base-2p"])
    model = build_model({"preset": "graph-tiny"}, dims=dims, levels={"level": "private", "critic_level": "full"},
                        num_actions=action_size(dims), layout=row_layout(dims, "private", "full"))
    mcts = agent("MCTS", model, sims=6, **params)
    assert mcts.pool.observation_size == model.obs_size
    game = _decisions(1)[0]
    leaf = cpp.ActorCriticEncoder(cpp.EncoderDims(**dims), cpp.InfoLevel.PRIVATE, cpp.InfoLevel.FULL)
    offset = row_layout(dims, "private", "full")["players"][0]
    full = make_encoder(dims, "full").encode(game, 0)
    expected = np.concatenate([make_encoder(dims, "private").encode(game, 0), full[offset : offset + 2 * 46]])
    assert np.array_equal(leaf.encode(game, 0), expected)
    suite = make_suite(({"game": "base-2p"},), turn_limit=60)
    games = schedule(suite, 2, boards=1)
    lineup = lambda: [agent("MCTS", model, sims=6, **params), agent("R")]
    assert play(lineup(), suite, games) == play_reference(lineup(), games, suite=suite)


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


@pytest.mark.parametrize("num_envs, threads", [(0, 1), (-2, 1), (4, 0)])
def test_non_positive_envs_or_threads_raise(num_envs, threads):
    """The old Python stepper spun forever with num_envs <= 0 while the VecEnv clamped it."""
    suite = make_suite(({"game": "base-2p"},), turn_limit=50)
    with pytest.raises(ValueError, match="positive"):
        play([agent("R"), agent("F")], suite, schedule(suite, 2, boards=1), num_envs=num_envs, threads=threads)


SMALL = {"name": "small-3p", "num_players": 3, "map": {"type": "blob", "land_tiles": [10, 14]}}


def test_small_maps_play():
    """Suites whose maps are all smaller than BASE used to crash the VecEnv at construction."""
    suite = make_suite(({"game": SMALL}, {"game": {**SMALL, "name": "small-3p-b", "map": {"type": "hexagon", "radius": 1}}}),
                  turn_limit=120)
    games = [g for g in schedule(suite, 2, boards=2) if g.scenario == 0]
    lineup = lambda: [agent("POLICY", tiny_model([SMALL])), agent("F")]
    rows = play(lineup(), suite, games, fill=agent("R"), num_envs=3)
    assert len(rows) == len(games)
    assert rows == play_reference(lineup(), games, suite=suite, fill=agent("R"))


def test_settlement_capacity_matches_the_engine():
    """The VecEnv refuses exactly the maps settlement_capacity says cannot seat 2n settlements."""
    from catanrl.eval.harness import settlement_capacity

    for tiles in range(1, 9):
        spec = {"name": "tiny", "num_players": 2, "map": {"type": "blob", "land_tiles": tiles}}
        config = __import__("catanrl.games", fromlist=["load_game"]).load_game(spec)
        catan_map = cpp.CatanMap.from_spec(cpp.MapSpec.from_dict(config.map_spec(0)))
        for players in range(2, 7):
            scenario = cpp.Scenario(num_players=players, maps=[catan_map])
            dims = cpp.EncoderDims(max_players=6, max_tiles=40, max_nodes=200, max_edges=200)
            try:
                cpp.VecEnv([scenario], num_envs=1, dims=dims)
                accepted = True
            except ValueError as error:
                assert "seat" in str(error)
                accepted = False
            assert accepted == (settlement_capacity(catan_map) >= 2 * players), (tiles, players)


def test_unseatable_maps_are_refused_before_any_game():
    """Refused up front (the old Python stepper scored them as stuck 1/N rows; the VecEnv refused
    them in start_games, after other games had played)."""
    crowded = {"name": "crowded-6p", "num_players": 6, "map": {"type": "blob", "land_tiles": 5}}  # seats <= 11
    suite = make_suite(({"game": "base-2p"}, {"game": crowded}), turn_limit=100)
    games = schedule(suite, 2, boards=1)
    played = []
    with pytest.raises(ValueError, match="cannot seat"):
        play_schedule([agent("R"), agent("F")], games, fill=agent("R"), suite=suite, on_game=played.append)
    assert played == []
