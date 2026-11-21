"""Test 35: MCTS over likely worlds (``world_choice=likely``): an agent with ``dets=8`` plays
``base-2p-fr`` on the CPU with a tiny random network, at both roots and with reuse; its
searches hold the most likely deals of the hidden cards, weighted by their probabilities,
whatever the decision's seed; the default stays the sampled worlds."""

from __future__ import annotations

import cppanatron as cpp
import numpy as np
import pytest
from eval_helpers import agent, make_suite, schedule, tiny_model

from catanrl import games as game_configs
from catanrl.eval.harness import play_schedule

GAME = "base-2p-fr"
SUITE = make_suite(({"game": GAME},), turn_limit=24)
GAMES = schedule(SUITE, 2, boards=1)
MODEL = tiny_model([GAME], level="private", seed=5)


def mcts(**params):
    return agent("MCTS", MODEL, infer_rows=16, sims=16, dets=8, pending=4, **params)


def hidden_cards_position(hidden: int, seed: int = 1):
    """A ``base-2p-fr`` game where the seat to move has a choice and the other seat holds at
    least ``hidden`` unplayed development cards, with the mover's hand tracker."""
    config = game_configs.load_game(GAME)
    for start in range(seed, seed + 40):
        game = config.cppanatron_game(start, start)
        trackers = [cpp.HandTracker(game, seat) for seat in range(2)]
        rng = np.random.default_rng(start)
        for step in range(1500):
            actions = game.playable_actions
            action = actions[int(rng.integers(len(actions)))]
            before = game.copy()
            game.execute(action)
            for tracker in trackers:
                tracker.observe(before, action, game)
            if game.winning_color is not None:
                break
            mover = game.current_player_index
            if (step % 10 == 0 and len(game.playable_actions) >= 2
                    and trackers[mover].development_card_count(1 - mover) >= hidden):
                return game, trackers[mover]
    raise AssertionError("no random game reached the position")


def decide(player, game, tracker, seed: int) -> int:
    """One ``act`` on ``game`` as a stream's first decision; returns the search's id (with
    ``reuse`` the search stays in the pool)."""
    rows = np.stack([player.encode(game, game.current_player_index)])
    masks = rows[:, -player.action_space.size:]
    choice = player.act(rows, masks, np.array([seed], dtype=np.uint64), games=[game], trackers=[tracker],
                        streams=np.array([0], dtype=np.int64), first=np.array([True]))
    assert masks[0, int(choice[0])]
    return player._live[0][0]


@pytest.mark.parametrize("root", ["puct", "gumbel"])
def test_an_agent_plays_base_2p_fr_over_likely_worlds(root):
    player = mcts(world_choice="likely", root=root, reuse=True)
    assert player.config.world_choice == cpp.WorldChoice.LIKELY and player.config.determinizations == 8
    assert player.needs_trackers and player.reuse
    rows = []
    play_schedule([player, agent("F")], GAMES, suite=SUITE, on_game=rows.append, num_envs=2)
    assert len(rows) == len(GAMES) == 2
    counters = player.counters
    assert counters["decisions"] == sum(row["decisions"][row["seats"].index(0)] for row in rows) > 0
    assert counters["fresh"] > 0 and counters["fresh"] + counters["reused"] == counters["decisions"]
    assert player.search_stats["simulations"] >= 16 * counters["fresh"]


@pytest.mark.parametrize("root", ["puct", "gumbel"])
def test_the_agents_worlds_are_the_most_likely_deals(root):
    game, tracker = hidden_cards_position(3)
    deals = cpp.likely_worlds(game, tracker)
    assert len(deals) > 8
    kept = sum(probability for _, probability in deals[:8])
    expected = [probability / kept for _, probability in deals[:8]]
    player = mcts(world_choice="likely", root=root, reuse=True)
    for seed in (3, 4):  # the worlds are the same whatever the decision's seed
        search = decide(player, game, tracker, seed)
        assert player.pool.stats(search).worlds == 8
        np.testing.assert_allclose(player.pool.world_weights(search), expected, rtol=1e-12)
        by_world = player.pool.world_visits(search)
        np.testing.assert_array_equal(by_world.sum(axis=0), player.pool.visits(search))
        assert by_world.sum() == 16

    sampled = mcts(root=root, reuse=True)
    assert sampled.config.world_choice == cpp.WorldChoice.SAMPLE
    search = decide(sampled, game, tracker, 3)
    shares = sampled.pool.world_weights(search)
    assert np.allclose(shares * 8, np.round(shares * 8)) and shares.sum() == pytest.approx(1.0)
