"""MCTS ``deadline``: a search that runs out of wall clock stops between rounds, after at
least one (a fresh search's first round expands its root, so at worst the move is the
prior's best), with nothing pending, and plays its most-visited move so far; the agent counts
it as ``cut``, a reusing seat re-roots a cut search as any other, and a caller's
``stop_at`` (the ladder bridges' budget for one move) ends the search earlier still."""

from __future__ import annotations

import time

import cppanatron as cpp
from eval_helpers import agent, tiny_model

from catanrl.search import run_searches
from test_mcts_reuse import decide, position

MODEL = tiny_model(["base-2p", "base-3p"], level="full", seed=11)


def mcts(**params):
    return agent("MCTS", MODEL, infer_rows=8, **params)


def test_a_search_past_its_stop_runs_one_round_and_stays_whole():
    searcher = mcts(sims=4096, dets=1)
    game = position(3)
    assert searcher.needs_trackers  # dets=1: one world dealt from the mover's tracker
    sid = searcher.pool.add(game, seed=1, tracker=cpp.HandTracker(game, game.current_player_index),
                            deal=searcher.tracker_mode)
    rows = run_searches(searcher.pool, searcher.evaluator, batch_rows=8, stop=time.monotonic() - 1.0)
    assert rows > 0 and searcher.pool.pending == 0 and not searcher.pool.done(sid)
    # One round of a fresh search expands its root: no simulation yet, so select plays the
    # prior's best (the raw policy), the floor a cut search never goes below.
    assert searcher.pool.stats(sid).simulations < 4096
    legal = {int(searcher.action_space.index(game, a)) for a in game.playable_actions}
    assert int(searcher.pool.select(sid, 0.0)) in legal


def test_stop_at_cuts_the_search_and_the_choice_is_legal():
    searcher = mcts(sims=4096, dets=1)
    game = position(5)
    searcher.stop_at = time.monotonic() - 1.0
    [choice] = decide(searcher, [game], [0], [True])
    assert searcher.counters["cut"] == 1 and searcher.counters["decisions"] == 1
    assert int(choice) in {int(searcher.action_space.index(game, a)) for a in game.playable_actions}
    assert searcher.search_stats["simulations"] < 4096


def test_a_deadline_long_enough_cuts_nothing():
    searcher = mcts(sims=16, dets=1, deadline=600.0)
    decide(searcher, [position(5)], [0], [True])
    assert searcher.counters["cut"] == 0 and searcher.search_stats["simulations"] >= 16


def test_a_cut_search_is_re_rooted_like_any_other():
    """Cut after a little search (a cut at zero simulations leaves the chosen child
    unvisited, which the engine refuses to re-root, and the next decision searches fresh)."""
    searcher = mcts(sims=1 << 30, dets=1, reuse=True, deadline=0.3)
    found = False
    for seed in range(40):
        game = position(seed)
        [choice] = decide(searcher, [game], [0], [True])
        action = searcher.action_space.decode(game, int(choice))
        nxt = game.copy()
        nxt.execute(action)
        if nxt.current_player_index != game.current_player_index or len(nxt.playable_actions) < 2 \
                or nxt.is_discarding:
            continue
        before = searcher.counters["reused"]
        decide(searcher, [nxt], [0], [False])
        if searcher.counters["reused"] > before:
            found = True
            break
    assert found and searcher.counters["cut"] == searcher.counters["decisions"] >= 2
    assert searcher.pool.pending == 0
