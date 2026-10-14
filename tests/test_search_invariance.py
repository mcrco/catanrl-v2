"""Test 22: SearchPool visit counts do not depend on the pool size, the leaf
batch size or the threads (strict evaluator; one machine)."""

from __future__ import annotations

import cppanatron as cpp
import numpy as np
import pytest
from eval_standins import tiny_model

from catanrl import games
from catanrl.search import BatchEvaluator, make_pool, run_searches, search_config

MODEL = tiny_model(["base-3p"], level="full", seed=5)
DIMS = MODEL.dims


def positions(count=9, seed=0):
    """Games mid-play with every seat's hand tracker fed from the start."""
    config = games.load_game("base-3p")
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        game = config.cppanatron_game(map_seed=i, seed=100 + i)
        trackers = [cpp.HandTracker(game, seat) for seat in range(3)]
        for _ in range(int(rng.integers(20, 160))):
            actions = game.playable_actions
            if game.winning_color is not None or not actions:
                break
            action = actions[int(rng.integers(len(actions)))]
            before = game.copy()
            game.execute(action)
            for tracker in trackers:
                tracker.observe(before, action, game)
        if game.winning_color is None:
            out.append((game, trackers[game.current_player_index]))
    return out


POSITIONS = positions()


def visits(params, *, pool_size, batch_rows, threads, infer_rows=8):
    evaluator = BatchEvaluator(MODEL, device="cpu", infer_rows=infer_rows, mode="strict")
    config = search_config(params, max_players=DIMS["max_players"], threads=threads)
    result = []
    for start in range(0, len(POSITIONS), pool_size):
        pool = make_pool(DIMS, "full", config)
        group = POSITIONS[start : start + pool_size]
        ids = [pool.add(game, seed=1000 + start + i, tracker=tracker if config.determinizations else None)
               for i, (game, tracker) in enumerate(group)]
        run_searches(pool, evaluator, batch_rows=batch_rows)
        result += [pool.visits(i).tolist() for i in ids]
    return result


@pytest.mark.parametrize("params", [
    {"sims": 24, "oracle": True},
    {"sims": 24, "dets": 3},
    {"sims": 16, "oracle": True, "root": "gumbel", "gumbel_m": 4},
])
def test_visits_are_invariant(params):
    reference = visits(params, pool_size=len(POSITIONS), batch_rows=64, threads=1)
    assert all(sum(v) > 0 for v in reference)
    for pool_size, batch_rows, threads in ((1, 1, 1), (4, 3, 4), (len(POSITIONS), 5, 2)):
        assert visits(params, pool_size=pool_size, batch_rows=batch_rows, threads=threads) == reference


def test_strict_rows_do_not_depend_on_their_batch():
    evaluator = BatchEvaluator(MODEL, device="cpu", infer_rows=4, mode="strict")
    encoder = cpp.GraphEncoder(cpp.EncoderDims(**DIMS), cpp.InfoLevel.FULL)
    rows = np.stack([encoder.encode(game, game.current_player_index) for game, _ in POSITIONS])
    together = evaluator(rows)
    for i in range(len(rows)):
        alone = evaluator(rows[i : i + 1])
        assert np.array_equal(alone[0][0], together[0][i]) and np.array_equal(alone[1][0], together[1][i])


def test_pools_in_lockstep_search_like_pools_alone():
    """run_pools batches several pools' leaves together; each search is unchanged."""
    from catanrl.search import run_pools

    evaluator = BatchEvaluator(MODEL, device="cpu", infer_rows=8, mode="strict")
    budgets = (16, 4)
    alone, together = [], []
    for sims in budgets:
        pool = make_pool(DIMS, "full", search_config({"sims": sims, "oracle": True}, max_players=3))
        ids = [pool.add(game, seed=50 + i) for i, (game, _) in enumerate(POSITIONS)]
        run_searches(pool, evaluator, batch_rows=len(ids))
        alone.append([pool.visits(i).tolist() for i in ids])
    pools = [make_pool(DIMS, "full", search_config({"sims": sims, "oracle": True}, max_players=3)) for sims in budgets]
    ids = [[pool.add(game, seed=50 + i) for i, (game, _) in enumerate(POSITIONS)] for pool in pools]
    run_pools(pools, evaluator, batch_rows=[len(i) for i in ids])
    together = [[pool.visits(i).tolist() for i in pool_ids] for pool, pool_ids in zip(pools, ids)]
    assert together == alone
