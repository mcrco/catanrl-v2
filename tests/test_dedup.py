"""Test 33: DedupEvaluator returns bitwise what the wrapped evaluator would: repeats
within a batch and across batches get the first evaluation's outputs, so MCTS with
``dedup`` (the default) searches exactly as without it; the golden and invariance tests
run with it on."""

from __future__ import annotations

import cppanatron as cpp
import numpy as np
import pytest
import torch
from eval_helpers import agent, tiny_model
from test_search_invariance import DIMS, MODEL, POSITIONS

from catanrl.search import BatchEvaluator, DedupEvaluator

ENCODER = cpp.GraphEncoder(cpp.EncoderDims(**DIMS), cpp.InfoLevel.FULL)
ROWS = np.stack([ENCODER.encode(game, game.current_player_index) for game, _ in POSITIONS])


def evaluators(model=MODEL, device="cpu", **kwargs):
    plain = BatchEvaluator(model, device=device, infer_rows=4, mode="strict")
    return plain, DedupEvaluator(BatchEvaluator(model, device=device, infer_rows=4, mode="strict"), **kwargs)


def same(a, b):
    return np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_repeats_copy_the_first_occurrence_within_and_across_batches():
    plain, dedup = evaluators()
    batch = ROWS[[0, 1, 0, 2, 1, 0, 3]]
    assert same(dedup(batch), plain(batch))
    assert dedup.counters == {"rows": 7, "forwarded": 4, "batch_hits": 3}
    assert dedup.evaluator.forwards == 4  # strict on CPU: one forward per distinct row
    again = ROWS[[3, 4, 0]]
    assert same(dedup(again), plain(again))
    assert dedup.counters == {"rows": 10, "forwarded": 5, "batch_hits": 3, "cache_hits": 2}
    assert same(dedup(ROWS[:0]), plain(ROWS[:0]))


def test_hashing_runs_only_where_a_merge_can_save_a_forward(monkeypatch):
    class Stub:  # a strict CUDA evaluator's shape, without CUDA
        obs_size, num_actions, num_players, infer_rows, recurrent = 4, 3, 2, 64, False
        device, mode = torch.device("cuda"), "strict"

    wrapper = DedupEvaluator(Stub())
    assert not wrapper.active(64) and wrapper.active(65)
    Stub.mode = "fast"
    assert wrapper.active(1)
    Stub.mode, Stub.device = "strict", torch.device("cpu")
    assert wrapper.active(1)
    Stub.device = torch.device("cuda")
    assert not DedupEvaluator(Stub()).active(1)
    monkeypatch.setenv("CATANRL_DEDUP_FORCE", "1")
    assert DedupEvaluator(Stub()).active(1)


def test_the_cache_is_bounded_and_can_be_off():
    _, dedup = evaluators(cache_rows=2)
    dedup(ROWS[[0, 1, 2]])  # row 0 is the oldest and goes
    dedup(ROWS[[0]])
    assert dedup.counters["forwarded"] == 4 and dedup.counters["cache_hits"] == 0
    dedup(ROWS[[0]])
    assert dedup.counters["cache_hits"] == 1
    _, off = evaluators(cache_rows=0)
    off(ROWS[[0]])
    off(ROWS[[0]])
    assert off.counters["forwarded"] == 2 and off.counters["cache_hits"] == 0
    dedup.clear()
    dedup(ROWS[[0]])
    assert dedup.counters["forwarded"] == 5


def test_a_recurrent_network_is_refused():
    class Recurrent:
        recurrent = True

    with pytest.raises(ValueError, match="recurrent"):
        DedupEvaluator(Recurrent())


def test_mcts_searches_the_same_with_dedup_on_and_off():
    """Choices and visits of the live searches (reuse keeps them inspectable), dedup on vs off."""
    on = agent("MCTS", MODEL, infer_rows=8, sims=12, dets=2, reuse=True)
    off = agent("MCTS", MODEL, infer_rows=8, sims=12, dets=2, reuse=True, dedup=False)
    assert on.dedup and isinstance(on.evaluator, DedupEvaluator)
    assert not off.dedup and isinstance(off.evaluator, BatchEvaluator)
    games = [game for game, _ in POSITIONS[:4]]
    trackers = [tracker for _, tracker in POSITIONS[:4]]
    rows = np.stack([on.encode(g, g.current_player_index) for g in games])
    masks, seeds = rows[:, -on.action_space.size:], np.arange(4, dtype=np.uint64)
    streams, first = np.arange(4), np.ones(4, dtype=bool)
    choices = [a.act(rows, masks, seeds, games=games, trackers=trackers, streams=streams, first=first)
               for a in (on, off)]
    assert np.array_equal(choices[0], choices[1])
    for stream in range(4):
        assert on.pool.visits(on._live[stream][0]).tolist() == off.pool.visits(off._live[stream][0]).tolist()
    assert on.evaluator.counters["rows"] == off.evaluator.rows > 0


@pytest.mark.gpu
def test_forced_on_cuda_is_bitwise(monkeypatch):
    monkeypatch.setenv("CATANRL_DEDUP_FORCE", "1")
    model = tiny_model(["base-3p"], level="full", seed=5)
    plain, dedup = evaluators(model, device="cuda")
    batch = ROWS[[0, 1, 0, 2, 1, 0, 3]]
    assert same(dedup(batch), plain(batch))
    assert dedup.counters["forwarded"] == 4 and dedup.counters["batch_hits"] == 3
