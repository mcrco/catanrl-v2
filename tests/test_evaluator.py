"""The batched evaluator: CUDA graphs agree with eager, strict rows do not
depend on their batch, and fast mode pads to powers of two."""

from __future__ import annotations

import cppanatron as cpp
import numpy as np
import pytest
import torch
from eval_standins import tiny_model

from catanrl import games
from catanrl.search import BatchEvaluator

MODEL = tiny_model(["base-2p"], level="full", seed=2, preset="graph-s")


def rows(count=24, seed=0):
    config = games.load_game("base-2p")
    encoder = cpp.GraphEncoder(cpp.EncoderDims(**MODEL.dims), cpp.InfoLevel.FULL)
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        game = config.cppanatron_game(map_seed=i, seed=i)
        for _ in range(int(rng.integers(0, 200))):
            if game.winning_color is not None:
                break
            actions = game.playable_actions
            game.execute(actions[int(rng.integers(len(actions)))])
        out.append(encoder.encode(game, game.current_player_index))
    return np.stack(out)


ROWS = rows()


def test_fast_buckets_are_powers_of_two():
    evaluator = BatchEvaluator(MODEL, device="cpu", infer_rows=16, mode="fast")
    assert [evaluator.bucket(n) for n in (1, 2, 3, 5, 9, 16, 40)] == [1, 2, 4, 8, 16, 16, 16]
    assert BatchEvaluator(MODEL, device="cpu", infer_rows=16, mode="strict").bucket(3) == 1


def test_values_are_probabilities_and_logits_masked():
    logits, values = BatchEvaluator(MODEL, device="cpu", infer_rows=8, mode="fast")(ROWS)
    mask = ROWS[:, -MODEL.num_actions:]
    assert np.all((logits > -1e8) == (mask > 0))
    assert np.all((values > 0) & (values < 1))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("mode", ["fast", "strict"])
def test_cuda_graphs_match_eager(mode):
    graphed = BatchEvaluator(MODEL, device="cuda", infer_rows=16, mode=mode, cuda_graphs=True)
    eager = BatchEvaluator(MODEL, device="cuda", infer_rows=16, mode=mode, cuda_graphs=False)
    for n in (1, 5, 16, 24):
        g_logits, g_values = graphed(ROWS[:n])
        e_logits, e_values = eager(ROWS[:n])
        assert np.allclose(g_logits, e_logits, atol=1e-4) and np.allclose(g_values, e_values, atol=1e-5)
        assert np.array_equal(g_logits.argmax(1), e_logits.argmax(1))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_strict_cuda_rows_do_not_depend_on_their_batch():
    evaluator = BatchEvaluator(MODEL, device="cuda", infer_rows=8, mode="strict")
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        together = evaluator(ROWS)
        shuffled = np.random.default_rng(1).permutation(len(ROWS))
        apart = evaluator(ROWS[shuffled])
        for i in range(len(ROWS)):
            alone = evaluator(ROWS[i : i + 1])
            assert np.array_equal(alone[0][0], together[0][i]) and np.array_equal(alone[1][0], together[1][i])
        assert np.array_equal(apart[0], together[0][shuffled])
    finally:
        torch.use_deterministic_algorithms(previous)
