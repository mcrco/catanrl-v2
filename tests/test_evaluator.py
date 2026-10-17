"""The batched evaluator: CUDA graphs agree with eager (and, emulated on CPU,
replay what the static inputs hold), strict rows do not depend on their
batch, and fast mode pads to powers of two."""

from __future__ import annotations

import cppanatron as cpp
import numpy as np
import pytest
import torch
from eval_helpers import tiny_model

from catanrl import games
from catanrl.search import BatchEvaluator

MODEL = tiny_model(["base-2p"], level="full", seed=2, preset="graph-s")
LSTM = tiny_model(["base-2p"], level="full", seed=2, preset="lstm-tiny")


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


def memories(n: int, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return tuple(rng.standard_normal((n, LSTM.hidden_size), dtype=np.float32) for _ in range(2))


def check_graphs_match_eager(graphed: BatchEvaluator, eager: BatchEvaluator) -> None:
    """Padded and chunked batches match; a recurrent step threads the memory; and a
    stateless call after a step is the zero-state function, not the last step's memory."""
    for n in (1, 5, 8, 24):
        g_logits, g_values = graphed(ROWS[:n])
        e_logits, e_values = eager(ROWS[:n])
        assert np.allclose(g_logits, e_logits, atol=1e-4) and np.allclose(g_values, e_values, atol=1e-5)
        assert np.array_equal(g_logits.argmax(1), e_logits.argmax(1))
    if not graphed.recurrent:
        return
    h, c = memories(24)
    for n in (5, 24):
        for g, e in zip(graphed.step(ROWS[:n], h[:n], c[:n]), eager.step(ROWS[:n], h[:n], c[:n])):
            assert np.allclose(g, e, atol=1e-4)
    stateless = graphed(ROWS[:5])
    zero = eager.step(ROWS[:5], np.zeros_like(h[:5]), np.zeros_like(c[:5]))
    assert np.allclose(stateless[0], zero[0], atol=1e-4) and np.allclose(stateless[1], zero[1], atol=1e-5)


def emulate_graphs(evaluator: BatchEvaluator, monkeypatch) -> BatchEvaluator:
    """Run the CUDA-graph path on CPU: a captured graph is a replay that recomputes
    the static outputs from whatever the static inputs hold at the time."""

    def capture(forward):
        outputs = tuple(out.clone() for out in forward())

        class Graph:
            def replay(self):
                for target, out in zip(outputs, forward()):
                    target.copy_(out)

        return Graph(), outputs

    monkeypatch.setattr(evaluator, "_capture", capture)
    evaluator.use_graphs = True
    return evaluator


@pytest.mark.parametrize("model", [MODEL, LSTM], ids=["graph", "lstm"])
def test_graph_replays_read_their_static_inputs(model, monkeypatch):
    """The graph path emulated on CPU (fast mode, so the buckets pad and chunk)."""
    graphed = emulate_graphs(BatchEvaluator(model, device="cpu", infer_rows=8, mode="fast"), monkeypatch)
    eager = BatchEvaluator(model, device="cpu", infer_rows=8, mode="fast", cuda_graphs=False)
    check_graphs_match_eager(graphed, eager)
    assert sorted(graphed._graphs) == [1, 8]
    assert len(graphed._graphs[8][1]) == (3 if model.recurrent else 1)  # rows (, h, c) static inputs


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("model", [MODEL, LSTM], ids=["graph", "lstm"])
@pytest.mark.parametrize("mode", ["fast", "strict"])
def test_cuda_graphs_match_eager(model, mode):
    graphed = BatchEvaluator(model, device="cuda", infer_rows=16, mode=mode, cuda_graphs=True)
    eager = BatchEvaluator(model, device="cuda", infer_rows=16, mode=mode, cuda_graphs=False)
    check_graphs_match_eager(graphed, eager)


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
