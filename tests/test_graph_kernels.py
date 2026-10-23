"""kernels=fused: the shared-trunk graph network on custom CUDA kernels gives the
module's outputs (to the last bits) on real rows, padded dims included; its strict
rows do not depend on their batch; a board the neighbour lists cannot hold raises; and
the evaluator refuses what the kernels do not run."""

from __future__ import annotations

import shutil

import cppanatron as cpp
import numpy as np
import pytest
import torch

from catanrl import games
from catanrl.envs.encoding import action_size, dims_for, row_layout
from catanrl.models import build_model
from catanrl.search import BatchEvaluator

GRAPH_S = {"hidden": 96, "global_hidden": 64, "layers": 3, "head_hidden": 128}
needs_kernels = pytest.mark.skipif(not torch.cuda.is_available() or shutil.which("nvcc") is None,
                                   reason="needs CUDA and nvcc (the kernels build on first use)")


def shared_model(scenarios, level="private", seed=0, **args):
    torch.manual_seed(seed)
    dims = dims_for(list(scenarios))
    model = build_model({"class": "graph", "args": {**GRAPH_S, "shared": True, **args}}, dims=dims,
                        levels={"level": level}, num_actions=action_size(dims), layout=row_layout(dims, level))
    # Trained-scale weights: random init leaves the logits near zero, where errors hide.
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(3.0)
    return model


def rows(model, scenario, count=48, seed=0, level=cpp.InfoLevel.PRIVATE):
    config = games.load_game(scenario)
    encoder = cpp.GraphEncoder(cpp.EncoderDims(**model.dims), level)
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        game = config.cppanatron_game(map_seed=i, seed=i)
        for _ in range(int(rng.integers(0, 300))):
            if game.winning_color is not None:
                break
            actions = game.playable_actions
            game.execute(actions[int(rng.integers(len(actions)))])
        out.append(encoder.encode(game, game.current_player_index))
    return np.stack(out)


def assert_close(model, got, want, obs):
    mask = obs[:, model.mask_offset: model.mask_offset + model.num_actions] > 0
    assert np.array_equal(got[0] > -1e8, mask) and np.all(got[0][~mask] == want[0][~mask])
    scale = max(1.0, float(np.abs(want[0][mask]).max()))
    assert np.abs(got[0][mask] - want[0][mask]).max() <= 2e-6 * scale + 1e-5
    assert np.abs(got[1] - want[1]).max() <= 1e-5


@pytest.mark.gpu
@needs_kernels
@pytest.mark.parametrize("scenarios,scenario", [
    (["base-2p"], "base-2p"),
    (["base-2p", "random-4p"], "base-2p"),     # a small board padded to larger dims
    (["base-2p", "random-4p"], "random-4p"),
    (["base-4p", "extension-6p"], "extension-6p"),
], ids=["2p", "2p-padded", "random-4p", "6p"])
@pytest.mark.parametrize("mode", ["strict", "fast"])
def test_fused_kernels_match_the_module(scenarios, scenario, mode):
    model = shared_model(scenarios)
    obs = rows(model, scenario)
    want = BatchEvaluator(model, device="cuda", infer_rows=16, mode=mode)(obs)
    got = BatchEvaluator(model, device="cuda", infer_rows=16, mode=mode, kernels="fused")(obs)
    assert_close(model, got, want, obs)


@pytest.mark.gpu
@needs_kernels
def test_strict_fused_rows_do_not_depend_on_their_batch():
    model = shared_model(["base-2p"])
    obs = rows(model, "base-2p", count=24)
    evaluator = BatchEvaluator(model, device="cuda", infer_rows=8, mode="strict", kernels="fused")
    together = evaluator(obs)
    shuffled = np.random.default_rng(1).permutation(len(obs))
    apart = evaluator(obs[shuffled])
    for i in range(len(obs)):
        alone = evaluator(obs[i: i + 1])
        assert np.array_equal(alone[0][0], together[0][i]) and np.array_equal(alone[1][0], together[1][i])
    assert np.array_equal(apart[0], together[0][shuffled]) and np.array_equal(apart[1], together[1][shuffled])


@pytest.mark.gpu
@needs_kernels
def test_a_node_with_four_edges_is_refused():
    model = shared_model(["base-2p"])
    obs = rows(model, "base-2p", count=2)
    offset, _ = model.layout["edge_nodes"]
    for e in range(4):  # node 0 on four edges
        obs[1, offset + 2 * e: offset + 2 * e + 2] = (0, 10 + e)
    evaluator = BatchEvaluator(model, device="cuda", infer_rows=8, mode="strict", kernels="fused")
    evaluator(obs[:1])
    with pytest.raises(RuntimeError, match="hex boards only"):
        evaluator(obs)


def test_the_evaluator_refuses_what_the_kernels_do_not_run():
    model = shared_model(["base-2p"])
    with pytest.raises(ValueError, match="CUDA"):
        BatchEvaluator(model, device="cpu", infer_rows=8, kernels="fused")
    with pytest.raises(ValueError, match="kernels must be"):
        BatchEvaluator(model, device="cpu", infer_rows=8, kernels="triton")
    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="CUDA"):
            BatchEvaluator(model, device="cuda", infer_rows=8, kernels="fused", heads="policy")
        separate = shared_model(["base-2p"], shared=False)
        with pytest.raises(ValueError, match="shared-trunk"):
            BatchEvaluator(separate, device="cuda", infer_rows=8, kernels="fused")
