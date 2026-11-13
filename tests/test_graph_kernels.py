"""kernels=fused: the shared-trunk graph network on custom CUDA kernels gives the
module's outputs (to the last bits) on real rows, padded dims included; its strict
rows do not depend on their batch; a board the neighbour lists cannot hold raises; and
the evaluator refuses what the kernels do not run. kernels=fused16: the same forward
with half GEMMs stays within rounding of fused, masks and finiteness intact, and a half
past its range sends the batch to the float32 kernels."""

from __future__ import annotations

import gc
import shutil

import cppanatron as cpp
import numpy as np
import pytest
import torch
from catanatron.registry import SpecError

from catanrl import games
from catanrl.agents import check_spec_syntax
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
@pytest.mark.parametrize("kernels", ["fused", "fused16"])
def test_strict_fused_rows_do_not_depend_on_their_batch(kernels):
    model = shared_model(["base-2p"])
    obs = rows(model, "base-2p", count=24)
    evaluator = BatchEvaluator(model, device="cuda", infer_rows=8, mode="strict", kernels=kernels)
    together = evaluator(obs)
    shuffled = np.random.default_rng(1).permutation(len(obs))
    apart = evaluator(obs[shuffled])
    for i in range(len(obs)):
        alone = evaluator(obs[i: i + 1])
        assert np.array_equal(alone[0][0], together[0][i]) and np.array_equal(alone[1][0], together[1][i])
    assert np.array_equal(apart[0], together[0][shuffled]) and np.array_equal(apart[1], together[1][shuffled])


@pytest.mark.gpu
@needs_kernels
@pytest.mark.parametrize("kernels", ["fused", "fused16"])
def test_a_node_with_four_edges_is_refused(kernels):
    model = shared_model(["base-2p"])
    obs = rows(model, "base-2p", count=2)
    offset, _ = model.layout["edge_nodes"]
    for e in range(4):  # node 0 on four edges
        obs[1, offset + 2 * e: offset + 2 * e + 2] = (0, 10 + e)
    evaluator = BatchEvaluator(model, device="cuda", infer_rows=8, mode="strict", kernels=kernels)
    evaluator(obs[:1])
    with pytest.raises(RuntimeError, match="hex boards only"):
        evaluator(obs)


def test_the_evaluator_refuses_what_the_kernels_do_not_run():
    model = shared_model(["base-2p"])
    with pytest.raises(ValueError, match="CUDA"):
        BatchEvaluator(model, device="cpu", infer_rows=8, kernels="fused")
    with pytest.raises(ValueError, match="kernels=fused16 .* CUDA"):
        BatchEvaluator(model, device="cpu", infer_rows=8, kernels="fused16")
    with pytest.raises(ValueError, match="kernels must be"):
        BatchEvaluator(model, device="cpu", infer_rows=8, kernels="triton")
    with pytest.raises(SpecError, match="not one of torch, fused"):  # self-play's: no agent searches with it
        check_spec_syntax("MCTS:ckpt=exp/v@best:kernels=fused16")
    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="CUDA"):
            BatchEvaluator(model, device="cuda", infer_rows=8, kernels="fused", heads="policy")
        separate = shared_model(["base-2p"], shared=False)
        with pytest.raises(ValueError, match="shared-trunk"):
            BatchEvaluator(separate, device="cuda", infer_rows=8, kernels="fused")
        with pytest.raises(ValueError, match="CUDA"):
            BatchEvaluator(model, device="cuda", infer_rows=8, kernels="fused16", heads="policy")
        with pytest.raises(ValueError, match="shared-trunk"):
            BatchEvaluator(separate, device="cuda", infer_rows=8, kernels="fused16")


# ---------------------------------------------------------------- kernels=fused16


@pytest.mark.gpu
@needs_kernels
@pytest.mark.parametrize("scenarios,scenario", [
    (["base-2p"], "base-2p"),
    (["base-2p", "random-4p"], "base-2p"),
    (["base-2p", "random-4p"], "random-4p"),
    (["base-4p", "extension-6p"], "extension-6p"),
], ids=["2p", "2p-padded", "random-4p", "6p"])
@pytest.mark.parametrize("mode", ["strict", "fast"])
def test_half_kernels_stay_within_rounding_of_the_float_ones(scenarios, scenario, mode):
    """float32 outputs in fused's conventions: the same masks to the bit, everything finite, and
    the rest within half's rounding in the trunk (these rows: logits within 1e-3 of their scale,
    values within 5e-4, KL up to 1.5e-5; the bounds leave a factor of four or more)."""
    model = shared_model(scenarios)
    obs = rows(model, scenario)
    want = BatchEvaluator(model, device="cuda", infer_rows=16, mode=mode, kernels="fused")(obs)
    evaluator = BatchEvaluator(model, device="cuda", infer_rows=16, mode=mode, kernels="fused16")
    got = evaluator(obs)
    mask = obs[:, model.mask_offset: model.mask_offset + model.num_actions] > 0
    assert got[0].dtype == np.float32 and got[1].dtype == np.float32 and evaluator.exact_forwards == 0
    assert np.array_equal(got[0] > -1e8, mask) and np.all(got[0][~mask] == want[0][~mask])
    assert np.isfinite(got[0]).all() and np.isfinite(got[1]).all()
    scale = max(1.0, float(np.abs(want[0][mask]).max()))
    assert 0 < np.abs(got[0][mask] - want[0][mask]).max() <= 4e-3 * scale
    assert np.abs(got[1] - want[1]).max() <= 2e-3
    log_want, log_got = (torch.log_softmax(torch.from_numpy(x[0]).double(), 1) for x in (want, got))
    assert float((log_want.exp() * (log_want - log_got)).sum(1).max()) <= 1e-4
    # the best action: fused's, or one fused holds within rounding of it
    best = np.take_along_axis(want[0], got[0].argmax(1)[:, None], 1)[:, 0]
    assert np.all(best >= want[0].max(1) - 4e-3 * scale)
    # exact(): the float32 kernels on the same copies of the weights
    if mode == "strict":
        exact = evaluator.exact(obs)
        assert np.array_equal(exact[0], want[0]) and np.array_equal(exact[1], want[1])


@pytest.mark.gpu
@needs_kernels
@pytest.mark.parametrize("name,factor", [
    ("backbone.layers.1.node_self.weight", 1e4),   # updates past 65504: inf from the GEMM
    ("backbone.layers.0.tile_self.weight", 1e6),   # weights that are inf as halves: nan updates, 0 after a relu
    ("backbone.layers.2.edge_from_node.weight", 1e3),
    ("policy_head.settlement.0.weight", 1e4),
    ("policy_head.road.0.weight", 1e4),
    ("policy_head.road.2.weight", 1e3),
    ("policy_head.robber_tile.weight", 1e3),
], ids=["update", "weights", "edges", "settlement", "road-1", "road-2", "robber"])
@pytest.mark.parametrize("mode", ["strict", "fast"])
def test_a_half_past_its_range_is_evaluated_in_float(name, factor, mode):
    """Weights no half forward can carry: the kernels flag it (bit 2), whatever their outputs
    look like, and the evaluator returns the float32 kernels' outputs for the batch."""
    model = shared_model(["base-2p"])
    with torch.no_grad():
        dict(model.named_parameters())[name].mul_(factor)
    obs = rows(model, "base-2p", count=24)
    want = BatchEvaluator(model, device="cuda", infer_rows=8, mode=mode, kernels="fused")(obs)
    evaluator = BatchEvaluator(model, device="cuda", infer_rows=8, mode=mode, kernels="fused16")
    got = evaluator(obs)
    assert evaluator.exact_forwards == 3
    assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])
    mask = obs[:, model.mask_offset: model.mask_offset + model.num_actions] > 0
    assert np.isfinite(got[0][mask]).all() and np.isfinite(got[1]).all()
    with torch.inference_mode():
        assert int(evaluator.fused.forward(torch.from_numpy(obs).cuda())[2][0]) == 2


@pytest.mark.gpu
@needs_kernels
def test_half_kernels_convert_the_weights_of_their_moment():
    """As fused: built from copies of the weights, so a change shows only in a new evaluator,
    and exact() (the float32 kernels on those copies) is what tells a stale one from rounding."""
    model = shared_model(["base-2p"])
    obs = rows(model, "base-2p", count=8)
    evaluator = BatchEvaluator(model, device="cuda", infer_rows=8, kernels="fused16")
    before, exact = evaluator(obs), evaluator.exact(obs)
    with torch.no_grad():
        dict(model.named_parameters())["backbone.layers.0.node_self.weight"].mul_(1.1)
    after = evaluator(obs)
    assert np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1])
    assert np.array_equal(evaluator.exact(obs)[1], exact[1])
    module = BatchEvaluator(model, device="cuda", infer_rows=8)(obs)
    assert np.abs(exact[1] - module[1]).max() > 1e-3
    fresh = BatchEvaluator(model, device="cuda", infer_rows=8, kernels="fused16")
    assert np.abs(fresh.exact(obs)[1] - module[1]).max() <= 1e-5 and np.abs(fresh(obs)[1] - module[1]).max() <= 2e-3


@pytest.mark.gpu
@needs_kernels
@pytest.mark.parametrize("first,second", [("fused16", "fused16"), ("fused16", "fused"), ("fused", "fused16")])
def test_half_graphs_do_not_share_a_workspace_with_other_graphs(first, second):
    """cuBLAS's workspace of the capture stream belongs to the first graph captured and goes
    with it: an evaluator captured later must still replay once that one is gone and the cache
    emptied (BatchEvaluator._own_workspace; an illegal memory access otherwise)."""
    model = shared_model(["base-2p"])
    obs = rows(model, "base-2p", count=64)
    earlier = BatchEvaluator(model, device="cuda", infer_rows=64, kernels=first)
    earlier(obs)
    later = BatchEvaluator(model, device="cuda", infer_rows=64, kernels=second)
    want = later(obs)
    del earlier
    gc.collect()
    torch.cuda.empty_cache()
    junk = torch.full((1 << 26,), float("nan"), device="cuda")  # 256 MiB over what was freed
    del junk
    for _ in range(2):
        got = later(obs)
        assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])
