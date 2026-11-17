"""kernels=fused for the seat-set network (model class ``seat``, models/seat_kernels.py): the
shared-trunk network on the graph kernels plus the seats' own gives the module's outputs (to
the last bits) on rows of two, three and four seats, with and without the dice input, on
BASE and hexagon boards, with the attention and without it; its strict rows do not depend on
their batch; what an absent slot holds changes nothing; a board the neighbour lists cannot
hold raises; and what the kernels do not run is refused. kernels=fused16: the same forward
with the board's GEMMs in half precision stays within rounding of fused, masks and finiteness
intact, and a half past its range sends the batch to the float32 kernels. With ``-m "gpu and
runs"``: the trained network (gen-base-2/main@e3800) on rows of games it plays itself."""

from __future__ import annotations

import gc
import shutil

import numpy as np
import pytest
import torch
from test_seat_model import Columns

from catanrl import games
from catanrl.envs.encoding import action_size, dims_for, make_encoder, row_layout
from catanrl.models import build_model
from catanrl.models.graph import GraphValueHead
from catanrl.models.graph_kernels import FusedGraph, FusedGraph16
from catanrl.search import BatchEvaluator

#: What gen-base-2 trains on, by seats: every table size in rows of four slots.
MIXTURE = ["base-2p-fr", "base-3p", "base-4p", "hexagon-3p"]
needs_kernels = pytest.mark.skipif(not torch.cuda.is_available() or shutil.which("nvcc") is None,
                                   reason="needs CUDA and nvcc (the kernels build on first use)")


def shared_model(scenarios=MIXTURE, level="private", dice=True, seed=0, preset="graph-seat-s", **args):
    torch.manual_seed(seed)
    dims = dims_for(list(scenarios))
    levels = {"level": level, **({"dice_state": True} if dice else {})}
    model = build_model({"preset": preset, "args": {"shared": True, **args}}, dims=dims, levels=levels,
                        num_actions=action_size(dims), layout=row_layout(dims, level, None, dice))
    # Trained-scale weights: random init leaves the logits near zero, where errors hide, and
    # the dice input's weights and the attention's turn-order bias at zero. The value head's
    # output layer the other way: at the trunk's scale its sigmoids saturate, hiding as much.
    generator = torch.Generator().manual_seed(seed + 100)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(3.0)
            if not p.abs().sum():
                p.add_(torch.randn(p.shape, generator=generator) * 0.3)
        for module in model.modules():
            if isinstance(module, GraphValueHead):
                module.mlp[-1].weight.mul_(0.1)
    return model


def encoder_of(model):
    return make_encoder(model.dims, model.levels["level"], bool(model.levels.get("dice_state")))


def rows(model, scenario, count=48, seed=0):
    config = games.load_game(scenario)
    encoder = encoder_of(model)
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
    return np.stack(out).astype(np.float32)


def mixed_rows(model, count=12):
    """Rows of two, three and four seats on both boards, interleaved (a mixture's batch)."""
    by_scenario = [rows(model, scenario, count=count, seed=i) for i, scenario in enumerate(MIXTURE)]
    return np.stack(by_scenario, axis=1).reshape(-1, by_scenario[0].shape[1])


def assert_close(model, got, want, obs):
    mask = obs[:, model.mask_offset: model.mask_offset + model.num_actions] > 0
    assert np.array_equal(got[0] > -1e8, mask) and np.all(got[0][~mask] == want[0][~mask])
    scale = max(1.0, float(np.abs(want[0][mask]).max()))
    assert np.abs(got[0][mask] - want[0][mask]).max() <= 2e-6 * scale + 1e-5
    assert np.abs(got[1] - want[1]).max() <= 1e-5


#: (the model's scenarios, the rows' scenario or None for the mixture's, dice input, model args)
CASES = {
    "2-seats": (MIXTURE, "base-2p-fr", True, {}),
    "3-seats": (MIXTURE, "base-3p", True, {}),
    "4-seats": (MIXTURE, "base-4p", True, {}),
    "hexagon": (MIXTURE, "hexagon-3p", True, {}),
    "mixed": (MIXTURE, None, True, {}),
    "2-seats-no-dice": (MIXTURE, "base-2p-fr", False, {}),
    "4-seats-no-dice": (MIXTURE, "base-4p", False, {}),
    "hexagon-no-dice": (MIXTURE, "hexagon-3p", False, {}),
    "2-slots": (["base-2p-fr"], "base-2p-fr", True, {}),            # a 2p-only run's rows
    "6-seats": (["base-4p", "extension-6p"], "extension-6p", True, {}),
    "mean-mixing": (MIXTURE, None, True, {"attention": False}),
    "mean-mixing-no-dice": (MIXTURE, None, False, {"attention": False}),
    "outcome-head": (MIXTURE, None, True, {"outcome": True}),      # training only: left out
    "wide-seats": (MIXTURE, None, True, {"seat_hidden": 80, "relation_hidden": 6, "attention_hidden": 32,
                                         "attention_heads": 2}),
}


def case(name, level="private"):
    scenarios, scenario, dice, args = CASES[name]
    model = shared_model(scenarios, level, dice, seed=len(name), **args)
    return model, (mixed_rows(model) if scenario is None else rows(model, scenario))


@pytest.mark.gpu
@needs_kernels
@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("mode", ["strict", "fast"])
def test_fused_kernels_match_the_module(name, mode):
    model, obs = case(name)
    seats = obs[:, model.layout["players"][0]:][:, : model.num_players * model.layout["players"][1][1]]
    present = seats.reshape(len(obs), model.num_players, -1)[..., model.backbone.present].sum(1)
    wanted = {"2-seats": {2}, "3-seats": {3}, "4-seats": {4}, "hexagon": {3}, "mixed": {2, 3, 4}, "6-seats": {6}}
    assert set(present.astype(int)) == wanted.get(name.removesuffix("-no-dice"), set(present.astype(int)))
    want = BatchEvaluator(model, device="cuda", infer_rows=16, mode=mode)(obs)
    got = BatchEvaluator(model, device="cuda", infer_rows=16, mode=mode, kernels="fused")(obs)
    assert_close(model, got, want, obs)
    held = want[1][want[1] != 0]  # the seats present: values that could show an error
    assert len(held) == present.sum() and held.min() > 0.02 and held.max() < 0.98 and held.std() > 0.02


@pytest.mark.gpu
@needs_kernels
def test_fused_kernels_match_the_module_at_the_full_level():
    model = shared_model(["base-2p-fr"], level="full")
    obs = rows(model, "base-2p-fr")
    want = BatchEvaluator(model, device="cuda", infer_rows=16)(obs)
    assert_close(model, BatchEvaluator(model, device="cuda", infer_rows=16, kernels="fused")(obs), want, obs)


@pytest.mark.gpu
@needs_kernels
@pytest.mark.parametrize("kernels", ["fused", "fused16"])
def test_strict_fused_rows_do_not_depend_on_their_batch(kernels):
    model = shared_model()
    obs = mixed_rows(model, count=6)
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
@pytest.mark.parametrize("scenario,seats", [("base-2p-fr", 2), ("base-3p", 3)])
def test_what_an_absent_slot_holds_changes_no_fused_output(kernels, scenario, seats):
    """The module's property (test_seat_model), held to the bit by the kernels: they read an
    absent slot's features, ownership channels and dice features only times its ``present``."""
    model = shared_model()
    obs = rows(model, scenario, count=16)
    columns = Columns(model.dims, "private", None, True)
    noisy = obs.copy()
    rng = np.random.default_rng(6)
    for k in range(seats, model.num_players):  # everything of the slot but its `present` flag
        held = columns.seat(k, present=False)
        noisy[:, held] = rng.random((len(obs), len(held)), dtype=np.float32) * 4.0 - 1.0
    assert (noisy != obs).any(axis=0).sum() > 100 * (model.num_players - seats)
    evaluator = BatchEvaluator(model, device="cuda", infer_rows=16, kernels=kernels)
    want, got = evaluator(obs), evaluator(noisy)
    assert np.array_equal(want[0], got[0]) and np.array_equal(want[1], got[1])


@pytest.mark.gpu
@needs_kernels
@pytest.mark.parametrize("kernels", ["fused", "fused16"])
def test_a_node_with_four_edges_is_refused(kernels):
    model = shared_model()
    obs = rows(model, "base-4p", count=2)
    offset, _ = model.layout["edge_nodes"]
    for e in range(4):  # node 0 on four edges
        obs[1, offset + 2 * e: offset + 2 * e + 2] = (0, 10 + e)
    evaluator = BatchEvaluator(model, device="cuda", infer_rows=8, mode="strict", kernels=kernels)
    evaluator(obs[:1])
    with pytest.raises(RuntimeError, match="hex boards only"):
        evaluator(obs)


def test_the_evaluator_refuses_what_the_seat_kernels_do_not_run():
    model = shared_model(dice=False)
    with pytest.raises(ValueError, match="kernels=fused .* CUDA"):
        BatchEvaluator(model, device="cpu", infer_rows=8, kernels="fused")
    with pytest.raises(ValueError, match="kernels=fused16 .* CUDA"):
        BatchEvaluator(model, device="cpu", infer_rows=8, kernels="fused16")
    for graph in (FusedGraph, FusedGraph16):  # the graph class's own: they name the seats'
        with pytest.raises(ValueError, match="seat_kernels.FusedSeat"):
            graph(model)
    if torch.cuda.is_available():
        for kernels in ("fused", "fused16"):
            with pytest.raises(ValueError, match="CUDA"):
                BatchEvaluator(model, device="cuda", infer_rows=8, kernels=kernels, heads="policy")
            with pytest.raises(ValueError, match="shared-trunk seat-set"):
                BatchEvaluator(shared_model(dice=False, shared=False), device="cuda", infer_rows=8, kernels=kernels)
            with pytest.raises(ValueError, match="without an aux head"):
                BatchEvaluator(shared_model(level="private", dice=False, aux=True), device="cuda", infer_rows=8,
                               kernels=kernels)
            if shutil.which("nvcc") is not None:  # the widths are the built extensions'
                with pytest.raises(ValueError, match="built for hidden 96, not 16"):
                    BatchEvaluator(shared_model(dice=False, preset="graph-seat-tiny"), device="cuda", infer_rows=8,
                                   kernels=kernels)
                with pytest.raises(ValueError, match="seat_hidden up to 256, not 288"):
                    BatchEvaluator(shared_model(dice=False, seat_hidden=288), device="cuda", infer_rows=8,
                                   kernels=kernels)


# ---------------------------------------------------------------- kernels=fused16


@pytest.mark.gpu
@needs_kernels
@pytest.mark.parametrize("name", ["2-seats", "4-seats", "hexagon", "mixed", "2-seats-no-dice", "2-slots", "6-seats",
                                  "mean-mixing"])
@pytest.mark.parametrize("mode", ["strict", "fast"])
def test_half_kernels_stay_within_rounding_of_the_float_ones(name, mode):
    """As the graph class's (test_graph_kernels): float32 outputs in fused's conventions, the
    same masks to the bit, everything finite, and the rest within half's rounding in the trunk
    (measured on these rows: logits within 1.2e-3 of their scale, values within 1.2e-3, the
    policies' KL within 5e-5; the bounds leave a factor of two to three)."""
    model, obs = case(name)
    want = BatchEvaluator(model, device="cuda", infer_rows=16, mode=mode, kernels="fused")(obs)
    evaluator = BatchEvaluator(model, device="cuda", infer_rows=16, mode=mode, kernels="fused16")
    got = evaluator(obs)
    mask = obs[:, model.mask_offset: model.mask_offset + model.num_actions] > 0
    assert got[0].dtype == np.float32 and got[1].dtype == np.float32 and evaluator.exact_forwards == 0
    assert np.array_equal(got[0] > -1e8, mask) and np.all(got[0][~mask] == want[0][~mask])
    assert np.isfinite(got[0]).all() and np.isfinite(got[1]).all()
    scale = max(1.0, float(np.abs(want[0][mask]).max()))
    assert 0 < np.abs(got[0][mask] - want[0][mask]).max() <= 4e-3 * scale
    assert np.abs(got[1] - want[1]).max() <= 4e-3
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
    ("backbone.layers.1.node_self.weight", 1e4),        # updates past 65504: inf from the GEMM
    ("backbone.layers.0.tile_self.weight", 1e6),        # weights that are inf as halves
    ("backbone.layers.2.edge_from_node.weight", 1e3),
    ("backbone.layers.0.seat_messages.weight", 1e6),    # a seat's messages past 65504 at its entities
    ("backbone.layers.1.node_from_seat.weight", 1e6),   # the board's weights on them
    ("backbone.layers.2.edge_from_seat.weight", 1e6),
    ("backbone.layers.1.tile_from_seat.weight", 1e6),
    ("policy_head.settlement.0.weight", 1e4),
    ("policy_head.road.0.weight", 1e4),
    ("policy_head.robber_tile.weight", 1e3),
], ids=["update", "weights", "edges", "messages", "node-seats", "edge-seats", "tile-seats", "settlement", "road",
        "robber"])
@pytest.mark.parametrize("mode", ["strict", "fast"])
def test_a_half_past_its_range_is_evaluated_in_float(name, factor, mode):
    """Weights no half forward can carry: the kernels flag it (bit 2), whatever their outputs
    look like, and the evaluator returns the float32 kernels' outputs for the batch."""
    model = shared_model()
    with torch.no_grad():
        dict(model.named_parameters())[name].mul_(factor)
    obs = rows(model, "base-4p", count=24)
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
    model = shared_model()
    obs = rows(model, "base-3p", count=8)
    evaluator = BatchEvaluator(model, device="cuda", infer_rows=8, kernels="fused16")
    before, exact = evaluator(obs), evaluator.exact(obs)
    with torch.no_grad():
        dict(model.named_parameters())["backbone.layers.0.seat_self.weight"].mul_(1.1)
    after = evaluator(obs)
    assert np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1])
    assert np.array_equal(evaluator.exact(obs)[1], exact[1])
    module = BatchEvaluator(model, device="cuda", infer_rows=8)(obs)
    assert np.abs(exact[1] - module[1]).max() > 1e-3
    fresh = BatchEvaluator(model, device="cuda", infer_rows=8, kernels="fused16")
    assert np.abs(fresh.exact(obs)[1] - module[1]).max() <= 1e-5 and np.abs(fresh(obs)[1] - module[1]).max() <= 4e-3


@pytest.mark.gpu
@needs_kernels
@pytest.mark.parametrize("first,second", [("fused16", "fused16"), ("fused16", "fused"), ("fused", "fused16")])
def test_half_graphs_do_not_share_a_workspace_with_other_graphs(first, second):
    """BatchEvaluator._own_workspace (test_graph_kernels): an evaluator captured later must
    still replay once an earlier half one is gone and the cache emptied."""
    model = shared_model()
    obs = mixed_rows(model, count=16)
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


# ---------------------------------------------------------------- the trained network


def self_play_rows(model, scenario, count=8, keep=256, seed=0):
    """Rows of games the network plays itself on the GPU (its sampled policy at every seat's
    decision), ``keep`` of them spread over the games' whole length."""
    config = games.load_game(scenario)
    encoder = encoder_of(model)
    live = [config.cppanatron_game(map_seed=1000 + seed + i, seed=seed + i) for i in range(count)]
    generator = torch.Generator(device="cuda").manual_seed(seed)
    out = []
    for _ in range(1500):
        live = [game for game in live if game.winning_color is None]
        if not live:
            break
        obs = np.stack([encoder.encode(game, game.current_player_index) for game in live]).astype(np.float32)
        out.append(obs)
        with torch.inference_mode():
            policy = torch.softmax(model(torch.from_numpy(obs).cuda())[0], dim=1)
        for game, index in zip(live, torch.multinomial(policy, 1, generator=generator)[:, 0].tolist()):
            game.execute(encoder.action_space.decode(game, index))
    out = np.concatenate(out)
    return out[np.linspace(0, len(out) - 1, min(keep, len(out))).astype(int)]


@pytest.mark.gpu
@pytest.mark.runs
@needs_kernels
@pytest.mark.parametrize("scenario", ["base-2p-fr", "base-3p", "base-4p", "hexagon-3p"])
def test_the_trained_network_on_its_own_games(scenario):
    """gen-base-2/main@e3800 (mix-graph-seat-s-2: graph-seat-s, private level, dice input): the
    kernels against the module on rows of games it plays itself, within the bounds
    seat_kernels.py documents, and fused16 within half's rounding of fused."""
    from catanrl.runs import RunError, load_checkpoint

    try:
        checkpoint = load_checkpoint("gen-base-2/main@e3800#ac792590f9f1")
    except (RunError, OSError, ValueError):
        pytest.skip("gen-base-2/main@e3800 is not in the runs directory")
    model = checkpoint.model("all", "cuda")
    assert model.spec["class"] == "seat" and model.levels["level"] == "private" and model.levels["dice_state"]
    obs = self_play_rows(model, scenario)
    assert len(obs) == 256
    want = BatchEvaluator(model, device="cuda", infer_rows=64)(obs)
    got = BatchEvaluator(model, device="cuda", infer_rows=64, kernels="fused")(obs)
    mask = obs[:, model.mask_offset: model.mask_offset + model.num_actions] > 0
    assert np.array_equal(got[0] > -1e8, mask) and np.all(got[0][~mask] == want[0][~mask])
    scale = max(1.0, float(np.abs(want[0][mask]).max()))
    assert np.abs(got[0][mask] - want[0][mask]).max() <= 1e-6 * scale + 1e-5
    assert np.abs(got[1] - want[1]).max() <= 1e-6
    best = np.take_along_axis(want[0], got[0].argmax(1)[:, None], 1)[:, 0]
    assert np.all(best >= want[0].max(1) - 1e-6 * scale - 1e-5)
    evaluator = BatchEvaluator(model, device="cuda", infer_rows=64, kernels="fused16")
    half = evaluator(obs)
    assert evaluator.exact_forwards == 0 and np.array_equal(half[0] > -1e8, mask)
    assert np.abs(half[0][mask] - got[0][mask]).max() <= 4e-3 * scale and np.abs(half[1] - got[1]).max() <= 4e-3
    print(f"\n{scenario}: {len(obs)} rows, logit scale {scale:.1f}; fused vs module: logits "
          f"{np.abs(got[0][mask] - want[0][mask]).max():.2e}, values {np.abs(got[1] - want[1]).max():.2e}; "
          f"fused16 vs fused: logits {np.abs(half[0][mask] - got[0][mask]).max():.2e}, "
          f"values {np.abs(half[1] - got[1]).max():.2e}")
