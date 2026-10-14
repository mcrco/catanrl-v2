"""Networks over graph rows: layout reading, masking, dims-agnosticism, values."""

from __future__ import annotations

import cppanatron as cpp
import numpy as np
import pytest
import torch

from catanrl import games
from catanrl.envs.encoding import action_size, dims_for, make_encoder, row_layout
from catanrl.models import build_model, resolve_spec

BASE_2P = {"max_players": 2, "max_tiles": 19, "max_nodes": 54, "max_edges": 72}
WIDE_4P = {"max_players": 4, "max_tiles": 30, "max_nodes": 90, "max_edges": 120}


def model_for(preset, dims=BASE_2P, level="private", seed=0, **args):
    torch.manual_seed(seed)
    spec = {"preset": preset} if not args else {**resolve_spec({"preset": preset}), "args": {
        **resolve_spec({"preset": preset})["args"], **args}}
    return build_model(spec, dims=dims, levels={"level": level}, num_actions=action_size(dims),
                       layout=row_layout(dims, level)).eval()


def played_games(name="base-2p", count=6, plies=80, seed=0):
    """Games at assorted points: initial placement, mid-game, robber prompts."""
    config = games.load_game(name)
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        game = config.cppanatron_game(map_seed=seed + i, seed=seed + i)
        for _ in range(int(rng.integers(0, plies))):
            if game.winning_color is not None:
                break
            actions = game.playable_actions
            game.execute(actions[int(rng.integers(len(actions)))])
        out.append(game)
    return out


def encode(games_list, dims, level="private"):
    encoder = make_encoder(dims, level)
    return torch.from_numpy(np.stack([encoder.encode(g, g.current_player_index) for g in games_list]))


@pytest.mark.parametrize("preset", ["graph-tiny", "mlp"])
def test_logits_are_masked_and_values_are_probabilities(preset):
    model = model_for(preset)
    obs = encode(played_games(), BASE_2P)
    logits, values = model(obs)
    mask = obs[:, model.mask_offset : model.mask_offset + model.num_actions]
    assert logits.shape == (obs.shape[0], model.num_actions)
    assert torch.all((logits > -1e8) == (mask > 0))
    assert torch.all(mask.gather(1, logits.argmax(1, keepdim=True)) > 0)
    assert values.shape == (obs.shape[0], 2)
    assert torch.all((values > 0) & (values < 1))


def test_fresh_value_head_predicts_one_over_n_per_seat():
    dims = {"max_players": 4, "max_tiles": 19, "max_nodes": 54, "max_edges": 72}
    model = model_for("graph-s", dims)
    obs = encode(played_games("base-4p", count=4), dims)
    _, values = model(obs)
    assert torch.allclose(values.sum(1), torch.ones(4), atol=0.1)
    assert torch.allclose(values, torch.full_like(values, 0.25), atol=0.05)


def test_graph_presets_match_v1_sizes():
    model = model_for("graph-s")
    count = lambda prefix: sum(p.numel() for k, p in model.named_parameters() if k.startswith(prefix))
    assert 0.40e6 < count("actor.") < 0.55e6
    assert 0.30e6 < count("critic.") < 0.45e6
    assert model.parts == {"actor": ["actor."], "critic": ["critic."]}
    assert model.dims_agnostic


def _names(dims):
    return list(cpp.GraphActionSpace(cpp.EncoderDims(**dims)).names())


def test_graph_network_is_dims_agnostic():
    """The same weights give the same outputs on rows padded to larger dims
    (more seats, tiles, nodes and edges)."""
    small = model_for("graph-tiny", BASE_2P, seed=3)
    large = model_for("graph-tiny", WIDE_4P, seed=99)
    large.load_state_dict(small.state_dict())  # identical parameter shapes
    played = played_games(count=8)
    logits_s, values_s = small(encode(played, BASE_2P))
    logits_l, values_l = large(encode(played, WIDE_4P))
    names_l = {name: i for i, name in enumerate(_names(WIDE_4P))}
    columns = [names_l[name] for name in _names(BASE_2P)]
    assert torch.allclose(logits_s, logits_l[:, columns], atol=1e-4)
    assert torch.allclose(values_s, values_l[:, :2], atol=1e-5)
    assert torch.all(values_l[:, 2:] == 0)


def test_graph_network_reads_topology_per_sample():
    """One network evaluates a batch that mixes maps of different shapes."""
    dims = dims_for(["base-4p", "random-4p"])
    model = model_for("graph-tiny", dims)
    mixed = played_games("base-4p", count=3) + played_games("random-4p", count=3)
    obs = encode(mixed, dims)
    batched, _ = model(obs)
    for i in range(obs.shape[0]):
        alone, _ = model(obs[i : i + 1])
        assert torch.allclose(batched[i], alone[0], atol=1e-5)


def test_mlp_is_not_dims_agnostic():
    assert not model_for("mlp").dims_agnostic
    small, large = model_for("mlp", BASE_2P), model_for("mlp", WIDE_4P)
    with pytest.raises(RuntimeError):
        large.load_state_dict(small.state_dict())


def test_spec_is_resolved_and_rebuilds_identically():
    model = model_for("graph-tiny")
    assert model.spec == resolve_spec(model.spec)
    rebuilt = build_model(model.spec, dims=BASE_2P, levels={"level": "private"},
                          num_actions=action_size(BASE_2P), layout=row_layout(BASE_2P, "private"))
    rebuilt.load_state_dict(model.state_dict())
    obs = encode(played_games(count=3), BASE_2P)
    assert torch.equal(model(obs)[0], rebuilt.eval()(obs)[0])


def test_bad_specs_raise():
    with pytest.raises(ValueError):
        resolve_spec({"preset": "graph-xl"})
    with pytest.raises(ValueError):
        resolve_spec({"class": "graph", "args": {"hiden": 3}})
    with pytest.raises(ValueError):
        build_model({"preset": "graph-tiny"}, dims=BASE_2P, levels={"level": "private"}, num_actions=7)
    layout = row_layout(BASE_2P, "private")
    layout["nodes"] = (layout["nodes"][0] + 1, layout["nodes"][1])
    with pytest.raises(ValueError):
        build_model({"preset": "graph-tiny"}, dims=BASE_2P, levels={"level": "private"},
                    num_actions=action_size(BASE_2P), layout=layout)


def test_shared_trunk_variant():
    model = model_for("graph-tiny", shared=True)
    assert model.parts["actor"] == ["backbone.", "policy_head."]
    logits, values = model(encode(played_games(count=2), BASE_2P))
    assert logits.shape[0] == values.shape[0] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_graph_forward_is_deterministic_on_cuda():
    model = model_for("graph-tiny").cuda()
    obs = encode(played_games(count=8), BASE_2P).cuda()
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        with torch.no_grad():
            first = model(obs)
            second = model(obs)
            one = model(obs[3:4])
    finally:
        torch.use_deterministic_algorithms(previous)
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    assert torch.allclose(first[0][3], one[0][0], atol=1e-5)


def test_critic_reads_the_critic_players_block():
    """With obs.critic_level above obs.level, the critic reads E5's critic_players block."""
    layout = row_layout(BASE_2P, "private", "full")
    assert layout["critic_players"] == (make_encoder(BASE_2P, "private").size, (2, 46))
    torch.manual_seed(0)
    model = build_model({"preset": "graph-tiny"}, dims=BASE_2P, levels={"level": "private", "critic_level": "full"},
                        num_actions=action_size(BASE_2P), layout=layout).eval()
    played = played_games(count=4, plies=200)
    private, full = encode(played, BASE_2P, "private"), encode(played, BASE_2P, "full")
    players_offset, _ = layout["players"]
    width = 2 * 46
    rows = torch.cat((private, full[:, players_offset : players_offset + width]), dim=1)
    logits, values = model(rows)
    other = rows.clone()
    other[:, -width:] = private[:, players_offset : players_offset + width]
    logits2, values2 = model(other)
    assert torch.equal(logits, logits2)
    assert not torch.equal(values, values2)
