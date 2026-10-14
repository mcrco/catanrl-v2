"""Test 6: checkpoints load anywhere, self-test, and refuse drift."""

from __future__ import annotations

import pytest
import torch

from catanrl import checkpoint as ck
from catanrl.checkpoint import (
    Checkpoint,
    CheckpointError,
    DimsMismatch,
    EncoderDrift,
    ModelDrift,
    SchemaMismatch,
    WeightsCorrupt,
    load_parts,
    save,
)
from catanrl.envs.encoding import action_size, dims_for, row_layout
from catanrl.models import build_model


def tiny(preset: str = "graph-tiny", games=("base-2p",), level: str = "private", critic: str | None = None,
         seed: int = 0):
    torch.manual_seed(seed)
    dims = dims_for(list(games))
    critic_level = critic if critic != level else None
    return build_model({"preset": preset}, dims=dims, levels={"level": level, "critic_level": critic},
                       num_actions=action_size(dims), layout=row_layout(dims, level, critic_level))


def saved(tmp_path, model, name="e000000.pt", epoch=0):
    config = {"env": {"scenarios": [{"game": "base-2p"}]}, "model": model.spec}
    return save(tmp_path / name, model, run=None, epoch=epoch, agent_steps=0, config=config, code={})


def rows(model, n=4):
    positions = ck.select_positions(model.dims, n)
    return torch.from_numpy(ck.encode_positions(positions, model.dims, model.levels["level"],
                                                model.levels["critic_level"]))


@pytest.mark.parametrize(("preset", "critic"), [("graph-tiny", None), ("mlp", None), ("graph-tiny", "full")])
def test_round_trip_gives_identical_outputs(tmp_path, preset, critic):
    model = tiny(preset, critic=critic).eval()
    ref = saved(tmp_path, model)
    payload = torch.load(ref.path, weights_only=True)  # plain data
    assert payload["format"] == "catanrl.ckpt" and payload["obs"]["dims"] == model.dims
    loaded = Checkpoint.load(ref.path).model("all")
    obs = rows(model)
    with torch.no_grad():
        for a, b in zip(model(obs), loaded(obs)):
            assert torch.equal(a, b)
    assert loaded.catanrl_ref == str(ref.path) and loaded.catanrl_allow_drift is False
    assert saved(tmp_path, model).weights_sha256 == ref.weights_sha256  # same epoch and weights: no-op
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    with pytest.raises(CheckpointError):
        saved(tmp_path, model)


def resave(path, mutate):
    payload = torch.load(path, weights_only=True)
    mutate(payload)
    torch.save(payload, path)


def test_tampered_weights_and_layouts_raise(tmp_path):
    model = tiny()
    ref = saved(tmp_path, model)
    raw = bytearray(ref.path.read_bytes())
    weight = max(model.state_dict().values(), key=lambda t: t.numel()).numpy().tobytes()
    raw[raw.index(weight) + len(weight) // 2] ^= 0x01  # one bit of the largest weight tensor
    (tmp_path / "flipped.pt").write_bytes(bytes(raw))
    with pytest.raises(WeightsCorrupt):
        Checkpoint.load(tmp_path / "flipped.pt")
    resave(ref.path, lambda p: p["state_dict"][next(iter(p["state_dict"]))].add_(1e-3))
    with pytest.raises(WeightsCorrupt):
        Checkpoint.load(ref.path)
    ref = saved(tmp_path, tiny(), name="e2.pt", epoch=2)
    resave(ref.path, lambda p: p["obs"]["layout_hash"].update(actor="0" * 16))
    with pytest.raises(SchemaMismatch):
        Checkpoint.load(ref.path).model()


def test_encoder_and_model_drift_raise_unless_allowed(tmp_path, monkeypatch):
    path = saved(tmp_path, tiny()).path
    real = ck.encode_positions

    def patched(*args, **kwargs):
        obs = real(*args, **kwargs)
        obs[0, 0] += 1.0
        return obs

    monkeypatch.setattr(ck, "encode_positions", patched)
    with pytest.raises(EncoderDrift):
        Checkpoint.load(path).model()
    assert Checkpoint.load(path).model(allow_drift=True).catanrl_allow_drift is True
    monkeypatch.setattr(ck, "encode_positions", real)

    from catanrl.models import GraphModel

    original = GraphModel.policy_logits
    monkeypatch.setattr(GraphModel, "policy_logits", lambda self, obs: original(self, obs) + 0.01)
    with pytest.raises(ModelDrift):
        Checkpoint.load(path).model()
    Checkpoint.load(path).model("critic")  # the critic still reproduces


def test_graph_runs_at_larger_dims_and_mlp_refuses(tmp_path):
    big = dims_for(["base-4p"])
    graph = Checkpoint.load(saved(tmp_path, tiny("graph-tiny")).path).model(dims=big)
    assert graph.dims == big
    mlp = Checkpoint.load(saved(tmp_path, tiny("mlp"), name="mlp.pt").path)
    with pytest.raises(DimsMismatch):
        mlp.model(dims=big)


def test_init_from_parts(tmp_path):
    source = tiny(seed=1)
    path = saved(tmp_path, source).path
    target = tiny(seed=2)
    load_parts(target, {"ckpt": str(path), "parts": ["actor"]})
    state, src = target.state_dict(), source.state_dict()
    assert all(torch.equal(state[k], src[k]) for k in state if k.startswith("actor."))
    assert not all(torch.equal(state[k], src[k]) for k in state if k.startswith("critic."))
    with pytest.raises(CheckpointError):
        load_parts(tiny("graph-s"), {"ckpt": str(path), "parts": ["actor"]})


def test_every_policy_head_is_self_tested(tmp_path, monkeypatch):
    """A change to logits of actions illegal in every stored position still drifts."""
    from catanrl.envs.encoding import make_encoder
    from catanrl.models import GraphModel

    model = tiny()
    path = saved(tmp_path, model).path
    offset, (size,) = make_encoder(model.dims, "private").action_space.layout()["city"]
    original = GraphModel.policy_logits

    def shifted(self, obs):
        out = original(self, obs).clone()
        out[:, offset:offset + size] += 0.5
        return out

    monkeypatch.setattr(GraphModel, "policy_logits", shifted)
    with pytest.raises(ModelDrift):
        Checkpoint.load(path).model()


def test_positions_carry_their_maps(tmp_path, monkeypatch):
    from catanrl import maps

    path = saved(tmp_path, tiny()).path
    real = maps.generate
    monkeypatch.setattr(maps, "generate", lambda config, seed: real(config, seed + 1))
    Checkpoint.load(path).model()  # a map-generator change is not an encoder drift


def test_self_tests_ignore_a_leaked_autocast(tmp_path):
    with torch.autocast("cpu", dtype=torch.bfloat16):
        ref = saved(tmp_path, tiny())
    Checkpoint.load(ref.path).model()


def test_positions_cover_prompts_and_full_tables():
    from catanrl.envs.encoding import dims_for

    kinds = {ck._kind(ck._position_game(p)) for p in ck.packaged_positions()}
    assert kinds >= {"turn", "turn+rolled", "MOVE_ROBBER", "DISCARD", "BUILD_INITIAL_SETTLEMENT",
                     "BUILD_INITIAL_ROAD", "road_building"}
    chosen = ck.select_positions(dims_for(["base-4p"]), 8)
    assert all(p["game"]["num_players"] == 4 for p in chosen)
    assert all(p["seat"] == ck._position_game(p).current_player_index for p in chosen)
