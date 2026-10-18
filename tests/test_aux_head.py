"""The auxiliary hidden-information head (``model.args.aux`` on a shared graph
trunk, ``train.aux_coef``): the hidden-feature set against the encoder, the
head's shape and masking, fairness (no head reads the critic block), the
loss, training with the knob at 0 (the plain PPO, bitwise) and above 0 (the
aux loss falls), and a checkpoint with the head loading, self-testing and
playing through the harness and ``catanrl eval`` without the block."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from run_helpers import train
from test_models import BASE_2P, BASE_4P, encode
from train_configs import tiny_ppo

from catanrl import games
from catanrl.agents import resolve_agent
from catanrl.checkpoint import Checkpoint, ModelDrift, _declare, _position_game, load_parts
from catanrl.envs.encoding import (
    HIDDEN_FEATURES,
    action_size,
    hidden_feature_indices,
    make_encoder,
    row_layout,
)
from catanrl.models import GraphModel, build_model
from catanrl.models.rows import canonical_feature_names, feature_names
from catanrl.train import puffer
from catanrl.train.puffer_policy import PufferPolicy, ValueBatch

AUX = {"preset": "graph-tiny", "args": {"shared": True, "aux": True}}
SHARED = {"preset": "graph-tiny", "args": {"shared": True}}


def aux_model(dims=BASE_2P, seed=0, critic="full"):
    torch.manual_seed(seed)
    return build_model(AUX, dims=dims, levels={"level": "private", "critic_level": critic},
                       num_actions=action_size(dims), layout=row_layout(dims, "private", critic)).eval()


def declared_positions(name: str, count: int = 6, seed: int = 0) -> list[dict]:
    """Positions of random games with chosen hands and dev cards for every seat, so each
    hidden feature takes a non-zero value for some other seat."""
    cfg = games.load_game(name)
    rng = np.random.default_rng(seed)
    hands = [[2, 1, 0, 1, 3], [0, 2, 1, 1, 0], [1, 0, 3, 0, 1], [1, 1, 1, 1, 1]]
    decks = [[1, 1, 0, 0, 1], [1, 0, 1, 0, 1], [1, 0, 0, 1, 1], [2, 0, 0, 0, 0]]
    out = []
    for i in range(count):
        game = cfg.cppanatron_game(map_seed=seed + i, seed=seed + i)
        for _ in range(int(rng.integers(20, 120))):
            if game.winning_color is not None:
                break
            actions = game.playable_actions
            game.execute(actions[int(rng.integers(len(actions)))])
        pos = {"game": cfg.to_dict(), "map_seed": seed + i, "map": cfg.map_spec(seed + i), "seed": seed + i,
               "seat": game.current_player_index, **_declare(game)}
        for k, player in enumerate(pos["players"]):
            player["resources"] = hands[(k + i) % 4]
            player["development_cards"] = decks[(k + i) % 4]
        out.append(pos)
    return out


def rows_with_block(positions, dims, critic="full"):
    """Rows as the VecEnv writes them at (private, critic): the private row then the critic's players block."""
    actor, other = make_encoder(dims, "private"), make_encoder(dims, critic)
    rows = []
    for pos in positions:
        game = _position_game(pos)
        rows.append(np.concatenate([actor.encode(game, pos["seat"]), other.encode_players(game, pos["seat"])]))
    return torch.from_numpy(np.stack(rows).astype(np.float32))


# ---------------------------------------------------------------- the hidden features


@pytest.mark.parametrize(("name", "dims"), [("base-2p", BASE_2P), ("base-4p", BASE_4P)])
def test_hidden_features_are_what_private_rows_zero_for_other_seats(name, dims):
    """Over declared positions, the players-block columns that differ between the private and
    the full row are exactly HIDDEN_FEATURES for the other seats, and none for the own seat."""
    positions = declared_positions(name)
    names = feature_names(dims, "private")["players"]
    offset, (players, width) = row_layout(dims, "private")["players"]
    encoders = {level: make_encoder(dims, level) for level in ("private", "public", "full")}
    blocks = {}
    for level, encoder in encoders.items():
        rows = np.stack([encoder.encode(_position_game(p), p["seat"]) for p in positions])
        blocks[level] = rows[:, offset : offset + players * width].reshape(len(positions), players, width)
    private, public, full = blocks["private"], blocks["public"], blocks["full"]
    assert not (private[:, 0] != full[:, 0]).any()                      # own seat: identical at every level
    differing = (private[:, 1:] != full[:, 1:]).any(axis=(0, 1))
    assert {names[i] for i in np.flatnonzero(differing)} == set(HIDDEN_FEATURES)
    shown_at_public = (private[:, 1:] != public[:, 1:]).any(axis=(0, 1))
    assert {names[i] for i in np.flatnonzero(shown_at_public)} == {f for f in HIDDEN_FEATURES if f.startswith("resource_")}
    assert (full[:, 1:][..., [names.index(f) for f in HIDDEN_FEATURES]] >= 0).all()
    assert (full[:, 1:][..., [names.index(f) for f in HIDDEN_FEATURES]] <= 1).all()


def test_hidden_feature_indices_follow_the_feature_order():
    canonical = canonical_feature_names()["players"]
    index = hidden_feature_indices(canonical)
    assert [canonical[i] for i in index] == list(HIDDEN_FEATURES) and len(index) == 15
    assert index == hidden_feature_indices(feature_names(BASE_4P, "full")["players"])  # same names at every level
    with pytest.raises(ValueError, match="hidden features"):
        hidden_feature_indices([n for n in canonical if n != "dev_knight"])


# ---------------------------------------------------------------- the head


def test_aux_head_predicts_the_other_seats_and_targets_come_from_the_block():
    model = aux_model(BASE_4P)
    assert isinstance(model, GraphModel) and model.aux and model.has_critic_block and not model.reads_critic_block
    assert model.parts == {"actor": ["backbone.", "policy_head."], "critic": ["value_head."], "aux": ["aux_head."]}
    positions = declared_positions("base-2p", 3) + declared_positions("base-4p", 3)
    obs = rows_with_block(positions, BASE_4P)
    with torch.no_grad():
        _logits, _values, present, predictions = model.heads_with_aux(obs)
        targets, others = model.aux_targets(obs)
        assert torch.equal(predictions, model.aux_predictions(obs))
    assert predictions.shape == targets.shape == (6, 3, 15) and others.shape == (6, 3)
    assert others.tolist() == [[1.0, 0.0, 0.0]] * 3 + [[1.0, 1.0, 1.0]] * 3        # a 2p game has one other seat
    assert torch.equal(others, present[:, 1:])
    # The targets are the full row's hidden features of slots 1..3.
    offset, (players, width) = row_layout(BASE_4P, "full")["players"]
    full = encode([_position_game(p) for p in positions], BASE_4P, "full")[:, offset : offset + players * width]
    full = full.reshape(6, players, width)[:, 1:]
    names = feature_names(BASE_4P, "full")["players"]
    assert torch.equal(targets, full[..., [names.index(f) for f in HIDDEN_FEATURES]])
    assert bool((targets[3:] > 0).any(dim=(0, 1)).all())                         # every feature is exercised in 4p
    assert bool((targets[:3, 1:] == 0).all())                                     # absent seats are all zero


def test_no_head_reads_the_critic_block():
    """Policy and value are identical whatever the block holds (fairness); the aux targets
    change with it; a model built without the block gives the same outputs and predictions."""
    model = aux_model(BASE_2P)
    positions = declared_positions("base-2p", 4)
    rows = rows_with_block(positions, BASE_2P)
    swapped = rows_with_block(positions, BASE_2P, critic="private")               # the block as the private players
    with torch.no_grad():
        logits, values, _present, predictions = model.heads_with_aux(rows)
        logits2, values2, _present2, predictions2 = model.heads_with_aux(swapped)
        assert torch.equal(logits, logits2) and torch.equal(values, values2) and torch.equal(predictions, predictions2)
        assert not torch.equal(model.aux_targets(rows)[0], model.aux_targets(swapped)[0])
        plain = build_model(AUX, dims=BASE_2P, levels={"level": "private"}, num_actions=action_size(BASE_2P),
                            layout=row_layout(BASE_2P, "private")).eval()
        plain.load_state_dict(model.state_dict())
        assert not plain.has_critic_block
        private = rows[:, : plain.obs_size]
        assert torch.equal(plain(private)[0], logits) and torch.equal(plain.aux_predictions(private), predictions)
        with pytest.raises(ValueError, match="critic block"):
            plain.aux_targets(private)
    with pytest.raises(ValueError, match="shared trunk cannot read"):
        build_model(SHARED, dims=BASE_2P, levels={"level": "private", "critic_level": "full"},
                    num_actions=action_size(BASE_2P), layout=row_layout(BASE_2P, "private", "full"))
    with pytest.raises(ValueError, match="shared trunk"):
        build_model({"preset": "graph-tiny", "args": {"aux": True}}, dims=BASE_2P, levels={"level": "private"},
                    num_actions=action_size(BASE_2P), layout=row_layout(BASE_2P, "private"))


def test_aux_head_leaves_the_other_parts_initial_weights_unchanged():
    """Built last: for one seed the trunk, policy and value weights equal the plain shared model's."""
    torch.manual_seed(3)
    plain = build_model(SHARED, dims=BASE_2P, levels={"level": "private"}, num_actions=action_size(BASE_2P),
                        layout=row_layout(BASE_2P, "private"))
    with_aux = aux_model(BASE_2P, seed=3)
    state, other = plain.state_dict(), with_aux.state_dict()
    assert set(state) < set(other) and all(k.startswith("aux_head.") for k in set(other) - set(state))
    assert all(torch.equal(state[k], other[k]) for k in state)


def test_aux_loss_is_the_masked_mean_squared_error():
    policy = PufferPolicy(aux_model(BASE_4P), num_rows=8, aux_coef=0.5)
    torch.manual_seed(0)
    predictions, targets = torch.randn(5, 3, 15), torch.rand(5, 3, 15)
    others = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    batch = ValueBatch(torch.arange(5), torch.zeros(5, 4), torch.ones(5, 4), 5, (predictions, targets, others))
    weighted, terms = policy.aux_loss(batch)
    expected = ((predictions - targets) ** 2 * others.unsqueeze(-1)).sum() / (others.sum() * 15)
    assert torch.allclose(terms["aux"], expected) and torch.allclose(weighted, 0.5 * expected)
    # Masked seats do not count, whatever is predicted for them.
    noisy = predictions.clone()
    noisy[others == 0] = 100.0
    assert torch.allclose(policy.aux_loss(ValueBatch(batch.index, batch.logits, batch.present, 5,
                                                     (noisy, targets, others)))[1]["aux"], expected)
    summary = policy.pop_aux()
    pairs = others.sum()
    for i, name in enumerate(HIDDEN_FEATURES):
        mask = others.unsqueeze(-1)
        mse = (((predictions - targets) ** 2 * mask).sum(dim=(0, 1))[i] * 2) / (2 * pairs)
        mean = (targets * mask).sum(dim=(0, 1))[i] / pairs
        var = ((targets ** 2) * mask).sum(dim=(0, 1))[i] / pairs - mean ** 2
        assert summary[f"mse_{name}"] == pytest.approx(float(mse), rel=1e-5)
        assert summary[f"var_{name}"] == pytest.approx(float(var), rel=1e-4, abs=1e-6)
    assert policy.pop_aux() == {} and policy.aux_loss(ValueBatch(batch.index, batch.logits, batch.present, 5)) is None
    with pytest.raises(ValueError, match="aux head"):
        PufferPolicy(build_model(SHARED, dims=BASE_2P, levels={"level": "private"}, num_actions=action_size(BASE_2P),
                                 layout=row_layout(BASE_2P, "private")), num_rows=8, aux_coef=0.1)
    with pytest.raises(ValueError, match="critic block"):
        PufferPolicy(build_model(AUX, dims=BASE_2P, levels={"level": "private"}, num_actions=action_size(BASE_2P),
                                 layout=row_layout(BASE_2P, "private")), num_rows=8, aux_coef=0.1)


# ---------------------------------------------------------------- training


def aux_config(aux_coef: float, **train) -> dict:
    cfg = tiny_ppo(epochs=3, **train)
    cfg["obs"] = {"level": "private", "critic_level": "full"}
    cfg["model"] = AUX
    cfg["train"]["aux_coef"] = aux_coef
    return cfg


@pytest.fixture
def forked_self_tests(monkeypatch):
    """Checkpoint self-tests rebuild a network, which draws from torch's RNG: forked here so
    the two runs compared below sample identically whatever the network's size."""
    from catanrl import checkpoint

    original = checkpoint.make_self_test

    def forked(*args, **kwargs):
        with torch.random.fork_rng():
            return original(*args, **kwargs)

    monkeypatch.setattr(checkpoint, "make_self_test", forked)


def test_aux_coef_zero_is_the_plain_training(forked_self_tests):
    """With the head and the block in place but aux_coef 0, the run equals the plain shared
    model's bitwise: losses per epoch, the shared parts' weights, the policy's logits."""
    plain_cfg = tiny_ppo(epochs=3)
    plain_cfg["model"] = SHARED
    plain = train(puffer.train_ppo, plain_cfg, "plain")
    off = train(puffer.train_ppo, aux_config(0.0), "aux-off")
    assert plain.status == off.status == "finished" and len(plain.rows) == len(off.rows) == 3
    keys = [k for k in plain.rows[0] if k.startswith(("losses/", "critic/", "environment/"))]
    assert "losses/aux" not in off.rows[0] and not any(k.startswith("aux/") for k in off.rows[0])
    for a, b in zip(plain.rows, off.rows):
        assert {k: a[k] for k in keys} == {k: b[k] for k in keys}
    plain_ckpt, off_ckpt = Checkpoint.load(plain.saved[-1].path), Checkpoint.load(off.saved[-1].path)
    assert plain_ckpt.ref.epoch == off_ckpt.ref.epoch == 3
    assert all(torch.equal(plain_ckpt.state_dict[k], off_ckpt.state_dict[k]) for k in plain_ckpt.state_dict)
    assert "aux" in off_ckpt.header["self_test"] and "aux" not in plain_ckpt.header["self_test"]
    net, with_head = plain_ckpt.model("all"), off_ckpt.model("all")
    obs = rows_with_block(declared_positions("base-2p", 3), BASE_2P)
    with torch.no_grad():
        assert torch.equal(net(obs[:, : net.obs_size])[0], with_head(obs)[0])


def test_aux_loss_trains_the_head_on_fixed_rows():
    """Adam on the aux loss alone, over fixed rows, cuts it by more than four times."""
    model = aux_model(BASE_2P, seed=1).train()
    policy = PufferPolicy(model, num_rows=2, aux_coef=1.0)
    rows = rows_with_block(declared_positions("base-2p", 12, seed=4), BASE_2P)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    def step():
        _, _, present, predictions = model.heads_with_aux(rows)
        targets, others = model.aux_targets(rows)
        batch = ValueBatch(torch.arange(len(rows)), torch.zeros(len(rows), 2), present, len(rows),
                           (predictions, targets, others))
        weighted, terms = policy.aux_loss(batch)
        return weighted, float(terms["aux"])

    first = step()[1]
    for _ in range(60):
        optimizer.zero_grad()
        weighted, _ = step()
        weighted.backward()
        optimizer.step()
    last = step()[1]
    assert first > 0 and last < first / 4, (first, last)


def test_aux_head_learns_in_training_and_is_logged_per_feature():
    """losses/aux and aux/mse_<feature>, aux/var_<feature> are logged; over eight tiny epochs
    the head's R^2 on the other seat's actual VP (its public VP plus hidden VP cards: the
    easy one) rises from below zero (a fresh head predicts ~0, worse than the mean) to above
    0.3. The loss itself is not monotone in a short run: hands fill as games progress."""
    cfg = aux_config(1.0, learning_rate=3e-3, update_epochs=4)
    cfg["train"]["total_timesteps"] = 8 * 2 * 8 * 8   # 8 epochs
    writer = train(puffer.train_ppo, cfg, "aux-on")
    assert writer.status == "finished"
    rows = writer.rows
    assert len(rows) == 8 and all(row["losses/aux"] > 0 for row in rows)
    for name in HIDDEN_FEATURES:
        assert f"aux/mse_{name}" in rows[-1] and f"aux/var_{name}" in rows[-1]
    mse = [rows[-1][f"aux/mse_{name}"] for name in HIDDEN_FEATURES]
    assert rows[-1]["losses/aux"] == pytest.approx(sum(mse) / len(mse), rel=0.05)   # the epoch's mean over features

    def r2(row, name):
        return 1.0 - row[f"aux/mse_{name}"] / row[f"aux/var_{name}"]

    trajectory = [r2(row, "actual_vps") for row in rows]
    assert trajectory[0] < 0.0 and trajectory[-1] > 0.3, trajectory
    # The checkpoint carries the head: it loads, self-tests (the head too), warm-starts, and plays.
    ref = writer.saved[-1]
    ckpt = Checkpoint.load(ref.path)
    assert ckpt.header["parts"]["aux"] == ["aux_head."] and ckpt.header["obs"]["critic_level"] == "full"
    loaded = ckpt.model("all")
    assert loaded.aux and loaded.has_critic_block
    target = aux_model(BASE_2P, seed=9)
    load_parts(target, {"ckpt": str(ref.path), "parts": ["actor", "critic"]})
    assert all(torch.equal(target.state_dict()[k], ckpt.state_dict[k]) for k in ckpt.state_dict
               if not k.startswith("aux_head."))
    original = GraphModel.aux_predictions
    GraphModel.aux_predictions = lambda self, obs: original(self, obs) + 0.01
    try:
        Checkpoint.load(ref.path).model("actor")           # the actor's self-test does not pin the head
        with pytest.raises(ModelDrift, match="aux"):
            Checkpoint.load(ref.path).model("all")
    finally:
        GraphModel.aux_predictions = original
    _plays_without_the_block(ref)


def _plays_without_the_block(ref):
    from eval_helpers import make_suite, schedule

    from catanrl.cli import main
    from catanrl.eval.harness import play_schedule

    agent = resolve_agent(f"POLICY:ckpt={ref.path}").native(device="cpu", infer_rows=4)
    assert agent.model.aux and not agent.model.has_critic_block
    assert agent.encoder.size == agent.model.obs_size == row_layout(agent.dims, "private")["action_mask"][0] + \
        row_layout(agent.dims, "private")["action_mask"][1][0]
    suite = make_suite(({"game": "base-2p"},), turn_limit=60)
    rows = []
    play_schedule([agent, resolve_agent("F").native(device="cpu", infer_rows=4)], schedule(suite, 2, boards=2),
                  suite=suite, on_game=rows.append, threads=1, num_envs=2, device="cpu", infer_rows=4)
    assert len(rows) == 4 and all(r["winner"] in (0, 1, None) for r in rows)
    assert main(["eval", f"POLICY:ckpt={ref.path}", "F", "--suite", "heldout-2p", "--boards", "1", "--threads", "1",
                 "--num-envs", "2"]) == 0
    mcts = resolve_agent(f"MCTS:ckpt={ref.path}:sims=4").native(device="cpu", infer_rows=4)
    assert not mcts.model.has_critic_block and mcts.pool.observation_size == mcts.model.obs_size


# ---------------------------------------------------------------- self-tests per part


def test_the_aux_part_is_pinned_by_its_own_self_test_and_run_only_when_compared(tmp_path):
    """``part="aux"`` (what load_parts of the head asks for) compares the head's predictions;
    the ``actor`` and ``critic`` self-tests never run the head; an unknown part is refused
    and a checkpoint without the head has no aux part."""
    from catanrl.checkpoint import CheckpointError, save

    path = tmp_path / "e0.pt"
    save(path, aux_model(BASE_2P, seed=2), run=None, epoch=0, agent_steps=0, config=None, code=None)
    stored = Checkpoint.load(path)
    target = aux_model(BASE_2P, seed=9)
    head = [k for k in stored.state_dict if k.startswith("aux_head.") and k.endswith("weight")]  # biases start at 0
    assert head and not any(torch.equal(target.state_dict()[k], stored.state_dict[k]) for k in head)
    calls, original = [], GraphModel.aux_predictions

    def drifted(self, obs):
        calls.append(len(obs))
        return original(self, obs) + 0.01

    GraphModel.aux_predictions = drifted
    try:
        Checkpoint.load(path).model("actor")
        Checkpoint.load(path).model("critic")
        assert calls == []  # neither self-test runs the head
        with pytest.raises(ModelDrift, match="aux"):
            load_parts(target, {"ckpt": str(path), "parts": ["aux"]})
        assert len(calls) == 1
        assert not any(torch.equal(target.state_dict()[k], stored.state_dict[k]) for k in head)  # nothing copied
        with pytest.raises(ModelDrift, match="aux"):
            Checkpoint.load(path).model("all")
    finally:
        GraphModel.aux_predictions = original
    refs = load_parts(target, {"ckpt": str(path), "parts": ["aux"]})
    assert [r.weights_sha256 for r in refs] == [stored.ref.weights_sha256]
    assert all(torch.equal(target.state_dict()[k], stored.state_dict[k]) for k in head)
    with pytest.raises(ValueError, match="part must be one of"):
        stored.self_test(part="head")
    plain = tmp_path / "plain.pt"
    save(plain, build_model(SHARED, dims=BASE_2P, levels={"level": "private"}, num_actions=action_size(BASE_2P),
                            layout=row_layout(BASE_2P, "private")), run=None, epoch=0, agent_steps=0, config=None,
         code=None)
    with pytest.raises(CheckpointError, match="no aux head"):
        Checkpoint.load(plain).self_test(part="aux")
