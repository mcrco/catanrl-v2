"""Refitting a critic under a fixed policy: ``train.freeze`` and ``init_from.split_shared``.

* ``split_shared``: a shared-trunk graph checkpoint loads into separate actor and critic
  networks that compute exactly its logits and values, bit for bit (a tiny network here;
  ``-m runs``: puffer-graph2/public@e3600 on real rows).
* ``freeze: [actor]``: after critic-only updates the actor's parameters and the policy's
  logits are the start's bit for bit, the critic's parameters moved, and the optimizer holds
  the critic's parameters only (a split network under Adam and Muon; on a shared trunk the
  value head alone trains). Refusals: unknown parts, every parameter, a policy that would
  still read a trained parameter, muon_exclude on a frozen parameter, a frozen part init_from
  does not load, and frozen weights that are not init_from's at a launch.
* A relaunch of a run whose first launch died before its e0 commit loads init_from again (it
  used to resume into the seed's random network, then freeze it).
* The critic's target under ``critic_targets: lambda``, ``critic_gamma: 1``, ``gae_lambda: 1``:
  each seat's outcome (1 for its win, 0 for a loss or a turn-limit truncation) where its game
  ends inside the rollout segment, else the stored value at the segment's last step (whose
  own step has no target); the loss is binary cross-entropy on the sigmoid.
* MCTS over a split network evaluates its leaves with the critic network's values.
"""

from __future__ import annotations

import cppanatron as cpp
import numpy as np
import pytest
import torch
from eval_helpers import agent
from run_helpers import train
from train_configs import tiny_ppo

from catanrl import games as game_configs
from catanrl.checkpoint import (
    Checkpoint,
    CheckpointError,
    encode_positions,
    load_parts,
    save,
    select_positions,
)
from catanrl.envs.encoding import action_size, dims_for, row_layout
from catanrl.models import build_model
from catanrl.search import make_pool, run_searches
from catanrl.train import puffer
from catanrl.train.puffer import check_frozen_init, check_policy_frozen, freeze_parts, starts_fresh, train_ppo
from catanrl.train.puffer_policy import PufferPolicy

DIMS = dims_for(["base-2p"])


def network(*, shared: bool, level: str = "private", critic_level: str | None = None, seed: int = 0,
            noise: float = 0.3, **args) -> torch.nn.Module:
    """graph-tiny at base-2p, its weights jittered so every head's output is far from its init."""
    torch.manual_seed(seed)
    critic = critic_level if critic_level not in (None, level) else None
    model = build_model({"preset": "graph-tiny", "args": {"shared": shared, **args}}, dims=DIMS,
                        levels={"level": level, "critic_level": critic_level}, num_actions=action_size(DIMS),
                        layout=row_layout(DIMS, level, critic))
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * noise)
    return model.eval()


def rows(level: str = "private", count: int = 48, seed: int = 0) -> torch.Tensor:
    """Real rows: the checkpoint self-test's positions, then random games' decisions."""
    out = [encode_positions(select_positions(DIMS, 8), DIMS, level)]
    encoder = cpp.GraphEncoder(cpp.EncoderDims(**DIMS), getattr(cpp.InfoLevel, level.upper()))
    config = game_configs.load_game("base-2p")
    rng = np.random.default_rng(seed)
    for i in range(count):
        game = config.cppanatron_game(map_seed=100 + i, seed=i)
        for _ in range(int(rng.integers(0, 300))):
            if game.winning_color is not None:
                break
            actions = game.playable_actions
            game.execute(actions[int(rng.integers(len(actions)))])
        out.append(encoder.encode(game, game.current_player_index)[None])
    return torch.from_numpy(np.concatenate(out).astype(np.float32))


def outputs(model, obs: torch.Tensor) -> dict[str, torch.Tensor]:
    """Logits, values and every action's logit (the mask opened), as the self-test reads them."""
    offset, (size,) = model.layout["action_mask"]
    opened = obs.clone()
    opened[:, offset:offset + size] = 1.0
    with torch.no_grad():
        logits, values = model(obs)
        open_logits, _ = model(opened)
    return {"logits": logits, "values": values, "open_logits": open_logits}


def assert_same(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor], keys=("logits", "values", "open_logits")):
    for key in keys:
        assert torch.equal(a[key], b[key]), key


def saved(model, tmp_path, name: str = "shared.pt") -> str:
    path = tmp_path / name
    save(path, model, run=None, epoch=0, agent_steps=0, config=None, code=None)
    return str(path)


SPLIT = {"parts": ["actor", "critic"], "split_shared": True}


# ---------------------------------------------------------------- split_shared


@pytest.mark.parametrize("level", ["private", "public"])
def test_a_split_network_is_the_shared_one_bit_for_bit(tmp_path, level):
    shared = network(shared=True, level=level)
    split = network(shared=False, level=level, seed=1)
    load_parts(split, {"ckpt": saved(shared, tmp_path), **SPLIT})
    obs = rows(level)
    assert_same(outputs(split, obs), outputs(shared, obs))
    state = split.state_dict()
    for key, value in shared.state_dict().items():  # each network got its own copy of the trunk
        if key.startswith("backbone."):
            rest = key[len("backbone."):]
            assert torch.equal(state[f"actor.trunk.{rest}"], value) and torch.equal(state[f"critic.trunk.{rest}"], value)
    # The values are the critic network's and the logits the actor's.
    before = outputs(split, obs)
    with torch.no_grad():
        split.critic.trunk.layers[0].node_self.weight.add_(0.5)
    after = outputs(split, obs)
    assert torch.equal(after["logits"], before["logits"]) and not torch.equal(after["values"], before["values"])
    with torch.no_grad():
        split.actor.trunk.layers[0].node_self.weight.add_(0.5)
    assert not torch.equal(outputs(split, obs)["logits"], before["logits"])


def test_one_part_alone_and_the_plain_path(tmp_path):
    shared = network(shared=True)
    path = saved(shared, tmp_path)
    split = network(shared=False, seed=1)
    untouched = {k: v.clone() for k, v in split.state_dict().items() if k.startswith("actor.")}
    load_parts(split, {"ckpt": path, "parts": ["critic"], "split_shared": True})
    obs = rows(count=8)
    assert torch.equal(outputs(split, obs)["values"], outputs(shared, obs)["values"])
    assert all(torch.equal(split.state_dict()[k], v) for k, v in untouched.items())
    # Without split_shared a shared checkpoint's keys are not a split network's.
    with pytest.raises(CheckpointError, match="keys differ"):
        load_parts(network(shared=False), {"ckpt": path, "parts": ["actor"]})


def test_split_shared_refusals(tmp_path):
    shared = saved(network(shared=True), tmp_path)
    separate = saved(network(shared=False), tmp_path, "separate.pt")
    with pytest.raises(CheckpointError, match="model.args.shared: false"):
        load_parts(network(shared=True), {"ckpt": shared, **SPLIT})
    with pytest.raises(CheckpointError, match="needs a shared-trunk graph checkpoint"):
        load_parts(network(shared=False), {"ckpt": separate, **SPLIT})
    with pytest.raises(CheckpointError, match="model args"):
        load_parts(network(shared=False, hidden=24), {"ckpt": shared, **SPLIT})
    with pytest.raises(CheckpointError, match="its rows are private"):
        load_parts(network(shared=False, level="public"), {"ckpt": shared, **SPLIT})
    with pytest.raises(CheckpointError, match="critic at another level"):
        load_parts(network(shared=False, critic_level="full"), {"ckpt": shared, **SPLIT})
    mlp = build_model({"preset": "mlp", "args": {"hidden": 16, "blocks": 1}}, dims=DIMS,
                      levels={"level": "private"}, num_actions=action_size(DIMS), layout=row_layout(DIMS, "private"))
    with pytest.raises(CheckpointError, match="a mlp network has no split"):
        load_parts(mlp, {"ckpt": shared, **SPLIT})


@pytest.mark.runs
def test_puffer_graph2_public_e3600_splits_bit_for_bit():
    """The refit's start, read from the real runs directory (read only): its split network
    gives its logits and values bit for bit on its self-test rows and on public rows of
    random base-2p games."""
    from catanrl.runs import load_checkpoint

    ref = "puffer-graph2/public@e3600#3cd78e6dafac"
    ckpt = load_checkpoint(ref)
    shared = ckpt.model("all")
    assert shared.spec["args"]["shared"] and ckpt.levels["level"] == "public" and ckpt.dims == DIMS
    spec = {**ckpt.header["model"], "args": {**ckpt.header["model"]["args"], "shared": False}}
    split = build_model(spec, dims=ckpt.dims, levels=ckpt.levels, num_actions=action_size(ckpt.dims),
                        layout=row_layout(ckpt.dims, "public")).eval()
    load_parts(split, {"ckpt": ref, **SPLIT})
    obs = torch.cat((ckpt.header["self_test"]["obs"], rows("public", count=200, seed=5)))
    assert_same(outputs(split, obs), outputs(shared, obs))
    assert sum(p.numel() for p in shared.parameters()) == 541_540
    assert sum(p.numel() for p in split.parameters()) == 843_652


# ---------------------------------------------------------------- freeze


def test_freeze_parts_and_its_refusals():
    model = network(shared=False)
    frozen = freeze_parts(model, ["actor"])
    assert frozen and all(name.startswith("actor.") for name in frozen)
    assert all(p.requires_grad == name.startswith("critic.") for name, p in model.named_parameters())
    check_policy_frozen(model)
    model.actor.policy.other[0].weight.requires_grad_(True)  # a policy parameter left training
    with pytest.raises(ValueError, match="the policy's logits read parameters that still train"):
        check_policy_frozen(model)
    with pytest.raises(ValueError, match="parts are"):
        freeze_parts(network(shared=False), ["aux"])
    with pytest.raises(ValueError, match="nothing would train"):
        freeze_parts(network(shared=True), ["actor", "critic"])
    shared = network(shared=True)
    freeze_parts(shared, ["actor"])  # the head-only arm: the trunk and the policy head
    assert [n for n, p in shared.named_parameters() if p.requires_grad] == [
        n for n, _ in shared.named_parameters() if n.startswith("value_head.")]
    check_policy_frozen(shared)
    shared = network(shared=True)
    freeze_parts(shared, ["critic"])  # the value head alone: the policy still trains
    with pytest.raises(ValueError, match="still train"):
        check_policy_frozen(shared)


def critic_refit(tmp_path, *, shared: bool = False, optimizer: str = "adam", epochs: int = 4, **train_keys):
    """A tiny critic-only run from a jittered shared checkpoint (split unless ``shared``)."""
    source = network(shared=True)
    path = saved(source, tmp_path)
    keys = {"freeze": ["actor"], "critic_targets": "lambda", "critic_gamma": 1.0, "gae_lambda": 1.0,
            "learning_rate": 0.01, "update_epochs": 2, **train_keys}
    if optimizer == "muon":
        keys.update(optimizer="muon", adam_beta1=0.95, adam_eps=1e-12, max_minibatch_size=32, prio_alpha=0.8,
                    prio_beta0=0.2)
    config = tiny_ppo(epochs=epochs, **keys)
    config["model"] = {"preset": "graph-tiny", "args": {"shared": shared}}
    config["init_from"] = {"ckpt": path, "parts": ["actor", "critic"], "split_shared": not shared}
    if shared:
        config["init_from"].pop("split_shared")
    writer = train(train_ppo, config, f"refit-{optimizer}-{'shared' if shared else 'split'}")
    assert writer.status == "finished"
    return source, writer


def test_frozen_parts_must_be_init_froms(tmp_path):
    source = network(shared=True)
    init = {"ckpt": saved(source, tmp_path), **SPLIT}
    split = network(shared=False, seed=1)
    with pytest.raises(ValueError, match="random initialization"):
        check_frozen_init(split, None, ["actor"])
    with pytest.raises(ValueError, match="random initialization"):
        check_frozen_init(split, {**init, "parts": ["critic"]}, ["actor"])
    with pytest.raises(ValueError, match="differ from init_from's"):
        check_frozen_init(split, init, ["actor"])
    load_parts(split, init)
    check_frozen_init(split, init, ["actor"])
    check_frozen_init(split, init, ["actor", "critic"])
    with torch.no_grad():
        split.critic.value.mlp[-1].bias.add_(1.0)  # a trained critic: only frozen parts are compared
    check_frozen_init(split, init, ["actor"])
    with pytest.raises(ValueError, match="differ from init_from's"):
        check_frozen_init(split, init, ["critic"])
    check_frozen_init(split, None, [])  # nothing frozen, nothing checked
    check_frozen_init(network(shared=True), None, ["aux"])  # only actor and critic come from init_from


def test_starts_fresh(tmp_path):
    class Writer:
        def __init__(self, mode, path=None):
            self.mode, self.path = mode, path
    assert starts_fresh(Writer("new")) and starts_fresh(object())
    assert starts_fresh(Writer("resume", tmp_path))  # died before its first commit
    (tmp_path / "trainer_state.pt").write_bytes(b"")
    assert not starts_fresh(Writer("resume", tmp_path)) and not starts_fresh(Writer("done", tmp_path))
    assert not starts_fresh(Writer("resume"))


@pytest.mark.train
@pytest.mark.parametrize("shared", [False, True])
def test_a_relaunch_after_a_death_before_e0_loads_init_from(tmp_path, monkeypatch, shared):
    """A first launch killed between its launch row and the e0 commit (a kill, Ctrl-C, an OOM
    allocating the rollout) leaves a resumable run with no trainer state: the relaunch must load
    init_from, not resume into the seed's random network and freeze that."""
    source = network(shared=True)
    config = tiny_ppo(epochs=2, freeze=["actor"], critic_targets="lambda", critic_gamma=1.0, gae_lambda=1.0)
    config["model"] = {"preset": "graph-tiny", "args": {"shared": shared}}
    config["init_from"] = {"ckpt": saved(source, tmp_path), **(SPLIT if not shared else {"parts": ["actor", "critic"]})}
    name = f"early-{'shared' if shared else 'split'}"

    def die(self):
        raise KeyboardInterrupt("killed before e0")
    with monkeypatch.context() as patch:
        patch.setattr(puffer.CatanPuffeRL, "save_checkpoint", die)
        with pytest.raises(KeyboardInterrupt):
            train(train_ppo, config, name)
    writer = train(train_ppo, config, name)
    assert writer.status == "finished"
    launches = writer._run_json()["launches"]
    assert [l["mode"] for l in launches] == ["new", "resume"] and launches[-1]["resumed_from"] == -1
    obs = rows(count=8)
    first, last = writer.saved[0], writer.saved[-1]
    assert first.epoch == 0 and writer._run_json()["checkpoints"][0]["launch"] == launches[-1]["launch"]
    assert_same(outputs(model_at(first), obs), outputs(source, obs))
    assert_same(outputs(model_at(last), obs), outputs(source, obs), keys=("logits", "open_logits"))


@pytest.mark.train
def test_a_resume_refuses_a_frozen_actor_that_is_not_init_froms(tmp_path):
    """The committed state of a run that resumed into a random network before the fix (or any
    state whose frozen weights drifted) is refused at the next launch, before it trains."""
    source = network(shared=True)
    config = tiny_ppo(epochs=4, freeze=["actor"], critic_targets="lambda", critic_gamma=1.0, gae_lambda=1.0)
    config["model"] = {"preset": "graph-tiny", "args": {"shared": False}}
    config["init_from"] = {"ckpt": saved(source, tmp_path), **SPLIT}
    writer = train(train_ppo, config, "drifted", stop_after=2)
    assert writer.status == "interrupted"
    path = writer.path / "trainer_state.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    key = next(k for k in state["model"] if k.startswith("actor."))
    state["model"][key] = state["model"][key] + 1.0
    torch.save(state, path)
    with pytest.raises(ValueError, match="differ from init_from's"):
        train(train_ppo, config, "drifted")


def weights(ref) -> dict[str, torch.Tensor]:
    return Checkpoint.load(ref.path).state_dict


def model_at(ref) -> torch.nn.Module:
    return Checkpoint.load(ref.path).model("all")


@pytest.mark.train
@pytest.mark.parametrize("optimizer", ["adam", "muon"])
def test_a_critic_refit_leaves_the_policy_bit_for_bit(tmp_path, optimizer):
    source, writer = critic_refit(tmp_path, optimizer=optimizer)
    first, last = writer.saved[0], writer.saved[-1]
    assert first.epoch == 0 and last.epoch == 4
    obs = rows(count=16)
    # e0 is the split start: the source's logits and values exactly.
    assert_same(outputs(model_at(first), obs), outputs(source, obs))
    start, end = weights(first), weights(last)
    actor = [k for k in start if k.startswith("actor.")]
    critic = [k for k in start if k.startswith("critic.")]
    assert actor and critic and len(actor) + len(critic) == len(start)
    assert all(torch.equal(start[k], end[k]) for k in actor)
    assert sum(not torch.equal(start[k], end[k]) for k in critic) == len(critic)  # every critic tensor moved
    trained = outputs(model_at(last), obs)
    assert_same(trained, outputs(source, obs), keys=("logits", "open_logits"))
    assert not torch.equal(trained["values"], outputs(source, obs)["values"])
    # The optimizer holds the critic's parameters only.
    state = writer.load_trainer_state()
    held = [i for group in state["optimizer"]["param_groups"] for i in group["params"]]
    named = dict(model_at(last).named_parameters())
    assert len(held) == len([n for n in named if n.startswith("critic.")])
    effective = writer._run_json()["launches"][-1]["effective"]["frozen"]
    assert effective == {"parts": ["actor"],
                         "trained_parameters": sum(p.numel() for n, p in named.items() if n.startswith("critic."))}
    # The policy never moved, so the PPO ratio is 1: KL 0, nothing clipped.
    for row in writer.rows:
        assert abs(row["losses/approx_kl"]) < 1e-6 and row["losses/clipfrac"] == 0.0


@pytest.mark.train
def test_a_shared_trunk_refit_trains_the_value_head_alone(tmp_path):
    source, writer = critic_refit(tmp_path, shared=True)
    start, end = weights(writer.saved[0]), weights(writer.saved[-1])
    head = [k for k in start if k.startswith("value_head.")]
    assert head and all(torch.equal(start[k], end[k]) for k in start if k not in head)
    assert all(not torch.equal(start[k], end[k]) for k in head)
    obs = rows(count=16)
    assert_same(outputs(model_at(writer.saved[-1]), obs), outputs(source, obs), keys=("logits", "open_logits"))


@pytest.mark.train
def test_muon_exclude_cannot_name_a_frozen_parameter(tmp_path):
    with pytest.raises(ValueError, match="not the optimizer's"):
        critic_refit(tmp_path, optimizer="muon", epochs=1, muon_exclude=["actor.policy.other.2.weight"],
                     muon_exclude_lr=1e-4)


# ---------------------------------------------------------------- the critic's target


def outcome_or_bootstrap(values: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
    """Per row and step t < H-1: the reward at the first step after t that ends a game (the
    terminal flag arrives with the next game's first observation), else the stored value at
    the segment's last step. The last step's own entry is its value (no advantage)."""
    out = values.clone()
    segments, horizon = values.shape
    for s in range(segments):
        for t in range(horizon - 1):
            ends = torch.nonzero(dones[s, t + 1:]).flatten()
            out[s, t] = rewards[s, t + 1 + ends[0]] if ends.numel() else values[s, horizon - 1]
    return out


def test_the_undiscounted_monte_carlo_target_on_a_segment():
    """critic_gamma 1, gae_lambda 1, every ratio 1 (what critic_targets: lambda hands the value
    loss): seat 0 wins at step 3, the next game is cut at the turn limit at step 6."""
    horizon = 8
    values = torch.rand(2, horizon)
    rewards, dones = torch.zeros(2, horizon), torch.zeros(2, horizon)
    rewards[0, 3] = 1.0                       # seat 0's row: the win
    dones[:, 3] = dones[:, 6] = 1.0           # both rows: the game's end, then a truncation (pays 0)
    advantages = puffer.pufferl.compute_puff_advantage(values, rewards, dones, torch.ones(2, horizon),
                                                       torch.zeros(2, horizon), 1.0, 1.0, 1.0, 1.0)
    returns = advantages + values
    expected = torch.tensor([[1, 1, 1, 0, 0, 0, values[0, 7], values[0, 7]],
                             [0, 0, 0, 0, 0, 0, values[1, 7], values[1, 7]]])
    assert torch.allclose(returns, expected, atol=1e-6)
    assert torch.allclose(returns, outcome_or_bootstrap(values, rewards, dones), atol=1e-6)


@pytest.mark.train
def test_a_refit_trains_on_outcomes_or_the_segment_end(tmp_path, monkeypatch):
    """In a real critic-only run (one scenario cut at a 4-turn limit, so truncations occur):
    every target the value loss reads is the outcome-or-bootstrap above, clamped to [0, 1],
    under binary cross-entropy with the last step of each segment weighted 0."""
    trainers, checked = [], []
    init = puffer.CatanPuffeRL.__init__

    def keep(self, *args, **kwargs):
        trainers.append(self)
        init(self, *args, **kwargs)

    value_loss = PufferPolicy.value_loss

    def hook(self, outputs, returns, *, idx, values, advantages):
        trainer = trainers[0]
        expected = outcome_or_bootstrap(values, trainer.rewards, trainer.terminals)
        assert torch.allclose(advantages + values, expected, atol=1e-5)
        checked.append(int(trainer.terminals.sum()))
        return value_loss(self, outputs, returns, idx=idx, values=values, advantages=advantages)

    monkeypatch.setattr(puffer.CatanPuffeRL, "__init__", keep)
    monkeypatch.setattr(PufferPolicy, "value_loss", hook)
    config = tiny_ppo(epochs=3, freeze=["actor"], critic_targets="lambda", critic_gamma=1.0, gae_lambda=1.0)
    config["model"] = {"preset": "graph-tiny", "args": {"shared": True}}
    config["init_from"] = {"ckpt": saved(network(shared=True), tmp_path), "parts": ["actor", "critic"]}
    config["env"]["scenarios"][0]["turn_limit"] = 4
    writer = train(train_ppo, config, "targets")
    assert writer.status == "finished" and checked and max(checked) > 0
    assert max(row.get("environment/self/truncated", 0.0) for row in writer.rows) > 0


# ---------------------------------------------------------------- the search


def test_mcts_reads_the_critic_networks_values():
    """MCTS (the search harness's spec: Gumbel root, dets 1) over a split network: the batched
    evaluator returns the critic network's values, and a critic that predicts a constant makes
    every root value that constant whatever the actor."""
    split = network(shared=False, level="public")
    search = agent("MCTS", split, infer_rows=8, sims=16, dets=1, root="gumbel", c_puct=0.75, gumbel_m=4)
    obs = rows("public", count=8)
    logits, values = search.evaluator(obs.numpy())
    with torch.no_grad():  # the strict CPU evaluator runs one row at a time
        expected = [split(obs[i:i + 1]) for i in range(obs.shape[0])]
    assert np.array_equal(logits, torch.cat([e[0] for e in expected]).numpy())
    assert np.array_equal(values, torch.cat([e[1] for e in expected]).numpy())
    with torch.no_grad():
        critic_values = torch.sigmoid(split.value_logits(obs)[0]).numpy()
    assert np.allclose(values, critic_values, atol=1e-6)  # the critic network's (2p: every seat present)

    with torch.no_grad():
        last = split.critic.value.mlp[-1]
        last.weight.zero_()
        last.bias.fill_(1.0)  # 2p: logit 1 + logit(1/2) = 1
    game = game_configs.load_game("base-2p").cppanatron_game(3, 3)
    for _ in range(30):
        game.execute(game.playable_actions[0])
    while len(game.playable_actions) < 2:
        game.execute(game.playable_actions[0])
    roots = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        with torch.no_grad():
            for p in split.actor.parameters():
                p.add_(torch.randn_like(p) * 0.3)
        mcts = agent("MCTS", split, infer_rows=8, sims=16, dets=1, root="gumbel", c_puct=0.75, gumbel_m=4)
        pool = make_pool(mcts.dims, mcts.level, mcts.config)
        sid = pool.add(game, seed=7, tracker=cpp.HandTracker(game, game.current_player_index))
        run_searches(pool, mcts.evaluator, batch_rows=mcts.config.max_pending)
        roots.append(pool.value(sid))
    for value in roots:
        assert np.allclose(value, 1.0 / (1.0 + np.exp(-1.0)), atol=1e-6)
