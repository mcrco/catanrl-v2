"""The PufferLib env over VecEnv and the policy adapter's value bookkeeping."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from catanrl.envs.encoding import action_size, row_layout
from catanrl.envs.puffer import make_vecenv, plan_env, scenario_specs
from catanrl.models import CatanModel, build_model
from catanrl.train.puffer_policy import PufferPolicy, ValueBatch, seat_rows


def small_plan(*entries, maps=4, seed=1):
    return plan_env({"scenarios": [{**e, "maps": maps} for e in entries]}, pool_seed=seed)


def test_map_pool_is_seeded_and_recorded():
    a = small_plan({"game": "base-2p"}, {"game": "random-4p"})
    b = small_plan({"game": "base-2p"}, {"game": "random-4p"})
    c = small_plan({"game": "base-2p"}, {"game": "random-4p"}, seed=2)
    assert a.pool == b.pool and a.pool["hash"] != c.pool["hash"]
    assert [len(s["maps"]) for s in a.pool["scenarios"]] == [4, 4]
    assert a.dims["max_players"] == 4


def test_scenario_specs_refuse_unknown_keys_and_bad_seats():
    with pytest.raises(ValueError):
        scenario_specs([{"game": "base-2p", "mapz": 3}])
    with pytest.raises(ValueError):
        scenario_specs([{"game": "base-2p", "seats": ["learner"]}])
    with pytest.raises(ValueError):
        scenario_specs([{"game": "base-2p", "seats": ["F", "R"]}])
    names = [s.name for s in scenario_specs(["base-2p", "base-2p"])]
    assert names == ["base-2p", "base-2p.1"]


def test_env_binds_puffer_buffers_zero_copy():
    plan = small_plan({"game": "base-4p", "seats": ["learner", "F", "learner", "R"]})
    vec = make_vecenv(plan, num_envs=6, level="private", critic_level=None, threads=2, seed=5)
    env = vec.envs[0]
    assert vec.num_agents == 24 and np.shares_memory(env.observations, vec.observations)
    vec.async_reset(5)
    obs, *_ = vec.recv()
    mask = obs[:, -vec.single_action_space.n :]
    acting = mask[:, -1] == 0
    assert np.array_equal(acting, env.acting.astype(bool))
    assert acting.reshape(6, 4).sum(1).tolist() == [1] * 6  # one acting row per env
    rng = np.random.default_rng(0)
    for _ in range(200):
        mask = obs[:, -vec.single_action_space.n :]
        vec.send((rng.random(mask.shape) * mask).argmax(1).astype(np.int32))
        obs, *_ = vec.recv()
    assert env.decisions > 0


class SlotModel(CatanModel):
    """Every acting row predicts (k + 1) / 10 for seat slot k."""

    dims_agnostic = True

    def __init__(self, dims, num_actions):
        level = "private"
        super().__init__({"class": "stub"}, dims, {"level": level, "critic_level": level},
                         row_layout(dims, level))
        players_offset, (_, width) = self.layout["players"]
        self.players = (players_offset, width)
        self.dummy = torch.nn.Parameter(torch.zeros(()))

    def heads(self, obs):
        offset, width = self.players
        players = obs[:, offset : offset + self.num_players * width].reshape(-1, self.num_players, width)
        present = players[..., 0]
        probs = (torch.arange(self.num_players, dtype=torch.float32) + 1.0) / 10.0
        logits = torch.logit(probs).expand(obs.shape[0], -1) + 0 * self.dummy
        mask = obs[:, self.mask_offset : self.mask_offset + self.num_actions]
        return torch.where(mask > 0, 0.0, -1e9), logits, present


def test_waiting_rows_take_the_acting_rows_prediction_for_their_seat():
    plan = small_plan({"game": "base-4p", "seats": ["learner", "F", "learner", "R"]},
                      {"game": "base-2p"})
    vec = make_vecenv(plan, num_envs=8, level="private", critic_level=None, threads=2, seed=3)
    env = vec.envs[0]
    policy = PufferPolicy(SlotModel(plan.dims, vec.single_action_space.n), num_rows=vec.num_agents)
    vec.async_reset(3)
    for _ in range(30):
        obs, *_ = vec.recv()
        logits, values = policy.forward_eval(torch.from_numpy(obs.copy()), {"env_id": slice(0, vec.num_agents)})
        for e in range(8):
            n = int(env.num_players[e])
            rows = slice(e * 4, e * 4 + 4)
            actor = int(np.flatnonzero(env.acting[rows])[0])
            for seat in range(4):
                expected = 0.0 if seat >= n else ((seat - actor) % n + 1) / 10.0
                assert values[e * 4 + seat].item() == pytest.approx(expected, abs=1e-6)
        waiting = env.acting == 0
        assert torch.all(logits[torch.from_numpy(waiting)][:, -1] == 0)
        vec.send(torch.distributions.Categorical(logits=logits).sample().numpy().astype(np.int32))
    calibration = policy.pop_calibration()
    assert calibration["value_sum_2p"] == pytest.approx(0.3)
    assert calibration["value_sum_4p"] == pytest.approx(1.0)


def test_seat_rows_wrap_around_the_table():
    rows = torch.tensor([4 * 2 + 3, 4 * 0 + 1])
    present = torch.tensor([[1.0, 1, 1, 1], [1, 1, 0, 0]])
    target, valid = seat_rows(rows, present, 4)
    assert target[0].tolist() == [11, 8, 9, 10]
    assert target[1][:2].tolist() == [1, 0] and valid[1].tolist() == [True, True, False, False]


def test_value_loss_targets_partner_rows_and_writes_back_values():
    """Slot k of an acting row is trained on seat (actor + k)'s return at the same step."""
    torch.manual_seed(0)
    num_envs, players, horizon = 3, 2, 5
    rows = num_envs * players
    values = torch.rand(rows, horizon)
    advantages = torch.rand(rows, horizon) * 0.2 - 0.1
    returns = (advantages + values).clamp(0, 1)
    policy = PufferPolicy.__new__(PufferPolicy)
    torch.nn.Module.__init__(policy)
    policy.max_players, policy.num_rows = players, rows
    idx = torch.tensor([3, 0])          # segments (= rows) in the minibatch
    # Minibatch row f = local_segment * horizon + t. Row 3 (env 1, seat 1) acts at t = 1;
    # row 0 (env 0, seat 0) acts at t = 2 and t = 4 (the last step: no target).
    index = torch.tensor([0 * horizon + 1, 1 * horizon + 2, 1 * horizon + 4])
    logits = torch.zeros(3, players, requires_grad=True)
    present = torch.ones(3, players)
    outputs = ValueBatch(index, logits, present, rows=2 * horizon)
    newvalue, loss = policy.value_loss(outputs, returns.flatten(), idx=idx, values=values.clone(),
                                       advantages=advantages)
    loss.backward()
    grad = logits.grad
    # BCE gradient is (p - target) / count with p = 0.5: recover the targets.
    count = 4  # two acting steps x two seats (the last step is excluded)
    targets = 0.5 - grad * count
    assert targets[0].tolist() == pytest.approx([returns[3, 1].item(), returns[2, 1].item()], abs=1e-5)
    assert targets[1].tolist() == pytest.approx([returns[0, 2].item(), returns[1, 2].item()], abs=1e-5)
    assert torch.all(grad[2] == 0)
    stored = newvalue.view(2, horizon)
    assert stored[0, 1].item() == pytest.approx(0.5) and stored[1, 2].item() == pytest.approx(0.5)


def test_policy_trains_only_on_acting_rows():
    plan = small_plan({"game": "base-2p"})
    vec = make_vecenv(plan, num_envs=4, level="private", critic_level=None, threads=1, seed=2)
    model = build_model({"preset": "graph-tiny"}, dims=plan.dims, levels="private",
                        num_actions=action_size(plan.dims), layout=row_layout(plan.dims, "private"))
    policy = PufferPolicy(model, num_rows=vec.num_agents)
    vec.async_reset(2)
    obs, *_ = vec.recv()
    logits, outputs = policy(torch.from_numpy(obs.copy()))
    assert outputs.index.tolist() == np.flatnonzero(vec.envs[0].acting).tolist()
    waiting = torch.from_numpy(vec.envs[0].acting == 0)
    probs = torch.softmax(logits[waiting], dim=1)
    assert torch.all(probs[:, -1] == 1)


def test_critic_level_rows_match_the_model():
    """obs.critic_level above obs.level: VecEnv appends critic_players and the model reads it."""
    plan = small_plan({"game": "base-2p"})
    vec = make_vecenv(plan, num_envs=4, level="private", critic_level="full", threads=1, seed=4)
    layout = row_layout(plan.dims, "private", "full")
    assert vec.single_observation_space.shape[0] == layout["critic_players"][0] + 2 * 46
    model = build_model({"preset": "graph-tiny"}, dims=plan.dims,
                        levels={"level": "private", "critic_level": "full"},
                        num_actions=action_size(plan.dims), layout=layout)
    policy = PufferPolicy(model, num_rows=vec.num_agents)
    vec.async_reset(4)
    obs, *_ = vec.recv()
    with torch.no_grad():
        _logits, values = policy.forward_eval(torch.from_numpy(obs.copy()))
    assert values.shape == (8,) and torch.all((values > 0) & (values < 1))


@pytest.mark.parametrize("maps", [0, -3, 2.5, True, "16"])
def test_map_pools_must_hold_maps(maps):
    """An empty pool would silently train on BASE boards instead of the game's generator."""
    with pytest.raises(ValueError, match="maps"):
        scenario_specs([{"game": "random-4p", "maps": maps}])
