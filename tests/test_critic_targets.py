"""train.critic_targets and critic_gamma (PufferLib's catanrl-v2 patch in PuffeRL.train).

* ``vtrace`` (the default) hands the value-loss hook the policy's V-trace
  advantages; ``lambda`` hands it plain GAE lambda-returns from the same stored
  values (every ratio 1, clips 1) at ``critic_gamma`` (default gamma), while
  the policy's advantages stay V-trace's.
* On a toy race with exact values (tests/critic_toy.py), advantage-correlated
  V-trace ratios inflate the per-seat critic's seat sum under ``vtrace`` and
  not under ``lambda`` (the critic4p diagnosis); ``critic_gamma: 1.0`` trains
  toward undiscounted win probabilities.
* The options are checked before training (with muon_exclude's).
"""

from __future__ import annotations

import numpy as np
import pytest
from run_helpers import train
from train_configs import tiny_ppo

from catanrl.train import puffer
from catanrl.train.puffer import _critic_options
from catanrl.train.puffer_policy import PufferPolicy


def gae(values, rewards, dones, gamma: float, lam: float) -> np.ndarray:
    """Plain GAE along each row; the last step's advantage is 0 (PufferLib's kernel)."""
    values, rewards, dones = (np.asarray(x, np.float64) for x in (values, rewards, dones))
    adv = np.zeros_like(values)
    last = np.zeros(values.shape[0])
    for t in range(values.shape[1] - 2, -1, -1):
        alive = 1.0 - dones[:, t + 1]
        delta = rewards[:, t + 1] + gamma * values[:, t + 1] * alive - values[:, t]
        last = delta + gamma * lam * last * alive
        adv[:, t] = last
    return adv


def spied_training(monkeypatch, name: str, **train_keys):
    """Every advantage call (its inputs at call time) and what the value-loss hook received."""
    calls, hooked, trainers = [], [], []
    original = puffer.pufferl.compute_puff_advantage

    def record(values, rewards, dones, ratio, advantages, gamma, lam, rho, c):
        inputs = [x.clone() for x in (values, rewards, dones, ratio)]
        out = original(values, rewards, dones, ratio, advantages, gamma, lam, rho, c)
        calls.append({"inputs": inputs, "trainer_ratio": ratio is trainers[0].ratio, "gamma": gamma, "lam": lam,
                      "rho": rho, "c": c, "out": out})
        return out

    value_loss = PufferPolicy.value_loss

    def hook(self, outputs, returns, *, idx, values, advantages):
        hooked.append((len(calls), advantages))
        return value_loss(self, outputs, returns, idx=idx, values=values, advantages=advantages)

    init = puffer.CatanPuffeRL.__init__

    def keep(self, *args, **kwargs):
        trainers.append(self)
        init(self, *args, **kwargs)

    monkeypatch.setattr(puffer.pufferl, "compute_puff_advantage", record)
    monkeypatch.setattr(PufferPolicy, "value_loss", hook)
    monkeypatch.setattr(puffer.CatanPuffeRL, "__init__", keep)
    writer = train(puffer.train_ppo, tiny_ppo(epochs=2, gamma=0.99, gae_lambda=0.9, update_epochs=2, **train_keys),
                   name)
    assert writer.status == "finished" and hooked
    return calls, hooked


@pytest.mark.train
def test_lambda_targets_are_plain_gae_at_the_critic_discount(monkeypatch):
    calls, hooked = spied_training(monkeypatch, "lam", critic_targets="lambda", critic_gamma=0.9)
    assert len(calls) == 2 * len(hooked)
    for n, advantages in hooked:
        policy_call, critic_call = calls[n - 2], calls[n - 1]
        assert policy_call["trainer_ratio"] and policy_call["gamma"] == 0.99  # V-trace, as before
        assert advantages is critic_call["out"]
        values, rewards, dones, ratio = critic_call["inputs"]
        assert not critic_call["trainer_ratio"] and bool((ratio == 1).all())
        assert (critic_call["gamma"], critic_call["lam"], critic_call["rho"], critic_call["c"]) == (0.9, 0.9, 1.0, 1.0)
        assert np.allclose(advantages.numpy(), gae(values, rewards, dones, 0.9, 0.9), atol=1e-5)
    # the stored values change between micro-batches (write-back), and so do the targets
    assert not np.allclose(hooked[0][1].numpy(), hooked[-1][1].numpy())


@pytest.mark.train
def test_vtrace_targets_are_the_policy_advantages(monkeypatch):
    calls, hooked = spied_training(monkeypatch, "vt")
    assert len(calls) == len(hooked)
    for n, advantages in hooked:
        assert calls[n - 1]["trainer_ratio"] and advantages is calls[n - 1]["out"]


@pytest.mark.train
def test_lambda_targets_remove_the_toy_excess():
    """Critic started at the exact values, policy frozen: with every ratio 1 both target
    modes are the same training (the control; its small drift is the toy's fresh Adam on a
    lookup table). Injected PPO-like ratios push the seat sum up under V-trace targets and
    leave lambda targets where the control is. The logged truth_sum matches the exact sum."""
    from critic_toy import run_toy

    control = run_toy(critic_targets="lambda", beta=0.0, sigma=0.0)
    assert run_toy(critic_targets="vtrace", beta=0.0, sigma=0.0)["excess"] == control["excess"]
    vtrace = run_toy(critic_targets="vtrace")
    fixed = run_toy(critic_targets="lambda")
    assert vtrace["excess"] - control["excess"] > 0.05            # measured +0.094 in 12 epochs
    assert abs(fixed["excess"] - control["excess"]) < 0.005       # measured 0.0001
    assert vtrace["log"][-1]["value_sum_4p"] - fixed["log"][-1]["value_sum_4p"] > 0.05
    truth = np.mean([row["truth_sum_4p"] for row in fixed["log"][-4:]])
    assert truth == pytest.approx(fixed["true_sum"], abs=0.01)


@pytest.mark.train
def test_critic_gamma_one_trains_toward_undiscounted_wins():
    """From the exact discounted values, critic_gamma 1.0 raises the seat sum toward 1 (every
    toy game has a winner), and truth_sum, now undiscounted, is exactly 1."""
    from critic_toy import run_toy

    discounted = run_toy(critic_targets="lambda")
    undiscounted = run_toy(critic_targets="lambda", critic_gamma=1.0, init_gamma=0.999)
    assert undiscounted["true_sum"] == pytest.approx(1.0, abs=1e-6)
    assert all(row["truth_sum_4p"] == pytest.approx(1.0) for row in undiscounted["log"])
    gain = undiscounted["log"][-1]["value_sum_4p"] - discounted["log"][-1]["value_sum_4p"]
    assert gain > 0.015                                            # measured +0.029 in 12 epochs
    assert all(row["target_sum_4p"] > row["value_sum_4p"] for row in undiscounted["log"])


def test_critic_options():
    base = {"learning_rate": 0.005, "gamma": 0.999}
    assert _critic_options(base) == {"critic_targets": "vtrace", "critic_gamma": None, "muon_exclude": [],
                                     "muon_exclude_lr": None}
    options = _critic_options({**base, "critic_targets": "lambda", "critic_gamma": "1",
                               "muon_exclude": ["value_head.mlp.4.weight"], "muon_exclude_lr": "5e-5"})
    assert options["critic_gamma"] == 1.0 and options["muon_exclude_lr"] == 5e-5
    for bad in ({"critic_targets": "retrace"}, {"critic_gamma": 1.0}, {"critic_targets": "lambda", "critic_gamma": 0},
                {"muon_exclude": ["x"]}, {"muon_exclude_lr": 1e-4}, {"muon_exclude": "x", "muon_exclude_lr": 1e-4}):
        with pytest.raises(ValueError):
            _critic_options({**base, **bad})
