"""A toy n-player race through the real training path, with exact values.

PuffeRL (V-trace, prioritized segments, the acting mask, micro-batches)
trains catanrl's :class:`~catanrl.train.puffer_policy.PufferPolicy` (value
fill, the per-seat value-loss hook, write-back) on a game small enough for
dynamic programming to give every seat's exact discounted win probability.

The game: ``P`` seats in turn order, scores 0..K-1. A turn is 1-3 decisions
(uniform). At each decision the actor picks action 0 or 1: the action equal
to its score's parity scores with probability ``QG``, the other with ``QB``.
The first to ``K`` wins (1 to its row, every row terminal, a new game at once
with a random first seat). The network is two lookup tables (policy logits
and per-slot value logits) indexed by the state relative to the actor.

:func:`run_toy` freezes the uniform policy, starts the critic at the exact
values, and injects PPO-like V-trace ratios into the policy's advantage call
only: on acting steps ``log r = f (beta z + sigma eps)``, ``z`` the step's
standardized plain advantage and ``f`` ramping over the epoch's first half
of micro-batches (PPO's ratios drift away from 1 as the epoch reuses its
samples, and correlate with each sample's own advantage). So the policy is
the same in every variant and the critic's drift is the targets' alone.
"""

from __future__ import annotations

import gymnasium
import numpy as np
import pufferlib
import pufferlib.vector
import torch
from torch import nn

from catanrl._puffer import import_pufferl
from catanrl.train.puffer_policy import PufferPolicy

pufferl = import_pufferl()
QG, QB = 0.16, 0.08


def encode(rel: np.ndarray, turns_left: np.ndarray, k: int) -> np.ndarray:
    """State index of scores relative to the actor (slot 0 first) and the turn's decisions left."""
    idx = np.zeros(rel.shape[0], np.int64)
    for slot in range(rel.shape[1]):
        idx = idx * k + rel[:, slot]
    return idx * 3 + (turns_left - 1)


class ToyEnv(pufferlib.PufferEnv):
    def __init__(self, num_envs: int = 256, players: int = 4, target: int = 6, buf=None, seed: int = 0):
        self.p, self.k = players, target
        self.num_games = num_envs
        self.num_agents = num_envs * players
        self.single_observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(3)
        self.render_mode = None
        super().__init__(buf)
        self.rng = np.random.default_rng(seed)

    def _new(self, games):
        self.scores[games] = 0
        self.cur[games] = self.rng.integers(0, self.p, games.size)
        self.left[games] = self.rng.integers(1, 4, games.size)

    def _write(self):
        n, p = self.num_games, self.p
        obs = self.observations.reshape(n, p, 4)
        obs[:] = 0
        obs[:, :, 3] = 1  # waiting rows: only the no-op
        g = np.arange(n)
        rel = np.stack([self.scores[g, (self.cur + slot) % p] for slot in range(p)], 1)
        obs[g, self.cur, 0] = encode(rel, self.left, self.k)
        obs[g, self.cur, 1:3] = 1
        obs[g, self.cur, 3] = 0

    def reset(self, seed=None):
        n = self.num_games
        self.scores = np.zeros((n, self.p), np.int64)
        self.cur = np.zeros(n, np.int64)
        self.left = np.zeros(n, np.int64)
        self._new(np.arange(n))
        self.rewards[:] = 0
        self.terminals[:] = 0
        self.truncations[:] = 0
        self._write()
        return self.observations, []

    def step(self, actions):
        n, p = self.num_games, self.p
        g = np.arange(n)
        action = np.asarray(actions).reshape(n, p)[g, self.cur]
        scored = self.rng.random(n) < np.where(action == self.scores[g, self.cur] % 2, QG, QB)
        self.scores[g, self.cur] += scored
        self.rewards[:] = 0
        self.terminals[:] = 0
        won = self.scores[g, self.cur] >= self.k
        self.rewards.reshape(n, p)[won, self.cur[won]] = 1
        self.terminals.reshape(n, p)[won] = 1
        self.left -= 1
        passing = ~won & (self.left == 0)
        self.cur[passing] = (self.cur[passing] + 1) % p
        self.left[passing] = self.rng.integers(1, 4, int(passing.sum()))
        self._new(np.nonzero(won)[0])
        self._write()
        return self.observations, self.rewards, self.terminals, self.truncations, []

    def close(self):
        pass


class ToyModel(nn.Module):
    """The CatanModel surface PufferPolicy uses, over two lookup tables."""

    dims_agnostic = True
    mask_offset, noop = 1, 2

    def __init__(self, players: int, target: int):
        super().__init__()
        self.num_players = players
        self.states = target ** players * 3
        self.pi = nn.Embedding(self.states, 2)
        self.v = nn.Embedding(self.states, players)
        nn.init.zeros_(self.pi.weight)
        nn.init.zeros_(self.v.weight)

    def acting(self, obs):
        return obs[:, self.mask_offset + self.noop] == 0

    def waiting_logits(self, rows, device):
        logits = torch.full((rows, 3), -1e9, device=device)
        logits[:, 2] = 0
        return logits

    def heads(self, obs):
        idx = obs[:, 0].long()
        logits = torch.cat((self.pi(idx), torch.full((idx.numel(), 1), -1e9)), 1)
        return logits, self.v(idx), torch.ones(idx.numel(), self.num_players)


def true_values(players: int, target: int, gamma: float, pi1: np.ndarray) -> np.ndarray:
    """V[state, slot]: slot's discounted win probability under the policy (P(action 1) = pi1)."""
    p, k = players, target
    states = k ** p * 3
    s = np.arange(states)
    left = s % 3 + 1
    rest = s // 3
    rel = np.zeros((states, p), np.int64)
    for slot in reversed(range(p)):
        rel[:, slot] = rest % k
        rest //= k
    good = rel[:, 0] % 2 == 1
    q = np.where(good, pi1 * QG + (1 - pi1) * QB, (1 - pi1) * QG + pi1 * QB)
    win = rel[:, 0] + 1 >= k
    scored = rel.copy()
    scored[:, 0] = np.minimum(rel[:, 0] + 1, k - 1)
    same_s, same_f = encode(scored, np.maximum(left - 1, 1), k), encode(rel, np.maximum(left - 1, 1), k)
    next_s = [encode(np.roll(scored, -1, axis=1), np.full(states, t), k) for t in (1, 2, 3)]
    next_f = [encode(np.roll(rel, -1, axis=1), np.full(states, t), k) for t in (1, 2, 3)]
    passing = (left == 1)[:, None]
    won = np.zeros((states, p))
    won[:, 0] = 1
    values = np.zeros((states, p))
    for _ in range(20000):
        shifted = np.roll(values, 1, axis=1)  # the next actor's slot k-1 is the old actor's slot k
        cont_s = np.where(passing, sum(shifted[i] for i in next_s) / 3, values[same_s])
        cont_f = np.where(passing, sum(shifted[i] for i in next_f) / 3, values[same_f])
        new = q[:, None] * np.where(win[:, None], won, gamma * cont_s) + (1 - q[:, None]) * gamma * cont_f
        if np.abs(new - values).max() < 1e-10:
            return new
        values = new
    raise RuntimeError("value iteration did not converge")


class _Quiet(pufferl.PuffeRL):
    def print_dashboard(self, *args, **kwargs):
        pass

    def close(self):  # PuffeRL's saves a checkpoint
        self.vecenv.close()


class _NoLogger:
    run_id = "toy"

    def log(self, logs, step):
        pass

    def close(self, *args, **kwargs):
        pass


def run_toy(*, critic_targets: str, critic_gamma: float | None = None, epochs: int = 12, gamma: float = 0.999,
            init_gamma: float | None = None, beta: float = 0.1, sigma: float = 0.1, num_envs: int = 256,
            players: int = 4, target: int = 6, seed: int = 0, measure_epochs: int = 4) -> dict:
    """Train the critic for ``epochs`` (policy frozen, uniform) from the exact values at
    ``init_gamma`` (default: the critic's discount); return the critic's excess over the
    exact values (at the critic's discount) on the decisions of the last ``measure_epochs``
    rollouts, and the logged critic metrics per epoch."""
    torch.manual_seed(seed)
    critic = gamma if critic_gamma is None else critic_gamma
    vecenv = pufferlib.vector.make(ToyEnv, env_kwargs={"num_envs": num_envs, "players": players, "target": target},
                                   num_envs=1, backend=pufferlib.vector.Serial, seed=seed)
    model = ToyModel(players, target)
    model.pi.weight.requires_grad_(False)  # uniform, frozen
    pi1 = np.full(model.states, 0.5)
    start = true_values(players, target, critic if init_gamma is None else init_gamma, pi1)
    truth = true_values(players, target, critic, pi1)
    with torch.no_grad():
        model.v.weight.copy_(torch.logit(torch.as_tensor(start, dtype=torch.float32).clamp(1e-6, 1 - 1e-6)))
    policy = PufferPolicy(model, num_rows=vecenv.num_agents, critic_gamma=critic)
    config = dict(
        adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8, amp=False, anneal_lr=False, bptt_horizon=64,
        checkpoint_interval=10 ** 9, clip_coef=0.2, compile=False, compile_mode="default", cpu_offload=False,
        critic_gamma=critic_gamma, critic_targets=critic_targets, data_dir="/nonexistent", device="cpu",
        ent_coef=0.0, env="toy", gae_lambda=0.95, gamma=gamma, learning_rate=0.02, max_grad_norm=4.0,
        max_minibatch_size=4096, min_lr_ratio=0.1, minibatch_size=16384, optimizer="adam", precision="float32",
        prio_alpha=0.8, prio_beta0=0.2, seed=seed, torch_deterministic=False, total_timesteps=10 ** 12,
        update_epochs=2, use_rnn=False, vf_clip_coef=0.2, vf_coef=2.0, vtrace_c_clip=1.0, vtrace_rho_clip=1.0,
        batch_size=vecenv.num_agents * 64,
    )
    trainer = _Quiet(config, vecenv, policy, logger=_NoLogger())
    trainer.utilization.stop()

    original = pufferl.compute_puff_advantage
    calls = {"policy": 0}
    noise: dict[str, torch.Tensor] = {}

    def injected(values, rewards, dones, ratio, advantages, g, lam, rho, c):
        if ratio is not trainer.ratio:  # not the policy's V-trace call
            return original(values, rewards, dones, ratio, advantages, g, lam, rho, c)
        i = calls["policy"] % trainer.total_minibatches
        calls["policy"] += 1
        plain = original(values, rewards, dones, torch.ones_like(ratio), torch.zeros_like(advantages), g, lam, rho, c)
        acting = trainer.observations[..., 3] == 0
        if i == 0:
            noise["eps"] = torch.randn(ratio.shape)
        z = (plain - plain[acting].mean()) / (plain[acting].std() + 1e-8)
        f = min(1.0, i / (trainer.total_minibatches / 2))
        rr = torch.where(acting, torch.exp(f * (beta * z + sigma * noise["eps"])), torch.ones_like(ratio))
        return original(values, rewards, dones, rr, advantages, g, lam, rho, c)

    pufferl.compute_puff_advantage = injected
    log, visits = [], np.zeros(model.states)
    try:
        for epoch in range(epochs):
            trainer.evaluate()
            if epoch >= epochs - measure_epochs:
                obs = trainer.observations.reshape(-1, 4)
                np.add.at(visits, obs[obs[:, 3] == 0, 0].long().numpy(), 1)
            policy.snapshot_values(trainer.values)
            trainer.train()
            log.append(policy.pop_calibration())
    finally:
        pufferl.compute_puff_advantage = original
        trainer.close()
    with torch.no_grad():
        pred = torch.sigmoid(model.v.weight).numpy()
    w = visits / visits.sum()
    return {
        "excess": float((w * (pred.sum(1) - truth.sum(1))).sum()),
        "excess_slot": (w[:, None] * (pred - truth)).sum(0).tolist(),
        "true_sum": float((w * truth.sum(1)).sum()),
        "log": log,
    }
