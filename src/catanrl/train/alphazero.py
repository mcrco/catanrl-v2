"""AlphaZero / expert iteration on cppanatron (DESIGN 1.6).

``train_alphazero(variant, writer)`` alternates, in one process:

1. **Self-play.** A VecEnv (auto-reset, the ``env`` scenarios as in PPO:
   mixed maps and player counts) plays games; every acting row is searched
   with a SearchPool, all searches in lockstep with their leaves evaluated in
   shared GPU batches (the eval's BatchEvaluator and CUDA graphs; with
   ``kernels: fused`` on the fused CUDA kernels, rebuilt from the weights
   after every learning step, which they copy). Search is
   on the true game (``dets: 0``, fast) or on worlds sampled from the deciding
   seat's hand tracker (``dets: K``, fair; ``VecEnv(trackers=True)``). The
   move played is sampled from the search policy at a temperature that decays
   over the first ``temperature_moves`` decisions of each game (the engine's
   ``select``: visits for PUCT roots, the improved policy for Gumbel roots,
   which also add Gumbel noise themselves; PUCT roots can add Dirichlet noise).
   Playout cap randomization (KataGo): a ``full_search_prob`` fraction of
   moves get ``sims`` simulations and become policy targets, the rest get
   ``fast_sims``, without the Dirichlet noise, and serve as value targets
   only, or with ``record_fast: false`` are played but not stored (KataGo
   records its full searches only). A search sends up to ``pending``
   (``fast_pending``) leaves to the network per round, counted as visits
   while they wait; at 1 a step takes ``sims + 1`` rounds of one leaf a search.
   ``runtime.search_groups`` (unset: 2) splits a step's searches into groups
   that take turns on the GPU while the engine collects and applies the
   others' leaves (:mod:`catanrl.search.pipeline`); 1 is the serial loop.
2. **Targets.** Policy: the search's ``policy_target`` (the visit
   distribution, or Gumbel's improved policy), over legal actions; or, with
   ``policy_target: gumbel`` (Gumbel roots), Gumbel's improved policy rebuilt
   from the same search's root statistics and the network's outputs at its
   root with the target's own sigma constants (:func:`gumbel_target`), e.g.
   gentler than the engine's, which at 2048 simulations puts the best-valued
   action about 100 logits up whatever its lead in Q, and optionally a
   temperature on its prior term (``target_prior_temperature``: a sharp prior
   otherwise keeps a move the deeper search plays near zero). The search
   itself (Sequential Halving, the move played) keeps the engine's sigma and
   priors either way. With ``policy_target: pruned`` (PUCT roots) it is the
   visit distribution without the visits that the root's Dirichlet noise
   forced (:func:`pruned_visits`): an action keeps the visits its value
   earns under the network's own prior, so the noise still shapes the search
   and the move played but is not taught back to the network.
   Value: each seat's outcome, seat-relative like the value head (slot k =
   the seat k places after the row's), 1 to the winner and 0 to the others;
   a truncated game gives every seat 1/n (the eval convention) or 0
   (``truncation_value: zero``).
3. **Replay.** A FIFO of ``replay_capacity`` positions sampled uniformly:
   float16 rows (exact for the integer topology blocks, about 1e-3 relative
   on features), float16 policy targets, float32 value targets. On a 2p BASE
   graph-s row (2,564 floats) a position takes about 5.8 KB: 200k positions
   are about 1.2 GB of RAM, and the trainer state (the buffer included, for
   an exact resume) as much on disk.
4. **Learning.** ``updates_per_iteration`` Adam(W) steps on minibatches:
   cross-entropy of the network's policy to the policy targets plus
   ``value_coef`` times each seated slot's binary cross-entropy to the value
   targets, weight decay as L2. The network is any ``build_model`` network,
   so PPO and DAgger checkpoints warm-start it through ``init_from``.

Checkpoints, select evals and ``best`` follow ``train_ppo`` (one epoch = one
iteration). A resume restores the weights, optimizer, replay buffer, counters
and RNG; games in flight restart.

Every iteration also logs :class:`Monitors` (``oos/*``, ``search/overrule``
and friends, ``target/*`` with ``policy_target: gumbel`` or ``pruned``,
``drift/*``):
logging-only health checks that read the network, the searched rows and
replay and feed nothing back.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import cppanatron as cpp
import numpy as np
import torch
import torch.nn.functional as F

from catanrl.envs.encoding import critic_level_of, encoder_dims, info_level
from catanrl.envs.puffer import plan_env
from catanrl.models import CatanModel, feature_names
from catanrl.search import BatchEvaluator, make_pool, run_pools, search_config
from catanrl.search.pipeline import Group, Pipeline, run_groups
from catanrl.seeding import derive_seed, determinism, rng_state, seed_everything, set_rng_state
from catanrl.train.puffer import build_policy_model, check_bindable, resolve_device, starts_fresh
from catanrl.train.select import Selector

#: The ``train`` keys AlphaZero reads (configs/train/alphazero.yaml has the same values).
ALPHAZERO_KEYS: dict[str, Any] = {
    "algo": "alphazero",
    "iterations": 100,
    "steps_per_iteration": 32,
    "root": "gumbel",
    "sims": 64,
    "fast_sims": 16,
    "pending": 1,
    "fast_pending": 1,
    "full_search_prob": 0.25,
    "record_fast": True,
    "dets": 0,
    "c_puct": 0.75,
    "gumbel_m": 16,
    "dirichlet_noise": True,
    "temperature_start": 1.0,
    "temperature_end": 0.0,
    "temperature_moves": 30,
    "truncation_value": "share",
    "policy_target": "engine",
    "target_c_visit": 50.0,
    "target_c_scale": 0.1,
    "target_rescale": True,
    "target_prior_temperature": 1.0,
    "replay_capacity": 200_000,
    "min_replay": 5_000,
    "updates_per_iteration": 64,
    "minibatch_size": 512,
    "learning_rate": 1.0e-4,
    "weight_decay": 1.0e-4,
    "value_coef": 1.0,
    "max_grad_norm": 1.0,
    "search_rows": 512,
    "kernels": "torch",
    "checkpoint_interval": 10,
}

#: ``runtime.search_groups`` when a config leaves it unset: self-play's searches in two groups
#: that take turns on the device (SelfPlay; 1 is the serial loop, run_pools).
SEARCH_GROUPS = 2


def check_train(train: Mapping[str, Any]) -> dict[str, Any]:
    missing = sorted(set(ALPHAZERO_KEYS) - set(train))
    unknown = sorted(set(train) - set(ALPHAZERO_KEYS))
    if missing or unknown:
        raise ValueError(f"alphazero train keys: missing {missing}, unknown {unknown}")
    train = dict(train)
    if train["root"] not in ("puct", "gumbel"):
        raise ValueError(f"train.root must be puct or gumbel, not {train['root']!r}")
    if train["truncation_value"] not in ("share", "zero"):
        raise ValueError(f"train.truncation_value must be share or zero, not {train['truncation_value']!r}")
    if not 0.0 < float(train["full_search_prob"]) <= 1.0:
        raise ValueError("train.full_search_prob must be in (0, 1]")
    if int(train["sims"]) < 1 or int(train["fast_sims"]) < 1 or int(train["dets"]) < 0:
        raise ValueError("train.sims and train.fast_sims must be positive and train.dets non-negative")
    if int(train["pending"]) < 1 or int(train["fast_pending"]) < 1:
        raise ValueError("train.pending and train.fast_pending count a search's leaves per network round (>= 1)")
    if train["kernels"] not in ("torch", "fused"):
        raise ValueError(f"train.kernels must be torch or fused, not {train['kernels']!r}")
    if train["policy_target"] not in ("engine", "gumbel", "pruned"):
        raise ValueError(f"train.policy_target must be engine, gumbel or pruned, not {train['policy_target']!r}")
    if train["policy_target"] == "gumbel" and train["root"] != "gumbel":
        raise ValueError("train.policy_target: gumbel rebuilds a Gumbel root's improved policy; a PUCT root's "
                         "target is its visits (policy_target: engine or pruned)")
    if train["policy_target"] == "pruned" and train["root"] != "puct":
        raise ValueError("train.policy_target: pruned prunes a PUCT root's visits by its PUCT scores; a Gumbel "
                         "root's target is its improved policy (policy_target: engine or gumbel)")
    if train["policy_target"] == "pruned" and int(train["dets"]) > 1:
        raise ValueError("train.policy_target: pruned reads one root world's prior, visits and Q; with dets > 1 "
                         "every world's tree runs PUCT on its own and the pool reports their sums (dets 0 or 1)")
    if not (float(train["target_c_visit"]) >= 0.0 and float(train["target_c_scale"]) >= 0.0):
        raise ValueError("train.target_c_visit and train.target_c_scale must be non-negative")
    if not float(train["target_prior_temperature"]) > 0.0:
        raise ValueError("train.target_prior_temperature must be positive")
    constants = ("target_c_visit", "target_c_scale", "target_rescale", "target_prior_temperature")
    if train["policy_target"] != "gumbel" and any(train[k] != ALPHAZERO_KEYS[k] for k in constants):
        raise ValueError(f"train.{', '.join(constants)} shape policy_target: gumbel; the engine's target "
                         "has the search's own sigma and a pruned one has none (set policy_target: gumbel)")
    return train


def temperature(train: Mapping[str, Any], ply: int) -> float:
    """Linear from temperature_start at a game's first decision to temperature_end
    after temperature_moves decisions."""
    moves = int(train["temperature_moves"])
    start, end = float(train["temperature_start"]), float(train["temperature_end"])
    if moves <= 0 or ply >= moves:
        return end
    return start + (end - start) * ply / moves


def value_targets(winner: int | None, seat: int, num_players: int, max_players: int,
                  truncation: str) -> tuple[np.ndarray, np.ndarray]:
    """Seat-relative outcome targets of a row at ``seat`` and the slots present.

    Slot k is the seat k places after ``seat`` (the value head's order). A
    truncated game (``winner`` None) gives every seat 1/n (``share``) or 0.
    """
    targets = np.zeros(max_players, dtype=np.float32)
    present = np.zeros(max_players, dtype=np.float32)
    present[:num_players] = 1.0
    if winner is None:
        if truncation == "share":
            targets[:num_players] = 1.0 / num_players
    else:
        targets[(winner - seat) % num_players] = 1.0
    return targets, present


#: The engine's floor on a Gumbel root's log-prior (kMinLogPrior, search_pool.cpp).
MIN_LOG_PRIOR = -69.0


def _log_softmax(scores: np.ndarray) -> np.ndarray:
    top = scores.max(-1, keepdims=True)
    return scores - top - np.log(np.exp(scores - top).sum(-1, keepdims=True))


def _world_log_priors(logits: np.ndarray, edges: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Each root world's log-prior over the root's actions ``edges`` ([W, E], from logits [A]
    or [W, A]) as the engine's expand builds a node's priors, before any Dirichlet noise: a
    softmax of ``logits / temperature`` over those actions, in which non-finite logits get
    nothing; a world without a finite one is uniform. Float64 (the engine stores float32)."""
    z = np.atleast_2d(np.asarray(logits, dtype=np.float64))[:, edges] / float(temperature)
    finite = np.isfinite(z)
    with np.errstate(invalid="ignore"):  # a world without a finite logit: -inf - -inf, replaced below
        log_prior = _log_softmax(np.where(finite, z, -np.inf))
    log_prior[~finite.any(1)] = -np.log(len(edges))
    return log_prior


def gumbel_target(logits: np.ndarray, legal: np.ndarray, visits: np.ndarray, q: np.ndarray, root_value: float,
                  *, c_visit: float = 50.0, c_scale: float = 0.1, rescale: bool = True,
                  prior_temperature: float = 1.0) -> np.ndarray:
    """Gumbel's improved policy at a searched root, softmax(log prior + sigma(completed Q)),
    over the codec's index space (float64 [A], zero off ``legal``), built as the engine's
    ``policy_target`` builds it (search_pool.cpp: expand, prepare_root, gumbel_sigma) but
    with these sigma constants. At the defaults, the engine's (``SearchConfig``'s
    gumbel_c_visit, gumbel_c_scale and gumbel_rescale_values, which the trainer's searches
    keep, and no temperature), it is the engine's target up to the engine's float32 priors
    and output.

    * ``logits``: the network's policy logits at the root, [A] or one row per root world
      [W, A]. Each world's prior is a softmax over ``legal`` (non-finite logits get none; a
      world without a finite one is uniform), its log floored at :data:`MIN_LOG_PRIOR`, and
      the root's log-prior is the worlds' mean. ``prior_temperature`` T divides the logits
      first, so the prior is softmax(logits / T) (and weights v_mix as such); T = 1 is exact.
    * ``legal``: the root's actions (bool [A]); ``visits`` and ``q``: the pool's
      ``visits(id)`` and ``q(id)`` (NaN where unvisited, never read there).
    * ``root_value``: the network's value of the root for its player (each world's clamped
      to [0, 1], then averaged), which completes an unvisited action's Q as v_mix: the root
      value and the prior-weighted mean Q of the visited actions, weighted 1 and their total
      visits.

    Completed Q is min-max rescaled over ``legal`` (``rescale``; else clamped to [0, 1]),
    and sigma = (``c_visit`` + the most visits of an action) * ``c_scale`` * that. The
    arithmetic is float64, the softmaxes in log space.
    """
    legal = np.asarray(legal, dtype=bool)
    target = np.zeros(legal.shape, dtype=np.float64)
    edges = np.flatnonzero(legal)
    if not len(edges):
        return target
    log_prior = np.maximum(_world_log_priors(logits, edges, prior_temperature), MIN_LOG_PRIOR).mean(0)
    n = np.asarray(visits, dtype=np.float64)[edges]
    seen = n > 0
    completed = np.full(len(edges), float(root_value))
    if seen.any():
        weight = np.maximum(np.exp(_log_softmax(log_prior))[seen], np.finfo(np.float64).tiny)
        values = np.asarray(q, dtype=np.float64)[edges][seen]
        total = n.sum()
        completed[:] = (float(root_value) + total * (weight * values).sum() / weight.sum()) / (total + 1.0)
        completed[seen] = values
    if rescale:
        low = completed.min()
        completed = (completed - low) / max(completed.max() - low, 1e-8)
    else:
        completed = np.clip(completed, 0.0, 1.0)
    target[edges] = np.exp(_log_softmax(log_prior + (c_visit + n.max()) * c_scale * completed))
    return target


def pruned_visits(logits: np.ndarray, legal: np.ndarray, visits: np.ndarray, q: np.ndarray, *,
                  c_puct: float) -> np.ndarray:
    """A searched PUCT root's visit counts without the visits its Dirichlet noise forced, over
    the codec's index space (float64 [A], zero off ``legal``); ``policy_target: pruned``'s
    target is their distribution. The idea is KataGo's pruning of forced playouts from its
    policy targets: the noise still decides where the search looks and which move is played,
    but an action keeps no more visits than its value earns under the network's own prior.

    With N and Q the root's visits and values, P the network's prior without the noise and T
    the root's visits in all, an action at n visits has the engine's PUCT score
    (search_pool.cpp, select_edge) ``Q + c_puct * P * sqrt(T + 1) / (1 + n)``.

    * The most visited action keeps its visits, and its score at them is the bar S. Ties go
      to the larger P, then to the first in the codec's order: the engine's rule (``select``
      at temperature 0), except that the engine's tie-breaking prior is the noised one it
      searched with.
    * Another visited action a whose Q alone reaches S keeps its visits. Otherwise it keeps
      at most the fewest visits at which its score is no higher than S,
      ``ceil(max(0, c_puct * P_a * sqrt(T + 1) / (S - Q_a) - 1))``, and an action that this
      cuts to one visit or none keeps none.

    * ``logits``: the network's policy logits at the root, [A] or [1, A]; P is the engine's
      root prior before the noise (:func:`_world_log_priors`). One root world only (``dets``
      0 or 1): with more, each world's tree runs PUCT on its own prior, visits and Q, of
      which the pool reports the sums, so several rows of logits are refused.
    * ``legal``: the root's actions (bool [A]); ``visits`` and ``q``: the pool's
      ``visits(id)`` and ``q(id)`` (NaN where unvisited, never read there).
    * ``c_puct``: the search's (``SearchConfig.c_puct``, engine units).

    A root without a visit gives zeros, as the engine's target does. The arithmetic is float64.
    """
    legal = np.asarray(legal, dtype=bool)
    kept = np.zeros(legal.shape, dtype=np.float64)
    edges = np.flatnonzero(legal)
    n = np.asarray(visits, dtype=np.float64)[edges]
    if not n.sum():
        return kept
    log_prior = _world_log_priors(logits, edges)
    if len(log_prior) != 1:
        raise ValueError(f"pruned_visits reads one root world's logits, not {len(log_prior)}")
    prior = np.exp(log_prior[0])
    values = np.asarray(q, dtype=np.float64)[edges]
    most = np.flatnonzero(n == n.max())
    best = most[np.argmax(prior[most])]
    exploration = float(c_puct) * np.sqrt(n.sum() + 1.0)
    gap = values[best] + exploration * prior[best] / (1.0 + n[best]) - values  # S - Q
    with np.errstate(divide="ignore", invalid="ignore"):  # no gap, or NaN at an unvisited action: not cut
        earned = np.ceil(np.maximum(exploration * prior / gap - 1.0, 0.0))
    cut = (gap > 0.0) & (earned < n)
    cut[best] = False
    kept[edges] = np.where(cut & (earned > 1.0), earned, np.where(cut, 0.0, n))
    return kept


# ---------------------------------------------------------------- replay


class ReplayBuffer:
    """A FIFO of positions: float16 rows and policy targets, float32 value targets."""

    def __init__(self, capacity: int, obs_size: int, num_actions: int, max_players: int) -> None:
        self.capacity = int(capacity)
        self.shapes = (int(obs_size), int(num_actions), int(max_players))
        self.size = 0
        self.next = 0
        self.added = 0
        self.obs = self.policy = self.has_policy = self.value = self.present = None

    def _allocate(self) -> None:
        obs_size, num_actions, players = self.shapes
        self.obs = np.zeros((self.capacity, obs_size), dtype=np.float16)
        self.policy = np.zeros((self.capacity, num_actions), dtype=np.float16)
        self.has_policy = np.zeros(self.capacity, dtype=bool)
        self.value = np.zeros((self.capacity, players), dtype=np.float32)
        self.present = np.zeros((self.capacity, players), dtype=np.float32)

    def add(self, obs, policy, has_policy, value, present) -> None:
        if self.obs is None:
            self._allocate()
        count = len(obs)
        keep = slice(max(0, count - self.capacity), count)  # more than fits: the newest
        slots = (self.next + np.arange(count)[keep] - keep.start) % self.capacity
        for name, values in (("obs", obs), ("policy", policy), ("has_policy", has_policy), ("value", value),
                             ("present", present)):
            getattr(self, name)[slots] = np.asarray(values)[keep]
        self.next = (self.next + len(slots)) % self.capacity
        self.size = min(self.size + len(slots), self.capacity)
        self.added += count

    def batch(self, index: np.ndarray, device: str) -> dict[str, torch.Tensor]:
        def tensor(array, dtype=torch.float32):
            return torch.from_numpy(np.ascontiguousarray(array[index])).to(device=device, dtype=dtype)

        return {"obs": tensor(self.obs), "policy": tensor(self.policy), "has_policy": tensor(self.has_policy, torch.bool),
                "value": tensor(self.value), "present": tensor(self.present)}

    def state(self) -> dict[str, Any]:
        """The buffer for the trainer state, oldest first (``weights_only``-loadable)."""
        if self.size == 0:
            return {"size": 0, "added": self.added}
        order = np.arange(self.next - self.size, self.next) % self.capacity
        arrays = {name: torch.from_numpy(np.ascontiguousarray(getattr(self, name)[order]))
                  for name in ("obs", "policy", "has_policy", "value", "present")}
        return {"size": self.size, "added": self.added, **arrays}

    def restore(self, state: Mapping[str, Any]) -> None:
        self.size = self.next = 0
        self.added = int(state["added"])
        count = int(state["size"])
        if not count:
            return
        self._allocate()
        keep = slice(max(0, count - self.capacity), count)
        for name in ("obs", "policy", "has_policy", "value", "present"):
            values = state[name][keep].numpy()
            getattr(self, name)[: len(values)] = values
        self.size = min(count, self.capacity)
        self.next = self.size % self.capacity


# ---------------------------------------------------------------- self-play


def make_vec(plan: Any, levels: Mapping[str, Any], *, num_envs: int, seed: int, threads: int,
             trackers: bool = False) -> cpp.VecEnv:
    """A VecEnv of ``plan``'s scenarios writing the rows of a network at ``levels``."""
    critic = critic_level_of(levels["level"], levels["critic_level"])
    return cpp.VecEnv(plan.scenarios, num_envs=num_envs, dims=encoder_dims(plan.dims),
                      level=info_level(levels["level"]), seed=seed, num_threads=threads,
                      critic_level=None if critic is None else info_level(critic), trackers=trackers)


class RootOutputs:
    """A SearchPool for :func:`run_pools` that keeps the network's outputs at its searches' roots.

    A search's root worlds are its first leaves (``collect``: roots first, and no other leaf
    before every root is expanded), so the first ``worlds`` rows collected for a search
    registered with :meth:`expect` are its roots: this keeps their legal masks (the rows'
    ``mask`` block), logits and values, the very outputs (rows, batches, evaluator) the search
    expanded its root with. Everything else passes through to the pool.
    """

    def __init__(self, pool: cpp.SearchPool, mask: slice) -> None:
        self.pool, self.mask = pool, mask
        self.worlds: dict[int, int] = {}
        self.roots: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        self.outstanding = 0  # root rows not collected yet: none after the first round or so
        self.batch: list[tuple[int, int, np.ndarray]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.pool, name)

    def expect(self, sid: int) -> None:
        """Keep the root outputs of search ``sid``, just added (its worlds are drawn)."""
        self.worlds[sid] = int(self.pool.stats(sid).worlds)
        self.roots[sid] = []
        self.outstanding += self.worlds[sid]

    def collect(self, observations: np.ndarray, ids: np.ndarray) -> int:
        n = self.pool.collect(observations, ids)
        self.batch = []
        if self.outstanding:
            issued: dict[int, int] = {}
            for k, sid in enumerate(ids[:n].tolist()):
                if sid in self.worlds and len(self.roots[sid]) + issued.get(sid, 0) < self.worlds[sid]:
                    issued[sid] = issued.get(sid, 0) + 1
                    self.batch.append((k, sid, observations[k, self.mask] > 0))
            self.outstanding -= len(self.batch)
        return n

    def apply(self, logits: np.ndarray, values: np.ndarray) -> None:
        self.pool.apply(logits, values)
        for k, sid, legal in self.batch:
            self.roots[sid].append((legal, logits[k].copy(), values[k].copy()))
        self.batch = []

    def root(self, sid: int) -> tuple[np.ndarray, np.ndarray, float]:
        """(legal actions, logits per world [W, A], value) of search ``sid``'s root, as the engine
        keeps them: the actions every world offers, and each world's value for the root's player
        (slot 0) clamped to [0, 1], averaged."""
        rows = self.roots[sid]
        if len(rows) != self.worlds[sid]:
            raise RuntimeError(f"search {sid}: {len(rows)} root evaluations for {self.worlds[sid]} worlds")
        legal = np.logical_and.reduce([legal for legal, _, _ in rows])
        value = float(np.mean([min(max(float(values[0]), 0.0), 1.0) for _, _, values in rows]))
        return legal, np.stack([logits for _, logits, _ in rows]), value


class Rounds:
    """An evaluator for :func:`run_pools` that counts its calls: one per round of leaves."""

    def __init__(self, evaluator: BatchEvaluator) -> None:
        self.evaluator, self.count = evaluator, 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.evaluator, name)

    def __call__(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.count += 1
        return self.evaluator(observations)


@dataclass
class _Pending:
    obs: np.ndarray
    policy: np.ndarray | None
    seat: int
    num_players: int


class SelfPlay:
    """The VecEnv, two search pools (full and fast budgets) and the games' pending positions.

    ``groups`` (``runtime.search_groups``) splits the games into that many search groups, each
    with its own two pools (game e searches in group e % groups), and runs a step's searches
    with :func:`~catanrl.search.pipeline.run_groups`: the device evaluates one group's leaves
    while the engine collects and applies the others'. One group is ``run_pools``. The searches
    are the same either way (pipeline.py says to which bits)."""

    def __init__(self, plan: Any, train: Mapping[str, Any], *, model: CatanModel, num_envs: int, threads: int,
                 seed: int, device: str, mode: str, groups: int = 1) -> None:
        if int(groups) < 1:
            raise ValueError("runtime.search_groups must be at least 1")
        self.train = dict(train)
        self.seed = int(seed)
        self.dims = dict(plan.dims)
        levels = model.levels
        self.dets = int(train["dets"])
        self.vec = make_vec(plan, levels, num_envs=num_envs, seed=self.seed, threads=threads, trackers=self.dets > 0)
        self.buffers = self.vec.allocate()
        self.vec.reset(self.seed)
        self.players = int(self.vec.max_players)
        self.model, self.device, self.mode = model, device, mode
        self.evaluator = self._evaluator()

        def pool(sims: int, pending: int, noise: bool) -> cpp.SearchPool:
            params = {"sims": sims, "dets": self.dets, "oracle": self.dets == 0, "root": train["root"],
                      "c_puct": train["c_puct"], "gumbel_m": train["gumbel_m"], "pending": pending, "noise": noise}
            config = search_config(params, max_players=self.players, threads=threads)
            return make_pool(self.dims, levels["level"], config, critic_level=levels["critic_level"])

        #: Leaves a search sends per round, by pool (full, fast).
        self.leaves = int(train["pending"]), int(train["fast_pending"])
        noise = bool(train["dirichlet_noise"]) and train["root"] == "puct"
        self.full_pool = pool(int(train["sims"]), self.leaves[0], noise)
        self.fast_pool = pool(int(train["fast_sims"]), self.leaves[1], False)  # fast searches: no noise (KataGo)
        #: Each search group's (full, fast) pools; the first group's are the two above.
        self.groups = [(self.full_pool, self.fast_pool)] + [
            (pool(int(train["sims"]), self.leaves[0], noise), pool(int(train["fast_sims"]), self.leaves[1], False))
            for _ in range(int(groups) - 1)]
        self.batches: dict = {}  # run_groups' page-locked rows, kept across steps and evaluators
        #: policy_target: gumbel's sigma constants (None: the engine's target, or the pruned one).
        self.sigma = None if train["policy_target"] != "gumbel" else {
            "c_visit": float(train["target_c_visit"]), "c_scale": float(train["target_c_scale"]),
            "rescale": bool(train["target_rescale"]), "prior_temperature": float(train["target_prior_temperature"])}
        self.mask = slice(model.mask_offset, model.mask_offset + model.num_actions)
        self.pending: list[list[_Pending]] = [[] for _ in range(num_envs)]
        self.ply = np.zeros(num_envs, dtype=np.int64)
        self.decisions = 0
        self.games = 0
        #: Sees each step's full-budget searches (rows and targets); logging only.
        self.monitor: Monitors | None = None

    def _evaluator(self) -> BatchEvaluator:
        return BatchEvaluator(self.model, device=self.device, infer_rows=int(self.train["search_rows"]),
                              mode=self.mode, kernels=self.train["kernels"])

    def refresh(self) -> None:
        """After the weights changed (learn(), a resume). The module's forward reads them in place;
        the fused kernels read copies made when the evaluator was built (FusedGraph), so
        ``kernels: fused`` rebuilds it, its CUDA graphs included."""
        if self.train["kernels"] == "fused":
            self.evaluator = None  # its graphs and static rows go before the new ones are captured
            self.evaluator = self._evaluator()

    def check_kernels(self, rows: int = 64) -> None:
        """``kernels: fused``: the evaluator's values against the module's own on acting rows. The
        kernels are within 1e-6 of the module (graph_kernels.py); an evaluator left on older weights
        is not, and self-play would search with them unnoticed."""
        if self.train["kernels"] != "fused":
            return
        obs = self.buffers["observations"][np.flatnonzero(self.buffers["acting"])[:rows]]
        with torch.inference_mode():
            want = self.model(torch.from_numpy(obs).to(self.device))[1].float().cpu().numpy()
        gap = float(np.abs(self.evaluator(obs)[1] - want).max()) if len(obs) else 0.0
        if gap > 1e-4:
            raise RuntimeError(f"kernels: fused: the evaluator's values are {gap:.2g} from the network's "
                               "(its weights are stale: refresh() after every change)")

    def step(self, iteration: int, step: int, rng: np.random.Generator, replay: ReplayBuffer,
             stats: dict[str, float]) -> None:
        """Search every acting row, play the chosen moves and file finished games."""
        b = self.buffers
        acting = np.flatnonzero(b["acting"])
        full = rng.random(len(acting)) < float(self.train["full_search_prob"])
        searches = []
        targets = []  # the full searches' own policy targets (pool.policy_target), in acting order, for the monitor
        rebuilt = []  # policy_target: gumbel's or pruned's targets of the same searches (the ones trained on), and
        pruned = []  # pruned's share of each of those searches' root visits that it removed, for the monitor
        groups = len(self.groups)
        roots = ([RootOutputs(pools[0], self.mask) for pools in self.groups]
                 if self.train["policy_target"] != "engine" else None)
        try:
            for i, row in enumerate(acting):
                env, seat = int(b["env_index"][row]), int(b["seat"][row])
                pool = self.groups[env % groups][0 if full[i] else 1]
                tracker = self.vec.tracker(env, seat) if self.dets else None
                seed = derive_seed(self.seed, "selfplay", iteration, step, env)
                searches.append((pool, pool.add(self.vec.game(env), seed=seed, tracker=tracker)))
                if roots is not None and full[i]:
                    roots[env % groups].expect(searches[-1][1])
            evaluator = Rounds(self.evaluator)
            # a round's rows: every search's quota (collect caps a search by the room left in its buffer)
            if groups == 1:
                leaves = run_pools([self.full_pool if roots is None else roots[0], self.fast_pool], evaluator,
                                   batch_rows=[int(full.sum()) * self.leaves[0], int((~full).sum()) * self.leaves[1]])
            else:
                of = b["env_index"][acting] % groups
                leaves, evaluator.count = run_groups(
                    [Group([pools[0] if roots is None else roots[g], pools[1]],
                           [int((full & (of == g)).sum()) * self.leaves[0], int((~full & (of == g)).sum()) * self.leaves[1]])
                     for g, pools in enumerate(self.groups)], Pipeline(self.evaluator, self.batches))
            for i, row in enumerate(acting):
                env, seat = int(b["env_index"][row]), int(b["seat"][row])
                pool, sid = searches[i]
                policy = None
                if full[i]:
                    if roots is None:
                        policy = pool.policy_target(sid).astype(np.float16)
                        targets.append(policy)
                    else:  # the root's actions: the real game's (the row's mask) that every world offers
                        legal, logits, value = roots[env % groups].root(sid)
                        legal = legal & (b["observations"][row, self.mask] > 0)
                        if self.sigma is not None:
                            policy = gumbel_target(logits, legal, pool.visits(sid), pool.q(sid), value,
                                                   **self.sigma).astype(np.float16)
                        else:  # policy_target: pruned, at the c_puct the search scored its edges with
                            visits = pool.visits(sid)
                            kept = pruned_visits(logits, legal, visits, pool.q(sid), c_puct=pool.config.c_puct)
                            policy = (kept / max(kept.sum(), 1.0)).astype(np.float16)
                            total = float(visits[legal].sum())
                            pruned.append(1.0 - kept.sum() / total if total else 0.0)
                        rebuilt.append(policy)
                        if self.monitor is not None:  # search/* keep reading the search's own target
                            targets.append(pool.policy_target(sid).astype(np.float16))
                    search = pool.stats(sid)
                    stats["depth"] += search.mean_depth
                    stats["max_depth"] += search.max_depth
                b["actions"][row] = pool.select(sid, temperature(self.train, int(self.ply[env])))
                if full[i] or self.train["record_fast"]:
                    self.pending[env].append(_Pending(b["observations"][row].astype(np.float16), policy, seat,
                                                      int(b["num_players"][env])))
                self.ply[env] += 1
        finally:
            for pool in (pool for pools in self.groups for pool in pools):
                if pool.pending:
                    pool.cancel()
                for sid in pool.ids:
                    pool.remove(sid)
        stats["leaves"] += leaves
        stats["rounds"] = stats.get("rounds", 0.0) + evaluator.count
        stats["full"] += int(full.sum())
        self.decisions += len(acting)
        if self.monitor is not None and targets:  # before step() rewrites rows
            self.monitor.searched(b["observations"][acting[full]], np.stack(targets),
                                  np.stack(rebuilt) if rebuilt else None, np.array(pruned) if pruned else None)
        self.vec.step()
        self._finish(replay)

    def _finish(self, replay: ReplayBuffer) -> None:
        b = self.buffers
        players = self.players
        for env in np.flatnonzero(b["terminals"].reshape(-1, players)[:, 0]):
            rows = self.pending[env]
            self.pending[env] = []
            self.ply[env] = 0
            self.games += 1
            if not rows:
                continue
            rewards = b["rewards"][env * players : (env + 1) * players]
            winner = int(np.argmax(rewards)) if rewards.max() > 0 else None
            targets = [value_targets(winner, r.seat, r.num_players, players, self.train["truncation_value"])
                       for r in rows]
            num_actions = self.full_pool.action_size
            replay.add(
                np.stack([r.obs for r in rows]),
                np.stack([r.policy if r.policy is not None else np.zeros(num_actions, np.float16) for r in rows]),
                np.array([r.policy is not None for r in rows]),
                np.stack([t for t, _ in targets]),
                np.stack([p for _, p in targets]),
            )


# ---------------------------------------------------------------- learning


def policy_stats(target: torch.Tensor, net_logp: torch.Tensor, net_probs: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per row: the target's entropy, the network's entropy over its legal actions and
    KL(target || net). learn()'s train/target_entropy, net_entropy and kl_target_net are their
    minibatch means. ``net_logp`` is the masked log-policy (illegal actions below -1e8) and
    ``net_probs`` its exp."""
    legal = net_logp > -1e8
    safe = torch.where(legal, net_logp, torch.zeros_like(net_logp))
    target_log = target.clamp_min(1e-12).log()
    return -(target * target_log).sum(1), -(net_probs * safe).sum(1), (target * (target_log - safe)).sum(1)


def learn(model: CatanModel, optimizer: torch.optim.Optimizer, replay: ReplayBuffer, train: Mapping[str, Any],
          device: str, rng: np.random.Generator) -> dict[str, float]:
    """``updates_per_iteration`` gradient steps on uniform minibatches; their mean statistics."""
    model.train()
    sums: dict[str, float] = {}
    players = replay.shapes[2]
    by_n = np.zeros((players + 1, 3))  # count, predicted seat-sum, target seat-sum
    updates = int(train["updates_per_iteration"])
    for _ in range(updates):
        batch = replay.batch(rng.integers(0, replay.size, int(train["minibatch_size"])), device)
        logits, value_logits, _ = model.heads(batch["obs"])
        logp = F.log_softmax(logits.float(), dim=1)
        has = batch["has_policy"]
        target = batch["policy"][has]
        target = target / target.sum(1, keepdim=True).clamp_min(1e-8)
        policy_loss = -(target * logp[has]).sum(1).mean() if bool(has.any()) else logp.new_zeros(())
        present = batch["present"]
        value_loss = (F.binary_cross_entropy_with_logits(value_logits.float(), batch["value"], reduction="none")
                      * present).sum() / present.sum().clamp_min(1.0)
        loss = policy_loss + float(train["value_coef"]) * value_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train["max_grad_norm"]))
        optimizer.step()
        with torch.no_grad():
            probs = logp.exp()
            if bool(has.any()):
                for key, value in zip(("target_entropy", "net_entropy", "kl_target_net"),
                                      policy_stats(target, logp[has], probs[has])):
                    sums[key] = sums.get(key, 0.0) + float(value.mean()) / updates
            for key, value in (("policy_loss", policy_loss), ("value_loss", value_loss)):
                sums[key] = sums.get(key, 0.0) + float(value) / updates
            predicted = (torch.sigmoid(value_logits.float()) * present).sum(1)
            n = present.sum(1).long().cpu().numpy()
            np.add.at(by_n, (n, 0), 1.0)
            np.add.at(by_n, (n, 1), predicted.cpu().numpy())
            np.add.at(by_n, (n, 2), (batch["value"] * present).sum(1).cpu().numpy())
    for n in range(2, players + 1):
        if by_n[n, 0]:
            sums[f"value_sum_{n}p"] = by_n[n, 1] / by_n[n, 0]
            sums[f"target_sum_{n}p"] = by_n[n, 2] / by_n[n, 0]
    model.eval()
    return sums


# ---------------------------------------------------------------- monitors

#: Positions in the drift probe, and the probe games that supply them (each env's first game).
PROBE_ROWS = 4096
PROBE_ENVS = 64
PROBE_MAX_STEPS = 4000
#: The probe's phases (they overlap): initial placements, discards and robber moves (the row's
#: prompt flags), a development card playable, and a build option on a normal turn (not a free road).
PHASES = ("setup", "discard", "robber", "dev", "build")


def phases_of(model: CatanModel, rows: np.ndarray) -> dict[str, np.ndarray]:
    """Which of :data:`PHASES` each row [n, obs] is in, from its global flags and action mask
    (a phase whose flag or actions the rows lack is all False)."""
    names = feature_names(model.dims, model.levels["level"])["global"]
    offset = int(model.layout["global"][0])
    none = np.zeros(len(rows), dtype=bool)

    def flag(name: str) -> np.ndarray:
        return rows[:, offset + names.index(name)] > 0 if name in names else none

    actions = cpp.GraphActionSpace(encoder_dims(model.dims)).names()
    legal = rows[:, model.mask_offset : model.mask_offset + model.num_actions] > 0

    def playable(prefixes: tuple[str, ...]) -> np.ndarray:
        if len(actions) != legal.shape[1]:
            return none
        return legal[:, [i for i, name in enumerate(actions) if name.startswith(prefixes)]].any(1)

    return {
        "setup": flag("prompt_build_initial_settlement") | flag("prompt_build_initial_road"),
        "discard": flag("prompt_discard"),
        "robber": flag("prompt_move_robber"),
        "dev": playable(("PLAY_", "MONOPOLY_", "YEAR_OF_PLENTY_")),
        "build": flag("prompt_play_turn") & ~flag("road_building") & playable(("SETTLEMENT@", "CITY@", "ROAD@")),
    }


class Monitors:
    """Logging-only health checks of a run (forwards of the network in chunks of ``rows``).

    * ``oos/*`` (before learn()): the value head out of sample, on the positions of the games
      that finished during the iteration (in replay but not yet trained on): ``value_bce``
      (per seated slot, train/value_loss's normalization) and ``value_brier`` against the outcome
      targets the trainer uses; per player count n the mean seat-sums ``value_sum_{n}p`` and
      ``target_sum_{n}p`` and their gap ``sum_gap_{n}p``; ``rows``.
    * ``search/*`` (self-play): among the iteration's full-budget searches at non-forced
      decisions, ``overrule``, the fraction whose target's argmax is not the network's masked
      argmax at that root; ``target_entropy``, ``net_entropy`` and ``kl_target_net`` on those
      rows (:func:`policy_stats`, learn()'s train/* definitions, here out of sample);
      ``overrule_rows``. The target is the search's own (``pool.policy_target``) under either
      ``policy_target``, so these series compare across runs.
    * ``target/*`` (self-play, ``policy_target: gumbel`` or ``pruned``): ``overrule``, ``entropy``
      and ``kl_net``, the same three on the same rows for the rebuilt or pruned targets the
      network trains on; with ``pruned`` also ``pruned``, the share of a search's root visits
      that the pruning removed (the mean over those searches).
    * ``drift/*`` (after learn(), so epoch e is checkpoint e's network): ``kl_init``, the mean
      KL(init || net), and ``agree_init``, the argmax agreement, on a fixed probe set, overall
      and per phase (``_{phase}`` of :data:`PHASES`, with ``rows_{phase}``). The probe is up to
      :data:`PROBE_ROWS` non-forced decisions drawn uniformly from :data:`PROBE_ENVS` games that
      the starting network plays with its raw policy (sampled), in their own VecEnv and RNG, with
      the starting network's masked log-policy on them. It is drawn once when a run starts and
      kept in the trainer state; a state without one (an older run) logs no drift.

    Nothing reads them back: the RNG streams, VecEnv, searches, targets, replay and updates are
    the same with or without them (test_monitors_are_logging_only).
    """

    def __init__(self, model: CatanModel, *, device: str, rows: int) -> None:
        self.model, self.device, self.rows = model, device, max(1, int(rows))
        self.mask = slice(model.mask_offset, model.mask_offset + model.num_actions)
        self.probe: dict[str, torch.Tensor] | None = None
        self.phases: dict[str, torch.Tensor] = {}
        self.begin()

    def begin(self) -> None:
        """Clear the iteration's sums and time."""
        self.searches = np.zeros(5)  # rows, overruled, target entropy, net entropy, KL(target || net)
        self.rebuilt = np.zeros(5)  # the same for policy_target: gumbel's or pruned's targets
        self.pruned = np.zeros(2)  # policy_target: pruned: rows, their shares of root visits removed
        self.seconds = 0.0

    def _forward(self, obs: np.ndarray, head: str) -> torch.Tensor:
        """Masked policy logits or value logits of rows (float16 or float32): float32 on the CPU."""
        out = []
        with torch.inference_mode():
            for start in range(0, len(obs), self.rows):
                chunk = torch.from_numpy(np.ascontiguousarray(obs[start : start + self.rows], dtype=np.float32))
                chunk = chunk.to(self.device)
                result = self.model.policy_logits(chunk) if head == "policy" else self.model.value_logits(chunk)[0]
                out.append(result.float().cpu())
        return torch.cat(out)

    # ------------------------------------------------------------ self-play

    def searched(self, obs: np.ndarray, targets: np.ndarray, rebuilt: np.ndarray | None = None,
                 pruned: np.ndarray | None = None) -> None:
        """One self-play step's full-budget searches: their roots' rows, the searches' own float16
        targets, (``policy_target: gumbel`` or ``pruned``) the rebuilt or pruned ones and
        (``pruned``) each search's share of root visits removed."""
        start = time.time()
        keep = (obs[:, self.mask] > 0).sum(1) > 1  # VecEnv plays forced moves itself: normally every row
        if keep.any():
            logp = F.log_softmax(self._forward(obs[keep], "policy"), dim=1)
            for sums, chosen in ((self.searches, targets), (self.rebuilt, rebuilt)):
                if chosen is None:
                    continue
                target = torch.from_numpy(chosen[keep].astype(np.float32))
                target = target / target.sum(1, keepdim=True).clamp_min(1e-8)
                entropies = policy_stats(target, logp, logp.exp())
                overruled = (target.argmax(1) != logp.argmax(1)).sum()
                sums += [int(keep.sum()), float(overruled), *(float(value.sum()) for value in entropies)]
            if pruned is not None:
                self.pruned += [int(keep.sum()), float(pruned[keep].sum())]
        self.seconds += time.time() - start

    # ------------------------------------------------------------ before and after learn()

    def before_learn(self, replay: ReplayBuffer, count: int) -> dict[str, float]:
        """``search/*`` and ``target/*`` of the iteration and ``oos/*`` of the newest ``count``
        replay rows (the positions of the games that finished during the iteration)."""
        start = time.time()
        logs: dict[str, float] = {}
        rows, overruled, target_entropy, net_entropy, kl = self.searches
        if rows:
            logs.update({"search/overrule": float(overruled / rows), "search/overrule_rows": int(rows),
                         "search/target_entropy": float(target_entropy / rows),
                         "search/net_entropy": float(net_entropy / rows), "search/kl_target_net": float(kl / rows)})
        rows, overruled, target_entropy, _, kl = self.rebuilt
        if rows:
            logs.update({"target/overrule": float(overruled / rows), "target/entropy": float(target_entropy / rows),
                         "target/kl_net": float(kl / rows)})
        rows, share = self.pruned
        if rows:
            logs["target/pruned"] = float(share / rows)
        count = min(int(count), replay.size)
        if count:
            index = (replay.next - count + np.arange(count)) % replay.capacity
            players = replay.shapes[2]
            by_n = np.zeros((players + 1, 3))  # rows, predicted seat-sum, target seat-sum
            bce = brier = seats = 0.0
            for first in range(0, count, self.rows):
                chunk = index[first : first + self.rows]
                logits = self._forward(replay.obs[chunk], "value")
                target, present = torch.from_numpy(replay.value[chunk]), torch.from_numpy(replay.present[chunk])
                predicted = torch.sigmoid(logits)
                bce += float((F.binary_cross_entropy_with_logits(logits, target, reduction="none") * present).sum())
                brier += float(((predicted - target) ** 2 * present).sum())
                seats += float(present.sum())
                n = present.sum(1).long().numpy()
                np.add.at(by_n, (n, 0), 1.0)
                np.add.at(by_n, (n, 1), (predicted * present).sum(1).numpy())
                np.add.at(by_n, (n, 2), (target * present).sum(1).numpy())
            logs.update({"oos/rows": count, "oos/value_bce": bce / max(seats, 1.0),
                         "oos/value_brier": brier / max(seats, 1.0)})
            for n in range(2, players + 1):
                if by_n[n, 0]:
                    value, target_sum = by_n[n, 1] / by_n[n, 0], by_n[n, 2] / by_n[n, 0]
                    logs.update({f"oos/value_sum_{n}p": float(value), f"oos/target_sum_{n}p": float(target_sum),
                                 f"oos/sum_gap_{n}p": float(abs(value - target_sum))})
        self.seconds += time.time() - start
        return logs

    def drift(self) -> dict[str, float]:
        """``drift/*``: the network against the starting network on the probe."""
        if self.probe is None:
            return {}
        start = time.time()
        logp = F.log_softmax(self._forward(self.probe["obs"].numpy(), "policy"), dim=1)
        init = self.probe["init_logp"]
        kl = (init.exp() * torch.where(init > -1e8, init - logp, torch.zeros_like(init))).sum(1)
        agree = (init.argmax(1) == logp.argmax(1)).float()
        logs: dict[str, float] = {"drift/rows": len(kl), "drift/kl_init": float(kl.mean()),
                                  "drift/agree_init": float(agree.mean())}
        for name, rows in self.phases.items():
            if bool(rows.any()):
                logs.update({f"drift/kl_init_{name}": float(kl[rows].mean()),
                             f"drift/agree_init_{name}": float(agree[rows].mean()),
                             f"drift/rows_{name}": int(rows.sum())})
        self.seconds += time.time() - start
        return logs

    # ------------------------------------------------------------ the probe

    def capture(self, plan: Any, *, seed: int, threads: int) -> None:
        """Draw the probe with the network as it is now (the run's start); see the class doc."""
        vec = make_vec(plan, self.model.levels, num_envs=PROBE_ENVS, seed=derive_seed(seed, "env"), threads=threads)
        b = vec.allocate()
        vec.reset(derive_seed(seed, "env"))
        rng = np.random.default_rng(derive_seed(seed, "moves"))
        players = int(vec.max_players)
        done = np.zeros(PROBE_ENVS, dtype=bool)
        kept = []
        for _ in range(PROBE_MAX_STEPS):
            acting = np.flatnonzero(b["acting"])
            if not len(acting):
                break
            obs = b["observations"][acting]
            logits = self._forward(obs, "policy").numpy()
            b["actions"][acting] = np.argmax(logits + rng.gumbel(size=logits.shape), axis=1)  # a softmax sample
            kept.append(obs[~done[b["env_index"][acting]]].astype(np.float16))
            vec.step()
            done |= b["terminals"].reshape(-1, players)[:, 0] > 0
            if done.all():
                break
        rows = np.concatenate(kept) if kept else np.zeros((0, int(vec.observation_size)), dtype=np.float16)
        rows = rows[(rows[:, self.mask] > 0).sum(1) > 1]
        if len(rows) > PROBE_ROWS:
            rows = rows[np.sort(rng.choice(len(rows), PROBE_ROWS, replace=False))]
        if len(rows):
            self.restore({"probe_obs": torch.from_numpy(rows),
                          "probe_init_logp": F.log_softmax(self._forward(rows, "policy"), dim=1)})

    def state(self) -> dict[str, torch.Tensor]:
        """The probe for the trainer state (``weights_only``-loadable)."""
        if self.probe is None:
            return {}
        return {"probe_obs": self.probe["obs"], "probe_init_logp": self.probe["init_logp"]}

    def restore(self, state: Mapping[str, Any]) -> None:
        if state.get("probe_obs") is None:
            self.probe, self.phases = None, {}
            return
        self.probe = {"obs": state["probe_obs"], "init_logp": state["probe_init_logp"]}
        self.phases = {name: torch.from_numpy(rows)
                       for name, rows in phases_of(self.model, self.probe["obs"].numpy()).items()}


# ---------------------------------------------------------------- the loop


def train_alphazero(variant: Any, writer: Any, *, monitors: bool = True) -> None:
    """AlphaZero on ``variant`` (a :class:`catanrl.config.Variant`) into ``writer``'s run.
    ``monitors=False`` (tests) leaves out :class:`Monitors`, which change nothing else."""
    cfg = variant.config
    runtime = dict(cfg.get("runtime") or {})
    strict = runtime.get("determinism", "fast") == "strict"
    with determinism("strict" if strict else "fast"):
        _train(cfg, writer, check_train(cfg["train"]), runtime, strict, monitors)


def _train(cfg: Mapping[str, Any], writer: Any, train: dict[str, Any], runtime: Mapping[str, Any], strict: bool,
           monitoring: bool = True) -> None:
    seed = int(cfg["seed"])
    launch = int(writer.launch)
    env_cfg = cfg["env"]
    pool_seed = (env_cfg.get("map_pool") or {}).get("seed")
    plan = plan_env(env_cfg, pool_seed=derive_seed(seed, "map_pool") if pool_seed is None else int(pool_seed))
    seed_everything(derive_seed(seed, "init"))
    model = build_policy_model(cfg, plan.dims)
    check_bindable(model, plan)
    if getattr(model, "recurrent", False):
        raise ValueError("train.algo: alphazero evaluates single positions; a recurrent network trains with ppo")
    if cfg.get("init_from") and starts_fresh(writer):
        from catanrl.checkpoint import load_parts

        load_parts(model, cfg["init_from"])
    device = resolve_device(runtime.get("device", "auto"))
    model.to(device).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train["learning_rate"]),
                                  weight_decay=float(train["weight_decay"]))
    selfplay = SelfPlay(plan, train, model=model, num_envs=int(env_cfg["num_envs"]),
                        threads=int(runtime.get("threads", 1)), seed=derive_seed(seed, "rollout", launch),
                        device=device, mode="strict" if strict else "fast",
                        groups=int(runtime.get("search_groups") or SEARCH_GROUPS))
    replay = ReplayBuffer(int(train["replay_capacity"]), selfplay.vec.observation_size, selfplay.vec.action_size,
                          selfplay.players)
    monitors = Monitors(model, device=device, rows=int(train["search_rows"])) if monitoring else None
    selfplay.monitor = monitors
    eval_runtime = {key: runtime[key] for key in ("device", "num_envs", "threads", "infer_rows") if key in runtime}
    selector = Selector((cfg.get("eval") or {}).get("select"), writer, eval_runtime,
                        monitors=(cfg.get("eval") or {}).get("monitor"))
    iteration = 0

    def save() -> None:
        ref = writer.save_checkpoint(model, epoch=iteration, agent_steps=selfplay.decisions)
        writer.save_trainer_state({
            "epoch": iteration, "decisions": selfplay.decisions, "games": selfplay.games,
            "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "optimizer": copy.deepcopy(optimizer.state_dict()), "replay": replay.state(), "rng": rng_state(),
            "map_pool": plan.pool["hash"], **({"monitors": monitors.state()} if monitors is not None else {}),
        })
        if selector.due(iteration):
            selector.run(ref, model)

    state = writer.load_trainer_state() if writer.mode == "resume" else None
    if state is not None:
        if state.get("map_pool") != plan.pool["hash"]:
            raise RuntimeError("the map pool changed since the run started; refusing to resume")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        replay.restore(state["replay"])
        iteration = int(state["epoch"])
        selfplay.decisions, selfplay.games = int(state["decisions"]), int(state["games"])
        set_rng_state(state["rng"])
        selfplay.refresh()
        if monitors is not None:
            monitors.restore(state.get("monitors") or {})
        selector.restore()
        state = None  # restored: free the loaded copy (its replay is as large as the buffer) for the launch
    else:
        if monitors is not None:
            monitors.capture(plan, seed=derive_seed(seed, "probe"), threads=int(runtime.get("threads", 1)))
        save()
    selector.catch_up(model)

    status, saved = "finished", iteration
    launched, first, total = time.time(), iteration, int(train["iterations"])  # this launch, for the ETA
    limit = runtime.get("time_limit_hours")
    while iteration < int(train["iterations"]):
        if writer.should_stop():
            status = writer.stop_reason or "interrupted"
            break
        start = time.time()
        rng = np.random.default_rng(derive_seed(seed, "selfplay", iteration))
        decisions, games = selfplay.decisions, selfplay.games
        stats = {"leaves": 0.0, "rounds": 0.0, "full": 0.0, "depth": 0.0, "max_depth": 0.0}
        added = replay.added
        model.eval()
        selfplay.check_kernels()
        if monitors is not None:
            monitors.begin()
        for step in range(int(train["steps_per_iteration"])):
            selfplay.step(iteration, step, rng, replay, stats)
        played = time.time() - start
        health = monitors.before_learn(replay, replay.added - added) if monitors is not None else {}
        learned = {}
        if replay.size >= int(train["min_replay"]):
            learned = learn(model, optimizer, replay, train, device,
                            np.random.default_rng(derive_seed(seed, "replay", iteration)))
            selfplay.refresh()
        if monitors is not None:
            health.update(monitors.drift())
            health["throughput/monitor_s"] = monitors.seconds
        iteration += 1
        new = selfplay.decisions - decisions
        episodes = selfplay.vec.pop_stats()
        full = max(stats["full"], 1.0)
        logs = {
            "epoch": iteration, "decisions": selfplay.decisions, "games": selfplay.games, "replay": replay.size,
            "positions": replay.added, "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "selfplay/decisions_per_s": new / played, "selfplay/leaves_per_s": stats["leaves"] / played,
            "selfplay/games_per_hour": (selfplay.games - games) * 3600.0 / played,
            "selfplay/full_fraction": stats["full"] / max(new, 1),
            "selfplay/rounds": stats["rounds"] / int(train["steps_per_iteration"]),  # network rounds per step
            "search/mean_depth": stats["depth"] / full, "search/max_depth": stats["max_depth"] / full,
            "environment/episodes": episodes["episodes"], "environment/truncated": episodes["truncated"],
            "environment/length": episodes["length"], "environment/learner_win": episodes["learner_win"],
            "throughput/selfplay_s": played, "throughput/iteration_s": time.time() - start,
            **{f"train/{k}": v for k, v in learned.items()}, **health, "map_pool": plan.pool["hash"],
        }
        writer.log(logs, selfplay.decisions)
        # one line per iteration; the ETA is this launch's pace (checkpoints and selects included)
        # over the iterations left, or the time limit if that comes first
        eta, end = (time.time() - launched) / (iteration - first) * (total - iteration), ""
        if limit is not None and launched + float(limit) * 3600 - time.time() < eta:
            eta, end = max(launched + float(limit) * 3600 - time.time(), 0.0), " (time limit)"
        print(f"{writer.ref}: iteration {iteration}/{total}: {selfplay.games} games (+{selfplay.games - games}), "
              f"{replay.added} positions (+{replay.added - added}), {logs['selfplay/games_per_hour']:.0f} games/h, "
              f"{logs['selfplay/leaves_per_s']:.0f} leaves/s, {logs['throughput/iteration_s']:.0f} s, "
              f"ETA {eta / 3600:.1f} h{end}", flush=True)
        if iteration % int(train["checkpoint_interval"]) == 0:
            save()
            saved = iteration
    if saved != iteration:
        save()
    writer.finish(status)
