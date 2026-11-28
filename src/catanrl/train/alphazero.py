"""AlphaZero / expert iteration on cppanatron (DESIGN 1.6).

``train_alphazero(variant, writer)`` alternates, in one process:

1. **Self-play.** A VecEnv (auto-reset, the ``env`` scenarios as in PPO:
   mixed maps and player counts) plays games; every acting row is searched
   with a SearchPool, all searches in lockstep with their leaves evaluated in
   shared GPU batches (the eval's BatchEvaluator and CUDA graphs; with
   ``kernels: fused`` on the fused CUDA kernels, rebuilt from the weights
   after every learning step, which they copy; ``kernels: fused16`` on their
   half-precision variant, self-play only). Search is
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
   records its full searches only). ``record_fast_prob`` in (0, 1] (optional;
   unset 1: all of them) stores each fast-searched move with that probability,
   drawn from the iteration's self-play generator after the full-search draws
   (no draw at 1).
   Logged: ``selfplay/recorded_full`` and ``selfplay/recorded_fast`` (the
   iteration's decisions stored with and without a policy target) and
   ``train/policy_rows`` (the share of a minibatch's rows that have one). A
   search sends up to ``pending`` (``fast_pending``) leaves to the network per
   round, counted as visits while they wait; at 1 a step takes ``sims + 1``
   rounds of one leaf a search.
   With ``reuse: true`` (optional; PUCT roots on the true game) a game keeps
   its search tree from one decision to the next, whichever seat moves: when
   the game moved by exactly the action played, and that action was searched
   and deterministic, the next search starts from that action's subtree
   (``SearchPool.reroot``) and the visits the subtree carries count towards
   its budget (KataGo's visit cap): a full search tops its root up to
   ``sims`` visits, a fast one to ``fast_sims``, and a root that has them
   already is played from as it is. Every full search still draws its own
   Dirichlet noise, over the root's priors as the network gave them. A chance
   action (a roll, a bought card, a steal), a move the VecEnv played by
   itself (forced), the end of the game and the end of the iteration (the
   network changes) drop the tree, and the next search starts empty.
   ``root_policy_temperature`` and ``forced_playouts`` (optional; PUCT roots)
   are KataGo's two other ways to widen a full search's root, the engine's
   settings of the same names: the root's priors are raised to
   ``1 / root_policy_temperature`` and renormalized before the noise is
   mixed in, and a visited root action with fewer than
   ``sqrt(forced_playouts * prior * visits of the root)`` visits is searched
   before PUCT is asked. Like the noise they shape the full searches only
   (a fast one runs as if they were unset), and ``policy_target: pruned``
   takes out of the target what they add.
   ``runtime.search_groups`` (unset: 2) splits a step's searches into groups
   that take turns on the GPU while the engine collects and applies the
   others' leaves (:mod:`catanrl.search.pipeline`); 1 is the serial loop.
   ``runtime.eval_cache_mb`` (unset: none) gives the pools an evaluation
   cache of that many megabytes (:class:`catanrl.search.pool.NetworkCache`):
   a leaf whose row the network has evaluated since its weights last
   changed, or that another leaf of the same round has, takes that row's
   outputs and no row in the batch. The searches are the same (to the bit
   with ``runtime.determinism: strict``; with ``fast`` a row's outputs
   depend on its batch's size in their last bits, with or without a cache).
   Logged: ``selfplay/cache_hits`` (the share of the iteration's leaves that
   took no row) and ``selfplay/cache_replaced`` (the share of the rows
   stored that displaced a row still in use: the table is small when it
   nears 1); ``selfplay/leaves_per_s`` counts every leaf, and
   ``selfplay/rounds`` the rounds that reached the network.
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
   earns under the network's own prior, so the noise (and the root policy
   temperature and the forced playouts) still shapes the search and the move
   played but is not taught back to the network. With ``reuse``
   a target counts every visit of its root, carried and new, and the pruned
   one reads the prior from the tree (``SearchPool.root_priors``).
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

With ``model.args.outcome`` (a shared graph trunk) the network carries the
outcome head (:class:`catanrl.models.graph.GraphOutcomeHead`) and learn() adds
two auxiliary losses, ``outcome_vp_coef`` and ``outcome_turns_coef`` (optional
train keys, above 0 exactly when the head exists) times a Huber loss
(:data:`OUTCOME_HUBER_DELTA`) on

* each seated slot's final victory points over the game's points to win
  (seat-relative like the value targets; hidden victory point cards counted), and
* the turns left until the game ends over
  :data:`~catanrl.models.graph.OUTCOME_TURNS`,

on the positions of games that ended with a winner: a game cut at the turn
limit has no such labels and its positions are masked from both losses (as are
those of a game whose end the VecEnv did not report by itself: a second game
that finished in the same step). The replay holds the labels as integers, P + 4
bytes a position: the final victory points per slot (uint8), the points to win
(uint8), the turns left (uint16) and whether they are known (bool). The head is
training only: self-play, the searches, the monitors' policy and value forwards
and every agent run ``forward`` / ``heads``, which never touch it. Logged with
the other ``train/*``: ``outcome_vp_loss``, ``outcome_turns_loss``, their mean
absolute errors in victory points and in turns (``outcome_vp_mae``,
``outcome_turns_mae``) and ``outcome_known`` (the labelled share of the sampled
rows); and by :class:`Monitors` out of sample, ``oos/outcome_vp_mae`` and
``oos/outcome_turns_mae`` (with ``oos/outcome_rows``).

Four more pieces after KataGo, each optional (train keys that are off when
absent; a config without them hashes, and the trainer runs, as before they
existed):

* **A lower-noise value target** (``value_search_blend`` w, ``value_blend_iterations``
  n). The value target of a position that had a full search is
  ``(1 - w') * game result + w' * search value``, with w' = w * min(1, i / n) at
  the run's iteration i (n = 0: w at once). The search value is the root's action
  values averaged by the visits the policy target keeps (``policy_target: pruned``:
  the pruned visits; else all of them), for the seat to move, and 1 minus that for
  the other seat (:func:`search_value`): 2-player games only, since the engine keeps
  a root action's value for the mover alone. The replay holds it beside the game
  result (``search_value`` float32 [P], ``has_search_value``), so w' also weighs
  the positions already stored, and the monitors and ``target_sum_*`` keep reading
  the game result. The loss stays the binary cross-entropy, on the soft target.
  Logged: ``train/value_blend`` (w') and ``train/search_value_gap`` (the sampled
  rows' mean |search value - game result|, the mover's).
* **Replay sampling by policy surprise** (``surprise_weight`` s, ``surprise_cap``).
  A full search's surprise is KL(policy target || the network's prior at its root,
  before the noise), in nats (:func:`policy_surprise`; float32 per position, 0
  without a policy target). learn() draws position i with probability
  ``(1 - s) / N + s * c_i / sum(c)``, c_i its surprise capped at ``surprise_cap``
  times the buffer's mean (0: no cap), with no importance correction
  (:func:`surprise_probabilities`). Logged: ``train/surprise_sampled_mean`` and
  ``train/surprise_sampled_max`` (N * p_i over the rows drawn; 1 is uniform),
  ``train/surprise_ess`` (the distribution's effective sample size over N) and
  ``train/surprise_mean`` (the buffer's mean surprise).
* **A soft policy head** (``model.args.soft_policy``; ``soft_policy_coef``,
  ``soft_policy_temperature`` T, unset 4). A second policy head on the shared
  trunk learns the policy target raised to 1 / T over the legal actions
  (:func:`soft_policy_target`): cross-entropy, on the rows with a policy target,
  times ``soft_policy_coef`` (above 0 exactly when the head exists). Training only,
  like the outcome head: nothing plays from it. Logged: ``train/soft_policy_loss``
  and ``train/soft_target_entropy``.
* **Smaller searches in decided games** (``decided_value`` d, ``decided_lookback``
  k, ``decided_full_search_prob``, ``decided_fast_sims``; 2-player games). Every
  search, full or fast, names the game's leader and the leader's value (the
  search value above; a fast search's by all its visits). A game is decided at a
  decision when its last k searches all gave one seat at least d; it is then
  searched at the full budget with probability ``decided_full_search_prob``
  instead of ``full_search_prob``, and its fast searches run ``decided_fast_sims``
  (unset: ``fast_sims``; a budget of its own needs ``reuse``, whose searches each
  carry theirs). Full searches keep ``sims``. With ``record_fast_prob`` below 1 a
  decided game's fast-searched moves are stored with probability
  ``record_fast_prob * decided_full_search_prob / full_search_prob``, thinned by
  as much as its full searches (KataGo likewise records fewer positions from
  lopsided games); with the decided logic off, or at 1, no game is thinned
  further. Logged: ``selfplay/decided`` (the share of decisions made in decided
  games) and ``selfplay/sims_per_decision`` (the simulations a search ran, on
  top of what a re-rooted tree carried).

With ``obs.dice_state`` the self-play VecEnv, every search leaf and the replay
carry the dice input (:data:`catanrl.envs.encoding.DICE_STATE`: 127 more floats
on a 2p base row, 254 bytes of a replay position), and every iteration logs
``dice/weight_norm*`` and, from the first learning step on, the sensitivity
probe ``dice/probe_*`` on a fixed batch of replay positions that rides in the
trainer state (:mod:`catanrl.train.dice_probe`).

Every iteration also logs :class:`Monitors` (``oos/*``, ``search/overrule``
and friends, ``target/*`` with ``policy_target: gumbel`` or ``pruned``,
``drift/*``):
logging-only health checks that read the network, the searched rows and
replay and feed nothing back.
"""

from __future__ import annotations

import copy
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import cppanatron as cpp
import numpy as np
import torch
import torch.nn.functional as F

from catanrl.config import check_alphazero_kata
from catanrl.envs.encoding import (
    critic_level_of,
    dice_state_of,
    encoder_dims,
    info_level,
    make_encoder,
)
from catanrl.envs.puffer import POINTS_KEYS, STYLE_KEYS, plan_env
from catanrl.models import CatanModel, feature_names
from catanrl.models.graph import OUTCOME_TURNS
from catanrl.search import BatchEvaluator, make_pool, run_pools, search_config
from catanrl.search.pipeline import Group, Pipeline, run_groups
from catanrl.search.pool import DIRICHLET_ALPHA, NetworkCache
from catanrl.seeding import derive_seed, determinism, rng_state, seed_everything, set_rng_state
from catanrl.train.dice_probe import PROBE_ROWS as DICE_PROBE_ROWS
from catanrl.train.dice_probe import DiceProbe
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

#: Optional ``train`` keys and the values that mean "unset" (catanrl.config.TRAIN_OPTIONAL: absent
#: from configs/train/alphazero.yaml, and a config without them hashes as it did before they existed).
ALPHAZERO_OPTIONAL: dict[str, Any] = {"reuse": False, "outcome_vp_coef": 0.0, "outcome_turns_coef": 0.0}
#: KataGo's forced playouts and root policy temperature in the full searches (a PUCT root; the
#: engine's ``SearchConfig`` settings of the same names, where 0.0 and 1.0 are off).
ALPHAZERO_OPTIONAL["forced_playouts"] = 0.0
ALPHAZERO_OPTIONAL["root_policy_temperature"] = 1.0
#: The outcome losses' Huber delta, in the heads' units: quadratic within half the points to
#: win and within half of OUTCOME_TURNS, linear beyond (a 300-turn game does not swamp a batch).
OUTCOME_HUBER_DELTA = 0.5
#: The replay's outcome labels (a network with the outcome head), all integers.
OUTCOME_LABELS = ("final_vp", "vp_to_win", "turns_left", "outcome_known")

#: ``runtime.search_groups`` when a config leaves it unset: self-play's searches in two groups
#: that take turns on the device (SelfPlay; 1 is the serial loop, run_pools).
SEARCH_GROUPS = 2
#: More optional keys, after KataGo (the module docstring's four pieces; checked by
#: catanrl.config.check_alphazero_kata).
ALPHAZERO_OPTIONAL.update({
    "value_search_blend": 0.0,
    "value_blend_iterations": 0,
    "surprise_weight": 0.0,
    "surprise_cap": 0.0,
    "soft_policy_coef": 0.0,
    "soft_policy_temperature": 4.0,
    "decided_value": 0.0,
    "decided_lookback": 0,
    "decided_full_search_prob": 0.0,
    "decided_fast_sims": 0,
})
#: The share of the fast-searched moves that record_fast: true stores (1.0: all of them, no draw).
ALPHAZERO_OPTIONAL["record_fast_prob"] = 1.0
#: The replay's search labels (value_search_blend, surprise_weight): the full search's value
#: per slot, whether the position has one, and its policy surprise.
SEARCH_LABELS = ("search_value", "has_search_value", "surprise")


def check_train(train: Mapping[str, Any]) -> dict[str, Any]:
    missing = sorted(set(ALPHAZERO_KEYS) - set(train))
    unknown = sorted(set(train) - set(ALPHAZERO_KEYS) - set(ALPHAZERO_OPTIONAL))
    if missing or unknown:
        raise ValueError(f"alphazero train keys: missing {missing}, unknown {unknown}")
    train = dict(train)
    check_alphazero_kata(train)
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
    if train["kernels"] not in ("torch", "fused", "fused16"):
        raise ValueError(f"train.kernels must be torch, fused or fused16, not {train['kernels']!r}")
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
    reuse = train.get("reuse", ALPHAZERO_OPTIONAL["reuse"])
    if reuse is not True and reuse is not False:
        raise ValueError(f"train.reuse must be true or false, not {reuse!r}")
    if reuse and train["root"] != "puct":
        raise ValueError("train.reuse counts the visits a re-rooted tree carries towards a PUCT search's "
                         "budget; a Gumbel root plans its visits from an empty root (root: puct)")
    if reuse and int(train["dets"]) > 0:
        raise ValueError("train.reuse keeps one tree per game on the true game (dets: 0); sampled worlds "
                         "hold one seat's knowledge and do not carry over to the next seat's decision")
    check_root_exploration(train)
    for key in ("outcome_vp_coef", "outcome_turns_coef"):
        value = train.get(key, 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0.0:
            raise ValueError(f"train.{key} is a non-negative number, not {value!r}")
    return train


def check_root_exploration(train: Mapping[str, Any]) -> None:
    """train.forced_playouts and train.root_policy_temperature (optional): finite numbers, the
    first non-negative and the second positive, and set only for a PUCT root."""
    for key, positive in (("forced_playouts", False), ("root_policy_temperature", True)):
        value = train.get(key, ALPHAZERO_OPTIONAL[key])
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)
                or value < 0.0 or (positive and value == 0.0)):
            raise ValueError(f"train.{key} is a finite {'positive' if positive else 'non-negative'} number, "
                             f"not {value!r}")
        if value != ALPHAZERO_OPTIONAL[key] and train["root"] != "puct":
            raise ValueError(f"train.{key} shapes a PUCT root's selection; Sequential Halving plans a Gumbel "
                             "root's visits itself (root: puct)")


def root_exploration(train: Mapping[str, Any]) -> dict[str, float]:
    """The full searches' forced_playouts and root_policy_temperature, as ``search_config`` params
    and as a search's own options (``SearchPool.add`` / ``reroot``); off when unset."""
    return {key: float(train.get(key, unset)) for key, unset in ALPHAZERO_OPTIONAL.items()
            if key in ("forced_playouts", "root_policy_temperature")}


def outcome_coefs(train: Mapping[str, Any]) -> tuple[float, float]:
    """(outcome_vp_coef, outcome_turns_coef); 0 when unset."""
    return float(train.get("outcome_vp_coef", 0.0)), float(train.get("outcome_turns_coef", 0.0))


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


def root_log_priors(pool: cpp.SearchPool, sid: int) -> np.ndarray:
    """The log of search ``sid``'s root priors before any Dirichlet noise, one row per root world
    ([W, A], -inf off the root's actions): ``logits`` for :func:`pruned_visits` at a root the
    network was not asked about in this search (``reuse``: a re-rooted root was evaluated as a
    leaf of an earlier one). The tree's own float32 priors (``SearchPool.root_priors``), so
    within float32 rounding of the softmax of the network's logits that built them."""
    with np.errstate(divide="ignore"):  # zero off the root's actions
        return np.log(pool.root_priors(sid).astype(np.float64))


def search_value(weights: np.ndarray, q: np.ndarray) -> float | None:
    """A searched root's value for the seat to move: its action values ``q`` (the pool's
    ``q(id)``: NaN where unvisited) averaged by ``weights`` over the actions that have both, in
    [0, 1]. The weights are the visits the policy target keeps (:func:`pruned_visits`' for
    ``policy_target: pruned``, the pool's ``visits(id)`` otherwise), so the value is that of
    the play the network is taught, without the visits the root's noise forced. None: no
    action has a weight and a value."""
    weights, q = np.asarray(weights, dtype=np.float64), np.asarray(q, dtype=np.float64)
    seen = (weights > 0.0) & np.isfinite(q)
    if not seen.any():
        return None
    return float(np.clip((weights[seen] * q[seen]).sum() / weights[seen].sum(), 0.0, 1.0))


def policy_surprise(target: np.ndarray, log_priors: np.ndarray) -> float:
    """KL(policy target || the network's prior at the root), in nats: how far the search moved
    the policy (KataGo's policy surprise, which ``surprise_weight`` samples the replay by).
    ``target`` [A] is the position's policy target (normalized here) and ``log_priors`` the
    root's log-priors before any Dirichlet noise ([A] or one row per root world [W, A], whose
    priors are averaged; :func:`root_log_priors`), floored at :data:`MIN_LOG_PRIOR` where the
    target has an action the prior gives nothing. A target without mass gives 0."""
    target = np.asarray(target, dtype=np.float64)
    total = target.sum()
    if not total > 0.0:
        return 0.0
    prior = np.exp(np.atleast_2d(np.asarray(log_priors, dtype=np.float64))).mean(0)
    kept = target > 0.0
    share = target[kept] / total
    with np.errstate(divide="ignore"):
        log_prior = np.maximum(np.log(prior[kept]), MIN_LOG_PRIOR)
    return float(max((share * (np.log(share) - log_prior)).sum(), 0.0))


def surprise_probabilities(surprise: np.ndarray, weight: float, cap: float = 0.0) -> np.ndarray:
    """The replay's sampling distribution under ``surprise_weight`` s = ``weight`` (float64 [N]):
    ``(1 - s) / N + s * c_i / sum(c)`` with c_i position i's surprise, capped at ``cap`` times
    the mean surprise when ``cap`` > 0 (KataGo's policy surprise weighting, without its
    importance correction: surprising positions simply train more often). NaN marks a surprise
    that is not known (positions restored from a state that had none) and counts as the mean of
    the known ones. No surprise at all: uniform."""
    c = np.asarray(surprise, dtype=np.float64)
    known = ~np.isnan(c)
    c = np.where(known, c, c[known].mean() if known.any() else 0.0)
    if cap > 0.0:
        c = np.minimum(c, float(cap) * c.mean())
    total = c.sum()
    if not total > 0.0:
        return np.full(len(c), 1.0 / len(c))
    return (1.0 - float(weight)) / len(c) + float(weight) * c / total


def soft_policy_target(target: torch.Tensor, legal: torch.Tensor, temperature: float) -> torch.Tensor:
    """The soft policy head's target (KataGo's soft policy, as Canopy builds it): the policy
    target raised to 1 / ``temperature`` over the legal actions, ``((target + 1e-7) * legal) **
    (1 / T)`` normalized per row. The 1e-7 gives every legal action a share (0.018 of the best
    one's at T = 4), so the head also learns which moves are playable at all."""
    soft = ((target + 1e-7) * legal).pow(1.0 / float(temperature))
    return soft / soft.sum(1, keepdim=True).clamp_min(1e-12)


def value_blend(train: Mapping[str, Any], iteration: int) -> float:
    """The search value's share of the value target at the run's ``iteration`` (0 at a fresh
    start): ``value_search_blend``, ramped linearly from 0 over ``value_blend_iterations``."""
    ramp = int(train.get("value_blend_iterations", 0))
    return float(train.get("value_search_blend", 0.0)) * (min(1.0, iteration / ramp) if ramp else 1.0)


# ---------------------------------------------------------------- replay


class ReplayBuffer:
    """A FIFO of positions: float16 rows and policy targets, float32 value targets. With
    ``outcome`` (a network with the outcome head) also each position's outcome labels
    (:data:`OUTCOME_LABELS`): the final victory points of each slot (uint8, seat-relative), the
    points to win (uint8), the turns left until the game ended (uint16) and whether those are
    known (bool; False: a game cut at the turn limit). With ``search`` (value_search_blend,
    surprise_weight) also its search labels (:data:`SEARCH_LABELS`): the full search's value of
    each slot (float32, seat-relative like the value targets), whether it has one (bool) and
    its policy surprise (float32 nats; NaN: not known, see :meth:`restore`)."""

    def __init__(self, capacity: int, obs_size: int, num_actions: int, max_players: int, *,
                 outcome: bool = False, search: bool = False) -> None:
        self.capacity = int(capacity)
        self.shapes = (int(obs_size), int(num_actions), int(max_players))
        self.size = 0
        self.next = 0
        self.added = 0
        self.outcome = bool(outcome)
        self.search = bool(search)
        self.names = (("obs", "policy", "has_policy", "value", "present") + (OUTCOME_LABELS if outcome else ())
                      + (SEARCH_LABELS if search else ()))
        self.obs = self.policy = self.has_policy = self.value = self.present = None
        self.final_vp = self.vp_to_win = self.turns_left = self.outcome_known = None
        self.search_value = self.has_search_value = self.surprise = None

    def _allocate(self) -> None:
        obs_size, num_actions, players = self.shapes
        self.obs = np.zeros((self.capacity, obs_size), dtype=np.float16)
        self.policy = np.zeros((self.capacity, num_actions), dtype=np.float16)
        self.has_policy = np.zeros(self.capacity, dtype=bool)
        self.value = np.zeros((self.capacity, players), dtype=np.float32)
        self.present = np.zeros((self.capacity, players), dtype=np.float32)
        if self.outcome:
            self.final_vp = np.zeros((self.capacity, players), dtype=np.uint8)
            self.vp_to_win = np.zeros(self.capacity, dtype=np.uint8)
            self.turns_left = np.zeros(self.capacity, dtype=np.uint16)
            self.outcome_known = np.zeros(self.capacity, dtype=bool)
        if self.search:
            self.search_value = np.zeros((self.capacity, players), dtype=np.float32)
            self.has_search_value = np.zeros(self.capacity, dtype=bool)
            self.surprise = np.zeros(self.capacity, dtype=np.float32)

    def add(self, obs, policy, has_policy, value, present, outcome: Mapping[str, Any] | None = None,
            search: Mapping[str, Any] | None = None) -> None:
        """``outcome``: the positions' :data:`OUTCOME_LABELS` (a buffer with them takes them
        with every add; one without takes none). ``search``: their :data:`SEARCH_LABELS`, under
        the same rule."""
        if self.outcome != (outcome is not None):
            raise ValueError("a replay buffer with outcome labels takes them with every add, and one without "
                             "takes none")
        if self.search != (search is not None):
            raise ValueError("a replay buffer with search labels takes them with every add, and one without "
                             "takes none")
        if self.obs is None:
            self._allocate()
        count = len(obs)
        keep = slice(max(0, count - self.capacity), count)  # more than fits: the newest
        slots = (self.next + np.arange(count)[keep] - keep.start) % self.capacity
        for name, values in (("obs", obs), ("policy", policy), ("has_policy", has_policy), ("value", value),
                             ("present", present), *((name, outcome[name]) for name in OUTCOME_LABELS
                                                     if outcome is not None),
                             *((name, search[name]) for name in SEARCH_LABELS if search is not None)):
            getattr(self, name)[slots] = np.asarray(values)[keep]
        self.next = (self.next + len(slots)) % self.capacity
        self.size = min(self.size + len(slots), self.capacity)
        self.added += count

    def batch(self, index: np.ndarray, device: str) -> dict[str, torch.Tensor]:
        def tensor(array, dtype=torch.float32, stored=None):
            rows = array[index] if stored is None else array[index].astype(stored)
            return torch.from_numpy(np.ascontiguousarray(rows)).to(device=device, dtype=dtype)

        out = {"obs": tensor(self.obs), "policy": tensor(self.policy), "has_policy": tensor(self.has_policy, torch.bool),
               "value": tensor(self.value), "present": tensor(self.present)}
        if self.outcome:  # the labels as floats, in their own units (victory points, turns)
            out.update({"final_vp": tensor(self.final_vp), "vp_to_win": tensor(self.vp_to_win),
                        "turns_left": tensor(self.turns_left, stored=np.int32),
                        "outcome_known": tensor(self.outcome_known, torch.bool)})
        if self.search:
            out.update({"search_value": tensor(self.search_value),
                        "has_search_value": tensor(self.has_search_value, torch.bool)})
        return out

    def state(self) -> dict[str, Any]:
        """The buffer for the trainer state, oldest first (``weights_only``-loadable)."""
        if self.size == 0:
            return {"size": 0, "added": self.added}
        order = np.arange(self.next - self.size, self.next) % self.capacity
        # turns_left as int32: torch has no uint16 that every version loads
        arrays = {name: torch.from_numpy(np.ascontiguousarray(
                      getattr(self, name)[order].astype(np.int32 if name == "turns_left" else getattr(self, name).dtype)))
                  for name in self.names}
        return {"size": self.size, "added": self.added, **arrays}

    def restore(self, state: Mapping[str, Any]) -> None:
        self.size = self.next = 0
        self.added = int(state["added"])
        count = int(state["size"])
        if not count:
            return
        self._allocate()
        keep = slice(max(0, count - self.capacity), count)
        for name in self.names:
            if name in OUTCOME_LABELS and name not in state:
                continue  # a state from before the labels: its positions have none (outcome_known False)
            if name in SEARCH_LABELS and name not in state:
                # a state from before the search labels: its positions have no search value, and the
                # surprise of those with a policy target is not known (NaN: the mean of the known ones)
                if name == "surprise":
                    self.surprise[: len(state["has_policy"][keep])] = np.where(
                        state["has_policy"][keep].numpy(), np.nan, 0.0)
                continue
            values = state[name][keep].numpy()
            getattr(self, name)[: len(values)] = values
        self.size = min(count, self.capacity)
        self.next = self.size % self.capacity


# ---------------------------------------------------------------- self-play


def make_vec(plan: Any, levels: Mapping[str, Any], *, num_envs: int, seed: int, threads: int,
             trackers: bool = False) -> cpp.VecEnv:
    """A VecEnv of ``plan``'s scenarios writing the rows of a network at ``levels`` (with the
    dice input when the network reads it, ``obs.dice_state``)."""
    critic = critic_level_of(levels["level"], levels["critic_level"])
    return cpp.VecEnv(plan.scenarios, num_envs=num_envs, dims=encoder_dims(plan.dims),
                      level=info_level(levels["level"]), seed=seed, num_threads=threads,
                      critic_level=None if critic is None else info_level(critic), trackers=trackers,
                      dice_state=dice_state_of(levels))


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
    #: The game's turn count at the position (read with the outcome head only: its turns-left label).
    turn: int = 0
    #: The full search's value for the row's seat (value_search_blend; None: it has none) and
    #: its policy surprise (surprise_weight; 0 without a policy target).
    search_value: float | None = None
    surprise: float = 0.0


class SelfPlay:
    """The VecEnv, two search pools (full and fast budgets) and the games' pending positions.
    With ``reuse`` one pool for both budgets, each search with its own, and every game's tree.

    ``groups`` (``runtime.search_groups``) splits the games into that many search groups, each
    with its own pools (game e searches in group e % groups, so a kept tree stays in its pool),
    and runs a step's searches with :func:`~catanrl.search.pipeline.run_groups`: the device
    evaluates one group's leaves while the engine collects and applies the others'. One group is
    ``run_pools``. The searches are the same either way (pipeline.py says to which bits).

    ``cache_mb`` (``runtime.eval_cache_mb``; 0: none) is the megabytes of the evaluation cache
    every pool shares (:class:`~catanrl.search.pool.NetworkCache`): leaves whose rows the
    network has evaluated since its weights last changed take no row in the batch. Every step
    starts by clearing it unless the evaluator and its network's weights are the ones it was
    filled with, so no caller has to say when they changed."""

    def __init__(self, plan: Any, train: Mapping[str, Any], *, model: CatanModel, num_envs: int, threads: int,
                 seed: int, device: str, mode: str, groups: int = 1, cache_mb: float = 0.0) -> None:
        if int(groups) < 1:
            raise ValueError("runtime.search_groups must be at least 1")
        if float(cache_mb) < 0:
            raise ValueError("runtime.eval_cache_mb must be 0 (no cache) or the cache's megabytes")
        self.train = dict(train)
        self.seed = int(seed)
        self.dims = dict(plan.dims)
        levels = model.levels
        self.dets = int(train["dets"])
        self.vec = make_vec(plan, levels, num_envs=num_envs, seed=self.seed, threads=threads, trackers=self.dets > 0)
        self.buffers = self.vec.allocate()
        self.vec.reset(self.seed)
        self.players = int(self.vec.max_players)
        #: The replay takes each full search's value (value_search_blend) and policy surprise
        #: (surprise_weight) as its search labels (SEARCH_LABELS).
        self.record_value = float(train.get("value_search_blend", 0.0)) > 0.0
        self.record_surprise = float(train.get("surprise_weight", 0.0)) > 0.0
        self.search = self.record_value or self.record_surprise
        #: decided_value (0: off): a game is decided once its last decided_lookback searches gave
        #: one seat at least this value. Each game's (leader, its value) of those searches.
        self.decided = float(train.get("decided_value", 0.0))
        self.leaders = [deque(maxlen=max(int(train.get("decided_lookback", 0)), 1)) for _ in range(num_envs)]
        if (self.record_value or self.decided) and self.players > 2:
            raise ValueError("train.value_search_blend and train.decided_value read the search's value for both "
                             "seats of a 2-player game (the engine keeps a root action's value for the seat to "
                             f"move alone); these games seat up to {self.players}")
        self.model, self.device, self.mode = model, device, mode
        self.evaluator = self._evaluator()
        #: The pools' evaluation cache (None: none), synced with the evaluator at every step.
        self.cache = (NetworkCache(self.evaluator.num_actions, self.evaluator.num_players, float(cache_mb))
                      if float(cache_mb) > 0 else None)

        def pool(sims: int, pending: int, noise: bool, **explore: float) -> cpp.SearchPool:
            params = {"sims": sims, "dets": self.dets, "oracle": self.dets == 0, "root": train["root"],
                      "c_puct": train["c_puct"], "gumbel_m": train["gumbel_m"], "pending": pending, "noise": noise}
            params.update(explore)
            config = search_config(params, max_players=self.players, threads=threads)
            return make_pool(self.dims, levels["level"], config, critic_level=levels["critic_level"],
                             dice_state=dice_state_of(levels), cache=self.cache.table if self.cache else None)

        #: Leaves a search sends per round, by pool (full, fast).
        self.leaves = int(train["pending"]), int(train["fast_pending"])
        noise = bool(train["dirichlet_noise"]) and train["root"] == "puct"
        #: reuse: a game's tree serves full and fast searches alike, so both run in one pool, each
        #: search with its own budget, leaves per round and noise (``options``: full, fast).
        self.reuse = bool(train.get("reuse", False))
        #: The full searches' forced playouts and root policy temperature (off in the fast ones).
        self.explore = root_exploration(train)


        def pools() -> list[cpp.SearchPool]:
            full = pool(int(train["sims"]), self.leaves[0], noise, **self.explore)
            # fast searches: no noise, forced playouts or root policy temperature (KataGo)
            return [full] if self.reuse else [full, pool(int(train["fast_sims"]), self.leaves[1], False)]

        #: Each search group's pools: [full, fast], or with reuse the one pool of both. The first
        #: group's are full_pool and fast_pool.
        self.groups = [pools() for _ in range(int(groups))]
        self.full_pool, self.fast_pool = self.groups[0][0], self.groups[0][-1]
        #: Every pool, group after group.
        self.pools = [pool for pools in self.groups for pool in pools]
        self.batches: dict = {}  # run_groups' page-locked rows, kept across steps and evaluators
        self.options = tuple(
            {"simulations": int(train[sims]), "max_pending": leaves, "dirichlet_alpha": DIRICHLET_ALPHA * noisy}
            for sims, leaves, noisy in (("sims", self.leaves[0], noise), ("fast_sims", self.leaves[1], False)))
        self.options[0].update(self.explore)
        self.options[1].update(root_exploration({}))  # the pool's are the full searches'
        #: reuse: each game's last search, kept for its next decision: (pool, search id, the game
        #: it searched, the action played).
        self.trees: dict[int, tuple[cpp.SearchPool, int, cpp.Game, int]] = {}
        self.codec = make_encoder(self.dims, levels["level"]).action_space if self.reuse else None
        # a decided game's fast search: the fast search at decided_fast_sims (unset: fast_sims, the same)
        self.decided_options = {**self.options[1], "simulations": int(train.get("decided_fast_sims", 0))
                                or int(train["fast_sims"])}
        #: policy_target: gumbel's sigma constants (None: the engine's target, or the pruned one).
        self.sigma = None if train["policy_target"] != "gumbel" else {
            "c_visit": float(train["target_c_visit"]), "c_scale": float(train["target_c_scale"]),
            "rescale": bool(train["target_rescale"]), "prior_temperature": float(train["target_prior_temperature"])}
        self.mask = slice(model.mask_offset, model.mask_offset + model.num_actions)
        self.pending: list[list[_Pending]] = [[] for _ in range(num_envs)]
        #: The outcome head's labels are recorded (model.args.outcome), and the games each env
        #: has finished as this object saw them (against VecEnv.last_outcome's count).
        self.outcome = bool(getattr(model, "outcome", False))
        self.finished = np.zeros(num_envs, dtype=np.int64)
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
        the fused kernels read copies made when the evaluator was built (FusedGraph, FusedSeat), so
        ``kernels: fused`` rebuilds it, its CUDA graphs included (``fused16`` too: its halves
        are converted from those copies)."""
        if self.train["kernels"] != "torch":
            self.evaluator = None  # its graphs and static rows go before the new ones are captured
            self.evaluator = self._evaluator()
        if self.cache is not None:  # (step() would: the weights are not the ones it was filled with)
            self.cache.sync(self.evaluator)

    def check_kernels(self, rows: int = 64) -> None:
        """``kernels: fused``: the evaluator's values against the module's own on acting rows. The
        kernels are within 1e-6 of the module (graph_kernels.py; the seat-set network's too,
        seat_kernels.py); an evaluator left on older weights is not, and self-play would search
        with them unnoticed.

        ``kernels: fused16``: the same check on the float32 kernels over the copies its halves were
        converted from (the evaluator's ``exact``), as sharp as fused's, then the half-precision
        values themselves, which rounding moves by more (at most 1.5e-3 on 53,000 real rows of two
        networks; graph_kernels.py): within 1e-2."""
        kernels = self.train["kernels"]
        if kernels == "torch":
            return
        obs = self.buffers["observations"][np.flatnonzero(self.buffers["acting"])[:rows]]
        with torch.inference_mode():
            want = self.model(torch.from_numpy(obs).to(self.device))[1].float().cpu().numpy()
        exact = self.evaluator(obs)[1] if kernels == "fused" else self.evaluator.exact(obs)[1]
        gap = float(np.abs(exact - want).max()) if len(obs) else 0.0
        if gap > 1e-4:
            raise RuntimeError(f"kernels: {kernels}: the evaluator's values are {gap:.2g} from the network's "
                               "(its weights are stale: refresh() after every change)")
        if kernels == "fused16":
            gap = float(np.abs(self.evaluator(obs)[1] - want).max()) if len(obs) else 0.0
            if gap > 1e-2:
                raise RuntimeError(f"kernels: fused16: the half-precision values are {gap:.2g} from the "
                                   "network's (rounding moves them by about 1e-3: train with kernels: fused)")

    def pool_of(self, env: int, full: bool) -> cpp.SearchPool:
        """The pool that searches game ``env`` at the full or the fast budget."""
        return self.groups[env % len(self.groups)][0 if full else -1]

    def _search(self, pool: cpp.SearchPool, env: int, full: bool, seed: int, stats: dict[str, float]) -> int:
        """``reuse``: game ``env``'s search for the decision it is at. Its kept tree, re-rooted at the
        action it played, when the game is that tree's game plus the action and nothing else (a
        chance outcome, a forced move the VecEnv played, a reset) and the engine re-roots there (a
        searched, deterministic action); a new search otherwise, the kept tree dropped. Either way
        with this decision's budget and noise: a re-rooted root's visits count towards the budget
        (a fast search of a decided game: decided_fast_sims)."""
        game, options = self.vec.game(env), self.options[0 if full else 1]
        if not full and self._decided(env):
            options = self.decided_options
        kept = self.trees.pop(env, None)
        sid = None
        if kept is not None:
            owner, sid, previous, chosen = kept
            expected = previous.copy()
            expected.execute(self.codec.decode(previous, chosen))
            if owner is pool and expected.search_equivalent(game) and pool.reroot(sid, chosen, **options):
                carried = min(int(pool.visits(sid).sum()), options["simulations"]) / options["simulations"]
                stats["reused"] = stats.get("reused", 0.0) + 1.0
                key = "carried_full" if full else "carried_fast"
                stats[key] = stats.get(key, 0.0) + carried
            else:
                owner.remove(sid)
                sid = None
        if sid is None:
            sid = pool.add(game, seed=seed, **options)
        self.trees[env] = (pool, sid, game, -1)  # the action: once it is chosen
        return sid

    def drop_trees(self) -> None:
        """``reuse``: forget every game's tree (the end of an iteration: the network that built
        them is about to change), so each game's next search starts empty."""
        for pool, sid, _, _ in self.trees.values():
            pool.remove(sid)
        self.trees.clear()

    def step(self, iteration: int, step: int, rng: np.random.Generator, replay: ReplayBuffer,
             stats: dict[str, float]) -> None:
        """Search every acting row, play the chosen moves and file finished games."""
        b = self.buffers
        acting = np.flatnonzero(b["acting"])
        # decided_value: the rows whose game is decided, searched in full with a probability of their own
        decided = np.array([self._decided(int(env)) for env in b["env_index"][acting]], dtype=bool)
        full = rng.random(len(acting)) < np.where(decided, float(self.train.get("decided_full_search_prob", 0.0)),
                                                  float(self.train["full_search_prob"]))
        # record_fast: the fast-searched rows that enter the replay too. record_fast_prob below 1 draws
        # them (no draw at 1), in a decided game at the probability scaled as its full searches' is
        stored = np.full(len(acting), bool(self.train["record_fast"]))
        keep = float(self.train.get("record_fast_prob", 1.0))
        if self.train["record_fast"] and keep < 1.0:
            scale = float(self.train.get("decided_full_search_prob", 0.0)) / float(self.train["full_search_prob"])
            stored = rng.random(len(acting)) < np.where(decided, keep * scale, keep)
        stored |= full
        searches = []
        targets = []  # the full searches' own policy targets (pool.policy_target), in acting order, for the monitor
        rebuilt = []  # policy_target: gumbel's or pruned's targets of the same searches (the ones trained on), and
        pruned = []  # pruned's share of each of those searches' root visits that it removed, for the monitor
        groups = len(self.groups)
        rebuild = self.train["policy_target"] != "engine"
        answered = 0
        if self.cache is not None:  # before any search: nothing of other weights is served
            self.cache.sync(self.evaluator)
            answered = self.cache.answered()
        # reuse: a re-rooted root is not evaluated again, so its prior is read from the tree instead
        roots = [RootOutputs(pools[0], self.mask) for pools in self.groups] if rebuild and not self.reuse else None
        try:
            rows = {id(pool): 0 for pool in self.pools}
            for i, row in enumerate(acting):
                env, seat = int(b["env_index"][row]), int(b["seat"][row])
                pool = self.pool_of(env, bool(full[i]))
                rows[id(pool)] += self.leaves[0 if full[i] else 1]
                tracker = self.vec.tracker(env, seat) if self.dets else None
                seed = derive_seed(self.seed, "selfplay", iteration, step, env)
                if self.reuse:
                    searches.append((pool, self._search(pool, env, bool(full[i]), seed, stats)))
                    continue
                searches.append((pool, pool.add(self.vec.game(env), seed=seed, tracker=tracker)))
                if roots is not None and full[i]:
                    roots[env % groups].expect(searches[-1][1])
            evaluator = Rounds(self.evaluator)
            # each group's pools (the full one behind its RootOutputs) and their rows a round: every search's
            # quota (collect caps a search by the room left in its buffer)
            batches = [([pools[0] if roots is None else roots[g], *pools[1:]], [rows[id(pool)] for pool in pools])
                       for g, pools in enumerate(self.groups)]
            if groups == 1:
                leaves = run_pools(batches[0][0], evaluator, batch_rows=batches[0][1])
            else:
                leaves, evaluator.count = run_groups([Group(*batch) for batch in batches],
                                                     Pipeline(self.evaluator, self.batches))
            for i, row in enumerate(acting):
                env, seat = int(b["env_index"][row]), int(b["seat"][row])
                pool, sid = searches[i]
                policy = None
                kept = None  # policy_target: pruned's visits, which also weigh the search value
                searched, surprise = None, 0.0  # the search's value for the seat to move, its policy surprise
                if full[i]:
                    if not rebuild:
                        policy = pool.policy_target(sid).astype(np.float16)
                        targets.append(policy)
                    else:  # the root's actions: the real game's (the row's mask) that every world offers
                        if roots is None:  # reuse (pruned; the true game is the one world)
                            legal, logits = b["observations"][row, self.mask] > 0, root_log_priors(pool, sid)
                        else:
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
                    if self.record_value or self.decided:  # by the visits the policy target keeps
                        searched = search_value(pool.visits(sid) if kept is None else kept, pool.q(sid))
                    if self.record_surprise:
                        surprise = policy_surprise(policy, root_log_priors(pool, sid))
                elif self.decided:
                    searched = search_value(pool.visits(sid), pool.q(sid))
                if self.decided:  # the game's leader by this search (no value: nobody leads)
                    mover = 0.5 if searched is None else searched
                    self.leaders[env].append((seat if mover >= 0.5 else 1 - seat, max(mover, 1.0 - mover)))
                    stats["decided"] = stats.get("decided", 0.0) + float(decided[i])
                    stats["simulations"] = stats.get("simulations", 0.0) + float(pool.stats(sid).simulations)
                b["actions"][row] = pool.select(sid, temperature(self.train, int(self.ply[env])))
                if self.reuse:
                    self.trees[env] = (*self.trees[env][:3], int(b["actions"][row]))
                if stored[i]:
                    self.pending[env].append(_Pending(b["observations"][row].astype(np.float16), policy, seat,
                                                      int(b["num_players"][env]),
                                                      int(self.vec.num_turns(env)) if self.outcome else 0,
                                                      searched if self.record_value and full[i] else None,
                                                      surprise))
                self.ply[env] += 1
        except BaseException:
            self.trees.clear()  # a step that failed keeps no tree: the searches go below
            raise
        finally:
            kept = {(id(pool), sid) for pool, sid, _, _ in self.trees.values()}
            for pool in self.pools:
                if pool.pending:
                    pool.cancel()
                for sid in pool.ids:
                    if (id(pool), sid) not in kept:
                        pool.remove(sid)
        if self.cache is not None:  # the leaves that took no row: counted as leaves all the same
            answered = self.cache.answered() - answered
            stats["cached"] = stats.get("cached", 0.0) + answered
        stats["leaves"] += leaves + answered
        stats["rounds"] = stats.get("rounds", 0.0) + evaluator.count
        stats["full"] += int(full.sum())
        stats["recorded_fast"] = stats.get("recorded_fast", 0.0) + int((stored & ~full).sum())
        self.decisions += len(acting)
        if self.monitor is not None and targets:  # before step() rewrites rows
            self.monitor.searched(b["observations"][acting[full]], np.stack(targets),
                                  np.stack(rebuilt) if rebuilt else None, np.array(pruned) if pruned else None)
        self.vec.step()
        self._finish(replay)

    def _decided(self, env: int) -> bool:
        """Whether ``env``'s game is decided at its next decision (``decided_value``): its last
        ``decided_lookback`` searches, full or fast, all gave one seat a value of at least
        ``decided_value``. A game with fewer searches so far is not; never when off."""
        recent = self.leaders[env]
        return bool(self.decided and len(recent) == recent.maxlen and len({leader for leader, _ in recent}) == 1
                    and min(value for _, value in recent) >= self.decided)

    def _last_outcome(self, env: int) -> dict[str, Any]:
        """``VecEnv.last_outcome`` of the game ``env`` just finished, with ``known``: its end
        labels that game's positions. Not when the game was cut at the turn limit (no winner,
        no end to count turns to), nor when it is not the one game the env finished since the
        last terminal seen here."""
        last = dict(self.vec.last_outcome(env))
        last["known"] = not last["truncated"] and int(last["games"]) == int(self.finished[env]) + 1
        self.finished[env] = int(last["games"])
        return last

    def _finish(self, replay: ReplayBuffer) -> None:
        b = self.buffers
        players = self.players
        for env in np.flatnonzero(b["terminals"].reshape(-1, players)[:, 0]):
            rows = self.pending[env]
            self.pending[env] = []
            self.ply[env] = 0
            self.leaders[env].clear()
            self.games += 1
            if int(env) in self.trees:  # reuse: the game is over (won or truncated) and reset
                pool, sid, _, _ = self.trees.pop(int(env))
                pool.remove(sid)
            last = self._last_outcome(int(env)) if self.outcome else None
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
                outcome_labels(rows, last, players) if last is not None else None,
                search_labels(rows, players) if self.search else None,
            )


def outcome_labels(rows: list[_Pending], last: Mapping[str, Any], max_players: int) -> dict[str, np.ndarray]:
    """The :data:`OUTCOME_LABELS` of a finished game's positions from ``VecEnv.last_outcome``
    (``last``; ``known: False`` in it: no labels, all masked). Slot k of ``final_vp`` is the
    seat k places after the row's (the value targets' order)."""
    count = len(rows)
    labels = {"final_vp": np.zeros((count, max_players), np.uint8), "vp_to_win": np.zeros(count, np.uint8),
              "turns_left": np.zeros(count, np.uint16), "outcome_known": np.zeros(count, bool)}
    if not last.get("known", True):
        return labels
    points = np.asarray(last["victory_points"], dtype=np.int64)
    for i, row in enumerate(rows):
        n = row.num_players
        labels["final_vp"][i, :n] = np.clip(points[(row.seat + np.arange(n)) % n], 0, 255)
        labels["turns_left"][i] = min(max(int(last["turns"]) - row.turn, 0), 65535)
    labels["vp_to_win"][:] = min(int(last["vps_to_win"]), 255)
    labels["outcome_known"][:] = True
    return labels


def search_labels(rows: list[_Pending], max_players: int) -> dict[str, np.ndarray]:
    """The :data:`SEARCH_LABELS` of a game's positions (2 players where a row has a search
    value): the row's seat has the search's value v and the other seat 1 - v, in the value
    targets' slot order."""
    count = len(rows)
    labels = {"search_value": np.zeros((count, max_players), np.float32),
              "has_search_value": np.zeros(count, bool), "surprise": np.zeros(count, np.float32)}
    for i, row in enumerate(rows):
        labels["surprise"][i] = row.surprise
        if row.search_value is not None:
            labels["search_value"][i, :2] = row.search_value, 1.0 - row.search_value
            labels["has_search_value"][i] = True
    return labels


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


def outcome_losses(vp: torch.Tensor, turns: torch.Tensor, batch: Mapping[str, torch.Tensor]
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The outcome head's two Huber losses on a replay batch and their mean absolute errors in
    victory points and in turns, over the labelled rows (``outcome_known``; the victory points
    over the seated slots of those): ``vp`` [B, P] is the head's final victory points over the
    points to win, ``turns`` [B] its turns left over OUTCOME_TURNS. No labelled row: zeros."""
    known = batch["outcome_known"].float()
    seats = batch["present"] * known[:, None]
    to_win = batch["vp_to_win"].clamp_min(1.0)[:, None]
    vp_gap = vp - batch["final_vp"] / to_win
    turns_gap = turns - batch["turns_left"] / OUTCOME_TURNS
    zeros = torch.zeros_like
    vp_loss = (F.huber_loss(vp_gap, zeros(vp_gap), reduction="none", delta=OUTCOME_HUBER_DELTA) * seats).sum()
    turns_loss = (F.huber_loss(turns_gap, zeros(turns_gap), reduction="none", delta=OUTCOME_HUBER_DELTA) * known).sum()
    n_seats, n_rows = seats.sum().clamp_min(1.0), known.sum().clamp_min(1.0)
    with torch.no_grad():
        vp_mae = (vp_gap.abs() * to_win * seats).sum() / n_seats
        turns_mae = (turns_gap.abs() * known).sum() * OUTCOME_TURNS / n_rows
    return vp_loss / n_seats, turns_loss / n_rows, vp_mae, turns_mae


def learn(model: CatanModel, optimizer: torch.optim.Optimizer, replay: ReplayBuffer, train: Mapping[str, Any],
          device: str, rng: np.random.Generator, iteration: int = 0) -> dict[str, float]:
    """``updates_per_iteration`` gradient steps on uniform minibatches; their mean statistics.
    A network with the outcome head adds its two losses (:func:`outcome_losses`) at
    ``outcome_vp_coef`` and ``outcome_turns_coef``, and one with the soft policy head its
    cross-entropy to the soft target (:func:`soft_policy_target`) at ``soft_policy_coef``.
    ``surprise_weight`` draws the minibatches by policy surprise instead
    (:func:`surprise_probabilities`, built once from the buffer as it is), and
    ``value_search_blend`` mixes the search value into the value target of the rows that have
    one, at its weight for the run's ``iteration`` (:func:`value_blend`)."""
    model.train()
    outcome = bool(getattr(model, "outcome", False))
    vp_coef, turns_coef = outcome_coefs(train)
    soft = bool(getattr(model, "soft_policy", False))
    soft_coef = float(train.get("soft_policy_coef", 0.0))
    soft_temperature = float(train.get("soft_policy_temperature", ALPHAZERO_OPTIONAL["soft_policy_temperature"]))
    mask = slice(model.mask_offset, model.mask_offset + model.num_actions)
    blended = float(train.get("value_search_blend", 0.0)) > 0.0
    blend = value_blend(train, iteration)
    surprise = float(train.get("surprise_weight", 0.0))
    if (blended or surprise > 0.0) and not replay.search:
        raise ValueError("train.value_search_blend and train.surprise_weight read the replay's search labels "
                         "(ReplayBuffer(search=True))")
    sums: dict[str, float] = {}
    players = replay.shapes[2]
    by_n = np.zeros((players + 1, 3))  # count, predicted seat-sum, target seat-sum
    updates = int(train["updates_per_iteration"])
    cumulative = None
    if surprise > 0.0:  # the sampling distribution over the buffer's rows, and N * p of the rows drawn
        p = surprise_probabilities(replay.surprise[: replay.size], surprise, float(train.get("surprise_cap", 0.0)))
        cumulative = np.cumsum(p)
        drawn = []
    for _ in range(updates):
        if cumulative is None:
            batch = replay.batch(rng.integers(0, replay.size, int(train["minibatch_size"])), device)
        else:
            index = np.searchsorted(cumulative, rng.random(int(train["minibatch_size"])) * cumulative[-1], side="right")
            index = np.minimum(index, replay.size - 1)
            drawn.append(p[index] * replay.size)
            batch = replay.batch(index, device)
        if outcome or soft:
            logits, value_logits, _, extras = model.training_heads(batch["obs"])
        else:
            logits, value_logits, _ = model.heads(batch["obs"])
        logp = F.log_softmax(logits.float(), dim=1)
        has = batch["has_policy"]
        target = batch["policy"][has]
        target = target / target.sum(1, keepdim=True).clamp_min(1e-8)
        policy_loss = -(target * logp[has]).sum(1).mean() if bool(has.any()) else logp.new_zeros(())
        present = batch["present"]
        value_target = batch["value"]
        extra: tuple[tuple[str, torch.Tensor], ...] = ()
        if blended:  # the rows with a search value: the game result and it; the others: the game result
            searched = batch["has_search_value"]
            value_target = torch.where(searched[:, None],
                                       (1.0 - blend) * batch["value"] + blend * batch["search_value"], batch["value"])
            gap = ((batch["search_value"][:, 0] - batch["value"][:, 0]).abs() * searched).sum()
            extra = (("value_blend", blend), ("search_value_gap", gap / searched.sum().clamp_min(1.0)))
        value_loss = (F.binary_cross_entropy_with_logits(value_logits.float(), value_target, reduction="none")
                      * present).sum() / present.sum().clamp_min(1.0)
        loss = policy_loss + float(train["value_coef"]) * value_loss
        if outcome:
            vp_loss, turns_loss, vp_mae, turns_mae = outcome_losses(extras["final_vp"], extras["turns_left"], batch)
            loss = loss + vp_coef * vp_loss + turns_coef * turns_loss
            extra += (("outcome_vp_loss", vp_loss), ("outcome_turns_loss", turns_loss), ("outcome_vp_mae", vp_mae),
                      ("outcome_turns_mae", turns_mae), ("outcome_known", batch["outcome_known"].float().mean()))
        if soft:  # on the rows with a policy target, over the row's legal actions (its mask block)
            soft_target = soft_policy_target(target, batch["obs"][has][:, mask] > 0, soft_temperature)
            soft_logp = F.log_softmax(extras["soft_policy"].float(), dim=1)[has]
            soft_loss = -(soft_target * soft_logp).sum(1).mean() if bool(has.any()) else logp.new_zeros(())
            loss = loss + soft_coef * soft_loss
            entropy = -(soft_target * soft_target.clamp_min(1e-12).log()).sum(1)
            extra += (("soft_policy_loss", soft_loss),
                      ("soft_target_entropy", entropy.mean() if bool(has.any()) else logp.new_zeros(())))
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
            for key, value in (("policy_loss", policy_loss), ("value_loss", value_loss),
                               ("policy_rows", has.float().mean()), *extra):
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
    if cumulative is not None:
        drawn = np.concatenate(drawn)
        known = replay.surprise[: replay.size]
        known = known[~np.isnan(known)]
        sums.update({"surprise_sampled_mean": float(drawn.mean()), "surprise_sampled_max": float(drawn.max()),
                     "surprise_ess": float(1.0 / (replay.size * (p * p).sum())),
                     "surprise_mean": float(known.mean()) if len(known) else 0.0})
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
    names = feature_names(model.dims, model.levels["level"], dice_state_of(model.levels))["global"]
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
      ``target_sum_{n}p`` and their gap ``sum_gap_{n}p``; ``rows``. With the outcome head
      (``model.args.outcome``) also ``outcome_vp_mae`` and ``outcome_turns_mae``, its mean
      absolute errors in victory points (per seated slot) and in turns on those positions'
      labelled rows (``outcome_rows`` of them).
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
            if replay.outcome and getattr(self.model, "outcome", False):
                logs.update(self._outcome(replay, index))
        self.seconds += time.time() - start
        return logs

    def _outcome(self, replay: ReplayBuffer, index: np.ndarray) -> dict[str, float]:
        """``oos/outcome_*``: the outcome head's mean absolute errors on the replay rows ``index``."""
        vp = turns = seats = rows = 0.0
        with torch.inference_mode():
            for first in range(0, len(index), self.rows):
                batch = replay.batch(index[first : first + self.rows], self.device)
                known = batch["outcome_known"].float()
                if not bool(known.any()):
                    continue
                _, _, _, final_vp, turns_left = self.model.heads_with_outcome(batch["obs"])
                _, _, vp_mae, turns_mae = outcome_losses(final_vp, turns_left, batch)
                n_seats, n_rows = float((batch["present"] * known[:, None]).sum()), float(known.sum())
                vp, turns = vp + float(vp_mae) * n_seats, turns + float(turns_mae) * n_rows
                seats, rows = seats + n_seats, rows + n_rows
        if not rows:
            return {"oos/outcome_rows": 0}
        return {"oos/outcome_rows": int(rows), "oos/outcome_vp_mae": vp / max(seats, 1.0),
                "oos/outcome_turns_mae": turns / rows}

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


def parents_state(cfg: Mapping[str, Any], state: Mapping[str, Any]) -> bool:
    """Whether ``state`` is the one catanrl fork copied from the parent: a continued run
    (``restore: weights+optimizer``) that has not committed yet, so the state is at the epoch of
    the checkpoint it continues. Such a launch may play another map pool than the parent's (a
    fork with ``--set env.scenarios=...``): the replay holds positions, not boards, and the
    fork's first commit stores its own pool, which its resumes must then match."""
    init = cfg.get("init_from") or {}
    return (init.get("restore") == "weights+optimizer"
            and str(init.get("ckpt", "")).partition("#")[0].endswith(f"@e{int(state['epoch'])}"))


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
    if (max(outcome_coefs(train)) > 0.0) != bool(getattr(model, "outcome", False)):
        raise ValueError("train.outcome_vp_coef / outcome_turns_coef weigh the outcome head's losses: one of them "
                         "is above 0 exactly when the network has the head (model.args.outcome)")
    if (float(train.get("soft_policy_coef", 0.0)) > 0.0) != bool(getattr(model, "soft_policy", False)):
        raise ValueError("train.soft_policy_coef weighs the soft policy head's loss: it is above 0 exactly when "
                         "the network has the head (model.args.soft_policy)")
    device = resolve_device(runtime.get("device", "auto"))
    model.to(device).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train["learning_rate"]),
                                  weight_decay=float(train["weight_decay"]))
    selfplay = SelfPlay(plan, train, model=model, num_envs=int(env_cfg["num_envs"]),
                        threads=int(runtime.get("threads", 1)), seed=derive_seed(seed, "rollout", launch),
                        device=device, mode="strict" if strict else "fast",
                        groups=int(runtime.get("search_groups") or SEARCH_GROUPS),
                        cache_mb=float(runtime.get("eval_cache_mb") or 0.0))
    replay = ReplayBuffer(int(train["replay_capacity"]), selfplay.vec.observation_size, selfplay.vec.action_size,
                          selfplay.players, outcome=selfplay.outcome, search=selfplay.search)
    monitors = Monitors(model, device=device, rows=int(train["search_rows"])) if monitoring else None
    selfplay.monitor = monitors
    # obs.dice_state: the dice input's usage checks (dice/*, catanrl.train.dice_probe), every iteration
    dice_probe = DiceProbe(model) if dice_state_of(model.levels) else None
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
            **(dice_probe.state() if dice_probe is not None else {}),
        })
        if selector.due(iteration):
            selector.run(ref, model)

    state = writer.load_trainer_state() if writer.mode == "resume" else None
    if state is not None:
        if state.get("map_pool") != plan.pool["hash"] and not parents_state(cfg, state):
            raise RuntimeError("the map pool changed since the run started; refusing to resume")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        # The stored state has the options it was saved under too: a continued run (catanrl fork with
        # --set) learns at its own config's rate and weight decay, on the stored moments
        for group in optimizer.param_groups:
            group["lr"], group["weight_decay"] = float(train["learning_rate"]), float(train["weight_decay"])
        replay.restore(state["replay"])
        iteration = int(state["epoch"])
        selfplay.decisions, selfplay.games = int(state["decisions"]), int(state["games"])
        set_rng_state(state["rng"])
        selfplay.refresh()
        if monitors is not None:
            monitors.restore(state.get("monitors") or {})
        if dice_probe is not None:
            dice_probe.restore(state)
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
        cache_stats = selfplay.cache.table.stats if selfplay.cache is not None else None
        added = replay.added
        model.eval()
        selfplay.check_kernels()
        if monitors is not None:
            monitors.begin()
        for step in range(int(train["steps_per_iteration"])):
            selfplay.step(iteration, step, rng, replay, stats)
        selfplay.drop_trees()  # reuse: no tree outlives the network that built it
        played = time.time() - start
        health = monitors.before_learn(replay, replay.added - added) if monitors is not None else {}
        learned = {}
        if replay.size >= int(train["min_replay"]):
            if dice_probe is not None and dice_probe.rows is None:
                # the probe's fixed batch: every k-th replay position, the first time the learner trains
                dice_probe.take(replay.obs[: replay.size : max(replay.size // DICE_PROBE_ROWS, 1)][:DICE_PROBE_ROWS])
            learned = learn(model, optimizer, replay, train, device,
                            np.random.default_rng(derive_seed(seed, "replay", iteration)), iteration)
            selfplay.refresh()
        if dice_probe is not None:
            health.update(dice_probe.metrics(model))
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
            # the decisions stored for the replay (they enter it when their game ends): the full searches,
            # with a policy target, and the fast ones record_fast (record_fast_prob) kept, value only
            "selfplay/recorded_full": stats["full"], "selfplay/recorded_fast": stats.get("recorded_fast", 0.0),
            "selfplay/rounds": stats["rounds"] / int(train["steps_per_iteration"]),  # network rounds per step
            "search/mean_depth": stats["depth"] / full, "search/max_depth": stats["max_depth"] / full,
            "environment/episodes": episodes["episodes"], "environment/truncated": episodes["truncated"],
            "environment/length": episodes["length"], "environment/learner_win": episodes["learner_win"],
            # how the games were played: final victory points (games with a winner), development
            # cards, Largest Army and Longest Road changing hands
            **{f"environment/{key}": episodes[key] for key in (*POINTS_KEYS, *STYLE_KEYS)},
            "throughput/selfplay_s": played, "throughput/iteration_s": time.time() - start,
            **{f"train/{k}": v for k, v in learned.items()}, **health, "map_pool": plan.pool["hash"],
        }
        if selfplay.reuse:
            # searches that started from a kept tree, and the share of its visit cap a search started
            # with (0 for a new tree), over the full and over the fast searches
            logs["selfplay/reused"] = stats.get("reused", 0.0) / max(new, 1)
            logs["selfplay/carried_full"] = stats.get("carried_full", 0.0) / full
            logs["selfplay/carried_fast"] = stats.get("carried_fast", 0.0) / max(new - stats["full"], 1.0)
        if selfplay.cache is not None:
            # runtime.eval_cache_mb: the share of the leaves that took their outputs from the cache
            # (or from another leaf's row of their round), and of the rows it stored the share that
            # displaced a row still in use (near 1: the table is too small for an iteration)
            now = selfplay.cache.table.stats
            logs["selfplay/cache_hits"] = stats.get("cached", 0.0) / max(stats["leaves"], 1.0)
            logs["selfplay/cache_replaced"] = (now.replaced - cache_stats.replaced) / max(
                now.stored - cache_stats.stored, 1)
        if selfplay.decided:
            # decided_value: the share of decisions made in decided games, and the simulations a
            # search ran (a re-rooted one on top of the visits it carried)
            logs["selfplay/decided"] = stats.get("decided", 0.0) / max(new, 1)
            logs["selfplay/sims_per_decision"] = stats.get("simulations", 0.0) / max(new, 1)
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
