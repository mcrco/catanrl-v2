"""Driving ``cppanatron.SearchPool`` with a :class:`BatchEvaluator`.

``SearchPool`` runs many searches in lockstep (one pending leaf per search by
default, v1's fastest design): ``collect`` advances every search to its next
leaf and writes the leaves' rows, the evaluator runs them on the GPU in one
batch, and ``apply`` expands and backs up. :func:`run_searches` loops that to
completion; :func:`search_config` maps agent parameters onto the engine's
``SearchConfig``.

:class:`NetworkCache` is the engine's evaluation cache (``cppanatron.EvalCache``)
for the pools of one network: a leaf whose row was evaluated before, or is
another leaf's of the same round, takes that row's outputs instead of a row of
its own in the batch. The searches are the ones without it wherever a row's
outputs do not depend on its batch (a strict evaluator: to the bit). Self-play's
searches have one (``runtime.eval_cache_mb``) and so has every MCTS agent, of
:func:`eval_cache_mb` megabytes.
"""

from __future__ import annotations

import os
import time
import weakref
from collections.abc import Mapping, Sequence
from typing import Any

import cppanatron as cpp
import numpy as np

from catanrl.envs.encoding import critic_level_of, encoder_dims, info_level, make_encoder
from catanrl.search.evaluator import BatchEvaluator

#: Dirichlet noise at PUCT roots when an agent asks for noise.
DIRICHLET_ALPHA = 0.3
DIRICHLET_FRACTION = 0.25

#: Environment variable setting the engine's ``SearchConfig.free_backlog`` for every pool this
#: process builds: the most tree nodes that ``remove``, ``advance`` and ``reroot`` may leave for
#: the pool's threads to free (0: each call frees its own, as before the engine had the option).
#: Where the memory is freed, like ``threads``: no result and no key depends on it
#: (catanrl.agents.spec.search_details), so it is the environment's, not an agent's or a run's.
FREE_BACKLOG_ENV = "CATANRL_FREE_BACKLOG"

#: Environment variable setting the megabytes of an MCTS agent's evaluation cache (the engine's,
#: :class:`NetworkCache`: one table per agent and network): :data:`EVAL_CACHE_MB` when unset, and
#: 0 for the Python path the agents had before it (``DedupEvaluator``, catanrl.search.dedup).
#: Which rows reach the network, never what a row's outputs are (strict evaluator: the games are
#: the same to the bit at any size), so like ``threads`` it is the environment's and in no key.
EVAL_CACHE_ENV = "CATANRL_EVAL_CACHE_MB"
#: The megabytes of an MCTS agent's evaluation cache when :data:`EVAL_CACHE_ENV` is unset.
EVAL_CACHE_MB = 256.0


def free_backlog() -> dict[str, int]:
    """``SearchConfig``'s ``free_backlog`` argument from :data:`FREE_BACKLOG_ENV`; none when it
    is unset (the engine's default; an engine build without the option takes no argument)."""
    text = os.environ.get(FREE_BACKLOG_ENV, "").strip()
    if not text:
        return {}
    try:
        nodes = int(text)
    except ValueError:
        nodes = -1
    if nodes < 0:
        raise ValueError(f"{FREE_BACKLOG_ENV} must be a number of tree nodes (0 or more), not {text!r}")
    return {"free_backlog": nodes}


def eval_cache_mb() -> float:
    """The megabytes of an MCTS agent's evaluation cache: :data:`EVAL_CACHE_ENV`'s, or
    :data:`EVAL_CACHE_MB` when it is unset; 0 is no engine cache (the Python dedup instead)."""
    text = os.environ.get(EVAL_CACHE_ENV, "").strip()
    if not text:
        return EVAL_CACHE_MB
    try:
        megabytes = float(text)
    except ValueError:
        megabytes = -1.0
    if not 0 <= megabytes < float("inf"):
        raise ValueError(f"{EVAL_CACHE_ENV} must be a number of megabytes (0: the Python dedup, no engine "
                         f"cache), not {text!r}")
    return megabytes


def search_config(params: Any, *, max_players: int, threads: int | None = None) -> cpp.SearchConfig:
    """The engine ``SearchConfig`` of MCTS agent params (catanrl.agents.spec.MctsParams).

    ``dets = K`` samples K root worlds from the hand tracker, or with
    ``world_choice = likely`` takes the K most likely deals of the hidden cards
    and searches each by its probability among them; ``oracle =
    true`` searches the true game (``determinizations = 0``). ``c_puct`` is
    in engine units (0.75 = v1's 1.5). ``transpositions`` turns on the engine's
    transposition table (one node for a position reached by reordered actions).
    ``root_policy_temperature`` (unset: 1.0) and ``forced_playouts`` (unset: 0.0)
    are the engine's root exploration settings of the same names (PUCT roots;
    self-play's full searches, catanrl.train.alphazero): no MCTS agent has them.
    ``free_backlog`` is not a parameter: :data:`FREE_BACKLOG_ENV` sets it.
    """
    get = params.get if isinstance(params, Mapping) else lambda key, default=None: getattr(params, key, default)
    oracle = bool(get("oracle", False))
    dets = int(get("dets", 1))
    if not oracle and dets < 1:
        raise ValueError("dets must be at least 1 (or oracle=true for the true game)")
    if oracle and dets > 1:
        raise ValueError("oracle=true searches the true game; it takes no dets")
    root = str(get("root", "puct"))
    roots = {"puct": cpp.RootSelection.PUCT, "gumbel": cpp.RootSelection.GUMBEL}
    if root not in roots:
        raise ValueError(f"root must be puct or gumbel, not {root!r}")
    noise = bool(get("noise", False))
    threads = int(get("threads", 0) if threads is None else threads)
    world_choice = str(get("world_choice", "sample") or "sample")
    if world_choice not in ("sample", "likely"):
        raise ValueError(f"world_choice must be sample or likely, not {world_choice!r}")
    if oracle and world_choice != "sample":
        raise ValueError("oracle=true searches the true game; it takes no world_choice")
    # Passed only when set, as free_backlog: the engine's default is the sampled worlds.
    worlds = {"world_choice": cpp.WorldChoice.LIKELY} if world_choice == "likely" else {}
    return cpp.SearchConfig(
        c_puct=float(get("c_puct", 0.75)),
        simulations=int(get("sims", 256)),
        max_pending=int(get("pending", 1)),
        root=roots[root],
        dirichlet_alpha=DIRICHLET_ALPHA if noise and root == "puct" else 0.0,
        dirichlet_fraction=DIRICHLET_FRACTION,
        gumbel_considered=int(get("gumbel_m", 16)),
        gumbel_scale=1.0,
        determinizations=0 if oracle else dets,
        max_players=int(max_players),
        threads=threads,
        transpositions=bool(get("transpositions", False)),
        root_policy_temperature=float(get("root_policy_temperature", 1.0)),
        forced_playouts=float(get("forced_playouts", 0.0)),
        **worlds,
        **free_backlog(),
    )


class NetworkCache:
    """A ``cppanatron.EvalCache`` of ``megabytes`` that answers for one network's weights only.

    ``table`` goes to every pool of the network (``make_pool(cache=...)``): the pools then
    write, and count, a row only for a leaf whose outputs are not known (``SearchPool.cache``),
    and a round that needs no row never reaches the caller, so the loops here and in
    pipeline.py run as they do without a cache.

    The table's outputs are the evaluator's for the weights it had when they were stored, so
    :meth:`sync` belongs before every run of searches (between runs: never between a collect
    and its apply, which the engine refuses). It clears the table unless the evaluator is the
    object of the last sync (the fused kernels read copies of the weights made when their
    evaluator was built) and no parameter or buffer of its network was written since: the
    tensors are the same objects on the same memory with the same version counters, which
    torch advances on every in-place write (an optimizer step, ``load_state_dict``). A caller
    therefore cannot forget to clear after a change it made by any of those means; a tensor
    whose counter cannot be read (made in inference mode) clears on every sync, and so does an
    evaluator that shows no network (no ``model``: a wrapper around one). The counters
    do not see a write through ``tensor.data`` or through a NumPy view of a tensor's memory
    (nothing in this package writes weights that way): ``table.clear()`` after one.

    The table is a fixed number of entries (``megabytes`` over the bytes of one entry: the
    logits and values of a row and 32 bytes), taken from the system as they are written; a new
    row replaces the entry of its set of four that was used longest ago.
    """

    def __init__(self, num_actions: int, num_players: int, megabytes: float) -> None:
        if not megabytes > 0:
            raise ValueError("the evaluation cache's size (megabytes) must be above 0")
        entries = int(float(megabytes) * 2**20) // cpp.EvalCache.entry_bytes(int(num_actions), int(num_players))
        self.table = cpp.EvalCache(int(num_actions), int(num_players), entries=max(entries, 4))
        self._evaluator: weakref.ref | None = None
        self._weights: tuple | None = None

    @staticmethod
    def weights(model: Any) -> tuple | None:
        """What changes whenever one of the network's tensors is written or replaced; None if
        that cannot be told."""
        try:
            return tuple((id(t), t.data_ptr(), t._version) for t in (*model.parameters(), *model.buffers()))
        except RuntimeError:  # an inference tensor has no version counter
            return None

    def sync(self, evaluator: Any) -> bool:
        """Clear the table unless it holds this evaluator's outputs for its network's weights
        as they are now; whether it was cleared."""
        model = getattr(evaluator, "model", None)
        weights = self.weights(model) if model is not None else None
        same = (self._evaluator is not None and self._evaluator() is evaluator
                and weights is not None and weights == self._weights)
        if not same:
            self.table.clear()
            self._evaluator, self._weights = weakref.ref(evaluator), weights
        return not same

    def answered(self) -> int:
        """Leaves that took their outputs from the table or from another leaf's row so far."""
        stats = self.table.stats
        return int(stats.hits + stats.repeats)


def make_pool(dims: Mapping[str, int], level: str, config: cpp.SearchConfig,
              critic_level: str | None = None, dice_state: bool = False,
              cache: cpp.EvalCache | None = None) -> cpp.SearchPool:
    """A pool whose leaf rows are the network's training rows.

    With a critic at another level the leaves carry the critic block too
    (``ActorCriticEncoder``, engine E8); with determinizations a leaf is a
    sampled world, whose sampled hidden cards a full-information critic sees.
    ``dice_state`` (a network with the dice input) gives every leaf row its
    position's dice state: a leaf is a game (or a world, a copy of it) with
    its own dice state, as the search's chance nodes leave it.
    ``cache`` (a :class:`NetworkCache`'s ``table``) is the evaluation cache the
    pool shares with the other pools of its network.
    """
    if critic_level_of(level, critic_level) is not None:
        encoder = cpp.ActorCriticEncoder(encoder_dims(dims), info_level(level), info_level(critic_level),
                                         dice_state=bool(dice_state))
    else:
        encoder = make_encoder(dims, level, dice_state)
    pool = cpp.SearchPool(encoder.action_space, encoder, config)
    if cache is not None:
        pool.cache = cache
    return pool


def run_searches(pool: cpp.SearchPool, evaluator: BatchEvaluator, *, batch_rows: int,
                 stop: float | None = None) -> int:
    """Collect, evaluate and apply until every search in the pool is done, or ``stop``
    passes (see :func:`run_pools`); returns leaf rows."""
    return run_pools([pool], evaluator, batch_rows=[batch_rows], stop=stop)


def run_pools(pools: Sequence[cpp.SearchPool], evaluator: BatchEvaluator, *, batch_rows: Sequence[int],
              stop: float | None = None) -> int:
    """Run several pools in lockstep, their leaves evaluated in one batch per round
    (e.g. full- and fast-budget searches of playout cap randomization); returns leaf rows.

    A row's outputs do not depend on its batch (strict evaluator), so the
    searches are the same as run pool by pool.

    ``stop`` (a ``time.monotonic()`` instant) ends the run between rounds once it has
    passed, after at least one round, with nothing pending: the searches are cut short
    but whole (``select`` plays the most visits so far, ``advance`` re-roots them).
    """
    for pool in pools:
        if pool.observation_size != evaluator.obs_size:
            raise ValueError("the pool's rows and the network's differ (level or dims)")
        if pool.action_size != evaluator.num_actions or pool.max_players != evaluator.num_players:
            raise ValueError("the pool's action space or value width differs from the network's")
    buffers = [(np.empty((max(1, rows), pool.observation_size), dtype=np.float32), np.empty(max(1, rows), dtype=np.int64))
               for pool, rows in zip(pools, batch_rows)]
    total = 0
    while True:
        if stop is not None and total and time.monotonic() >= stop:
            return total
        counts = []
        try:
            for pool, (observations, ids) in zip(pools, buffers):
                counts.append(pool.collect(observations, ids))
            n = sum(counts)
            if not n:
                return total
            rows = [observations[:k] for (observations, _), k in zip(buffers, counts)]
            logits, values = evaluator(rows[0] if len(rows) == 1 else np.concatenate(rows))
            start = 0
            for pool, k in zip(pools, counts):
                if k:
                    pool.apply(logits[start : start + k], values[start : start + k])
                start += k
        except BaseException:
            # Withdraw collected leaves, so the pools stay usable and the error is the real one.
            for pool in pools:
                if pool.pending:
                    pool.cancel()
            raise
        total += n
