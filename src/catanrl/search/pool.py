"""Driving ``cppanatron.SearchPool`` with a :class:`BatchEvaluator`.

``SearchPool`` runs many searches in lockstep (one pending leaf per search by
default, v1's fastest design): ``collect`` advances every search to its next
leaf and writes the leaves' rows, the evaluator runs them on the GPU in one
batch, and ``apply`` expands and backs up. :func:`run_searches` loops that to
completion; :func:`search_config` maps agent parameters onto the engine's
``SearchConfig``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import cppanatron as cpp
import numpy as np

from catanrl.envs.encoding import critic_level_of, encoder_dims, info_level, make_encoder
from catanrl.search.evaluator import BatchEvaluator

#: Dirichlet noise at PUCT roots when an agent asks for noise.
DIRICHLET_ALPHA = 0.3
DIRICHLET_FRACTION = 0.25


def search_config(params: Any, *, max_players: int, threads: int | None = None) -> cpp.SearchConfig:
    """The engine ``SearchConfig`` of MCTS agent params (catanrl.agents.spec.MctsParams).

    ``dets = K`` samples K root worlds from the hand tracker; ``oracle =
    true`` searches the true game (``determinizations = 0``). ``c_puct`` is
    in engine units (0.75 = v1's 1.5).
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
    return cpp.SearchConfig(
        c_puct=float(get("c_puct", 0.75)),
        simulations=int(get("sims", 256)),
        root=roots[root],
        dirichlet_alpha=DIRICHLET_ALPHA if noise and root == "puct" else 0.0,
        dirichlet_fraction=DIRICHLET_FRACTION,
        gumbel_considered=int(get("gumbel_m", 16)),
        gumbel_scale=1.0,
        determinizations=0 if oracle else dets,
        max_players=int(max_players),
        threads=threads,
    )


def make_pool(dims: Mapping[str, int], level: str, config: cpp.SearchConfig,
              critic_level: str | None = None) -> cpp.SearchPool:
    """A pool whose leaf rows are the network's training rows.

    With a critic at another level the leaves carry the critic block too
    (``ActorCriticEncoder``, engine E8); with determinizations a leaf is a
    sampled world, whose sampled hidden cards a full-information critic sees.
    """
    if critic_level_of(level, critic_level) is not None:
        encoder = cpp.ActorCriticEncoder(encoder_dims(dims), info_level(level), info_level(critic_level))
    else:
        encoder = make_encoder(dims, level)
    return cpp.SearchPool(encoder.action_space, encoder, config)


def run_searches(pool: cpp.SearchPool, evaluator: BatchEvaluator, *, batch_rows: int) -> int:
    """Collect, evaluate and apply until every search in the pool is done; returns leaf rows."""
    return run_pools([pool], evaluator, batch_rows=[batch_rows])


def run_pools(pools: Sequence[cpp.SearchPool], evaluator: BatchEvaluator, *, batch_rows: Sequence[int]) -> int:
    """Run several pools in lockstep, their leaves evaluated in one batch per round
    (e.g. full- and fast-budget searches of playout cap randomization); returns leaf rows.

    A row's outputs do not depend on its batch (strict evaluator), so the
    searches are the same as run pool by pool.
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
