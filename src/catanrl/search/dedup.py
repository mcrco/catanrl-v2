"""Leaf rows evaluated once: :class:`DedupEvaluator` merges the byte-identical rows of a
batch and keeps recent rows' outputs across batches.

Under the strict contract (:mod:`catanrl.search.evaluator`: a row's outputs never depend
on its batch) ``f(rows)[i] == f(rows)[j]`` whenever ``rows[i] == rows[j]`` bytewise, so
copying the first occurrence's outputs to every repeat gives exactly what the wrapped
evaluator would, and so does reading them back from an earlier batch of the same network
(the eval harness already rests on reruns of ``f`` agreeing). Nothing here enters a
record's key. Repeats come from a determinized search's worlds (the roots are issued
together, and at ``private`` level worlds that differ only in hidden cards encode the same
row for the mover), from one position reached by two action orders in a tree, and from
positions another seat's unseen draw or steal leaves identical for the mover.

Hashing costs about 10 us per 4p row, so the wrapper hashes only when a merge can remove a
forward: on CPU (strict mode runs one forward per row), in ``fast`` mode, or when the
batch is larger than ``infer_rows`` (a chunk can go). A strict CUDA batch below
``infer_rows`` is one padded graph replay whatever its row count, so it passes through
untouched; ``CATANRL_DEDUP_FORCE=1`` hashes it anyway, to count a GPU run's repeats.
"""

from __future__ import annotations

import hashlib
import os
from collections import Counter, OrderedDict

import numpy as np

from catanrl.search.evaluator import BatchEvaluator


class DedupEvaluator:
    """``__call__(rows) -> (logits, values)`` as the wrapped evaluator's, with every distinct
    row forwarded once per batch (first-occurrence order) and ``cache_rows`` recent rows'
    outputs kept (LRU) for later batches. ``counters``: ``rows`` seen, ``forwarded`` to the
    network, ``batch_hits`` (repeats within a batch) and ``cache_hits``."""

    def __init__(self, evaluator: BatchEvaluator, *, cache_rows: int = 65536) -> None:
        if getattr(evaluator, "recurrent", False):
            raise ValueError("a recurrent network's rows carry a memory: there is nothing to merge")
        self.evaluator = evaluator
        self.cache_rows = int(cache_rows)
        self.force = os.environ.get("CATANRL_DEDUP_FORCE") == "1"
        self._cache: OrderedDict[bytes, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.counters: Counter[str] = Counter()

    @property
    def obs_size(self) -> int:
        return self.evaluator.obs_size

    @property
    def num_actions(self) -> int:
        return self.evaluator.num_actions

    @property
    def num_players(self) -> int:
        return self.evaluator.num_players

    @property
    def device(self):
        return self.evaluator.device

    @property
    def mode(self) -> str:
        return self.evaluator.mode

    @property
    def infer_rows(self) -> int:
        return self.evaluator.infer_rows

    def active(self, rows: int) -> bool:
        """Whether merging ``rows`` rows can save a forward."""
        return self.force or self.device.type == "cpu" or self.mode == "fast" or rows > self.infer_rows

    def clear(self) -> None:
        self._cache.clear()

    def __call__(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        observations = np.ascontiguousarray(observations, dtype=np.float32)
        n = len(observations)
        self.counters["rows"] += n
        if n == 0 or not self.active(n):
            self.counters["forwarded"] += n
            return self.evaluator(observations)
        if observations.ndim != 2 or observations.shape[1] != self.obs_size:
            raise ValueError(f"expected rows [n, {self.obs_size}], got {observations.shape}")
        keys = [hashlib.blake2b(row.tobytes(), digest_size=16).digest() for row in observations]
        logits = np.empty((n, self.num_actions), dtype=np.float32)
        values = np.empty((n, self.num_players), dtype=np.float32)
        source: dict[bytes, int] = {}  # key -> the row whose outputs every repeat copies
        forward: list[int] = []
        for i, key in enumerate(keys):
            if key in source:
                self.counters["batch_hits"] += 1
                continue
            source[key] = i
            cached = self._cache.get(key)
            if cached is None:
                forward.append(i)
            else:
                self._cache.move_to_end(key)
                logits[i], values[i] = cached
                self.counters["cache_hits"] += 1
        if forward:
            out_logits, out_values = self.evaluator(observations[forward])
            logits[forward], values[forward] = out_logits, out_values
            for j, i in enumerate(forward):
                self._remember(keys[i], out_logits[j], out_values[j])
        self.counters["forwarded"] += len(forward)
        for i, key in enumerate(keys):
            j = source[key]
            if j != i:
                logits[i], values[i] = logits[j], values[j]
        return logits, values

    def _remember(self, key: bytes, logits: np.ndarray, values: np.ndarray) -> None:
        if self.cache_rows <= 0:
            return
        self._cache[key] = (logits.copy(), values.copy())
        while len(self._cache) > self.cache_rows:
            self._cache.popitem(last=False)
