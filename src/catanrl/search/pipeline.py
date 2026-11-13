"""A pipelined round loop: the GPU evaluates one group's leaves while the CPU collects
and applies the others'.

:func:`~catanrl.search.pool.run_pools` is strictly serial: every pool collects, the
evaluator runs the round's rows and returns, every pool applies, so the GPU idles while
the engine's threads descend and back up and they idle while it computes. Here the
searches are split into groups (:class:`Group`: pools that collect, evaluate and apply
together, as ``run_pools`` runs its pools), and :func:`run_groups` keeps every group but
one queued on the device::

    collect A, submit A, collect B, submit B
    until done: wait for A, apply A, collect A, submit A      (the GPU runs B meanwhile)
                wait for B, apply B, collect B, submit B      (and A meanwhile)

so a round of every group costs about max(GPU, CPU) rather than their sum. The loop is
one thread: only one group's pools run at a time (each on its own worker threads), and
the device runs the submitted groups in order.

:class:`Pipeline` is the evaluator's side. It wraps a
:class:`~catanrl.search.evaluator.BatchEvaluator` and runs the same forwards (its
buckets, CUDA graphs and kernels, whatever it was built with) without waiting for them:
a group's rows are collected straight into page-locked memory (:class:`Batch`), its
chunks are queued one after another (non-blocking uploads into the graphs' static inputs,
replays and non-blocking downloads into page-locked outputs, all on one stream, which
runs them in order), and only :meth:`Pipeline.result` waits, on an event recorded behind
the group's last download.

Every search is independent of the others (its own seed and tree), so the searches are
those of ``run_pools`` wherever a row's outputs do not depend on its batch: with a strict
evaluator, bit for bit, for any number of groups. A fast evaluator's bucket sizes follow
the chunk sizes, which follow the groups; on CUDA a row's outputs are the same bits
within one bucket size (whatever its position and neighbours) and differ in the last
bits between bucket sizes, as they already do from round to round in ``run_pools``.

Without a CUDA device nothing is asynchronous: :meth:`Pipeline.submit` computes at once
(the evaluator's own forwards, row by row in strict mode) and the loop is ``run_pools``
group by group.

Two things of ``BatchEvaluator._run`` that a queue has to keep. The fused kernels' flag
of every chunk is read when its outputs are back (:meth:`Pipeline.flagged`, a chunk at a
time, with the chunk's rows still in the batch: the place to evaluate one again). And
the CUDA graphs are the evaluator's own, captured by it (``BatchEvaluator._graph``) on
this thread with nothing queued on the device: :meth:`Pipeline.submit` waits for what is
queued before a bucket's first use (torch 2.14 lets the graphs of a stream share a cuBLAS
workspace, and a capture may free or move it under a running replay).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, field

import cppanatron as cpp
import numpy as np
import torch

from catanrl.search.evaluator import BatchEvaluator


class Batch:
    """A group's rows and outputs in host memory (page-locked for a CUDA device): the pools
    collect into ``observations`` and ``ids``, and :meth:`Pipeline.result` returns views of
    the outputs' first rows."""

    def __init__(self, evaluator: BatchEvaluator, capacity: int) -> None:
        pin = evaluator.device.type == "cuda"
        self.capacity = int(capacity)
        self._shape = self._shape_for(evaluator)
        self._rows = torch.empty((self.capacity, evaluator.obs_size), dtype=torch.float32, pin_memory=pin)
        self._logits = torch.empty((self.capacity, evaluator.num_actions), dtype=torch.float32, pin_memory=pin)
        self._values = torch.empty((self.capacity, evaluator.num_players), dtype=torch.float32, pin_memory=pin)
        #: The kernels' overflow flag of each forward (FusedGraph), read by result().
        self._flags = torch.zeros(self.capacity, dtype=torch.int32, pin_memory=pin)
        self._flagged = 0
        self.observations: np.ndarray = self._rows.numpy()
        self.ids = np.empty(self.capacity, dtype=np.int64)
        self._step = 1  # the rows of a chunk (a forward) of the last submit
        self._event: torch.cuda.Event | None = None
        self._called: tuple[np.ndarray, np.ndarray] | None = None  # the outputs of an evaluator that is called
        self.rows = 0  # submitted and not yet returned by result()

    @staticmethod
    def _shape_for(evaluator: BatchEvaluator) -> tuple:
        return (evaluator.obs_size, evaluator.num_actions, evaluator.num_players, evaluator.infer_rows,
                evaluator.device.type == "cuda")

    def fits(self, evaluator: BatchEvaluator) -> bool:
        """Whether the buffers are ``evaluator``'s shapes (and page-locked if it needs them so)."""
        return self._shape == self._shape_for(evaluator)


class Pipeline:
    """Deferred forwards of ``evaluator`` (stateless, both heads or one): ``submit`` queues a
    batch's first rows and returns, ``result`` waits for them. Several batches may be queued;
    their results come in the order submitted. The evaluator stays usable on its own between
    a ``result`` and the next ``submit`` (its graphs' static rows are shared).

    ``batches`` keeps the page-locked buffers by index: pass the same dict to the pipelines of
    successive evaluators of one network shape (self-play rebuilds its evaluator after every
    learning step) and they are allocated once."""

    def __init__(self, evaluator: BatchEvaluator, batches: dict[int, Batch] | None = None) -> None:
        if evaluator.recurrent:
            raise ValueError("the pipeline evaluates single positions (a recurrent network has a memory per row)")
        self.evaluator = evaluator
        self.batches: dict[int, Batch] = {} if batches is None else batches
        #: Whether the forwards are BatchEvaluator's own, queued here. A subclass that evaluates
        #: its own way (its __call__, _evaluate or _run) or a wrapper is called instead, a group
        #: at a time.
        self.queued = all(getattr(type(evaluator), name, None) is getattr(BatchEvaluator, name)
                          for name in ("__call__", "_evaluate", "_run"))

    def batch(self, index: int, rows: int) -> Batch:
        """Batch ``index``, kept between calls, with room for ``rows`` rows."""
        batch = self.batches.get(index)
        if batch is not None and batch.rows:
            raise RuntimeError("the batch is queued: take its result first")
        if batch is None or batch.capacity < rows or not batch.fits(self.evaluator):
            step = self.evaluator.infer_rows  # whole forwards, so a few more rows do not allocate again
            batch = self.batches[index] = Batch(self.evaluator, max(1, -(-rows // step)) * step)
        return batch

    def submit(self, batch: Batch, rows: int) -> None:
        """Queue the forwards of ``batch.observations[:rows]``. The rows must stay as they are
        until :meth:`result` returns."""
        ev = self.evaluator
        if not 0 < rows <= batch.capacity or batch.rows:
            raise ValueError(f"{rows} rows for a batch of {batch.capacity} with {batch.rows} queued")
        if torch.is_autocast_enabled() or (ev.device.type == "cuda" and torch.is_autocast_enabled("cuda")):
            raise RuntimeError("evaluate with autocast off (PuffeRL leaks bf16 autocast)")
        step = 1 if ev.mode == "strict" and ev.device.type == "cpu" else ev.infer_rows  # BatchEvaluator._evaluate's
        batch.rows, batch._flagged, batch._step = rows, 0, step
        if not self.queued:
            batch._called = ev(batch.observations[:rows])
            return
        with torch.inference_mode():
            for chunk, start in enumerate(range(0, rows, step)):
                k = min(step, rows - start)
                size = ev.bucket(k)
                source = batch._rows[start : start + k]
                ev.rows += k
                ev.forwards += 1
                if ev.use_graphs:
                    if size not in ev._graphs:
                        self._settle()  # a capture, with nothing running on the device
                    graph, static, outputs = ev._graph(size)
                    static[0][:k].copy_(source, non_blocking=True)
                    if size > k:
                        static[0][k:].copy_(static[0][:1].expand(size - k, -1))
                    graph.replay()
                else:
                    tensor = source.to(ev.device, non_blocking=True)
                    if size > k:
                        tensor = torch.cat((tensor, tensor[:1].expand(size - k, -1)), dim=0)
                    outputs = ev._forward(tensor)
                # Before the next replay overwrites the static outputs (one stream: in order).
                batch._logits[start : start + k].copy_(outputs[0][:k], non_blocking=True)
                batch._values[start : start + k].copy_(outputs[1][:k], non_blocking=True)
                if len(outputs) > 2:  # the fused kernels' overflow flag
                    batch._flags[chunk : chunk + 1].copy_(outputs[2].reshape(-1)[:1], non_blocking=True)
                    batch._flagged = chunk + 1
            if ev.device.type == "cuda":
                batch._event = torch.cuda.Event()
                batch._event.record(torch.cuda.current_stream(ev.device))

    def result(self, batch: Batch) -> tuple[np.ndarray, np.ndarray]:
        """(logits [rows, A], values [rows, P]) of the rows submitted: views of the batch's
        outputs, valid until its next ``submit``. Waits for the device to reach them."""
        rows, batch.rows = batch.rows, 0
        if not rows:
            raise ValueError("nothing was submitted for this batch")
        if batch._called is not None:
            called, batch._called = batch._called, None
            return called
        if batch._event is not None:
            batch._event.synchronize()
            batch._event = None
        flags = batch._flags.numpy()[: batch._flagged]
        for chunk in np.flatnonzero(flags):
            start = int(chunk) * batch._step
            self.flagged(batch, start, min(batch._step, rows - start), int(flags[chunk]))
        return batch._logits.numpy()[:rows], batch._values.numpy()[:rows]

    def flagged(self, batch: Batch, start: int, rows: int, flag: int) -> None:
        """THE PER-CHUNK HOOK. A chunk (``batch.observations[start:start + rows]``, one forward)
        whose forward raised the kernels' flag, called by :meth:`result` once the outputs are
        back and before they are returned: the chunk's rows are still in the batch, so it can be
        evaluated again here, synchronously (write ``batch._logits`` and ``batch._values``
        ``[start:start + rows]``), or refused. What ``BatchEvaluator._run`` does with the flag:

        - bit 2 (``kernels=fused16``: a half overflowed in the forward) with an evaluator that
          has ``_exact``: the chunk again on the float32 kernels, counted in the evaluator's
          ``exact_forwards``; the flag is then the one that forward returns;
        - any flag left (bit 1: a node with more neighbours than the kernels take) raises."""
        ev = self.evaluator
        exact = getattr(ev, "_exact", None)
        if flag & 2 and exact is not None:
            ev.exact_forwards += 1
            results = exact(batch.observations[start : start + rows], ev.bucket(rows))
            batch._logits.numpy()[start : start + rows] = results[0]
            batch._values.numpy()[start : start + rows] = results[1]
            flag = int(np.any(results[-1]))
        if flag:
            raise RuntimeError("kernels=fused: a node with more than three neighbours, edges or tiles "
                               "(hex boards only); play this board with kernels=torch")

    def _settle(self) -> None:
        """Wait for everything queued on the device (the batches keep their results)."""
        if self.evaluator.device.type == "cuda":
            torch.cuda.synchronize(self.evaluator.device)

    def drain(self) -> None:
        """Wait for everything queued and forget it (after an error: nothing still writes
        into the batches)."""
        try:
            if self.evaluator.device.type == "cuda":
                torch.cuda.synchronize(self.evaluator.device)
        finally:
            for batch in self.batches.values():
                batch.rows, batch._event, batch._called = 0, None, None


@dataclass
class Group:
    """Pools whose leaves share a batch per round, with each pool's rows per round (as
    ``run_pools`` takes them). ``rounds`` and ``leaves`` count the last run's."""

    pools: Sequence[cpp.SearchPool]
    batch_rows: Sequence[int]
    rounds: int = 0
    leaves: int = 0
    _counts: list[int] = field(default_factory=list, repr=False)

    def collect(self, batch: Batch) -> int:
        """Every pool's next leaves, one after another in the batch; their number."""
        self._counts, n = [], 0
        for pool, rows in zip(self.pools, self.batch_rows):
            room = max(1, rows)
            k = pool.collect(batch.observations[n : n + room], batch.ids[n : n + room])
            self._counts.append(k)
            n += k
        return n

    def apply(self, logits: np.ndarray, values: np.ndarray) -> None:
        start = 0
        for pool, k in zip(self.pools, self._counts):
            if k:
                pool.apply(logits[start : start + k], values[start : start + k])
            start += k
        self.rounds += 1
        self.leaves += start


def run_groups(groups: Sequence[Group], pipeline: Pipeline) -> tuple[int, int]:
    """Run every group's searches to the end, the groups taking turns on the device (see the
    module); returns (leaf rows, rounds: the most any group took, what ``run_pools`` would
    take for all the pools at once).

    One group is ``run_pools`` with the pipeline's buffers. On an error the collected leaves
    are withdrawn, so the pools stay usable (as ``run_pools`` leaves them).
    """
    ev = pipeline.evaluator
    for group in groups:
        for pool in group.pools:
            if pool.observation_size != ev.obs_size:
                raise ValueError("the pool's rows and the network's differ (level or dims)")
            if pool.action_size != ev.num_actions or pool.max_players != ev.num_players:
                raise ValueError("the pool's action space or value width differs from the network's")
        group.rounds = group.leaves = 0
    batches = [pipeline.batch(i, sum(max(1, rows) for rows in group.batch_rows)) for i, group in enumerate(groups)]
    queue: deque[tuple[Group, Batch]] = deque()
    try:
        for group, batch in zip(groups, batches):
            n = group.collect(batch)
            if n:
                pipeline.submit(batch, n)
                queue.append((group, batch))
        while queue:
            group, batch = queue.popleft()
            group.apply(*pipeline.result(batch))
            n = group.collect(batch)
            if n:
                pipeline.submit(batch, n)
                queue.append((group, batch))
    except BaseException:
        # Withdraw collected leaves, so the pools stay usable and the error is the real one.
        with suppress(Exception):  # a device that failed fails again here: the first error is the one to raise
            pipeline.drain()
        for group in groups:
            for pool in group.pools:
                if pool.pending:
                    pool.cancel()
        raise
    return sum(group.leaves for group in groups), max((group.rounds for group in groups), default=0)
