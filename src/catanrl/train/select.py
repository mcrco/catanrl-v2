"""The select eval: which checkpoint of a run is ``best``.

DESIGN 6.1 orders a checkpoint as: the ``.pt``, then the trainer state (the
commit), then the select eval, which may move ``best``. The trainer state
therefore never knows the newest checkpoint's select result, so the best
metric is not kept there: on resume, :meth:`Selector.restore` rebuilds it
from the select records of every committed checkpoint (cached records, read
with ``catanrl.eval.results.find``; nothing is replayed).
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from typing import Any

import torch


def lineup_for(select: Mapping[str, Any], ref: Any) -> list[str]:
    return [str(spec).replace("${self}", str(ref)) for spec in select["lineup"]]


def select_eval(select: Mapping[str, Any], ref: Any, eval_runtime: Mapping[str, Any]) -> tuple[Any, float]:
    """Run ``eval.select`` on a checkpoint: (record, metric).

    Isolated from training: autocast off (PuffeRL leaks bf16 autocast), a
    forked torch RNG and strict determinism.
    """
    from catanrl.eval.results import evaluate
    from catanrl.seeding import determinism

    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.autocast("cuda", enabled=False), torch.random.fork_rng(devices=devices), determinism("strict"):
        record = evaluate(lineup_for(select, ref), select["suite"], fill=select.get("fill"), **eval_runtime)
    return record, metric_of(select, record)


def recorded_metric(select: Mapping[str, Any], ref: Any) -> float | None:
    """The metric of a checkpoint's existing select record, or None."""
    from catanrl.eval.results import find

    record = find(lineup_for(select, ref), select["suite"], fill=select.get("fill"))
    return None if record is None else metric_of(select, record)


def metric_of(select: Mapping[str, Any], record: Any) -> float:
    return float(record.summary[select.get("metric", "score_rate")])


class Selector:
    """Runs select evals at checkpoints and moves the writer's ``best`` alias."""

    def __init__(self, select: Mapping[str, Any] | None, writer: Any, eval_runtime: Mapping[str, Any]) -> None:
        self.select = dict(select) if select else None
        self.writer = writer
        self.eval_runtime = dict(eval_runtime)
        self.best: float | None = None

    def due(self, epoch: int) -> bool:
        return bool(self.select) and int(epoch) % int(self.select.get("every", 1)) == 0

    def run(self, ref: Any, model: torch.nn.Module | None = None) -> None:
        if not self.select:
            return
        context = _eval_mode(model) if model is not None else contextlib.nullcontext()
        with context:
            record, metric = select_eval(self.select, ref, self.eval_runtime)
        if self.best is None or metric > self.best:
            self.best = metric
            self.writer.set_alias("best", ref, record=record)

    def restore(self) -> None:
        """The best metric over the committed checkpoints' select records."""
        self.best = None
        if not self.select:
            return
        for ref in self.writer.checkpoints():
            if self.due(ref.epoch):
                metric = recorded_metric(self.select, ref)
                if metric is not None and (self.best is None or metric > self.best):
                    self.best = metric


@contextlib.contextmanager
def _eval_mode(model: torch.nn.Module):
    training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(training)
