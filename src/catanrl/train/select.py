"""The select eval: which checkpoint of a run is ``best``.

DESIGN 6.1 orders a checkpoint as: the ``.pt``, then the trainer state (the
commit), then the select eval, which may move ``best``. The trainer state
therefore never knows the newest checkpoint's select result, so the best
metric is not kept there: on resume, :meth:`Selector.restore` reads it from
the run's ``best`` alias (the last decision written), and the writer's
pending selects (the last commit's included) replay their cached records
through :meth:`Selector.run`, redoing a decision a crash cut off.
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
        record = evaluate(lineup_for(select, ref), select["suite"], fill=select.get("fill"), restart_partial=True,
                          **eval_runtime)
    return record, metric_of(select, record)


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
        """The metric of the run's current ``best`` (not of a select record whose alias
        was never written, so rerunning that select moves ``best`` to it)."""
        self.best = None
        row = self.writer.aliases().get("best") if self.select else None
        if row is not None and row.get("metric") is not None:
            self.best = float(row["metric"])


@contextlib.contextmanager
def _eval_mode(model: torch.nn.Module):
    training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(training)
