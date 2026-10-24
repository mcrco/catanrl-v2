"""The select eval: which checkpoint of a run is ``best``; and the monitors, evals
logged at checkpoints that decide nothing.

DESIGN 6.1 orders a checkpoint as: the ``.pt``, then the trainer state (the
commit), then the select eval, which may move ``best``. The trainer state
therefore never knows the newest checkpoint's select result, so the best
metric is not kept there: on resume, :meth:`Selector.restore` reads it from
the run's ``best`` alias (the last decision written), and the writer's
pending selects (the last commit's included) replay their cached records
through :meth:`Selector.catch_up`, redoing a decision a crash cut off.

``eval.monitor`` entries (``{name, suite, lineup, every, metric, fill}``) run
at the same point, before the select eval, at their own cadence: each is an
eval record like the select's (slot 0 is the checkpoint, ``${self}``) and a
metrics row of its own at the checkpoint's step, ``select/<name>/score_rate``,
``wilson_low``, ``wilson_high``, ``games``, ``seat<k>``, ``<metric>`` and
``eval_id`` (the record), which the wandb mirror logs as it comes. A resume
replays the monitors whose rows metrics.jsonl lacks (cached records, so only
the row is new). ``eval.select: {from: NAME}`` makes monitor NAME the select
eval (catanrl.config), so ``best`` follows it and its select is the same record.

A monitor against a fixed opponent (``${champion}``: ``eval.champion`` or the
run's ``init_from`` checkpoint, pinned when the run is created) plays the same
suite schedule (the same boards, every seat order) at every checkpoint, so two
checkpoints' records of it pair game for game: take the eval ids from their rows
(``select/<name>/eval_id`` at the two epochs) and run
``uv run python tools/search_probe.py --pair <id at eA> <id at eB>`` (the delta
and McNemar's p on slot 0), or :func:`catanrl.eval.results.paired` when both
records ran under one code id.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from typing import Any

import torch


def lineup_for(select: Mapping[str, Any], ref: Any) -> list[str]:
    return [str(spec).replace("${self}", str(ref)) for spec in select["lineup"]]


def select_eval(select: Mapping[str, Any], ref: Any, eval_runtime: Mapping[str, Any]) -> tuple[Any, float]:
    """Run ``eval.select`` (or a monitor) on a checkpoint: (record, metric).

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


def monitor_row(monitor: Mapping[str, Any], record: Any, metric: float) -> dict[str, Any]:
    """A monitor's result as metrics keys: ``select/<name>/...`` (slot 0's score rate,
    Wilson interval, games, per-seat rates, the monitor's metric and the record's id)."""
    from catanrl.runs import MONITOR_PREFIX

    s, prefix = record.summary, f"{MONITOR_PREFIX}{monitor['name']}"
    row = {f"{prefix}/score_rate": s["score_rate"], f"{prefix}/wilson_low": s["wilson"][0],
           f"{prefix}/wilson_high": s["wilson"][1], f"{prefix}/games": s["games"]}
    for seat, rate in (s["slots"][0].get("per_seat") or {}).items():
        row[f"{prefix}/seat{seat}"] = rate
    row.setdefault(f"{prefix}/{monitor.get('metric', 'score_rate')}", metric)
    row[f"{prefix}/eval_id"] = record.eval_id
    return row


def _every(entry: Mapping[str, Any], epoch: int) -> bool:
    return int(epoch) % int(entry.get("every", 1)) == 0


class Selector:
    """Runs monitor and select evals at checkpoints and moves the writer's ``best`` alias."""

    def __init__(self, select: Mapping[str, Any] | None, writer: Any, eval_runtime: Mapping[str, Any],
                 monitors: Sequence[Mapping[str, Any]] | None = None) -> None:
        self.select = dict(select) if select else None
        self.monitors = [dict(m) for m in monitors or []]
        self.writer = writer
        self.eval_runtime = dict(eval_runtime)
        self.best: float | None = None

    def due(self, epoch: int) -> bool:
        return (bool(self.select) and _every(self.select, epoch)) or any(_every(m, epoch) for m in self.monitors)

    def run(self, ref: Any, model: torch.nn.Module | None = None) -> None:
        """A new checkpoint's evals: the monitors due at its epoch, then the select eval if due."""
        self._evaluate(ref, model, [m for m in self.monitors if _every(m, ref.epoch)],
                       select=bool(self.select) and _every(self.select, ref.epoch))

    def catch_up(self, model: torch.nn.Module | None = None) -> None:
        """What the committed checkpoints still owe after a launch's start: the monitor rows
        metrics.jsonl lacks, then the pending selects (the last commit's always: its alias
        decision may be lost)."""
        if self.monitors:
            for ref, names in self.writer.pending_monitors(self.monitors):
                self._evaluate(ref, model, [m for m in self.monitors if m["name"] in names], select=False)
        if self.select:
            for ref in self.writer.pending_selects():
                self._evaluate(ref, model, [], select=True)

    def _evaluate(self, ref: Any, model: torch.nn.Module | None, monitors: list[dict], *, select: bool) -> None:
        if not monitors and not select:
            return
        context = _eval_mode(model) if model is not None else contextlib.nullcontext()
        with context:
            for monitor in monitors:  # first: a select ``from`` a monitor then finds its record cached
                record, metric = select_eval(monitor, ref, self.eval_runtime)
                steps = int(ref.agent_steps)
                self.writer.log({"epoch": int(ref.epoch), "agent_steps": steps,
                                 **monitor_row(monitor, record, metric)}, steps)
            if select:
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
