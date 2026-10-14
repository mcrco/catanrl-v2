"""Eval records: cached by exactly what decides the outcome (DESIGN 6.6).

::

    key = {"schema", "harness", "determinism", "code": code_id("eval"),
           "suite": suite.key(...),            # map hashes and schedule hash included
           "lineup": [agent.key_dict(), ...], "fill": ...}
    eval_id = digest(key)  # runs/<slot-0 run or _evals>/evals/<suite>/<eval_id>.json

``evaluate(lineup, suite)`` returns the record if it exists. Otherwise it
locks the id, plays only the games missing from ``<eval_id>.games.jsonl``
(one appended row per finished game, so a crash resumes exactly) and writes
the record once. Each start appends its execution (device class, torch,
``infer_rows``); resuming under another one needs ``allow_mixed``, which
marks the record. A forfeit makes the record ``valid: false``.
"""

from __future__ import annotations

import fcntl
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from catanrl import canon
from catanrl.eval.stats import _paired_comparison, binomial_two_sided, wilson_interval
from catanrl.eval.suites import Suite, load_suite

SCHEMA = 1


class ExecutionMismatch(RuntimeError):
    pass


def play_schedule(*args, **kwargs):  # indirection: tests replace the harness
    from catanrl.eval.harness import play_schedule as play

    return play(*args, **kwargs)


@dataclass(frozen=True)
class EvalRecord:
    eval_id: str
    key: dict
    summary: dict
    path: Path
    data: dict = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def load(cls, path: str | Path) -> EvalRecord:
        path = Path(path)
        data = canon.read_json(path)
        return cls(eval_id=data["eval_id"], key=data["key"], summary=data["summary"], path=path, data=data)

    @property
    def games_path(self) -> Path:
        return self.path.with_suffix(".games.jsonl")

    def games(self) -> Iterator[dict]:
        yield from _latest_rows(self.games_path).values()

    @property
    def valid(self) -> bool:
        return bool(self.data.get("valid", True))


# ---------------------------------------------------------------- keys and places


def _resolve(lineup: Sequence[Any], suite: Suite, fill: Any, harness: str):
    from catanrl.agents import ResolvedAgent, resolve_agent

    games = suite.games()
    agents = [a if isinstance(a, ResolvedAgent) else resolve_agent(a, harness=harness, games=games) for a in lineup]
    fill_agent = None
    if fill is not None:
        fill_agent = fill if isinstance(fill, ResolvedAgent) else resolve_agent(fill, harness=harness, games=games)
    return agents, fill_agent


def eval_key(lineup: Sequence[Any], suite: Suite, *, harness: str, fill: Any = None) -> dict:
    from catanrl.provenance import code_id

    return {"schema": SCHEMA, "harness": harness, "determinism": suite.determinism,
            "code": code_id("eval", harness),
            "suite": suite.key(len(lineup), fill=fill is not None),
            "lineup": [a.key_dict() for a in lineup], "fill": fill.key_dict() if fill is not None else None}


def _record_path(lineup: Sequence[Any], suite: Suite, eval_id: str) -> Path:
    from catanrl.runs import runs_root

    first = lineup[0].checkpoint
    if first is not None and first.run:
        experiment, variant = first.run.split("/")
        root = runs_root() / experiment / variant / "evals"
    else:
        root = runs_root() / "_evals"
    return root / suite.name / f"{eval_id}.json"


def find(lineup: Sequence[Any], suite: str | Mapping | Suite, *, fill: Any = None,
         harness: str = "native") -> EvalRecord | None:
    suite = load_suite(suite)
    agents, fill_agent = _resolve(lineup, suite, fill, harness)
    eval_id = canon.digest(eval_key(agents, suite, harness=harness, fill=fill_agent))
    path = _record_path(agents, suite, eval_id)
    return EvalRecord.load(path) if path.exists() else None


# ---------------------------------------------------------------- scoring


def _latest_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for row in canon.read_jsonl(path):
        rows[row["game_id"]] = row
    return rows


def scores(row: Mapping[str, Any], num_slots: int) -> list[float]:
    """Per slot: 1 for the winner, 1/N per seat for a truncation (re-scored from the outcome)."""
    out = [0.0] * num_slots
    seats = row["seats"]
    if row.get("winner") is not None:
        out[row["winner"]] += 1.0
    elif row.get("truncated"):
        for slot in seats:
            out[slot] += 1.0 / len(seats)
    return out


def summarize(rows: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> dict:
    """Score rates with Wilson intervals, per-seat rates and win shares vs 1/N."""
    num_slots = len(labels)
    slots = []
    for slot, label in enumerate(labels):
        mine = [r for r in rows if slot in r["seats"]]
        score = sum(scores(r, num_slots + 1)[slot] for r in mine)
        wins = sum(1 for r in mine if r.get("winner") == slot)
        per_seat = {}
        for r in mine:
            for seat, s in enumerate(r["seats"]):
                if s == slot:
                    entry = per_seat.setdefault(seat, [0.0, 0])
                    entry[0] += scores(r, num_slots + 1)[slot] / r["seats"].count(slot)
                    entry[1] += 1
        sizes = {len(r["seats"]) for r in mine}
        n = len(mine)
        info = {"label": label, "games": n, "score_rate": score / n if n else float("nan"),
                "wilson": list(wilson_interval(score, n)), "wins": wins,
                "per_seat": {str(seat): s / c for seat, (s, c) in sorted(per_seat.items())}}
        if len(sizes) == 1 and n:
            players = sizes.pop()
            info["expected"] = 1.0 / players
            info["win_share_p"] = binomial_two_sided(wins, n, 1.0 / players)
        slots.append(info)
    forfeits = sum(sum(r.get("forfeits") or []) for r in rows)
    return {"games": len(rows), "score_rate": slots[0]["score_rate"], "wilson": slots[0]["wilson"],
            "truncated": sum(1 for r in rows if r.get("truncated")), "forfeits": forfeits, "slots": slots}


def _execution(device: str, infer_rows: int) -> dict:
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cls = torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else "cpu"
    return {"device_class": cls, "torch": torch.__version__, "infer_rows": int(infer_rows)}


# ---------------------------------------------------------------- evaluate


def evaluate(lineup: Sequence[Any], suite: str | Mapping | Suite, *, fill: Any = None, harness: str = "native",
             recompute: bool = False, allow_mixed: bool = False, device: str | None = None,
             num_envs: int | None = None, threads: int | None = None, infer_rows: int | None = None) -> EvalRecord:
    """The record of ``lineup`` on ``suite``: cached, or played (resuming a partial one)."""
    if harness != "native":
        raise NotImplementedError("only the native harness evaluates; Catanatron players are for play and parity")
    suite = load_suite(suite)
    agents, fill_agent = _resolve(lineup, suite, fill, harness)
    key = eval_key(agents, suite, harness=harness, fill=fill_agent)
    eval_id = canon.digest(key)
    path = _record_path(agents, suite, eval_id)
    if path.exists() and not recompute:
        return EvalRecord.load(path)
    device, infer_rows = device or "auto", int(infer_rows or 256)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = os.open(path.with_suffix(".lock"), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists() and not recompute:
            return EvalRecord.load(path)
        if recompute and path.exists():
            result = verify(EvalRecord.load(path), games=None, device=device, num_envs=num_envs, threads=threads)
            if result["differing"]:
                raise RuntimeError(f"{eval_id}: {len(result['differing'])} games differ on replay")
            return EvalRecord.load(path)
        return _play(agents, fill_agent, suite, key, eval_id, path, allow_mixed=allow_mixed, device=device,
                     num_envs=num_envs, threads=threads, infer_rows=infer_rows)
    finally:
        if path.exists():
            path.with_suffix(".lock").unlink(missing_ok=True)  # waiters return the record
        fcntl.flock(lock, fcntl.LOCK_UN)
        os.close(lock)


def _play(agents, fill_agent, suite, key, eval_id, path, *, allow_mixed, device, num_envs, threads,
          infer_rows) -> EvalRecord:
    from catanrl import provenance

    games_path = path.with_suffix(".games.jsonl")
    segments_path = path.with_suffix(".segments.jsonl")
    execution = _execution(device, infer_rows)
    segments = canon.read_jsonl(segments_path)
    mixed = any(s["execution"] != execution for s in segments)
    if mixed and not allow_mixed:
        raise ExecutionMismatch(f"{eval_id} was started under {segments[0]['execution']}, now {execution}; "
                                "pass allow_mixed to finish it here (the record is marked mixed)")
    segment = len(segments)
    canon.append_jsonl(segments_path, {"segment": segment, "execution": execution, "time": time.time(),
                                       "provenance": provenance.capture(kind="eval")})
    schedule = suite.schedule(len(agents), fill=fill_agent is not None)
    done = _latest_rows(games_path)
    missing = [g for g in schedule if g.game_id not in done]
    if missing:
        natives = [a.native(device=device, infer_rows=infer_rows, mode=suite.determinism) for a in agents]
        fill_native = fill_agent.native(device=device, infer_rows=infer_rows, mode=suite.determinism) \
            if fill_agent is not None else None

        def on_game(row: dict) -> None:
            canon.append_jsonl(games_path, {**row, "segment": segment})

        play_schedule(natives, missing, fill=fill_native, device=device, num_envs=int(num_envs or 256),
                      threads=int(threads or 8), infer_rows=infer_rows, on_game=on_game, suite=suite)
    rows = _latest_rows(games_path)
    if set(rows) != {g.game_id for g in schedule}:
        raise RuntimeError(f"{eval_id}: the harness did not play every scheduled game")
    labels = [a.label for a in agents]
    ordered = [rows[g.game_id] for g in schedule]
    summary = summarize(ordered, labels)
    record = {
        "eval_id": eval_id, "key": key, "summary": summary, "suite": suite.name,
        "specs": [a.canonical for a in agents], "fill": fill_agent.canonical if fill_agent else None,
        "refs": [str(a.checkpoint) if a.checkpoint else None for a in agents],
        "fair": [a.fair for a in agents], "labels": labels,
        "allow_drift": any(bool(getattr(a.params, "allow_drift", False)) for a in agents),
        "valid": summary["forfeits"] == 0, "mixed": mixed,
        "games_sha256": canon.digest([{k: v for k, v in r.items() if k != "segment"} for r in ordered], 64),
        "segments": canon.read_jsonl(segments_path), "created": time.time(),
    }
    canon.write_once_json(path, record)
    return EvalRecord.load(path)


# ---------------------------------------------------------------- comparisons and checks


def paired(a: EvalRecord, b: EvalRecord, *, slot_a: int = 0, slot_b: int = 0) -> dict:
    """Delta and exact McNemar of two records on the same games and opponents."""
    if a.key["suite"]["schedule_hash"] != b.key["suite"]["schedule_hash"]:
        raise ValueError("paired records must play the same schedule (schedule hashes differ)")
    others_a = [d for i, d in enumerate(a.key["lineup"]) if i != slot_a] + [a.key["fill"]]
    others_b = [d for i, d in enumerate(b.key["lineup"]) if i != slot_b] + [b.key["fill"]]
    if others_a != others_b:
        raise ValueError("paired records must face the same opponents")
    rows_a, rows_b = {r["game_id"]: r for r in a.games()}, {r["game_id"]: r for r in b.games()}
    if set(rows_a) != set(rows_b):
        raise ValueError(f"paired game ids differ: {sorted(set(rows_a) ^ set(rows_b))[:5]}")
    ids = sorted(rows_a)
    result = _paired_comparison([rows_a[i].get("winner") == slot_a for i in ids],
                                [rows_b[i].get("winner") == slot_b for i in ids])
    return {**result, "a": a.eval_id, "b": b.eval_id}


def verify(record: EvalRecord | str | Path, *, games: int | None = 16, device: str | None = None,
           num_envs: int | None = None, threads: int | None = None) -> dict:
    """Replay the first ``games`` games with this code and machine; count identical rows."""
    if not isinstance(record, EvalRecord):
        record = EvalRecord.load(record)
    data = record.data
    suite = load_suite(data["suite"]) if isinstance(data["suite"], str) else load_suite(data["suite"])
    suite = Suite(**{**suite.__dict__, "boards": record.key["suite"]["boards"]})
    agents, fill_agent = _resolve(data["specs"], suite, data.get("fill"), record.key["harness"])
    stored = {r["game_id"]: r for r in record.games()}
    schedule = suite.schedule(len(agents), fill=fill_agent is not None)
    chosen = schedule if games is None else schedule[: int(games)]
    infer_rows = int(data["segments"][0]["execution"]["infer_rows"]) if data.get("segments") else 256
    natives = [a.native(device=device or "auto", infer_rows=infer_rows, mode=suite.determinism) for a in agents]
    fill_native = fill_agent.native(device=device or "auto", infer_rows=infer_rows, mode=suite.determinism) \
        if fill_agent else None
    found: dict[str, dict] = {}
    play_schedule(natives, chosen, fill=fill_native, device=device or "auto", num_envs=int(num_envs or 256),
                  threads=int(threads or 8), infer_rows=infer_rows, on_game=lambda r: found.__setitem__(r["game_id"], r),
                  suite=suite)

    def same(x: dict, y: dict) -> bool:
        strip = lambda r: {k: v for k, v in r.items() if k != "segment"}  # noqa: E731
        return canon.canonical_json(strip(x)) == canon.canonical_json(strip(y))

    differing = [g.game_id for g in chosen if not same(found[g.game_id], stored[g.game_id])]
    return {"identical": len(chosen) - len(differing), "of": len(chosen), "differing": differing}


def stale(pattern: str = "*") -> list[EvalRecord]:
    """Records whose outcome code differs from this checkout's."""
    from catanrl.provenance import code_id
    from catanrl.runs import runs_root

    import fnmatch

    root = runs_root()
    pattern = pattern if "/" in pattern else f"{pattern}/*"
    paths = [p for p in sorted(root.glob("*/*/evals/*/*.json"))
             if fnmatch.fnmatchcase(f"{p.parents[3].name}/{p.parents[2].name}", pattern)]
    out = []
    for path in paths + sorted(root.glob("_evals/*/*.json")):
        record = EvalRecord.load(path)
        if record.key["code"] != code_id("eval", record.key["harness"]):
            out.append(record)
    return out
