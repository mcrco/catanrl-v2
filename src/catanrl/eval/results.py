"""Eval records: cached by exactly what decides the outcome (DESIGN 6.6).

::

    key = {"schema", "harness", "determinism", "code": code_id("eval"),
           "suite": suite.key(...),            # map hashes and schedule hash included
           "lineup": [agent.key_dict(), ...], "fill": ...}
    eval_id = digest(key)  # runs/<slot-0 run or _evals>/evals/<suite>/<eval_id>.json

``evaluate(lineup, suite)`` returns the record if it exists. Otherwise it
locks the id, plays only the games missing from ``<eval_id>.games.jsonl``
(one appended row per finished game, so a crash resumes exactly) under the
suite's determinism flags, and writes the record once. Each start appends a
segment: its execution (device class, CPU capability, torch, numpy, and
``infer_rows`` on GPUs) and a provenance capture with patches. Resuming
under another execution needs ``allow_mixed`` (the record is marked) or
``restart_partial`` (the partial games are set aside). A forfeit makes the
record ``valid: false``.

``harness="catanatron"`` plays the same schedule with Catanatron players
(:mod:`catanrl.eval.catanatron`: the only harness for ``LLM`` agents); the
harness is in the key, with Catanatron's code in ``code``. Rows of LLM seats
carry an ``llm`` entry (per-game decisions, tokens, retries, fallbacks, wall
time) that the summary totals per slot and ``verify`` ignores. Rows of a game
with a domestic trade (a ``TRADE`` seat at the table) carry ``trades`` (offers,
accepts and confirms per seat), an outcome the summary totals per slot; other
rows are as they always were. A record with
a real language model (not a stub) in its lineup is :class:`Unreplayable`:
``verify`` refuses it unless forced, since the model never answers the same
way twice and every replayed game is paid for.
"""

from __future__ import annotations

import fcntl
import fnmatch
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


HARNESSES = ("native", "catanatron")
#: Row keys that are not outcomes (dropped from games_sha256 and from verify's comparison).
NOT_OUTCOME = ("segment", "llm")


class Unreplayable(RuntimeError):
    """A record whose games no replay reproduces: a real language model decided them."""


def real_llm_slots(key: Mapping[str, Any]) -> list[int]:
    """Lineup slots (``len(lineup)`` for the fill) played by a language model other than a stub."""
    from catanrl.agents.llm.model import is_stub

    slots = [*key.get("lineup", []), key.get("fill")]
    return [i for i, d in enumerate(slots)
            if d and d.get("key") == "LLM" and not is_stub(str((d.get("params") or {}).get("model", "")))]


def play_schedule(*args, **kwargs):  # indirection: tests replace the harness
    from catanrl.eval.harness import play_schedule as play

    return play(*args, **kwargs)


def play_catanatron(*args, **kwargs):  # indirection: tests replace the harness
    from catanrl.eval.catanatron import play_catanatron as play

    return play(*args, **kwargs)


def _play_games(agents, fill_agent, games, suite, harness, *, device, num_envs, threads, infer_rows, on_game) -> None:
    """``games`` with ``agents`` (resolved for ``harness``), rows to ``on_game``."""
    if harness == "catanatron":
        play_catanatron(agents, games, fill=fill_agent, suite=suite, threads=int(threads or 1), on_game=on_game)
        return
    natives = [a.native(device=device, infer_rows=infer_rows, mode=suite.determinism) for a in agents]
    fill_native = fill_agent.native(device=device, infer_rows=infer_rows, mode=suite.determinism) \
        if fill_agent is not None else None
    play_schedule(natives, games, fill=fill_native, device=device, num_envs=int(num_envs or 256),
                  threads=int(threads or 8), infer_rows=infer_rows, suite=suite, on_game=on_game)


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

    def suite(self) -> Suite:
        """The suite exactly as keyed (not as its YAML now reads)."""
        return Suite.from_key(self.data["suite"], self.key["suite"])


# ---------------------------------------------------------------- keys and places


def _resolve(lineup: Sequence[Any], suite: Suite, fill: Any, harness: str):
    """Resolved agents for this suite's games (a resolved agent is re-resolved from its
    canonical spec, so its fairness is judged on these games)."""
    from catanrl.agents import ResolvedAgent, resolve_agent

    def one(spec: Any):
        spec = spec.canonical if isinstance(spec, ResolvedAgent) else spec
        return resolve_agent(spec, harness=harness, games=suite.games())

    return [one(a) for a in lineup], (one(fill) if fill is not None else None)


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
        root = runs_root() / first.run / "evals"
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


def all_records(pattern: str = "*") -> list[Path]:
    """Record files of runs matching ``pattern``, plus those under ``_evals``."""
    from catanrl.runs import runs_root

    root = runs_root()
    pattern = pattern if "/" in pattern else f"{pattern}/*"
    paths = [p for p in sorted(root.glob("*/*/evals/*/*.json"))
             if fnmatch.fnmatchcase(f"{p.parents[3].name}/{p.parents[2].name}", pattern)]
    return paths + sorted(root.glob("_evals/*/*.json"))


# ---------------------------------------------------------------- scoring


def _latest_rows(path: Path) -> dict[str, dict]:
    return {row["game_id"]: row for row in canon.read_jsonl(path)}


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
    """Score rates with Wilson intervals, per-seat rates, and win shares vs 1/N
    tested on the decided games (a truncation is nobody's loss)."""
    num_slots = len(labels)
    slots = []
    for slot, label in enumerate(labels):
        mine = [r for r in rows if slot in r["seats"]]
        score = sum(scores(r, num_slots + 1)[slot] for r in mine)
        per_seat: dict[int, list] = {}
        for r in mine:
            for seat, s in enumerate(r["seats"]):
                if s == slot:
                    entry = per_seat.setdefault(seat, [0.0, 0])
                    entry[0] += scores(r, num_slots + 1)[slot] / r["seats"].count(slot)
                    entry[1] += 1
        decided = [r for r in mine if r.get("winner") is not None]
        wins = sum(1 for r in decided if r["winner"] == slot)
        n = len(mine)
        info = {"label": label, "games": n, "score_rate": score / n if n else float("nan"),
                "wilson": list(wilson_interval(score, n)), "wins": wins, "decided": len(decided),
                "per_seat": {str(seat): s / c for seat, (s, c) in sorted(per_seat.items())}}
        sizes = {len(r["seats"]) for r in mine}
        if len(sizes) == 1 and decided:
            players = sizes.pop()
            info["expected"] = 1.0 / players
            info["win_share_p"] = binomial_two_sided(wins, len(decided), 1.0 / players)
        llm = [r["llm"][slot] for r in mine if r.get("llm") and slot < len(r["llm"]) and r["llm"][slot]]
        if llm:
            info["llm"] = llm_totals(llm)
        trading = [r for r in mine if r.get("trades")]  # written only by trading lineups, and only when non-zero
        if trading:
            info["trades"] = trade_totals(trading, slot, n)
        slots.append(info)
    return {"games": len(rows), "score_rate": slots[0]["score_rate"], "wilson": slots[0]["wilson"],
            "truncated": sum(1 for r in rows if r.get("truncated")),
            "forfeits": sum(sum(r.get("forfeits") or []) for r in rows), "slots": slots}


def trade_totals(rows: Sequence[Mapping[str, Any]], slot: int, games: int) -> dict:
    """One slot's domestic trades over its ``games``: offers it made, offers it accepted and
    trades it confirmed at its seats (a row's ``trades`` holds each per seat; a game without
    the field had none), with per-game means."""
    keys = ("offers", "accepts", "confirms")
    totals = {k: 0 for k in keys}
    for r in rows:
        for seat, s in enumerate(r["seats"]):
            if s == slot:
                for k in keys:
                    totals[k] += int(r["trades"][k][seat])
    return {**totals, "per_game": {k: totals[k] / games for k in keys}}


def llm_totals(stats: Sequence[Mapping[str, Any]]) -> dict:
    """One slot's LLM stats over its games: sums, per-game means, and whether any token count
    was an estimate (the stubs; a provider without usage) rather than the SDK's usage."""
    keys = ("decisions", "forced", "calls", "prompt_tokens", "completion_tokens", "retries", "fallbacks", "wall_s")
    totals = {k: sum(s.get(k, 0) for s in stats) for k in keys}
    n = len(stats)
    out = {"games": n, **totals, "estimated": any(s.get("estimated") for s in stats)}
    out["per_game"] = {k: totals[k] / n for k in keys}
    out["fallback_rate"] = totals["fallbacks"] / totals["decisions"] if totals["decisions"] else 0.0
    return out


def _execution(device: str, infer_rows: int) -> dict:
    """What may move a row's bits: the device class (and on CPU the kernel ISA), torch,
    numpy (sampled moves), and on GPUs the padded inference shape."""
    import numpy as np
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        return {"device_class": torch.cuda.get_device_name(torch.device(device)), "torch": torch.__version__,
                "numpy": np.__version__, "infer_rows": int(infer_rows)}
    # Strict CPU evaluation runs one row at a time, so infer_rows does not matter there.
    return {"device_class": "cpu", "cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "torch": torch.__version__, "numpy": np.__version__}


def _strict(suite: Suite):
    """The suite's determinism flags, with autocast off (PuffeRL leaks bf16 autocast)."""
    import contextlib

    import torch

    from catanrl.seeding import determinism

    stack = contextlib.ExitStack()
    stack.enter_context(torch.autocast("cuda", enabled=False))
    stack.enter_context(torch.autocast("cpu", enabled=False))
    stack.enter_context(determinism(suite.determinism))
    return stack


# ---------------------------------------------------------------- evaluate


def evaluate(lineup: Sequence[Any], suite: str | Mapping | Suite, *, fill: Any = None, harness: str = "native",
             recompute: bool = False, allow_mixed: bool = False, restart_partial: bool = False,
             device: str | None = None, num_envs: int | None = None, threads: int | None = None,
             infer_rows: int | None = None) -> EvalRecord:
    """The record of ``lineup`` on ``suite``: cached, or played (resuming a partial one).

    ``recompute`` replays every game of an existing record and raises if any differs.
    """
    if harness not in HARNESSES:
        raise ValueError(f"harness must be one of {HARNESSES}, not {harness!r}")
    suite = load_suite(suite)
    agents, fill_agent = _resolve(lineup, suite, fill, harness)
    key = eval_key(agents, suite, harness=harness, fill=fill_agent)
    eval_id = canon.digest(key)
    path = _record_path(agents, suite, eval_id)
    if path.exists() and not recompute:
        return EvalRecord.load(path)
    # Catanatron players run their networks on the CPU at batch 1 unless their spec says otherwise.
    device = device or ("cpu" if harness == "catanatron" else "auto")
    infer_rows = int(infer_rows or 256)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = os.open(path.with_suffix(".lock"), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            if recompute:
                result = verify(EvalRecord.load(path), games=None, device=device, num_envs=num_envs, threads=threads)
                if result["differing"]:
                    raise RuntimeError(f"{eval_id}: {len(result['differing'])} games differ on replay")
            return EvalRecord.load(path)
        return _play(agents, fill_agent, suite, key, eval_id, path, harness=harness, allow_mixed=allow_mixed,
                     restart_partial=restart_partial, device=device, num_envs=num_envs, threads=threads,
                     infer_rows=infer_rows)
    finally:
        if path.exists():
            path.with_suffix(".lock").unlink(missing_ok=True)  # waiters return the record
        fcntl.flock(lock, fcntl.LOCK_UN)
        os.close(lock)


def _segment(path: Path, eval_id: str, segment: int, execution: dict) -> dict:
    """Record this start: provenance (with patches and keep refs) in its own file, a small row."""
    from catanrl import provenance

    prov = provenance.capture(kind="eval")
    provenance.check_launch(prov, {})
    folder = path.parent / "provenance"
    name = f"{eval_id}.start{len(list(folder.glob(f'{eval_id}.start*.json')))}"  # unique across restarts
    prov["run"]["launch_name"] = name
    provenance.save_patches(prov, folder)
    provenance.keep_commits(prov)
    canon.write_once_json(folder / f"{name}.json", prov)
    row = {"segment": segment, "execution": execution, "time": time.time(), "dirty": provenance.any_dirty(prov),
           "git": prov["git"]["repos"]["catanrl-v2"]["head"], "provenance": f"provenance/{name}.json"}
    canon.append_jsonl(path.with_suffix(".segments.jsonl"), row)
    return row


def _play(agents, fill_agent, suite, key, eval_id, path, *, harness="native", allow_mixed, restart_partial, device,
          num_envs, threads, infer_rows) -> EvalRecord:
    games_path, segments_path = path.with_suffix(".games.jsonl"), path.with_suffix(".segments.jsonl")
    execution = _execution(device, infer_rows)
    segments = canon.read_jsonl(segments_path)
    if any(s["execution"] != execution for s in segments):
        if restart_partial:  # nothing cites a partial record: set its games aside and start over
            stamp = int(time.time())
            for p in (games_path, segments_path):
                if p.exists():
                    p.rename(p.with_name(f"{p.name}.discarded-{stamp}"))
            segments = []
        elif not allow_mixed:
            raise ExecutionMismatch(f"{eval_id} was started under {segments[0]['execution']}, now {execution}; "
                                    "pass allow_mixed to finish it here (the record is marked mixed)")
    segment = len(segments)
    _segment(path, eval_id, segment, execution)
    schedule = suite.schedule(len(agents), fill=fill_agent is not None)
    done = _latest_rows(games_path)
    missing = [g for g in schedule if g.game_id not in done]
    if missing:
        with _strict(suite):
            _play_games(agents, fill_agent, missing, suite, harness, device=device, num_envs=num_envs,
                        threads=threads, infer_rows=infer_rows,
                        on_game=lambda row: canon.append_jsonl(games_path, {**row, "segment": segment}))
    rows = _latest_rows(games_path)
    if set(rows) != {g.game_id for g in schedule}:
        raise RuntimeError(f"{eval_id}: the harness did not play every scheduled game")
    labels = [a.label for a in agents]
    ordered = [rows[g.game_id] for g in schedule]
    summary = summarize(ordered, labels)
    segments = canon.read_jsonl(segments_path)
    record = {
        "eval_id": eval_id, "key": key, "summary": summary, "suite": suite.name,
        "specs": [a.canonical for a in agents], "fill": fill_agent.canonical if fill_agent else None,
        "refs": [str(a.checkpoint) if a.checkpoint else None for a in agents],
        # PREDICT's second checkpoint per slot (None elsewhere): both are pinned in the key.
        "predictors": [str(a.predictor) if getattr(a, "predictor", None) else None for a in agents],
        # TRADE's critic per slot (None elsewhere), pinned in the key the same way.
        "critics": [str(a.critic) if getattr(a, "critic", None) else None for a in agents],
        "fair": [a.fair for a in agents], "labels": labels,
        "allow_drift": any(bool(getattr(a.params, "allow_drift", False)) for a in agents),
        "valid": summary["forfeits"] == 0, "mixed": len({canon.digest(s["execution"]) for s in segments}) > 1,
        "games_sha256": canon.digest([_strip(r) for r in ordered], 64),
        "segments": segments, "created": time.time(),
    }
    canon.write_once_json(path, record)
    return EvalRecord.load(path)


# ---------------------------------------------------------------- comparisons and checks


def paired(a: EvalRecord, b: EvalRecord, *, slot_a: int = 0, slot_b: int = 0) -> dict:
    """Delta and exact McNemar of two records that differ only in one slot's agent."""
    if slot_a != slot_b:
        raise ValueError("paired records compare the same slot (same seat order against the same opponents)")
    if a.key["suite"]["schedule_hash"] != b.key["suite"]["schedule_hash"]:
        raise ValueError("paired records must play the same schedule (schedule hashes differ)")
    for part in ("code", "determinism", "harness"):
        if a.key[part] != b.key[part]:
            raise ValueError(f"paired records differ in {part}")
    others_a = [d for i, d in enumerate(a.key["lineup"]) if i != slot_a] + [a.key["fill"]]
    others_b = [d for i, d in enumerate(b.key["lineup"]) if i != slot_b] + [b.key["fill"]]
    if others_a != others_b or len(a.key["lineup"]) != len(b.key["lineup"]):
        raise ValueError("paired records must face the same opponents in the same seats")
    rows_a, rows_b = {r["game_id"]: r for r in a.games()}, {r["game_id"]: r for r in b.games()}
    if set(rows_a) != set(rows_b):
        raise ValueError(f"paired game ids differ: {sorted(set(rows_a) ^ set(rows_b))[:5]}")
    ids = sorted(rows_a)
    result = _paired_comparison([rows_a[i].get("winner") == slot_a for i in ids],
                                [rows_b[i].get("winner") == slot_b for i in ids])
    return {**result, "a": a.eval_id, "b": b.eval_id}


def verify(record: EvalRecord | str | Path, *, games: int | None = 16, device: str | None = None,
           num_envs: int | None = None, threads: int | None = None, force: bool = False) -> dict:
    """Replay the first ``games`` games with this code and machine; count identical rows.

    The suite and agents are rebuilt from the record's key; inputs that no
    longer resolve to that key (other weights at a path, an edited game) raise.
    A record with a real language model in its lineup is refused
    (:class:`Unreplayable`) unless ``force``: it never replays identically and
    every replayed game is paid for.
    """
    if not isinstance(record, EvalRecord):
        record = EvalRecord.load(record)
    if (slots := real_llm_slots(record.key)) and not force:
        labels = record.data.get("labels") or []
        names = ", ".join(labels[s] if s < len(labels) else f"slot {s}" for s in slots)
        raise Unreplayable(f"{record.eval_id}: {names} is a real language model, which never replays identically, "
                           "and every replayed game is paid for; catanrl verify --force (verify(force=True)) "
                           "spends it anyway")
    data, suite = record.data, record.suite()
    agents, fill_agent = _resolve(data["specs"], suite, data.get("fill"), record.key["harness"])
    rebuilt = eval_key(agents, suite, harness=record.key["harness"], fill=fill_agent)
    changed = [k for k in record.key if k != "code" and rebuilt.get(k) != record.key[k]]
    if changed:
        raise RuntimeError(f"{record.eval_id}: its inputs changed ({', '.join(changed)}); this is not a replay")
    stored = {r["game_id"]: r for r in record.games()}
    schedule = suite.schedule(len(agents), fill=fill_agent is not None)
    chosen = schedule if games is None else schedule[: int(games)]
    first = (data.get("segments") or [{}])[0].get("execution") or {}
    infer_rows = int(first.get("infer_rows", 256))
    harness = record.key["harness"]
    found: dict[str, dict] = {}
    with _strict(suite):
        _play_games(agents, fill_agent, chosen, suite, harness,
                    device=device or ("cpu" if harness == "catanatron" else "auto"), num_envs=num_envs,
                    threads=threads, infer_rows=infer_rows, on_game=lambda r: found.__setitem__(r["game_id"], r))
    differing = [g.game_id for g in chosen
                 if canon.canonical_json(_strip(found[g.game_id])) != canon.canonical_json(_strip(stored[g.game_id]))]
    return {"identical": len(chosen) - len(differing), "of": len(chosen), "differing": differing}


def _strip(row: Mapping[str, Any]) -> dict:
    return {k: v for k, v in row.items() if k not in NOT_OUTCOME}


def stale(pattern: str = "*") -> list[EvalRecord]:
    """Records whose outcome code differs from this checkout's."""
    from catanrl.provenance import code_id

    records = [EvalRecord.load(p) for p in all_records(pattern)]
    return [r for r in records if r.key["code"] != code_id("eval", r.key["harness"])]
