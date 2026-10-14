"""``catanrl report-data``: the report cites queries, never typed numbers (DESIGN 6.6).

``report/results.yaml``::

    supersedes: [b1b053a]          # refuse checkpoints trained before these commits
    cite:
      g1-vs-F:  {lineup: ["POLICY:ckpt=arch-lineage/graph-s-s42.g1@ref", F], suite: heldout-2p, oracle_ok: true}
      mcts-vs-raw: {paired: [mcts-vs-F, g1-vs-F]}
      arch:     {lineup: ["POLICY:ckpt=arch-lineage/*.g1@best", F], suite: heldout-2p, group_by: model.preset}

becomes ``report/data/results.json`` (rate, interval, n, p, eval id,
canonical specs, ``run@eN#sha``, flags), which Typst reads with ``json()``.
The same file pins every ref: a ref or wildcard that now resolves
differently is a diff that needs ``update``. Refused: non-heldout suites,
unfair slots under test (unless ``oracle_ok``), invalid or drift-allowed
records, and checkpoints whose training lineage predates a ``supersedes``
commit. Dirty, mixed or ``fast`` rows are marked † and ``final`` refuses
them (and re-plays ``verify_games`` games of every cited record).
"""

from __future__ import annotations

import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from catanrl import canon

ENTRY_KEYS = {"lineup", "suite", "fill", "oracle_ok", "group_by", "paired"}


class ReportError(RuntimeError):
    pass


def _expand(lineup: list[str]) -> list[tuple[Any, list[str]]]:
    """Slot 0's wildcard run (``exp/*.g1``) over finished runs: [(run or None, lineup)]."""
    from catanrl.runs import list_runs

    m = re.search(r"(?:ckpt=|^[A-Z]+:)([^:@#]*\*[^:@#]*)", lineup[0])
    if not m:
        return [(None, lineup)]
    runs = list_runs(m.group(1), status="finished")
    return [(run, [lineup[0].replace(m.group(1), run.ref), *lineup[1:]]) for run in runs]


def _descends(commit: str, head: str, repo: Path) -> bool:
    return subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", commit, head],
                          capture_output=True, check=False).returncode == 0


def _lineage_heads(agent: Any) -> list[str]:
    """The superproject HEAD of every launch of the checkpoint's run and its init_from ancestors."""
    from catanrl.runs import load_experiment

    ref = agent.checkpoint
    if ref.run is None:
        from catanrl.checkpoint import Checkpoint

        return [Checkpoint.load(ref.path).header["code"].get("git")]
    run = load_experiment(ref.run)
    runs = [run] + [load_experiment(r.run) for r in run.lineage() if r.run]
    return [p["git"]["repos"]["catanrl-v2"]["head"] for r in runs for p in r.provenance()]


def _flags(record: Any) -> list[str]:
    flags = []
    if record.data.get("mixed"):
        flags.append("mixed")
    if record.key.get("determinism") == "fast":
        flags.append("fast")
    if any(s["provenance"]["git"]["repos"]["catanrl-v2"]["dirty"] for s in record.data.get("segments", [])):
        flags.append("dirty")
    return flags


def _row(name: str, lineup: list[str], entry: dict, *, supersedes: list[str], repo: Path, run_missing: bool,
         final: bool, verify_games: int, problems: list[str]) -> dict | None:
    from catanrl.eval.results import _resolve, evaluate, find, verify
    from catanrl.eval.suites import load_suite

    suite = load_suite(entry["suite"])
    if suite.purpose != "heldout":
        problems.append(f"{name}: {suite.name} is a {suite.purpose} suite; the report cites heldout suites only")
        return None
    agents, fill = _resolve(lineup, suite, entry.get("fill"), "native")
    if not agents[0].fair and not entry.get("oracle_ok"):
        problems.append(f"{name}: {agents[0].label} sees hidden information (add oracle_ok: true to cite it)")
    if agents[0].checkpoint is not None:
        for head in _lineage_heads(agents[0]):
            for commit in supersedes:
                if not head or not _descends(commit, head, repo):
                    problems.append(f"{name}: {agents[0].label} was trained at {str(head)[:12]}, before {commit}")
    record = find(agents, suite, fill=fill)
    if record is None:
        if not run_missing:
            fill_arg = f" --fill {fill.canonical}" if fill else ""
            problems.append(f"{name}: no record; run: catanrl eval {' '.join(a.canonical for a in agents)} "
                            f"--suite {suite.name}{fill_arg}")
            return None
        record = evaluate(agents, suite, fill=fill)
    if not record.data["valid"]:
        problems.append(f"{name}: record {record.eval_id} is invalid (a bot forfeited)")
    if record.data.get("allow_drift"):
        problems.append(f"{name}: record {record.eval_id} was made with allow_drift")
    flags = _flags(record)
    if final:
        if flags:
            problems.append(f"{name}: --final refuses {', '.join(flags)} record {record.eval_id}")
        check = verify(record, games=verify_games)
        if check["differing"]:
            problems.append(f"{name}: {len(check['differing'])} of {check['of']} games differ on replay")
    slot = record.summary["slots"][0]
    return {"rate": slot["score_rate"], "interval": slot["wilson"], "n": slot["games"],
            "p": slot.get("win_share_p"), "eval_id": record.eval_id, "specs": record.data["specs"],
            "refs": record.data["refs"], "suite": suite.name, "flags": flags, "dagger": bool(flags),
            "record": record}


def report_data(spec: str | Path, *, run_missing: bool = False, update: bool = False, final: bool = False,
                verify_games: int = 16) -> Path:
    from catanrl.eval.results import paired
    from catanrl.provenance import repo_root

    spec = Path(spec)
    data = yaml.safe_load(spec.read_text()) or {}
    supersedes = [str(c) for c in data.get("supersedes") or []]
    repo = repo_root()
    problems: list[str] = []
    entries: dict[str, Any] = {}
    records: dict[str, Any] = {}
    for name, entry in (data.get("cite") or {}).items():
        unknown = set(entry) - ENTRY_KEYS
        if unknown:
            problems.append(f"{name}: unknown keys {sorted(unknown)}")
            continue
        if "paired" in entry:
            continue
        rows = []
        for run, lineup in _expand([str(s) for s in entry["lineup"]]):
            row = _row(name, lineup, entry, supersedes=supersedes, repo=repo, run_missing=run_missing, final=final,
                       verify_games=verify_games, problems=problems)
            if row is not None:
                if entry.get("group_by") and run is not None:
                    from catanrl.config import get_path

                    row["group"] = str(get_path(run.config, entry["group_by"]))
                rows.append(row)
        if not rows:
            if "*" in str(entry["lineup"][0]):
                problems.append(f"{name}: the wildcard matches no finished run")
            continue
        records[name] = rows[0]["record"]
        for row in rows:
            row.pop("record")
        if entry.get("group_by"):
            groups: dict[str, list[float]] = {}
            for row in rows:
                groups.setdefault(row["group"], []).append(row["rate"])
            entries[name] = {"rows": rows, "groups": {
                g: {"mean": statistics.mean(r), "sd": statistics.stdev(r) if len(r) > 1 else None, "n": len(r)}
                for g, r in sorted(groups.items())}}
        else:
            entries[name] = rows[0] if len(rows) == 1 else {"rows": rows}
    for name, entry in (data.get("cite") or {}).items():
        if "paired" not in entry:
            continue
        a, b = entry["paired"]
        if a not in records or b not in records:
            problems.append(f"{name}: pairs {a} and {b}, which are not both cited single records")
            continue
        try:
            entries[name] = {k: v for k, v in paired(records[a], records[b]).items()}
        except ValueError as error:
            problems.append(f"{name}: {error}")
    out = spec.parent / "data" / "results.json"
    pins = {name: [r["specs"] for r in (e.get("rows") or [e])] for name, e in entries.items() if "specs" in e
            or "rows" in e}
    if out.exists() and not update:
        old = canon.read_json(out).get("pins", {})
        moved = [f"{name}: {old[name]} -> {pins[name]}" for name in pins if name in old and old[name] != pins[name]]
        if moved:
            problems.append("refs resolve differently than pinned (rerun with --update to accept):\n  "
                            + "\n  ".join(moved))
    if problems:
        raise ReportError("\n".join(problems))
    canon.atomic_write_json(out, {"generated": time.time(), "supersedes": supersedes, "final": final,
                                  "entries": entries, "pins": pins})
    return out
