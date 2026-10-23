"""Test 34: tools/search_probe.py: a probe report names every counter, zero or not, and
``--pair`` scores a slot on the games two records share, across code and schedule (said
in the report), refusing another suite, other opponents or an empty intersection."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from eval_helpers import agent, tiny_model

from catanrl.eval.results import EvalRecord

spec = importlib.util.spec_from_file_location("search_probe", Path(__file__).resolve().parents[1] / "tools" / "search_probe.py")
search_probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(search_probe)

SUITE = {"seed": 5, "purpose": "heldout", "boards": 4, "turn_limit": 100, "determinism": "strict",
         "scenarios": [{"game": "base-2p"}]}
AGENT = {"key": "POLICY", "params": {}}
OTHER = {"key": "MCTS", "params": {"sims": 8}}


def write_record(path: Path, eval_id: str, winners: dict[str, int | None], *, code="c1", boards=4,
                 lineup=(AGENT, AGENT), fill=None, suite=SUITE) -> EvalRecord:
    key = {"code": code, "suite": {**suite, "boards": boards, "schedule_hash": f"h{boards}"},
           "lineup": list(lineup), "fill": fill}
    path.write_text(json.dumps({"eval_id": eval_id, "key": key, "summary": {}}))
    rows = [{"game_id": g, "seats": [0, 1], "winner": w, "truncated": w is None} for g, w in winners.items()]
    path.with_suffix(".games.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return EvalRecord.load(path)


def test_a_probe_report_names_every_counter():
    report = search_probe.probe(agent("MCTS", tiny_model(), sims=4, reuse=True))
    assert report["counters"] == {name: 0 for name in search_probe.COUNTERS}
    assert report["reuse_rate"] == 0 and report["per_decision"] == {}
    assert {name: report["dedup"][name] for name in search_probe.DEDUP} == {name: 0 for name in search_probe.DEDUP}
    assert report["dedup"]["batch_hit_fraction"] == 0 and report["dedup"]["forwarded_fraction"] == 0
    assert "dedup" not in search_probe.probe(agent("MCTS", tiny_model(), sims=4, dedup=False))


def test_pair_scores_the_shared_games_and_says_what_differs(tmp_path):
    long = write_record(tmp_path / "long.json", "aaaa", {"s0b0r0": 0, "s0b0r1": 1, "s0b1r0": 0, "s0b1r1": None})
    short = write_record(tmp_path / "short.json", "bbbb", {"s0b0r0": 1, "s0b0r1": 1}, code="c2", boards=1)
    result = search_probe.pair(long, short)
    assert result["games"] == 2 and result["of"] == [4, 2] and (result["a"], result["b"]) == ("aaaa", "bbbb")
    assert result["code_differs"] and result["schedule_differs"]
    assert (result["only_a"], result["only_b"], result["delta"]) == (1, 0, 0.5)
    assert result["rate_a"] == 0.5 and result["rate_b"] == 0.0
    same = search_probe.pair(long, write_record(tmp_path / "same.json", "cccc", {"s0b0r0": 1, "s0b0r1": 1,
                                                                                 "s0b1r0": 0, "s0b1r1": 0}))
    assert same["games"] == 4 and not same["code_differs"] and not same["schedule_differs"]
    # Slot 1's score on the same shared games (the opponents are then checked from its side).
    flipped = search_probe.pair(long, short, slot=1)
    assert (flipped["rate_a"], flipped["rate_b"], flipped["delta"]) == (0.5, 1.0, -0.5)
    with pytest.raises(ValueError, match="no slot"):
        search_probe.pair(long, short, slot=2)
    with pytest.raises(ValueError, match="same suite"):
        search_probe.pair(long, write_record(tmp_path / "suite.json", "dddd", {"s0b0r0": 0},
                                             suite={**SUITE, "turn_limit": 200}))
    with pytest.raises(ValueError, match="opponents"):
        search_probe.pair(long, write_record(tmp_path / "opp.json", "eeee", {"s0b0r0": 0}, lineup=(AGENT, OTHER)))
    with pytest.raises(ValueError, match="share no game"):
        search_probe.pair(long, write_record(tmp_path / "none.json", "ffff", {"s0b9r0": 0}))


def test_the_command_line_pairs_by_path_and_writes_the_report(tmp_path, capsys):
    a = write_record(tmp_path / "a.json", "aaaa", {"s0b0r0": 0, "s0b0r1": 1})
    b = write_record(tmp_path / "b.json", "bbbb", {"s0b0r0": 1, "s0b0r1": 1}, code="c2")
    out = tmp_path / "report.json"
    search_probe.main(["--pair", str(a.path), str(b.path), "--json", str(out)])
    report = json.loads(out.read_text())
    assert report["games"] == 2 and report["code_differs"] and not report["schedule_differs"]
    assert json.loads(capsys.readouterr().out) == report
    with pytest.raises(SystemExit):
        search_probe.main([])
    with pytest.raises(SystemExit):
        search_probe.main(["POLICY:x", "--pair", str(a.path), str(b.path)])


def test_an_eval_id_stored_by_several_runs_resolves_when_their_games_agree():
    """A select of weights another run already played in the same search keys the same eval id
    (critic-refit-2p/head's e0 repeats 856a0421): the id resolves while the copies hold the same
    games, and is refused, naming the files, when they do not."""
    from catanrl.runs import runs_root

    def stored(run: str, eval_id: str, games_sha256: str, winners) -> Path:
        folder = runs_root() / run / "evals" / "select-2p-500"
        folder.mkdir(parents=True)
        path = folder / f"{eval_id}.json"
        write_record(path, eval_id, winners)
        path.write_text(json.dumps({**json.loads(path.read_text()), "games_sha256": games_sha256}))
        return path

    first = stored("az/main", "856a0421185ce86f", "g1", {"s0b0r0": 0})
    stored("refit/head", "856a0421185ce86f", "g1", {"s0b0r0": 0})
    stored("other/x", "856a9999aaaaaaaa", "g3", {"s0b0r0": 1})
    assert search_probe.record("856a0421").path == first
    assert search_probe.record("856a0421185ce86f").eval_id == "856a0421185ce86f"
    with pytest.raises(SystemExit, match="3 records match"):
        search_probe.record("856a")
    stored("refit/main", "856a0421185ce86f", "g2", {"s0b0r0": 1})
    with pytest.raises(SystemExit, match="hold different games"):
        search_probe.record("856a0421")
