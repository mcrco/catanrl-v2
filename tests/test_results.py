"""Test 11: eval records with a fake harness: ids, caching, resume, mixed
execution, forfeits, re-scoring and pairing."""

from __future__ import annotations

import zlib

import pytest

from catanrl.eval import results
from catanrl.eval.results import ExecutionMismatch, evaluate, find, paired, summarize

SUITE = {"name": "fake", "purpose": "heldout", "seed": 5, "boards": 4, "scenarios": [{"game": "base-2p"}],
         "turn_limit": 100}


class FakeHarness:
    def __init__(self, fail_after: int | None = None, forfeit: str | None = None, bias: int = 0):
        self.calls, self.played, self.fail_after, self.forfeit, self.bias = 0, [], fail_after, forfeit, bias

    def __call__(self, lineup, games, *, fill, device, num_envs, threads, infer_rows, on_game, suite):
        self.calls += 1
        for g in games:
            if self.fail_after is not None and len(self.played) >= self.fail_after:
                raise KeyboardInterrupt
            self.played.append(g.game_id)
            seat = (zlib.crc32(g.game_id.encode()) + self.bias) % g.num_players
            truncated = g.board == 3
            on_game({"game_id": g.game_id, "seats": list(g.seats), "winner": None if truncated else g.seats[seat],
                     "winner_seat": None if truncated else seat, "turns": 50, "truncated": truncated,
                     "stuck": False, "forfeits": [int(g.game_id == self.forfeit), 0], "decisions": [1, 1]})


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("CATANRL_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr("catanrl.provenance.code_id", lambda kind="eval", harness="native": {"pinned": "x"})


def use(monkeypatch, harness):
    monkeypatch.setattr(results, "play_schedule", harness)
    return harness


def test_eval_id_is_pinned_and_records_are_cached(monkeypatch):
    harness = use(monkeypatch, FakeHarness())
    record = evaluate(["R", "F"], SUITE)
    assert record.eval_id == "499785eb59c85e27"
    assert record.path.parent.parent.name == "_evals" and record.data["valid"]
    assert record.summary["games"] == 8 and len(list(record.games())) == 8
    again = evaluate(["R", "F"], SUITE)
    assert (again.eval_id, harness.calls) == (record.eval_id, 1)
    assert find(["R", "F"], SUITE).eval_id == record.eval_id
    assert find(["F", "R"], SUITE) is None  # lineup order is not normalized


def test_a_partial_record_plays_only_missing_games(monkeypatch):
    first = use(monkeypatch, FakeHarness(fail_after=3))
    with pytest.raises(KeyboardInterrupt):
        evaluate(["R", "F"], SUITE)
    assert find(["R", "F"], SUITE) is None
    second = use(monkeypatch, FakeHarness())
    record = evaluate(["R", "F"], SUITE)
    assert len(first.played) == 3 and len(second.played) == 5
    assert not set(first.played) & set(second.played)
    assert [s["segment"] for s in record.data["segments"]] == [0, 1] and not record.data["mixed"]


def test_mixed_execution_needs_allow_mixed(monkeypatch):
    use(monkeypatch, FakeHarness(fail_after=2))
    with pytest.raises(KeyboardInterrupt):
        evaluate(["R", "F"], SUITE, infer_rows=8)
    use(monkeypatch, FakeHarness())
    with pytest.raises(ExecutionMismatch):
        evaluate(["R", "F"], SUITE, infer_rows=16)
    assert evaluate(["R", "F"], SUITE, infer_rows=16, allow_mixed=True).data["mixed"]


def test_forfeits_invalidate_and_truncations_rescore(monkeypatch):
    use(monkeypatch, FakeHarness(forfeit="s0b1r0"))
    record = evaluate(["R", "F"], SUITE)
    assert not record.data["valid"] and record.summary["forfeits"] == 1
    rows = list(record.games())
    assert record.summary["truncated"] == 2
    total = sum(s["score_rate"] * s["games"] for s in record.summary["slots"])
    assert total == pytest.approx(8.0)  # every game's point is shared out, truncations by 1/N
    rows = [dict(r, truncated=True, winner=None) for r in rows]  # re-scored from rows alone
    assert summarize(rows, ["a", "b"])["score_rate"] == pytest.approx(0.5)


def test_paired_joins_and_refuses_mismatches(monkeypatch):
    use(monkeypatch, FakeHarness())
    a = evaluate(["R", "F"], SUITE)
    use(monkeypatch, FakeHarness(bias=1))
    b = evaluate(["R:", "F"], {**SUITE, "name": "fake2"})  # same schedule, another record
    result = paired(a, b)
    assert result["games"] == 8 and result["only_a"] + result["only_b"] > 0
    other = evaluate(["R", "F"], {**SUITE, "boards": 2, "name": "fake3"})
    with pytest.raises(ValueError, match="schedule"):
        paired(a, other)
    vs_r = evaluate(["R", "R"], SUITE)
    with pytest.raises(ValueError, match="opponents"):
        paired(a, vs_r)
    rows = b.games_path.read_text().splitlines()
    b.games_path.write_text("\n".join(rows[:-1]) + "\n")
    with pytest.raises(ValueError, match="game ids differ"):
        paired(a, b)
