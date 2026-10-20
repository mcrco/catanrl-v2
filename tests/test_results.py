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
    def __init__(self, fail_after: int | None = None, forfeit: str | None = None, bias: int = 0,
                 trades: bool = False):
        self.calls, self.played, self.fail_after, self.forfeit, self.bias = 0, [], fail_after, forfeit, bias
        self.trades = trades

    def __call__(self, lineup, games, *, fill, device, num_envs, threads, infer_rows, on_game, suite):
        self.calls += 1
        for g in games:
            if self.fail_after is not None and len(self.played) >= self.fail_after:
                raise KeyboardInterrupt
            self.played.append(g.game_id)
            seat = (zlib.crc32(g.game_id.encode()) + self.bias) % g.num_players
            truncated = g.board == 3
            row = {"game_id": g.game_id, "seats": list(g.seats), "winner": None if truncated else g.seats[seat],
                   "winner_seat": None if truncated else seat, "turns": 50, "truncated": truncated,
                   "stuck": False, "forfeits": [int(g.game_id == self.forfeit), 0], "decisions": [1, 1]}
            if self.trades and not truncated:  # as the harness: the field only on a game with a trade
                row["trades"] = {"offers": [2 * int(s == 0) for s in g.seats], "accepts": [int(s != 0) for s in g.seats],
                                 "confirms": [int(s == 0) for s in g.seats],
                                 "cancels": [int(s == 0 and g.board % 2 == 0) for s in g.seats]}
            on_game(row)


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


def test_mixed_execution_needs_allow_mixed_or_a_restart(monkeypatch):
    machine = {"device_class": "A"}
    monkeypatch.setattr(results, "_execution", lambda device, infer_rows: dict(machine))
    use(monkeypatch, FakeHarness(fail_after=2))
    with pytest.raises(KeyboardInterrupt):
        evaluate(["R", "F"], SUITE)
    machine["device_class"] = "B"
    use(monkeypatch, FakeHarness())
    with pytest.raises(ExecutionMismatch):
        evaluate(["R", "F"], SUITE)
    assert evaluate(["R", "F"], SUITE, allow_mixed=True).data["mixed"]
    use(monkeypatch, FakeHarness(fail_after=2))
    with pytest.raises(KeyboardInterrupt):
        evaluate(["R", "R"], SUITE)
    machine["device_class"] = "C"
    restarted = use(monkeypatch, FakeHarness())
    record = evaluate(["R", "R"], SUITE, restart_partial=True)  # select/final evals: nothing cites a partial
    assert len(restarted.played) == 8 and not record.data["mixed"]


def test_cpu_execution_ignores_infer_rows():
    assert results._execution("cpu", 8) == results._execution("cpu", 256)
    assert "cpu_capability" in results._execution("cpu", 8)


def test_games_are_played_under_strict_flags(monkeypatch):
    import torch

    seen = []

    def harness(*args, **kwargs):
        seen.append(torch.are_deterministic_algorithms_enabled())
        FakeHarness()(*args, **kwargs)

    use(monkeypatch, harness)
    evaluate(["R", "F"], SUITE)
    assert seen == [True] and not torch.are_deterministic_algorithms_enabled()


def test_win_share_is_tested_on_decided_games():
    rows = [{"game_id": str(i), "seats": [0, 1], "winner": i % 2, "truncated": False} for i in range(1000)]
    rows += [{"game_id": f"t{i}", "seats": [0, 1], "winner": None, "truncated": True} for i in range(100)]
    slot = summarize(rows, ["a", "b"])["slots"][0]
    assert slot["decided"] == 1000 and slot["win_share_p"] == pytest.approx(1.0)


def test_verify_rebuilds_the_keyed_suite_and_refuses_changed_inputs(monkeypatch, tmp_path):
    import shutil

    import torch

    from catanrl.checkpoint import save
    from catanrl.envs.encoding import action_size, dims_for, row_layout
    from catanrl.models import build_model

    def model(seed):
        torch.manual_seed(seed)
        dims = dims_for(["base-2p"])
        return build_model({"preset": "graph-tiny"}, dims=dims, levels="private", num_actions=action_size(dims),
                           layout=row_layout(dims, "private"))

    pt = tmp_path / "e000000.pt"
    save(pt, model(0), run=None, epoch=0, agent_steps=0, config={}, code={})
    use(monkeypatch, FakeHarness())
    inline = {"purpose": "heldout", "seed": 3, "boards": 2, "scenarios": [{"game": "base-2p"}], "turn_limit": 50}
    record = evaluate([f"POLICY:ckpt={pt}", "F"], inline)
    assert results.verify(record, games=None)["identical"] == 4  # an inline suite replays from its key
    shutil.move(pt, tmp_path / "old.pt")
    save(pt, model(1), run=None, epoch=0, agent_steps=0, config={}, code={})
    with pytest.raises(Exception, match="not #|inputs changed"):
        results.verify(record)


def test_fairness_is_judged_on_the_suite_games(monkeypatch, tmp_path):
    import torch

    from catanrl.agents import resolve_agent
    from catanrl.checkpoint import save
    from catanrl.envs.encoding import action_size, dims_for, row_layout
    from catanrl.models import build_model

    torch.manual_seed(0)
    dims = dims_for(["base-2p"])
    net = build_model({"preset": "graph-tiny"}, dims=dims, levels="public", num_actions=action_size(dims),
                      layout=row_layout(dims, "public"))
    pt = save(tmp_path / "pub.pt", net, run=None, epoch=0, agent_steps=0, config={}, code={}).path
    agent = resolve_agent(f"POLICY:ckpt={pt}")
    assert agent.fair  # judged without games: 2p dims
    use(monkeypatch, FakeHarness())
    suite = {**SUITE, "scenarios": [{"game": "base-4p"}], "boards": 1}
    record = evaluate([agent, "R", "R", "R"], suite)
    assert record.data["fair"][0] is False


def test_paired_needs_the_same_slot_and_code(monkeypatch):
    use(monkeypatch, FakeHarness())
    a = evaluate(["R", "F"], SUITE)
    b = evaluate(["F", "R"], SUITE)
    with pytest.raises(ValueError, match="same slot"):
        paired(a, b, slot_a=0, slot_b=1)


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


def test_trade_rows_are_totalled_and_other_records_stay_as_they_were(monkeypatch):
    """Rows of a trading lineup carry ``trades`` (an outcome: written, verified and summed per
    slot); a record without the field is today's byte for byte (games digest and summary)."""
    from catanrl import canon

    use(monkeypatch, FakeHarness())
    plain = evaluate(["R", "F"], SUITE)
    assert plain.data["games_sha256"] == "aae9172f0d46f48968b17d3ee3e8bebcd3fb1858eec725923133cc82d302653b"
    assert canon.digest(plain.summary) == "df56191dcb18dcb3" and plain.data["critics"] == [None, None]
    assert all("trades" not in s for s in plain.summary["slots"])
    use(monkeypatch, FakeHarness(trades=True))
    record = evaluate(["R", "F"], {**SUITE, "name": "fake-trades"})
    assert sum("trades" in r for r in record.games()) == 6
    assert record.data["games_sha256"] != plain.data["games_sha256"]
    slots = record.summary["slots"]
    assert all(set(r["trades"]) == set(results.TRADE_KEYS) for r in record.games() if "trades" in r)
    assert slots[0]["trades"] == {"offers": 12, "accepts": 0, "confirms": 6, "cancels": 4, "cancel_rate": 0.4,
                                  "per_game": {"offers": 1.5, "accepts": 0.0, "confirms": 0.75, "cancels": 0.5}}
    assert slots[1]["trades"] == {"offers": 0, "accepts": 6, "confirms": 0, "cancels": 0, "cancel_rate": 0.0,
                                  "per_game": {"offers": 0.0, "accepts": 0.75, "confirms": 0.0, "cancels": 0.0}}
    assert results.verify(record, games=None)["identical"] == 8
    use(monkeypatch, FakeHarness())  # a replay that trades differently is not a replay
    assert results.verify(record, games=None)["identical"] == 2  # the two truncated games: no field either way


def test_paired_joins_a_trade_proposer_with_its_policy_control(monkeypatch, tmp_path):
    """[TRADE, N x3] pairs with [POLICY, N x3] (the control, where nobody offers: slots 1-3
    identical in both keys); a table whose other slots differ is refused."""
    import torch

    from catanrl.agents.spec import ResolvedAgent
    from catanrl.checkpoint import save
    from catanrl.envs.encoding import action_size, dims_for, row_layout
    from catanrl.models import build_model

    torch.manual_seed(0)
    dims = dims_for(["base-2p"])
    net = build_model({"preset": "graph-tiny"}, dims=dims, levels="private", num_actions=action_size(dims),
                      layout=row_layout(dims, "private"))
    pt = save(tmp_path / "net.pt", net, run=None, epoch=0, agent_steps=0, config={}, code={}).path
    monkeypatch.setattr(ResolvedAgent, "native", lambda self, **kw: None)  # the fake harness runs no agent
    suite = {**SUITE, "scenarios": [{"game": "base-4p"}], "boards": 2}
    n = f"TRADE:ckpt={pt}:propose=false"
    use(monkeypatch, FakeHarness(trades=True))
    a = evaluate([f"TRADE:ckpt={pt}", n, n, n], suite)
    assert a.data["critics"] == [a.data["refs"][0]] * 4 and a.data["fair"] == [True] * 4
    assert a.summary["slots"][0]["trades"]["confirms"] == 8  # no board truncates: a trade in every game
    use(monkeypatch, FakeHarness(bias=1))
    b = evaluate([f"POLICY:ckpt={pt}", n, n, n], suite)
    assert b.data["critics"][0] is None and b.data["critics"][1:] == a.data["critics"][1:]
    result = paired(a, b)
    assert result["games"] == 8 and result["only_a"] + result["only_b"] > 0
    for other in ([f"POLICY:ckpt={pt}"] * 4, [f"TRADE:ckpt={pt}", n, n, f"{n}:accept_margin=1.0"]):
        with pytest.raises(ValueError, match="opponents"):
            paired(a, evaluate(other, suite))


def test_catanrl_eval_prints_a_trading_slot_s_totals(monkeypatch, tmp_path, capsys):
    """``catanrl eval`` prints the per-slot trade totals the summary holds (the pilot's
    per-game counts and cancel rate), as it prints an LLM slot's per-game stats."""
    import json

    from catanrl.cli import main

    use(monkeypatch, FakeHarness(trades=True))
    path = tmp_path / "fake-cli.yaml"
    path.write_text(json.dumps({k: v for k, v in SUITE.items() if k != "name"}))
    assert main(["eval", "R", "F", "--suite", str(path), "--threads", "1"]) == 0
    out = capsys.readouterr().out
    assert "per game: 1.50 offers, 0.00 accepts, 0.75 confirms, 0.50 cancels; cancel rate 0.40" in out
    assert "per game: 0.00 offers, 0.75 accepts, 0.00 confirms, 0.00 cancels; cancel rate 0.00" in out
