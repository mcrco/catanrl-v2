"""``kernels=fused16`` in an MCTS spec (the half-precision kernels): a value of the keyed field
with a canonical spec, a key and an eval id of its own, which leaves every spec without it what
it was; a strict evaluator resolves a half overflow row by row, so a row's outputs are the
row's whatever its batch (what the agent's cache and a replay rest on); the agent checks the
half values against the float32 kernels' at a model's first decision. On a GPU: its strict
games do not depend on the cache or the run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from catanatron.registry import SpecError
from eval_helpers import agent, tiny_model
from test_agent_cache import GAMES, MODEL, decide, mcts, play, positions
from test_agent_spec import _rebuilds_unchanged, ckpts  # noqa: F401  (the fixture)
from test_graph_kernels import needs_kernels, shared_model

from catanrl import canon
from catanrl.agents import check_spec_syntax, resolve_agent
from catanrl.agents.native import HALF_VALUE_TOLERANCE
from catanrl.agents.spec import MctsParams, _fmt, keyed_fields
from catanrl.eval import results
from catanrl.eval.suites import Suite
from catanrl.search import BatchEvaluator, DedupEvaluator
from catanrl.search.pool import EVAL_CACHE_ENV

STORED_MCTS_KEYS = Path(__file__).with_name("data") / "stored_mcts_keys.json"
REST = "sims=8:dets=1:pending=64:reuse=true:transpositions=true"


def test_half_precision_kernels_are_a_key_of_their_own(ckpts):  # noqa: F811
    """fused16's outputs are within rounding of fused's, not its bits: its canonical spec, key
    and eval id are its own, apart from fused's and from the module's by the one field."""
    suite = Suite(name="t", purpose="dev", seed=1, boards=1, scenarios=({"game": "base-2p"},), turn_limit=500)

    def eval_id(resolved) -> str:
        return canon.digest(results.eval_key([resolved, resolve_agent("F")], suite, harness="native"))

    base, torch_, fused, half = (resolve_agent(f"MCTS:ckpt=exp/v:{REST}{tail}")
                                 for tail in ("", ":kernels=torch", ":kernels=fused", ":kernels=fused16"))
    assert half.canonical == base.canonical + ":kernels=fused16"
    assert resolve_agent(half.canonical).canonical == half.canonical
    assert half.key_dict()["params"] == {**base.key_dict()["params"], "kernels": "fused16"}
    # The key is fused's but for that value: the search, the checkpoint and the rest are the same.
    assert {**half.key_dict(), "params": fused.key_dict()["params"]} == fused.key_dict()
    assert len({eval_id(resolved) for resolved in (base, fused, half)}) == 3
    assert half.native  # built on a CUDA device only, as fused
    with pytest.raises(ValueError, match="kernels=fused16 .* CUDA"):
        half.native(device="cpu")
    check_spec_syntax("MCTS:ckpt=exp/v@best:kernels=fused16")
    with pytest.raises(SpecError, match="not one of torch, fused, fused16"):
        resolve_agent("MCTS:ckpt=exp/v:sims=8:kernels=half")

    # A spec without it is what it was: no kernels part and no kernels in the key by default
    # (and for kernels=torch), fused its own part and value.
    assert base.canonical == ("MCTS:ckpt=" + base.canonical.split("ckpt=")[1].split(":")[0] + ":sims=8:dets=1"
                              ":oracle=false:root=puct:c_puct=0.75:gumbel_m=16:temperature=0.0:noise=false"
                              ":allow_drift=false:tracker=pool:reuse=true:transpositions=true")
    assert torch_.canonical == base.canonical and torch_.key_dict() == base.key_dict()
    assert eval_id(torch_) == eval_id(base)
    assert fused.canonical == base.canonical + ":kernels=fused"
    assert base.key_dict()["params"] == {
        "sims": 8, "dets": 1, "oracle": False, "root": "puct", "c_puct": 0.75, "gumbel_m": 16, "temperature": 0.0,
        "noise": False, "allow_drift": False, "tracker": "pool", "reuse": True, "transpositions": True}
    assert fused.key_dict()["params"] == {**base.key_dict()["params"], "kernels": "fused"}


def test_stored_fused_keys_and_specs_rebuild_unchanged():
    """The stored records that searched with kernels=fused (three of them in the file, the
    standard row's among them): their key params, their search config and their canonical
    spec, to the character, rebuild from the params as they are read today."""
    entries = [entry for entry in json.loads(STORED_MCTS_KEYS.read_text()) if entry["params"].get("kernels")]
    assert len(entries) >= 3 and {entry["params"]["kernels"] for entry in entries} == {"fused"}
    for entry in entries:
        _rebuilds_unchanged(entry)
        params = MctsParams(ckpt="x", **entry["params"], pending=entry["search"]["max_pending"])
        ckpt = entry["spec"].split(":")[1]
        rebuilt = ":".join(["MCTS", ckpt, *(f"{name}={_fmt(getattr(params, name))}"
                                           for name in keyed_fields(params) if name != "ckpt")])
        assert rebuilt == entry["spec"] and ":kernels=fused" in rebuilt
    assert any(entry["search"]["max_pending"] == 64 and entry["params"]["sims"] == 1024 for entry in entries)


# ------------------------------------------------------------ the strict evaluator's overflow

EVAL_MODEL = tiny_model(["base-2p"], level="full", seed=2)
ROWS = np.stack([agent("POLICY", EVAL_MODEL).encode(game, game.current_player_index) for game in positions(12)])


class Halves(BatchEvaluator):
    """kernels=fused16 stood in on the CPU. A forward has ``infer_rows`` rows in strict mode, as
    on a GPU; a row's "half" outputs are the network's own for that row alone with the logits
    one higher; the forward's flag has bit 2 when any row of it (padding included) is in
    ``bad``: an overflow is a forward's, not a row's. ``_exact`` answers the network's own."""

    def __init__(self, bad=(), mode="strict", left=0) -> None:
        super().__init__(EVAL_MODEL, device="cpu", infer_rows=8, mode=mode)
        self.plain = BatchEvaluator(EVAL_MODEL, device="cpu", infer_rows=8, mode="strict")  # a row per forward
        self.kernels, self.fused = "fused16", object()
        self.bad, self.left = {ROWS[i].tobytes() for i in bad}, left
        self.sizes: list[int] = []

    def bucket(self, rows: int) -> int:
        return self.infer_rows if self.mode == "strict" else super().bucket(rows)

    def _forward(self, obs, state=None):
        rows = obs.numpy()
        self.sizes.append(len(rows))
        logits, values = self.plain(rows)
        flag = 2 * any(row.tobytes() in self.bad for row in rows) | self.left
        return torch.from_numpy(logits + np.float32(1)), torch.from_numpy(values), torch.full((1,), flag, dtype=torch.int32)

    def _exact(self, chunk, size):
        logits, values = self.plain(chunk)
        return logits, values, np.full(1, self.left, dtype=np.int32)


def expected(bad) -> tuple[np.ndarray, np.ndarray]:
    """Every row's outputs by the row alone: the float32 ones for a row that overflows, the
    half ones for any other."""
    logits, values = BatchEvaluator(EVAL_MODEL, device="cpu", infer_rows=8, mode="strict")(ROWS)
    half = logits + np.float32(1)
    half[list(bad)] = logits[list(bad)]
    return half, values


def test_a_strict_half_overflow_is_resolved_row_by_row():
    """Strict rows do not depend on their batch, with kernels=fused16 too: a row that overflows
    takes the float32 kernels' outputs and every other row its half ones, wherever the row
    sits, whatever shares its chunk and whether or not the chunk is padded."""
    clean = Halves()
    logits, values = clean._run(ROWS[:8])
    assert np.array_equal(logits, expected(())[0][:8]) and np.array_equal(values, expected(())[1][:8])
    assert (clean.exact_forwards, clean.exact_rows, clean.forwards, clean.rows) == (0, 0, 1, 8)

    one = Halves(bad=[3])
    logits, values = one._run(ROWS[:8])
    assert np.array_equal(logits, expected([3])[0][:8]) and np.array_equal(values, expected([3])[1][:8])
    # The chunk, its halves, the flagged half's halves and the flagged pair's rows: 7 forwards
    # of the forward's own size, and one row in float32.
    assert (one.exact_forwards, one.exact_rows, one.forwards, one.rows) == (1, 1, 7, 8)
    assert one.sizes == [8] * 7

    rng = np.random.default_rng(0)
    for bad in ([3], [0], [7], [2, 3, 9], list(range(12))):
        want = expected(bad)
        evaluator = Halves(bad=bad)
        for size in (8, 8, 8, 5, 3, 2, 1, 1):
            index = rng.choice(12, size=size, replace=False)
            logits, values = evaluator._run(ROWS[index])
            assert np.array_equal(logits, want[0][index]) and np.array_equal(values, want[1][index])
        first = np.array([bad[0], *(i for i in range(12) if i != bad[0])][:5])  # padded with a row that overflows
        logits, _ = evaluator._run(ROWS[first])
        assert np.array_equal(logits, want[0][first])
        assert evaluator.exact_rows >= evaluator.exact_forwards > 0
    assert Halves(bad=list(range(12)))._run(ROWS[:8])[0].tolist() == expected(range(12))[0][:8].tolist()


def test_a_fast_half_overflow_takes_the_chunk_and_a_flag_left_raises():
    """Fast mode, whose rows depend on their batch anyway, evaluates the flagged chunk again in
    float32 as it always did; a flag that is left after the overflow is dealt with (the
    topology's) raises in both modes."""
    fast = Halves(bad=[3], mode="fast")
    logits, values = fast._run(ROWS[:8])
    plain = BatchEvaluator(EVAL_MODEL, device="cpu", infer_rows=8, mode="strict")(ROWS[:8])
    assert np.array_equal(logits, plain[0]) and np.array_equal(values, plain[1])
    assert (fast.exact_forwards, fast.exact_rows, fast.forwards) == (1, 8, 1)
    for mode in ("strict", "fast"):
        for bad in ((), (3,)):
            with pytest.raises(RuntimeError, match="hex boards only"):
                Halves(bad=bad, mode=mode, left=1)._run(ROWS[:8])


# ------------------------------------------------------------ the agent's check

def halve(searcher, offset: float) -> list[int]:
    """The agent's evaluator made to say it runs the half kernels, with float32 values
    ``offset`` away from the ones it answers; the rows of each check it is asked for."""
    inner = searcher.evaluator.evaluator if isinstance(searcher.evaluator, DedupEvaluator) else searcher.evaluator
    inner.kernels, checked = "fused16", []

    def exact(rows):
        checked.append(len(rows))
        logits, values = inner(rows)
        return logits, values + np.float32(offset)

    inner.exact = exact
    return checked


@pytest.mark.parametrize("megabytes", [None, 0.0], ids=["cache", "python"])
def test_half_values_are_checked_at_a_models_first_decision(monkeypatch, megabytes):
    """kernels=fused16: before a model's first searches the agent holds the half values of the
    acting rows against the float32 kernels' (``evaluator.exact``), once per model; within the
    tolerance the decisions and the evaluator's counters are those of an agent that checks
    nothing, past it the agent refuses to search."""
    assert HALF_VALUE_TOLERANCE == 1e-2
    games = positions()

    def inner(searcher):
        return searcher.evaluator.evaluator if megabytes == 0 else searcher.evaluator

    plain = mcts(monkeypatch, megabytes)
    want = [decide(plain, games), decide(plain, games)]
    first = mcts(monkeypatch, megabytes)
    checked = halve(first, 5e-3)
    assert [decide(first, games), decide(first, games)] == want and checked == [len(games)]
    assert (inner(first).rows, inner(first).forwards) == (inner(plain).rows, inner(plain).forwards)
    assert (inner(first).exact_forwards, inner(first).exact_rows) == (0, 0)
    evaluator = inner(first)
    first._set_model(MODEL)  # another model on the agent: checked at its own first decision
    assert inner(first) is not evaluator
    again = halve(first, -5e-3)
    assert decide(first, games) == want[0] and decide(first, games) == want[1]
    assert again == [len(games)] and checked == [len(games)]

    for offset in (2e-2, float("nan")):
        bad = mcts(monkeypatch, megabytes)
        checked = halve(bad, offset)
        for _ in range(2):  # refused at every decision: nothing was searched, nothing is left
            with pytest.raises(RuntimeError, match="kernels=fused16.*search this network with kernels=fused"):
                decide(bad, games)
            assert not list(bad.pool.ids) and bad.counters["decisions"] == 0 and inner(bad).rows == 0
        assert checked == [len(games)] * 2
    many = mcts(monkeypatch, megabytes, dets=1)
    checked = halve(many, 0.0)
    decide(many, positions(4) * 20)  # the first 64 rows of the decision
    assert checked == [64]


# ------------------------------------------------------------ on a GPU

def half_agent(monkeypatch, megabytes, model, kernels="fused16", **params):
    if megabytes is None:
        monkeypatch.delenv(EVAL_CACHE_ENV, raising=False)
    else:
        monkeypatch.setenv(EVAL_CACHE_ENV, repr(megabytes))
    return agent("MCTS", model, infer_rows=32, device="cuda",
                 **{"sims": 24, "dets": 1, "pending": 8, "reuse": True, "transpositions": True, "kernels": kernels,
                    **params})


@pytest.mark.gpu
@needs_kernels
def test_strict_half_precision_games_depend_on_neither_the_cache_nor_the_run(monkeypatch):
    """Strict games of a fused16 agent: row for row the same through the engine's cache, the
    Python dedup and no dedup, and from one run (one agent, one set of CUDA graphs) to the
    next; fused's games are the same three ways too."""
    model = shared_model(["base-2p"])
    for kernels in ("fused16", "fused"):
        played = [play(half_agent(monkeypatch, megabytes, model, kernels, dedup=dedup), num_envs=4, device="cuda")
                  for megabytes, dedup in ((None, True), (None, True), (0.0, True), (None, False))]
        assert played[0] == played[1] == played[2] == played[3] and len(played[0]) == len(GAMES)
    cached = half_agent(monkeypatch, None, model)
    play(cached, num_envs=4, device="cuda")
    stats = cached.cache_stats
    assert stats["hits"] > 0 and stats["forwarded"] == cached.evaluator.rows < stats["leaves"]
    assert cached.evaluator.exact_forwards == 0 == cached.evaluator.exact_rows


@pytest.mark.gpu
@needs_kernels
def test_a_strict_half_overflow_on_the_kernels_is_a_rows_own():
    """The kernels themselves: with weights that overflow a half on some rows and not on others,
    a strict evaluator answers every row the same alone, in a full batch and beside other rows
    (the float32 kernels' outputs where the row overflows, the half ones elsewhere)."""
    from test_graph_kernels import rows as kernel_rows

    model = shared_model(["base-2p"])
    obs = kernel_rows(model, "base-2p", count=48)

    def overflowing(factor: float):
        """(the model with a layer's weight times ``factor``, which rows overflow a half alone)."""
        scaled = shared_model(["base-2p"])
        with torch.no_grad():
            dict(scaled.named_parameters())["backbone.layers.1.node_self.weight"].mul_(factor)
        probe = BatchEvaluator(scaled, device="cuda", infer_rows=8, mode="strict", kernels="fused16")
        return scaled, np.array([int(probe._replay(obs[i:i + 1], None, 8)[-1][0]) & 2 for i in range(len(obs))],
                                dtype=bool)

    # Between a factor at which no row overflows and one at which every row does, one at which some do.
    low, high = 1.0, 1e4
    assert not overflowing(low)[1].any() and overflowing(high)[1].all()
    for _ in range(24):
        scale, some = overflowing((low * high) ** 0.5)
        if some.all():
            high = (low * high) ** 0.5
        elif not some.any():
            low = (low * high) ** 0.5
        else:
            break
    else:
        pytest.skip("no factor found at which some rows overflow and others do not")
    evaluator = BatchEvaluator(scale, device="cuda", infer_rows=8, mode="strict", kernels="fused16")
    exact = BatchEvaluator(scale, device="cuda", infer_rows=8, mode="strict", kernels="fused")(obs)
    together = evaluator(obs)
    assert evaluator.exact_forwards > 0 and evaluator.exact_rows == int(some.sum())
    assert np.array_equal(together[0][some], exact[0][some]) and np.array_equal(together[1][some], exact[1][some])
    assert not np.array_equal(together[0][~some], exact[0][~some])  # the others kept their half outputs
    order = np.random.default_rng(1).permutation(len(obs))
    shuffled = evaluator(obs[order])
    assert np.array_equal(shuffled[0], together[0][order]) and np.array_equal(shuffled[1], together[1][order])
    for i in (int(np.flatnonzero(some)[0]), int(np.flatnonzero(~some)[0])):
        single = evaluator(obs[i:i + 1])
        assert np.array_equal(single[0][0], together[0][i]) and np.array_equal(single[1][0], together[1][i])
