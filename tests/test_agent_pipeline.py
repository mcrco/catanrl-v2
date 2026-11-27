"""The MCTS agent's searches through self-play's pipeline (``run_groups``: on a CUDA device,
and on a CPU with ``CATANRL_EVAL_PIPELINE=1``) and in search groups (``groups=2``): in strict
mode the games are the serial loop's (``run_pools``, ``CATANRL_EVAL_PIPELINE=0``) move for
move, with every row, counter and search statistic, through the engine's cache, with and
without reuse, transpositions and sampled worlds; a strict half overflow is resolved row by
row in the queue as in the evaluator's own call; neither the variable nor ``groups`` is in a
key, and a record played on one loop replays on the other under the same id; a stream's kept
tree stays in its env's pool; an error empties every pool; and ``stop`` cuts ``run_groups``
between rounds as it cuts ``run_pools``. On a GPU: the same games on the three kernels, a
real overflow in the queue, and a batch's one event."""

from __future__ import annotations

import time
import zlib

import cppanatron as cpp
import numpy as np
import pytest
import torch
from catanatron.registry import SpecError
from eval_helpers import agent
from test_agent_cache import GAMES as CACHE_GAMES
from test_agent_cache import MODEL as CACHE_MODEL
from test_agent_cache import SUITE as CACHE_SUITE
from test_agent_fused16 import ROWS as HALF_ROWS
from test_agent_fused16 import Halves, expected
from test_graph_kernels import needs_kernels, shared_model
from test_graph_kernels import rows as kernel_rows
from test_mcts_reuse import CHANCE, GAMES, MODEL, SUITE, decide, position
from test_search_pipeline import MODEL as PIPE_MODEL
from test_search_pipeline import ROWS as PIPE_ROWS
from test_search_pipeline import Searches, strict

from catanrl import canon
from catanrl.agents import native, resolve_agent
from catanrl.checkpoint import save
from catanrl.eval import results
from catanrl.eval.harness import play_schedule
from catanrl.eval.suites import Suite
from catanrl.search import (
    BatchEvaluator,
    DedupEvaluator,
    Group,
    Pipeline,
    eval_pipeline,
    run_groups,
    run_pools,
)
from catanrl.search import pipeline as pipeline_module
from catanrl.search.pipeline import EVAL_PIPELINE_ENV
from catanrl.search.pool import EVAL_CACHE_ENV

BOARD = [g for g in GAMES if g.board == 0]  # a 2p board and a 3p board in every rotation
TWELVE = 12 * cpp.EvalCache.entry_bytes(MODEL.num_actions, MODEL.num_players) / 2**20  # a table of twelve rows


def mcts(monkeypatch, piped: bool | None, groups: int = 1, megabytes: float | None = 4.0, model=MODEL,
         sims: int = 6, **params):
    """An MCTS agent built with ``CATANRL_EVAL_PIPELINE`` at ``piped`` (None: unset, the
    default) and the cache's megabytes (None: unset; 0: the Python dedup)."""
    for name, text in ((EVAL_PIPELINE_ENV, None if piped is None else str(int(piped))),
                       (EVAL_CACHE_ENV, None if megabytes is None else repr(megabytes))):
        if text is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, text)
    return agent("MCTS", model, infer_rows=8, sims=sims, groups=groups, **params)


def play(first, others=("F", "R"), games=BOARD, suite=SUITE, **kwargs):
    """The games' rows and, by its decision seed (one per decision of a seat of a game, whatever
    the order the harness asks in), every move of ``first``, with the numbers of its search to
    the bit where the search is kept (``reuse``)."""
    rows, moves, act = [], {}, first.act

    def recorded(obs, masks, seeds, **memory):
        choices = act(obs, masks, seeds, **memory)
        for i, (seed, choice) in enumerate(zip(seeds, choices)):
            numbers: tuple = ()
            if first.reuse:
                stream = int(memory["streams"][i])
                pool, sid = first._pool_of(stream), first._live[stream][0]
                numbers = (pool.visits(sid).tobytes(), pool.q(sid).tobytes(), pool.value(sid).tobytes())
            moves[int(seed)] = (int(choice), *numbers)
        return choices

    first.act = recorded
    play_schedule([first, *(agent(key) for key in others)], games, suite=suite, on_game=rows.append, **kwargs)
    return {row["game_id"]: row for row in rows}, moves


@pytest.fixture
def loops(monkeypatch):
    """The loops the agents' searches ran in: the groups of each ``run_groups`` and the pools
    of each ``run_pools``."""
    seen: dict[str, list[int]] = {"groups": [], "pools": []}
    grouped, pooled = native.run_groups, native.run_pools

    def groups(members, pipeline, **kwargs):
        seen["groups"].append(len(members))
        return grouped(members, pipeline, **kwargs)

    def pools(members, evaluator, **kwargs):
        seen["pools"].append(len(members))
        return pooled(members, evaluator, **kwargs)

    monkeypatch.setattr(native, "run_groups", groups)
    monkeypatch.setattr(native, "run_pools", pools)
    return seen


# ---------------------------------------------------------------- the loop


def test_the_loop_is_the_environments(monkeypatch):
    monkeypatch.delenv(EVAL_PIPELINE_ENV, raising=False)
    assert eval_pipeline() is None
    for text, piped in (("0", False), ("1", True), (" 1 ", True), ("", None)):
        monkeypatch.setenv(EVAL_PIPELINE_ENV, text)
        assert eval_pipeline() is piped
        assert agent("MCTS", MODEL, sims=4).piped is piped
    for text in ("2", "true", "-1"):
        monkeypatch.setenv(EVAL_PIPELINE_ENV, text)
        with pytest.raises(ValueError, match="CATANRL_EVAL_PIPELINE must be 0 .* or 1"):
            eval_pipeline()
        with pytest.raises(ValueError, match="CATANRL_EVAL_PIPELINE"):
            agent("MCTS", MODEL, sims=4)


def test_the_pipeline_queues_batch_evaluators_own_forwards(monkeypatch):
    """``_queue``: the pipeline on a CUDA device and where the variable asks for it, of the
    evaluator as it is at the decision; the serial loop on a CPU, at ``=0``, and for every
    evaluator whose forwards are not ``BatchEvaluator``'s own."""
    default = mcts(monkeypatch, None)
    assert default.piped is None and default._queue() is None and default._pipeline is None  # a CPU
    assert mcts(monkeypatch, False)._queue() is None

    first = mcts(monkeypatch, True)
    pipeline = first._queue()
    assert isinstance(pipeline, Pipeline) and pipeline.queued and pipeline.resolve
    assert pipeline.evaluator is first.evaluator and pipeline.batches is first._batches
    assert first._queue() is pipeline  # one per evaluator
    assert first.pools == [first.pool] and first.pool.cache is first.cache.table

    # The Python dedup's wrapper, and one without dedup at all: the wrapper is called, the other queued.
    python = mcts(monkeypatch, True, megabytes=0.0)
    assert isinstance(python.evaluator, DedupEvaluator) and python._queue() is None
    assert isinstance(mcts(monkeypatch, True, megabytes=0.0, dedup=False)._queue(), Pipeline)

    # Callers swap the evaluator (the analyst's ONNX session, the tests' failing wrappers).
    class Wrapper:
        def __init__(self, inner):
            self.inner = inner
            self.obs_size, self.num_actions, self.num_players = inner.obs_size, inner.num_actions, inner.num_players

        def __call__(self, rows):
            return self.inner(rows)

    class Scaled(BatchEvaluator):
        def __call__(self, observations):
            logits, values = super().__call__(observations)
            return logits * 2, values

    healthy = first.evaluator
    first.evaluator = Wrapper(healthy)
    assert first._queue() is None
    first.evaluator = Scaled(MODEL, device="cpu", infer_rows=8)
    assert first._queue() is None  # a call of its own is not queued
    first.evaluator = other = BatchEvaluator(MODEL, device="cpu", infer_rows=8)
    swapped = first._queue()
    assert swapped is not pipeline and swapped.evaluator is other and swapped.batches is first._batches
    first.evaluator = healthy
    assert first._queue().evaluator is healthy


CASES = {
    "oracle": {"oracle": True},
    "dets": {"dets": 1},
    "pending": {"sims": 12, "dets": 1, "pending": 4},
    "transpositions": {"sims": 16, "dets": 1, "transpositions": True, "pending": 4},
    "reuse": {"dets": 1, "reuse": True},
    "reuse-tt": {"sims": 16, "dets": 1, "reuse": True, "transpositions": True, "pending": 4},
    "reuse-tt-oracle": {"sims": 16, "oracle": True, "reuse": True, "transpositions": True},
    "reuse-gumbel": {"sims": 16, "dets": 2, "reuse": True, "root": "gumbel"},
}
#: Counters that do not depend on num_envs (dropped_first does: test_reuse_is_deterministic).
COUNTERS = ("decisions", "fresh", "reused", "refused_diverged", "refused_engine", "cut")


@pytest.mark.parametrize("case", sorted(CASES))
def test_games_are_the_serial_loops_through_the_pipeline_and_in_groups(case, monkeypatch, loops):
    params = CASES[case]
    serial = mcts(monkeypatch, False, **params)
    reference = play(serial, num_envs=16)
    assert len(reference[0]) == len(BOARD) and len(reference[1]) == serial.counters["decisions"] > 0
    assert not loops["groups"] and set(loops["pools"]) == {1}

    # One group: the serial loop's rounds, so the same forwards and the same counts of the cache.
    piped = mcts(monkeypatch, True, **params)
    loops["pools"].clear()
    assert play(piped, num_envs=16) == reference
    assert not loops["pools"] and set(loops["groups"]) == {1}
    assert piped.counters == serial.counters and piped.search_stats == serial.search_stats
    assert (piped.evaluator.rows, piped.evaluator.forwards) == (serial.evaluator.rows, serial.evaluator.forwards)
    assert piped.cache_stats == serial.cache_stats and piped.cache_stats["forwarded"] == piped.evaluator.rows > 0

    # Groups, on the one cache: through the pipeline, and in lockstep where it is not asked for (a CPU).
    for groups, num_envs, queued in ((2, 5, True), (3, 16, True), (2, 5, None)):
        first = mcts(monkeypatch, queued, groups=groups, **params)
        assert len(first.pools) == groups and first.pool is first.pools[0]
        assert all(pool.cache is first.cache.table for pool in first.pools)
        loops["groups"].clear()
        loops["pools"].clear()
        assert play(first, num_envs=num_envs) == reference
        ran, idle = (loops["groups"], loops["pools"]) if queued else (loops["pools"], loops["groups"])
        assert max(ran) == groups and not idle  # the searches did run as groups
        for name in COUNTERS:
            assert first.counters[name] == serial.counters[name], name
        assert first.search_stats == serial.search_stats
        stats = first.cache_stats
        assert stats["leaves"] == serial.cache_stats["leaves"] and stats["forwarded"] == first.evaluator.rows
        assert sum(len(pool) for pool in first.pools) == len(first._live)
        assert all(pool.pending == 0 for pool in first.pools)
    assert bool(serial._live) == bool(params.get("reuse"))


@pytest.mark.parametrize("megabytes", [TWELVE, 0.0, None], ids=["twelve-rows", "python", "no-dedup"])
def test_groups_play_the_same_games_whatever_skips_the_repeated_rows(megabytes, monkeypatch, loops):
    """A table of twelve rows, which the groups' pools replace all the time under one another;
    the Python dedup, whose wrapper is not queued (the pools run in lockstep through it); and
    no dedup at all."""
    params = {"sims": 12, "dets": 2, "reuse": True, "pending": 4, "dedup": megabytes is not None}
    serial = mcts(monkeypatch, False, megabytes=megabytes, **params)
    reference = play(serial, num_envs=16)
    first = mcts(monkeypatch, True, groups=2, megabytes=megabytes, **params)
    loops["pools"].clear()
    assert play(first, num_envs=5) == reference
    assert first.search_stats == serial.search_stats
    if megabytes == 0:
        assert isinstance(first.evaluator, DedupEvaluator) and max(loops["pools"]) == 2 and not loops["groups"]
        assert first.evaluator.counters["rows"] == serial.evaluator.counters["rows"]
    else:
        assert max(loops["groups"]) == 2 and not loops["pools"]
    if megabytes == TWELVE:
        assert first.cache.table.capacity == 12 and first.cache_stats["replaced"] > 0
        assert first.cache_stats["leaves"] == serial.cache_stats["leaves"]
    if megabytes is None:
        assert first.cache is None and first.evaluator.rows == serial.evaluator.rows


# ---------------------------------------------------------------- a strict half overflow


class Chunks(Halves):
    """``Halves`` whose own call forwards ``infer_rows`` rows at a time in strict mode, as on a
    GPU (``chunk_rows``, which the pipeline's chunks are too)."""

    chunk_rows = property(lambda self: self.infer_rows)


def test_the_pipeline_resolves_a_strict_half_overflow_as_the_evaluators_call():
    """``Pipeline(resolve=True)``, the agents': every row's outputs are the row's own (the
    float32 ones where it overflows, the half ones elsewhere) wherever it sits and whatever is
    queued with it, on the forwards the evaluator's own call takes."""
    for bad in ((), [3], [0], [7, 8], [2, 3, 9], list(range(12))):
        want = expected(bad)
        called, queued = Chunks(bad=bad), Chunks(bad=bad)
        own = called(HALF_ROWS)
        assert np.array_equal(own[0], want[0]) and np.array_equal(own[1], want[1])
        pipeline = Pipeline(queued, resolve=True)
        assert pipeline.queued
        batch = pipeline.batch(0, 12)
        batch.observations[:12] = HALF_ROWS
        pipeline.submit(batch, 12)
        assert queued.sizes == [8, 8]  # the chunks of 8 and of 4 rows, each in the forward's size
        logits, values = pipeline.result(batch)
        assert np.array_equal(logits, want[0]) and np.array_equal(values, want[1])
        for name in ("rows", "forwards", "exact_forwards", "exact_rows"):
            assert getattr(queued, name) == getattr(called, name), name
        assert queued.sizes == called.sizes and queued.exact_rows == len(bad)

    # Two batches queued, rows in any order: the second's forwards are behind the first's resolve.
    rng = np.random.default_rng(0)
    want = expected([2, 3, 9])
    pipeline = Pipeline(Chunks(bad=[2, 3, 9]), resolve=True)
    a, b = pipeline.batch(0, 12), pipeline.batch(1, 5)
    for _ in range(4):
        first, second = rng.permutation(12), rng.choice(12, size=5, replace=False)
        a.observations[:12], b.observations[:5] = HALF_ROWS[first], HALF_ROWS[second]
        pipeline.submit(a, 12)
        pipeline.submit(b, 5)
        for batch, index in ((a, first), (b, second)):
            logits, values = pipeline.result(batch)
            assert np.array_equal(logits, want[0][index]) and np.array_equal(values, want[1][index])
    assert pipeline.evaluator.exact_rows >= pipeline.evaluator.exact_forwards > 0


def test_without_resolve_and_in_fast_mode_a_flagged_chunk_is_evaluated_whole():
    """Self-play's pipeline (no ``resolve``) and a fast evaluator's take the float32 outputs for
    the whole flagged chunk, as they did; a flag left after the overflow raises either way."""
    plain = BatchEvaluator(Halves().model, device="cpu", infer_rows=8, mode="strict")(HALF_ROWS)
    for evaluator, resolve in ((Chunks(bad=[3]), False), (Chunks(bad=[3], mode="fast"), True)):
        pipeline = Pipeline(evaluator, resolve=resolve)
        batch = pipeline.batch(0, 12)
        batch.observations[:12] = HALF_ROWS
        pipeline.submit(batch, 12)
        logits, values = pipeline.result(batch)
        assert np.array_equal(logits[:8], plain[0][:8]) and np.array_equal(logits[8:], plain[0][8:] + np.float32(1))
        assert np.array_equal(values, plain[1])
        assert (evaluator.exact_forwards, evaluator.exact_rows, evaluator.forwards) == (1, 8, 2)
    for bad in ((), (3,)):
        pipeline = Pipeline(Chunks(bad=bad, left=1), resolve=True)
        batch = pipeline.batch(0, 12)
        batch.observations[:12] = HALF_ROWS
        pipeline.submit(batch, 12)
        with pytest.raises(RuntimeError, match="hex boards only"):
            pipeline.result(batch)


class Halving(BatchEvaluator):
    """``kernels=fused16`` stood in on the CPU for an agent's searches. A forward has
    ``infer_rows`` rows in strict mode, as on a GPU; a row's "half" outputs are the network's
    own for that row alone with the logits halved (another search than the float32 ones'); the
    forward's flag has bit 2 when any row of it (padding included) overflows, which about one
    row in ``every`` does, by its bytes. ``_exact`` answers the network's own."""

    def __init__(self, model, every: int = 4) -> None:
        super().__init__(model, device="cpu", infer_rows=8, mode="strict")
        self.plain = BatchEvaluator(model, device="cpu", infer_rows=8, mode="strict")  # a row per forward
        self.kernels, self.fused, self.every = "fused16", object(), every

    chunk_rows = property(lambda self: self.infer_rows)

    def bucket(self, rows: int) -> int:
        return self.infer_rows

    def overflows(self, row: np.ndarray) -> bool:
        return zlib.crc32(row.tobytes()) % self.every == 0

    def _forward(self, obs, state=None):
        rows = obs.numpy()
        logits, values = self.plain(rows)
        flag = 2 * any(self.overflows(row) for row in rows)
        return (torch.from_numpy(logits * np.float32(0.5)), torch.from_numpy(values),
                torch.full((1,), flag, dtype=torch.int32))

    def _exact(self, chunk, size):
        logits, values = self.plain(chunk)
        return logits, values, np.zeros(1, dtype=np.int32)


class Alone:
    """What ``Halving`` owes every row whatever its forward: the row's float32 outputs where it
    overflows, its half ones elsewhere. A wrapper: called, never queued."""

    def __init__(self, halving: Halving) -> None:
        self.halving = halving
        self.obs_size, self.num_actions, self.num_players = halving.obs_size, halving.num_actions, halving.num_players

    def __call__(self, rows):
        logits, values = self.halving.plain(rows)
        for i, row in enumerate(rows):
            if not self.halving.overflows(row):
                logits[i] *= np.float32(0.5)
        return logits, values


def test_an_agents_strict_half_overflow_plays_the_serial_loops_games(monkeypatch, loops):
    """The agent's searches on an evaluator whose forwards overflow all the time: through the
    pipeline, alone and in groups, the games of the serial loop on the same forwards, which are
    the games of an evaluator that answers every row by itself."""
    params = {"sims": 12, "dets": 1, "reuse": True, "pending": 4}

    def build(piped, groups=1) -> tuple:
        first = mcts(monkeypatch, piped, groups=groups, **params)
        first.evaluator = Halving(MODEL)
        return first, first.evaluator

    serial, called = build(False)
    reference = play(serial, num_envs=16)
    assert called.exact_rows >= called.exact_forwards > 0 and not loops["groups"]
    assert called.forwards < called.rows + 8 * called.exact_rows  # forwards of several rows
    piped, queued = build(True)
    assert play(piped, num_envs=16) == reference and set(loops["groups"]) == {1}
    for name in ("rows", "forwards", "exact_forwards", "exact_rows"):
        assert getattr(queued, name) == getattr(called, name), name
    assert piped.cache_stats == serial.cache_stats and piped.search_stats == serial.search_stats
    two, _ = build(True, groups=2)
    assert play(two, num_envs=5) == reference and max(loops["groups"]) == 2
    alone = mcts(monkeypatch, True, **params)
    alone.evaluator = Alone(called)
    assert play(alone, num_envs=16) == reference
    # The chunk taken whole (self-play's way) would be another evaluator: other games.
    whole, _ = build(True)
    whole._queue().resolve = False
    assert play(whole, num_envs=16) != reference


# ---------------------------------------------------------------- keys and records


def test_neither_the_loop_nor_groups_is_in_a_key_and_a_record_replays_on_the_other_loop(tmp_path, monkeypatch, loops):
    """Like ``threads`` and the cache's size: the agent's canonical spec and key and an eval's
    key and id are the same whatever the variable and ``groups`` say. A record played in the
    serial loop is found by a spec with groups (nothing is played again), replays through the
    pipeline game for game, and is the record the groups play."""
    monkeypatch.setenv("CATANRL_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv(EVAL_CACHE_ENV, "4")
    path = save(tmp_path / "e000000.pt", CACHE_MODEL, run=None, epoch=0, agent_steps=0,
                config={"model": CACHE_MODEL.spec}, code={}).path
    spec = f"MCTS:ckpt={path}:sims=8:dets=2:reuse=true:transpositions=true"
    suite = Suite(name="t", purpose="dev", seed=3, boards=2, scenarios=({"game": "base-2p"},), turn_limit=140)

    def keys(spec: str = spec) -> tuple:
        resolved = resolve_agent(spec)
        key = results.eval_key([resolved, resolve_agent("F")], suite, harness="native")
        return resolved.canonical, resolved.key_dict(), key, canon.digest(key)

    monkeypatch.delenv(EVAL_PIPELINE_ENV, raising=False)
    unset = keys()
    assert "groups" not in unset[0]
    assert not {"groups", "pipeline", "eval_pipeline"} & (set(unset[1]["params"]) | set(unset[1]["search"]))
    for text in ("0", "1"):
        monkeypatch.setenv(EVAL_PIPELINE_ENV, text)
        assert keys() == unset
        assert keys(spec + ":groups=2") == unset and keys(spec + ":groups=1") == unset
    assert resolve_agent(spec + ":groups=2").params.groups == 2 and resolve_agent(spec).params.groups == 1
    for bad in ("0", "-1"):
        with pytest.raises(SpecError, match="groups counts search groups"):
            resolve_agent(f"{spec}:groups={bad}")
    with pytest.raises(ValueError, match="groups counts search groups"):
        agent("MCTS", MODEL, sims=4, groups=0)

    def rows(record) -> dict:
        return {row["game_id"]: results._strip(row) for row in record.games()}

    monkeypatch.setenv(EVAL_PIPELINE_ENV, "0")
    record = results.evaluate([spec, "F"], suite, device="cpu", num_envs=3)
    assert record.eval_id == unset[3] and record.data["specs"][0] == unset[0] and record.summary["games"] == 4
    assert set(loops["pools"]) == {1} and not loops["groups"]
    monkeypatch.setenv(EVAL_PIPELINE_ENV, "1")
    loops["pools"].clear()
    assert results.verify(record, games=None, device="cpu", num_envs=2) == {"identical": 4, "of": 4, "differing": []}
    assert set(loops["groups"]) == {1} and not loops["pools"]

    # The groups' own record, in runs of its own: the same id, the same games.
    monkeypatch.setenv("CATANRL_RUNS_DIR", str(tmp_path / "other"))
    loops["groups"].clear()
    grouped = results.evaluate([spec + ":groups=2", "F"], suite, device="cpu", num_envs=4)
    assert max(loops["groups"]) == 2 and not loops["pools"]
    assert grouped.eval_id == record.eval_id and grouped.path != record.path
    assert grouped.data["specs"] == record.data["specs"] and grouped.key == record.key
    assert rows(grouped) == rows(record) and len(rows(record)) == 4

    monkeypatch.setenv("CATANRL_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(results, "play_schedule", lambda *args, **kwargs: pytest.fail("the record was not found"))
    assert results.evaluate([spec + ":groups=2", "F"], suite, device="cpu").eval_id == record.eval_id
    assert results.find([spec + ":groups=2", "F"], suite).eval_id == record.eval_id


# ---------------------------------------------------------------- the groups' pools


def test_a_streams_searches_are_its_envs_pools(monkeypatch):
    """Streams are ``env * max_players + seat``: env 0 and env 2 search in pool 0, env 1 and
    env 3 in pool 1, at every decision, so a kept tree is re-rooted, dropped and forgotten
    where it lives; the searches are those of one group."""
    one = mcts(monkeypatch, True, reuse=True, oracle=True)
    two = mcts(monkeypatch, True, groups=2, reuse=True, oracle=True)
    assert two.dims["max_players"] == 3
    games, streams = [position(1), position(2), position(3), position(4)], [0, 5, 9, 6]
    choices = [decide(a, games, streams=streams, first=[True] * 4) for a in (one, two)]
    assert np.array_equal(*choices)
    assert [len(pool) for pool in two.pools] == [2, 2] and len(one.pool) == 4
    for stream in streams:
        pool = two.pools[stream // 3 % 2]
        assert pool is two._pool_of(stream) and two._live[stream][0] in pool.ids
        assert pool.visits(two._live[stream][0]).tolist() == one.pool.visits(one._live[stream][0]).tolist()
    # Each stream's next decision, where its move kept the turn and drew nothing: re-rooted
    # in its own pool, with the visits one group's re-rooted search has.
    following, kept = [], []
    for game, stream, chosen in zip(games, streams, choices[1]):
        action = two.action_space.decode(game, int(chosen))
        after = game.copy()
        after.execute(action)
        if action.action_type not in CHANCE and after.current_player_index == game.current_player_index \
                and len(after.playable_actions) >= 2:
            following.append(after)
            kept.append(stream)
    assert {stream // 3 % 2 for stream in kept} == {0, 1}, "the positions no longer re-root in both pools"
    for a in (one, two):
        decide(a, following, streams=kept, first=[False] * len(kept))
        assert a.counters["reused"] == len(kept) and a.counters["fresh"] == 4
    assert [len(pool) for pool in two.pools] == [2, 2]
    for stream in kept:
        assert two._pool_of(stream).visits(two._live[stream][0]).tolist() == \
            one.pool.visits(one._live[stream][0]).tolist()
    # Another seat of env 1 starts a game: env 1's search (stream 5, pool 1) is dropped.
    decide(two, games[:1], streams=[4], first=[True])
    assert two.counters["dropped_first"] == 1 and set(two._live) == {0, 4, 6, 9}
    assert [len(pool) for pool in two.pools] == [2, 2]
    two.forget(3)  # stream 9, pool 1
    two.forget(0)  # stream 0, pool 0
    assert two.counters["dropped_over"] == 2 and set(two._live) == {4, 6}
    assert [len(pool) for pool in two.pools] == [1, 1]


def searched(first, games) -> np.ndarray:
    """One ``act`` on ``games`` without the harness's memory (no reuse)."""
    rows = np.stack([first.encode(g, g.current_player_index) for g in games])
    trackers = [cpp.HandTracker(g, g.current_player_index) for g in games]
    return first.act(rows, rows[:, -first.action_space.size:], np.arange(100, 100 + len(games), dtype=np.uint64),
                     games=games, trackers=trackers)


def test_without_reuse_a_batchs_rows_alternate_and_the_pools_end_empty(monkeypatch):
    first, one = mcts(monkeypatch, True, groups=2, dets=2), mcts(monkeypatch, True, dets=2)
    games = [position(seed) for seed in (1, 2, 3)]
    seen, run = [], native.run_groups

    def counted(groups, pipeline, **kwargs):
        seen.append([(len(group.pools[0]), list(group.batch_rows)) for group in groups])
        return run(groups, pipeline, **kwargs)

    monkeypatch.setattr(native, "run_groups", counted)
    assert np.array_equal(searched(first, games), searched(one, games))
    assert seen == [[(2, [2]), (1, [1])], [(3, [3])]]  # rows 0 and 2, row 1; a leaf per search and round
    # One row: one group, the other pool is left out.
    assert np.array_equal(searched(first, games[:1]), searched(one, games[:1]))
    assert seen[2] == [(1, [1])]
    assert all(len(pool) == 0 for pool in first.pools) and first.search_stats == one.search_stats


def test_an_error_empties_every_pool_and_the_streams_restart(monkeypatch):
    first = mcts(monkeypatch, True, groups=2, reuse=True, oracle=True, dedup=False)  # every leaf is a forward
    games, streams = [position(5), position(6), position(7)], [0, 3, 7]  # envs 0, 1 and 2
    decide(first, games, streams=streams, first=[True] * 3)
    assert [len(pool) for pool in first.pools] == [2, 1]
    forward, calls = first.evaluator._forward, []

    def failing(*args):
        calls.append(1)
        if len(calls) == 4:  # the first group's second round, with the second group's leaves queued
            raise MemoryError("evaluator failed")
        return forward(*args)

    monkeypatch.setattr(first.evaluator, "_forward", failing)
    with pytest.raises(MemoryError, match="evaluator failed"):
        decide(first, games, streams=streams, first=[True] * 3)
    assert all(len(pool) == 0 and pool.pending == 0 for pool in first.pools) and not first._live
    assert len(first._batches) == 2 and all(batch.rows == 0 for batch in first._batches.values())
    monkeypatch.setattr(first.evaluator, "_forward", forward)
    with pytest.raises(ValueError, match="no live search"):
        decide(first, games, streams=streams, first=[False] * 3)
    assert len(decide(first, games, streams=streams, first=[True] * 3)) == 3
    assert [len(pool) for pool in first.pools] == [2, 1]


def test_a_bind_keeps_the_groups(monkeypatch):
    first = mcts(monkeypatch, True, groups=2, reuse=True, oracle=True)
    pools, batches, pipeline = first.pools, first._batches, first._queue()
    first.bind({**first.dims, "max_players": 4})
    assert first.dims["max_players"] == 4 and len(first.pools) == 2 and first.pools[0] is not pools[0]
    assert first.pool is first.pools[0] and all(pool.max_players == 4 for pool in first.pools)
    assert all(pool.cache is first.cache.table for pool in first.pools)
    bound = first._queue()
    assert bound is not pipeline and bound.evaluator is first.evaluator and bound.batches is batches


# ---------------------------------------------------------------- stop


def numbers(searches) -> list[tuple]:
    return [(pool.visits(sid).tobytes(), pool.q(sid).tobytes(), pool.stats(sid).simulations, pool.done(sid))
            for pool, sid in searches]


def test_stop_ends_run_groups_between_rounds_as_run_pools():
    """A ``stop`` already passed: every group runs one round (the first is never skipped),
    nothing is left pending, the searches hold what ``run_pools`` leaves them with the same
    ``stop``, and running on finishes them as if never stopped."""
    evaluator, past = strict(), time.monotonic() - 1.0
    budget = [i % 3 == 0 for i in range(12)]
    serial, grouped, whole = (Searches(n, group, budget) for n, group in
                              ((1, [0] * 12), (3, [2 * (i % 2) for i in range(12)]), (1, [0] * 12)))
    reference = serial.add(0)
    leaves = run_pools(serial.pools[0], evaluator, batch_rows=serial.batch_rows[0], stop=past)
    searches = grouped.add(0)
    members = [Group(pools, rows) for pools, rows in zip(grouped.pools, grouped.batch_rows)]
    pipeline = Pipeline(evaluator)
    assert run_groups(members, pipeline, stop=past) == (leaves, 1)
    assert [g.rounds for g in members] == [1, 0, 1]  # the middle group has no searches
    assert all(pool.pending == 0 for pools in grouped.pools for pool in pools)
    assert all(batch.rows == 0 for batch in pipeline.batches.values())
    assert numbers(searches) == numbers(reference)
    # One round: a fresh search's is its root's evaluation.
    assert leaves == len(searches) and not any(sims or done for *_, sims, done in numbers(searches))
    # Not yet passed: the searches run to the end, the ones cut short as the ones never stopped.
    run_groups(members, pipeline, stop=time.monotonic() + 3600.0)
    uncut = whole.add(0)
    run_pools(whole.pools[0], evaluator, batch_rows=whole.batch_rows[0])
    assert numbers(searches) == numbers(uncut) and all(done for *_, done in numbers(searches))
    assert grouped.read(searches) == whole.read(uncut)


def test_stop_passing_mid_run_applies_the_queued_rounds_and_collects_no_more(monkeypatch):
    """The clock passes ``stop`` at the third look (after group 0's second round): the round
    group 1 has queued is applied, and neither collects again."""

    class Clock:
        looks = 0

        @classmethod
        def monotonic(cls) -> float:
            cls.looks += 1
            return 0.0 if cls.looks < 3 else 10.0

    s = Searches(2, [i % 2 for i in range(12)], [False] * 12)
    searches = s.add(0)
    members = [Group(pools, rows) for pools, rows in zip(s.pools, s.batch_rows)]
    monkeypatch.setattr(pipeline_module, "time", Clock)
    leaves, rounds = run_groups(members, Pipeline(strict()), stop=5.0)
    assert Clock.looks == 4 and rounds == 2 and [g.rounds for g in members] == [2, 2]
    assert leaves == sum(g.leaves for g in members) > 0
    assert all(pool.pending == 0 for pools in s.pools for pool in pools)
    assert not any(pool.done(sid) for pool, sid in searches)


@pytest.mark.parametrize("groups", [1, 2])
def test_a_cut_search_plays_its_most_visited_move_through_the_pipeline_too(groups, monkeypatch):
    """``stop_at`` (the bridges' budget; ``deadline`` goes the same way) already passed: one
    round per search, counted as cut, the moves those of the serial loop cut the same way."""
    params = {"sims": 64, "pending": 4, "dets": 1, "reuse": True}
    one, two = mcts(monkeypatch, False, **params), mcts(monkeypatch, True, groups=groups, **params)
    games, streams = [position(1), position(2), position(3), position(4)], [0, 3, 7, 10]
    choices = []
    for a in (one, two):
        a.stop_at = time.monotonic() - 1.0
        choices.append(decide(a, games, streams, [True] * 4))
    assert np.array_equal(*choices)
    assert one.counters["cut"] == two.counters["cut"] == 4 and two.search_stats == one.search_stats
    # One round each: the roots' evaluations, and the prior's best move of every search.
    assert two.search_stats["evaluations"] == 4 and two.search_stats["simulations"] == 0
    assert all(pool.pending == 0 for pool in two.pools)
    assert [len(pool) for pool in two.pools] == ([4] if groups == 1 else [2, 2])
    # Without the cutoff the same agents search their budget.
    for a in (one, two):
        a.stop_at = None
        decide(a, games, streams, [True] * 4)
    assert one.counters["cut"] == two.counters["cut"] == 4
    assert two.search_stats["simulations"] == one.search_stats["simulations"] >= 4 * 64


# ---------------------------------------------------------------- CUDA


@pytest.mark.gpu
@needs_kernels
@pytest.mark.parametrize("kernels", ["fused16", "fused", "torch"])
def test_cuda_games_are_the_serial_loops(kernels, monkeypatch, loops):
    """On the device, where the pipeline is the agents' loop unless the variable says 0 and a
    forward has ``infer_rows`` rows: the serial loop's games, move for move with every kept
    search's numbers, through the pipeline and in two and three groups, on each kernels; one
    group forwards the serial loop's rows in its forwards."""
    model = shared_model(["base-2p"])

    def build(piped, groups=1):
        for name, text in ((EVAL_PIPELINE_ENV, None if piped is None else str(int(piped))), (EVAL_CACHE_ENV, None)):
            if text is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, text)
        return agent("MCTS", model, infer_rows=32, device="cuda", sims=24, dets=1, pending=8, reuse=True,
                     transpositions=True, kernels=kernels, groups=groups)

    def played(first, num_envs):
        return play(first, others=("F",), games=CACHE_GAMES, suite=CACHE_SUITE, num_envs=num_envs, device="cuda")

    serial = build(False)
    reference = played(serial, 4)
    assert len(reference[0]) == len(CACHE_GAMES) and set(loops["pools"]) == {1} and not loops["groups"]
    for groups, num_envs in ((1, 4), (2, 4), (2, 3), (3, 4)):
        first = build(None, groups)
        assert first.piped is None and isinstance(first._queue(), Pipeline) and first._queue().resolve
        loops["groups"].clear()
        loops["pools"].clear()
        assert played(first, num_envs) == reference
        assert max(loops["groups"]) == groups and not loops["pools"]
        assert all(batch._rows.is_pinned() and batch._marker is not None for batch in first._batches.values())
        for name in COUNTERS:
            assert first.counters[name] == serial.counters[name], name
        assert first.search_stats == serial.search_stats
        assert first.cache_stats["leaves"] == serial.cache_stats["leaves"]
        assert first.cache_stats["forwarded"] == first.evaluator.rows
        assert (first.evaluator.exact_forwards, first.evaluator.exact_rows) == (0, 0)
        if groups == 1:
            assert first.cache_stats == serial.cache_stats and first.counters == serial.counters
            assert (first.evaluator.rows, first.evaluator.forwards) == (serial.evaluator.rows, serial.evaluator.forwards)


@pytest.mark.gpu
@needs_kernels
def test_cuda_a_strict_half_overflow_in_the_queue_is_a_rows_own():
    """The kernels themselves, with weights that overflow a half on some rows and not on others:
    two batches queued behind one another on ``Pipeline(resolve=True)`` return, row for row,
    what the strict evaluator's own call returns, on as many forwards."""
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
    order = np.random.default_rng(1).permutation(len(obs))
    called = BatchEvaluator(scale, device="cuda", infer_rows=8, mode="strict", kernels="fused16")
    together, shuffled = called(obs), called(obs[order])
    assert called.exact_rows == 2 * int(some.sum()) > 0
    queued = BatchEvaluator(scale, device="cuda", infer_rows=8, mode="strict", kernels="fused16")
    pipeline = Pipeline(queued, resolve=True)
    a, b = pipeline.batch(0, len(obs)), pipeline.batch(1, len(obs))
    a.observations[: len(obs)], b.observations[: len(obs)] = obs, obs[order]
    pipeline.submit(a, len(obs))
    pipeline.submit(b, len(obs))
    for batch, want in ((a, together), (b, shuffled)):
        logits, values = pipeline.result(batch)
        assert np.array_equal(logits, want[0]) and np.array_equal(values, want[1])
    assert np.array_equal(shuffled[0], together[0][order])
    for name in ("rows", "forwards", "exact_forwards", "exact_rows"):
        assert getattr(queued, name) == getattr(called, name), name
    # Self-play's pipeline takes the flagged chunks whole: more float32 rows, the same overflowing ones.
    whole = BatchEvaluator(scale, device="cuda", infer_rows=8, mode="strict", kernels="fused16")
    pipeline = Pipeline(whole)
    batch = pipeline.batch(0, len(obs))
    batch.observations[: len(obs)] = obs
    pipeline.submit(batch, len(obs))
    logits, _ = pipeline.result(batch)
    assert np.array_equal(logits[some], together[0][some]) and whole.exact_rows >= int(some.sum())


@pytest.mark.gpu
def test_cuda_a_batch_keeps_one_event():
    """A batch's event is made once and recorded again by every submit: the results behind it
    are the evaluator's turn after turn, with another batch queued, a long queue in front, and
    after a drain."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    reference = BatchEvaluator(PIPE_MODEL, device="cuda", infer_rows=8, mode="strict")
    pipeline = Pipeline(BatchEvaluator(PIPE_MODEL, device="cuda", infer_rows=8, mode="strict"))
    long = np.concatenate([PIPE_ROWS] * 20)
    a, b = pipeline.batch(0, len(long)), pipeline.batch(1, 16)
    assert a._marker is None and a._event is None
    markers = None
    for turn in range(8):
        first = long if turn % 4 == 3 else PIPE_ROWS[turn : turn + 20]  # every fourth: 100 forwards in front of b's
        second = PIPE_ROWS[20 - turn : 33 - turn]
        a.observations[: len(first)], b.observations[: len(second)] = first, second
        pipeline.submit(a, len(first))
        pipeline.submit(b, len(second))
        assert a._event is a._marker and b._event is b._marker and a._marker is not b._marker
        markers = markers or (a._marker, b._marker)
        assert a._marker is markers[0] and b._marker is markers[1]
        for batch, chunk in ((a, first), (b, second)):
            logits, values = pipeline.result(batch)
            assert batch._event is None and batch._marker.query()
            want = reference(chunk)
            assert np.array_equal(logits, want[0]) and np.array_equal(values, want[1])
    a.observations[: len(long)] = long
    pipeline.submit(a, len(long))
    pipeline.drain()
    assert a.rows == 0 and a._event is None and a._marker is markers[0]
    a.observations[:5] = PIPE_ROWS[:5]
    pipeline.submit(a, 5)
    logits, values = pipeline.result(a)
    want = reference(PIPE_ROWS[:5])
    assert np.array_equal(logits, want[0]) and np.array_equal(values, want[1])
