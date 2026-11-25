"""The MCTS agent's evaluation cache (``MctsAgent.cache``: the engine's, ``CATANRL_EVAL_CACHE_MB``
megabytes, 256 unless set): whole games are the ones of the Python dedup (``=0``) and of no
dedup at all, row for row, on fewer forwarded rows; the variable is in no key and a record
played on one path replays on the other under the same id; a table serves one network (a new
model starts an empty one, a written weight or another evaluator clears it)."""

from __future__ import annotations

import copy

import cppanatron as cpp
import numpy as np
import pytest
import torch
from eval_helpers import agent, make_suite, schedule, tiny_model

from catanrl import canon
from catanrl import games as game_configs
from catanrl.agents import resolve_agent
from catanrl.checkpoint import save
from catanrl.envs.encoding import dims_for
from catanrl.eval import results
from catanrl.eval.harness import play_schedule
from catanrl.eval.suites import Suite
from catanrl.search import BatchEvaluator, DedupEvaluator, NetworkCache, eval_cache_mb
from catanrl.search.pool import EVAL_CACHE_ENV, EVAL_CACHE_MB

SUITE = make_suite(({"game": "base-2p"},), turn_limit=140)
GAMES = schedule(SUITE, 2, boards=2)
MODEL = tiny_model(["base-2p"], level="private", seed=3)
ENTRY = cpp.EvalCache.entry_bytes(MODEL.num_actions, MODEL.num_players)
TWELVE = 12 * ENTRY / 2**20  # megabytes of a table of twelve rows: it replaces all the time


def mcts(monkeypatch, megabytes: float | None, model=MODEL, **params):
    """An MCTS agent built with the variable at ``megabytes`` (None: unset, the default)."""
    if megabytes is None:
        monkeypatch.delenv(EVAL_CACHE_ENV, raising=False)
    else:
        monkeypatch.setenv(EVAL_CACHE_ENV, repr(megabytes))
    return agent("MCTS", model, infer_rows=8, **{"sims": 10, "dets": 2, "reuse": True, **params})


def play(first, **kwargs) -> list[dict]:
    rows = []
    play_schedule([first, agent("F")], GAMES, suite=SUITE, on_game=rows.append, **kwargs)
    return sorted(rows, key=lambda row: row["game_id"])


def positions(count: int = 4) -> list[cpp.Game]:
    out = []
    for seed in range(count):
        game = game_configs.load_game("base-2p").cppanatron_game(seed, seed)
        rng = np.random.default_rng(seed)
        for _ in range(20 + 15 * seed):
            game.execute(game.playable_actions[int(rng.integers(len(game.playable_actions)))])
        while len(game.playable_actions) < 2:
            game.execute(game.playable_actions[0])
        out.append(game)
    return out


def decide(mcts, games) -> tuple:
    """First decisions of ``games``' streams: the choices and each live search's numbers, to the bit."""
    rows = np.stack([mcts.encode(g, g.current_player_index) for g in games])
    trackers = [cpp.HandTracker(g, g.current_player_index) for g in games]
    choices = mcts.act(rows, rows[:, -mcts.action_space.size:], np.arange(len(games), dtype=np.uint64) + 40,
                       games=games, trackers=trackers, streams=np.arange(len(games)),
                       first=np.ones(len(games), dtype=bool))
    searches = [mcts._live[stream][0] for stream in range(len(games))]
    return (choices.tolist(), [mcts.pool.visits(sid).tobytes() for sid in searches],
            [mcts.pool.q(sid).tobytes() for sid in searches], [mcts.pool.value(sid).tobytes() for sid in searches])


def test_the_size_is_the_environments(monkeypatch):
    monkeypatch.delenv(EVAL_CACHE_ENV, raising=False)
    assert eval_cache_mb() == EVAL_CACHE_MB == 256
    for text, megabytes in (("0", 0.0), (" 64 ", 64.0), ("0.5", 0.5), ("", EVAL_CACHE_MB)):
        monkeypatch.setenv(EVAL_CACHE_ENV, text)
        assert eval_cache_mb() == megabytes
    for bad in ("-1", "many", "inf", "nan"):
        monkeypatch.setenv(EVAL_CACHE_ENV, bad)
        with pytest.raises(ValueError, match=EVAL_CACHE_ENV):
            eval_cache_mb()
        with pytest.raises(ValueError, match=EVAL_CACHE_ENV):  # the agent reads it when it is built
            agent("MCTS", MODEL, sims=4)
        assert agent("MCTS", MODEL, sims=4, dedup=False).cache is None  # no dedup: nothing to size


def test_an_agent_skips_repeated_rows_one_way(monkeypatch):
    """The engine's cache by default, the Python dedup at 0, neither with ``dedup=false``."""
    default = mcts(monkeypatch, None)
    assert isinstance(default.cache, NetworkCache) and isinstance(default.evaluator, BatchEvaluator)
    assert default.pool.cache is default.cache.table
    assert default.cache.table.capacity == (int(EVAL_CACHE_MB * 2**20) // ENTRY) // 4 * 4
    assert (default.cache.table.action_size, default.cache.table.max_players) == (MODEL.num_actions, MODEL.num_players)
    assert len(default.cache.table) == 0 and default.cache_stats["capacity"] == default.cache.table.capacity
    small = mcts(monkeypatch, TWELVE)
    assert small.cache.table.capacity == 12 and small.cache is not default.cache
    python = mcts(monkeypatch, 0.0)
    assert python.cache is None and python.pool.cache is None and isinstance(python.evaluator, DedupEvaluator)
    assert python.cache_stats == {}
    for megabytes in (None, 0.0):
        off = mcts(monkeypatch, megabytes, dedup=False)
        assert off.cache is None and off.pool.cache is None and isinstance(off.evaluator, BatchEvaluator)


@pytest.mark.parametrize("params", [{"dets": 2}, {"dets": 0, "oracle": True, "transpositions": True},
                                    {"dets": 1, "reuse": False, "pending": 4}],
                         ids=["worlds", "oracle-transpositions", "fresh-searches"])
def test_games_are_the_same_through_the_cache_the_python_dedup_and_neither(monkeypatch, params):
    """Strict rows: every game's row is the same on the four paths (the table at 256 MB and at
    twelve rows, which replaces all the time), and the cache forwards fewer rows than no dedup."""
    engine, small, python = (mcts(monkeypatch, megabytes, **params) for megabytes in (None, TWELVE, 0.0))
    off = mcts(monkeypatch, None, dedup=False, **params)
    played = [play(first, num_envs=3) for first in (engine, small, python, off)]
    assert played[0] == played[1] == played[2] == played[3] and len(played[0]) == len(GAMES)
    assert engine.counters == small.counters == python.counters == off.counters
    assert engine.search_stats == small.search_stats == python.search_stats == off.search_stats
    stats = engine.cache_stats
    # Every leaf the uncached agent forwarded was searched here too; the rest took no row.
    assert stats["leaves"] == off.evaluator.rows == python.evaluator.counters["rows"] == small.cache_stats["leaves"]
    assert stats["forwarded"] == engine.evaluator.rows < off.evaluator.rows
    assert stats["hits"] > 0 and stats["hits"] + stats["repeats"] + stats["forwarded"] == stats["leaves"]
    assert stats["replaced"] < stats["stored"] // 20 and stats["clears"] == 1  # the first decision's sync
    assert small.cache_stats["replaced"] > 0 and small.cache_stats["entries"] == 12
    assert engine.evaluator.rows <= small.evaluator.rows < off.evaluator.rows


def test_the_cache_is_in_no_key_and_a_record_replays_through_it(tmp_path, monkeypatch):
    """Like ``threads``: the agent's canonical spec and key and an eval's key and id are the
    same whatever the variable says. A record played with the Python dedup is found under the
    engine cache (nothing is played again) and replays through it game for game."""
    monkeypatch.setenv("CATANRL_RUNS_DIR", str(tmp_path / "runs"))
    path = save(tmp_path / "e000000.pt", MODEL, run=None, epoch=0, agent_steps=0, config={"model": MODEL.spec},
                code={}).path
    spec = f"MCTS:ckpt={path}:sims=8:dets=2:reuse=true:transpositions=true"
    suite = Suite(name="t", purpose="dev", seed=3, boards=2, scenarios=({"game": "base-2p"},), turn_limit=140)

    def keys() -> tuple:
        resolved = resolve_agent(spec)
        key = results.eval_key([resolved, resolve_agent("F")], suite, harness="native")
        return resolved.canonical, resolved.key_dict(), key, canon.digest(key)

    monkeypatch.delenv(EVAL_CACHE_ENV, raising=False)
    unset = keys()
    assert not {"dedup", "cache", "eval_cache_mb"} & (set(unset[1]["params"]) | set(unset[1]["search"]))
    for megabytes in ("0", "1", "64"):
        monkeypatch.setenv(EVAL_CACHE_ENV, megabytes)
        assert keys() == unset

    monkeypatch.setenv(EVAL_CACHE_ENV, "0")
    record = results.evaluate([spec, "F"], suite, device="cpu", num_envs=3)
    assert record.eval_id == unset[3] and record.data["specs"][0] == unset[0] and record.summary["games"] == 4
    monkeypatch.delenv(EVAL_CACHE_ENV)
    synced = []
    sync = NetworkCache.sync
    monkeypatch.setattr(NetworkCache, "sync", lambda self, evaluator: synced.append(self) or sync(self, evaluator))
    replayed = results.verify(record, games=None, device="cpu", num_envs=2)
    assert replayed == {"identical": 4, "of": 4, "differing": []}
    assert synced and len({id(cache) for cache in synced}) == 1  # the replay's searches went through one table
    monkeypatch.setattr(results, "play_schedule", lambda *args, **kwargs: pytest.fail("the record was not found"))
    assert results.evaluate([spec, "F"], suite, device="cpu").eval_id == record.eval_id
    assert results.find([spec, "F"], suite).eval_id == record.eval_id


def test_a_table_serves_one_network(monkeypatch):
    """A model set on the agent starts an empty table of its own (``_set_model``: the constructor
    and ``bind``), so its decisions are a new agent's; a weight written in place, which the
    agent is not told about, clears the table at the next decision."""
    games = positions()
    first = mcts(monkeypatch, None)
    table = first.cache
    decide(first, games)
    assert first.cache_stats["entries"] > 0
    other = tiny_model(["base-2p"], level="private", seed=8)
    first._set_model(other)
    assert first.cache is not table and first.pool.cache is first.cache.table is not table.table
    assert len(first.cache.table) == 0 and first.cache_stats["leaves"] == 0
    assert decide(first, games) == decide(mcts(monkeypatch, None, model=other, dedup=False), games)
    assert decide(first, games) != decide(mcts(monkeypatch, None, dedup=False), games)  # MODEL's decisions

    wide = dims_for(["base-2p", "base-4p"])
    table = first.cache
    first.bind(wide)
    assert first.cache is not table and first.pool.cache is first.cache.table
    assert first.cache.table.max_players == first.evaluator.num_players == 4 != table.table.max_players
    first.bind(wide)  # the dims it has: nothing is rebuilt
    assert first.pool.cache is first.cache.table and first.cache_stats["clears"] == 0

    model = copy.deepcopy(MODEL)
    written = mcts(monkeypatch, None, model=model)
    before = decide(written, games)
    assert decide(written, games) == before and written.cache_stats["clears"] == 1  # the same weights: kept
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(1.5)
    after = decide(written, games)
    assert written.cache_stats["clears"] == 2
    assert after == decide(mcts(monkeypatch, None, model=copy.deepcopy(model), dedup=False), games) != before


def test_another_evaluator_clears_the_table(monkeypatch):
    """The table holds one evaluator's outputs: an agent handed another (a wrapper that shows
    no network, as the tests' failing evaluators) clears before every decision's searches."""
    class Wrapper:
        def __init__(self, inner):
            self.inner = inner
            self.obs_size, self.num_actions, self.num_players = inner.obs_size, inner.num_actions, inner.num_players

        def __call__(self, rows):
            return self.inner(rows)

    games = positions(2)
    first = mcts(monkeypatch, None)
    healthy, expected = first.evaluator, decide(first, games)
    first.evaluator = Wrapper(healthy)
    assert decide(first, games) == expected and first.cache_stats["clears"] == 2
    assert decide(first, games) == expected and first.cache_stats["clears"] == 3
    first.evaluator = healthy
    assert decide(first, games) == expected and first.cache_stats["clears"] == 4
    assert decide(first, games) == expected and first.cache_stats["clears"] == 4
