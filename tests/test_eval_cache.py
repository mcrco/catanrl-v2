"""The evaluation cache (``cppanatron.EvalCache`` behind ``catanrl.search.pool.NetworkCache``):
searches and self-play with it are the ones without to the bit (strict evaluator), fewer rows
reach the network, and nothing of older weights is ever served."""

from __future__ import annotations

import cppanatron as cpp
import numpy as np
import pytest
import torch
from eval_helpers import tiny_model

from catanrl import games
from catanrl.config import Variant, config_hash, resolve
from catanrl.envs.puffer import plan_env
from catanrl.runs import RunWriter, load_experiment
from catanrl.search import BatchEvaluator, Group, NetworkCache, Pipeline, make_pool, run_groups, run_pools, search_config
from catanrl.train.alphazero import ALPHAZERO_KEYS, ReplayBuffer, SelfPlay, train_alphazero
from catanrl.train.puffer import build_policy_model

MODEL = tiny_model(["base-2p"], level="full", seed=4)
DIMS = MODEL.dims
PARAMS = {"sims": 40, "pending": 4, "oracle": True, "noise": True}
ENTRY = cpp.EvalCache.entry_bytes(MODEL.num_actions, MODEL.num_players)


def positions(count=6, seed=0):
    config = games.load_game("base-2p")
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        game = config.cppanatron_game(map_seed=i, seed=50 + i)
        for _ in range(int(rng.integers(10, 120))):
            actions = game.playable_actions
            if game.winning_color is not None or not actions:
                break
            game.execute(actions[int(rng.integers(len(actions)))])
        if game.winning_color is None:
            out.append(game)
    return out


class Counted:
    """A strict evaluator that counts its rows and calls."""

    def __init__(self, model=MODEL) -> None:
        self.evaluator = BatchEvaluator(model, device="cpu", infer_rows=8, mode="strict")
        self.rows = self.calls = 0

    def __getattr__(self, name):
        return getattr(self.evaluator, name)

    def __call__(self, observations):
        self.rows += len(observations)
        self.calls += 1
        return self.evaluator(observations)


def pool_with(cache: NetworkCache | None, threads=2, **params) -> cpp.SearchPool:
    config = search_config({**PARAMS, **params}, max_players=DIMS["max_players"], threads=threads)
    return make_pool(DIMS, "full", config, cache=cache.table if cache else None)


def numbers(pool, sid) -> tuple:
    """A search's results, to the bit (q is nan where an action has no visits)."""
    stats = pool.stats(sid)
    return (pool.visits(sid).tobytes(), pool.q(sid).tobytes(), pool.policy_target(sid).tobytes(),
            pool.value(sid).tobytes(), pool.root_priors(sid).tobytes(), stats.simulations, stats.nodes)


# ---------------------------------------------------------------- the pools


def test_megabytes_are_entries_and_the_table_belongs_to_one_network():
    cache = NetworkCache(MODEL.num_actions, MODEL.num_players, 1.0)
    assert cache.table.capacity == (2**20 // ENTRY) // 4 * 4 and cache.table.bytes <= 2**20
    assert (cache.table.action_size, cache.table.max_players) == (MODEL.num_actions, MODEL.num_players)
    assert NetworkCache(MODEL.num_actions, MODEL.num_players, 1e-9).table.capacity == 4  # one set at least
    for megabytes in (0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="megabytes"):
            NetworkCache(MODEL.num_actions, MODEL.num_players, megabytes)
    pool = pool_with(cache)
    assert pool.cache is cache.table and pool_with(None).cache is None
    with pytest.raises(ValueError):  # another network's output sizes
        pool.cache = cpp.EvalCache(MODEL.num_actions + 1, MODEL.num_players, entries=16)
    assert pool.cache is cache.table
    pool.cache = None
    assert pool.cache is None


@pytest.mark.parametrize("megabytes", [4.0, 12 * ENTRY / 2**20], ids=["roomy", "twelve-entries"])
def test_searches_with_a_cache_are_the_ones_without(megabytes):
    """Two moves of the same games (the second searches under the first's positions again), two
    searches of one game with one seed (the same leaves in every round) and a table that keeps
    everything or twelve rows: every search's numbers are the uncached pool's, on fewer rows."""
    results, rows = [], []
    for cached in (False, True):
        cache = NetworkCache(MODEL.num_actions, MODEL.num_players, megabytes) if cached else None
        evaluator, pool = Counted(), pool_with(cache)
        if cache is not None:
            cache.sync(evaluator)
        codec, out = pool.config and cpp.GraphEncoder(cpp.EncoderDims(**DIMS), cpp.InfoLevel.FULL).action_space, []
        boards = [game.copy() for game in positions()]
        for move in range(2):
            searches = [pool.add(game, seed=7 * move + i) for i, game in enumerate(boards)]
            searches.append(pool.add(boards[0], seed=7 * move))  # the first search's twin
            run_pools([pool], evaluator, batch_rows=[len(searches) * PARAMS["pending"]])
            assert all(pool.done(sid) for sid in searches) and pool.pending == 0
            out.append([numbers(pool, sid) for sid in searches])
            assert out[-1][0] == out[-1][-1]
            for game, sid in zip(boards, searches):
                game.execute(codec.decode(game, pool.select(sid)))
            for sid in searches:
                pool.remove(sid)
        results.append(out)
        rows.append(evaluator.rows)
        if cache is not None:
            stats = cache.table.stats
            assert stats.leaves == rows[0] and evaluator.rows + stats.hits + stats.repeats == stats.leaves
            assert stats.repeats > 0 and stats.hits > 0 and cache.answered() == stats.hits + stats.repeats
            assert len(cache.table) == stats.entries <= cache.table.capacity
            assert stats.entries + stats.replaced == stats.stored
            if megabytes < 1:
                assert stats.replaced > 0 and stats.entries == 12  # the table is full and replaces
            else:
                assert stats.replaced < stats.stored // 20  # (a set of four can fill before the table)
    assert results[0] == results[1]
    assert rows[1] < rows[0]


@pytest.mark.parametrize("groups", [1, 2])
def test_a_round_that_needs_no_row_is_not_the_end_of_a_search(groups):
    """A search the table knows from end to end: the network sees its root (a root always gets a
    row) and no other row, yet the search runs its whole budget, under run_pools and run_groups."""
    cache = NetworkCache(MODEL.num_actions, MODEL.num_players, 4.0)
    evaluator = Counted()
    cache.sync(evaluator)
    game = positions()[0]
    pools = [pool_with(cache, noise=False) for _ in range(groups)]

    def search(seed):
        sids = [pool.add(game, seed=seed) for pool in pools]
        if groups == 1:
            run_pools(pools, evaluator, batch_rows=[PARAMS["pending"]])
        else:
            run_groups([Group([pool], [PARAMS["pending"]]) for pool in pools], Pipeline(evaluator.evaluator))
        out = [numbers(pool, sid) for pool, sid in zip(pools, sids)]
        assert all(pool.done(sid) and pool.stats(sid).simulations == PARAMS["sims"] for pool, sid in zip(pools, sids))
        for pool, sid in zip(pools, sids):
            pool.remove(sid)
        return out

    first, before = search(3), cache.table.stats
    assert before.leaves > groups * PARAMS["sims"] // 2 and before.stored > 10
    rows = evaluator.rows
    again, after = search(3), cache.table.stats
    assert again == first and after.stored == before.stored
    assert after.leaves - before.leaves == before.leaves  # the same leaves again
    assert after.hits + after.repeats - before.hits - before.repeats == before.leaves - groups  # all but the roots
    if groups == 1:
        assert evaluator.rows - rows == 1


def test_a_failed_round_is_withdrawn_with_a_cache_too():
    """An evaluator error mid-search: the leaves are withdrawn (the cached ones too), nothing is
    counted for the round, and the search then finishes as the uncached one does."""
    cache = NetworkCache(MODEL.num_actions, MODEL.num_players, 4.0)
    game = positions()[1]

    class Failing(Counted):
        def __call__(self, observations):
            if self.calls == 5:
                self.calls += 1
                raise RuntimeError("the device fell over")
            return super().__call__(observations)

    results = []
    for cached in (True, False):
        evaluator, pool = Failing(), pool_with(cache if cached else None)
        if cached:
            cache.sync(evaluator)
        sids = [pool.add(game, seed=1), pool.add(game, seed=1), pool.add(game, seed=2)]
        with pytest.raises(RuntimeError, match="fell over"):
            run_pools([pool], evaluator, batch_rows=[12])
        assert pool.pending == 0 and not pool.done(sids[0])
        run_pools([pool], evaluator, batch_rows=[12])
        results.append([numbers(pool, sid) for sid in sids])
        if cached:
            stats = cache.table.stats
            assert stats.leaves == evaluator.rows + stats.hits + stats.repeats and stats.repeats > 0
    assert results[0] == results[1]


# ---------------------------------------------------------------- whose weights


def test_sync_clears_whenever_the_evaluator_or_its_weights_changed():
    model = tiny_model(["base-2p"], level="full", seed=9)
    evaluator = BatchEvaluator(model, device="cpu", infer_rows=8, mode="strict")
    cache = NetworkCache(model.num_actions, model.num_players, 1.0)
    row = np.arange(model.obs_size, dtype=np.float32)

    def fill():
        cache.table.store(row, np.zeros(model.num_actions, np.float32), np.zeros(model.num_players, np.float32))
        assert cache.table.find(row) is not None

    def cleared() -> bool:
        clears = cache.table.stats.clears
        out = cache.sync(evaluator)
        assert cache.table.stats.clears == clears + out and (not out or cache.table.find(row) is None)
        return out

    assert cleared() and not cleared()  # first sight of the evaluator, then nothing new
    fill()
    assert not cleared() and cache.table.find(row) is not None
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    sum(p.sum() for p in model.parameters()).backward()
    optimizer.step()  # an optimizer step: in-place writes
    assert cleared() and not cleared()
    fill()
    model.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})  # even the same values
    assert cleared() and not cleared()
    fill()
    with torch.no_grad():
        next(iter(model.parameters()))[..., :1] += 1.0  # one element of one tensor
    assert cleared() and not cleared()
    fill()
    with torch.no_grad():
        next(iter(model.buffers())).copy_(next(iter(model.buffers())))  # a buffer
    assert cleared() and not cleared()
    fill()
    parameter = next(iter(model.parameters()))
    parameter.data = parameter.data.clone()  # other memory under the same tensor
    assert cleared() and not cleared()
    fill()
    evaluator = BatchEvaluator(model, device="cpu", infer_rows=8, mode="strict")  # e.g. SelfPlay.refresh
    assert cleared() and not cleared()
    # the old evaluator is gone: an object that happens to take its place is still another one
    fill()
    del evaluator
    evaluator = BatchEvaluator(model, device="cpu", infer_rows=8, mode="strict")
    assert cleared() and not cleared()


def test_a_tensor_without_a_version_counter_clears_at_every_sync():
    model = tiny_model(["base-2p"], level="full", seed=9)
    with torch.inference_mode():
        model.register_buffer("made_in_inference_mode", torch.zeros(3))
    evaluator = BatchEvaluator(model, device="cpu", infer_rows=8, mode="strict")
    cache = NetworkCache(model.num_actions, model.num_players, 1.0)
    assert NetworkCache.weights(model) is None
    assert cache.sync(evaluator) and cache.sync(evaluator)


# ---------------------------------------------------------------- self-play

TRAIN = {**ALPHAZERO_KEYS, "iterations": 2, "steps_per_iteration": 24, "sims": 4, "fast_sims": 2,
         "full_search_prob": 0.5, "min_replay": 1, "updates_per_iteration": 3, "minibatch_size": 16,
         "replay_capacity": 500, "checkpoint_interval": 1, "search_rows": 16}
ENV = {"scenarios": [{"game": "base-2p", "maps": 4, "turn_limit": 8}], "num_envs": 6}  # short games


def self_play(groups: int, cache_mb: float = 0.0, level="private", **train) -> SelfPlay:
    cfg = {"model": {"preset": "graph-tiny"}, "obs": {"level": level}}
    plan = plan_env(ENV, pool_seed=1)
    torch.manual_seed(0)
    model = build_policy_model(cfg, plan.dims).eval()
    return SelfPlay(plan, {**TRAIN, **train}, model=model, num_envs=6, threads=2, seed=5, device="cpu",
                    mode="strict", groups=groups, cache_mb=cache_mb)


def played(groups: int, cache_mb: float = 0.0, steps=40, level="private", between=None, **train):
    """Self-play's moves, replay rows and counters over ``steps`` steps in ``groups`` groups, and
    the SelfPlay itself. ``between(play, step)`` runs before each step."""
    play = self_play(groups, cache_mb, level, **train)
    replay = ReplayBuffer(5000, play.vec.observation_size, play.vec.action_size, play.players)
    stats = {"leaves": 0.0, "full": 0.0, "depth": 0.0, "max_depth": 0.0, "rounds": 0.0}
    rng = np.random.default_rng(0)
    actions = []
    for step in range(steps):
        if between is not None:
            between(play, step)
        play.step(0, step, rng, replay, stats)
        actions.append(play.buffers["actions"].copy())
    play.drop_trees()  # reuse: the trees kept for the next step
    assert all(len(pool) == 0 and pool.pending == 0 for pool in play.pools)
    n = replay.size
    counters = {k: v for k, v in stats.items() if k not in ("rounds", "cached")}  # rounds: the network's
    return ((np.stack(actions), replay.obs[:n].copy(), replay.policy[:n].copy(), replay.value[:n].copy(),
             replay.has_policy[:n].copy(), [r.policy for rows in play.pending for r in rows], play.games,
             play.decisions, counters), stats, play)


def same(a, b) -> bool:
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, list | tuple):
        return len(a) == len(b) and all(x is y is None or same(x, y) for x, y in zip(a, b))
    return a == b


REUSE = {"reuse": True, "root": "puct", "dets": 0, "sims": 16, "fast_sims": 8, "pending": 3, "fast_pending": 2,
         "policy_target": "pruned", "forced_playouts": 2.0, "root_policy_temperature": 1.1}
SETTINGS = {
    "reuse": ("full", REUSE),
    "hidden-hands": ("private", {"sims": 12, "pending": 3, "fast_pending": 2}),
    "pruned": ("full", {"sims": 16, "pending": 3, "policy_target": "pruned", "root": "puct", "dets": 0}),
    "gumbel": ("private", {"sims": 12, "pending": 4, "root": "gumbel", "policy_target": "gumbel"}),
}


@pytest.mark.parametrize("setting,groups", [("reuse", 1), ("reuse", 2), ("hidden-hands", 2), ("pruned", 1),
                                            ("gumbel", 2)])
def test_self_play_with_a_cache_is_self_play_without(setting, groups):
    """Tree reuse in one pool a group, hidden hands (sampled worlds), both budgets in two pools,
    the rebuilt targets' root outputs (RootOutputs reads the roots' rows: a root keeps its row):
    the moves, the replay's rows and targets, the targets still pending and every counter, the
    leaves too, with a roomy table and with one of twenty rows; fewer rows reach the network."""
    level, train = SETTINGS[setting]
    reference, plain, _ = played(groups, level=level, **train)
    assert len(reference[1]) > 20 and "cached" not in plain
    hits = []
    for megabytes in (64.0, 20 * ENTRY / 2**20):
        result, stats, play = played(groups, megabytes, level=level, **train)
        assert same(result, reference), (setting, megabytes)
        table = play.cache.table.stats
        assert stats["cached"] == table.hits + table.repeats > 0 and table.leaves == plain["leaves"]
        assert stats["rounds"] <= plain["rounds"] and table.clears == 1
        assert table.entries + table.replaced == table.stored
        assert table.replaced > 0 and table.entries <= 20 if megabytes < 1 else table.replaced < table.stored // 20
        hits.append(table.hits)
    assert hits[0] > hits[1]  # the roomy table answers more


def nudge(play, step, at=(13, 27)):
    """The weights change mid-run without a word to SelfPlay (an optimizer's in-place step)."""
    if step in at:
        generator = torch.Generator().manual_seed(step)
        with torch.no_grad():
            for parameter in play.model.parameters():
                parameter.add_(0.05 * torch.randn(parameter.shape, generator=generator))


def test_nothing_of_older_weights_is_served(monkeypatch, groups=2):
    """The weights move twice mid-run and nobody calls refresh(): the cached run is still the
    uncached one, having cleared its table at the next step each time. With the check taken out
    (the control: the table keeps the old weights' outputs) the run is another one."""
    reference, _, _ = played(groups, level="full", between=nudge, **REUSE)
    unmoved, _, _ = played(groups, level="full", **REUSE)
    assert not same(reference, unmoved)
    result, stats, play = played(groups, 8.0, level="full", between=nudge, **REUSE)
    assert same(result, reference)
    assert play.cache.table.stats.clears == 3 and stats["cached"] > 0
    monkeypatch.setattr(NetworkCache, "weights", staticmethod(lambda model: ()))
    stale, _, play = played(groups, 8.0, level="full", between=nudge, **REUSE)
    assert play.cache.table.stats.clears == 1 and not same(stale, reference)


def test_a_new_evaluator_and_refresh_clear_the_cache():
    """Another evaluator (the fused kernels' weights are copies made when theirs was built):
    cleared at the next step; refresh() clears at once."""
    other = {}

    def swap(play, step):
        if step == 20:
            torch.manual_seed(1)
            other["model"] = build_policy_model({"model": {"preset": "graph-tiny"}, "obs": {"level": "full"}},
                                                play.dims).eval()
            play.evaluator = BatchEvaluator(other["model"], device="cpu", infer_rows=16, mode="strict")

    reference, _, _ = played(1, level="full", between=swap, **REUSE)
    result, _, play = played(1, 8.0, level="full", between=swap, **REUSE)
    assert same(result, reference) and play.cache.table.stats.clears == 2
    entries = len(play.cache.table)
    assert entries > 0
    play.refresh()  # torch kernels: the same evaluator and weights, nothing to clear
    assert len(play.cache.table) == entries
    with torch.no_grad():
        next(iter(other["model"].parameters())).mul_(1.0)
    play.refresh()
    assert len(play.cache.table) == 0 and play.cache.table.stats.clears == 3


def test_self_play_refuses_a_negative_cache():
    with pytest.raises(ValueError, match="eval_cache_mb"):
        self_play(1, -1.0)
    play = self_play(2, 0.0)
    assert play.cache is None and all(pool.cache is None for pool in play.pools)
    play = self_play(2, 1.0)
    assert all(pool.cache is play.cache.table for pool in play.pools) and len(play.pools) == 4


# ---------------------------------------------------------------- the runtime key


def variant(name: str, **runtime) -> Variant:
    raw = {"seed": 3, "env": {**ENV, "num_envs": 4}, "obs": {"level": "full"}, "model": {"preset": "graph-tiny"},
           "train": {**TRAIN, **REUSE},
           "runtime": {"device": "cpu", "threads": 2, "determinism": "strict", **runtime}}
    cfg = resolve(raw, experiment="az", variant=name)
    return Variant("az", name, None, {}, cfg, config_hash(cfg))


def test_eval_cache_mb_is_a_runtime_key_outside_the_hash():
    unset, on, none = variant("a"), variant("a", eval_cache_mb=64), variant("a", eval_cache_mb=None)
    assert "eval_cache_mb" not in unset.config["runtime"] and "eval_cache_mb" not in none.config["runtime"]
    assert on.config["runtime"]["eval_cache_mb"] == 64
    assert unset.config_hash == on.config_hash == none.config_hash


@pytest.mark.train
def test_runs_with_and_without_a_cache_train_the_same_weights(tmp_path, monkeypatch):
    """runtime.eval_cache_mb unset, roomy and tiny, in one group and two: the same checkpoints
    (the cache outlives no learn()), and the cached runs log their hits."""
    monkeypatch.chdir(tmp_path)
    shas, metrics = {}, {}
    for name, runtime in (("unset", {}), ("roomy", {"eval_cache_mb": 16}), ("tiny", {"eval_cache_mb": 0.02}),
                          ("serial", {"eval_cache_mb": 16, "search_groups": 1})):
        with RunWriter.open(variant(name, **runtime), argv=["test"]) as writer:
            train_alphazero(writer.variant_for_training, writer)
        run = load_experiment(f"az/{name}")
        shas[name], metrics[name] = [c.weights_sha256 for c in run.checkpoints()], run.metrics()
    assert len(shas["unset"]) == 3 and len(set(shas["unset"])) == 3
    assert shas["roomy"] == shas["tiny"] == shas["serial"] == shas["unset"]
    assert not any(key.startswith("selfplay/cache") for row in metrics["unset"] for key in row)
    for name in ("roomy", "tiny", "serial"):
        for row, plain in zip(metrics[name], metrics["unset"]):
            assert 0.0 < row["selfplay/cache_hits"] < 1.0 and 0.0 <= row["selfplay/cache_replaced"] <= 1.0
            assert row["selfplay/rounds"] <= plain["selfplay/rounds"]
    assert metrics["tiny"][-1]["selfplay/cache_replaced"] > metrics["roomy"][-1]["selfplay/cache_replaced"]
    assert metrics["tiny"][-1]["selfplay/cache_hits"] < metrics["roomy"][-1]["selfplay/cache_hits"]
