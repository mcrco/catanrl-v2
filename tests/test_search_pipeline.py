"""The pipelined round loop: the searches of ``run_groups`` are ``run_pools``'s for any
number of groups (strict evaluator), the pipeline's outputs are the evaluator's, and
self-play in search groups plays, records and trains what the serial loop does."""

from __future__ import annotations

import cppanatron as cpp
import numpy as np
import pytest
import torch
from eval_helpers import tiny_model
from test_evaluator import emulate_graphs

from catanrl import games
from catanrl.config import Variant, config_hash, resolve
from catanrl.envs.puffer import plan_env
from catanrl.runs import RunWriter, load_experiment
from catanrl.search import (
    BatchEvaluator,
    Group,
    Pipeline,
    make_pool,
    run_groups,
    run_pools,
    search_config,
)
from catanrl.train.alphazero import ALPHAZERO_KEYS, ReplayBuffer, Rounds, SelfPlay, train_alphazero
from catanrl.train.puffer import build_policy_model

MODEL = tiny_model(["base-2p"], level="full", seed=4)
DIMS = MODEL.dims
BUDGETS = ({"sims": 24, "pending": 4, "oracle": True, "noise": True}, {"sims": 6, "pending": 2, "oracle": True})


def positions(count=12, seed=0):
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


def rows(count=40):
    encoder = cpp.GraphEncoder(cpp.EncoderDims(**DIMS), cpp.InfoLevel.FULL)
    return np.stack([encoder.encode(game, game.current_player_index) for game in positions(count, seed=1)])


ROWS = rows()
CODEC = cpp.GraphEncoder(cpp.EncoderDims(**DIMS), cpp.InfoLevel.FULL).action_space


def strict(model=MODEL, infer_rows=8) -> BatchEvaluator:
    return BatchEvaluator(model, device="cpu", infer_rows=infer_rows, mode="strict")


class Searches:
    """The same games searched for a few moves, each with its own seed and one of two budgets
    (game i: BUDGETS[budget[i]]), in ``groups`` groups of two pools: game i in group
    ``group[i]``. ``play`` runs a move's searches with ``run`` and returns every search's
    numbers, then plays the selected actions."""

    def __init__(self, groups: int, group, budget, threads=2) -> None:
        self.games = [game.copy() for game in positions()]
        self.group, self.budget = list(group), list(budget)
        configs = [search_config(params, max_players=DIMS["max_players"], threads=threads) for params in BUDGETS]
        self.pools = [[make_pool(DIMS, "full", config) for config in configs] for _ in range(groups)]
        self.batch_rows = [[0, 0] for _ in range(groups)]

    def add(self, move: int) -> list[tuple[cpp.SearchPool, int]]:
        searches = []
        self.batch_rows = [[0, 0] for _ in self.pools]
        for i, game in enumerate(self.games):
            pool = self.pools[self.group[i]][self.budget[i]]
            searches.append((pool, pool.add(game, seed=1000 * move + i)))
            self.batch_rows[self.group[i]][self.budget[i]] += BUDGETS[self.budget[i]]["pending"]
        return searches

    def read(self, searches) -> list[tuple]:
        out = []
        for game, (pool, sid) in zip(self.games, searches):
            action = pool.select(sid, 1.0)
            # bytes: the very bits (q is nan where an action has no visits)
            out.append((pool.visits(sid).tobytes(), pool.q(sid).tobytes(), pool.policy_target(sid).tobytes(), action,
                        pool.stats(sid).simulations))
            pool.remove(sid)
            game.execute(CODEC.decode(game, action))
        return out


def serial(evaluator, moves=3):
    """run_pools over one group: the reference."""
    s = Searches(1, [0] * 12, [i % 3 == 0 for i in range(12)])
    out = []
    for move in range(moves):
        searches = s.add(move)
        run_pools(s.pools[0], evaluator, batch_rows=s.batch_rows[0])
        out.append(s.read(searches))
    return out


def grouped(evaluator, group, groups, moves=3, pipeline=None):
    s = Searches(groups, group, [i % 3 == 0 for i in range(12)])
    pipeline = pipeline or Pipeline(evaluator)
    out, rounds = [], []
    for move in range(moves):
        searches = s.add(move)
        members = [Group(pools, rows) for pools, rows in zip(s.pools, s.batch_rows)]
        leaves, most = run_groups(members, pipeline)
        assert leaves == sum(g.leaves for g in members) and most == max(g.rounds for g in members)
        rounds.append([g.rounds for g in members])
        out.append(s.read(searches))
    return out, rounds


# ---------------------------------------------------------------- the pipeline's outputs


@pytest.mark.parametrize("mode", ["strict", "fast", "graphs"])
def test_queued_batches_return_the_evaluators_outputs(mode, monkeypatch):
    """Two batches queued before either result is read, in buffers that are reused: the
    evaluator's own outputs and counters (fast mode pads and chunks; the graph path is
    emulated on CPU, where a replay reads the static rows as they are at that moment)."""
    def build():
        evaluator = BatchEvaluator(MODEL, device="cpu", infer_rows=8, mode="strict" if mode == "strict" else "fast")
        return emulate_graphs(evaluator, monkeypatch) if mode == "graphs" else evaluator

    reference, pipeline = build(), Pipeline(build())
    a, b = pipeline.batch(0, 20), pipeline.batch(1, 13)
    assert (a.capacity, b.capacity) == (24, 16) and pipeline.batch(0, 24) is a
    for first, second in ((ROWS[:20], ROWS[20:33]), (ROWS[5:8], ROWS[9:10]), (ROWS[14:31], ROWS[:13])):
        a.observations[: len(first)] = first
        b.observations[: len(second)] = second
        pipeline.submit(a, len(first))
        pipeline.submit(b, len(second))
        for batch, chunk in ((a, first), (b, second)):
            logits, values = pipeline.result(batch)
            expected = reference(chunk)
            assert np.array_equal(logits, expected[0]) and np.array_equal(values, expected[1])
    assert (pipeline.evaluator.rows, pipeline.evaluator.forwards) == (reference.rows, reference.forwards)
    with pytest.raises(ValueError, match="nothing was submitted"):
        pipeline.result(a)
    pipeline.submit(a, 3)
    with pytest.raises(RuntimeError, match="queued"):
        pipeline.batch(0, 3)
    with pytest.raises(ValueError, match="3 queued"):
        pipeline.submit(a, 3)
    pipeline.drain()
    assert pipeline.batch(0, 3) is a and pipeline.batch(0, 25) is not a  # grown in whole forwards
    assert pipeline.batches[0].capacity == 32


def test_batches_outlive_an_evaluator_of_the_same_shapes():
    """Self-play builds an evaluator per iteration: the buffers are kept while they fit."""
    batches: dict = {}
    first = Pipeline(strict(), batches).batch(0, 10)
    assert Pipeline(strict(), batches).batch(0, 10) is first
    assert Pipeline(strict(infer_rows=4), batches).batch(0, 10) is not first
    with pytest.raises(ValueError, match="recurrent"):
        Pipeline(strict(tiny_model(["base-2p"], level="full", seed=2, preset="lstm-tiny")))


class Scaled(BatchEvaluator):
    """An evaluator with its own __call__ (as the tests' and anyone's subclasses have)."""

    def __call__(self, observations):
        logits, values = super().__call__(observations)
        return logits * np.float32(3.0), values


def test_an_evaluator_with_its_own_call_is_called():
    evaluator = Scaled(MODEL, device="cpu", infer_rows=8, mode="strict")
    pipeline = Pipeline(evaluator)
    assert not pipeline.queued and Pipeline(strict()).queued and not Pipeline(Rounds(strict())).queued
    batch = pipeline.batch(0, 9)
    batch.observations[:9] = ROWS[:9]
    pipeline.submit(batch, 9)
    logits, values = pipeline.result(batch)
    expected = evaluator(ROWS[:9])
    assert np.array_equal(logits, expected[0]) and np.array_equal(values, expected[1])
    split = [i % 2 for i in range(12)]
    assert grouped(evaluator, split, 2)[0] == serial(evaluator)


class Flagging(BatchEvaluator):
    """Forwards that return the fused kernels' third output: ``flags[n]`` for the n-th forward
    (0: none), with logits that are off wherever a flag is raised."""

    flags: dict[int, int] = {}  # noqa: RUF012
    calls = 0

    def _forward(self, obs, state=None):
        logits, values = super()._forward(obs, state)
        flag = self.flags.get(self.calls, 0)
        self.calls += 1
        return logits + flag, values, torch.full((1,), flag, dtype=torch.int32)


class Recomputing(Flagging):
    """With kernels=fused16's way out: ``_exact`` evaluates a chunk again (here the network's
    own forward, in the chunk's bucket) and returns its outputs and the flag left."""

    exact_forwards = 0
    left = 0

    def _exact(self, chunk, size):
        self.chunks = [*getattr(self, "chunks", []), (chunk.copy(), size)]
        batch = torch.from_numpy(chunk)
        batch = torch.cat((batch, batch[:1].expand(size - len(chunk), -1)), dim=0)
        with torch.inference_mode():
            logits, values = BatchEvaluator._forward(self, batch)
        return logits[: len(chunk)].numpy(), values[: len(chunk)].numpy(), np.full(1, self.left, dtype=np.int32)


def test_a_flagged_chunk_is_evaluated_again_or_refused():
    """The kernels' flag is read a chunk at a time when the outputs are back: bit 2 has that
    chunk (and no other) evaluated again by the evaluator's ``_exact`` from the rows still in
    the batch; a flag that is left, or one the evaluator has no way out of, raises."""
    def build(cls, **flags):
        evaluator = cls(MODEL, device="cpu", infer_rows=8, mode="fast")
        evaluator.flags = {int(n): flag for n, flag in flags.items()}
        if cls is Recomputing:
            evaluator.kernels = "fused16"  # the kernels whose flag has a way out
        return Pipeline(evaluator)

    def run(pipeline, rows=ROWS[:21]):
        batch = pipeline.batch(0, len(rows))
        batch.observations[: len(rows)] = rows
        pipeline.submit(batch, len(rows))
        return pipeline.result(batch)

    expected = BatchEvaluator(MODEL, device="cpu", infer_rows=8, mode="fast")(ROWS[:21])
    pipeline = build(Recomputing, **{"1": 2, "2": 2})  # forwards of rows 8..15 and 16..20 (bucket 8)
    assert pipeline.queued
    logits, values = run(pipeline)
    assert np.array_equal(logits, expected[0]) and np.array_equal(values, expected[1])
    ev = pipeline.evaluator
    assert ev.exact_forwards == 2 and [size for _, size in ev.chunks] == [8, 8]
    assert np.array_equal(ev.chunks[0][0], ROWS[8:16]) and np.array_equal(ev.chunks[1][0], ROWS[16:21])
    logits, values = run(pipeline)  # no flag any more: nothing is evaluated again
    assert np.array_equal(logits, expected[0]) and ev.exact_forwards == 2

    for cls, flags, left in ((Recomputing, {"0": 1}, 0), (Recomputing, {"2": 3}, 1), (Flagging, {"1": 2}, 0)):
        pipeline = build(cls, **flags)
        pipeline.evaluator.left = left
        with pytest.raises(RuntimeError, match="more than three neighbours"):
            run(pipeline)
        assert pipeline.batches[0].rows == 0
        pipeline.evaluator.flags = {}
        assert np.array_equal(run(pipeline)[0], expected[0])  # the pipeline and its batch are usable

    class Counting(Pipeline):  # the hook is the pipeline's to override
        seen = ()

        def flagged(self, batch, start, rows, flag):
            self.seen += ((start, rows, flag),)

    evaluator = Flagging(MODEL, device="cpu", infer_rows=8, mode="fast")
    evaluator.flags = {0: 2, 2: 1}
    pipeline = Counting(evaluator)
    run(pipeline)
    assert pipeline.seen == ((0, 8, 2), (16, 5, 1))


def test_a_bucket_is_captured_with_nothing_queued(monkeypatch):
    """A CUDA graph is captured by the evaluator itself, after the pipeline waited for what is
    queued on the device, once per bucket (graphs emulated on CPU: the order of the calls)."""
    evaluator = emulate_graphs(BatchEvaluator(MODEL, device="cpu", infer_rows=8, mode="fast"), monkeypatch)
    pipeline, order = Pipeline(evaluator), []
    capture = evaluator._capture
    monkeypatch.setattr(evaluator, "_capture", lambda forward: order.append("capture") or capture(forward))
    monkeypatch.setattr(pipeline, "_settle", lambda: order.append("settle"))
    a, b = pipeline.batch(0, 20), pipeline.batch(1, 8)
    a.observations[:20], b.observations[:3] = ROWS[:20], ROWS[20:23]
    pipeline.submit(a, 20)  # buckets 8, 8 and 4
    pipeline.submit(b, 3)  # bucket 4 again
    assert order == ["settle", "capture", "settle", "capture"] and sorted(evaluator._graphs) == [4, 8]
    expected = BatchEvaluator(MODEL, device="cpu", infer_rows=8, mode="fast")
    for batch, chunk in ((a, ROWS[:20]), (b, ROWS[20:23])):
        assert np.array_equal(pipeline.result(batch)[0], expected(chunk)[0])
    b.observations[:2] = ROWS[:2]
    pipeline.submit(b, 2)  # a new bucket
    pipeline.submit(a, 20)
    assert order[4:] == ["settle", "capture"]
    assert np.array_equal(pipeline.result(b)[0], expected(ROWS[:2])[0])
    assert np.array_equal(pipeline.result(a)[0], expected(ROWS[:20])[0])


# ---------------------------------------------------------------- the searches


def test_the_searches_are_run_pools_for_any_number_of_groups():
    """Every search's visits, q, policy target and selected action over three moves, with
    one, two and three groups, groups of unequal budgets (one finishes its rounds before
    the others) and a group without searches."""
    evaluator = strict()
    reference = serial(evaluator)
    assert all(sims in (24, 6) for move in reference for *_, sims in move)
    one, rounds = grouped(evaluator, [0] * 12, 1)
    assert one == reference
    serial_rounds = [r[0] for r in rounds]
    for groups, group in ((2, [i % 2 for i in range(12)]), (3, [i % 3 for i in range(12)]),
                          (2, [0] * 12), (3, [2 * (i % 2) for i in range(12)])):
        results, rounds = grouped(evaluator, group, groups)
        assert results == reference, (groups, group)
        assert [max(r) for r in rounds] == serial_rounds  # the slowest search's rounds, as in one group
        if 1 not in group and groups == 3:
            assert all(r[1] == 0 for r in rounds)  # the empty group never reaches the evaluator
    # group 0 holds every small search (i % 3 == 0) and nothing else: it is done first
    results, rounds = grouped(evaluator, [int(i % 3 != 0) for i in range(12)], 2)
    assert results == reference and all(0 < r[0] < r[1] for r in rounds)


def test_the_graph_path_runs_the_same_searches(monkeypatch):
    """Graphs emulated on CPU, their static rows shared by both groups: strict (a row at a
    time on CPU), and fast in one group, where the chunks and so the buckets are run_pools's."""
    reference = serial(strict())
    evaluator = emulate_graphs(strict(), monkeypatch)
    results, _ = grouped(evaluator, [i % 2 for i in range(12)], 2)
    assert results == reference and sorted(evaluator._graphs) == [1]
    fast = [emulate_graphs(BatchEvaluator(MODEL, device="cpu", infer_rows=8, mode="fast"), monkeypatch)
            for _ in range(2)]
    assert grouped(fast[0], [0] * 12, 1)[0] == serial(fast[1])
    assert sorted(fast[0]._graphs) == sorted(fast[1]._graphs) and len(fast[0]._graphs) > 1
    assert (fast[0].rows, fast[0].forwards) == (fast[1].rows, fast[1].forwards)


def test_an_error_mid_round_leaves_the_pools_usable(monkeypatch):
    """As run_pools: the leaves collected are withdrawn (every group's, also the ones queued
    behind the failing forward), the error is the evaluator's, and the searches run on."""
    evaluator = strict()
    s = Searches(2, [i % 2 for i in range(12)], [i % 3 == 0 for i in range(12)])
    searches = s.add(0)
    forward, calls = evaluator._forward, []

    def failing(*args):
        calls.append(1)
        if len(calls) == 150:  # a few rounds in
            raise RuntimeError("the device fell over")
        return forward(*args)

    monkeypatch.setattr(evaluator, "_forward", failing)
    pipeline = Pipeline(evaluator)
    members = [Group(pools, rows) for pools, rows in zip(s.pools, s.batch_rows)]
    with pytest.raises(RuntimeError, match="fell over"):
        run_groups(members, pipeline)
    assert sum(g.rounds for g in members) > 0
    assert all(pool.pending == 0 for pools in s.pools for pool in pools)
    assert all(batch.rows == 0 for batch in pipeline.batches.values())
    assert not all(pool.done(sid) for pool, sid in searches)
    run_groups(members, pipeline)
    assert all(pool.done(sid) for pool, sid in searches)
    assert all(pool.stats(sid).simulations == BUDGETS[b]["sims"] for (pool, sid), b in zip(searches, s.budget))


def test_pools_of_another_network_are_refused():
    other = tiny_model(["base-3p"], level="full", seed=1)
    pool = make_pool(other.dims, "full", search_config(BUDGETS[0], max_players=other.dims["max_players"], threads=1))
    with pytest.raises(ValueError, match="differ"):
        run_groups([Group([pool], [4])], Pipeline(strict()))
    assert run_groups([], Pipeline(strict())) == (0, 0)


# ---------------------------------------------------------------- self-play

TRAIN = {**ALPHAZERO_KEYS, "iterations": 2, "steps_per_iteration": 24, "sims": 4, "fast_sims": 2,
         "full_search_prob": 0.5, "min_replay": 1, "updates_per_iteration": 3, "minibatch_size": 16,
         "replay_capacity": 500, "checkpoint_interval": 1, "search_rows": 16}
ENV = {"scenarios": [{"game": "base-2p", "maps": 4, "turn_limit": 8}], "num_envs": 6}  # short games


def played(groups: int, steps=40, level="private", **train):
    """Self-play's moves, replay rows and counters over ``steps`` steps in ``groups`` groups."""
    cfg = {"model": {"preset": "graph-tiny"}, "obs": {"level": level}}
    plan = plan_env(ENV, pool_seed=1)
    torch.manual_seed(0)
    model = build_policy_model(cfg, plan.dims).eval()
    play = SelfPlay(plan, {**TRAIN, **train}, model=model, num_envs=6, threads=2, seed=5, device="cpu",
                    mode="strict", groups=groups)
    replay = ReplayBuffer(5000, play.vec.observation_size, play.vec.action_size, play.players)
    stats = {"leaves": 0.0, "full": 0.0, "depth": 0.0, "max_depth": 0.0, "rounds": 0.0}
    rng = np.random.default_rng(0)
    actions = []
    for step in range(steps):
        play.step(0, step, rng, replay, stats)
        actions.append(play.buffers["actions"].copy())
    play.drop_trees()  # reuse: the trees kept for the next step
    assert all(len(pool) == 0 for pools in play.groups for pool in pools) and len(play.groups) == groups
    n = replay.size
    return (np.stack(actions), replay.obs[:n].copy(), replay.policy[:n].copy(), replay.value[:n].copy(),
            replay.has_policy[:n].copy(), [r.policy for rows in play.pending for r in rows], play.games,
            play.decisions, stats)


def same(a, b) -> bool:
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, list):
        return len(a) == len(b) and all(x is y is None or same(x, y) for x, y in zip(a, b))
    return a == b


@pytest.mark.parametrize("train", [
    {},
    {"sims": 12, "pending": 3, "fast_pending": 2},
    {"sims": 12, "pending": 3, "policy_target": "pruned", "root": "puct", "full_search_prob": 1.0},
    {"sims": 12, "pending": 4, "root": "gumbel", "policy_target": "gumbel"},
], ids=["defaults", "pending", "pruned", "gumbel"])
def test_self_play_in_groups_is_the_serial_loops(train):
    """Hidden hands (a tracker and sampled worlds per search), both budgets, the rebuilt
    targets' root outputs per group: the moves, replay rows, pending targets and every
    counter (leaves and rounds too) of one group with two and with three."""
    reference = played(1, **train)
    assert reference[-1]["rounds"] > 40 and len(reference[1]) > 20
    for groups in (2, 3):
        assert all(same(a, b) for a, b in zip(played(groups, **train), reference)), groups
    with pytest.raises(ValueError, match="search_groups"):
        played(0, steps=0)


def test_reuse_in_groups_is_the_serial_loops():
    """train.reuse: a group has one pool for both budgets and a game stays in its group, so its
    kept tree is re-rooted in the pool that holds it; one group, two and three play, record and
    count (the re-rooted searches and their carried visits too) the same."""
    train = {"reuse": True, "root": "puct", "dets": 0, "sims": 12, "fast_sims": 6, "pending": 3, "fast_pending": 2,
             "policy_target": "pruned"}
    reference = played(1, level="full", **train)
    assert reference[-1]["reused"] > 20 and reference[-1]["carried_full"] > 0 and len(reference[1]) > 20
    for groups in (2, 3):
        assert all(same(a, b) for a, b in zip(played(groups, level="full", **train), reference)), groups


def variant(name: str, **runtime) -> Variant:
    raw = {"seed": 3, "env": {**ENV, "num_envs": 4}, "obs": {"level": "private"}, "model": {"preset": "graph-tiny"},
           "train": TRAIN, "runtime": {"device": "cpu", "threads": 2, "determinism": "strict", **runtime}}
    cfg = resolve(raw, experiment="az", variant=name)
    return Variant("az", name, None, {}, cfg, config_hash(cfg))


def test_search_groups_is_a_runtime_key_outside_the_hash():
    unset, two = variant("a"), variant("a", search_groups=2)
    assert "search_groups" not in unset.config["runtime"] and two.config["runtime"]["search_groups"] == 2
    assert unset.config_hash == two.config_hash


@pytest.mark.train
def test_runs_in_one_two_and_three_groups_train_the_same_weights(tmp_path, monkeypatch):
    """runtime.search_groups unset (two groups), 1, 2 and 3: the same checkpoints."""
    monkeypatch.chdir(tmp_path)
    shas = {}
    for name, runtime in (("unset", {}), ("one", {"search_groups": 1}), ("two", {"search_groups": 2}),
                          ("three", {"search_groups": 3})):
        with RunWriter.open(variant(name, **runtime), argv=["test"]) as writer:
            train_alphazero(writer.variant_for_training, writer)
        shas[name] = [c.weights_sha256 for c in load_experiment(f"az/{name}").checkpoints()]
    assert len(shas["one"]) == 3 and len(set(shas["one"])) == 3
    assert shas["unset"] == shas["two"] == shas["three"] == shas["one"]


# ---------------------------------------------------------------- CUDA


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["strict", "fast"])
def test_cuda_queued_batches_return_the_evaluators_outputs(mode):
    """On the device the forwards really are queued (page-locked rows, one stream): three
    batches behind one another return what the evaluator returns for the same rows (the
    same chunks in the same buckets, so the same bits), and the searches of two groups
    are the serial loop's with a strict evaluator."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    reference = BatchEvaluator(MODEL, device="cuda", infer_rows=8, mode=mode)
    pipeline = Pipeline(BatchEvaluator(MODEL, device="cuda", infer_rows=8, mode=mode))
    assert pipeline.evaluator.use_graphs
    batches = [pipeline.batch(i, 24) for i in range(3)]
    assert all(batch._rows.is_pinned() and batch._logits.is_pinned() for batch in batches)
    for _ in range(3):
        chunks = [ROWS[:20], ROWS[20:33], ROWS[33:36]]
        for batch, chunk in zip(batches, chunks):
            batch.observations[: len(chunk)] = chunk
            pipeline.submit(batch, len(chunk))
        for batch, chunk in zip(batches, chunks):
            logits, values = pipeline.result(batch)
            expected = reference(chunk)
            assert np.array_equal(logits, expected[0]) and np.array_equal(values, expected[1])
    if mode == "strict":
        assert grouped(pipeline.evaluator, [i % 2 for i in range(12)], 2, pipeline=pipeline)[0] == serial(reference)


@pytest.mark.gpu
def test_cuda_a_capture_waits_for_the_queue():
    """A bucket first used while another batch is queued: its graph is captured (by the
    evaluator) only when the device has finished that batch, and both results are right."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    reference = BatchEvaluator(MODEL, device="cuda", infer_rows=8, mode="fast")
    evaluator = BatchEvaluator(MODEL, device="cuda", infer_rows=8, mode="fast")
    pipeline, seen = Pipeline(evaluator), []
    capture = evaluator._capture

    def watched(forward):
        seen.append([batch._event is None or batch._event.query() for batch in pipeline.batches.values()])
        return capture(forward)

    evaluator._capture = watched
    long = np.concatenate([ROWS] * 40)  # 200 forwards queued
    a, b = pipeline.batch(0, len(long)), pipeline.batch(1, 8)
    a.observations[: len(long)], b.observations[:3] = long, ROWS[:3]
    pipeline.submit(a, len(long))
    pipeline.submit(b, 3)  # bucket 4: a capture behind a's queue
    assert len(seen) == 2 and all(all(done) for done in seen)
    for batch, chunk in ((a, long), (b, ROWS[:3])):
        logits, values = pipeline.result(batch)
        expected = reference(chunk)
        assert np.array_equal(logits, expected[0]) and np.array_equal(values, expected[1])
