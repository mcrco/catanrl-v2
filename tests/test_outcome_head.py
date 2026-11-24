"""The outcome head (``model.args.outcome``): each seat's final victory points and the turns
left, auxiliary targets of the AlphaZero trainer that nothing reads at play time.

* the head is built after every other part: the same seed gives the other parts the same
  weights and outputs with it and without (graph-tiny, graph-s and graph-m), and unset it
  leaves specs and config hashes as they were; the head and its two coefficients come together;
* a network with the dice input and the head warm-starts from a checkpoint that has neither
  and plays as it, bit for bit (``init_from.zero_new_inputs``; the head keeps its fresh
  weights), on every graph preset;
* the same weights with and without the head give the same logits and values through the
  module, the batch evaluator, POLICY, BELIEF and MCTS, the harness, self-play (one search
  group and the pipeline) and the fused kernels' folded weights (CPU, stub kernels; the device
  evaluators are marked ``gpu``);
* self-play labels the positions of finished games (seat-relative victory points, turns left)
  and masks those of games cut at the turn limit; the replay holds the labels in P + 4 bytes
  a position; the losses and their readouts in victory points and turns; a run from a plain
  checkpoint with the input and the head logs them and resumes.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_alphazero_dice import ENV, STATS, TRAIN, run
from test_dice_input import (
    DIMS,
    SEARCHERS,
    assert_plays_alike,
    balanced_games,
    decide,
    evaluate,
    first_projections,
    native,
    needs_kernels,
    real_rows,
    same_on_device,
    saved,
    trained_scale,
)

from catanrl import canon
from catanrl import checkpoint as ck
from catanrl.config import ConfigError, Variant, config_hash, resolve
from catanrl.envs.encoding import action_size, row_layout
from catanrl.envs.puffer import plan_env
from catanrl.models import build_model, resolve_spec
from catanrl.models.graph import OUTCOME_TURNS
from catanrl.runs import RunWriter, load_experiment
from catanrl.train.alphazero import (
    OUTCOME_HUBER_DELTA,
    OUTCOME_LABELS,
    ReplayBuffer,
    SelfPlay,
    _Pending,
    learn,
    outcome_labels,
    outcome_losses,
)
from catanrl.train.puffer import build_policy_model


@pytest.fixture(autouse=True)
def runs_dir(request, tmp_path, monkeypatch):
    if request.node.get_closest_marker("runs"):
        return  # the stored runs, read only
    monkeypatch.setenv("CATANRL_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.chdir(tmp_path)


def model_of(*, dice: bool = True, outcome: bool = True, seed: int = 0, preset: str = "graph-tiny",
             shared: bool = True):
    torch.manual_seed(seed)
    levels = {"level": "full", **({"dice_state": True} if dice else {})}
    args = {"shared": shared, **({"outcome": True} if outcome else {})}
    return build_model({"preset": preset, "args": args}, dims=DIMS, levels=levels,
                       num_actions=action_size(DIMS), layout=row_layout(DIMS, "full", None, dice)).eval()


def strip(full, bare):
    """``full``'s weights in ``bare``, a network without the outcome head."""
    bare.load_state_dict({k: v for k, v in full.state_dict().items() if not k.startswith("outcome_head.")})
    return bare


def pair(seed: int = 7, scale: bool = False, **kwargs):
    """(a network with the outcome head, the same weights without it)."""
    full = model_of(seed=seed, **kwargs)
    if scale:
        trained_scale(full)
    return full, strip(full, model_of(outcome=False, seed=seed + 50, **kwargs))


# ---------------------------------------------------------------- the network


@pytest.mark.parametrize("preset", ["graph-tiny", "graph-s", "graph-m"])
def test_the_head_is_built_last_and_changes_no_other_weight_or_output(preset):
    with_head, plain = model_of(preset=preset, seed=3), model_of(outcome=False, preset=preset, seed=3)
    mine, theirs = with_head.state_dict(), plain.state_dict()
    assert list(mine)[: len(theirs)] == list(theirs)
    assert all(torch.equal(mine[k], theirs[k]) for k in theirs)
    new = [k for k in mine if k not in theirs]
    assert new and all(k.startswith("outcome_head.") for k in new)
    assert with_head.outcome and not plain.outcome
    assert with_head.parts == {**plain.parts, "outcome": ["outcome_head."]}
    rows = real_rows(True)
    with torch.no_grad():
        assert_plays_alike(with_head, plain, rows, rows)
        heads = with_head.heads(rows)
        logits, value_logits, present, vp, turns = with_head.heads_with_outcome(rows)
    assert all(torch.equal(a, b) for a, b in zip(heads, (logits, value_logits, present)))
    assert vp.shape == (len(rows), 2) and turns.shape == (len(rows),)
    assert float(vp.abs().max()) < 0.5 and float(turns.abs().max()) < 0.5  # both start near 0
    with pytest.raises(ValueError, match="no outcome head"):
        plain.heads_with_outcome(rows)


def test_the_heads_size_on_graph_s():
    count = lambda m: sum(p.numel() for p in m.parameters())
    plain = count(model_of(outcome=False, preset="graph-s"))
    assert plain == 549_156 and count(model_of(preset="graph-s")) - plain == 135_938


def raw(*, outcome=None, dice: bool = True, init_from=None, **train):
    args = {"shared": True, **({} if outcome is None else {"outcome": outcome})}
    return {"seed": 3, "env": ENV, "obs": {"level": "full", **({"dice_state": True} if dice else {})},
            "model": {"preset": "graph-tiny", "args": args}, "train": {**TRAIN, **train},
            "runtime": {"device": "cpu", "threads": 2, "determinism": "strict"},
            "eval": {"select": None, "final": []}, **({"init_from": init_from} if init_from else {})}


def variant(name: str, **kwargs) -> Variant:
    cfg = resolve(raw(**kwargs), experiment="azo", variant=name)
    return Variant("azo", name, None, {}, cfg, config_hash(cfg))


def test_an_unset_head_keeps_specs_and_config_hashes():
    base = resolve_spec({"preset": "graph-s", "args": {"shared": True}})
    assert "outcome" not in base["args"]
    assert resolve_spec({"preset": "graph-s", "args": {"shared": True, "outcome": False}}) == base
    on = resolve_spec({"preset": "graph-s", "args": {"shared": True, "outcome": True}})
    assert on["args"] == {**base["args"], "outcome": True} and resolve_spec(on) == on
    with pytest.raises(ValueError, match="outcome"):
        resolve_spec({"preset": "graph-s", "args": {"outcome": 1}})
    with pytest.raises(ValueError, match="unknown graph args"):
        resolve_spec({"preset": "graph-s", "args": {"outcomes": True}})
    with pytest.raises(ValueError, match="shared trunk"):
        model_of(shared=False)
    plain = variant("a")
    unset = variant("a", outcome=False, outcome_vp_coef=0, outcome_turns_coef=0.0)
    assert plain.config_hash == unset.config_hash and plain.config == unset.config
    assert not {"outcome_vp_coef", "outcome_turns_coef"} & set(plain.config["train"])
    assert "outcome" not in plain.config["model"]["args"]
    on = variant("a", outcome=True, outcome_vp_coef=1, outcome_turns_coef=0.25)
    assert on.config_hash != plain.config_hash and on.config["model"]["args"]["outcome"] is True
    assert on.config["train"]["outcome_vp_coef"] == 1.0 and isinstance(on.config["train"]["outcome_vp_coef"], float)
    assert on.config["train"]["outcome_turns_coef"] == 0.25
    assert variant("a", outcome=True, outcome_turns_coef=0.25).config_hash != on.config_hash


@pytest.mark.parametrize("kwargs, match", [
    ({"outcome_vp_coef": 0.5}, "set model.args.outcome"),
    ({"outcome": True}, "outcome_vp_coef"),
    ({"outcome": True, "outcome_turns_coef": -1.0}, "non-negative"),
    ({"outcome": True, "outcome_vp_coef": "much"}, "non-negative"),
    ({"outcome": True, "outcome_vp_coef": True}, "non-negative"),
])
def test_the_head_and_its_coefficients_come_together(kwargs, match):
    with pytest.raises(ConfigError, match=match):
        resolve(raw(**kwargs), experiment="azo", variant="bad")


def test_only_the_alphazero_trainer_trains_the_head():
    ppo = {**raw(outcome=True), "train": {"algo": "ppo"}}
    with pytest.raises(ConfigError, match="trains with train.algo alphazero"):
        resolve(ppo, experiment="azo", variant="bad")


# ---------------------------------------------------------------- warm starts


@pytest.mark.parametrize("preset", ["graph-tiny", "graph-s", "graph-m"])
def test_a_warm_start_with_the_input_and_the_head_plays_as_its_source(tmp_path, preset):
    """The bundle's e0: every weight of the source copied, the dice inputs' zero, the head's
    fresh; the logits and values are the source's bit for bit, as built and as saved."""
    source = trained_scale(model_of(dice=False, outcome=False, preset=preset, seed=3))
    ref = saved(tmp_path, source)
    target = model_of(preset=preset, seed=4)
    fresh = {k: v.clone() for k, v in target.state_dict().items() if k.startswith("outcome_head.")}
    ck.load_parts(target, {"ckpt": str(ref.path), "parts": ["actor", "critic"], "zero_new_inputs": True})
    theirs, mine = source.state_dict(), target.state_dict()
    assert sorted(set(mine) - set(theirs)) == sorted([*ck.dice_input_keys(target), *fresh])
    assert all(torch.equal(mine[k], theirs[k]) for k in theirs)
    assert all(torch.equal(mine[k], fresh[k]) for k in fresh) and any(v.any() for v in fresh.values())
    assert_plays_alike(source, target, real_rows(False), real_rows(True))
    again = ck.Checkpoint.load(saved(tmp_path / "warm", target).path)
    loaded = again.model("all")
    assert loaded.outcome and again.header["model"]["args"]["outcome"] is True
    assert all(torch.equal(loaded.state_dict()[k], mine[k]) for k in mine)
    assert_plays_alike(source, again.model("actor"), real_rows(False), real_rows(True))


@pytest.mark.runs
@pytest.mark.parametrize("ref", ["az-2p-kata/main+1500@e1800#7c44ec80540a",
                                 "graph-m-2p-lr/main@e22711#673425d606b5"], ids=["e1800", "2p-graph-m-1"])
def test_real_checkpoints_warm_start_with_the_input_and_the_head(ref):
    """az-2p-bundle's start (e1800, graph-s) and best-2p (2p-graph-m-1, graph-m): with the dice
    input and the outcome head they play as stored."""
    from catanrl.runs import RunError, load_checkpoint

    try:
        stored = load_checkpoint(ref)
    except (RunError, OSError, ValueError):
        pytest.skip(f"{ref} is not in the runs directory")
    source = stored.model("all")
    assert source.shared and not stored.dice_state and not source.outcome
    spec = {**stored.header["model"], "args": {**stored.header["model"]["args"], "outcome": True}}
    torch.manual_seed(0)
    target = build_model(spec, dims=stored.dims, levels={"level": "full", "dice_state": True},
                         num_actions=action_size(stored.dims),
                         layout=row_layout(stored.dims, "full", None, True)).eval()
    ck.load_parts(target, {"ckpt": ref, "parts": ["actor", "critic"], "zero_new_inputs": True})
    assert target.outcome and len(ck.dice_input_keys(target)) == 2
    assert_plays_alike(source, target, real_rows(False), real_rows(True))


# ---------------------------------------------------------------- play time


@pytest.fixture
def ckpts(tmp_path):
    full, bare = pair()
    return saved(tmp_path / "head", full), saved(tmp_path / "bare", bare)


@pytest.mark.parametrize("spec", SEARCHERS)
def test_agents_play_the_head_network_as_the_same_weights_without(ckpts, spec):
    with_head, without = (native(spec.format(ref.path)) for ref in ckpts)
    answers = 0
    for game, seat in balanced_games(3, 400, 31):
        if len(game.playable_actions) < 2:
            continue
        (_, a, spy_a), (_, b, spy_b) = decide(with_head, game, seat), decide(without, game, seat)
        assert np.array_equal(a, b), spec
        assert len(spy_a.rows) == len(spy_b.rows) > 0
        for rows_a, rows_b in zip(spy_a.rows, spy_b.rows):
            assert np.array_equal(rows_a, rows_b)
            for out_a, out_b in zip(with_head.evaluator(rows_a), without.evaluator(rows_b)):
                assert np.array_equal(np.asarray(out_a), np.asarray(out_b))
            answers += len(rows_a)
    assert answers >= 8


@pytest.mark.parametrize("mode", ["strict", "fast"])
def test_the_batch_evaluator_answers_as_without_the_head(mode):
    full, bare = pair(preset="graph-s", scale=True)
    rows = real_rows(True)
    a, b = evaluate(full, rows, "torch", mode, device="cpu"), evaluate(bare, rows, "torch", mode, device="cpu")
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_the_harness_plays_the_head_network_as_the_same_weights_without(ckpts, tmp_path):
    from catanrl.eval.harness import play_schedule
    from catanrl.eval.suites import load_suite

    other = saved(tmp_path / "other", model_of(dice=False, outcome=False, seed=8))
    suite = dataclasses.replace(load_suite("select-2p-fr"), boards=2)
    games = suite.schedule(2)
    results = []
    for ref in ckpts:
        rows = []
        agents = [native(f"POLICY:ckpt={ref.path}"), native(f"POLICY:ckpt={other.path}")]
        play_schedule(agents, games, suite=suite, device="cpu", num_envs=2, threads=1, infer_rows=4,
                      on_game=rows.append)
        results.append(sorted(rows, key=lambda row: row["game_id"]))
    assert len(results[0]) == 4 and results[0] == results[1]


def same(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return True


def test_the_fused_kernels_take_the_head_network_and_fold_the_same_weights(monkeypatch):
    """kernels=fused reads the trunk, the policy head and the value head: a network whose only
    extra head is the outcome head is accepted, and every folded weight is that of the same
    weights without it. (CPU, stub kernels; tools/bundle_smoke.py runs the forward on the GPU.)"""
    from catanrl.models import graph_kernels

    monkeypatch.setattr(graph_kernels, "kernels", lambda: SimpleNamespace(hidden=96))
    full, bare = pair(preset="graph-s", scale=True)
    a, b = graph_kernels.FusedGraph(full), graph_kernels.FusedGraph(bare)
    tensors = [k for k, v in vars(a).items() if isinstance(v, (torch.Tensor, dict))]
    assert len(tensors) > 5 and same(vars(a), vars(b))
    rows = real_rows(True)
    with torch.no_grad():
        (glob_a, us_a), (glob_b, us_b) = first_projections(a, rows), first_projections(b, rows)
    assert torch.equal(glob_a, glob_b) and all(torch.equal(x, y) for x, y in zip(us_a, us_b))


@pytest.mark.gpu
@needs_kernels
@pytest.mark.parametrize("kernels, mode", [("torch", "strict"), ("torch", "fast"), ("fused", "strict"),
                                           ("fused", "fast"), ("fused16", "strict"), ("fused16", "fast")])
def test_the_device_evaluators_answer_as_without_the_head(tmp_path, kernels, mode):
    """On the GPU: the module, the fused kernels and their half-precision variant give the
    logits and values of the same weights without the head, bit for bit; and a network
    warm-started with the dice input and the head gives its source's."""
    full, bare = pair(preset="graph-s", scale=True)
    rows = real_rows(True)
    a, b = evaluate(full, rows, kernels, mode), evaluate(bare, rows, kernels, mode)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    source = trained_scale(model_of(dice=False, outcome=False, preset="graph-s", seed=3))
    target = model_of(preset="graph-s", seed=4)
    ck.load_parts(target, {"ckpt": str(saved(tmp_path, source).path), "parts": ["actor", "critic"],
                           "zero_new_inputs": True})
    was, now = evaluate(source, real_rows(False), kernels, mode), evaluate(target, rows, kernels, mode)
    assert same_on_device(was, now, kernels)  # the module on the wider rows: to rounding (test_dice_input.py)


# ---------------------------------------------------------------- self-play and the replay


def network(outcome: bool = True, seed: int = 0):
    torch.manual_seed(seed)
    cfg = {"model": {"preset": "graph-tiny", "args": {"shared": True, **({"outcome": True} if outcome else {})}},
           "obs": {"level": "full", "dice_state": True}}
    return build_policy_model(cfg, plan_env(ENV, pool_seed=1).dims).eval()


def selfplay(model, groups: int = 1, **train):
    plan = plan_env(ENV, pool_seed=1)
    play = SelfPlay(plan, {**TRAIN, **train}, model=model, num_envs=4, threads=2, seed=5, device="cpu", mode="strict",
                    groups=groups)
    replay = ReplayBuffer(5000, play.vec.observation_size, play.vec.action_size, play.players, outcome=play.outcome)
    return play, replay


@pytest.fixture
def as_won(monkeypatch):
    """Self-play that labels the games cut at the turn limit (10 turns here) as if they had
    ended there: finished games' labels without playing to 15 points. The ends it saw."""
    ends = []
    real = SelfPlay._last_outcome

    def last_outcome(self, env):
        last = real(self, env)
        assert last["truncated"] and not last["known"]
        last["known"] = True
        ends.append(last)
        return last

    monkeypatch.setattr(SelfPlay, "_last_outcome", last_outcome)
    return ends


@pytest.mark.parametrize("groups", [1, 2], ids=["one-group", "pipeline"])
def test_self_play_with_the_head_is_self_play_without_and_masks_cut_games(groups):
    """The searches never run the head: the same moves and targets; the games here are all cut
    at the turn limit, so no position has labels."""
    full = network(seed=3)
    bare = strip(full, network(False, seed=9))
    (a, replay_a), (b, replay_b) = selfplay(full, groups), selfplay(bare, groups)
    assert a.outcome and replay_a.outcome and not b.outcome and not replay_b.outcome
    rngs = np.random.default_rng(1), np.random.default_rng(1)
    stats = dict(STATS), dict(STATS)
    for step in range(100):
        a.step(0, step, rngs[0], replay_a, stats[0])
        b.step(0, step, rngs[1], replay_b, stats[1])
        assert np.array_equal(a.buffers["actions"], b.buffers["actions"]), step
    assert stats[0] == stats[1] and a.decisions == b.decisions and a.games == b.games > 0
    size = replay_a.size
    assert size == replay_b.size > 20
    for name in ("obs", "policy", "has_policy", "value", "present"):
        assert np.array_equal(getattr(replay_a, name)[:size], getattr(replay_b, name)[:size]), name
    assert replay_b.final_vp is None and "final_vp" not in replay_b.state()
    assert not replay_a.outcome_known[:size].any() and not replay_a.final_vp[:size].any()
    assert a.finished.sum() == a.games and all(a.vec.last_outcome(e)["truncated"] for e in range(4))


def test_self_play_labels_the_positions_of_a_finished_game(as_won):
    play, replay = selfplay(network(), record_fast=True)
    rng = np.random.default_rng(1)
    for step in range(100):
        play.step(0, step, rng, replay, dict(STATS))
    size = replay.size
    assert len(as_won) == play.games > 0 and size > 40
    assert replay.outcome_known[:size].all() and (replay.vp_to_win[:size] == 15).all()
    left = replay.turns_left[:size].astype(int)
    assert left.min() >= 0 and left.max() <= 10 and len(set(left.tolist())) > 3
    ends = {tuple(sorted(end["victory_points"])) for end in as_won}
    assert {tuple(sorted(row)) for row in replay.final_vp[:size].tolist()} == ends
    assert all(end["turns"] == 10 and end["vps_to_win"] == 15 for end in as_won)
    # a game's positions are stored in order: the turns left never grow within one
    assert (np.diff(left) > 0).sum() <= play.games


def test_outcome_labels_are_seat_relative():
    rows = [_Pending(None, None, 0, 2, 3), _Pending(None, None, 1, 2, 7), _Pending(None, None, 1, 2, 12)]
    last = {"games": 1, "turns": 9, "truncated": False, "vps_to_win": 15, "victory_points": [15, 6], "known": True}
    labels = outcome_labels(rows, last, 2)
    assert set(labels) == set(OUTCOME_LABELS)
    assert labels["final_vp"].tolist() == [[15, 6], [6, 15], [6, 15]] and labels["final_vp"].dtype == np.uint8
    assert labels["turns_left"].tolist() == [6, 2, 0] and labels["turns_left"].dtype == np.uint16
    assert labels["vp_to_win"].tolist() == [15] * 3 and labels["outcome_known"].tolist() == [True] * 3
    masked = outcome_labels(rows, {**last, "known": False}, 2)
    assert not any(values.any() for values in masked.values())
    # three seats of four slots: the row's own seat first, the empty slot zero
    padded = outcome_labels([_Pending(None, None, 2, 3, 0)], {**last, "victory_points": [3, 4, 10]}, 4)
    assert padded["final_vp"].tolist() == [[10, 3, 4, 0]]


def test_the_replay_holds_the_labels_in_six_bytes_a_position():
    rng = np.random.default_rng(0)

    def positions(count):
        return (rng.random((count, 10), dtype=np.float32).astype(np.float16),
                rng.random((count, 5), dtype=np.float32).astype(np.float16), rng.random(count) < 0.5,
                rng.random((count, 2), dtype=np.float32), np.ones((count, 2), dtype=np.float32))

    def labels(count):
        return {"final_vp": rng.integers(2, 16, (count, 2)).astype(np.uint8),
                "vp_to_win": np.full(count, 15, np.uint8),
                "turns_left": rng.integers(0, 600, count).astype(np.uint16), "outcome_known": rng.random(count) < 0.7}

    plain, labelled = ReplayBuffer(8, 10, 5, 2), ReplayBuffer(8, 10, 5, 2, outcome=True)
    with pytest.raises(ValueError, match="outcome labels"):
        labelled.add(*positions(3))
    with pytest.raises(ValueError, match="outcome labels"):
        plain.add(*positions(3), labels(3))
    plain.add(*positions(3))
    first, second = labels(5), labels(6)
    labelled.add(*positions(5), first)
    labelled.add(*positions(6), second)  # wraps: 11 positions through 8 slots
    width = lambda replay: sum(getattr(replay, name)[0].nbytes for name in replay.names)
    assert width(labelled) - width(plain) == 6 and labelled.names[-4:] == OUTCOME_LABELS
    state = labelled.state()
    newest = {name: np.concatenate((first[name], second[name]))[-8:] for name in OUTCOME_LABELS}
    for name in OUTCOME_LABELS:
        assert np.array_equal(state[name].numpy(), newest[name]), name
    again = ReplayBuffer(8, 10, 5, 2, outcome=True)
    again.restore(torch.load(_saved(state), weights_only=True))
    assert again.size == 8 and all(np.array_equal(getattr(again, name), state[name].numpy()) for name in again.names)
    assert again.turns_left.dtype == np.uint16 and again.outcome_known.dtype == bool
    batch = again.batch(np.arange(8), "cpu")
    assert batch["final_vp"].dtype == batch["turns_left"].dtype == batch["vp_to_win"].dtype == torch.float32
    assert batch["outcome_known"].dtype == torch.bool and batch["turns_left"].tolist() == newest["turns_left"].tolist()
    # a state from before the labels restores with every position unlabelled
    old = ReplayBuffer(8, 10, 5, 2, outcome=True)
    old.restore({k: v for k, v in state.items() if k not in OUTCOME_LABELS})
    assert old.size == 8 and not old.outcome_known.any() and np.array_equal(old.obs, again.obs)


def _saved(state):
    import io

    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return buffer


# ---------------------------------------------------------------- the losses


def test_the_losses_are_masked_hubers_and_the_readouts_victory_points_and_turns():
    t = torch.tensor
    batch = {"present": t([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0]]), "outcome_known": t([True, True, False]),
             "final_vp": t([[15.0, 6.0], [10.0, 0.0], [9.0, 9.0]]), "vp_to_win": t([15.0, 10.0, 15.0]),
             "turns_left": t([40.0, 250.0, 7.0])}
    vp, turns = torch.zeros(3, 2, requires_grad=True), torch.zeros(3, requires_grad=True)
    vp_loss, turns_loss, vp_mae, turns_mae = outcome_losses(vp, turns, batch)
    assert OUTCOME_HUBER_DELTA == 0.5 and OUTCOME_TURNS == 100.0
    # gaps 1, 0.4, 1 of the three labelled seats: linear beyond 0.5, quadratic within
    assert float(vp_loss) == pytest.approx((0.375 + 0.08 + 0.375) / 3)
    assert float(turns_loss) == pytest.approx((0.08 + 0.5 * (2.5 - 0.25)) / 2)  # 40 and 250 turns
    assert float(vp_mae) == pytest.approx((15 + 6 + 10) / 3) and float(turns_mae) == pytest.approx(145.0)
    (vp_loss + turns_loss).backward()
    assert not vp.grad[2].any() and not vp.grad[1, 1].any() and not turns.grad[2].any()  # masked
    assert vp.grad[0].all() and vp.grad[1, 0] and turns.grad[:2].all()
    exact = outcome_losses(batch["final_vp"] / batch["vp_to_win"][:, None], batch["turns_left"] / 100.0, batch)
    assert [float(x) for x in exact] == [0.0] * 4
    none = outcome_losses(vp, turns, {**batch, "outcome_known": t([False, False, False])})
    assert [float(x) for x in none] == [0.0] * 4


def test_learning_fits_the_labels(as_won):
    net = network(seed=2)
    play, replay = selfplay(net, record_fast=True)
    rng = np.random.default_rng(1)
    for step in range(100):
        play.step(0, step, rng, replay, dict(STATS))
    train = {**TRAIN, "outcome_vp_coef": 0.5, "outcome_turns_coef": 0.5, "updates_per_iteration": 20,
             "minibatch_size": 32, "learning_rate": 3e-3}
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-3)
    first = learn(net, optimizer, replay, train, "cpu", rng)
    for _ in range(5):
        last = learn(net, optimizer, replay, train, "cpu", rng)
    keys = ("outcome_vp_loss", "outcome_turns_loss", "outcome_vp_mae", "outcome_turns_mae")
    assert all(np.isfinite(first[k]) and np.isfinite(last[k]) and last[k] < first[k] for k in keys)
    assert first["outcome_known"] == pytest.approx(1.0) and np.isfinite(last["policy_loss"]) and np.isfinite(last["value_loss"])
    # without the head the same call has no such statistics
    bare = network(False)
    plain = learn(bare, torch.optim.AdamW(bare.parameters(), lr=1e-3), replay, TRAIN, "cpu", rng)
    assert not any(key.startswith("outcome") for key in plain)


# ---------------------------------------------------------------- a run


def test_a_run_from_a_plain_checkpoint_trains_the_head_logs_it_and_resumes(as_won, monkeypatch):
    """The bundle in small: a run without the input or the head, then one with both from its
    checkpoint (a fresh optimizer, an empty replay)."""
    assert run(variant("plain", dice=False, iterations=1)).status == "finished"
    start = load_experiment("azo/plain").resolve("latest")
    init = {"ckpt": "azo/plain@latest", "parts": ["actor", "critic"], "zero_new_inputs": True}
    v = variant("bundle", outcome=True, init_from=init, iterations=3, outcome_vp_coef=0.5, outcome_turns_coef=0.5,
                record_fast=True)
    real = RunWriter.should_stop

    def stop_after_two(self):
        if self.launch == 0 and len(canon.read_jsonl(self.path / "metrics.jsonl")) >= 2:
            self.stop_reason = "interrupted"
            return True
        return real(self)

    monkeypatch.setattr(RunWriter, "should_stop", stop_after_two)
    assert run(v, monitors=True).status == "interrupted"
    bundle = load_experiment("azo/bundle")
    e0 = ck.Checkpoint.load(bundle.checkpoints()[0].path)
    source, warm = ck.Checkpoint.load(start.path).model("all"), e0.model("all")
    theirs, mine = source.state_dict(), warm.state_dict()
    assert warm.outcome and e0.dice_state and all(torch.equal(mine[k], theirs[k]) for k in theirs)
    new = set(mine) - set(theirs)
    assert {k for k in new if not k.startswith("outcome_head.")} == set(ck.dice_input_keys(warm))
    assert any(k.startswith("outcome_head.") for k in new)
    state = torch.load(bundle.path / "trainer_state.pt", weights_only=False)
    replay = state["replay"]
    assert all(name in replay for name in OUTCOME_LABELS) and bool(replay["outcome_known"].all())
    assert len(replay["final_vp"]) == len(replay["obs"]) and replay["final_vp"].dtype == torch.uint8

    assert run(v, monitors=True).status == "finished"  # the resume
    metrics = load_experiment("azo/bundle").metrics()
    assert [m["launch"] for m in metrics] == [0, 0, 1]
    learned = [m for m in metrics if "train/policy_loss" in m]
    assert learned
    for m in learned:
        for key in ("outcome_vp_loss", "outcome_turns_loss", "outcome_vp_mae", "outcome_turns_mae"):
            assert np.isfinite(m[f"train/{key}"]) and m[f"train/{key}"] > 0.0, key
        assert m["train/outcome_known"] == pytest.approx(1.0) and 0.0 < m["train/outcome_vp_mae"] < 15.0
        assert 0.0 < m["train/outcome_turns_mae"] < 100.0
    probed = [m for m in metrics if "oos/outcome_rows" in m]
    assert probed and all(m["oos/outcome_rows"] > 0 and np.isfinite(m["oos/outcome_vp_mae"])
                          and np.isfinite(m["oos/outcome_turns_mae"]) for m in probed)
    final = ck.Checkpoint.load(load_experiment("azo/bundle").resolve("latest").path).model("all")
    moved = [k for k in mine if k.startswith("outcome_head.") and not torch.equal(final.state_dict()[k], mine[k])]
    assert moved  # the head trains
    assert not any(key.startswith(("train/outcome", "oos/outcome"))
                   for m in load_experiment("azo/plain").metrics() for key in m)
