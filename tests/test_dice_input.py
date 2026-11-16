"""The dice input (``obs.dice_odds``, approved by Marco 2026-11-10): each seat's next-roll odds,
``Game.roll_probabilities``, 11 per seat in the global block, seats relative to the observer.

* the features against ``roll_probabilities`` on real balanced games, in observer order;
* off, rows, layouts, levels, checkpoint headers and config hashes are as before;
* the config key and the checkpoint's ``obs.dice_odds`` round-trip, and the self-test pins
  declared balanced dice states;
* ``init_from.zero_new_inputs`` copies every weight and zeroes only the new inputs', so the
  warm-started network gives its source's logits and values on real rows;
* POLICY, BELIEF and MCTS (torch kernels) build their rows, worlds and leaves with the input
  from the checkpoint alone, the harness plays a dice network against one without it in
  either slot, and the fused kernels fold the input into their global projection;
* the training env's rows carry it, and a tiny PPO run from a checkpoint without it logs
  ``dice/*``, starts as its source and moves the dice weights off zero.
"""

from __future__ import annotations

import dataclasses
import json
import random
from types import SimpleNamespace

import cppanatron as cpp
import numpy as np
import pytest
import torch

from catanrl import checkpoint as ck
from catanrl.agents.native import make_native_agent
from catanrl.agents.spec import resolve_agent
from catanrl.config import ConfigError, config_hash, resolve
from catanrl.envs.encoding import (
    DICE_PREFIX,
    FAIR_ODDS,
    action_size,
    dims_for,
    make_encoder,
    row_layout,
)
from catanrl.games import load_game
from catanrl.models import build_model, normalize_levels
from catanrl.models.rows import canonical_feature_names, feature_names

GAME = "base-2p-fr"
DIMS = dims_for([GAME])


def model_of(dice: bool, *, level: str = "full", shared: bool = True, seed: int = 0, preset: str = "graph-tiny"):
    torch.manual_seed(seed)
    levels = {"level": level, **({"dice_odds": True} if dice else {})}
    return build_model({"preset": preset, "args": {"shared": shared}}, dims=DIMS, levels=levels,
                       num_actions=action_size(DIMS), layout=row_layout(DIMS, level, None, dice)).eval()


def saved(path, model, epoch=0):
    config = {"env": {"scenarios": [{"game": GAME}]}, "model": model.spec}
    return ck.save(path / f"e{epoch:06d}.pt", model, run=None, epoch=epoch, agent_steps=0, config=config, code={})


def balanced_games(n: int = 6, steps: int = 500, every: int = 23):
    """(game copy, seat to act) along random balanced games of base-2p-fr."""
    config = load_game(GAME)
    out = []
    for seed in range(n):
        game = config.cppanatron_game(seed, 100 + seed)
        rng = random.Random(seed)
        for step in range(steps):
            if game.winning_color is not None:
                break
            if step % every == 7:
                out.append((game.copy(), game.current_player_index))
            game.execute(rng.choice(game.playable_actions))
    return out


def odds_block(names, row, offset, players):
    first = names.index(f"{DICE_PREFIX}0_2")
    return row[offset + first: offset + first + 11 * players].reshape(players, 11)


# ---------------------------------------------------------------- the feature


def test_the_features_are_each_seats_roll_probabilities_in_observer_order():
    encoder = make_encoder(DIMS, "full", True)
    names = list(encoder.feature_names("global"))
    assert names[-22:] == [f"{DICE_PREFIX}{k}_{t}" for k in range(2) for t in range(2, 13)]
    offset = encoder.layout()["global"][0]
    differ = unfair = 0
    for game, _ in balanced_games():
        for observer in range(2):
            odds = odds_block(names, encoder.encode(game, observer), offset, 2)
            own = np.array(game.roll_probabilities(observer), dtype=np.float32)
            other = np.array(game.roll_probabilities(1 - observer), dtype=np.float32)
            assert np.array_equal(odds[0], own) and np.array_equal(odds[1], other)  # slot 0 is the observer
            assert abs(float(odds[0].sum()) - 1.0) < 1e-5
            differ += not np.array_equal(own, other)
            unfair += not np.allclose(own, FAIR_ODDS)
    assert differ > 0 and unfair > 0  # the seats' odds differ (seven balancing), and they are not fair


def test_canonical_names_pad_six_seats_and_fair_dice_read_fair():
    canonical = canonical_feature_names(dice_odds=True)["global"]
    plain = canonical_feature_names()["global"]
    assert canonical == plain + [f"{DICE_PREFIX}{k}_{t}" for k in range(6) for t in range(2, 13)]
    game = load_game("base-2p").cppanatron_game(1, 2)  # random dice
    encoder = make_encoder(DIMS, "private", True)
    odds = odds_block(list(encoder.feature_names("global")), encoder.encode(game, 1), encoder.layout()["global"][0], 2)
    assert np.array_equal(odds, np.tile(np.array(FAIR_ODDS, dtype=np.float32), (2, 1)))


# ---------------------------------------------------------------- off is unchanged


def test_off_is_bit_identical():
    for level in ("private", "full"):
        default, off = make_encoder(DIMS, level), make_encoder(DIMS, level, False)
        assert default.layout() == off.layout() and default.feature_names() == off.feature_names()
        for game, seat in balanced_games(2, 300, 41):
            assert np.array_equal(default.encode(game, seat), off.encode(game, seat))
        assert row_layout(DIMS, level) == row_layout(DIMS, level, None, False)
        assert feature_names(DIMS, level) == feature_names(DIMS, level, False)
        assert ck.layout_hash(DIMS, level) == ck.layout_hash(DIMS, level, False) != ck.layout_hash(DIMS, level, True)
    assert normalize_levels({"level": "full"}) == {"level": "full", "critic_level": "full"}
    assert normalize_levels({"level": "full", "dice_odds": False}) == {"level": "full", "critic_level": "full"}
    plain, dice = model_of(False), model_of(True)  # one seed: every shared weight drawn alike
    theirs, mine = plain.state_dict(), dice.state_dict()
    assert set(mine) - set(theirs) == {"backbone.dice_projection.weight"} and set(theirs) <= set(mine)
    assert all(torch.equal(mine[k], theirs[k]) for k in theirs)
    assert mine["backbone.dice_projection.weight"].shape == (theirs["backbone.global_projection.bias"].shape[0], 66)
    assert not mine["backbone.dice_projection.weight"].any()  # zero at initialization
    assert plain.backbone.dice_projection is None and plain.backbone.plain_globals == len(
        canonical_feature_names()["global"])


def test_an_unset_dice_input_keeps_the_config_hash():
    base = {"env": {"scenarios": [{"game": GAME}]}, "model": {"preset": "graph-tiny"}, "train": {"algo": "ppo"},
            "obs": {"level": "full"}}
    plain = resolve(base, experiment="t", variant="v")
    unset = resolve({**base, "obs": {"level": "full", "dice_odds": False}}, experiment="t", variant="v")
    on = resolve({**base, "obs": {"level": "full", "dice_odds": True}}, experiment="t", variant="v")
    assert "dice_odds" not in plain["obs"] and "dice_odds" not in unset["obs"] and on["obs"]["dice_odds"] is True
    assert config_hash(unset) == config_hash(plain) != config_hash(on)
    with pytest.raises(ConfigError, match="obs.dice_odds is true or false"):
        resolve({**base, "obs": {"level": "full", "dice_odds": "yes"}}, experiment="t", variant="v")
    with pytest.raises(ConfigError, match="train.algo ppo"):
        resolve({**base, "obs": {"level": "full", "dice_odds": True}, "train": {"algo": "alphazero"}},
                experiment="t", variant="v")
    init = {"ckpt": "x/y@e1", "zero_new_inputs": True}
    with pytest.raises(ConfigError, match="set obs.dice_odds: true"):
        resolve({**base, "init_from": init}, experiment="t", variant="v")
    with pytest.raises(ConfigError, match="restores weights only"):
        resolve({**base, "obs": {"level": "full", "dice_odds": True}, "init_from": {**init, "restore":
                 "weights+optimizer"}}, experiment="t", variant="v")


# ---------------------------------------------------------------- checkpoints


def test_the_checkpoint_stores_the_input_and_its_self_test_declares_dice_states(tmp_path):
    plain_ref, dice_ref = saved(tmp_path / "plain", model_of(False)), saved(tmp_path / "dice", model_of(True))
    plain, dice = ck.Checkpoint.load(plain_ref.path), ck.Checkpoint.load(dice_ref.path)
    assert set(plain.header["obs"]) == {"level", "critic_level", "dims", "layout_hash"} and not plain.dice_odds
    assert plain.levels == {"level": "full", "critic_level": "full"}
    assert dice.header["obs"]["dice_odds"] is True and dice.dice_odds
    assert dice.levels == {"level": "full", "critic_level": "full", "dice_odds": True}
    positions = json.loads(dice.header["self_test"]["positions"])
    assert all(p["game"]["dice_mode"] == "balanced" and "dice" in p for p in positions)
    assert len({json.dumps(p["dice"]) for p in positions}) > 4  # states of their own
    assert not any("dice" in p for p in json.loads(plain.header["self_test"]["positions"]))
    model = dice.model("all")
    assert model.levels["dice_odds"] and model.obs_size == model_of(True).obs_size
    obs = dice.header["self_test"]["obs"]
    names = feature_names(DIMS, "full", True)["global"]
    odds = obs[:, names.index(f"{DICE_PREFIX}0_2"): names.index(f"{DICE_PREFIX}0_2") + 22]
    assert not np.allclose(odds.numpy()[1:], np.tile(FAIR_ODDS, 2))  # the self-test reads unfair odds
    # A drifted dice feature is caught by the self-test.
    payload = torch.load(dice_ref.path, weights_only=True)
    payload["self_test"]["obs"][1, names.index(f"{DICE_PREFIX}0_7")] += 0.01
    torch.save(payload, tmp_path / "drift.pt")
    with pytest.raises(ck.EncoderDrift):
        ck.Checkpoint.load(tmp_path / "drift.pt").model("all")


# ---------------------------------------------------------------- the warm start


def real_rows(dice: bool, level: str = "full") -> torch.Tensor:
    encoder = make_encoder(DIMS, level, dice)
    return torch.from_numpy(np.stack([encoder.encode(g, s) for g, s in balanced_games()]))


def assert_plays_alike(a, b, rows_a, rows_b):
    """The same logits and values, bit for bit (the dice weights add exact zeros)."""
    with torch.no_grad():
        (la, va), (lb, vb) = a(rows_a), b(rows_b)
    assert torch.equal(la, lb) and torch.equal(va, vb)


@pytest.mark.parametrize("shared", [True, False])
def test_zero_new_inputs_copies_every_weight_and_plays_as_the_source(tmp_path, shared):
    source = model_of(False, shared=shared, seed=3)
    ref = saved(tmp_path, source)
    target = model_of(True, shared=shared, seed=4)
    with torch.no_grad():
        for key in ck.dice_input_keys(target):
            target.state_dict()[key].normal_()  # a fresh network's dice weights would be zero already
    with pytest.raises(ck.CheckpointError, match="init_from.zero_new_inputs"):  # the hint
        ck.load_parts(target, {"ckpt": str(ref.path)})
    ck.load_parts(target, {"ckpt": str(ref.path), "parts": ["actor", "critic"], "zero_new_inputs": True})
    theirs, mine = source.state_dict(), target.state_dict()
    new = sorted(set(mine) - set(theirs))
    assert new == sorted(ck.dice_input_keys(target)) == (["backbone.dice_projection.weight"] if shared else
                                                        ["actor.trunk.dice_projection.weight",
                                                         "critic.trunk.dice_projection.weight"])
    assert all(torch.equal(mine[k], theirs[k]) for k in theirs) and not any(mine[k].any() for k in new)
    assert_plays_alike(source, target, real_rows(False), real_rows(True))
    # nothing to zero: refused
    with pytest.raises(ck.CheckpointError, match="reads no input its source lacks"):
        ck.load_parts(model_of(False, shared=shared), {"ckpt": str(ref.path), "zero_new_inputs": True})


def test_zero_new_inputs_then_split_shared(tmp_path):
    source = model_of(False, shared=True, seed=5)
    ref = saved(tmp_path, source)
    target = model_of(True, shared=False, seed=6)
    ck.load_parts(target, {"ckpt": str(ref.path), "split_shared": True, "zero_new_inputs": True})
    assert_plays_alike(source, target, real_rows(False), real_rows(True))


@pytest.mark.runs
def test_fr3_warm_starts_into_graph_s_with_the_dice_input():
    """The experiment's start: fr3 (ppo-2p-fr3/full@e4350) in graph-s with the input plays as fr3."""
    from catanrl.runs import RunError, load_checkpoint

    try:
        fr3 = load_checkpoint("ppo-2p-fr3/full@e4350#43eca7ffdd41")
    except (RunError, OSError, ValueError):
        pytest.skip("ppo-2p-fr3/full@e4350 is not in the runs directory")
    source = fr3.model("all")
    assert source.levels == {"level": "full", "critic_level": "full"} and not fr3.dice_odds
    torch.manual_seed(0)
    target = build_model(fr3.header["model"], dims=fr3.dims, levels={"level": "full", "dice_odds": True},
                         num_actions=action_size(fr3.dims), layout=row_layout(fr3.dims, "full", None, True)).eval()
    ck.load_parts(target, {"ckpt": "ppo-2p-fr3/full@e4350#43eca7ffdd41", "parts": ["actor", "critic"],
                           "zero_new_inputs": True})
    assert_plays_alike(source, target, real_rows(False), real_rows(True))


# ---------------------------------------------------------------- consumers


@pytest.fixture
def dice_ckpt(tmp_path):
    return saved(tmp_path, model_of(True, seed=7))


@pytest.fixture
def plain_ckpt(tmp_path):
    return saved(tmp_path / "plain", model_of(False, seed=8))


def native(spec: str):
    return make_native_agent(resolve_agent(spec, games=[GAME]), device="cpu", infer_rows=4)


class Spy:
    """Wraps an evaluator: records every row batch it is asked for."""

    def __init__(self, evaluator) -> None:
        self.inner, self.rows = evaluator, []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __call__(self, rows):
        self.rows.append(np.array(rows))
        return self.inner(rows)


def test_policy_belief_and_mcts_read_the_input_from_the_checkpoint(dice_ckpt):
    names = feature_names(DIMS, "full", True)["global"]
    first = names.index(f"{DICE_PREFIX}0_2")
    game, seat = balanced_games(1, 300, 300)[0]
    expected = np.concatenate([np.array(game.roll_probabilities((seat + k) % 2), dtype=np.float32)
                               for k in range(2)])
    for spec in (f"POLICY:ckpt={dice_ckpt.path}", f"BELIEF:ckpt={dice_ckpt.path}:worlds=3",
                 f"MCTS:ckpt={dice_ckpt.path}:sims=8:dets=1:kernels=torch", f"MCTS:ckpt={dice_ckpt.path}:sims=8:oracle=true"):
        agent = native(spec)
        assert agent.dice_odds and agent.encoder.dice_odds and agent.model.levels["dice_odds"], spec
        row = agent.encode(game, seat)
        assert row.shape == (agent.model.obs_size,) and np.array_equal(row[first: first + 22], expected)
        spy = Spy(agent.evaluator)
        agent.evaluator = spy
        tracker = cpp.HandTracker(game, seat)
        agent.act(row[None], row[None, -agent.action_space.size:], np.array([5], dtype=np.uint64),
                  games=[game.copy()], trackers=[tracker])
        seen = np.concatenate(spy.rows)
        assert seen.shape[1] == agent.model.obs_size and len(seen)
        if agent.key == "POLICY":
            assert np.array_equal(seen[0], row)
        if agent.key == "BELIEF":  # every world is a copy of the game: its dice state too
            assert all(np.array_equal(r[first: first + 22], expected) for r in seen)
        if agent.key == "MCTS":
            assert agent.pool.observation_size == agent.model.obs_size
            assert np.array_equal(seen[0][first: first + 22], expected)  # the root's leaf
            odds = seen[:, first: first + 22].reshape(-1, 2, 11)
            assert np.allclose(odds.sum(-1), 1.0, atol=1e-5)  # every leaf reads its own state's odds
            assert len({o.tobytes() for o in odds}) > 1  # leaves after a roll moved the state on


def test_predict_and_trade_refuse_a_dice_network(dice_ckpt):
    from catanatron.registry import SpecError

    with pytest.raises(SpecError, match="dice input"):
        resolve_agent(f"TRADE:ckpt={dice_ckpt.path}", games=[GAME])


@pytest.mark.parametrize("dice_first", [True, False])
def test_the_harness_plays_a_dice_network_against_one_without(dice_ckpt, plain_ckpt, dice_first):
    """Either slot: the VecEnv encodes for slot 0's network and the other re-encodes from the
    games; every row an agent decides on is its own encoder's row of that game."""
    from catanrl.eval.harness import play_schedule
    from catanrl.eval.suites import load_suite

    specs = [f"POLICY:ckpt={dice_ckpt.path}", f"POLICY:ckpt={plain_ckpt.path}"]
    if not dice_first:
        specs.reverse()
    agents = [native(s) for s in specs]
    checked = []
    for agent in agents:
        agent.needs_games = True  # hand it the games, to compare its rows with them
        act = agent.act

        def spy(rows, masks, seeds, *, games=None, _agent=agent, _act=act, **kwargs):
            for row, game in zip(rows, games):
                assert np.array_equal(row, _agent.encode(game, game.current_player_index))
                checked.append(_agent.dice_odds)
            return _act(rows, masks, seeds, games=games, **kwargs)

        agent.act = spy
    suite = dataclasses.replace(load_suite("select-2p-fr"), boards=1)
    games = suite.schedule(2)
    rows = []
    play_schedule(agents, games, suite=suite, device="cpu", num_envs=2, threads=1, infer_rows=4,
                  on_game=rows.append)
    assert len(rows) == 2 and True in checked and False in checked


def test_the_fused_kernels_fold_the_dice_inputs_into_the_global_projection(monkeypatch):
    """kernels=fused reads the raw [global | players] span through weights folded from the
    canonical ones: with the input, its global projection is the module's (CPU, stub kernels)."""
    from catanrl.models import graph_kernels

    monkeypatch.setattr(graph_kernels, "kernels", lambda: SimpleNamespace(hidden=96))
    torch.manual_seed(0)
    model = build_model({"preset": "graph-s", "args": {"shared": True}}, dims=DIMS,
                        levels={"level": "full", "dice_odds": True}, num_actions=action_size(DIMS),
                        layout=row_layout(DIMS, "full", None, True)).eval()
    with torch.no_grad():
        model.backbone.dice_projection.weight.normal_()  # zero at initialization
    fused = graph_kernels.FusedGraph(model)
    rows = real_rows(True)
    trunk = model.backbone
    with torch.no_grad():
        r = trunk.reader(rows)
        plain = r.glob[:, : trunk.plain_globals]
        want = (trunk.global_projection(torch.cat((plain, r.players.flatten(1)), dim=1))
                + trunk.dice_projection(r.glob[:, trunk.plain_globals:]))
        got = torch.addmm(fused.bg, rows[:, fused.g_span[0]: fused.g_span[1]], fused.wgT)
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-4)


# ---------------------------------------------------------------- training


def test_the_training_env_writes_rows_with_the_input():
    from catanrl.envs.puffer import make_vecenv, plan_env

    plan = plan_env({"scenarios": [{"game": GAME, "maps": 2}], "num_envs": 2}, pool_seed=0)
    vec = make_vecenv(plan, num_envs=2, level="full", critic_level=None, threads=1, seed=1, dice_odds=True)
    env = vec.envs[0]
    assert env.vec.dice_odds and env.single_observation_space.shape[0] == model_of(True).obs_size
    env.reset(1)
    encoder = make_encoder(plan.dims, "full", True)
    for row in np.flatnonzero(env.acting):
        game = env.vec.game(int(row) // 2)
        assert np.array_equal(env.observations[row], encoder.encode(game, int(row) % 2))
    vec.close()


def test_a_tiny_run_from_a_plain_checkpoint_starts_as_it_and_trains_the_input(tmp_path, monkeypatch):
    from run_helpers import train
    from train_configs import tiny_ppo

    from catanrl.train import dice_probe
    from catanrl.train.dice_probe import STATE_KEY
    from catanrl.train.puffer import train_ppo

    source = model_of(False, seed=9)
    ref = saved(tmp_path, source)
    config = tiny_ppo(seed=4, epochs=4, critic_targets="lambda", critic_gamma=1.0)
    config["env"]["scenarios"] = [{"game": GAME, "maps": 4}]
    config["obs"] = {"level": "full", "dice_odds": True}
    config["model"] = {"preset": "graph-tiny", "args": {"shared": True}}
    config["init_from"] = {"ckpt": str(ref.path), "parts": ["actor", "critic"], "zero_new_inputs": True}
    monkeypatch.setattr(dice_probe, "PROBE_EPOCH", 2)  # the batch from epoch 2's rollout (default 10)
    writer = train(train_ppo, config, "dice")
    assert writer.status == "finished"
    e0 = ck.Checkpoint.load(writer.saved[0].path)
    assert e0.dice_odds
    assert_plays_alike(source, e0.model("all"), real_rows(False), real_rows(True))
    rows = [r for r in writer.rows if "dice/weight_norm" in r]
    assert len(rows) == 4 and [("dice/probe_kl" in r) for r in rows] == [False, True, True, True]
    probed = rows[1:]
    assert all(r["dice/probe_rows"] == probed[0]["dice/probe_rows"] > 0 for r in probed)
    assert 0.0 < rows[0]["dice/weight_norm"] < rows[-1]["dice/weight_norm"]
    assert all(np.isfinite(r["dice/probe_kl"]) and r["dice/probe_value_gap"] > 0.0 for r in probed)
    state = torch.load(writer.path / "trainer_state.pt", weights_only=False)
    assert state[STATE_KEY].shape == (probed[0]["dice/probe_rows"], model_of(True).obs_size)
