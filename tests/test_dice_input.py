"""The dice input (``obs.dice_state``): the balanced dice state in the global block and on each
tile (:data:`catanrl.envs.encoding.DICE_STATE`), seats relative to the observer.

* the features against the game's ``BalancedDice`` on real balanced games, in observer order,
  and the constants of random dice;
* off, rows (against digests from main's build), layouts, levels, checkpoint headers and
  config hashes are as before;
* the config key and the checkpoint's ``obs.dice_state`` round-trip, and the self-test pins
  declared balanced dice states;
* ``init_from.zero_new_inputs`` copies every weight and zeroes only the new inputs', so the
  warm-started network gives its source's logits and values on real rows, bit for bit;
* POLICY, BELIEF and MCTS (torch kernels) build their rows, worlds and leaves with the input
  from the checkpoint alone and, warm-started, choose as their source; the harness plays a
  dice network against one without it in either slot; the fused kernels keep the plain
  projections' GEMMs as the source's and read the dice features through two of their own
  (CPU, stub kernels; the whole forward on the device evaluators is marked ``gpu``);
* the training env's rows carry it, a tiny PPO run from a checkpoint without it logs
  ``dice/*``, starts as its source and moves the dice weights off zero, and the probe's fair
  rows are a fresh state's (tests/test_alphazero_dice.py has the AlphaZero trainer).
"""

from __future__ import annotations

import copy
import dataclasses
import json
import random
import shutil
from types import SimpleNamespace

import cppanatron as cpp
import numpy as np
import pytest
import torch
from dice_features import DiceColumns, global_names, tile_names

from catanrl import checkpoint as ck
from catanrl.agents.native import make_native_agent
from catanrl.agents.spec import resolve_agent
from catanrl.config import ConfigError, config_hash, resolve
from catanrl.envs.encoding import (
    DICE_ODDS_SCALE,
    DICE_PREFIX,
    FAIR_ODDS,
    action_size,
    dims_for,
    fresh_dice_features,
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
    levels = {"level": level, **({"dice_state": True} if dice else {})}
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


# ---------------------------------------------------------------- the features


def test_the_features_are_the_games_balanced_dice_in_observer_order():
    encoder = make_encoder(DIMS, "full", True)
    plain = make_encoder(DIMS, "full")
    columns = DiceColumns(encoder)
    assert len(columns.global_dice) == 14 * 2 + 23 == 51 and len(columns.tile_dice) == 2 + 2
    assert encoder.size == plain.size + 51 + 19 * 4 == plain.size + 127
    differ = unfair = aged = sevens = window = 0
    for game, _ in balanced_games():
        dice = game.balanced_dice
        for observer in range(2):
            row = encoder.encode(game, observer)
            columns.check(game, observer, row)
            glob = dict(zip(columns.global_dice, columns.glob(row)))
            own = np.array(game.roll_probabilities(observer), dtype=np.float32)
            other = np.array(game.roll_probabilities(1 - observer), dtype=np.float32)
            odds = np.array([[glob[f"dice_odds_{k}_{t}"] for t in range(2, 13)] for k in range(2)])
            assert np.allclose(odds[0], own * DICE_ODDS_SCALE) and np.allclose(odds[1], other * DICE_ODDS_SCALE)
            assert abs(float(odds[0].sum()) - DICE_ODDS_SCALE) < 1e-4  # slot 0 is the observer
            assert glob["dice_sevens_0"] == np.float32(min(dice.sevens[observer] / 10.0, 1.0))
            assert all(0.0 <= value <= 2.5 for value in columns.all(row))  # of order 1
            differ += not np.array_equal(own, other)
            unfair += not np.allclose(own, FAIR_ODDS)
            aged += glob["dice_draws_left"] < 1.0
            sevens += glob["dice_seven_share_0"] > 0
            window += max(glob[f"dice_recent_{t}"] for t in range(2, 13)) > 0.5
    # the seats' odds differ (seven balancing) and are not fair; decks age, windows repeat
    assert min(differ, unfair, aged, sevens, window) > 0


def test_canonical_names_pad_six_seats():
    canonical, plain = canonical_feature_names(dice_state=True), canonical_feature_names()
    assert canonical["global"] == plain["global"] + global_names(6) and len(global_names(6)) == 14 * 6 + 23
    assert canonical["tiles"] == plain["tiles"] + tile_names(6) and len(tile_names(6)) == 8
    assert all(canonical[b] == plain[b] for b in plain if b not in ("global", "tiles"))


@pytest.mark.parametrize("level", ["private", "full"])
def test_random_dice_read_a_fresh_state_at_every_roll(level):
    """Under random dice the dice features are constants: the fair odds for every seat, a full
    deck, an empty window, no sevens; a tile's odds are its pips feature."""
    encoder = make_encoder(DIMS, level, True)
    columns = DiceColumns(encoder)
    game = load_game("base-2p").cppanatron_game(1, 2)  # random dice
    assert game.balanced_dice is None
    rng = random.Random(0)
    fresh = np.array(fresh_dice_features(columns.global_names, 2)[-51:], dtype=np.float32)
    assert fresh_dice_features(columns.global_names, 2)[:-51] == [None] * (len(columns.global_names) - 51)
    seen = set()
    pips = columns.tile_feature_names.index("pips")
    for step in range(300):
        if game.winning_color is not None:
            break
        if step % 17 == 0:
            for observer in range(2):
                row = encoder.encode(game, observer)
                columns.check(game, observer, row)
                assert np.array_equal(columns.glob(row), fresh)
                tiles = row[columns.t0: columns.t0 + columns.num_tiles * columns.tile_width].reshape(
                    columns.num_tiles, columns.tile_width)
                numbered = np.array(columns.tile_numbers(row)) > 0
                assert np.array_equal(columns.tiles(row)[:, 0], tiles[:, pips])  # the fair odds: the pips
                assert np.array_equal(columns.tiles(row)[:, 1], tiles[:, pips])
                assert np.array_equal(columns.tiles(row)[:, 2], numbered.astype(np.float32))
                assert not columns.tiles(row)[:, 3].any()
                seen.add(columns.all(row).tobytes())
        game.execute(rng.choice(game.playable_actions))
    assert len(seen) == 1
    assert np.allclose(fresh[:22], np.tile(np.array(FAIR_ODDS) * DICE_ODDS_SCALE, 2))


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
    assert normalize_levels({"level": "full", "dice_state": False}) == {"level": "full", "critic_level": "full"}
    plain, dice = model_of(False), model_of(True)  # one seed: every shared weight drawn alike
    theirs, mine = plain.state_dict(), dice.state_dict()
    new = {"backbone.dice_projection.weight", "backbone.tile_dice_projection.weight"}
    assert set(mine) - set(theirs) == new and set(theirs) <= set(mine)
    assert all(torch.equal(mine[k], theirs[k]) for k in theirs)
    assert mine["backbone.dice_projection.weight"].shape == (theirs["backbone.global_projection.bias"].shape[0], 107)
    assert mine["backbone.tile_dice_projection.weight"].shape == (theirs["backbone.tile_projection.bias"].shape[0], 8)
    assert not any(mine[k].any() for k in new)  # zero at initialization
    trunk = plain.backbone
    assert trunk.dice_projection is None and trunk.tile_dice_projection is None
    assert trunk.plain_globals == len(canonical_feature_names()["global"])
    assert trunk.plain_tiles == len(canonical_feature_names()["tiles"]) == dice.backbone.plain_tiles
    assert (dice.backbone.dice_inputs, dice.backbone.tile_dice_inputs) == (107, 8)


#: sha256 (16 hex) of the rows without the input along seeded random games, from main's build
#: before the option existed (3f34b94, engine b926970): balanced dice and random, 2 and 4 seats.
MAIN_ROWS = {("base-2p-fr", "private"): "8b1ffd4388e1627f", ("base-2p-fr", "full"): "371789ed77ebcde1",
             ("base-4p", "private"): "cc1a60fdf5d11ca1", ("base-4p", "full"): "c15e847657ae9d74"}


@pytest.mark.parametrize("name, level", sorted(MAIN_ROWS))
def test_rows_without_the_input_are_mains_byte_for_byte(name, level):
    import hashlib

    config = load_game(name)
    encoder = make_encoder(dims_for([name]), level)
    digest = hashlib.sha256()
    for seed in range(4):
        game = config.cppanatron_game(seed, 100 + seed)
        rng = random.Random(seed)
        for step in range(400):
            if game.winning_color is not None:
                break
            if step % 23 == 7:
                for seat in range(game.num_players):
                    digest.update(encoder.encode(game, seat).tobytes())
            game.execute(rng.choice(game.playable_actions))
    assert digest.hexdigest()[:16] == MAIN_ROWS[name, level]


def test_an_unset_dice_input_keeps_the_config_hash():
    base = {"env": {"scenarios": [{"game": GAME}]}, "model": {"preset": "graph-tiny"}, "train": {"algo": "ppo"},
            "obs": {"level": "full"}}
    plain = resolve(base, experiment="t", variant="v")
    unset = resolve({**base, "obs": {"level": "full", "dice_state": False}}, experiment="t", variant="v")
    on = resolve({**base, "obs": {"level": "full", "dice_state": True}}, experiment="t", variant="v")
    assert "dice_state" not in plain["obs"] and "dice_state" not in unset["obs"] and on["obs"]["dice_state"] is True
    assert config_hash(unset) == config_hash(plain) != config_hash(on)
    with pytest.raises(ConfigError, match="obs.dice_state is true or false"):
        resolve({**base, "obs": {"level": "full", "dice_state": "yes"}}, experiment="t", variant="v")
    with pytest.raises(ConfigError, match="train.algo ppo or alphazero"):
        resolve({**base, "obs": {"level": "full", "dice_state": True}, "train": {"algo": "imitation"}},
                experiment="t", variant="v")
    az = resolve({**base, "obs": {"level": "full", "dice_state": True}, "train": {"algo": "alphazero"}},
                 experiment="t", variant="v")
    assert az["obs"]["dice_state"] is True
    init = {"ckpt": "x/y@e1", "zero_new_inputs": True}
    with pytest.raises(ConfigError, match="set obs.dice_state: true"):
        resolve({**base, "init_from": init}, experiment="t", variant="v")
    with pytest.raises(ConfigError, match="restores weights only"):
        resolve({**base, "obs": {"level": "full", "dice_state": True}, "init_from": {**init, "restore":
                 "weights+optimizer"}}, experiment="t", variant="v")


# ---------------------------------------------------------------- checkpoints


def test_the_checkpoint_stores_the_input_and_its_self_test_declares_dice_states(tmp_path):
    plain_ref, dice_ref = saved(tmp_path / "plain", model_of(False)), saved(tmp_path / "dice", model_of(True))
    plain, dice = ck.Checkpoint.load(plain_ref.path), ck.Checkpoint.load(dice_ref.path)
    assert set(plain.header["obs"]) == {"level", "critic_level", "dims", "layout_hash"} and not plain.dice_state
    assert plain.levels == {"level": "full", "critic_level": "full"}
    assert dice.header["obs"]["dice_state"] is True and dice.dice_state
    assert dice.levels == {"level": "full", "critic_level": "full", "dice_state": True}
    positions = json.loads(dice.header["self_test"]["positions"])
    assert all(p["game"]["dice_mode"] == "balanced" and "dice" in p for p in positions)
    assert len({json.dumps(p["dice"]) for p in positions}) > 4  # states of their own
    assert not any("dice" in p for p in json.loads(plain.header["self_test"]["positions"]))
    model = dice.model("all")
    assert model.levels["dice_state"] and model.obs_size == model_of(True).obs_size
    obs = dice.header["self_test"]["obs"]
    columns = DiceColumns(make_encoder(DIMS, "full", True))
    names = feature_names(DIMS, "full", True)["global"]
    fresh = np.array(fresh_dice_features(names, 2)[-51:], dtype=np.float32)
    states = [columns.glob(row) for row in obs.numpy()]
    assert sum(not np.allclose(state, fresh) for state in states) > 4  # the self-test reads used decks
    assert any(columns.tiles(row)[:, 3].any() for row in obs.numpy())  # and tiles whose number just came
    # A drifted dice feature, global or on a tile, is caught by the self-test.
    tile = next(t for t, number in enumerate(columns.tile_numbers(obs[1].numpy())) if number)
    for name, column in (("global", names.index(f"{DICE_PREFIX}odds_0_7")),
                         ("tile", columns.t0 + (tile + 1) * columns.tile_width - 2)):
        payload = torch.load(dice_ref.path, weights_only=True)
        payload["self_test"]["obs"][1, column] += 0.01
        torch.save(payload, tmp_path / f"drift-{name}.pt")
        with pytest.raises(ck.EncoderDrift):
            ck.Checkpoint.load(tmp_path / f"drift-{name}.pt").model("all")


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
    trunks = ["backbone"] if shared else ["actor.trunk", "critic.trunk"]
    assert new == sorted(ck.dice_input_keys(target)) == sorted(
        f"{trunk}.{layer}.weight" for trunk in trunks for layer in ("dice_projection", "tile_dice_projection"))
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
    assert source.levels == {"level": "full", "critic_level": "full"} and not fr3.dice_state
    torch.manual_seed(0)
    target = build_model(fr3.header["model"], dims=fr3.dims, levels={"level": "full", "dice_state": True},
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


def decide(agent, game, seat, seed=5, spy=True):
    """The agent's choice for ``seat`` in ``game``, with the row batches its evaluator saw
    and what it answered."""
    row = agent.encode(game, seat)
    watch = Spy(agent.evaluator)
    agent.evaluator = watch
    try:
        choice = agent.act(row[None], row[None, -agent.action_space.size:], np.array([seed], dtype=np.uint64),
                           games=[game.copy()], trackers=[cpp.HandTracker(game, seat)])
    finally:
        agent.evaluator = watch.inner
    return row, choice, watch


SEARCHERS = ("POLICY:ckpt={}", "BELIEF:ckpt={}:worlds=3", "MCTS:ckpt={}:sims=8:dets=1:kernels=torch",
             "MCTS:ckpt={}:sims=8:oracle=true")


def test_policy_belief_and_mcts_read_the_input_from_the_checkpoint(dice_ckpt):
    columns = DiceColumns(make_encoder(DIMS, "full", True))
    game, seat = balanced_games(1, 300, 300)[0]
    for spec in (s.format(dice_ckpt.path) for s in SEARCHERS):
        agent = native(spec)
        assert agent.dice_state and agent.encoder.dice_state and agent.model.levels["dice_state"], spec
        row, _, spy = decide(agent, game, seat)
        assert row.shape == (agent.model.obs_size,)
        columns.check(game, seat, row)
        seen = np.concatenate(spy.rows)
        assert seen.shape[1] == agent.model.obs_size and len(seen)
        if agent.key == "POLICY":
            assert np.array_equal(seen[0], row)
        if agent.key == "BELIEF":  # every world is a copy of the game: its dice state too
            assert all(np.array_equal(columns.glob(r), columns.glob(row)) for r in seen)
            assert all(np.array_equal(columns.tiles(r), columns.tiles(row)) for r in seen)
        if agent.key == "MCTS":
            assert agent.pool.observation_size == agent.model.obs_size
            assert np.array_equal(columns.all(seen[0]), columns.all(row))  # the root's leaf
            odds = np.stack([columns.glob(r)[:22] for r in seen]).reshape(-1, 2, 11)
            assert np.allclose(odds.sum(-1), DICE_ODDS_SCALE, atol=1e-4)  # every leaf reads its own state
            assert len({columns.all(r).tobytes() for r in seen}) > 1  # leaves after a roll moved the state on
            # a leaf's tile features are its own state's: each numbered tile's odds are that number's
            for r in seen:
                glob = dict(zip(columns.global_dice, columns.glob(r)))
                for number, tile in zip(columns.tile_numbers(r), columns.tiles(r)):
                    want = [glob[f"dice_odds_0_{number}"] * 1.2, glob[f"dice_deck_{number}"]] if number else [0, 0]
                    assert np.allclose([tile[0], tile[2]], want, atol=1e-5)


@pytest.mark.parametrize("spec", SEARCHERS)
def test_a_warm_started_network_searches_as_its_source(tmp_path, spec):
    """BELIEF and MCTS on a network warm-started with zero dice weights: the evaluator answers
    its source's logits and values bit for bit on every world and leaf, so the search visits
    the same nodes and chooses the same action."""
    source = model_of(False, seed=11)
    plain_ref = saved(tmp_path / "plain", source)
    target = model_of(True, seed=12)
    ck.load_parts(target, {"ckpt": str(plain_ref.path), "parts": ["actor", "critic"], "zero_new_inputs": True})
    dice_ref = saved(tmp_path / "dice", target)
    plain, dice = native(spec.format(plain_ref.path)), native(spec.format(dice_ref.path))
    assert dice.dice_state and not plain.dice_state
    answers = 0
    for game, seat in balanced_games(3, 400, 37):
        if len(game.playable_actions) < 2:
            continue
        (_, a, spy_a), (_, b, spy_b) = decide(plain, game, seat), decide(dice, game, seat)
        assert np.array_equal(a, b), spec
        assert len(spy_a.rows) == len(spy_b.rows) > 0
        for rows_a, rows_b in zip(spy_a.rows, spy_b.rows):
            assert rows_a.shape[0] == rows_b.shape[0] and rows_b.shape[1] == rows_a.shape[1] + 127
            for out_a, out_b in zip(plain.evaluator(rows_a), dice.evaluator(rows_b)):
                assert np.array_equal(np.asarray(out_a), np.asarray(out_b))
            answers += len(rows_a)
    assert answers >= 8


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
                checked.append(_agent.dice_state)
            return _act(rows, masks, seeds, games=games, **kwargs)

        agent.act = spy
    suite = dataclasses.replace(load_suite("select-2p-fr"), boards=1)
    games = suite.schedule(2)
    rows = []
    play_schedule(agents, games, suite=suite, device="cpu", num_envs=2, threads=1, infer_rows=4,
                  on_game=rows.append)
    assert len(rows) == 2 and True in checked and False in checked


def first_projections(fused, rows: torch.Tensor):
    """FusedGraph's first projections on CPU rows (the topology kernel's contiguous copies of
    the raw entity blocks made here)."""
    block = lambda b: rows[:, fused.spans[b][0]: fused.spans[b][0] + fused.spans[b][1][0] * fused.spans[b][1][1]
                           ].reshape(-1, fused.spans[b][1][1]).contiguous()
    return fused._first(rows, block("nodes"), block("tiles"), block("edges"))


def test_the_fused_kernels_keep_the_plain_gemms_and_read_the_dice_inputs_beside_them(monkeypatch):
    """kernels=fused reads the raw blocks through weights folded from the canonical ones. With
    the input the plain features go through the GEMMs of a network without it (the same
    weights, the same inputs) and the dice features through two of their own: zero dice
    weights give that network's first projections bit for bit, and trained ones the module's.
    (CPU, stub kernels: the arithmetic of the weights; tools/bundle_smoke.py checks the
    whole forward on the GPU.)"""
    from catanrl.models import graph_kernels

    monkeypatch.setattr(graph_kernels, "kernels", lambda: SimpleNamespace(hidden=96))
    plain, model = model_of(False, preset="graph-s"), model_of(True, preset="graph-s")  # one seed
    base, fused = graph_kernels.FusedGraph(plain), graph_kernels.FusedGraph(model)
    assert not base.dice and fused.dice
    assert fused.plain == {**base.plain, "global": base.plain["global"], "tiles": base.plain["tiles"]}
    assert base.plain["tiles"] == fused.spans["tiles"][1][1] - 4
    assert torch.equal(fused.wgT, base.wgT) and torch.equal(fused.bg, base.bg)
    assert all(torch.equal(fused.proj[b][k], base.proj[b][k]) for b in base.proj for k in ("wT", "b", "wiT"))
    assert fused.g_dice.shape == (51, fused.bg.shape[0]) and fused.t_dice.shape == (4, 96)
    assert not fused.g_dice.any() and not fused.t_dice.any()
    rows, plain_rows = real_rows(True), real_rows(False)
    with torch.no_grad():
        (glob, us), (base_glob, base_us) = first_projections(fused, rows), first_projections(base, plain_rows)
    assert torch.equal(glob, base_glob) and all(torch.equal(a, b) for a, b in zip(us, base_us))
    with torch.no_grad():
        model.backbone.dice_projection.weight.normal_()
        model.backbone.tile_dice_projection.weight.normal_()
    fused = graph_kernels.FusedGraph(model)
    trunk = model.backbone
    with torch.no_grad():
        glob, us = first_projections(fused, rows)
        r = trunk.reader(rows)
        want = torch.relu(trunk.global_projection(torch.cat((r.glob[:, : trunk.plain_globals], r.players.flatten(1)),
                                                            dim=1))
                          + trunk.dice_projection(r.glob[:, trunk.plain_globals:]))
        tiles = torch.relu(trunk.tile_projection(r.tiles[..., : trunk.plain_tiles])
                           + trunk.tile_dice_projection(r.tiles[..., trunk.plain_tiles:]))
        want_tiles = tiles.reshape(-1, 96) @ trunk.tile_injection[0].weight[:, :96].T
    assert torch.allclose(glob, want, rtol=1e-5, atol=1e-4)
    assert torch.allclose(us[1], want_tiles, rtol=1e-5, atol=1e-4)
    assert not torch.allclose(glob, base_glob) and not torch.allclose(us[1], base_us[1])  # the inputs are read


# ---------------------------------------------------------------- the device evaluators (GPU)

needs_kernels = pytest.mark.skipif(not torch.cuda.is_available() or shutil.which("nvcc") is None,
                                   reason="needs CUDA and nvcc (the kernels build on first use)")


def trained_scale(model):
    """Random init leaves the logits near zero, where errors hide (tests/test_graph_kernels.py)."""
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(3.0)
    return model


def evaluate(model, rows, kernels: str, mode: str, device: str = "cuda"):
    from catanrl.search import BatchEvaluator

    return BatchEvaluator(copy.deepcopy(model), device=device, infer_rows=16, mode=mode, kernels=kernels)(rows.numpy())


def same_on_device(a, b, kernels: str) -> bool:
    """Two (logits, values) results of one network read at two row widths: equal bit for bit on
    the fused kernels; on ``kernels=torch`` within rounding, with every row's best action the same."""
    (la, va), (lb, vb) = (tuple(np.asarray(x, dtype=np.float32) for x in r) for r in (a, b))
    if kernels != "torch":
        return bool(np.array_equal(la, lb) and np.array_equal(va, vb))
    return bool(np.allclose(la, lb, rtol=1e-5, atol=1e-4) and np.allclose(va, vb, atol=1e-5)
                and (la.argmax(1) == lb.argmax(1)).all())


def warm_started(tmp_path, preset: str = "graph-s"):
    """(a network without the input, the one with it warm-started from it)."""
    source = trained_scale(model_of(False, seed=3, preset=preset))
    ref = saved(tmp_path, source)
    target = model_of(True, seed=4, preset=preset)
    ck.load_parts(target, {"ckpt": str(ref.path), "parts": ["actor", "critic"], "zero_new_inputs": True})
    return source, target


@pytest.mark.gpu
@needs_kernels
@pytest.mark.parametrize("kernels, mode", [("torch", "strict"), ("torch", "fast"), ("fused", "strict"),
                                           ("fused", "fast"), ("fused16", "strict"), ("fused16", "fast")])
def test_a_warm_started_network_is_its_source_on_the_device_evaluators(tmp_path, kernels, mode):
    """The run's e0 (init_from.zero_new_inputs) on the GPU: the module, the fused kernels and
    their half-precision variant give the source's logits and values: the kernels bit for bit,
    the module to rounding (cuBLAS rounds a product of the wider rows' slices differently: 1e-5
    on these logits, measured on the RTX 2070 SUPER)."""
    source, target = warm_started(tmp_path)
    was, now = evaluate(source, real_rows(False), kernels, mode), evaluate(target, real_rows(True), kernels, mode)
    assert same_on_device(was, now, kernels)


@pytest.mark.gpu
@needs_kernels
@pytest.mark.parametrize("mode", ["strict", "fast"])
def test_the_fused_kernels_with_trained_dice_weights_match_the_module(tmp_path, mode):
    """tests/test_graph_kernels.py's bounds, for a network that reads the dice input."""
    _, model = warm_started(tmp_path)
    zero = evaluate(model, real_rows(True), "fused", mode)
    with torch.no_grad():
        for key in ck.dice_input_keys(model):
            model.state_dict()[key].normal_(std=0.1)
    rows = real_rows(True)
    mask = rows.numpy()[:, model.mask_offset: model.mask_offset + model.num_actions] > 0
    want = evaluate(model, rows, "torch", mode)
    scale = max(1.0, float(np.abs(want[0][mask]).max()))
    got = evaluate(model, rows, "fused", mode)
    assert np.array_equal(got[0] > -1e8, mask) and np.all(got[0][~mask] == want[0][~mask])
    assert np.abs(got[0][mask] - want[0][mask]).max() <= 2e-6 * scale + 1e-5
    assert np.abs(got[1] - want[1]).max() <= 1e-5
    assert np.abs(got[1] - zero[1]).max() > 1e-4  # the kernels read the input
    half = evaluate(model, rows, "fused16", mode)
    assert np.isfinite(half[0]).all() and np.isfinite(half[1]).all()
    assert np.abs(half[0][mask] - got[0][mask]).max() <= 4e-3 * scale
    assert np.abs(half[1] - got[1]).max() <= 2e-3


# ---------------------------------------------------------------- training


def test_the_training_env_writes_rows_with_the_input():
    from catanrl.envs.puffer import make_vecenv, plan_env

    plan = plan_env({"scenarios": [{"game": GAME, "maps": 2}], "num_envs": 2}, pool_seed=0)
    vec = make_vecenv(plan, num_envs=2, level="full", critic_level=None, threads=1, seed=1, dice_state=True)
    env = vec.envs[0]
    assert env.vec.dice_state and env.single_observation_space.shape[0] == model_of(True).obs_size
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
    config["obs"] = {"level": "full", "dice_state": True}
    config["model"] = {"preset": "graph-tiny", "args": {"shared": True}}
    config["init_from"] = {"ckpt": str(ref.path), "parts": ["actor", "critic"], "zero_new_inputs": True}
    monkeypatch.setattr(dice_probe, "PROBE_EPOCH", 2)  # the batch from epoch 2's rollout (default 10)
    writer = train(train_ppo, config, "dice")
    assert writer.status == "finished"
    e0 = ck.Checkpoint.load(writer.saved[0].path)
    assert e0.dice_state
    assert_plays_alike(source, e0.model("all"), real_rows(False), real_rows(True))
    rows = [r for r in writer.rows if "dice/weight_norm" in r]
    assert len(rows) == 4 and [("dice/probe_kl" in r) for r in rows] == [False, True, True, True]
    probed = rows[1:]
    assert all(r["dice/probe_rows"] == probed[0]["dice/probe_rows"] > 0 for r in probed)
    assert 0.0 < rows[0]["dice/weight_norm"] < rows[-1]["dice/weight_norm"]
    assert all(np.isfinite(r["dice/probe_kl"]) and r["dice/probe_value_gap"] > 0.0 for r in probed)
    state = torch.load(writer.path / "trainer_state.pt", weights_only=False)
    assert state[STATE_KEY].shape == (probed[0]["dice/probe_rows"], model_of(True).obs_size)


@pytest.mark.parametrize("game_name", ["base-2p-fr", "base-2p"])
def test_the_probes_fair_rows_are_a_fresh_states(game_name):
    """The probe's counterfactual: every dice feature as a fresh balanced state shows it (the
    row a random-dice game gives), everything else untouched; seat slots a game leaves empty
    stay zero."""
    from catanrl.train.dice_probe import FairDice, dice_weight_norms

    dims = dims_for(["base-4p"])
    torch.manual_seed(0)
    model = build_model({"preset": "graph-tiny", "args": {"shared": True}}, dims=dims,
                        levels={"level": "full", "dice_state": True}, num_actions=action_size(dims),
                        layout=row_layout(dims, "full", None, True)).eval()
    assert dice_weight_norms(model) == {"dice/weight_norm": 0.0, "dice/weight_norm_global": 0.0,
                                        "dice/weight_norm_tiles": 0.0}
    encoder = make_encoder(dims, "full", True)
    columns = DiceColumns(encoder)
    fair = FairDice(model)
    dice_columns = set(fair.columns)
    assert len(dice_columns) == len(columns.global_dice) + columns.num_tiles * len(columns.tile_dice)
    game = load_game(game_name).cppanatron_game(3, 4)
    rng = random.Random(1)
    fresh = np.array(fresh_dice_features(columns.global_names, 2)[-len(columns.global_dice):], dtype=np.float32)
    assert not fresh[22:44].any()  # the two empty slots' odds
    moved = 0
    for step in range(400):
        if game.winning_color is not None:
            break
        if step % 29 == 5:
            row = torch.from_numpy(encoder.encode(game, game.current_player_index))[None]
            out = fair.rows(row)
            assert np.array_equal(columns.glob(out[0].numpy()), fresh)
            tiles = columns.tiles(out[0].numpy())
            numbered = np.array(columns.tile_numbers(row[0].numpy())) > 0
            pips = row[0].numpy()[columns.t0: columns.t0 + columns.num_tiles * columns.tile_width].reshape(
                columns.num_tiles, columns.tile_width)[:, columns.tile_feature_names.index("pips")]
            assert np.array_equal(tiles[:, 0], pips) and np.array_equal(tiles[:, 1], pips)
            assert not tiles[:, 2:4].any()  # seats 2 and 3 are empty
            assert np.array_equal(tiles[:, 4], numbered.astype(np.float32)) and not tiles[:, 5].any()
            others = [c for c in range(row.shape[1]) if c not in dice_columns]
            assert torch.equal(out[:, others], row[:, others])
            if game_name == "base-2p":
                assert torch.equal(out, row)  # random dice: the row is already a fresh state's
            moved += not torch.equal(out, row)
        game.execute(rng.choice(game.playable_actions))
    assert (moved > 0) == (game_name == "base-2p-fr")
