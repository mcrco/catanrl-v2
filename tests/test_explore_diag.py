"""tools/explore_diag.py: positions of the trainer's own self-play that restore exactly, search arms
whose prior is reshaped at every node, a referee with common random numbers that judges a move
once, and the report's math."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cppanatron as cpp
import numpy as np
import pytest
import torch
from test_dice_input import DIMS, GAME, model_of

import catanrl.checkpoint as ck
from catanrl.envs.encoding import make_encoder
from catanrl.games import load_game
from catanrl.search.evaluator import BatchEvaluator
from catanrl.search.pool import make_pool, run_pools, search_config
from catanrl.seeding import derive_seed
from catanrl.train.alphazero import ALPHAZERO_KEYS

spec = importlib.util.spec_from_file_location(
    "explore_diag", Path(__file__).resolve().parents[1] / "tools" / "explore_diag.py")
diag = importlib.util.module_from_spec(spec)
sys.modules["explore_diag"] = diag  # its dataclasses look their module up by name
spec.loader.exec_module(diag)

TRAIN = {**ALPHAZERO_KEYS, "root": "puct", "policy_target": "pruned", "sims": 16, "fast_sims": 4, "pending": 4,
         "fast_pending": 2, "full_search_prob": 0.5, "dets": 0, "reuse": True, "steps_per_iteration": 16,
         "search_rows": 64}
ENV = {"scenarios": [{"game": GAME, "turn_limit": 300, "maps": 8}], "num_envs": 4}
CONFIG = {"env": ENV, "seed": 42, "train": TRAIN}
SEP = diag.SEP


@pytest.fixture(autouse=True)
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CATANRL_RUNS_DIR", str(tmp_path / "runs"))
    torch.set_num_threads(2)


def saved(path: Path, model) -> str:
    ref = ck.save(path / "e000000.pt", model, run=None, epoch=0, agent_steps=0,
                  config={**CONFIG, "model": model.spec}, code={})
    return str(ref.path)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory) -> str:
    return saved(tmp_path_factory.mktemp("ckpt"), model_of(True, seed=4))


@pytest.fixture(scope="module")
def setup(ckpt):
    return diag.load_setup(ckpt, device="cpu", pool_seed=None)


@pytest.fixture(scope="module")
def gathered(setup):
    return diag.gather(setup, games=4, envs=4, per_game=3, seed=5, threads=2, log=lambda _: None)


def same(a: np.ndarray, b: np.ndarray) -> bool:
    return np.array_equal(a, b, equal_nan=a.dtype.kind == "f")


# ---------------------------------------------------------------- arms and priors


def test_arms_parse_and_the_grid_names_its_cells():
    arm = diag.parse_arm("wide:visits=9600,temp=1.5,mix=0.25,noise=0,seed=7")
    assert arm == diag.Arm("wide", 9600, temp=1.5, mix=0.25, noise=False, seed=7) and not arm.plain
    assert diag.parse_arm("x:visits=5") == diag.Arm("x", 5) and diag.Arm("x", 5).plain
    for bad in ("x", "x:visits=5,depth=2", "a__b:visits=5", "x:visits=0", "x:visits=5,mix=1", "x:visits=5,temp=0"):
        with pytest.raises(ValueError):
            diag.parse_arm(bad)
    a1, a2, c, b = diag.default_arms(600, wide_visits=9600, wide_mix=0.25, wide_temp=1.0)
    assert (a1.name, a2.name, c.name, b.name) == ("A1", "A2", "C", "B")
    assert a1.visits == a2.visits == 600 and a1.seed != a2.seed and a1.noise and a2.noise and a1.plain
    assert (c.visits, c.plain, c.noise) == (9600, True, True) and (b.visits, b.mix, b.noise) == (9600, 0.25, True)
    grid = diag.grid_arms([600, 16384], [1, 1.5])
    assert [g.name for g in grid] == ["g600_t1", "g16384_t1", "g600_t1.5", "g16384_t1.5"]
    assert not any(g.noise for g in grid) and grid[3].temp == 1.5 and grid[3].visits == 16384
    assert diag.grid_arms([600], [2], mix=0.1)[0].name == "g600_t2_m0.1"


def test_reshaped_logits_give_the_tempered_prior_mixed_with_uniform():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 4, (6, 9)).astype(np.float32)
    legal = rng.random((6, 9)) < 0.6
    legal[:, 0] = True
    legal[4] = False  # no legal move (never a leaf): left alone
    legal[5] = [True] + [False] * 8  # a single legal move

    def softmax(z, mask):
        z = np.where(mask, z.astype(np.float64), -np.inf)
        p = np.exp(z - z.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True)

    rows = legal.any(1)

    for temp, mix in ((1.0, 0.0), (2.0, 0.0), (1.0, 0.25), (1.5, 0.1)):
        shaped = diag.reshape_logits(logits, legal, temp=temp, mix=mix)
        want = (1 - mix) * softmax(logits[rows] / temp, legal[rows]) + mix * legal[rows] / legal[rows].sum(1, keepdims=True)
        assert shaped.dtype == np.float32 and np.array_equal(shaped[~legal], logits[~legal])
        assert np.allclose(softmax(shaped[rows], legal[rows]), want, atol=1e-6)
    # a legal move the network rules out entirely still gets the uniform share
    steep = np.array([[0.0, -500.0, 0.0]], dtype=np.float32)
    p = softmax(diag.reshape_logits(steep, np.ones((1, 3), bool), mix=0.3), np.ones((1, 3), bool))
    assert np.allclose(p, [[0.45, 0.1, 0.45]], atol=1e-6)


def test_the_entropy_is_per_position_over_flat_legal_lists():
    start = np.array([0, 2, 5])
    p = np.array([0.5, 0.5, 1.0, 0.0, 0.0])
    assert np.allclose(diag.entropy(p, start), [np.log(2), 0.0])


def test_an_arms_prior_is_the_trees_prior_at_the_root_and_at_an_inner_node():
    model = model_of(True, seed=4)
    encoder = make_encoder(DIMS, "full", True)
    space = encoder.action_space
    game = load_game(GAME).cppanatron_game(3, 103)  # the first settlement: the next node is its road
    mask = slice(model.mask_offset, model.mask_offset + model.num_actions)
    config = search_config({"sims": 96, "dets": 0, "oracle": True, "root": "puct", "c_puct": 0.75, "pending": 4,
                            "noise": True}, max_players=2, threads=1)
    evaluator = BatchEvaluator(model, device="cpu", infer_rows=64, mode="strict")
    temp, mix = 2.0, 0.25

    def expected(g):
        row = encoder.encode(g, int(g.current_player_index)).astype(np.float32)[None]
        legal = row[:, mask] > 0
        logits = np.where(legal, evaluator(row)[0].astype(np.float64) / temp, -np.inf)
        p = np.exp(logits - logits.max())
        return ((1 - mix) * p / p.sum() + mix * legal / legal.sum())[0]

    pool = make_pool(DIMS, "full", config, dice_state=True)
    shaped = diag.Shaped(evaluator, mask, temp=temp, mix=mix)
    sid = pool.add(game, seed=1)
    leaves = run_pools([pool], shaped, batch_rows=[4])
    assert shaped.rows == leaves > 0
    assert np.allclose(pool.root_priors(sid)[0], expected(game), atol=1e-6)  # before the root's noise
    move = int(np.argmax(pool.visits(sid)))
    child = game.copy()
    child.execute(space.decode(child, move))
    assert pool.reroot(sid, move, simulations=96)
    assert np.allclose(pool.root_priors(sid)[0], expected(child), atol=1e-6)
    # the plain arm leaves the network's logits alone
    plain = diag.Shaped(evaluator, mask)
    row = encoder.encode(game, int(game.current_player_index)).astype(np.float32)[None]
    assert np.array_equal(plain(row)[0], evaluator(row)[0])
    assert (plain.obs_size, plain.num_actions) == (evaluator.obs_size, evaluator.num_actions)


# ---------------------------------------------------------------- positions


def test_positions_are_the_trainers_self_play_and_restore_exactly(setup, gathered):
    data, originals = gathered
    count = len(data["game"])
    assert len(data["g_seed"]) == 4 and 4 <= count <= 12 and len(originals) == count
    assert setup.pools["pool"] != setup.pools["train_pool"]  # boards outside the training pool
    per_game = np.bincount(data["game"], minlength=4)
    lengths = np.diff(data["g_start"])
    assert (per_game == np.minimum(lengths, 3)).all() and (lengths > 0).all()
    assert (data["nlegal"] >= 2).all()  # the VecEnv plays forced moves itself
    for i in range(count):
        g = int(data["game"][i])
        assert int(data["g_action"][data["g_start"][g] + data["ply"][i]]) == int(data["played"][i])
        winner = int(data["g_winner"][g])
        assert int(data["y"][i]) == int(winner == int(data["seat"][i]))
    games = diag.restore_games(setup.plan, setup.levels, data)
    for game, original in zip(games, originals):
        assert game.search_equivalent(original) and original.search_equivalent(game)
    diag.verify_positions(setup, data, originals)
    # a subset in another order restores the same games
    some = diag.restore_games(setup.plan, setup.levels, data, [count - 1, 0])
    assert some[0].search_equivalent(originals[-1]) and some[1].search_equivalent(originals[0])
    # a corrupted action list is caught
    broken = {**data, "g_seed": data["g_seed"] + np.uint64(1)}
    with pytest.raises(RuntimeError):
        diag.verify_positions(setup, broken, originals)


def test_gathering_is_seeded(setup, gathered):
    again, _ = diag.gather(setup, games=4, envs=4, per_game=3, seed=5, threads=2, log=lambda _: None)
    assert all(np.array_equal(gathered[0][key], again[key]) for key in ("g_seed", "g_action", "game", "ply"))


# ---------------------------------------------------------------- search


def test_an_arm_reads_the_trainers_target_off_its_own_tree(setup, gathered):
    data, _ = gathered
    games = diag.restore_games(setup.plan, setup.levels, data)
    facts = diag.root_facts(setup, games)
    start, action = facts["legal_start"], facts["legal_action"]
    assert np.array_equal(np.diff(start), data["nlegal"]) and len(facts["action_names"]) == setup.model.num_actions
    assert set(np.unique(facts["kind"])) <= set(range(len(diag.KINDS)))
    evaluator = BatchEvaluator(setup.model, device="cpu", infer_rows=64, mode="fast")
    run = lambda arm: diag.search_arm(setup, arm, games, start, action, evaluator=evaluator, seed=9,
                                      threads=2, inflight=3, log=lambda _: None)
    quiet, info = run(diag.Arm("quiet", 24, noise=False, seed=1))
    owner = np.repeat(np.arange(len(games)), np.diff(start))
    assert (np.bincount(owner, weights=quiet["visits"]) == 24).all()  # the budget is the root's visits
    assert np.allclose(quiet["prior"], facts["net_p"], atol=1e-5)  # a plain arm searches the network's prior
    assert np.allclose(np.bincount(owner, weights=quiet["target"]), 1.0, atol=1e-5)
    assert info["leaves"] > 0 and abs(info["prior_entropy"] - diag.entropy(facts["net_p"], start).mean()) < 1e-4
    for i in range(len(games)):  # the arm's move: the most visited (the pruned target's argmax)
        span = slice(start[i], start[i + 1])
        assert quiet["visits"][span][np.flatnonzero(action[span] == quiet["chosen"][i])[0]] == quiet["visits"][span].max()
    assert (quiet["tree_bytes"] > 0).all() and (quiet["max_depth"] >= 1).all()
    again, _ = run(diag.Arm("quiet", 24, noise=False, seed=1))
    assert all(np.array_equal(quiet[key], again[key], equal_nan=True) for key in ("visits", "q", "chosen"))
    wide, wide_info = run(diag.Arm("wide", 24, temp=2.0, mix=0.25, noise=False, seed=1))
    tempered = np.exp(facts["logit"].astype(np.float64) / 2.0)
    tempered /= np.bincount(owner, weights=tempered)[owner]
    assert np.allclose(wide["prior"], 0.75 * tempered + 0.25 / np.diff(start)[owner], atol=1e-5)
    assert wide_info["prior_entropy"] > info["prior_entropy"]
    assert diag.inflight_for(16384, tree_gb=6.0, max_inflight=512) == 6 * 2**30 // (16384 * diag.TREE_BYTES_PER_VISIT)
    assert diag.inflight_for(600, tree_gb=6.0, max_inflight=512) == 512


# ---------------------------------------------------------------- referee


def test_a_position_is_judged_once_per_distinct_move():
    arrays = {f"chosen{SEP}A1": np.array([3, 5, 7, 9]), f"chosen{SEP}B": np.array([3, 6, 7, 2]),
              f"chosen{SEP}g": np.array([3, 6, 8, 9])}
    assert diag.arm_names(arrays) == ["A1", "B", "g"]
    pos, move = diag.moves_to_judge(arrays, ["A1", "B", "g"])
    assert list(zip(pos.tolist(), move.tolist())) == [(1, 5), (1, 6), (2, 7), (2, 8), (3, 2), (3, 9)]
    pos, move = diag.moves_to_judge(arrays, ["A1", "B"])
    assert list(zip(pos.tolist(), move.tolist())) == [(1, 5), (1, 6), (3, 2), (3, 9)]


def test_sampling_is_the_inverse_cdf_over_the_legal_moves():
    logits = np.log(np.array([[0.2, 0.5, 0.1, 0.3], [0.2, 0.5, 0.1, 0.3]]))
    legal = np.array([[True, False, True, True], [True, True, False, False]])
    picks = lambda u: diag.sample_rows(logits, legal, np.array([u, u])).tolist()
    assert picks(0.0) == [0, 0] and picks(0.3) == [0, 1] and picks(0.4) == [2, 1] and picks(0.999999) == [3, 1]
    draws = np.random.default_rng(0).random(20000)
    freq = np.bincount(diag.sample_rows(np.repeat(logits[:1], 20000, 0), np.repeat(legal[:1], 20000, 0), draws), minlength=4)
    assert np.allclose(freq / 20000, [1 / 3, 0, 1 / 6, 1 / 2], atol=0.02)


def test_playouts_share_their_chance_across_a_positions_moves(setup, gathered):
    data, _ = gathered
    late = int(np.argmin(np.abs(data["ply"] - 60)))  # mid-game: far from the turn limit
    games = diag.restore_games(setup.plan, setup.levels, data, [late, late, late])
    encoder = make_encoder(DIMS, "full", True)
    space = encoder.action_space
    legal = np.flatnonzero(space.mask(games[0]))
    pos = np.array([late, late, late], dtype=np.int32)
    move = np.array([legal[0], legal[0], legal[1]], dtype=np.int16)
    ply = data["ply"][pos]
    run = lambda seed, **kw: diag.play_out(setup, games, pos, move, ply, playouts=3, seed=seed, **kw)
    outcome, length = run(11, batch=4)
    assert outcome.shape == (3, 3) and set(np.unique(outcome)) <= {0, 1, 2} and (length >= 0).all()
    # the same move twice: the same chance and the same samples give the same games
    assert np.array_equal(outcome[0], outcome[1]) and np.array_equal(length[0], length[1])
    again, again_length = run(11, batch=7)  # nor does the batching matter
    assert np.array_equal(outcome, again) and np.array_equal(length, again_length)
    _, other_length = run(12, batch=4)
    assert not np.array_equal(length, other_length)
    assert len(np.unique(length[0])) > 1  # the playouts of a move differ from each other
    # playout 0 of the first job, by hand
    game = games[0].copy()
    mover = int(game.current_player_index)
    hands = [cpp.HiddenHand(resources=list(p.resources), development_cards=list(p.development_cards),
                            development_card_owned_at_start=list(p.development_card_owned_at_start))
             for p in game.players]
    before = encoder.encode(game, mover).copy()
    game.redeal_hidden_cards(hands, derive_seed(11, "chance", late, 0))
    assert np.array_equal(encoder.encode(game, mover), before)  # every hand and the dice state stay
    game.execute(space.decode(game, int(move[0])))
    rng = np.random.default_rng(derive_seed(11, "sample", late, 0))
    decisions = 0
    while game.winning_player_index is None and game.num_turns < 300:
        actions = game.playable_actions
        if len(actions) == 1:
            game.execute(actions[0])
            continue
        row = encoder.encode(game, int(game.current_player_index)).astype(np.float32)[None]
        with torch.inference_mode():
            logits = setup.model.policy_logits(torch.from_numpy(row)).numpy()
        choice = diag.sample_rows(logits, row[:, setup.mask] > 0, np.array([rng.random()]))[0]
        game.execute(space.decode(game, int(choice)))
        decisions += 1
    winner = game.winning_player_index
    assert int(outcome[0, 0]) == (1 if winner is None else 2 * int(int(winner) == mover))
    assert int(length[0, 0]) == decisions
    # the trainer's fast search as the playout policy
    searched, searched_length = run(11, batch=4, sims=4, threads=2)
    assert np.array_equal(searched[0], searched[1]) and (searched_length > 0).any()


# ---------------------------------------------------------------- report


def synthetic():
    """Four positions with three legal moves each (actions 0, 1, 2); A2 differs from A1 at
    position 1, B at positions 1, 2 and 3."""
    start = np.array([0, 3, 6, 9, 12])
    action = np.tile(np.array([0, 1, 2], dtype=np.int16), 4)
    net_p = np.tile(np.array([0.9, 0.09, 0.01], dtype=np.float32), 4)
    arrays = {"legal_start": start, "legal_action": action, "net_p": net_p,
              "net_top": np.zeros(4, np.int16), "kind": np.array([0, 4, 4, 2], np.int8),
              "full": np.array([True, False, False, True]), "game": np.arange(4),
              "action_names": np.array(["ROAD@0", "ROAD@1", "MARITIME_4_WOOD_ORE"])}
    chosen = {"A1": [0, 0, 0, 0], "A2": [0, 1, 0, 0], "B": [0, 1, 2, 2]}
    visits = {"A1": [10, 0, 0, 8, 2, 0, 9, 1, 0, 7, 0, 3], "A2": [10, 0, 0, 4, 6, 0, 9, 1, 0, 9, 1, 0],
              "B": [10, 0, 0, 3, 7, 0, 2, 0, 8, 4, 0, 6]}
    q = {"A1": [.1, np.nan, np.nan, .1, 0, np.nan, .1, 0, np.nan, .1, np.nan, .3],
         "A2": [.1, np.nan, np.nan, .1, .2, np.nan, .1, 0, np.nan, .1, 0, np.nan],
         "B": [.1, np.nan, np.nan, .1, .3, np.nan, np.nan, np.nan, .5, .2, np.nan, .4]}
    info = {}
    for seed, name in enumerate(chosen):
        v = np.array(visits[name], dtype=np.uint32)
        arrays[f"chosen{SEP}{name}"] = np.array(chosen[name], dtype=np.int16)
        arrays[f"visits{SEP}{name}"] = v
        arrays[f"q{SEP}{name}"] = np.array(q[name], dtype=np.float32)
        arrays[f"prior{SEP}{name}"] = net_p if name != "B" else np.full(12, 1 / 3, np.float32)
        arrays[f"target{SEP}{name}"] = (v / 10).astype(np.float32)
        arrays[f"value{SEP}{name}"] = np.zeros(4, np.float32)
        arrays[f"depth{SEP}{name}"] = np.ones(4, np.float32)
        arrays[f"tree_bytes{SEP}{name}"] = np.full(4, 2**20, np.int64)
        info[name] = {"name": name, "visits": 10 if name != "B" else 40, "temp": 1.0, "mix": 0.0 if name != "B" else 1.0,
                      "noise": True, "seed": seed, "leaves": 40, "wall_s": 1.0}
    found = {"ckpt": "x@e0", "arms": info, "positions_digest": "d"}
    # wins of 4 playouts (outcome2: 2 a win, 1 cut): position 1 moves 0 and 1; 2: 0 and 2; 3: 0 and 2
    referee = {"pos": np.array([1, 1, 2, 2, 3, 3]), "move": np.array([0, 1, 0, 2, 0, 2]),
               "outcome2": np.array([[2, 2, 0, 0], [2, 2, 2, 0], [2, 0, 0, 0], [2, 2, 2, 2], [2, 2, 1, 1], [0, 0, 2, 2]],
                                    dtype=np.int8)}
    return found, arrays, referee


def test_the_report_counts_disagreements_and_pairs_the_referee():
    found, arrays, referee = synthetic()
    out = diag.report(found, arrays, referee, boot=200)
    assert out["positions"] == 4 and out["floor_rate"] == 0.25 and out["judged_moves"] == 6
    assert abs(out["net_entropy"] - float(-(np.array([.9, .09, .01]) * np.log([.9, .09, .01])).sum())) < 1e-6
    assert out["truncated_playouts"] == 2 / 24
    b = out["arms"]["B"]
    assert b["overrule"]["k"] == 3 and b["vs_reference"]["differ"]["rate"] == 0.75
    assert abs(b["differ_minus_floor"] - 0.5) < 1e-12 and out["arms"]["A1"]["vs_reference"]["differ"]["k"] == 0
    assert abs(b["prior_entropy"] - np.log(3)) < 1e-6 and abs(b["nats_added"] - (np.log(3) - out["net_entropy"])) < 1e-6
    assert b["vs_reference"]["by_kind"]["build"]["rate"] == 1.0 and b["vs_reference"]["by_kind"]["setup"]["k"] == 0
    w = b["vs_reference"]["where_differ"]
    # A1's visits on B's moves: 2 (position 1), 0 (2), 3 (3)
    assert (w["n"], w["other_visits_0"], w["other_visits_le2"], w["other_visits_median"]) == (3, 1 / 3, 2 / 3, 2.0)
    assert abs(w["net_prior_mean"] - (0.09 + 0.01 + 0.01) / 3) < 1e-6 and w["net_prior_below_1pct"] == 2 / 3  # float32 0.01 is below 0.01
    # B's own Q gap: .3 - .1, unknown (it never visited A1's move at position 2), .4 - .2
    assert abs(w["own_q_gap_mean"] - 0.2) < 1e-6 and w["own_q_gap_unknown"] == 1 / 3
    assert w["moves"] == {"ROAD -> MARITIME": 2, "ROAD -> ROAD": 1}
    r = b["vs_reference"]["referee"]
    # B's move minus A1's: +.25 (position 1), +.75 (2), .5 - .75 (3); 0 at position 0
    assert abs(r["per_position"]["mean"] - (0.25 + 0.75 - 0.25) / 4) < 1e-12 and r["per_position"]["n"] == 4
    assert abs(r["where_differ"]["mean"] - 0.25) < 1e-12 and r["where_differ"]["n"] == 3 and r["unjudged"] == 0
    assert r["where_differ"]["lo"] <= 0.25 <= r["where_differ"]["hi"]
    assert abs(r["by_kind"]["build"]["mean"] - 0.5) < 1e-12 and abs(r["by_kind"]["robber"]["mean"] + 0.25) < 1e-12
    floor = out["arms"]["A2"]["vs_reference"]["referee"]
    assert abs(floor["per_position"]["mean"] - 0.25 / 4) < 1e-12 and floor["where_differ"]["n"] == 1
    assert [(p["arm"], p["against"]) for p in out["pairs"]] == [("A2", "A1"), ("B", "A1")]
    curve = out["curve"]
    assert [point["arm"] for point in curve["temp 1, mix 0, noise on"]] == ["A1", "A2"]
    point = curve["temp 1, mix 1, noise on"][0]
    assert point["visits"] == 40 and point["differ"]["rate"] == 0.75
    assert abs(point["referee_per_position"]["mean"] - 0.1875) < 1e-12
    diag.print_report(out)
    json.dumps(out)
    # a move the referee has not judged leaves its position out of the paired difference
    partial = {key: value[:4] for key, value in referee.items()}
    r = diag.report(found, arrays, partial, boot=50)["arms"]["B"]["vs_reference"]["referee"]
    assert r["unjudged"] == 1 and r["per_position"]["n"] == 3 and r["where_differ"]["n"] == 2
    # without a referee the disagreement tables stand alone
    bare = diag.report(found, arrays, None, boot=50)
    assert "referee" not in bare["arms"]["B"]["vs_reference"] and bare["playouts"] is None
    diag.print_report(bare)


def test_actions_group_by_family():
    assert [diag.family(n) for n in ("ROAD@3", "ROBBER@4/1", "MARITIME_4_WOOD_ORE", "END_TURN", "DISCARD_WOOD")] == \
        ["ROAD", "ROBBER", "MARITIME", "END_TURN", "DISCARD"]


# ---------------------------------------------------------------- the four steps


def test_the_steps_chain_and_reruns_only_add(ckpt, tmp_path, capsys):
    pos, found, judged, out = (str(tmp_path / name) for name in ("pos.npz", "search.npz", "referee.npz", "report.json"))
    cpu = ["--device", "cpu", "--env-threads", "2", "--progress", "1000"]
    diag.main(["positions", "--ckpt", ckpt, "--out", pos, "--games", "4", "--envs", "4", "--seed", "5", *cpu])
    manifest, data = diag.load_npz(pos)
    assert manifest["positions"] == len(data["game"]) and manifest["restore_checked"] and manifest["git"]
    assert manifest["pool_seed"] != manifest["train_pool_seed"] and manifest["train"]["kernels"] == "torch"
    search = ["search", "--positions", pos, "--out", found, "--wide-visits", "40", *cpu]
    diag.main(search)
    first, arrays = diag.load_npz(found)
    assert list(first["arms"]) == ["A1", "A2", "C", "B"] == diag.arm_names(arrays)
    assert first["arms"]["A1"]["visits"] == 16 and first["arms"]["B"]["mix"] == 0.25 and first["arms"]["A1"]["noise"]
    assert first["arms"]["B"]["prior_entropy"] > first["arms"]["C"]["prior_entropy"]
    assert abs(first["arms"]["C"]["prior_entropy"] - first["arms"]["A1"]["prior_entropy"]) < 1e-6
    # a rerun with a grid keeps the finished arms and adds the new ones
    diag.main([*search, "--grid-visits", "16,32", "--grid-temps", "1,2"])
    second, more = diag.load_npz(found)
    assert list(second["arms"]) == ["A1", "A2", "C", "B", "g16_t1", "g32_t1", "g16_t2", "g32_t2"]
    assert all(same(arrays[key], more[key]) for key in arrays)
    assert second["arms"]["A1"] == first["arms"]["A1"] and len(second["runs"]) == 2
    assert second["runs"][1]["arms"] == ["g16_t1", "g32_t1", "g16_t2", "g32_t2"] and not second["arms"]["g32_t2"]["noise"]
    with pytest.raises(SystemExit, match="other settings"):
        diag.main(["search", "--positions", pos, "--out", found, "--wide-visits", "41", *cpu])
    with pytest.raises(SystemExit, match="repeat"):
        diag.main([*search, "--arm", "C:visits=41"])
    with pytest.raises(SystemExit, match="something else"):
        diag.main([*search, "--seed", "6"])
    referee = ["referee", "--positions", pos, "--search", found, "--out", judged, "--playouts", "2", "--batch", "64", *cpu]
    diag.main([*referee, "--arms", "A1,A2,C,B"])
    _, held = diag.load_npz(judged)
    want = diag.moves_to_judge(more, ["A1", "A2", "C", "B"])
    assert np.array_equal(held["pos"], want[0]) and np.array_equal(held["move"], want[1])
    diag.main(referee)  # every arm now: only the moves not judged yet are played
    two, both = diag.load_npz(judged)
    everything = diag.moves_to_judge(more, diag.arm_names(more))
    assert len(both["pos"]) == len(everything[0]) and len(two["runs"]) == 2
    assert set(zip(both["pos"].tolist(), both["move"].tolist())) == set(zip(everything[0].tolist(), everything[1].tolist()))
    assert np.array_equal(both["outcome2"][:len(held["pos"])], held["outcome2"])
    assert two["runs"][1]["moves"] == len(everything[0]) - len(want[0])
    diag.main(referee)
    assert diag.load_npz(judged)[0]["runs"][2]["moves"] == 0
    with pytest.raises(SystemExit, match="something else"):
        diag.main([*referee, "--playouts", "3"])
    capsys.readouterr()
    diag.main(["report", "--search", found, "--referee", judged, "--out", out, "--boot", "100"])
    printed = capsys.readouterr().out
    report = json.loads(Path(out).read_text())
    assert report["positions"] == manifest["positions"] and set(report["arms"]) == set(second["arms"])
    assert "A2 against A1" in printed and "curve" in printed and "temp 2, mix 0, noise off" in report["curve"]
    assert [p["visits"] for p in report["curve"]["temp 2, mix 0, noise off"]] == [16, 32]
    assert all(arm["vs_reference"]["referee"]["unjudged"] == 0 for arm in report["arms"].values())
    assert report["manifest"]["referee"]["playouts"] == 2
    # a smoke on the first positions only
    small = str(tmp_path / "small.npz")
    diag.main(["search", "--positions", pos, "--out", small, "--no-defaults", "--arm", "A1:visits=8", "--limit", "2", *cpu])
    assert len(diag.load_npz(small)[1][f"chosen{SEP}A1"]) == 2


def test_nothing_is_written_under_the_runs_directory(ckpt, tmp_path):
    with pytest.raises(SystemExit, match="runs directory"):
        diag.main(["positions", "--ckpt", ckpt, "--out", str(tmp_path / "runs" / "x" / "pos.npz"), "--device", "cpu"])
    with pytest.raises(SystemExit, match="go together"):
        diag.arm_list(diag.argparse.Namespace(no_defaults=True, arm=None, grid_visits="600", grid_temps=None,
                                              grid_mix=0.0, grid_seed=1), 600)
