"""tools/checkpoint_similarity.py: the positions of held-out self-play games (each player's own
row, with the dice input or without), every checkpoint's values and policies on them, and the
report's math (the pairwise matrices, the outcome scores, the series)."""

from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_dice_input import DIMS, GAME, model_of

import catanrl.checkpoint as ck
from catanrl.envs.encoding import make_encoder, row_layout, row_size

spec = importlib.util.spec_from_file_location(
    "checkpoint_similarity", Path(__file__).resolve().parents[1] / "tools" / "checkpoint_similarity.py")
similarity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(similarity)

ENV = {"scenarios": [{"game": GAME, "turn_limit": 40, "maps": 8}]}
CONFIG = {"env": ENV, "seed": 42}


def saved(path: Path, model, epoch: int = 0) -> str:
    """A checkpoint of ``model`` with a run config (the env and the seed of its training pool)."""
    ref = ck.save(path / f"e{epoch:06d}.pt", model, run=None, epoch=epoch, agent_steps=0,
                  config={**CONFIG, "model": model.spec}, code={})
    return str(ref.path)


def widened(source_path: str, seed: int = 9):
    """A network with the dice input warm-started from one without (``init_from.zero_new_inputs``)."""
    target = model_of(True, seed=seed)
    ck.load_parts(target, {"ckpt": source_path, "parts": ["actor", "critic"], "zero_new_inputs": True})
    return target.eval()


def played(players, **kwargs) -> dict[str, np.ndarray]:
    plan, pools = similarity.heldout_plan([CONFIG], None)
    data = similarity.play_positions(players, plan, **{"games": 8, "envs": 4, "per_game": 5, "seed": 3, **kwargs},
                                     log=lambda _: None)
    manifest = {"players": [f"p{i}@e0" for i in range(len(players))], "level": "full", "dims": dict(plan.dims),
                **pools, "games": len(data["g_id"]), "truncated_games": int(data["g_trunc"].sum())}
    return {"manifest": np.array(json.dumps(manifest)), **data}


# ---------------------------------------------------------------- play and score


def test_the_plain_columns_are_the_row_without_the_dice_input():
    columns = similarity.plain_columns(DIMS, "full")
    plain, dice = row_layout(DIMS, "full"), row_layout(DIMS, "full", dice_state=True)
    assert (len(columns), row_size(dice)) == (row_size(plain), 2691) and len(columns) == 2564
    assert np.array_equal(columns, np.sort(columns))
    names = [f"{block}.{i}" for block, (offset, shape) in dice.items() for i in range(int(np.prod(shape)))]
    kept = [names[c].partition(".")[0] for c in columns]
    assert {block: kept.count(block) for block in plain} == {block: int(np.prod(shape)) for block, (_, shape) in plain.items()}


def test_the_heldout_pool_is_held_out_of_every_networks_training_pool():
    plan, pools = similarity.heldout_plan([CONFIG, {**CONFIG, "seed": 43}, CONFIG], None)
    assert len(pools["train_pools"]) == 2 and pools["pool"] not in pools["train_pools"]
    assert pools["pool"] == plan.pool["hash"]
    other = {"env": {"scenarios": [{"game": GAME, "turn_limit": 40, "maps": 4}]}, "seed": 42}
    with pytest.raises(ValueError, match="different envs"):
        similarity.heldout_plan([CONFIG, other], None)


def test_play_keeps_a_sample_of_every_game_at_both_rows():
    players = [model_of(False, seed=1), model_of(True, seed=2)]
    data = played(players)
    n = len(data["game"])
    assert sorted(data["g_id"]) == list(range(8)) and np.bincount(data["g_player"]).tolist() == [4, 4]
    assert data["rows"].shape == (n, 2691) and data["rows"].dtype == np.float32 and data["legal"].shape == (n, 314)
    offset, (size,) = row_layout(DIMS, "full")["action_mask"]
    assert np.array_equal(data["rows"][:, data["plain_columns"]][:, offset:offset + size] > 0, data["legal"])
    assert data["legal"].sum(1).min() >= 1
    for gid, player, decisions, winner, truncated in zip(data["g_id"], data["g_player"], data["g_T"],
                                                         data["g_winner"], data["g_trunc"]):
        m = data["game"] == gid
        assert m.sum() == min(5, decisions) and (data["player"][m] == player).all()
        ply = data["ply"][m]
        assert (np.diff(ply) > 0).all() and 0 <= ply.min() and ply.max() < decisions  # distinct, in order
        assert np.array_equal(data["y"][m], (data["seat"][m] == winner).astype(np.int8))
        assert (data["truncated"][m] == truncated).all() and (winner == -1) == bool(truncated)
    assert (data["ply"] >= 5).any()  # a sample of the whole game, not its first decisions
    again = played(players)
    assert all(np.array_equal(data[name], again[name]) for name in data if name != "manifest")  # seeded
    other = played(players, seed=4)
    assert not np.array_equal(data["rows"], other["rows"])


def test_play_refuses_shares_that_do_not_divide_and_networks_of_another_row():
    plan, _ = similarity.heldout_plan([CONFIG], None)
    players = [model_of(False, seed=1), model_of(True, seed=2)]
    with pytest.raises(ValueError, match="multiple"):
        similarity.play_positions(players, plan, games=8, envs=3, per_game=5, seed=3)
    with pytest.raises(ValueError, match="one level"):
        similarity.play_positions([players[0], model_of(False, level="private")], plan, games=4, envs=2, per_game=5,
                                  seed=3)


def test_score_gives_each_network_its_own_row(tmp_path):
    plain, dice = model_of(False, seed=1), model_of(True, seed=2)
    with torch.no_grad():
        for key in ck.dice_input_keys(dice):
            dice.state_dict()[key].normal_(std=0.1)  # a fresh network's dice weights are zero: make it read them
    data = played([plain, dice])
    start, action = similarity.legal_index(data["legal"])
    assert np.array_equal(np.diff(start), data["legal"].sum(1))
    assert all(np.array_equal(action[start[i]:start[i + 1]], np.flatnonzero(data["legal"][i])) for i in (0, 7, len(start) - 2))
    for model, rows in ((plain, data["rows"][:, data["plain_columns"]]), (dice, data["rows"])):
        got = similarity.score_network(model, data, batch=16)  # several batches
        with torch.inference_mode():
            logits, values = model(torch.from_numpy(np.ascontiguousarray(rows)))
        want = torch.log_softmax(logits.double(), -1).numpy()
        assert np.allclose(got["value"], values[:, 0].numpy(), atol=1e-6)
        assert np.allclose(got["logp"], want[data["legal"]], atol=1e-6)
        assert np.array_equal(got["top"], want.argmax(1)) and np.allclose(got["top_p"], np.exp(want.max(1)), atol=1e-6)
        assert np.allclose(np.add.reduceat(np.exp(got["logp"].astype(np.float64)), start[:-1]), 1.0, atol=1e-5)
    blank = data["rows"].copy()
    blank[:, np.setdiff1d(np.arange(blank.shape[1]), data["plain_columns"])] = 0.0  # the dice columns
    assert not np.allclose(similarity.score_network(dice, dict(data, rows=blank))["value"], got["value"])  # it reads them
    with pytest.raises(ValueError, match="does not read the positions' rows"):
        similarity.score_network(model_of(False, level="private"), data)


def test_the_commands_end_to_end_and_a_warm_started_dice_network_is_its_source(tmp_path, capsys):
    """play, score and report over checkpoint files: a network with the dice input warm-started
    from one without (the bundle run's e0 of its parent's e1800) scores as its source on every
    position, each at its own row: zero distance in the report, and not to a third network."""
    plain = saved(tmp_path / "plain", model_of(False, seed=1))
    warm = saved(tmp_path / "warm", widened(plain), epoch=25)
    other = saved(tmp_path / "other", model_of(True, seed=2), epoch=50)
    positions, scores, out = (str(tmp_path / name) for name in ("positions.npz", "scores.npz", "report.json"))
    similarity.main(["play", "--players", plain, other, "--games", "8", "--envs", "4", "--per-game", "5", "--seed", "3",
                     "--threads", "1", "--out", positions])
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["games"] == 8 and manifest["dice_state"] == [False, True] and manifest["row_width"] == [2564, 2691]
    assert manifest["pool"] not in manifest["train_pools"]
    similarity.main(["score", positions, "--ckpts", plain, warm, other, "--threads", "1", "--out", scores])
    capsys.readouterr()
    data = dict(np.load(scores))
    stored = json.loads(str(data["manifest"]))
    assert [c["dice_state"] for c in stored["checkpoints"]] == [False, True, True]
    assert [c["iteration"] for c in stored["checkpoints"]] == [0, 25, 50]
    n = manifest["positions"]
    assert data["value"].shape == (3, n) and data["logp"].shape == (3, len(data["legal_action"]))
    assert np.array_equal(data["value"][0], data["value"][1]) and np.array_equal(data["logp"][0], data["logp"][1])
    similarity.main(["report", scores, "--boot", "50", "--json", out])
    text = capsys.readouterr().out
    assert f"{n} positions of 8 games" in text and text.count("\n  it ") == 3
    result = json.loads(Path(out).read_text())
    assert set(result["matrices"]) == {"all", f"player:{manifest['players'][0]}", f"player:{manifest['players'][1]}"}
    whole = result["matrices"]["all"]
    assert (whole["value_mad"][0][1], whole["top1_agree"][0][1], whole["js"][0][1]) == (0.0, 1.0, 0.0)
    assert whole["value_mad"][0][2] > 0 and whole["js"][0][2] > 0 and whole["value_mad"][1][2] == whole["value_mad"][0][2]
    assert result["series"]["to_previous"]["value_mad"][:2] == [None, 0.0]
    assert sum(result["positions"][name] for name in result["matrices"] if name != "all") == n


@pytest.mark.gpu
def test_score_on_the_gpu_is_the_cpus_to_rounding():
    plain, dice = model_of(False, seed=1), model_of(True, seed=2)
    data = played([plain, dice])
    for model in (plain, dice):
        cpu = similarity.score_network(model, data)
        gpu = similarity.score_network(model.to("cuda"), data, device="cuda")
        assert np.abs(cpu["value"] - gpu["value"]).max() < 1e-5 and np.abs(cpu["logp"] - gpu["logp"]).max() < 1e-4


@pytest.mark.runs
def test_the_bundle_runs_e0_values_positions_as_its_parents_e1800(tmp_path):
    """The real pair: az-2p-bundle/main@e0 (with the dice input) against az-2p-kata/main+1500@e1800
    (without), on positions of two held-out games each network's policy played from its own row
    (the CPU's module: the same values and policies; a GPU rounds the wider rows' products
    differently, tests/test_dice_input.py)."""
    from catanrl.runs import RunError, load_checkpoint

    try:
        ckpts = [load_checkpoint("az-2p-kata/main+1500@e1800#7c44ec80540a"), load_checkpoint("az-2p-bundle/main@e0")]
    except RunError:
        pytest.skip("az-2p-kata/main+1500 or az-2p-bundle/main is not in the runs directory")
    assert [c.dice_state for c in ckpts] == [False, True]
    plan, pools = similarity.heldout_plan([c.header["config"] for c in ckpts], None)
    assert len(pools["train_pools"]) == 1  # the bundle run trains on its parent's boards
    models = [c.model("all") for c in ckpts]
    data = similarity.play_positions(models, plan, games=2, envs=2, per_game=10, seed=3, log=lambda _: None)
    manifest = {"players": [str(c.ref) for c in ckpts], "level": "full", "dims": dict(plan.dims), **pools,
                "games": 2, "truncated_games": int(data["g_trunc"].sum())}
    data["manifest"] = np.array(json.dumps(manifest))
    assert len(data["game"]) == 20 and sorted(set(data["player"])) == [0, 1]
    plain = make_encoder(plan.dims, "full")
    assert data["rows"][:, data["plain_columns"]].shape == (20, plain.size) and plain.size == models[0].obs_size
    old, new = (similarity.score_network(model, data) for model in models)
    assert np.abs(old["value"] - new["value"]).max() <= 1e-6 and np.abs(old["logp"] - new["logp"]).max() <= 1e-5
    assert np.array_equal(old["top"], new["top"]) and 0.0 < old["value"].min() and old["value"].max() < 1.0
    refs = similarity.checkpoint_list([], ["az-2p-kata/main+1500", "az-2p-bundle/main"], 300)
    by_ref = {(c["run"], c["epoch"]): c["iteration"] for c in refs}
    assert by_ref[("az-2p-kata/main+1500", 1800)] == 1800 and by_ref[("az-2p-bundle/main", 0)] == 1800
    assert all(it == 1800 + epoch for (run, epoch), it in by_ref.items() if run == "az-2p-bundle/main")
    forked = similarity.checkpoint_list([], ["az-2p-kata/main+600"], 25)  # a fork keeps its parent's count
    assert [c["iteration"] for c in forked] == [325, 350, 375, 400, 410]  # and its last checkpoint is kept


# ---------------------------------------------------------------- report


def synthetic(k: int = 3, games: int = 40, per: int = 6, actions: int = 8, seed: int = 0) -> dict[str, np.ndarray]:
    """``k`` copies of one random checkpoint's scores on ``games * per`` positions of two players."""
    rng = np.random.default_rng(seed)
    n = games * per
    game = np.repeat(np.arange(games), per)
    legal = rng.random((n, actions)) < 0.5
    legal[np.arange(n), rng.integers(0, actions, n)] = True
    legal[np.arange(n), rng.integers(0, actions, n)] = True
    legal[legal.sum(1) < 2, :2] = True  # two legal actions at least: room for disjoint policies
    start, action = similarity.legal_index(legal)
    p = rng.random(len(action)) + 0.05
    p /= np.repeat(np.add.reduceat(p, start[:-1]), np.diff(start))
    top = np.array([action[a:b][np.argmax(p[a:b])] for a, b in itertools.pairwise(start)], np.int16)
    top_p = np.maximum.reduceat(p, start[:-1])
    y = np.repeat(rng.integers(0, 2, games), per).astype(np.int8)
    value = np.clip(0.3 + 0.3 * y + rng.normal(scale=0.05, size=n), 0.2, 0.8).astype(np.float32)
    manifest = {"positions": {"players": ["a@e0", "b@e0"], "games": games, "truncated_games": 0, "pool": "p",
                              "train_pools": ["t"]},
                "checkpoints": [{"ref": f"x@e{25 * i}", "run": "x" if i < 2 else "y", "epoch": 25 * i,
                                 "iteration": 25 * i} for i in range(k)]}
    return {"manifest": np.array(json.dumps(manifest)), "legal_start": start, "legal_action": action,
            "value": np.tile(value, (k, 1)), "top": np.tile(top, (k, 1)), "top_p": np.tile(top_p, (k, 1)),
            "logp": np.tile(np.log(p), (k, 1)), "game": game, "ply": np.tile(np.arange(per), games),
            "turn": np.tile(np.arange(per), games), "seat": np.zeros(n, np.int8), "player": (game % 2).astype(np.int8),
            "y": y, "truncated": np.zeros(n, bool)}


def test_identical_checkpoints_are_at_zero_distance_and_agree_everywhere():
    out = similarity.report(synthetic(), boot=50)
    for name in ("all", "player:a@e0", "player:b@e0"):
        m = out["matrices"][name]
        assert np.array_equal(m["value_mad"], np.zeros((3, 3))) and np.array_equal(m["js"], np.zeros((3, 3)))
        assert np.array_equal(m["top1_agree"], np.ones((3, 3))) and np.allclose(m["value_corr"], 1.0)
    assert out["positions"] == {"all": 240, "player:a@e0": 120, "player:b@e0": 120}
    series = out["series"]
    assert series["iteration"] == [0, 25, 50]
    assert series["to_first"]["value_mad"] == [0.0, 0.0, 0.0] and series["to_first"]["top1_agree"] == [1.0, 1.0, 1.0]
    assert series["to_previous"]["js"] == [None, 0.0, 0.0] and series["to_previous"]["top1_agree"] == [None, 1.0, 1.0]
    assert out["runs"] == [{"run": "x", "first": 0, "last": 1}, {"run": "y", "first": 2, "last": 2}]
    assert out["checkpoints"][0]["brier"] == out["checkpoints"][2]["brier"]


def test_a_shifted_value_is_at_its_shift_and_a_players_positions_are_their_own_matrix():
    data = synthetic()
    data["value"][1] += 0.1  # everywhere
    data["value"][2, data["player"] == 1] -= 0.05  # on the second player's positions alone
    out = similarity.report(data, boot=50)
    whole, first, second = (out["matrices"][name] for name in ("all", "player:a@e0", "player:b@e0"))
    assert whole["value_mad"][0][1] == pytest.approx(0.1, abs=1e-6) and whole["value_corr"][0][1] == pytest.approx(1.0)
    assert whole["value_mad"][0][2] == pytest.approx(0.025, abs=1e-6)
    assert first["value_mad"][0][2] == 0.0 and second["value_mad"][0][2] == pytest.approx(0.05, abs=1e-6)
    assert whole["value_mad"][1][2] == pytest.approx(0.125, abs=1e-6)  # 0.1 on one half, 0.15 on the other
    assert np.array_equal(whole["value_mad"], np.transpose(whole["value_mad"]))
    assert out["series"]["to_first"]["value_mad"] == pytest.approx([0.0, 0.1, 0.025], abs=1e-6)
    assert out["series"]["to_previous"]["value_mad"][1:] == pytest.approx([0.1, 0.125], abs=1e-6)
    assert whole["js"][0][1] == 0.0 and whole["top1_agree"][0][2] == 1.0  # the policies did not move


def test_the_jensen_shannon_divergence():
    start = np.array([0, 2, 4, 7])
    p = np.array([1.0, 0.0, 0.5, 0.5, 0.2, 0.3, 0.5])
    q = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.2])
    disjoint = np.array([0.0, 1.0, 0.5, 0.5, 0.2, 0.3, 0.5])
    js = similarity.js_divergence(p, q, start)
    mixed = -(0.75 * np.log2(0.75) + 0.25 * np.log2(0.25))
    assert js[0] == pytest.approx(mixed - 0.5) and js[1] == 0.0  # H(m) - (H(p) + H(q)) / 2, in bits
    assert np.array_equal(js, similarity.js_divergence(q, p, start))  # symmetric
    assert np.array_equal(similarity.js_divergence(p, p, start), np.zeros(3))
    assert similarity.js_divergence(p, disjoint, start)[0] == pytest.approx(1.0)  # one bit at most
    assert np.allclose(similarity.entropy(np.stack((p, q)), start)[:, :2], [[0.0, 1.0], [1.0, 1.0]])


def test_the_policy_matrices_on_known_policies():
    data = synthetic(k=3)
    start, action = data["legal_start"], data["legal_action"]
    first, last = start[:-1], start[1:] - 1
    moved = (data["player"] == 0) & (data["ply"] < 3)  # a quarter of the positions, half of the first player's
    one_hot = np.full(len(action), -np.inf)
    one_hot[first] = 0.0  # every position's first legal action
    other = np.full(len(action), -np.inf)
    other[np.where(moved, last, first)] = 0.0  # the moved positions' last legal action instead
    data["logp"] = np.stack((one_hot, other, data["logp"][2])).astype(np.float32)
    data["top"] = np.stack((action[first], action[np.where(moved, last, first)], data["top"][2]))
    out = similarity.report(data, boot=50)
    whole = out["matrices"]["all"]
    assert whole["js"][0][1] == pytest.approx(0.25) and whole["top1_agree"][0][1] == pytest.approx(0.75)
    assert np.array_equal(whole["js"], np.transpose(whole["js"])) and np.diagonal(whole["js"]).tolist() == [0.0] * 3
    assert np.array_equal(whole["top1_agree"], np.transpose(whole["top1_agree"]))
    p = np.exp(data["logp"].astype(np.float64))
    want = similarity.js_divergence(p[0], p[2], start).mean()
    assert whole["js"][0][2] == pytest.approx(want, abs=1e-6) and 0 < want < 1
    assert out["series"]["to_first"]["js"][1] == pytest.approx(0.25)
    assert [out["matrices"][name]["js"][0][1] for name in ("player:a@e0", "player:b@e0")] == [0.5, 0.0]
    assert [out["matrices"][name]["top1_agree"][0][1] for name in ("player:a@e0", "player:b@e0")] == [0.5, 1.0]


def test_the_outcome_scores_leave_truncated_games_out():
    data = synthetic(k=2)
    data["value"][1] = data["y"]  # the oracle
    cut = data["game"] >= 30  # ten games stopped at the turn limit: no winner, y is 0
    data["truncated"] = cut
    data["y"] = np.where(cut, 0, data["y"]).astype(np.int8)
    out = similarity.report(data, boot=200)
    assert out["scored_positions"]["all"] == 180 and out["scored_games"] == 30
    p, y = data["value"][0, ~cut].astype(np.float64), data["y"][~cut]
    first, oracle = out["checkpoints"]
    assert first["brier"][0] == pytest.approx(np.mean((p - y) ** 2), abs=1e-6)
    assert first["log_loss"][0] == pytest.approx(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)), abs=1e-6)
    assert first["brier"][1] < first["brier"][0] < first["brier"][2]  # the game-clustered interval
    assert oracle["brier"] == [0.0, 0.0, 0.0] and oracle["log_loss"][0] == pytest.approx(0.0, abs=1e-6)
    mine = data["player"][~cut] == 0
    assert first["by_player"]["a@e0"]["brier"][0] == pytest.approx(np.mean((p[mine] - y[mine]) ** 2), abs=1e-6)
    assert set(first["by_player"]) == {"a@e0", "b@e0"} and (first["ref"], first["iteration"]) == ("x@e0", 0)
    data["truncated"] = np.ones_like(cut)
    assert "brier" not in similarity.report(data, boot=10)["checkpoints"][0]  # nothing to score against


def test_the_figures_are_skipped_without_matplotlib(tmp_path, capsys):
    out = similarity.report(synthetic(), boot=10)
    paths = similarity.figures(out, tmp_path / "figures")
    if importlib.util.find_spec("matplotlib") is None:
        assert paths == [] and "matplotlib is not installed" in capsys.readouterr().err
    else:
        assert [p.name for p in paths] == ["value_mad.png", "top1_agree.png", "js.png"] and all(p.stat().st_size for p in paths)
