"""tools/critic_heldout.py: the held-out games of a frozen policy with every critic's values
per decision, and the report's metrics (log loss, Brier, AUC within turn bins, the
calibration-only baselines, paired deltas)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_critic_refit import SPLIT, network, saved

from catanrl.checkpoint import load_parts
from catanrl.envs.puffer import plan_env

spec = importlib.util.spec_from_file_location("critic_heldout", Path(__file__).resolve().parents[1] / "tools" / "critic_heldout.py")
critic_heldout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(critic_heldout)

ENV = {"scenarios": [{"game": "base-2p", "turn_limit": 40, "maps": 8}]}


def test_the_heldout_pool_shares_no_board_with_the_training_pool(monkeypatch):
    plan, pools = critic_heldout.heldout_plan(ENV, seed=42, pool_seed=None)
    train = plan_env(ENV, pool_seed=pools["train_pool_seed"])
    assert pools["train_pool"] == train.pool["hash"] != pools["pool"] == plan.pool["hash"]
    _, pinned = critic_heldout.heldout_plan({**ENV, "map_pool": {"seed": 7}}, seed=42, pool_seed=8)
    assert (pinned["train_pool_seed"], pinned["pool_seed"]) == (7, 8)  # the run's pinned pool seed is the one held out
    with pytest.raises(ValueError, match="the training pool's"):
        critic_heldout.heldout_plan(ENV, seed=42, pool_seed=pools["train_pool_seed"])
    with pytest.raises(ValueError, match="the training pool's"):
        critic_heldout.heldout_plan({**ENV, "map_pool": {"seed": 7}}, seed=42, pool_seed=7)
    import catanrl.envs.puffer as envs

    real = envs.plan_env
    monkeypatch.setattr(envs, "plan_env", lambda env, pool_seed: real(env, pool_seed=1))  # every seed, one pool
    with pytest.raises(ValueError, match="8 held-out boards are in the training pool"):
        critic_heldout.heldout_plan(ENV, seed=42, pool_seed=None)


def models(tmp_path):
    source = network(shared=True, level="private")
    init = {"ckpt": saved(source, tmp_path), **SPLIT}
    a, b = network(shared=False, seed=1), network(shared=False, seed=2)
    load_parts(a, init)
    load_parts(b, init)
    with torch.no_grad():
        b.critic.value.mlp[-1].bias.add_(0.5)  # another critic, the same actor
    return a, b


def test_play_records_every_critic_on_the_policys_games(tmp_path):
    a, b = models(tmp_path)
    plan, _ = critic_heldout.heldout_plan(ENV, seed=42, pool_seed=None)
    data = critic_heldout.play_games(a, [a, b, a], plan, games=4, envs=2, seed=3)
    n = len(data["t"])
    assert len(data["g_id"]) == 4 and sorted(data["g_id"]) == [0, 1, 2, 3] and data["g_T"].sum() == n
    assert data["z"].shape == (n, 3, 2) and np.array_equal(data["z"][:, 0], data["z"][:, 2])
    assert not np.array_equal(data["z"][:, 0], data["z"][:, 1])
    for gid, winner in zip(data["g_id"], data["g_winner"]):
        m = data["game"] == gid
        assert np.array_equal(data["t"][m], np.arange(m.sum()))
        assert np.array_equal(data["y"][m, 0], (data["seat"][m] == winner).astype(np.int8))
        assert np.array_equal(data["y"][m, 1], (1 - data["seat"][m] == winner).astype(np.int8))
    again = critic_heldout.play_games(a, [b], plan, games=4, envs=2, seed=3)
    assert np.array_equal(again["z"][:, 0], data["z"][:, 1])  # the same games, whatever is scored
    other = network(shared=False, seed=4)
    with pytest.raises(ValueError, match="not a frozen actor"):
        critic_heldout.play_games(a, [a, other], plan, games=2, envs=2, seed=3)


def synthetic(k: int = 3, games: int = 60, per: int = 30, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    game = np.repeat(np.arange(games), per)
    turn = np.tile(np.arange(per), games)
    strength = np.repeat(rng.normal(size=games), per)
    y0 = (strength + rng.normal(scale=0.5, size=game.size) > 0).astype(np.int8)
    y = np.stack((y0, 1 - y0), 1)
    z = np.empty((game.size, k, 2), np.float32)
    for i in range(k):
        logit = strength * (1 + i) + rng.normal(scale=0.1, size=game.size)
        z[:, i, 0], z[:, i, 1] = logit - 0.3, -logit - 0.3  # under-confident by the discount's kind of gap
    manifest = {"checkpoints": [f"x@e{25 * i}" for i in range(k)], "epochs": [25 * i for i in range(k)],
                "games": games, "decisions": int(game.size), "truncated": 0, "pool": "p", "train_pool": "t",
                "policy": "x@e0"}
    return {"manifest": np.array(json.dumps(manifest)), "game": game, "turn": turn, "y": y, "z": z}


def test_the_report_pairs_every_checkpoint_with_the_first_an_earlier_one_and_the_better_baseline():
    data = synthetic()
    out = critic_heldout.report(data, boot=200, back=50)
    first, second, third = out["checkpoints"]
    assert first["vs_first"]["bce0"] == [0.0, 0.0, 0.0] and first["vs_back"]["bce0"] == [0.0, 0.0, 0.0]
    assert (first["back_epoch"], second["back_epoch"], third["back_epoch"]) == (0, 25, 0)  # e50 looks back to e0
    assert third["vs_back"] == third["vs_first"]
    p = 1 / (1 + np.exp(-data["z"][:, 1].astype(np.float64)))
    bce0 = -np.mean(data["y"][:, 0] * np.log(p[:, 0]) + (1 - data["y"][:, 0]) * np.log(1 - p[:, 0]))
    assert second["bce0"][0] == pytest.approx(bce0) and second["bce0"][1] <= bce0 <= second["bce0"][2]
    assert second["vs_first"]["bce0"][0] == pytest.approx(second["bce0"][0] - first["bce0"][0])
    best = out["best_baseline"]
    assert second["vs_best_baseline"]["bce0"][0] == pytest.approx(second["bce0"][0] - out["baselines"][best]["bce0"][0])
    # here slot 1's logit falls as slot 0's rises, so normalizing keeps slot 0's order within a turn bin
    assert out["baselines"]["normalized"]["auc0_within_turn"] == pytest.approx(first["auc0_within_turn"])
    assert out["baselines"]["platt_turn"]["bce0"][0] < first["bce0"][0]  # it fixes the under-confidence


def test_auc_and_the_logistic_fit():
    rng = np.random.default_rng(1)
    score, y = rng.normal(size=200).round(1), rng.integers(0, 2, 200)
    pairs = [(s1 > s0) + 0.5 * (s1 == s0) for s1 in score[y == 1] for s0 in score[y == 0]]
    assert critic_heldout.auc(score, y) == pytest.approx(np.mean(pairs))
    x = np.stack((rng.normal(size=20000), np.ones(20000)), 1)
    y = (rng.random(20000) < 1 / (1 + np.exp(-(x @ np.array([1.5, -0.5]))))).astype(np.float64)
    assert critic_heldout.logistic(x, y) == pytest.approx([1.5, -0.5], abs=0.08)
