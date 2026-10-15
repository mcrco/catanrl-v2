"""The critic's calibration metrics on synthetic trajectories, against brute force:
OutcomeCalibration (truth_sum, per-slot calibration) and the rollout-vs-trained gap."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from catanrl.train.critic_metrics import OutcomeCalibration
from catanrl.train.puffer_policy import PufferPolicy, ValueBatch


def brute_force(finished: list[dict], gamma: float, max_players: int) -> dict[str, float]:
    """The metrics of OutcomeCalibration.pop() over ``finished`` games, decision by decision."""
    sums: dict[int, dict] = {}
    for game in finished:
        n, winner, decisions = game["n"], game["winner"], game["decisions"]
        total = sums.setdefault(n, {"games": 0, "decisions": 0, "pred": np.zeros(n), "real": np.zeros(n),
                                    "sq": np.zeros(n)})
        total["games"] += 1
        for t, (actor, pred) in enumerate(decisions):
            total["decisions"] += 1
            for k in range(n):
                y = gamma ** (len(decisions) - 1 - t) * float((actor + k) % n == winner)
                total["pred"][k] += pred[k]
                total["real"][k] += y
                total["sq"][k] += (pred[k] - y) ** 2
    out = {}
    for n, total in sums.items():
        d = total["decisions"]
        out[f"truth_sum_{n}p"] = total["real"].sum() / d
        out[f"finished_value_sum_{n}p"] = total["pred"].sum() / d
        out[f"finished_games_{n}p"] = float(total["games"])
        for k in range(n):
            out[f"pred_slot{k}_{n}p"] = total["pred"][k] / d
            out[f"real_slot{k}_{n}p"] = total["real"][k] / d
            out[f"brier_slot{k}_{n}p"] = total["sq"][k] / d
    return out


@pytest.mark.parametrize("gamma", [0.9, 1.0])
def test_outcome_calibration_matches_brute_force(gamma):
    """Envs of 4 seat rows play 2-, 3- and 4-player games of random lengths, winners
    (some truncated: nobody) and actors. A game's last decision is followed by a step
    whose rows are all terminal, 1 to the winner's row, and whose observation is the
    next game's first decision. Popped twice: games in flight carry over."""
    rng = np.random.default_rng(7)
    envs, players = 6, 4
    calibration = OutcomeCalibration(num_envs=envs, max_players=players, gamma=gamma)

    def new_game():
        n = int(rng.integers(2, players + 1))
        return {"n": n, "length": int(rng.integers(1, 40)), "winner": int(rng.integers(-1, n)), "decisions": []}

    games = [new_game() for _ in range(envs)]
    finished: list[dict] = []
    for step in range(600):
        reward = torch.zeros(envs * players)
        done = torch.zeros(envs * players, dtype=torch.bool)
        rows, probs, present = [], [], []
        for e in range(envs):
            game = games[e]
            if len(game["decisions"]) == game["length"]:
                done[e * players:(e + 1) * players] = True
                if game["winner"] >= 0:
                    reward[e * players + game["winner"]] = 1.0
                finished.append(game)
                games[e] = game = new_game()
            actor = int(rng.integers(0, game["n"]))
            pred = np.zeros(players, np.float32)
            pred[:game["n"]] = rng.random(game["n"])
            game["decisions"].append((actor, pred))
            rows.append(e * players + actor)
            probs.append(pred)
            present.append([1.0] * game["n"] + [0.0] * (players - game["n"]))
        calibration.observe(reward, done, torch.tensor(rows), torch.tensor(np.array(probs)), torch.tensor(present))
        if step in (299, 599):
            found = calibration.pop()
            expected = brute_force(finished, gamma, players)
            finished = []
            assert set(found) == set(expected)
            assert set(expected) >= {"truth_sum_2p", "truth_sum_3p", "truth_sum_4p", "brier_slot3_4p"}
            for key, value in expected.items():
                assert found[key] == pytest.approx(value, rel=1e-6, abs=1e-9), key
    if gamma == 1.0:  # undiscounted: the truth is the share of decisions in games somebody won
        assert 0.0 < found["truth_sum_4p"] < 1.0


class _Stub(nn.Module):
    num_players = 4

    def heads(self, obs):
        raise AssertionError("not called")


def test_rollout_vs_trained_gap():
    """The rollout network's predicted sum (the values stored before training) minus the
    network's in training, on the trained decisions (not a segment's last step)."""
    torch.manual_seed(0)
    envs, players, horizon = 3, 4, 5
    rows = envs * players
    policy = PufferPolicy(_Stub(), num_rows=rows)
    values = torch.rand(rows, horizon)
    policy.snapshot_values(values)
    rollout = values.clone()
    values.add_(0.25)  # write-backs since the snapshot do not count
    idx = torch.tensor([1, 6, 8, 11])
    local = torch.tensor([0, 0, 1, 2, 3, 3])
    t = torch.tensor([0, 4, 2, 1, 3, 4])
    present = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 0]],
                           dtype=torch.float32)
    logits = torch.randn(len(t), players)
    batch = ValueBatch(index=local * horizon + t, logits=logits, present=present, rows=len(idx) * horizon)
    policy.value_loss(batch, torch.zeros(len(idx) * horizon), idx=idx, values=values,
                      advantages=torch.zeros(rows, horizon))
    gaps: dict[int, list[float]] = {}
    for m in range(len(t)):
        if int(t[m]) == horizon - 1:
            continue
        row = int(idx[local[m]])
        env, seat, n = row // players, row % players, int(present[m].sum())
        stored = sum(float(rollout[env * players + (seat + k) % n, t[m]]) for k in range(n))
        trained = float(torch.sigmoid(logits[m, :n]).sum())
        gaps.setdefault(n, []).append(stored - trained)
    calibration = policy.pop_calibration()
    assert sorted(gaps) == [2, 3, 4]
    for n, values_n in gaps.items():
        assert calibration[f"rollout_vs_trained_gap_{n}p"] == pytest.approx(np.mean(values_n), abs=1e-6)
    assert "rollout_vs_trained_gap_4p" not in policy.pop_calibration()

