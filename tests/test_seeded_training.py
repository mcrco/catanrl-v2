"""Test 15: seeded strict CPU training is bitwise reproducible."""

from __future__ import annotations

import pytest
from run_helpers import train
from train_configs import tiny_ppo

from catanrl.train.puffer import train_ppo

pytestmark = pytest.mark.train


def run(tmp_path, name, seed):
    writer = train(train_ppo, tiny_ppo(seed=seed), name)
    assert writer.status == "finished"
    outcome = {k: v for row in writer.rows for k, v in row.items()
               if k.startswith(("losses/", "environment/", "critic/"))}
    per_epoch = [{k: v for k, v in row.items() if k.startswith(("losses/", "environment/", "critic/"))}
                 for row in writer.rows]
    return [(ref.epoch, ref.weights_sha256) for ref in writer.saved], per_epoch, outcome


def test_same_seed_same_weights_and_metrics(tmp_path):
    first = run(tmp_path, "a", seed=3)
    second = run(tmp_path, "b", seed=3)
    assert [epoch for epoch, _ in first[0]] == [0, 2, 4]
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert any(k.startswith("losses/") for k in first[2])


def test_other_seed_other_weights(tmp_path):
    first = run(tmp_path, "a", seed=3)
    other = run(tmp_path, "c", seed=4)
    assert first[0][0] != other[0][0]  # the init seed changes e000000
    assert first[0][-1] != other[0][-1]
