"""Seeded strict CPU training gives pinned weights and metrics.

Two tiny runs: tiny_ppo's Adam on 2p (self-play and against R), and a Muon
run in ppo-4p's shape (four-seat self-play, prioritized segments, cosine,
accumulated micro-batches, one shared trunk). The pinned values were taken
before the critic options (train.critic_targets, critic_gamma, muon_exclude)
existed, so at their defaults training stays bit-identical to the code
before them. Only the keys pinned are compared: metrics added later do not
count.

``CATANRL_UPDATE_GOLDEN=1`` rewrites tests/data/golden_train.json; review
the diff, since any change means training changed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from run_helpers import train
from train_configs import tiny_ppo

from catanrl.train.puffer import train_ppo

pytestmark = pytest.mark.train

GOLDEN = Path(__file__).with_name("data") / "golden_train.json"
PINNED = ("losses/", "environment/", "critic/value_sum_", "critic/target_sum_", "critic/sum_gap_",
          "learning_rate")


def muon_4p() -> dict:
    config = tiny_ppo(
        seed=5, optimizer="muon", learning_rate=0.005, anneal_lr=True, min_lr_ratio=0.1,
        adam_beta1=0.95, adam_eps=1e-12, gamma=0.999, update_epochs=2, vf_coef=2.0, max_grad_norm=4.0,
        ent_coef=0.005, minibatch_size=64, max_minibatch_size=32, prio_alpha=0.8, prio_beta0=0.2)
    config["env"]["scenarios"] = [{"game": "base-4p", "maps": 4, "turn_limit": 1000}]
    config["train"]["total_timesteps"] = 8 * 4 * 8 * 6  # 6 epochs of 8 games x 4 rows x 8 steps
    config["model"] = {"preset": "graph-tiny", "args": {"shared": True}}
    return config


CASES = {"adam-2p": lambda: tiny_ppo(seed=3), "muon-4p": muon_4p}


def record(name: str) -> dict:
    writer = train(train_ppo, CASES[name](), name)
    assert writer.status == "finished"
    return {
        "weights": [[ref.epoch, ref.weights_sha256] for ref in writer.saved],
        "metrics": [{k: v for k, v in row.items() if k.startswith(PINNED)} for row in writer.rows],
    }


@pytest.mark.parametrize("name", sorted(CASES))
def test_golden_training(name):
    found = record(name)
    golden = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
    if os.environ.get("CATANRL_UPDATE_GOLDEN"):
        golden[name] = found
        GOLDEN.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n")
    assert name in golden, "no golden training yet: run with CATANRL_UPDATE_GOLDEN=1"
    assert found["weights"] == golden[name]["weights"]
    assert len(found["metrics"]) == len(golden[name]["metrics"])
    for epoch, (row, pinned) in enumerate(zip(found["metrics"], golden[name]["metrics"])):
        assert {k: row.get(k) for k in pinned} == pinned, f"epoch row {epoch}"
