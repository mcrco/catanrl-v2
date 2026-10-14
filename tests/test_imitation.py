"""DAgger from F: labels, outcome targets and a short run."""

from __future__ import annotations

import cppanatron as cpp
import numpy as np
import pytest
import torch
from run_helpers import train

from catanrl.envs.puffer import plan_env
from catanrl.train.imitation import IMITATION_KEYS, _Collector, _Dataset, train_imitation
from catanrl.train.puffer import build_policy_model

CONFIG = {
    "seed": 5,
    "env": {"scenarios": [{"game": "base-2p", "maps": 4}, {"game": "base-3p", "maps": 4, "seats": ["learner", "F", "R"]}],
            "num_envs": 6},
    "obs": {"level": "private"},
    "model": {"preset": "graph-tiny"},
    "train": {**IMITATION_KEYS, "iterations": 2, "steps_per_iteration": 24, "checkpoint_interval": 1,
              "dataset_cap": 5000, "minibatch_size": 64},
    "runtime": {"device": "cpu", "threads": 2},
}


def test_labels_are_the_expert_and_targets_the_outcome():
    plan = plan_env(CONFIG["env"], pool_seed=1)
    model = build_policy_model(CONFIG, plan.dims)
    collector = _Collector(plan, CONFIG["train"], num_envs=6, threads=2, seed=3, level="private",
                           enemies=cpp.ValueEnemies.NEXT)
    dataset = _Dataset(10_000)
    collector.collect(model, "cpu", 400, beta=1.0, rng=np.random.default_rng(0), dataset=dataset)
    assert dataset.size > 0
    rows, labels, targets, present = dataset.arrays()
    size = int(collector.vec.action_size)
    masks = rows[:, -size:].astype(np.float32)
    assert np.all(masks[np.arange(len(labels)), labels] > 0)       # every label is legal
    assert np.all(targets.sum(1) <= 1) and np.all(targets <= present)  # one winner at most, a seated one
    assert np.any(targets.sum(1) == 1)


def test_dataset_drops_the_oldest_chunks():
    dataset = _Dataset(10)  # keeps whole chunks: the fewest newest chunks holding >= 10 rows
    for i in range(5):
        dataset.add(np.full((4, 3), i), np.zeros(4), np.zeros((4, 2)), np.ones((4, 2)))
    rows, *_ = dataset.arrays()
    assert dataset.size == len(rows) == 12
    assert sorted(set(rows[:, 0].tolist())) == [2, 3, 4]


def test_short_run(tmp_path):
    writer = train(train_imitation, CONFIG, "dagger")
    assert writer.status == "finished"
    assert [ref.epoch for ref in writer.saved] == [0, 1, 2]
    assert writer.rows[-1]["beta"] == pytest.approx(0.9)
    assert "imitation/accuracy" in writer.rows[-1]
    state = writer.load_trainer_state()
    assert state["epoch"] == 2 and isinstance(state["model"], dict)
    assert all(isinstance(v, torch.Tensor) for v in state["model"].values())


def test_unknown_or_missing_keys_raise(tmp_path):
    bad = {**CONFIG, "train": {**CONFIG["train"], "epochs": 3}}
    with pytest.raises(ValueError):
        train(train_imitation, bad, "bad")


def test_labels_are_value_action():
    """The engine's expert labels (E7) are F's value_action at every acting row."""
    plan = plan_env(CONFIG["env"], pool_seed=2)
    collector = _Collector(plan, CONFIG["train"], num_envs=6, threads=2, seed=4, level="private",
                           enemies=cpp.ValueEnemies.NEXT)
    vec, space = collector.vec, collector.vec.encoder.action_space
    buffers = collector.buffers
    checked = 0
    for _ in range(60):
        acting = np.flatnonzero(buffers["acting"])
        labels = vec.expert_actions(enemies=cpp.ValueEnemies.NEXT)
        for row in acting:
            game = vec.game(int(buffers["env_index"][row]))
            assert labels[row] == space.index(game, cpp.value_action(game, enemies=cpp.ValueEnemies.NEXT))
            buffers["actions"][row] = labels[row]
            checked += 1
        assert np.all(np.delete(labels, acting) == -1)
        vec.step()
    assert checked >= 300


def test_resume_restores_the_dataset_decisions_and_rng(tmp_path, monkeypatch):
    """A resumed DAgger run continues from the aggregate it stopped with."""
    from catanrl.train import imitation

    config = {**CONFIG, "train": {**CONFIG["train"], "iterations": 4, "checkpoint_interval": 2}}
    straight = train(train_imitation, config, "straight")
    stopped = train(train_imitation, config, "run", stop_after=2)
    saved = stopped.load_trainer_state()
    assert saved["epoch"] == 2 and saved["dataset"]["rows"].shape[0] == straight.rows[1]["dataset"]

    restored = {}
    original = imitation._Dataset.restore

    def spy(self, state):
        original(self, state)
        restored["size"] = self.size

    def spy_rng(state):
        set_rng = imitation.__dict__["_set_rng_state_original"]
        set_rng(state)
        restored["rng"] = torch.get_rng_state()

    monkeypatch.setattr(imitation._Dataset, "restore", spy)
    monkeypatch.setitem(imitation.__dict__, "_set_rng_state_original", imitation.set_rng_state)
    monkeypatch.setattr(imitation, "set_rng_state", spy_rng)
    resumed = train(train_imitation, config, "run")
    assert restored["size"] == saved["dataset"]["rows"].shape[0]
    assert torch.equal(restored["rng"], saved["rng"]["torch"])
    first = resumed.rows[0]
    assert first["epoch"] == 3 and first["decisions"] > saved["decisions"]
    assert first["dataset"] >= restored["size"] > 0  # the aggregate survives (games in flight restart)
