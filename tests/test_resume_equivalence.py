"""Test 16: a run stopped and resumed restores its weights, optimizer,
scheduler, epoch, step and RNG exactly, and continues the LR schedule.
Games in flight restart, so later weights are not compared."""

from __future__ import annotations

import pytest
import torch
from learning_standins import LocalVariant, LocalWriter
from train_configs import tiny_ppo

from catanrl.train.puffer import train_ppo

pytestmark = pytest.mark.train


def lr_by_epoch(writer):
    return {row["epoch"]: row["learning_rate"] for row in writer.metrics}


def equal(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def test_resume_continues_exactly(tmp_path, monkeypatch):
    config = tiny_ppo(epochs=4, anneal_lr=True, min_lr_ratio=0.1)
    straight = LocalWriter(tmp_path / "straight")
    train_ppo(LocalVariant(config), straight)

    stopped = LocalWriter(tmp_path / "resumed", stop_after_epochs=2)
    train_ppo(LocalVariant(config), stopped)
    assert stopped.status == "interrupted"
    saved = stopped.load_trainer_state()
    assert saved["epoch"] == 2

    # Capture the trainer's state right after restore.
    from catanrl.train import puffer

    restored = {}
    original = puffer.CatanPuffeRL.restore

    def spy(self, state):
        original(self, state)
        restored.update(self.state())

    monkeypatch.setattr(puffer.CatanPuffeRL, "restore", spy)
    resumed = LocalWriter(tmp_path / "resumed", mode="resume", launch=1)
    train_ppo(LocalVariant(config), resumed)
    assert resumed.status == "finished"

    for key in ("epoch", "global_step", "decisions", "model", "optimizer", "scheduler", "learning_rate",
                "map_pool"):
        assert equal(restored[key], saved[key]), key
    assert equal(restored["rng"]["torch"], saved["rng"]["torch"])
    assert [row["epoch"] for row in resumed.metrics] == [3, 4]
    assert lr_by_epoch(resumed) == {e: lr for e, lr in lr_by_epoch(straight).items() if e > 2}
    final, reference = resumed.load_trainer_state(), straight.load_trainer_state()
    assert final["scheduler"]["last_epoch"] == reference["scheduler"]["last_epoch"]
    steps = lambda state: {int(s["step"]) for s in state["optimizer"]["state"].values()}
    assert steps(final) == steps(reference)


def test_resume_keeps_the_kl_adapted_learning_rate(tmp_path, monkeypatch):
    """The committed state holds the learning rate the next epoch uses, adaptation included."""
    from catanrl.train import puffer

    used = {}
    original = puffer.CatanPuffeRL.train

    def train(self):
        used[(self.writer.path.name, self.epoch + 1)] = self.optimizer.param_groups[0]["lr"]
        return original(self)

    monkeypatch.setattr(puffer.CatanPuffeRL, "train", train)
    # A tiny KL target: every epoch divides the learning rate by 1.5.
    config = tiny_ppo(epochs=4, lr_schedule="kl_adaptive", kl_target=1e-9, lr_min=1e-6, lr_max=1e-1)
    train_ppo(LocalVariant(config), LocalWriter(tmp_path / "straight"))
    stopped = LocalWriter(tmp_path / "run", stop_after_epochs=2)
    train_ppo(LocalVariant(config), stopped)
    assert stopped.load_trainer_state()["learning_rate"] == used[("straight", 3)]
    train_ppo(LocalVariant(config), LocalWriter(tmp_path / "run", mode="resume", launch=1))
    assert used[("run", 3)] == used[("straight", 3)] != used[("straight", 2)]
