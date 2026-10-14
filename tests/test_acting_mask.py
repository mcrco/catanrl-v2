"""PuffeRL's acting-row mask: only steps whose row acted weigh in the
prioritized sampling, the advantage normalization and the loss/KL means."""

from __future__ import annotations

import pufferlib.pytorch
import pytest
import torch
from run_helpers import train
from train_configs import tiny_ppo

from catanrl.train import puffer
from catanrl.train.puffer_policy import PufferPolicy


def config(prio_alpha: float):
    cfg = tiny_ppo(epochs=2, learning_rate=0.0, update_epochs=2, prio_alpha=prio_alpha, prio_beta0=0.5)
    # Padding rows (2p games in 4-seat rows) and bot rows (R never acts).
    cfg["env"]["dims"] = {"max_players": 4, "max_tiles": 19, "max_nodes": 54, "max_edges": 72}
    return cfg


@pytest.mark.parametrize("prio_alpha", [0.0, 0.8])
def test_only_acting_steps_count(tmp_path, monkeypatch, prio_alpha):
    batches = []
    original = PufferPolicy.value_loss

    def spy(self, outputs, returns, *, idx, values, advantages):
        batches.append(idx.clone())
        return original(self, outputs, returns, idx=idx, values=values, advantages=advantages)

    trainers = []
    original_init = puffer.CatanPuffeRL.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        trainers.append(self)

    monkeypatch.setattr(PufferPolicy, "value_loss", spy)
    monkeypatch.setattr(puffer.CatanPuffeRL, "__init__", init)
    writer = train(puffer.train_ppo, config(prio_alpha), "mask")
    trainer = trainers[0]
    policy = trainer.uncompiled_policy
    acting = policy.acting_mask(trainer.observations)
    assert 0 < acting.float().mean() < 0.5  # most rows wait, are bots or padding
    # Segments that never act (bots, padding) are never sampled.
    last_epoch = batches[-trainer.total_minibatches:]
    for idx in last_epoch:
        assert bool(acting[idx].any(axis=1).all())
    # With learning rate 0 the logged entropy is the mean over minibatches of the
    # entropy of their acting steps (not diluted by the waiting rows' zeros).
    expected = []
    with torch.no_grad():
        for idx in last_epoch:
            obs = trainer.observations[idx].reshape(-1, trainer.observations.shape[-1])
            logits, _ = policy(obs)
            _, _, entropy = pufferlib.pytorch.sample_logits(logits, action=trainer.actions[idx])
            mask = acting[idx].reshape(-1).float()
            expected.append(float((entropy * mask).sum() / mask.sum()))
    logged = writer.rows[-1]["losses/entropy"]
    assert logged == pytest.approx(sum(expected) / len(expected), rel=1e-4)
