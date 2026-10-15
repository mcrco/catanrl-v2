"""train.muon_exclude: matched parameters are stepped by AdamW, the rest by the main
optimizer exactly as without them; AdamW's learning rate follows the main one's
fraction; its state rides in the main optimizer's state_dict."""

from __future__ import annotations

import copy

import heavyball
import pytest
import torch
from torch import nn

from catanrl.train.muon_exclude import KEY, exclude_from_muon

EXCLUDED = "2.weight"  # the 1 x 8 output layer


@pytest.fixture(autouse=True)
def _eager_heavyball(monkeypatch):
    monkeypatch.setattr(heavyball.utils, "compile_mode", None)  # same math, no inductor build


def muon(params, lr=0.01):
    """PuffeRL's Muon (pufferl.PuffeRL.__init__)."""
    return heavyball.ForeachMuon(params, lr=lr, betas=(0.95, 0.999), eps=1e-12, heavyball_momentum=True)


def net() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 1))


def backward(model: nn.Module, step: int) -> None:
    x = torch.randn(16, 6, generator=torch.Generator().manual_seed(step))
    (model(x) - 0.5).pow(2).mean().backward()


def test_matched_parameters_go_to_adamw_and_the_rest_to_muon():
    model, reference, plain = net(), net(), net()
    optimizer = muon(model.parameters())
    exclude_from_muon(optimizer, model, [EXCLUDED], lr=1e-3, base_lr=0.01)
    ref_muon = muon([p for name, p in reference.named_parameters() if name != EXCLUDED])
    ref_adamw = torch.optim.AdamW([reference[2].weight], lr=1e-3)
    plain_muon = muon(plain.parameters())  # Muon on everything
    for step in range(4):
        for m in (model, reference, plain):
            backward(m, step)
        optimizer.step()
        optimizer.zero_grad()
        for opt in (ref_muon, ref_adamw, plain_muon):
            opt.step()
            opt.zero_grad()
        for (name, p), q, r in zip(model.named_parameters(), reference.parameters(), plain.parameters()):
            assert torch.equal(p, q), (step, name)
            if step == 0:  # same gradients: only the excluded layer's step differs from Muon's
                assert torch.equal(p, r) != (name == EXCLUDED), name


def test_adamw_follows_the_main_learning_rate():
    model = net()
    optimizer = muon(model.parameters(), lr=0.01)
    adamw = exclude_from_muon(optimizer, model, ["2.*"], lr=1e-3, base_lr=0.01)
    assert [p for g in adamw.param_groups for p in g["params"]] == [model[2].weight, model[2].bias]
    for group in optimizer.param_groups:
        group["lr"] = 0.0025  # a cosine, the KL schedule or a restore
    backward(model, 0)
    optimizer.step()
    assert adamw.param_groups[0]["lr"] == pytest.approx(2.5e-4)


def test_adamw_state_rides_in_the_main_state_dict():
    model = net()
    optimizer = muon(model.parameters())
    adamw = exclude_from_muon(optimizer, model, [EXCLUDED], lr=1e-3, base_lr=0.01)
    for step in range(2):
        backward(model, step)
        optimizer.step()
        optimizer.zero_grad()
    state = copy.deepcopy(optimizer.state_dict())
    assert KEY in state and state[KEY]["state"][0]["step"] == 2
    fresh = net()
    restored = muon(fresh.parameters())
    adamw_restored = exclude_from_muon(restored, fresh, [EXCLUDED], lr=1e-3, base_lr=0.01)
    restored.load_state_dict(state)
    assert torch.equal(adamw_restored.state_dict()["state"][0]["exp_avg"], adamw.state_dict()["state"][0]["exp_avg"])
    assert KEY in state  # the caller's dict is left alone


def test_a_pattern_must_match():
    model = net()
    with pytest.raises(ValueError, match="match no parameter"):
        exclude_from_muon(muon(model.parameters()), model, ["value_head.*"], lr=1e-3, base_lr=0.01)


@pytest.mark.train
def test_training_routes_the_value_output_layer(tmp_path):
    """Through train_ppo (Adam as the main optimizer here): the critic's output layer has
    AdamW state, stepped at every optimizer step, and no state in the main optimizer."""
    from run_helpers import train
    from train_configs import tiny_ppo

    from catanrl.config import resolve
    from catanrl.envs.encoding import dims_for
    from catanrl.games import load_game
    from catanrl.train.puffer import build_policy_model, train_ppo

    name = "critic.value.mlp.4.weight"
    config = tiny_ppo(epochs=2, muon_exclude=[name], muon_exclude_lr=1e-4)
    optimizer = train(train_ppo, config, "excl").load_trainer_state()["optimizer"]
    assert optimizer[KEY]["param_groups"][0]["lr"] == pytest.approx(1e-4)
    steps = {int(s["step"]) for s in optimizer[KEY]["state"].values()}
    assert steps == {int(s["step"]) for s in optimizer["state"].values()} == {4}  # 2 epochs x 2 steps
    model = build_policy_model(resolve(config, experiment="t", variant="v"), dims_for([load_game("base-2p")]))
    names = [n for n, _ in model.named_parameters()]  # PuffeRL's optimizer order
    assert names.index(name) not in optimizer["state"]
    assert names.index("critic.value.mlp.4.bias") in optimizer["state"]
