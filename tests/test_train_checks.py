"""Checks train_ppo and train_imitation make before training."""

from __future__ import annotations

import pytest

from catanrl.envs.puffer import plan_env
from catanrl.train.puffer import build_policy_model, check_bindable


@pytest.mark.parametrize("preset, dims, ok", [
    ("mlp", "pool", False),       # trained at the pool's dims: evals (generator bounds) cannot bind it
    ("mlp", "auto", True),
    ("graph-tiny", "pool", True),  # dims-agnostic
])
def test_networks_must_be_bindable_by_evals(preset, dims, ok):
    plan = plan_env({"scenarios": [{"game": "random-4p", "maps": 8}], "dims": dims}, pool_seed=3)
    model = build_policy_model({"model": {"preset": preset}, "obs": {"level": "private"}}, plan.dims)
    if ok:
        check_bindable(model, plan)
    else:
        with pytest.raises(ValueError, match="dims-agnostic"):
            check_bindable(model, plan)
