"""Test 2: derive_seed equals v1, labels are checked, determinism restores flags."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from catanrl.seeding import derive_seed, determinism, rng_state, seed_everything, set_rng_state

#: Computed with v1's catanrl.utils.seeding.derive_seed.
V1 = {
    (42, "init"): 6085316617551129739,
    (42, "sample", 0): 9147917377732318731,
    (42, "rollout", 0): 650873587555472283,
    (123, "select", "abc", 0): 4882798855712450158,
    (0,): 6912158355717386040,
    (7, "map_pool"): 5170867417530434070,
    (67, "heldout", 3, "bots", 1): 1302919384634790851,
}


def test_derive_seed_equals_v1():
    for args, expected in V1.items():
        assert derive_seed(*args) == expected


def test_pipe_in_a_label_is_rejected():
    with pytest.raises(ValueError):
        derive_seed(1, "a|b")


def test_determinism_restores_flags():
    before = (torch.backends.cudnn.benchmark, torch.are_deterministic_algorithms_enabled(),
              torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.benchmark = True
    with determinism("strict"):
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cudnn.benchmark
        assert torch.get_float32_matmul_precision() == "highest"
    assert torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = before[0]
    assert (torch.backends.cudnn.benchmark, torch.are_deterministic_algorithms_enabled(),
            torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32) == before
    with pytest.raises(ValueError), determinism("loose"):
        pass


def test_rng_state_round_trips_through_weights_only(tmp_path):
    seed_everything(5)
    state = rng_state()
    torch.save(state, tmp_path / "rng.pt")
    expected = (random.random(), np.random.rand(), torch.rand(1).item())
    set_rng_state(torch.load(tmp_path / "rng.pt", weights_only=True))
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == expected
