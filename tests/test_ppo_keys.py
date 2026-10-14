"""Test 5: ppo.yaml plus the runner's keys are exactly the keys PuffeRL reads."""

from __future__ import annotations

import ast
from pathlib import Path

import pufferlib
import pytest
import yaml

from catanrl.train.puffer import CATANRL_TRAIN_KEYS, PUFFERL_KEYS, RUNNER_KEYS, puffer_config

ROOT = Path(__file__).resolve().parents[1]


def pufferl_keys() -> set[str]:
    """Every config["..."] and config.get("...") inside PuffeRL."""
    source = Path(pufferlib.__file__).with_name("pufferl.py").read_text()
    cls = next(node for node in ast.parse(source).body
               if isinstance(node, ast.ClassDef) and node.name == "PuffeRL")

    def is_config(node) -> bool:
        return (isinstance(node, ast.Name) and node.id == "config") or (
            isinstance(node, ast.Attribute) and node.attr == "config")

    keys = set()
    for node in ast.walk(cls):
        if isinstance(node, ast.Subscript) and is_config(node.value) and isinstance(node.slice, ast.Constant):
            keys.add(node.slice.value)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
                and is_config(node.func.value) and node.args and isinstance(node.args[0], ast.Constant)):
            keys.add(node.args[0].value)
    return keys


def test_pufferl_keys_are_known():
    assert pufferl_keys() == set(PUFFERL_KEYS)


def full_train() -> dict:
    keys = PUFFERL_KEYS - RUNNER_KEYS
    train = {key: 0 for key in keys}
    train.update(bptt_horizon=8, minibatch_size=64, update_epochs=1, algo="ppo")
    return train


def test_puffer_config_sets_every_key():
    config = puffer_config(full_train(), {"device": "cpu"}, rows=8, seed=1, data_dir="/tmp")
    assert set(config) == set(PUFFERL_KEYS)
    assert config["batch_size"] == 64 and config["compile"] is False


def test_puffer_config_refuses_runner_keys_and_gaps():
    with pytest.raises(ValueError):
        puffer_config({**full_train(), "device": "cuda"}, {}, rows=8, seed=1, data_dir="/tmp")
    train = full_train()
    del train["gamma"]
    with pytest.raises(ValueError):
        puffer_config(train, {}, rows=8, seed=1, data_dir="/tmp")


def test_ppo_yaml_matches_pufferl():
    path = ROOT / "configs" / "train" / "ppo.yaml"
    if not path.exists():
        pytest.skip("configs/train/ppo.yaml is not written yet")
    data = yaml.safe_load(path.read_text())
    train = data.get("train", data)
    assert (set(train) - CATANRL_TRAIN_KEYS) | RUNNER_KEYS == pufferl_keys()
