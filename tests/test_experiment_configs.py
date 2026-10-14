"""Test 17: every experiment under configs/experiments resolves and validates."""

from __future__ import annotations

from pathlib import Path

import pytest

from catanrl.config import load_plan, validate_plan

EXPERIMENTS = sorted((Path(__file__).resolve().parents[1] / "configs" / "experiments").glob("*.yaml"))


@pytest.mark.parametrize("path", EXPERIMENTS, ids=[p.stem for p in EXPERIMENTS])
def test_experiment_validates(path):
    assert validate_plan(path) == []
    plan = load_plan(path, smoke=True)
    assert plan.variants and all(v.config["runtime"]["device"] == "cpu" for v in plan.variants)


def test_arch_lineage_shape():
    plan = load_plan("arch-lineage")
    assert len(plan.variants) == 18
    g1 = plan.variant("graph-m-s43.g1")
    assert g1.depends_on == ("graph-m-s43.warm",) and g1.config["train"]["gamma"] == 1.0
    assert g1.config["init_from"]["ckpt"] == "arch-lineage/graph-m-s43.warm@3000"


def test_puffer_graph2_keeps_v1_shape():
    """v1 graph2's arithmetic: 132,096 agent steps per epoch, 64 minibatches of 4096 rows
    (16 optimizer steps), cosine T_max 3785 epochs, Muon momentum 0.95, one shared trunk
    with about graph2's parameter count (the notes cite 541,540 against 537,830), and v1's
    effective value/policy balance."""
    from catanrl.envs.encoding import dims_for
    from catanrl.games import load_game
    from catanrl.train.puffer import build_policy_model, puffer_config

    plan = load_plan("puffer-graph2")
    assert [v.name for v in plan.variants] == ["full", "public"]
    for v in plan.variants:
        cfg = v.config
        assert cfg["obs"]["critic_level"] is None and cfg["model"]["args"]["shared"]
        config = puffer_config(cfg["train"], cfg["runtime"], rows=2 * cfg["env"]["num_envs"], seed=0, data_dir=".")
        assert config["batch_size"] == 132_096
        assert config["total_timesteps"] // config["batch_size"] == 3785
        assert config["update_epochs"] * config["batch_size"] // config["max_minibatch_size"] == 64
        assert config["minibatch_size"] // config["max_minibatch_size"] == 4
        assert (config["optimizer"], config["adam_beta1"], config["anneal_lr"]) == ("muon", 0.95, True)
        # v1's waiting rows halved its policy and entropy terms: twice the value weight, clip at twice v1's
        assert (config["vf_coef"], config["max_grad_norm"]) == (2.0, 2.0)
        model = build_policy_model(cfg, dims_for([load_game("base-2p")]))
        assert sum(p.numel() for p in model.parameters()) == 541_540
