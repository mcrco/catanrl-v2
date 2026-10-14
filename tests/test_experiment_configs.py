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


def test_ppo_4p_is_puffer_graph2_per_decision():
    """ppo-4p is puffer-graph2's recipe at four rows per game, matched per decision (one row in
    four acts): about 66k decisions per epoch, 8192 decisions per optimizer step and 16 steps
    per epoch with no micro-batch left over, v1's 250M decisions over the same ~3800-epoch
    cosine, a clip scaled with the accumulated micro-batches, a private shared trunk, pure
    self-play, and a rollout buffer that leaves the 8 GB card room to train."""
    from catanrl.envs.encoding import dims_for, row_layout, row_size
    from catanrl.games import load_game
    from catanrl.train.puffer import build_policy_model, puffer_config, rollout_bytes

    ref = load_plan("puffer-graph2").variant("full").config
    plan = load_plan("ppo-4p")
    assert [v.name for v in plan.variants] == ["main"]
    cfg = plan.variants[0].config
    (scenario,) = cfg["env"]["scenarios"]
    assert scenario["game"]["name"] == "base-4p" and not scenario.get("seats")  # every seat learns
    assert scenario["turn_limit"] == 1000 and scenario["maps"] == 4096
    assert cfg["obs"] == {"level": "private", "critic_level": None} and cfg["model"]["args"]["shared"]

    same = ("optimizer", "learning_rate", "lr_schedule", "anneal_lr", "min_lr_ratio", "adam_beta1", "gamma",
            "gae_lambda", "update_epochs", "clip_coef", "vf_coef", "ent_coef", "bptt_horizon", "prio_alpha",
            "prio_beta0", "vtrace_rho_clip", "vtrace_c_clip", "precision", "amp", "checkpoint_interval")
    assert {k: cfg["train"][k] for k in same} == {k: ref["train"][k] for k in same}

    num_envs, ref_envs = cfg["env"]["num_envs"], ref["env"]["num_envs"]
    config = puffer_config(cfg["train"], cfg["runtime"], rows=4 * num_envs, seed=0, data_dir=".")
    ref_config = puffer_config(ref["train"], ref["runtime"], rows=2 * ref_envs, seed=0, data_dir=".")
    horizon = config["bptt_horizon"]
    assert config["batch_size"] == 262_144 and num_envs * horizon == 65_536   # decisions per epoch
    assert abs(num_envs / ref_envs - 1) < 0.01                                   # 2p: 66,048
    assert config["total_timesteps"] // 4 == ref_config["total_timesteps"] // 2 == 250_000_000
    assert config["total_timesteps"] // config["batch_size"] == 3814           # cosine T_max (2p: 3785)

    def steps(c, players):
        micro = c["update_epochs"] * c["batch_size"] // c["max_minibatch_size"]
        accumulate = c["minibatch_size"] // c["max_minibatch_size"]
        return micro, accumulate, c["minibatch_size"] // players                # decisions per step

    micro, accumulate, decisions = steps(config, 4)
    ref_micro, ref_accumulate, ref_decisions = steps(ref_config, 2)
    assert config["update_epochs"] * config["batch_size"] % config["max_minibatch_size"] == 0
    assert micro % accumulate == 0 and micro // accumulate == ref_micro // ref_accumulate == 16
    assert decisions == ref_decisions == 8192
    # PuffeRL sums the micro-batch means: twice the micro-batches, twice the clip
    assert config["max_grad_norm"] / accumulate == ref_config["max_grad_norm"] / ref_accumulate

    dims = dims_for([load_game("base-4p")])
    assert sum(p.numel() for p in build_policy_model(cfg, dims).parameters()) == 541_540
    obs = row_size(row_layout(dims, "private"))
    assert obs == 3202 and rollout_bytes(4 * num_envs, horizon, obs) < 3.2 * 2**30

    assert cfg["eval"]["select"] == {"suite": "select-4p", "lineup": ["POLICY:ckpt=${self}", "F", "F", "F"],
                                     "every": 50, "metric": "score_rate"}
