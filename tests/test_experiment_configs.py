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
    assert len(plan.variants) == 12
    ppo, g1 = plan.variant("graph-m-s43.ppo"), plan.variant("graph-m-s43.g1")
    assert ppo.config["train"]["algo"] == "ppo" and ppo.config["init_from"] is None and ppo.depends_on == ()
    assert g1.depends_on == ("graph-m-s43.ppo",) and g1.config["train"]["gamma"] == 1.0
    assert g1.config["init_from"]["ckpt"] == "arch-lineage/graph-m-s43.ppo@3000"


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
    assert abs(config["total_timesteps"] // 4 - ref_config["total_timesteps"] // 2) < 1_000_000  # ~250M decisions
    assert config["total_timesteps"] == 3800 * config["batch_size"]  # whole epochs, a multiple of the select interval

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


def test_ppo_4p_critic_fix_is_ppo_4p_but_the_critic():
    """Fine-tunes of ppo-4p's latest weights for 100 of its epochs at its cosine floor, all
    with lambda critic targets: gamma 0.999, the critic undiscounted, both undiscounted, and
    the value output layer off Muon; everything else is ppo-4p's recipe."""
    ref = load_plan("ppo-4p").variants[0].config
    plan = load_plan("ppo-4p-critic-fix")
    assert [v.name for v in plan.variants] == ["lambda", "lambda-critic-g1", "lambda-g1", "lambda-adamw"]
    changed = {"total_timesteps", "learning_rate", "anneal_lr", "gamma", "critic_targets", "critic_gamma",
               "muon_exclude", "muon_exclude_lr"}
    for v in plan.variants:
        cfg = v.config
        assert cfg["init_from"] == {"ckpt": "ppo-4p/main@latest", "parts": ["actor", "critic"], "restore": "weights"}
        assert {k: cfg[k] for k in ("seed", "env", "obs", "model")} == {k: ref[k] for k in ("seed", "env", "obs", "model")}
        assert cfg["eval"]["select"] == ref["eval"]["select"]
        assert {k: cfg["train"][k] for k in cfg["train"] if k not in changed} == \
            {k: ref["train"][k] for k in ref["train"] if k not in changed}
        assert cfg["train"]["total_timesteps"] // (4096 * 64) == 100
        assert cfg["train"]["learning_rate"] == ref["train"]["learning_rate"] * ref["train"]["min_lr_ratio"]
        assert cfg["train"]["critic_targets"] == "lambda"
    options = {v.name: {k: v.config["train"][k] for k in ("gamma", "critic_gamma", "muon_exclude")}
               for v in plan.variants}
    assert options == {
        "lambda": {"gamma": 0.999, "critic_gamma": None, "muon_exclude": []},
        "lambda-critic-g1": {"gamma": 0.999, "critic_gamma": 1.0, "muon_exclude": []},
        "lambda-g1": {"gamma": 1.0, "critic_gamma": None, "muon_exclude": []},
        "lambda-adamw": {"gamma": 0.999, "critic_gamma": None, "muon_exclude": ["value_head.mlp.4.weight"]},
    }


def test_az_2p_2048_starts_from_puffer_graph2_public():
    """The owner's az-2p-2048 (2026-10-21): puffer-graph2/public's network, rows, rules and
    training pool, so init_from loads e3600 whole (graph-s alone is unshared); fair Gumbel
    targets at 2048 simulations on every move; a select whose first 200 boards are
    select-2p's; raw finals only, the BELIEF anchor paired on heldout-2p; and the search
    finals (az-2p-2048-search) at exactly the training search."""
    from catanrl.eval.suites import load_suite

    ref = load_plan("puffer-graph2").variant("public").config
    (v,) = load_plan("az-2p-2048").variants
    cfg = v.config
    assert {k: cfg[k] for k in ("seed", "obs", "model")} == {k: ref[k] for k in ("seed", "obs", "model")}
    assert cfg["model"]["args"]["shared"] and cfg["obs"]["level"] == "public"
    assert cfg["env"]["scenarios"] == ref["env"]["scenarios"] and cfg["env"]["map_pool"] == ref["env"]["map_pool"]
    assert cfg["init_from"] == {"ckpt": "puffer-graph2/public@e3600", "parts": ["actor", "critic"]}
    train = cfg["train"]
    assert {k: train[k] for k in ("root", "sims", "full_search_prob", "dets", "gumbel_m", "dirichlet_noise",
                                  "iterations", "checkpoint_interval")} == {
        "root": "gumbel", "sims": 2048, "full_search_prob": 1.0, "dets": 1, "gumbel_m": 16, "dirichlet_noise": False,
        "iterations": 30, "checkpoint_interval": 2}

    select = cfg["eval"]["select"]
    assert select["lineup"] == ["POLICY:ckpt=${self}", "POLICY:ckpt=${init}"] and select["every"] == 1
    suite, small = load_suite(select["suite"]), load_suite("select-2p")
    assert (suite.purpose, suite.boards) == ("select", 2000)
    assert type(suite)(**{**suite.__dict__, "boards": small.boards}).schedule_hash(2) == small.schedule_hash(2)

    finals = [(f["suite"], f["lineup"]) for f in cfg["eval"]["final"]]
    best, anchor = f"POLICY:ckpt={v.ref}@best", "BELIEF:ckpt=puffer-graph2/full@e3350:worlds=16"
    assert finals == [("heldout-2p", [best, "POLICY:ckpt=${init}"]), ("heldout-2p", [best, "F"]),
                      ("heldout-2p", [best, anchor]), ("heldout-2p", [f"POLICY:ckpt={v.ref}@init", anchor])]

    search = (f"sims={train['sims']}:dets=1:root={train['root']}:c_puct={train['c_puct']}"
              f":gumbel_m={train['gumbel_m']}")
    (s,) = load_plan("az-2p-2048-search").variants
    assert s.kind == "eval" and [(f["suite"], f["lineup"]) for f in s.config["eval"]["final"]] == [
        ({"name": "heldout-2p", "boards": 500}, [f"MCTS:ckpt={v.ref}@{which}:{search}", "POLICY:ckpt=puffer-graph2/public@e3600"])
        for which in ("best", "init")]


def test_critic_refit_2p_is_puffer_graph2_public_with_only_the_critic_training():
    """Marco's critic refit (2026-10-22): puffer-graph2/public@e3600's network, rows, rules and
    training pool, the policy frozen (split into two trunks in main, the shared trunk in head),
    undiscounted outcome targets over 256-step segments in the start's rollout memory budget,
    uniform segments, one pass per rollout, 500 epochs; the search rows exactly
    az-2p-2048-soft's select (512 sims, select-2p-500) and gumbel-edge-2p's final (2048 sims,
    heldout-2p's first 500 boards), against the start's raw policy."""
    from catanrl.envs.encoding import dims_for, row_layout, row_size
    from catanrl.games import load_game
    from catanrl.train.puffer import build_policy_model, puffer_config, rollout_bytes

    ref = load_plan("puffer-graph2").variant("public").config
    soft = load_plan("az-2p-2048-soft").variants[0].config
    edge = load_plan("gumbel-edge-2p").variants[0].config
    plan = load_plan("critic-refit-2p")
    assert [v.name for v in plan.variants] == ["main", "head"]
    dims = dims_for([load_game("base-2p")])
    trained = {}
    for v in plan.variants:
        cfg = v.config
        assert {k: cfg[k] for k in ("seed", "obs")} == {k: ref[k] for k in ("seed", "obs")}
        assert cfg["env"]["scenarios"] == ref["env"]["scenarios"] and cfg["env"]["map_pool"] == ref["env"]["map_pool"]
        args = {k: a for k, a in cfg["model"]["args"].items() if k != "shared"}
        assert args == {k: a for k, a in ref["model"]["args"].items() if k != "shared"}
        assert cfg["model"]["args"]["shared"] == (v.name == "head")
        assert cfg["init_from"] == {"ckpt": "puffer-graph2/public@e3600", "parts": ["actor", "critic"],
                                    **({"split_shared": True} if v.name == "main" else {})}
        train = cfg["train"]
        assert train["freeze"] == ["actor"]
        assert (train["critic_targets"], train["critic_gamma"], train["gae_lambda"]) == ("lambda", 1.0, 1.0)
        assert (train["update_epochs"], train["prio_alpha"], train["optimizer"]) == (1, 0.0, "adam")
        rows = 2 * cfg["env"]["num_envs"]
        config = puffer_config(train, cfg["runtime"], rows=rows, seed=0, data_dir=".")
        assert config["batch_size"] == 264_192 and config["total_timesteps"] // config["batch_size"] == 500
        assert config["total_timesteps"] % config["batch_size"] == 0
        assert config["bptt_horizon"] == 256 and config["minibatch_size"] // config["max_minibatch_size"] == 4
        obs = row_size(row_layout(dims, "public"))
        start_rollout = rollout_bytes(2 * ref["env"]["num_envs"], ref["train"]["bptt_horizon"], obs)
        assert rollout_bytes(rows, config["bptt_horizon"], obs) == 2 * start_rollout < 2.6 * 2**30

        select = cfg["eval"]["select"]
        assert {k: select[k] for k in ("suite", "lineup", "metric")} == \
            {k: soft["eval"]["select"][k] for k in ("suite", "lineup", "metric")}
        assert select["every"] == 500  # e0 and the last epoch only
        (edge_final,) = edge["eval"]["final"]
        (final,) = cfg["eval"]["final"]
        assert final["suite"] == edge_final["suite"]
        assert final["lineup"] == [edge_final["lineup"][0].replace("puffer-graph2/public@e3600", f"{v.ref}@latest"),
                                   "POLICY:ckpt=${init}"]
        model = build_policy_model(cfg, dims)
        assert sum(p.numel() for p in model.parameters()) == (541_540 if v.name == "head" else 843_652)
        trained[v.name] = sum(p.numel() for n, p in model.named_parameters()
                              if not n.startswith(tuple(model.parts["actor"])))
    assert trained == {"main": 376_225, "head": 74_113}


def test_ppo_2p_fr_is_puffer_graph2_full_continued():
    """Marco's friendly-robber continuation (2026-10-24): puffer-graph2/full@best (e3350) trained
    on for ~1B agent steps on base-2p-fr (base-2p plus OpenHex's friendly robber and balanced dice), its recipe key
    for key but the budget and the learning rate (half the start's peak, the same 0.1x cosine),
    from its weights (init_from, a fresh optimizer); best follows the champion monitor (POLICY
    vs the start), F is a monitor; the finals on heldout-2p-fr pair @best and the start."""
    from catanrl.train.puffer import puffer_config

    ref = load_plan("puffer-graph2").variant("full").config
    (v,) = load_plan("ppo-2p-fr").variants
    cfg = v.config
    assert v.name == "full" and cfg["obs"] == ref["obs"] == {"level": "full", "critic_level": None}
    assert {k: cfg[k] for k in ("seed", "model", "runtime")} == {k: ref[k] for k in ("seed", "model", "runtime")}
    assert {k: cfg["env"][k] for k in cfg["env"] if k != "scenarios"} == \
        {k: ref["env"][k] for k in ref["env"] if k != "scenarios"}
    (scenario,), (start,) = cfg["env"]["scenarios"], ref["env"]["scenarios"]
    assert {**scenario, "game": None} == {**start, "game": None}  # turn limit, maps, weight, seats
    assert {**scenario["game"], "name": None, "friendly_robber": None, "dice_mode": None} == \
        {**start["game"], "name": None, "friendly_robber": None, "dice_mode": None}
    assert scenario["game"]["friendly_robber"] and not start["game"]["friendly_robber"]
    assert scenario["game"]["dice_mode"] == "balanced" and "dice_mode" not in start["game"]  # start: fair dice

    changed = {"total_timesteps", "learning_rate"}
    assert {k: cfg["train"][k] for k in cfg["train"] if k not in changed} == \
        {k: ref["train"][k] for k in ref["train"] if k not in changed}
    assert cfg["train"]["learning_rate"] == ref["train"]["learning_rate"] / 2
    config = puffer_config(cfg["train"], cfg["runtime"], rows=2 * cfg["env"]["num_envs"], seed=0, data_dir=".")
    epochs = config["total_timesteps"] // config["batch_size"]
    assert config["batch_size"] == 132_096 and config["total_timesteps"] == epochs * config["batch_size"]
    assert epochs == 7600 and epochs % config["checkpoint_interval"] == 0  # the last epoch is a select epoch

    assert cfg["init_from"] == {"ckpt": "puffer-graph2/full@best#3355b5059503", "parts": ["actor", "critic"]}
    assert "champion" not in cfg["eval"]  # ${champion} is ${init}: the start
    champion = {"name": "champion", "suite": "select-2p-fr", "lineup": ["POLICY:ckpt=${self}", "POLICY:ckpt=${init}"],
                "every": 50, "metric": "score_rate"}
    f = {"name": "f", "suite": "select-2p-fr", "lineup": ["POLICY:ckpt=${self}", "F"], "every": 50,
         "metric": "score_rate"}
    assert cfg["eval"]["monitor"] == [champion, f]
    assert cfg["eval"]["select"] == {**{k: x for k, x in champion.items() if k != "name"}, "from": "champion"}
    best, start_ref = f"{v.ref}@best", "${init}"
    finals = [(e["suite"], e["lineup"]) for e in cfg["eval"]["final"]]
    assert finals == [("heldout-2p-fr", lineup) for lineup in (
        [f"POLICY:ckpt={best}", "F"], [f"BELIEF:ckpt={best}:worlds=16", "F"],
        [f"POLICY:ckpt={start_ref}", "F"], [f"BELIEF:ckpt={start_ref}:worlds=16", "F"],
        [f"POLICY:ckpt={best}", f"POLICY:ckpt={start_ref}"],
        [f"BELIEF:ckpt={best}:worlds=16", f"BELIEF:ckpt={start_ref}:worlds=16"])]
