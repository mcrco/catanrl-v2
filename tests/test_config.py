"""Test 4: experiment resolution, validation, hashing and migrations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from catanrl import config
from catanrl.config import ConfigError, config_diff, config_hash, load_plan, migrate

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "experiments"
BASE = {
    "seed": 1,
    "env": {"scenarios": [{"game": "base-2p"}], "num_envs": 8},
    "model": {"preset": "graph-tiny"},
    "train": {"algo": "ppo", "total_timesteps": 1000},
}


def write(tmp_path: Path, name: str = "exp", **body) -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump({"name": name, **BASE, **body}, sort_keys=False))
    return path


def test_unknown_key_suggests_and_values_are_coerced(tmp_path):
    with pytest.raises(ConfigError, match="did you mean 'learning_rate'"):
        load_plan(write(tmp_path, train={"algo": "ppo", "learnin_rate": 1e-3}))
    with pytest.raises(ConfigError, match="did you mean 'num_envs'"):
        load_plan(write(tmp_path, env={"scenarios": ["base-2p"], "num_env": 8}))
    path = tmp_path / "exp.yaml"
    text = "name: exp\nseed: 1\nenv: {scenarios: [{game: base-2p}]}\nmodel: {preset: graph-tiny}\n" \
           "train: {learning_rate: 3e-4, total_timesteps: 1_000}\n"
    path.write_text(text)
    cfg = load_plan(path).variants[0].config
    assert cfg["train"]["learning_rate"] == 3e-4 and isinstance(cfg["train"]["learning_rate"], float)
    assert cfg["train"]["total_timesteps"] == 1000
    assert cfg["train"]["gamma"] == 0.99  # completed from configs/train/ppo.yaml
    assert cfg["model"]["class"] == "graph"  # completed from configs/models/graph-tiny.yaml


def test_execution_keys_under_train_raise(tmp_path):
    with pytest.raises(ConfigError, match="runtime"):
        load_plan(write(tmp_path, train={"algo": "ppo", "device": "cuda"}))


def test_vtrace_critic_targets_take_a_null_critic_gamma(tmp_path):
    """ppo.yaml defaults critic_gamma to 1.0 (lambda's discount), so a YAML pinning
    critic_targets: vtrace alone is refused at resolution, not by the trainer after its
    run dir exists."""
    with pytest.raises(ConfigError, match="critic_gamma"):
        load_plan(write(tmp_path, train={**BASE["train"], "critic_targets": "vtrace"}))
    pinned = load_plan(write(tmp_path, train={**BASE["train"], "critic_targets": "vtrace", "critic_gamma": None}))
    assert pinned.variants[0].config["train"]["critic_gamma"] is None
    train = load_plan(write(tmp_path)).variants[0].config["train"]
    assert (train["critic_targets"], train["critic_gamma"]) == ("lambda", 1.0)  # the defaults since 2026-10-19


def test_no_default_architecture(tmp_path):
    path = write(tmp_path)
    data = yaml.safe_load(path.read_text())
    del data["model"]
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="model"):
        load_plan(path)


def test_list_tied_and_exclude_axes(tmp_path):
    sweep = {
        "axes": {
            "seed": [1, 2],
            "info": {"pf": {"obs.level": "private", "obs.critic_level": "full"}, "ff": {"obs.level": "full"}},
        },
        "name": "{info}-s{seed}",
        "exclude": [{"seed": 2, "info": "ff"}],
    }
    plan = load_plan(write(tmp_path, sweep=sweep))
    names = [v.name for v in plan.variants]
    assert names == ["pf-s1", "ff-s1", "pf-s2"]
    pf = plan.variant("pf-s2").config
    assert (pf["seed"], pf["obs"]["level"], pf["obs"]["critic_level"]) == (2, "private", "full")
    assert plan.variant("pf-s1").labels == {"seed": "1", "info": "pf"}
    assert len({v.config_hash for v in plan.variants}) == 3


def test_name_collisions_and_conflicts(tmp_path):
    with pytest.raises(ConfigError, match="both named"):
        load_plan(write(tmp_path, sweep={"axes": {"seed": [1, 2]}, "name": "same"}))
    stages = {"a": {"train.gamma": 1.0}}
    with pytest.raises(ConfigError, match="stage a and a sweep axis"):
        load_plan(write(tmp_path, stages=stages, sweep={"axes": {"train.gamma": [0.9, 0.99]}}))
    with pytest.raises(ConfigError, match="must match"):
        load_plan(write(tmp_path, sweep={"axes": {"seed": [1]}, "name": "Upper{seed}"}))


def test_stages_replace_sections_and_depend_in_order(tmp_path):
    stages = {
        "ppo": {"train!": {"algo": "ppo", "gamma": 0.9}},
        "g1": {"init_from": {"ckpt": "${stages.ppo}@best"}, "train.gamma": 1.0},
    }
    plan = load_plan(write(tmp_path, stages=stages))
    ppo, g1 = plan.variants
    assert (ppo.name, g1.name) == ("main.ppo", "main.g1")
    # train! drops the body's total_timesteps for ppo.yaml's; a dotted key merges into the body's
    assert (ppo.config["train"]["total_timesteps"], ppo.config["train"]["gamma"]) == (400_000_000, 0.9)
    assert (g1.config["train"]["total_timesteps"], g1.config["train"]["gamma"]) == (1000, 1.0)
    assert ppo.config["train"]["ent_coef"] == 0.01 and ppo.config["init_from"] is None
    assert g1.config["init_from"]["ckpt"] == "exp/main.ppo@best"
    assert g1.depends_on == ("main.ppo",) and g1.ref == "exp/main.g1"
    with pytest.raises(ConfigError, match="no earlier stage"):
        load_plan(write(tmp_path, stages={"a": {"init_from": {"ckpt": "${stages.b}@best"}}, "b": {}}))


def test_interpolation_self_and_params(tmp_path):
    ev = {"select": {"suite": "select-2p", "lineup": ["POLICY:ckpt=${self}", "F"]},
          "final": [{"suite": "heldout-2p", "lineup": ["POLICY:ckpt=${run}@best:sample=${params.s}", "F"]}]}
    cfg = load_plan(write(tmp_path, eval=ev, params={"s": True})).variants[0].config
    assert cfg["eval"]["select"]["lineup"][0] == "POLICY:ckpt=${self}"
    assert cfg["eval"]["final"][0]["lineup"][0] == "POLICY:ckpt=exp/main@best:sample=True"
    bad = {"final": [{"suite": "heldout-2p", "lineup": ["POLICY:ckpt=${self}", "F"]}]}
    with pytest.raises(ConfigError, match="only allowed in eval.select"):
        load_plan(write(tmp_path, eval=bad))
    with pytest.raises(ConfigError, match="referenced nowhere"):
        load_plan(write(tmp_path, params={"unused": 1}))
    cfg = load_plan(write(tmp_path, meta={"note": "seed ${seed}"})).variants[0].config
    assert cfg["meta"]["note"] == "seed ${seed}"  # meta is not interpolated


def test_interpolation_init_needs_the_actor_from_init_from(tmp_path):
    ev = {"select": {"suite": "select-2p", "lineup": ["POLICY:ckpt=${self}", "POLICY:ckpt=${init}"]},
          "final": [{"suite": "heldout-2p", "lineup": ["MCTS:ckpt=${init}:sims=8", "F"]}]}
    start = {"ckpt": "other/run@best"}
    cfg = load_plan(write(tmp_path, eval=ev, init_from=start)).variants[0].config
    assert cfg["eval"]["select"]["lineup"][1] == "POLICY:ckpt=${init}"  # bound when the run is created
    assert cfg["eval"]["final"][0]["lineup"][0] == "MCTS:ckpt=${init}:sims=8"
    hybrid = {"ckpt": {"actor": "other/run@best", "critic": "third/run@best"}}
    assert load_plan(write(tmp_path, eval=ev, init_from=hybrid)).variants[0].config["eval"] == cfg["eval"]
    with pytest.raises(ConfigError, match="has none"):
        load_plan(write(tmp_path, eval=ev))
    with pytest.raises(ConfigError, match="has none"):
        load_plan(write(tmp_path, eval=ev, init_from={**start, "parts": ["critic"]}))
    with pytest.raises(ConfigError, match="only allowed in eval.select and eval.final"):
        load_plan(write(tmp_path, init_from={"ckpt": "${init}"}))


def test_alphazero_from_a_network_evaluates_head_to_head_against_it(tmp_path):
    az = {"train": {"algo": "alphazero"}, "init_from": {"ckpt": "other/run@best"}}
    cfg = load_plan(write(tmp_path, **az)).variants[0].config
    assert cfg["eval"]["select"] == {"suite": "select-2p", "lineup": ["POLICY:ckpt=${self}", "POLICY:ckpt=${init}"],
                                     "every": 1, "metric": "score_rate"}
    search = "sims=64:dets=1:root=gumbel:c_puct=0.75:gumbel_m=16"  # the training search, made fair
    assert [(e["suite"], e["lineup"]) for e in cfg["eval"]["final"]] == [
        ("heldout-2p", ["POLICY:ckpt=exp/main@best", "POLICY:ckpt=${init}"]),
        ({"name": "heldout-2p", "boards": 500}, [f"MCTS:ckpt=exp/main@best:{search}", "POLICY:ckpt=${init}"]),
        ({"name": "heldout-2p", "boards": 500}, [f"MCTS:ckpt=exp/main@init:{search}", "POLICY:ckpt=${init}"]),
        ("heldout-2p", ["POLICY:ckpt=exp/main@best", "F"]),
    ]
    assert config.validate_plan(write(tmp_path, **az)) == []
    # A full-information actor plays as BELIEF; the experiment's own keys sit above the defaults.
    full = load_plan(write(tmp_path, obs={"level": "full"}, eval={"select": {"every": 5}}, **az)).variants[0].config
    assert full["eval"]["select"]["lineup"] == ["BELIEF:ckpt=${self}:worlds=16", "BELIEF:ckpt=${init}:worlds=16"]
    assert full["eval"]["select"]["every"] == 5 and full["eval"]["final"][3]["lineup"][0].startswith("BELIEF:")
    stages = {"first": {}, "az": {"train!": az["train"], "init_from": {"ckpt": "${stages.first}@best"}}}
    first, second = load_plan(write(tmp_path, stages=stages)).variants
    assert first.config["eval"] == {"select": None, "final": []}
    assert second.config["eval"]["select"]["lineup"][1] == "POLICY:ckpt=${init}"
    # From scratch there is no start to play: the global defaults.
    scratch = load_plan(write(tmp_path, train=az["train"])).variants[0].config
    assert scratch["eval"] == {"select": None, "final": []}


def test_alphazero_head_to_head_follows_the_player_count(tmp_path):
    az = {"train": {"algo": "alphazero"}, "init_from": {"ckpt": "other/run@best"}}
    four = load_plan(write(tmp_path, env={"scenarios": [{"game": "base-4p"}]}, **az)).variants[0].config
    start = ["POLICY:ckpt=${init}"] * 3
    assert four["eval"]["select"]["suite"] == "select-4p"
    assert four["eval"]["select"]["lineup"] == ["POLICY:ckpt=${self}", *start]
    assert [e["suite"] for e in four["eval"]["final"]] == ["heldout-4p", {"name": "heldout-4p", "boards": 500},
                                                           {"name": "heldout-4p", "boards": 500}, "heldout-4p"]
    assert four["eval"]["final"][3]["lineup"][1:] == ["F", "F", "F"]
    # 'public' is only fair with two players: four players play it as BELIEF.
    public = load_plan(write(tmp_path, env={"scenarios": [{"game": "base-4p"}]}, obs={"level": "public"},
                             **az)).variants[0].config
    assert public["eval"]["select"]["lineup"][0] == "BELIEF:ckpt=${self}:worlds=16"
    island = load_plan(write(tmp_path, env={"scenarios": [{"game": "random-4p"}]}, **az)).variants[0].config
    assert island["eval"]["select"]["suite"] == "select-random-4p"
    # A mixture has no default suites: the experiment must choose its own head to head.
    mixed = {"scenarios": [{"game": "base-2p"}, {"game": "base-4p"}]}
    with pytest.raises(ConfigError, match="no default head-to-head suites"):
        load_plan(write(tmp_path, env=mixed, **az))
    own = {"select": {"suite": "select-2p", "lineup": ["POLICY:ckpt=${self}", "POLICY:ckpt=${init}"]}}
    assert load_plan(write(tmp_path, env=mixed, eval=own, **az)).variants[0].config["eval"]["select"]["suite"] == "select-2p"


def test_hash_ignores_meta_runtime_and_train_final(tmp_path):
    first = load_plan(write(tmp_path)).variants[0]
    other = load_plan(write(tmp_path, meta={"notes": "x"}, runtime={"device": "cpu", "threads": 3},
                            eval={"final": [{"suite": "heldout-2p", "lineup": ["F", "R"]}]})).variants[0]
    assert first.config_hash == other.config_hash
    changed = load_plan(write(tmp_path, seed=2)).variants[0]
    assert changed.config_hash != first.config_hash
    assert config_diff(first.config, changed.config) == ["seed: 1 -> 2"]
    a = {"kind": "eval", "eval": {"final": [{"suite": "x", "lineup": ["F"]}]}}
    b = {"kind": "eval", "eval": {"final": [{"suite": "x", "lineup": ["R"]}]}}
    assert config_hash(a) != config_hash(b)


def test_set_overrides_win(tmp_path):
    plan = load_plan(write(tmp_path, sweep={"axes": {"seed": [1, 2]}, "name": "s{seed}"}),
                     overrides=["train.gamma=1.0", "runtime.threads=3"], only="s2")
    assert [v.name for v in plan.variants] == ["s2"]
    assert plan.variants[0].config["train"]["gamma"] == 1.0
    with pytest.raises(ConfigError):
        load_plan(write(tmp_path), variant="nope")


def test_smoke_maps_epoch_refs_to_latest():
    plan = load_plan(FIXTURES / "fx-lineage.yaml", smoke=True, only="graph-tiny-s42.*")
    g1 = plan.variant("graph-tiny-s42.g1")
    assert plan.experiment == "fx-lineage+smoke"
    assert g1.config["init_from"]["ckpt"] == "fx-lineage+smoke/graph-tiny-s42.ppo@latest"
    assert g1.config["eval"]["select"]["suite"] == {"name": "select-2p", "boards": 2}
    assert g1.config["runtime"]["device"] == "cpu"


#: Pinned: a change here means every existing run of these configs refuses to resume,
#: unless MIGRATIONS covers it. Schema 2 added PPO's critic_targets, critic_gamma,
#: muon_exclude and muon_exclude_lr (schema 1's hashes: graph-tiny-s42.ppo 20ec34c2d9d0676a,
#: graph-tiny-s42.g1 15aeb57c1b609e46, mlp-s43.ppo f2a01f1ffa2e8221, mlp-s43.g1
#: 31e1ee123fb5fcd0, s16-oracle 3ee08aa7d06abdcb, s64-d4 aaa5309ffd128cac); schema 3 added
#: PPO's aux_coef (schema 2's hashes: graph-tiny-s42.ppo 9de4ad123b1571e1, graph-tiny-s42.g1
#: 8f20d8bd5eb0864c, mlp-s43.ppo 0afc662b53b55599, mlp-s43.g1 2b8b04b954ec769d, s16-oracle
#: 57fd361629823202, s64-d4 3b565fffa56c89eb). Stored schema-1 and schema-2 runs migrate to
#: the hashes below.
GOLDEN = {
    "fx-lineage": {
        "graph-tiny-s42.ppo": "d1bd50a558dcb0d1", "graph-tiny-s42.g1": "700cfba93f2a4e09",
        "mlp-s43.ppo": "3666bf72fc1f3b1f", "mlp-s43.g1": "94da3f005acf8a95",
    },
    "fx-mcts": {"s16-oracle": "03c620ea32b0d5f2", "s64-d4": "32c708b2b94de072"},
}


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_golden_fixture_hashes(name):
    plan = load_plan(FIXTURES / f"{name}.yaml")
    found = {v.name: v.config_hash for v in plan.variants}
    assert {k: found[k] for k in GOLDEN[name]} == GOLDEN[name]


def test_schema_1_ppo_configs_migrate_to_schema_2_hashes():
    """A run stored before schema 2 (no critic keys) hashed as schema 1 did and migrates
    to the current resolution's hash, so it still resumes."""
    schema_1 = {("fx-lineage", "graph-tiny-s42.ppo"): "20ec34c2d9d0676a",
                ("fx-mcts", "s64-d4"): "aaa5309ffd128cac"}
    for (name, variant), old_hash in schema_1.items():
        new = load_plan(FIXTURES / f"{name}.yaml").variant(variant)
        stored = json.loads(json.dumps(new.config))
        for key in ("critic_targets", "critic_gamma", "muon_exclude", "muon_exclude_lr", "aux_coef"):
            del stored["train"][key]
        if stored["model"]:
            del stored["model"]["args"]["aux"]
        assert config_hash(stored) == old_hash
        assert config_hash(migrate(stored, from_schema=1)) == new.config_hash


def test_schema_2_ppo_configs_migrate_to_schema_3_hashes(tmp_path):
    """A run stored under schema 2 (no aux_coef: the live runs of October 2026) hashed as
    schema 2 did and migrates to the current resolution's hash, so it still resumes."""
    schema_2 = {("fx-lineage", "graph-tiny-s42.ppo"): "9de4ad123b1571e1",
                ("fx-lineage", "mlp-s43.g1"): "2b8b04b954ec769d",
                ("fx-mcts", "s16-oracle"): "57fd361629823202"}
    for (name, variant), old_hash in schema_2.items():
        new = load_plan(FIXTURES / f"{name}.yaml").variant(variant)
        stored = json.loads(json.dumps(new.config))
        del stored["train"]["aux_coef"]
        if stored["model"] and stored["model"]["class"] == "graph":  # resolution fills the class default
            del stored["model"]["args"]["aux"]
        assert config_hash(stored) == old_hash
        assert config_hash(migrate(stored, from_schema=2)) == new.config_hash
    # The class default is a graph key: another class's stored args stay as they are.
    cnn = load_plan(write(tmp_path, model={"preset": "cnn-tiny"})).variants[0].config
    stored = json.loads(json.dumps(cnn))
    del stored["train"]["aux_coef"]
    assert "aux" not in migrate(stored, from_schema=2)["model"]["args"]
    assert config_hash(migrate(stored, from_schema=2)) == config_hash(cnn)


def _keyed(key: str, algo: str | None) -> str:
    return key.replace("train.", f"train.{algo}.", 1) if algo else key


def test_migrations_keep_hashes_and_cover_new_keys(monkeypatch, tmp_path):
    snapshot = json.loads((ROOT / "tests" / "data" / "config_schema.json").read_text())
    known_algos = {k.split(".")[1] for k in snapshot["keys"] if k.startswith("train.")}
    migrated = {_keyed(key, algo) for schema, key, _, algo in config.MIGRATIONS if schema > snapshot["schema"]}
    migrated |= {f"train.{a}.{key[6:]}" for schema, key, _, algo in config.MIGRATIONS
                 if algo is None and key.startswith("train.") for a in known_algos}
    new = {k for k in config.schema_keys() - set(snapshot["keys"])
           if not k.startswith("train.") or k.split(".")[1] in known_algos}  # a new algo has no old configs
    assert new <= migrated, f"new config keys without a MIGRATIONS entry: {sorted(new - migrated)}"
    # An old stored config plus a new key at its legacy value hashes like the new resolution.
    stored = load_plan(write(tmp_path)).variants[0].config
    old = json.loads(json.dumps(stored))
    del old["train"]["ent_coef"]
    monkeypatch.setattr(config, "MIGRATIONS", [(config.SCHEMA + 1, "train.ent_coef", 0.01, "ppo")])
    assert config_hash(migrate(old, from_schema=config.SCHEMA)) == config_hash(stored)
    assert config_hash(migrate(old, from_schema=config.SCHEMA + 1)) != config_hash(stored)
    # A PPO key's migration leaves other algos' configs alone.
    dagger = load_plan(write(tmp_path, train={"algo": "imitation"})).variants[0].config
    assert "ent_coef" not in migrate(dagger, from_schema=config.SCHEMA)["train"]


def test_tied_axes_on_one_section_merge(tmp_path):
    sweep = {"axes": {"gam": {"lo": {"train": {"gamma": 0.9}}, "hi": {"train": {"gamma": 0.999}}},
                      "ent": {"e0": {"train": {"ent_coef": 0.0}}, "e1": {"train": {"ent_coef": 0.1}}}},
             "name": "{gam}-{ent}"}
    plan = load_plan(write(tmp_path, sweep=sweep))
    got = {v.name: (v.config["train"]["gamma"], v.config["train"]["ent_coef"]) for v in plan.variants}
    assert got == {"lo-e0": (0.9, 0.0), "lo-e1": (0.9, 0.1), "hi-e0": (0.999, 0.0), "hi-e1": (0.999, 0.1)}
    assert len({v.config_hash for v in plan.variants}) == 4


def test_a_sweep_cell_can_replace_a_section(tmp_path):
    # `key!` in a cell replaces the base's section instead of merging into it (arch-ablation's lstm cell).
    sweep = {"axes": {"m": {"seeded": {"seed": 2}, "plain": {"model!": {"preset": "graph-tiny"}}}}, "name": "{m}"}
    plan = load_plan(write(tmp_path, model={"preset": "graph-tiny", "args": {"shared": True}}, sweep=sweep))
    got = {v.name: v.config["model"]["args"]["shared"] for v in plan.variants}
    assert got == {"seeded": True, "plain": False}


def test_a_head_to_head_row_depends_on_the_cell_it_names(tmp_path):
    # eval.final is not hashed, but its refs are bound when the variant is created (arch-ablation's
    # head-to-head rows), so a ref to a sibling cell orders the runner; ${experiment} keeps it in the
    # smoke store.
    row = {"suite": "heldout-2p", "lineup": ["POLICY:ckpt=${run}@best", "POLICY:ckpt=${experiment}/control@best"]}
    sweep = {"axes": {"m": {"control": {"seed": 2}, "other": {"eval.final": [row]}}}, "name": "{m}"}
    for smoke, experiment in ((False, "exp"), (True, "exp+smoke")):
        plan = load_plan(write(tmp_path, sweep=sweep), smoke=smoke)
        control, other = plan.variants
        assert (control.depends_on, other.depends_on) == ((), ("control",))
        assert other.config["eval"]["final"][0]["lineup"][1] == f"POLICY:ckpt={experiment}/control@best"
    with pytest.raises(ConfigError, match="depends on \\['missing'\\]"):
        row["lineup"][1] = "POLICY:ckpt=${experiment}/missing@best"
        load_plan(write(tmp_path, sweep=sweep))


def test_set_on_a_sweep_axis_is_refused(tmp_path):
    path = write(tmp_path, sweep={"axes": {"seed": [1, 2]}, "name": "s{seed}"})
    with pytest.raises(ConfigError, match="--set and a sweep axis"):
        load_plan(path, overrides=["seed=9"])


def test_excludes_are_checked(tmp_path):
    axes = {"train.learning_rate": [1.0e-3, 3.0e-4], "seed": [1, 2]}
    plan = load_plan(write(tmp_path, sweep={"axes": axes, "exclude": [{"train.learning_rate": "1e-3", "seed": 2}]}))
    assert len(plan.variants) == 3
    with pytest.raises(ConfigError, match="names no axis"):
        load_plan(write(tmp_path, sweep={"axes": axes, "exclude": [{"seeed": 1}]}))
    with pytest.raises(ConfigError, match="matches no cell"):
        load_plan(write(tmp_path, sweep={"axes": axes, "exclude": [{"seed": 3}]}))


def test_game_definitions_are_hashed(tmp_path):
    game = tmp_path / "g.yaml"
    game.write_text("name: g\nnum_players: 2\nmap: {type: BASE}\nvictory_points_to_win: 15\n")
    path = write(tmp_path, env={"scenarios": [{"game": str(game)}]})
    first = load_plan(path).variants[0]
    assert first.config["env"]["scenarios"][0]["game"]["victory_points_to_win"] == 15
    game.write_text("name: g\nnum_players: 2\nmap: {type: BASE}\nvictory_points_to_win: 10\n")
    assert load_plan(path).variants[0].config_hash != first.config_hash


def test_execution_keys_cover_the_runners():
    from catanrl.train.puffer import RUNNER_KEYS

    assert RUNNER_KEYS <= config.EXECUTION_KEYS


def test_heldout_overlap_compares_generated_maps(tmp_path):
    mini = {"name": "mini2", "num_players": 2, "map": {"type": "MINI"}}
    spelled = {**mini, "map": {"type": "MINI", "number_placement": "official_spiral"}}  # the same boards
    suite = {"purpose": "heldout", "seed": 5, "boards": 300, "turn_limit": 500}
    final = [{"suite": {**suite, "scenarios": [{"game": g}]}, "lineup": ["F", "R"]} for g in (mini, spelled)]
    path = write(tmp_path, env={"scenarios": [{"game": mini, "maps": 1024}]}, eval={"final": final})
    problems = config.validate_plan(path)
    assert len(problems) == 2 and all("training pool" in p for p in problems)
