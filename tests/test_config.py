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
        "dagger": {"train!": {"algo": "imitation", "iterations": 2}},
        "warm": {"init_from": {"ckpt": "${stages.dagger}@best"}},
    }
    plan = load_plan(write(tmp_path, stages=stages))
    dagger, warm = plan.variants
    assert (dagger.name, warm.name) == ("main.dagger", "main.warm")
    assert dagger.config["train"]["algo"] == "imitation" and "total_timesteps" not in dagger.config["train"]
    assert dagger.config["train"]["beta0"] == 1.0
    assert warm.config["init_from"]["ckpt"] == "exp/main.dagger@best"
    assert warm.depends_on == ("main.dagger",) and warm.ref == "exp/main.warm"
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
    assert g1.config["init_from"]["ckpt"] == "fx-lineage+smoke/graph-tiny-s42.warm@latest"
    assert g1.config["eval"]["select"]["suite"] == {"name": "select-2p", "boards": 2}
    assert g1.config["runtime"]["device"] == "cpu"


#: Pinned: a change here means every existing run of these configs refuses to resume.
GOLDEN = {
    "fx-lineage": {
        "graph-tiny-s42.dagger": "3f950ff65aff3360", "graph-tiny-s42.warm": "b98ad2aca9f60a58",
        "graph-tiny-s42.g1": "a5a7490459332231", "mlp-s43.g1": "c5eda8724ed73ed8",
    },
    "fx-mcts": {"s16-oracle": "3ee08aa7d06abdcb", "s64-d4": "aaa5309ffd128cac"},
}


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_golden_fixture_hashes(name):
    plan = load_plan(FIXTURES / f"{name}.yaml")
    found = {v.name: v.config_hash for v in plan.variants}
    assert {k: found[k] for k in GOLDEN[name]} == GOLDEN[name]


def test_migrations_keep_hashes_and_cover_new_keys(monkeypatch, tmp_path):
    snapshot = json.loads((ROOT / "tests" / "data" / "config_schema.json").read_text())
    migrated = {key for schema, key, _ in config.MIGRATIONS if schema > snapshot["schema"]}
    new = config.schema_keys() - set(snapshot["keys"])
    assert new <= migrated, f"new config keys without a MIGRATIONS entry: {sorted(new - migrated)}"
    # An old stored config plus a new key at its legacy value hashes like the new resolution.
    stored = load_plan(write(tmp_path)).variants[0].config
    old = json.loads(json.dumps(stored))
    del old["train"]["ent_coef"]
    monkeypatch.setattr(config, "MIGRATIONS", [(config.SCHEMA + 1, "train.ent_coef", 0.01)])
    assert config_hash(migrate(old, from_schema=config.SCHEMA)) == config_hash(stored)
    assert config_hash(migrate(old, from_schema=config.SCHEMA + 1)) != config_hash(stored)


def test_execution_keys_cover_the_runners():
    from catanrl.train.puffer import RUNNER_KEYS

    assert RUNNER_KEYS <= config.EXECUTION_KEYS
