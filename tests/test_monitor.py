"""``eval.monitor`` and ``eval.champion``: evals at checkpoints, logged as ``select/<name>/...``
rows, deciding nothing unless ``eval.select.from`` names one; absent (or unset), every config
hashes as it did before they existed."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from run_helpers import train
from train_configs import tiny_ppo

from catanrl.config import ConfigError, config_hash, load_plan, resolve, validate_plan
from catanrl.runs import is_monitor_row, load_experiment, monitor_evals
from catanrl.train import select as select_module
from catanrl.train.puffer import train_ppo

ROOT = Path(__file__).resolve().parents[1]
SELF_VS_F = ["POLICY:ckpt=${self}", "F"]


def with_eval(**eval_) -> dict:
    return {**tiny_ppo(epochs=4, checkpoint_interval=1), "eval": eval_}


def resolved(cfg: dict) -> dict:
    return resolve(cfg, experiment="test", variant="v")


# ---------------------------------------------------------------- config


def test_absent_or_unset_monitors_and_champion_keep_the_hash():
    select = {"suite": "select-2p", "lineup": SELF_VS_F, "every": 1}
    plain = resolved(with_eval(select=select))
    assert "monitor" not in plain["eval"] and "champion" not in plain["eval"]
    for unset in ({"monitor": []}, {"monitor": None}, {"champion": None}, {"monitor": [], "champion": None}):
        cfg = resolved(with_eval(select=select, **unset))
        assert config_hash(cfg) == config_hash(plain) and set(cfg["eval"]) == set(plain["eval"])
    monitored = resolved(with_eval(select=select, monitor=[{"name": "vsf", "suite": "select-2p", "lineup": SELF_VS_F}]))
    assert config_hash(monitored) != config_hash(plain)


def test_every_experiment_hashes_as_pinned():
    """configs/experiments/*.yaml resolve to the hashes pinned before eval.monitor existed, so
    stored runs still match. A deliberate config change re-pins: regenerate the file from
    ``{f"{v.experiment}/{v.name}": v.config_hash for each plan's variants}``."""
    pinned = json.loads((ROOT / "tests" / "data" / "experiment_hashes.json").read_text())
    got = {}
    for path in sorted((ROOT / "configs" / "experiments").glob("*.yaml")):
        got.update({f"{v.experiment}/{v.name}": v.config_hash for v in load_plan(path).variants})
    shared = sorted(set(pinned) & set(got))
    assert len(shared) >= 0.9 * len(pinned)
    assert {k: got[k] for k in shared} == {k: pinned[k] for k in shared}


@pytest.mark.parametrize("entry, match", [
    ({"suite": "select-2p", "lineup": SELF_VS_F}, "needs name, suite and lineup"),
    ({"name": "Bad Name", "suite": "select-2p", "lineup": SELF_VS_F}, "must match"),
    ({"name": "m", "suite": "select-2p", "lineup": ["F", "POLICY:ckpt=${self}"]}, "slot 0"),
    ({"name": "m", "suite": "select-2p", "lineup": SELF_VS_F, "every": 0}, "positive"),
    ({"name": "m", "suite": "select-2p", "lineup": SELF_VS_F, "evry": 5}, "unknown key eval.monitor.evry"),
])
def test_monitor_entries_are_checked(entry, match):
    with pytest.raises(ConfigError, match=match):
        resolved(with_eval(monitor=[entry]))


def test_monitor_shape_and_kind_are_checked():
    entry = {"name": "m", "suite": "select-2p", "lineup": SELF_VS_F}
    with pytest.raises(ConfigError, match="named 'm'"):
        resolved(with_eval(monitor=[entry, entry]))
    with pytest.raises(ConfigError, match="expected a list"):
        resolved(with_eval(monitor=entry))
    with pytest.raises(ConfigError, match="kind: eval"):
        resolved({**with_eval(monitor=[entry]), "kind": "eval"})
    with pytest.raises(ConfigError, match="unknown key 'eval.monitr'.*monitor"):
        resolved(with_eval(monitr=[entry]))


def test_champion_is_eval_champion_else_init():
    lineup = ["POLICY:ckpt=${self}", "POLICY:ckpt=${champion}"]
    entry = {"name": "champ", "suite": "select-2p", "lineup": lineup}
    with pytest.raises(ConfigError, match="has neither"):
        resolved(with_eval(monitor=[entry]))
    init = {"ckpt": "base/run@e100", "parts": ["actor", "critic"]}
    from_init = resolved({**with_eval(monitor=[entry]), "init_from": init})
    assert from_init["eval"]["monitor"][0]["lineup"] == ["POLICY:ckpt=${self}", "POLICY:ckpt=${init}"]
    named = resolved({**with_eval(monitor=[entry], champion="puffer-graph2/full@e3350"), "init_from": init})
    assert named["eval"]["monitor"][0]["lineup"] == lineup and named["eval"]["champion"] == "puffer-graph2/full@e3350"
    with pytest.raises(ConfigError, match=r"\$\{champion\} is only allowed"):
        resolved({**with_eval(monitor=[entry]), "init_from": {**init, "ckpt": "${champion}"}})


def test_select_from_a_monitor():
    champ = {"name": "champ", "suite": {"name": "select-2p", "boards": 50}, "lineup": SELF_VS_F, "every": 5}
    cfg = resolved(with_eval(monitor=[champ], select={"from": "champ"}))
    assert cfg["eval"]["select"] == {"suite": {"name": "select-2p", "boards": 50}, "lineup": SELF_VS_F, "every": 5,
                                     "from": "champ"}
    with pytest.raises(ConfigError, match="eval.select.suite belongs there"):
        resolved(with_eval(monitor=[champ], select={"from": "champ", "suite": "select-2p"}))
    with pytest.raises(ConfigError, match="no eval.monitor is named 'other'"):
        resolved(with_eval(monitor=[champ], select={"from": "other"}))


def write_experiment(tmp_path: Path, **eval_) -> Path:
    path = tmp_path / "mon.yaml"
    body = {"name": "mon", **{k: v for k, v in tiny_ppo().items() if k != "runtime"}, "eval": eval_}
    path.write_text(yaml.safe_dump(body))
    return path


def test_smoke_runs_shrink_monitors_and_validation_wants_a_select_suite(tmp_path):
    entry = {"name": "champ", "suite": "select-2p", "lineup": SELF_VS_F, "every": 50}
    path = write_experiment(tmp_path, monitor=[entry], select={"from": "champ"})
    assert validate_plan(path) == []
    smoke = load_plan(path, smoke=True).variants[0].config["eval"]
    assert smoke["monitor"][0]["every"] == smoke["select"]["every"] == 1
    assert smoke["monitor"][0]["suite"] == smoke["select"]["suite"] == {"name": "select-2p", "boards": 2}
    held = write_experiment(tmp_path, monitor=[{**entry, "suite": "heldout-2p"}])
    assert validate_plan(held) == ["main: eval.monitor uses heldout-2p, a heldout suite"]


# ---------------------------------------------------------------- training (evals faked)


def fake_record(name: str, epoch: int, rate: float):
    return SimpleNamespace(eval_id=f"{name}-e{epoch}", summary={
        "games": 4, "score_rate": rate, "wilson": [rate - 0.1, rate + 0.1], "truncated": 0, "forfeits": 0,
        "slots": [{"score_rate": rate, "per_seat": {"0": rate, "1": rate / 2}}]})


#: Scores by eval (a monitor's name, the select's ``from``, else "select") and epoch: vs F climbs, vs
#: the champion peaks at e2.
RATES = {"vsf": [0.1, 0.3, 0.5, 0.7, 0.9], "champ": [0.5, 0.4, 0.8, 0.6, 0.7], "select": [0.1, 0.3, 0.5, 0.7, 0.9]}


@pytest.fixture
def fake_evals(monkeypatch):
    played = []

    def fake(entry, ref, eval_runtime):
        name = entry.get("name") or entry.get("from") or "select"
        played.append((name, ref.epoch))
        record = fake_record(name, ref.epoch, RATES[name][ref.epoch])
        return record, record.summary["score_rate"]

    monkeypatch.setattr(select_module, "select_eval", fake)
    return played


MONITORS = [{"name": "vsf", "suite": "select-2p", "lineup": SELF_VS_F, "every": 1},
            {"name": "champ", "suite": "select-2p", "lineup": ["POLICY:ckpt=${self}", "R"], "every": 2}]


def monitor_rows(run) -> dict[tuple[int, str], dict]:
    out = {}
    for row in run.metrics():
        for name in monitor_evals(row):
            assert (row["epoch"], name) not in out, f"two rows for {name} at e{row['epoch']}"
            out[(row["epoch"], name)] = row
    return out


def test_monitors_log_rows_at_their_checkpoints_and_best_follows_select_from(fake_evals):
    w = train(train_ppo, with_eval(monitor=MONITORS, select={"from": "champ"}))
    run = load_experiment("test/v")
    steps = {c.epoch: c.agent_steps for c in run.checkpoints()}
    rows = monitor_rows(run)
    assert sorted(rows) == sorted([(e, "vsf") for e in range(5)] + [(e, "champ") for e in (0, 2, 4)])
    for (epoch, name), row in rows.items():
        assert is_monitor_row(row) and row["step"] == row["agent_steps"] == steps[epoch]
        p = f"select/{name}"
        rate = RATES[name][epoch]
        assert row[f"{p}/score_rate"] == rate and row[f"{p}/games"] == 4 and row[f"{p}/eval_id"] == f"{name}-e{epoch}"
        assert row[f"{p}/wilson_low"] == pytest.approx(rate - 0.1) and row[f"{p}/seat1"] == rate / 2
    epoch_rows = [r for r in run.metrics() if not is_monitor_row(r)]
    assert len(epoch_rows) == 4 and not any(k.startswith("select/") for r in epoch_rows for k in r)
    best = w.aliases()["best"]
    assert (best["epoch"], best["eval_id"], best["metric"]) == (2, "champ-e2", 0.8)  # not vs F's e4
    # A checkpoint's monitors, then its select (here the champ monitor's eval again: a cached record
    # in a real run); the launch's catch-up replays e0's select (the last commit's) and no monitor.
    assert fake_evals[:4] == [("vsf", 0), ("champ", 0), ("champ", 0), ("champ", 0)]
    assert fake_evals.count(("champ", 4)) == 2 and fake_evals.count(("vsf", 4)) == 1


def test_a_resume_logs_only_the_monitor_rows_a_crash_cut_off(fake_evals, monkeypatch):
    from catanrl.runs import RunWriter

    real = RunWriter.log

    def crash_at_e2(self, logs, step):
        if "select/vsf/eval_id" in logs and logs["epoch"] == 2:
            raise KeyboardInterrupt  # e2's vs-F record exists, its row was never written
        real(self, logs, step)

    monkeypatch.setattr(RunWriter, "log", crash_at_e2)
    cfg = with_eval(monitor=MONITORS, select={"suite": "select-2p", "lineup": SELF_VS_F, "every": 1})
    with pytest.raises(KeyboardInterrupt):
        train(train_ppo, cfg)
    monkeypatch.setattr(RunWriter, "log", real)
    assert (2, "vsf") not in monitor_rows(load_experiment("test/v"))
    before = len(fake_evals)
    train(train_ppo, cfg)
    rows = monitor_rows(load_experiment("test/v"))  # asserts one row per (epoch, monitor)
    assert sorted(rows) == sorted([(e, "vsf") for e in range(5)] + [(e, "champ") for e in (0, 2, 4)])
    replayed = fake_evals[before:]
    # The catch-up: e2's two monitors (the crash cut vs F's row and champ never ran), none already logged.
    assert replayed[:2] == [("vsf", 2), ("champ", 2)]
    assert not {("vsf", 0), ("vsf", 1), ("champ", 0)} & set(replayed)


# ---------------------------------------------------------------- training (real evals, CPU)


def test_a_champion_monitor_end_to_end(monkeypatch):
    """Real evals: a run whose champion is another run's checkpoint logs its monitor at every
    checkpoint, moves best by it, and two checkpoints' monitor records pair game for game."""
    import importlib.util

    tiny = {**tiny_ppo(epochs=2, checkpoint_interval=1), "runtime": {"device": "cpu", "threads": 2, "num_envs": 4,
                                                                         "infer_rows": 16, "determinism": "strict"}}
    train(train_ppo, tiny, "champ")
    suite = {"name": "select-2p", "boards": 2}
    champ = {"name": "champ", "suite": suite, "lineup": ["POLICY:ckpt=${self}", "POLICY:ckpt=${champion}"]}
    cfg = {**tiny, "eval": {"champion": "test/champ@latest", "monitor": [champ], "select": {"from": "champ"}}}
    w = train(train_ppo, cfg, "run")
    run = load_experiment("test/run")
    assert run.champion_ref.startswith("test/champ@e2#")  # pinned at creation
    rows = monitor_rows(run)
    assert sorted(rows) == [(0, "champ"), (1, "champ"), (2, "champ")]
    records = {e: next(run.path.glob(f"evals/select-2p/{row['select/champ/eval_id']}.json")) for (e, _), row in
               rows.items()}
    from catanrl.eval.results import EvalRecord

    loaded = {e: EvalRecord.load(p) for e, p in records.items()}
    for epoch, record in loaded.items():
        assert record.data["refs"][1] == run.champion_ref and record.summary["games"] == 4
        assert rows[(epoch, "champ")]["select/champ/score_rate"] == record.summary["score_rate"]
    best = w.aliases()["best"]
    top = max(loaded, key=lambda e: (loaded[e].summary["score_rate"], -e))
    assert best["epoch"] == top and best["eval_id"] == loaded[top].eval_id
    spec = importlib.util.spec_from_file_location("search_probe", ROOT / "tools" / "search_probe.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    report = probe.pair(loaded[0], loaded[2])
    assert report["games"] == 4 and not report["code_differs"] and not report["schedule_differs"]


def test_the_champion_is_pinned_at_creation_and_kept_by_forks(monkeypatch):
    from test_results import FakeHarness
    from test_runs import commit, make_variant, tiny

    from catanrl import runs
    from catanrl.eval import results
    from catanrl.runner import final_evals
    from catanrl.runs import RunWriter

    monkeypatch.setattr(results, "play_schedule", FakeHarness())
    with RunWriter.open(make_variant("p"), argv=["t"]) as w:
        commit(w, tiny(0), 0)
        w.finish("time_limit")
    start = str(load_experiment("exp/p").resolve("latest"))
    vs_champion = ["POLICY:ckpt=${self}", "POLICY:ckpt=${champion}"]
    monitors = [{"name": "champ", "suite": {"name": "select-2p", "boards": 1}, "lineup": vs_champion}]
    ev = {"champion": "exp/p@latest", "monitor": monitors, "select": {"from": "champ"},
          "final": [{"suite": {"name": "heldout-2p", "boards": 1}, "lineup": ["POLICY:ckpt=${run}@best",
                                                                             "POLICY:ckpt=${champion}"]}]}
    child = make_variant("c", eval=ev, runtime={"device": "cpu"})
    assert child.config["eval"]["select"]["lineup"] == vs_champion
    with RunWriter.open(child, argv=["t"]) as w:
        assert w.champion_ref == start and w.init_ref is None
        bound = w.variant_for_training.config["eval"]
        assert bound["monitor"][0]["lineup"] == bound["select"]["lineup"] == ["POLICY:ckpt=${self}",
                                                                              f"POLICY:ckpt={start}"]
        assert w.bindings()["POLICY:ckpt=${champion}"].startswith(f"POLICY:ckpt={start}:")  # keyed as written
        e0 = commit(w, tiny(1), 0)
        assert w.pending_monitors(bound["monitor"]) == [(e0, ["champ"])]
        assert w.pending_selects() == [e0]
        w.set_alias("best", e0, record=SimpleNamespace(eval_id="abc", summary={"score_rate": 0.5}))
        w.finish("time_limit")
    with RunWriter.open(load_experiment("exp/p").variant(), argv=["t"]) as w:  # the champion's run moves on
        w.load_trainer_state()
        commit(w, tiny(2), 1)
        w.finish("time_limit")
    assert str(load_experiment("exp/p").resolve("latest")) != start
    played = final_evals(load_experiment("exp/c"), child)
    assert played[0].data["refs"] == [str(e0), start]
    more = runs.fork("exp/c", as_="c+more")  # a continuation still plays the parent's champion
    assert more.config["eval"]["champion"] == start
    assert more.config["eval"]["monitor"][0]["lineup"] == more.config["eval"]["select"]["lineup"] == [
        "POLICY:ckpt=${self}", f"POLICY:ckpt={start}"]
    assert load_experiment("exp/c+more").champion_ref == start
