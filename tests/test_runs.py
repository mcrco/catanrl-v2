"""Test 7: the runs store under a foreign cwd: relaunches, locks, crashes at
each commit point, selectors, pins, fork and finish."""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch

from catanrl import canon, runs
from catanrl.checkpoint import Checkpoint
from catanrl.config import ConfigConflict, Variant, config_hash, resolve
from catanrl.envs.encoding import action_size, dims_for, row_layout
from catanrl.models import build_model
from catanrl.provenance import ProvenanceError
from catanrl.runs import RunError, RunLocked, RunWriter, load_checkpoint, load_experiment


@pytest.fixture(autouse=True)
def runs_dir(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    monkeypatch.setenv("CATANRL_RUNS_DIR", str(root))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    return root


def make_variant(name: str = "v", experiment: str = "exp", **over) -> Variant:
    raw = {"seed": 1, "env": {"scenarios": [{"game": "base-2p"}], "num_envs": 2}, "model": {"preset": "graph-tiny"},
           "train": {"total_timesteps": 100}, **over}
    cfg = resolve(raw, experiment=experiment, variant=name)
    return Variant(experiment=experiment, name=name, stage=None, labels={}, config=cfg, config_hash=config_hash(cfg))


def tiny(seed: int = 0):
    torch.manual_seed(seed)
    dims = dims_for(["base-2p"])
    return build_model({"preset": "graph-tiny"}, dims=dims, levels="private", num_actions=action_size(dims),
                       layout=row_layout(dims, "private"))


def commit(writer: RunWriter, model, epoch: int):
    ref = writer.save_checkpoint(model, epoch=epoch, agent_steps=10 * epoch)
    writer.save_trainer_state({"epoch": epoch, "model": model.state_dict()})
    return ref


def record(eval_id: str = "abc", rate: float = 0.5):
    return SimpleNamespace(eval_id=eval_id, summary={"score_rate": rate})


def test_relaunch_table(monkeypatch):
    v = make_variant()
    with RunWriter.open(v, argv=["t"]) as w:
        assert (w.mode, w.launch) == ("new", 0)
        with pytest.raises(RunLocked):
            RunWriter.open(v, argv=["t"])
        commit(w, tiny(), 0)
        w.finish("interrupted")
    with RunWriter.open(v, argv=["t"]) as w:
        assert (w.mode, w.launch) == ("resume", 1)
        w.finish("finished")
    with RunWriter.open(v, argv=["t"]) as w:
        assert w.mode == "done"
    with pytest.raises(ConfigConflict, match="seed: 1 -> 2"):
        RunWriter.open(make_variant(seed=2), argv=["t"])
    other = make_variant("w")
    with RunWriter.open(other, argv=["t"]) as w:
        w.finish("time_limit")
    monkeypatch.setattr("catanrl.provenance.resume_diff", lambda a, b: ["code train: a -> b"])
    with pytest.raises(ProvenanceError, match="code train"):
        RunWriter.open(other, argv=["t"])
    with RunWriter.open(other, argv=["t"], allow_code_change=True) as w:
        assert w.mode == "resume"
    launches = canon.read_json(load_experiment("exp/w").path / "run.json")["launches"]
    assert [launch["mixed_code"] for launch in launches] == [False, True]


def test_an_exited_child_is_dead(runs_dir):
    make_variant()  # validates the config in-process first
    code = textwrap.dedent(f"""
        import os
        os.environ["CATANRL_RUNS_DIR"] = {str(runs_dir)!r}
        from catanrl.config import Variant, config_hash, resolve
        from catanrl.runs import RunWriter
        raw = {{"seed": 1, "env": {{"scenarios": [{{"game": "base-2p"}}], "num_envs": 2}},
               "model": {{"preset": "graph-tiny"}}, "train": {{"total_timesteps": 100}}}}
        cfg = resolve(raw, experiment="exp", variant="v")
        RunWriter.open(Variant("exp", "v", None, {{}}, cfg, config_hash(cfg)), argv=["child"])
        os._exit(3)
    """)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, check=False)
    assert result.returncode == 3, result.stderr[-2000:]
    run = load_experiment("exp/v")
    assert run.status == "dead"
    with RunWriter.open(run.variant(), argv=["t"]) as w:
        assert (w.mode, w.launch) == ("resume", 1)


def test_crash_between_pt_and_commit_orphans_it():
    v = make_variant()
    model = tiny()
    with RunWriter.open(v, argv=["t"]) as w:
        commit(w, model, 0)
        commit(w, model, 2)
        w.save_checkpoint(model, epoch=4, agent_steps=40)  # crash before the commit
        w.finish("failed")
    assert [c.epoch for c in load_experiment("exp/v").checkpoints()] == [0, 2]
    with RunWriter.open(v, argv=["t"]) as w:
        state = w.load_trainer_state()
        assert state["epoch"] == 2
        assert [c.epoch for c in w.checkpoints()] == [0, 2]
        assert (w.path / "checkpoints" / "orphaned" / "launch1" / "e000004.pt").exists()
        assert w.resolve("latest").epoch == 2


def test_crash_between_commit_and_run_json_reconciles(monkeypatch):
    v = make_variant()
    model = tiny()
    with RunWriter.open(v, argv=["t"]) as w:
        commit(w, model, 0)
        real = canon.atomic_write_json

        def crash(path, obj):
            if path.name == "run.json":
                raise OSError("crash")
            real(path, obj)

        monkeypatch.setattr(canon, "atomic_write_json", crash)
        with pytest.raises(OSError):
            commit(w, model, 2)
        monkeypatch.setattr(canon, "atomic_write_json", real)
        w.finish("failed")
    assert [c.epoch for c in load_experiment("exp/v").checkpoints()] == [0]
    with RunWriter.open(v, argv=["t"]) as w:
        assert w.load_trainer_state()["epoch"] == 2
        assert [c.epoch for c in w.checkpoints()] == [0, 2]


def test_selectors_pins_and_portable_checkpoints(tmp_path):
    v = make_variant()
    with RunWriter.open(v, argv=["t"]) as w:
        e0 = commit(w, tiny(0), 0)
        e2 = commit(w, tiny(1), 2)
        w.set_alias("best", e0, record=record())
        w.set_alias("best", e0, record=record())  # unchanged: no new row
        runs.pin("exp/v@2", "ref", note="under the lock")
        w.finish("finished")
    run = load_experiment("exp")
    assert run.resolve("best").epoch == 0 and run.resolve("latest").epoch == 2
    assert run.resolve("init").epoch == 0 and str(run.resolve(2)) == str(run.resolve("e2")) == str(run.resolve("ref"))
    assert run.resolve(f"e2#{e2.weights_sha256[:12]}").weights_sha256 == e2.weights_sha256
    with pytest.raises(RunError, match="not #"):
        run.resolve(f"e2#{e0.weights_sha256[:12]}")
    with pytest.raises(RunError, match="selectors: best, init, latest, ref, e0, e2"):
        run.resolve("nope")
    assert len(canon.read_jsonl(run.path / "aliases.jsonl")) == 2
    for bad in ("best", "Upper", "e5", "12"):
        with pytest.raises(RunError):
            runs.pin("exp/v@2", bad)
    net = run.build_policy("best")
    assert net.catanrl_ref == str(e0) == f"exp/v@e0#{e0.weights_sha256[:12]}"
    assert load_checkpoint("exp/v@ref").ref.epoch == 2
    copy = tmp_path / "portable.pt"
    shutil.copy(e2.path, copy)
    shutil.rmtree(run.path)
    assert Checkpoint.load(copy).model().catanrl_ref == str(e2)  # identity, not location


def test_best_needs_a_select_record():
    with RunWriter.open(make_variant(), argv=["t"]) as w:
        e0 = commit(w, tiny(), 0)
        w.set_alias("best", e0)
        with pytest.raises(RunError, match="does not cite a select eval"):
            w.resolve("best")


def test_finish_and_fork_continue_a_time_limited_parent():
    parent = make_variant("p")
    with RunWriter.open(parent, argv=["t"]) as w:
        commit(w, tiny(), 0)
        commit(w, tiny(), 4)
        w.finish("time_limit")
    assert load_experiment("exp/p").status == "time_limit"
    child = runs.fork("exp/p", as_="p+more", overrides=["train.total_timesteps=500"])
    assert child.config["init_from"]["ckpt"] == "exp/p@latest"
    with RunWriter.open(child, argv=["t"]) as w:
        assert w.mode == "resume"
        assert w.load_trainer_state()["epoch"] == 4
        assert w.variant_for_training.config["init_from"]["ckpt"]["actor"].startswith("exp/p@e4#")
    assert [str(r) for r in load_experiment("exp/p+more").lineage()] == [str(load_experiment("exp/p").resolve(4))] * 2
    runs.finish("exp/p")
    assert load_experiment("exp/p").status == "finished"
    with pytest.raises(RunError):
        runs.finish("exp/p")


def test_missing_selects_rerun_after_a_crash(monkeypatch):
    from test_results import FakeHarness

    from catanrl.eval import results

    monkeypatch.setattr(results, "play_schedule", FakeHarness())
    select = {"suite": {"name": "select-2p", "boards": 2}, "lineup": ["POLICY:ckpt=${self}", "R"], "every": 2}
    v = make_variant(eval={"select": select})
    with RunWriter.open(v, argv=["t"]) as w:
        e0 = commit(w, tiny(0), 0)
        assert w.pending_selects() == [e0]  # crash before its select eval
        rec = results.evaluate([f"POLICY:ckpt={e0}", "R"], select["suite"])
        w.set_alias("best", e0, record=rec)
        e1 = commit(w, tiny(1), 1)  # off the select cadence
        e2 = commit(w, tiny(2), 2)  # crash before its select eval
        assert [r.epoch for r in w.pending_selects()] == [2]
        results.evaluate([f"POLICY:ckpt={e2}", "R"], select["suite"])  # crash before the alias
        assert [r.epoch for r in w.pending_selects()] == [2]  # the last commit's alias decision is redone
        assert e1.epoch == 1 and w.resolve("best").epoch == 0
