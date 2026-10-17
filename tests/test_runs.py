"""Test 7: the runs store under a foreign cwd: relaunches, locks, crashes at
each commit point, selectors, pins, fork and finish."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch

from catanrl import canon, config, runs
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
    assert Checkpoint.load(copy).model().catanrl_ref == f"{copy.resolve()}#{e2.weights_sha256[:12]}"


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
    e4 = str(load_experiment("exp/p").resolve(4))
    assert child.config["init_from"]["ckpt"] == e4
    with RunWriter.open(load_experiment("exp/p").variant(), argv=["t"]) as w:  # the parent moves on
        w.load_trainer_state()
        commit(w, tiny(), 6)
        w.finish("time_limit")
    with RunWriter.open(child, argv=["t"]) as w:  # the fork still continues e4
        assert w.mode == "resume"
        assert w.load_trainer_state()["epoch"] == 4
        assert w.variant_for_training.config["init_from"]["ckpt"]["actor"] == e4
    assert [str(r) for r in load_experiment("exp/p+more").lineage()] == [e4]
    with pytest.raises(RunError, match="latest commit"):
        runs.fork("exp/p@e4", as_="p+old")
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


def test_a_fork_evaluates_itself_not_its_parent():
    final = [{"suite": "heldout-2p", "lineup": ["POLICY:ckpt=${run}@best", "F"]}]
    with RunWriter.open(make_variant("p", eval={"final": final}), argv=["t"]) as w:
        commit(w, tiny(), 0)
        w.finish("time_limit")
    child = runs.fork("exp/p", as_="p+more")
    assert child.config["eval"]["final"][0]["lineup"][0] == "POLICY:ckpt=exp/p+more@best"
    assert load_experiment("exp/p+more").manifest["resolved"]["specs"] == {}  # its own: bound at eval time


def test_init_is_pinned_at_creation_and_kept_by_forks(monkeypatch):
    from test_results import FakeHarness

    from catanrl.eval import results
    from catanrl.runner import final_evals

    monkeypatch.setattr(results, "play_schedule", FakeHarness())
    with RunWriter.open(make_variant("p"), argv=["t"]) as w:
        commit(w, tiny(0), 0)
        w.finish("time_limit")
    start = str(load_experiment("exp/p").resolve("latest"))
    suite = {"name": "heldout-2p", "boards": 1}
    ev = {"select": {"suite": "select-2p", "lineup": ["POLICY:ckpt=${self}", "POLICY:ckpt=${init}"]},
          "final": [{"suite": suite, "lineup": ["POLICY:ckpt=${run}@best", "POLICY:ckpt=${init}"]}]}
    child = make_variant("c", init_from={"ckpt": "exp/p@latest"}, eval=ev, runtime={"device": "cpu"})
    with RunWriter.open(child, argv=["t"]) as w:
        assert w.init_ref == start
        assert w.variant_for_training.config["eval"]["select"]["lineup"] == [
            "POLICY:ckpt=${self}", f"POLICY:ckpt={start}"]  # what the trainer's select plays
        assert w.bindings()["POLICY:ckpt=${init}"].startswith(f"POLICY:ckpt={start}:")  # keyed as written
        e0 = commit(w, tiny(1), 0)
        assert w.pending_selects() == [e0]
        w.set_alias("best", e0, record=record())
        w.finish("time_limit")
    with RunWriter.open(load_experiment("exp/p").variant(), argv=["t"]) as w:  # the start's run moves on
        w.load_trainer_state()
        commit(w, tiny(2), 1)
        w.finish("time_limit")
    assert str(load_experiment("exp/p").resolve("latest")) != start
    played = final_evals(load_experiment("exp/c"), child)
    assert played[0].data["refs"] == [str(e0), start]
    more = runs.fork("exp/c", as_="c+more")  # a continuation still plays the parent's start
    assert more.config["eval"]["select"]["lineup"][1] == f"POLICY:ckpt={start}"
    assert more.config["eval"]["final"][0]["lineup"] == ["POLICY:ckpt=exp/c+more@best", f"POLICY:ckpt={start}"]
    with pytest.raises(RunError, match="there is none"):
        runs.expand_init("POLICY:ckpt=${init}", load_experiment("exp/p").init_ref)


def test_a_final_naming_another_experiments_run_waits_for_it(monkeypatch):
    """A control in another experiment (aux-hidden's arch-ablation/graph-s) is not a
    dependency the runner can order. Usable when the variant is created, it is pinned then,
    as every ref is; otherwise the variant is created and trained without it, the final
    waits (Unbound: not missing, nothing to play) until that run is usable, then pins its
    best once; the finals the run can play are played meanwhile."""
    from test_results import FakeHarness

    from catanrl.eval import results
    from catanrl.runner import _missing_finals, final_evals

    harness = FakeHarness()
    monkeypatch.setattr(results, "play_schedule", harness)
    suite = {"name": "heldout-2p", "boards": 1}
    control = "POLICY:ckpt=ctl/main@best"
    ev = {"final": [{"suite": suite, "lineup": ["POLICY:ckpt=${run}@best", "F"]},
                    {"suite": suite, "lineup": ["POLICY:ckpt=${run}@best", control]}]}
    v = make_variant("c", eval=ev, runtime={"device": "cpu"})
    with RunWriter.open(v, argv=["t"]) as w:  # no ctl/main anywhere: created all the same
        assert w.bindings() == {}
        e0 = commit(w, tiny(1), 0)
        w.set_alias("best", e0, record=record())
        w.finish("finished")
    run = load_experiment("exp/c")
    assert run.manifest["resolved"]["specs"] == {} and run.variant().depends_on == ()
    played = final_evals(run, v)
    assert len(played) == 1 and played[0].data["refs"][0] == str(e0)  # vs F is played, the control waits
    assert not _missing_finals(run, v) and run.bindings() == {}
    with pytest.raises(runs.Unbound, match="waits for ctl/main, which is not there yet"):
        runs.bound_lineup(run, ev["final"][1]["lineup"])
    with RunWriter.open(make_variant("main", experiment="ctl"), argv=["t"]) as w:  # there, but not usable
        ctl = commit(w, tiny(2), 0)
        w.set_alias("best", ctl, record=record())
        w.finish("interrupted")
    with pytest.raises(runs.Unbound, match="which is interrupted"):
        runs.bound_lineup(run, ev["final"][1]["lineup"])
    assert not _missing_finals(run, v) and len(final_evals(run, v)) == 1 and run.bindings() == {}
    with RunWriter.open(make_variant("d", eval=ev), argv=["t"]) as w:  # a sibling created now waits too
        assert w.bindings() == {}
    with RunWriter.open(load_experiment("ctl/main").variant(), argv=["t"]) as w:
        w.load_trainer_state()
        w.finish("time_limit")
    assert _missing_finals(run, v)
    played = final_evals(run, v)
    assert len(played) == 2 and played[1].data["refs"] == [str(e0), str(ctl)]
    assert run.bindings()[control].startswith(f"POLICY:ckpt={ctl}")
    with RunWriter.open(make_variant("e", eval=ev), argv=["t"]) as w:  # created once usable: pinned at creation
        assert w.bindings()[control].startswith(f"POLICY:ckpt={ctl}")
    with RunWriter.open(load_experiment("ctl/main").variant(), argv=["t"]) as w:  # the control moves on
        w.load_trainer_state()
        w.set_alias("best", commit(w, tiny(3), 1), record=record("def"))
        w.finish("time_limit")
    calls = harness.calls
    assert not _missing_finals(run, v)
    assert final_evals(run, v)[1].data["refs"] == [str(e0), str(ctl)] and harness.calls == calls  # pinned once


def test_a_kill_between_provenance_and_run_json_keeps_the_run_openable(monkeypatch):
    v = make_variant()
    real = canon.atomic_write_json

    def die_at_run_json(path, obj):
        if path.name == "run.json":
            raise KeyboardInterrupt  # a kill after provenance/launch0.json, before the launch row
        real(path, obj)

    monkeypatch.setattr(canon, "atomic_write_json", die_at_run_json)
    with pytest.raises(KeyboardInterrupt):
        RunWriter.open(v, argv=["t"])
    monkeypatch.setattr(canon, "atomic_write_json", real)
    with RunWriter.open(v, argv=["t"]) as w:
        assert w.launch == 1 and w.mode == "new"
        commit(w, tiny(), 0)
        w.finish("interrupted")
    with RunWriter.open(v, argv=["t"]) as w:  # the resume code check reads the first recorded launch
        assert (w.mode, w.launch) == ("resume", 2)


def test_a_failure_after_finishing_leaves_the_run_finished():
    v = make_variant()
    with RunWriter.open(v, argv=["t"]) as w:
        commit(w, tiny(), 0)
        w.finish("finished")
    with pytest.raises(RuntimeError), RunWriter.open(v, argv=["t"]):
        raise RuntimeError("a final eval failed")
    state = canon.read_json(load_experiment("exp/v").path / "run.json")
    assert state["status"] == "finished" and state["launches"][-1]["status"] == "failed"


def test_finish_reconciles_and_refuses_missing_selects(monkeypatch):
    from test_results import FakeHarness

    from catanrl.eval import results

    v = make_variant()
    with RunWriter.open(v, argv=["t"]) as w:
        commit(w, tiny(), 0)
        w.save_checkpoint(tiny(1), epoch=2, agent_steps=2)
        torch.save({"epoch": 2}, w.path / "trainer_state.pt")  # killed before run.json
        w.save_checkpoint(tiny(2), epoch=4, agent_steps=4)  # never committed
    runs.finish("exp/v")
    run = load_experiment("exp/v")
    assert run.status == "finished" and [c.epoch for c in run.checkpoints()] == [0, 2]
    assert (run.path / "checkpoints" / "orphaned").exists()

    monkeypatch.setattr(results, "play_schedule", FakeHarness())
    select = {"suite": {"name": "select-2p", "boards": 1}, "lineup": ["POLICY:ckpt=${self}", "R"]}
    sv = make_variant("s", eval={"select": select})
    with RunWriter.open(sv, argv=["t"]) as w:
        commit(w, tiny(), 0)
        w.finish("time_limit")
    with pytest.raises(RunError, match="lack select evals"):
        runs.finish("exp/s")


def test_stored_variants_survive_a_migration(monkeypatch):
    with RunWriter.open(make_variant(), argv=["t"]) as w:
        commit(w, tiny(), 0)
        w.finish("interrupted")
    monkeypatch.setattr(config, "MIGRATIONS", [(config.SCHEMA + 1, "train.new_key", 0.0, None)])
    stored = load_experiment("exp/v").variant()
    assert stored.config["train"]["new_key"] == 0.0
    with RunWriter.open(stored, argv=["t"]) as w:
        assert w.mode == "resume"


def test_metrics_hide_rolled_back_epochs_and_cut_lineage():
    v = make_variant()
    with RunWriter.open(v, argv=["t"]) as w:
        for epoch in (1, 2, 3):
            w.log({"epoch": epoch, "x": 0}, epoch)
            if epoch == 2:
                commit(w, tiny(), 2)
        w.finish("interrupted")
    with RunWriter.open(v, argv=["t"]) as w:
        w.load_trainer_state()
        w.log({"epoch": 3, "x": 1}, 3)
        w.finish("time_limit")
    rows = load_experiment("exp/v").metrics()
    assert [(r["launch"], r["epoch"]) for r in rows] == [(0, 1), (0, 2), (1, 3)]
    runs.fork("exp/v", as_="v+more")
    child = load_experiment("exp/v+more")
    assert [r["epoch"] for r in child.metrics(lineage=True)] == [1, 2]  # the parent up to e2, once


def test_lineage_walks_every_parent():
    with RunWriter.open(make_variant("root"), argv=["t"]) as w:
        commit(w, tiny(), 0)
        w.finish("finished")
    with RunWriter.open(make_variant("a", init_from={"ckpt": "exp/root@latest"}), argv=["t"]) as w:
        commit(w, tiny(), 0)
        w.finish("finished")
    with RunWriter.open(make_variant("b"), argv=["t"]) as w:
        commit(w, tiny(1), 0)
        w.finish("finished")
    hybrid = make_variant("h", init_from={"ckpt": {"actor": "exp/b@latest", "critic": "exp/a@latest"}})
    with RunWriter.open(hybrid, argv=["t"]) as w:
        w.finish("finished")
    assert {r.run for r in load_experiment("exp/h").lineage()} == {"exp/a", "exp/b", "exp/root"}


def test_names_resolve_in_the_runs_dir_not_the_cwd(tmp_path, monkeypatch):
    with RunWriter.open(make_variant(), argv=["t"]) as w:
        commit(w, tiny(), 0)
    decoy = tmp_path / "decoy"
    (decoy / "exp" / "v").mkdir(parents=True)
    shutil.copy(load_experiment("exp/v").path / "config.json", decoy / "exp" / "v" / "config.json")
    monkeypatch.chdir(decoy)
    assert load_experiment("exp/v").path == load_experiment(runs.runs_root() / "exp" / "v").path


def test_a_brief_status_probe_does_not_refuse_a_writer():
    import fcntl
    import os
    import threading

    v = make_variant()
    with RunWriter.open(v, argv=["t"]) as w:
        commit(w, tiny(), 0)
        w.finish("interrupted")
    fd = os.open(load_experiment("exp/v").path / ".lock", os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    threading.Timer(0.3, lambda: (fcntl.flock(fd, fcntl.LOCK_UN), os.close(fd))).start()
    with RunWriter.open(v, argv=["t"]) as w:
        assert w.mode == "resume"


def test_the_test_tier_never_reaches_wandb(monkeypatch):
    """The conftest keeps the mirror off for every test, whatever the shell exports
    (a pytest session must never create wandb runs; tests/test_wandb_mirror.py has the stub)."""
    assert os.environ.get("WANDB_MODE") == "disabled" and "CATANRL_WANDB_PROJECT" not in os.environ
    monkeypatch.setenv("CATANRL_WANDB_PROJECT", "catanrl-v2")  # as the operator's shell might
    with RunWriter.open(make_variant(), argv=["t"]) as w:
        assert w.mirror is None
        w.finish("finished")


def test_a_copied_pt_is_its_own_identity(tmp_path):
    with RunWriter.open(make_variant(), argv=["t"]) as w:
        e0 = commit(w, tiny(), 0)
        w.finish("finished")
    copy = tmp_path / "copy.pt"
    shutil.copy(e0.path, copy)
    warm = make_variant("warm", init_from={"ckpt": str(copy)})
    with RunWriter.open(warm, argv=["t"]) as w:
        bound = w.variant_for_training.config["init_from"]["ckpt"]["actor"]
        w.finish("finished")
    assert bound == f"{copy.resolve()}#{e0.weights_sha256[:12]}"
    shutil.rmtree(load_experiment("exp/v").path)  # the header's run is gone; the copy still serves
    assert [r.run for r in load_experiment("exp/warm").lineage()] == [None]
