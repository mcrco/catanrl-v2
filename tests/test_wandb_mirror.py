"""The wandb mirror (DESIGN 6.10): on only when configured, one stable run id
across relaunches and backfills, failures that only warn, and the select
curve and finals in the layout of the backfilled runs. wandb itself is a
stub module in ``sys.modules``: every ``init``, ``log``, summary and config
write is recorded, lower steps are dropped as wandb drops them, and ``Api``
answers for the runs ``init`` made. The conftest turns the mirror off for
every test; the fixture here turns it back on."""

from __future__ import annotations

import sys
import threading
import types
from types import SimpleNamespace

import pytest
import torch

from catanrl import canon
from catanrl import wandb_mirror as wm
from catanrl.config import Variant, config_hash, load_plan, resolve
from catanrl.envs.encoding import action_size, dims_for, row_layout
from catanrl.models import build_model
from catanrl.runner import run_variant
from catanrl.runs import RunWriter, load_experiment
from catanrl.wandb_mirror import FINAL_COLUMNS, FINAL_TABLE, THROUGH_KEY, Mirror, Settings


@pytest.fixture(autouse=True)
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CATANRL_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WANDB_MODE", raising=False)  # the conftest's "disabled": the switches decide here
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb"))


# ---------------------------------------------------------------- a stub wandb


class _Config(dict):
    def update(self, values, allow_val_change=False):
        dict.update(self, values)


class _Table:
    def __init__(self, columns, data):
        self.columns, self.data = list(columns), [list(r) for r in data]


class _Run:
    """What wandb.init returns: rows merge at an equal step, lower steps are dropped."""

    def __init__(self, kwargs, previous=None):
        self.kwargs = kwargs
        self.summary = dict(previous.summary) if previous else {}
        self.config = _Config(kwargs.get("config") or {})
        self.history = list(previous.history) if previous else []
        self.step = previous.step if previous else -1
        self.resumed = previous is not None
        self.dropped = []
        self.finished = False

    def log(self, data, step=None, commit=None):
        if step < self.step:
            self.dropped.append((step, dict(data)))
            return
        if self.history and self.history[-1][0] == step:
            self.history[-1][1].update(data)
        else:
            self.history.append((step, dict(data)))
        self.step = step

    def finish(self):
        self.finished = True


class _Api:
    """``wandb.Api()``: the stored runs by ``[entity/]project/id``, as the server has them."""

    def __init__(self, mod):
        self._mod = mod
        self.default_entity = "me"

    def run(self, path):
        self._mod.lookups.append(path)
        run = self._mod.store.get(path.rsplit("/", 1)[-1])
        if run is None:
            raise ValueError(f"Could not find run {path}")  # wandb's RunNotFoundError
        return SimpleNamespace(group=run.kwargs.get("group"), tags=list(run.kwargs.get("tags") or []))


def stub_wandb(monkeypatch, *, fail: str | None = None) -> types.ModuleType:
    """A ``wandb`` module whose runs persist by id across ``init`` calls (as resume does);
    ``fail`` makes ``init``, ``log`` or ``Api`` raise."""
    mod = types.ModuleType("wandb")
    mod.store, mod.runs, mod.lookups, mod.Table = {}, [], [], _Table

    def api():
        if fail == "api":
            raise RuntimeError("no network")
        return _Api(mod)

    def init(**kwargs):
        if fail == "init":
            raise RuntimeError("no network")
        run = _Run(kwargs, previous=mod.store.get(kwargs["id"]))
        if fail == "log":
            def boom(*args, **kw):
                raise RuntimeError("boom")

            run.log = boom
        mod.store[kwargs["id"]] = run
        mod.runs.append(run)
        return run

    mod.init, mod.Api = init, api
    monkeypatch.setitem(sys.modules, "wandb", mod)
    return mod


def warned(recwarn, text: str) -> bool:
    """Whether a mirror warning (from any thread) says ``text``."""
    return any(issubclass(w.category, RuntimeWarning) and text in str(w.message) for w in recwarn)


# ---------------------------------------------------------------- a run with rows and records


def make_variant(name="v", experiment="exp", **over) -> Variant:
    raw = {"seed": 1, "env": {"scenarios": [{"game": "base-2p"}], "num_envs": 2}, "model": {"preset": "graph-tiny"},
           "train": {"total_timesteps": 100}, **over}
    cfg = resolve(raw, experiment=experiment, variant=name)
    return Variant(experiment=experiment, name=name, stage=None, labels={}, config=cfg, config_hash=config_hash(cfg))


def tiny():
    torch.manual_seed(0)
    dims = dims_for(["base-2p"])
    return build_model({"preset": "graph-tiny"}, dims=dims, levels="private", num_actions=action_size(dims),
                       layout=row_layout(dims, "private"))


def commit(writer, model, epoch):
    ref = writer.save_checkpoint(model, epoch=epoch, agent_steps=10 * epoch)
    writer.save_trainer_state({"epoch": epoch, "model": model.state_dict()})
    return ref


def write_record(run, ref, *, suite, purpose, rate, opponents=("F",), eval_id=None, **summary_over):
    """A record as results.py writes it, for ``ref`` against ``opponents`` (``summary_over``
    replaces summary keys, to make a record the layout does not fit)."""
    eval_id = eval_id or canon.digest([str(ref), suite, list(opponents)])
    labels = [f"POLICY:{ref.run}@e{ref.epoch}", *opponents]
    summary = {"games": 4, "score_rate": rate, "wilson": [rate - 0.1, rate + 0.1], "truncated": 0, "forfeits": 0,
               "slots": [{"label": labels[0], "games": 4, "score_rate": rate, "wilson": [rate - 0.1, rate + 0.1],
                          "per_seat": {"0": rate, "1": rate / 2}, "expected": 0.5}], **summary_over}
    path = run.path / "evals" / suite / f"{eval_id}.json"
    canon.write_once_json(path, {"eval_id": eval_id, "key": {"suite": {"purpose": purpose}}, "summary": summary,
                                 "suite": suite, "labels": labels, "refs": [str(ref), *([None] * len(opponents))],
                                 "fair": [False] * (1 + len(opponents)), "valid": True})
    return SimpleNamespace(eval_id=eval_id, summary=summary, path=path)


def train_run(project="p"):
    """Three epochs with checkpoints at e0 and e2, selects for both, one final; returns the writer's run."""
    v = make_variant()
    with RunWriter.open(v, argv=["t"]) as w:
        model = tiny()
        e0 = commit(w, model, 0)
        write_record(w, e0, suite="select-2p", purpose="select", rate=0.2)
        w.set_alias("best", e0, record=SimpleNamespace(eval_id="r0", summary={"score_rate": 0.2}))
        w.log({"epoch": 1, "agent_steps": 10, "losses/entropy": 1.0, "map_pool": "abc"}, 10)
        e2 = commit(w, model, 2)
        record = write_record(w, e2, suite="select-2p", purpose="select", rate=0.6)
        w.set_alias("best", e2, record=record)
        w.log({"epoch": 2, "agent_steps": 20, "losses/entropy": 0.5, "map_pool": "abc"}, 20)
        w.log({"epoch": 3, "agent_steps": 30, "losses/entropy": 0.4, "map_pool": "abc"}, 30)
        w.finish("finished")
        write_record(w, e2, suite="heldout-2p", purpose="heldout", rate=0.7)
    return load_experiment("exp/v")


# ---------------------------------------------------------------- enable and disable


def test_off_unless_a_project_is_configured(monkeypatch):
    mod = stub_wandb(monkeypatch)
    with RunWriter.open(make_variant(), argv=["t"]) as w:
        assert w.mirror is None
    assert mod.runs == []
    assert wm.settings_for({"meta": {}}, {}) is None
    assert wm.settings_for({"meta": {"wandb": {"project": "p", "entity": "e", "tags": ["x"]}}}, {}) == \
        Settings(project="p", entity="e", tags=("x",))
    assert wm.settings_for({"meta": {"wandb": "p"}}, {}) == Settings(project="p")
    assert wm.settings_for({"meta": {}}, {"CATANRL_WANDB_PROJECT": "q"}) == Settings(project="q")
    assert wm.settings_for({"meta": {"wandb": {"project": "p"}}}, {"CATANRL_WANDB_PROJECT": "q"}).project == "q"
    assert wm.settings_for({"meta": {"wandb": {"project": "p"}}}, {"WANDB_MODE": "disabled"}) is None
    assert wm.settings_for({"meta": {"wandb": {"project": "p"}}}, {"WANDB_MODE": "offline"}) is not None


def test_smoke_runs_stay_out_unless_asked(monkeypatch):
    """``catanrl run EXP --smoke`` (and the -m train smoke tier) must not put EXP+smoke runs
    in the project EXP's YAML names, nor in an exported CATANRL_WANDB_PROJECT."""
    mod = stub_wandb(monkeypatch)
    cfg = load_plan("arch-lineage", smoke=True).variants[0].config  # the YAML names the real project
    assert cfg["meta"]["wandb"]["project"] == "catanrl-v2" and cfg["meta"]["smoke"] is True
    assert wm.settings_for(cfg, {}) is None
    assert wm.settings_for(cfg, {"CATANRL_WANDB_PROJECT": "q"}) is None
    assert wm.settings_for(cfg, {"CATANRL_WANDB_SMOKE": "1"}) == Settings(project="catanrl-v2")
    assert wm.settings_for(cfg, {"CATANRL_WANDB_SMOKE": "1", "CATANRL_WANDB_PROJECT": "q"}) == Settings(project="q")
    monkeypatch.setenv("CATANRL_WANDB_PROJECT", "p")
    with RunWriter.open(make_variant(meta={"smoke": True, "wandb": {"project": "p"}}), argv=["t"]) as w:
        assert w.mirror is None
        w.finish("finished")
    assert mod.runs == []
    monkeypatch.setenv("CATANRL_WANDB_SMOKE", "1")
    with RunWriter.open(make_variant("v2", meta={"smoke": True}), argv=["t"]) as w:
        assert isinstance(w.mirror, Mirror)
        w.finish("finished")
    assert [r.kwargs["name"] for r in mod.runs] == ["exp/v2"]


def test_a_missing_wandb_package_only_warns(monkeypatch, recwarn):
    monkeypatch.setitem(sys.modules, "wandb", None)  # import wandb raises ImportError
    monkeypatch.setenv("CATANRL_WANDB_PROJECT", "p")
    with RunWriter.open(make_variant(), argv=["t"]) as w:
        assert w.mirror is None
        w.finish("finished")
    assert load_experiment("exp/v").status == "finished"
    assert warned(recwarn, "not importable")


# ---------------------------------------------------------------- identity and resume


def test_run_id_is_stable_and_in_wandbs_charset():
    assert wm.run_id("exp/graph-s-s42.g1", "2fdb6f44ac56b5ff") == "exp-graph-s-s42.g1-2fdb6f44"
    assert wm.run_id("ppo-mixed+smoke/main", "6349dc9308fb69c2") == "ppo-mixed-smoke-main-6349dc93"


def test_a_relaunch_resumes_the_same_wandb_run(monkeypatch):
    mod = stub_wandb(monkeypatch)
    monkeypatch.setenv("CATANRL_WANDB_PROJECT", "p")
    monkeypatch.setenv("CATANRL_WANDB_ENTITY", "me")
    v = make_variant(meta={"wandb": {"group": "g", "tags": ["t"]}})
    with RunWriter.open(v, argv=["t"]) as w:
        assert isinstance(w.mirror, Mirror)
        commit(w, tiny(), 0)
        w.log({"epoch": 1, "agent_steps": 10, "x": 1.0}, 10)
        w.finish("interrupted")
    with RunWriter.open(v, argv=["t"]) as w:
        assert w.mode == "resume"
        w.log({"epoch": 2, "agent_steps": 20, "x": 2.0}, 20)
        w.finish("finished")
    first, second = mod.runs
    assert first.kwargs["id"] == second.kwargs["id"] == f"exp-v-{v.config_hash[:8]}"
    assert first.kwargs["resume"] == "allow" and second.resumed
    kw = first.kwargs
    assert (kw["project"], kw["entity"], kw["name"], kw["group"], kw["job_type"]) == ("p", "me", "exp/v", "g", "train")
    assert kw["tags"] == ["exp", "t"] and kw["notes"] == wm.LIVE_NOTES
    assert kw["config"]["experiment"] == "exp" and kw["config"]["variant"] == "exp/v"
    assert kw["config"]["config_hash"] == v.config_hash and kw["config"]["seed"] == 1
    assert set(kw["config"]["_provenance"]) == {"launch0"} and set(second.kwargs["config"]["_provenance"]) == \
        {"launch0", "launch1"}
    assert kw["config"]["_provenance"]["launch0"]["git"]["catanrl-v2"].keys() == {"head", "dirty", "branch"}
    assert first.config["steps_per_epoch"] == 10 and second.kwargs["config"]["steps_per_epoch"] == 10
    assert [(s, r["x"]) for s, r in second.history] == [(10, 1.0), (20, 2.0)] and second.dropped == []
    assert second.summary[THROUGH_KEY] == 20 and second.summary["status"] == "finished" and second.finished


def test_a_resync_keeps_the_runs_group_and_tags(monkeypatch):
    """The two legacy backfills sit in hand-chosen groups: syncing them again, or a
    relaunch under the mirror, must not move them to the experiment or drop a tag."""
    mod = stub_wandb(monkeypatch)
    run = train_run()
    wm.backfill(run, Settings(project="p", group="puffer-graph-2p"))
    id_ = mod.runs[0].kwargs["id"]
    assert mod.lookups == [f"p/{id_}"] and mod.runs[0].kwargs["group"] == "puffer-graph-2p"
    wm.backfill(run, Settings(project="p", entity="me"), tags=["backfill", "again"])
    assert mod.lookups[1] == f"me/p/{id_}" and mod.runs[1].kwargs["group"] == "puffer-graph-2p"
    assert mod.runs[1].kwargs["tags"] == ["backfill", "exp", "again"]
    monkeypatch.setenv("CATANRL_WANDB_PROJECT", "p")
    with RunWriter.open(run.variant(), argv=["t"]) as w:  # a relaunch of the finished run
        assert isinstance(w.mirror, Mirror)
    assert mod.runs[2].kwargs["group"] == "puffer-graph-2p" and mod.runs[2].kwargs["tags"] == ["backfill", "exp", "again"]
    wm.backfill(run, Settings(project="p", group="moved"))  # a configured group wins
    assert mod.runs[3].kwargs["group"] == "moved"
    monkeypatch.setenv("WANDB_MODE", "offline")  # no server to ask: derived, as for a new run
    wm.backfill(run, Settings(project="p"))
    assert len(mod.lookups) == 4 and mod.runs[4].kwargs["group"] == "exp"


def test_lineage_tags_follow_init_from(monkeypatch):
    stub_wandb(monkeypatch)
    with RunWriter.open(make_variant("parent"), argv=["t"]) as w:
        commit(w, tiny(), 0)
        w.finish("finished")
    child = make_variant("child", init_from={"ckpt": "exp/parent@e0", "parts": ["actor"]})
    with RunWriter.open(child, argv=["t"]) as w:
        assert wm.lineage_tags(w) == ["exp", "from:exp/parent"]
        w.finish("finished")


# ---------------------------------------------------------------- failures only warn


@pytest.mark.parametrize("fail", ["init", "log"])
def test_wandb_failures_only_warn(monkeypatch, recwarn, fail):
    stub_wandb(monkeypatch, fail=fail)
    monkeypatch.setenv("CATANRL_WANDB_PROJECT", "p")
    with RunWriter.open(make_variant(), argv=["t"]) as w:
        commit(w, tiny(), 0)
        w.log({"epoch": 1, "agent_steps": 10, "x": 1.0}, 10)
        w.finish("finished")
    run = load_experiment("exp/v")
    assert run.status == "finished" and [r["x"] for r in run.metrics()] == [1.0]
    assert warned(recwarn, "wandb.init failed" if fail == "init" else "wandb log failed")


def test_an_unreachable_api_only_warns_live_and_stops_a_backfill(monkeypatch, recwarn):
    mod = stub_wandb(monkeypatch, fail="api")
    monkeypatch.setenv("CATANRL_WANDB_PROJECT", "p")
    with RunWriter.open(make_variant(), argv=["t"]) as w:
        w.log({"epoch": 1, "agent_steps": 10, "x": 1.0}, 10)
        w.finish("finished")
    (wb,) = mod.runs
    assert wb.kwargs["group"] == "exp" and dict(wb.history)[10]["x"] == 1.0 and wb.finished
    assert warned(recwarn, "group and tags")
    with pytest.raises(RuntimeError, match="no network"):  # a re-sync blind to the server would move a group
        wm.backfill(load_experiment("exp/v"), Settings(project="p"))


def test_a_select_record_that_does_not_fit_costs_only_itself(monkeypatch, recwarn):
    """A select whose summary the layout cannot read is skipped with a warning; the
    epoch's row, the other selects and the finals still reach wandb."""
    mod = stub_wandb(monkeypatch)
    monkeypatch.setenv("CATANRL_WANDB_PROJECT", "p")
    with RunWriter.open(make_variant(), argv=["t"]) as w:
        model = tiny()
        e0 = commit(w, model, 0)
        write_record(w, e0, suite="select-2p", purpose="select", rate=0.2)
        w.log({"epoch": 1, "agent_steps": 10, "x": 1.0}, 10)
        e2 = commit(w, model, 2)
        write_record(w, e2, suite="select-2p", purpose="select", rate=0.6, slots=[])  # no slot 0
        w.log({"epoch": 2, "agent_steps": 20, "x": 2.0}, 20)
        w.log({"epoch": 3, "agent_steps": 30, "x": 3.0}, 30)
        w.finish("finished")
        write_record(w, e2, suite="heldout-2p", purpose="heldout", rate=0.7)
    (wb,) = mod.runs
    rows = dict(wb.history)
    assert list(rows) == [0, 10, 20, 30] and rows[20] == {"epoch": 2, "agent_steps": 20, "x": 2.0}
    assert rows[0]["select/select-2p vs F/score_rate"] == 0.2 and wb.summary[THROUGH_KEY] == 30
    assert wb.summary["final/heldout-2p/POLICY:exp/v@e2 vs F"] == 0.7 and wb.summary["status"] == "finished"
    assert warned(recwarn, "not mirrored") and not warned(recwarn, "wandb log failed")


def test_a_slow_wandb_finish_does_not_hold_the_run(monkeypatch, recwarn):
    stub_wandb(monkeypatch)
    monkeypatch.setenv("CATANRL_WANDB_PROJECT", "p")
    monkeypatch.setattr(wm, "FINISH_TIMEOUT_S", 0.05)
    gate = threading.Event()
    monkeypatch.setattr(_Run, "finish", lambda self: gate.wait(10))
    with RunWriter.open(make_variant(), argv=["t"]) as w:
        w.finish("finished")
    gate.set()
    assert load_experiment("exp/v").status == "finished"
    assert warned(recwarn, "still syncing")


# ---------------------------------------------------------------- the layout


def test_select_curve_and_finals_in_the_backfilled_layout(monkeypatch):
    mod = stub_wandb(monkeypatch)
    monkeypatch.setenv("CATANRL_WANDB_PROJECT", "p")
    run = train_run()
    (wb,) = mod.runs
    assert wb.finished and wb.dropped == []
    rows = dict(wb.history)
    assert list(rows) == [0, 10, 20, 30]
    sel = "select/select-2p vs F"
    assert rows[0] == {"epoch": 0, "agent_steps": 0, f"{sel}/score_rate": 0.2, f"{sel}/wilson_low": pytest.approx(0.1),
                       f"{sel}/wilson_high": pytest.approx(0.3), f"{sel}/games": 4, f"{sel}/seat0": 0.2,
                       f"{sel}/seat1": 0.1}
    assert rows[10] == {"epoch": 1, "agent_steps": 10, "losses/entropy": 1.0}  # launch, step and strings dropped
    assert rows[20]["losses/entropy"] == 0.5 and rows[20][f"{sel}/score_rate"] == 0.6  # merged into e2's row
    assert "select" not in " ".join(rows[30])
    assert wb.summary[THROUGH_KEY] == 30 and wb.summary["status"] == "finished"
    assert (wb.summary["best/ref"], wb.summary["best/epoch"], wb.summary["best/metric"]) == \
        (run.aliases()["best"]["ref"], 2, 0.6)
    assert wb.summary["final/heldout-2p/POLICY:exp/v@e2 vs F"] == 0.7
    table = wb.summary[FINAL_TABLE]
    assert table.columns == FINAL_COLUMNS
    record = run.evals("heldout-2p")[0]
    assert table.data == [["heldout-2p", "POLICY:exp/v@e2 vs F", 4, 0.7, pytest.approx(0.6), pytest.approx(0.8), 0.5,
                           0, "[false, false]", record.eval_id]]
    assert wb.config["steps_per_epoch"] == 10


def test_a_select_of_a_checkpoint_newer_than_the_row_waits_for_its_row(monkeypatch):
    """A late worker (here: stuck in wandb.init) finds e2's select record on disk while
    e1's row is still to log; wandb needs the rows in step order."""
    mod = stub_wandb(monkeypatch)
    monkeypatch.setenv("CATANRL_WANDB_PROJECT", "p")
    gate, real_init = threading.Event(), mod.init
    mod.init = lambda **kwargs: (gate.wait(10), real_init(**kwargs))[1]
    with RunWriter.open(make_variant(), argv=["t"]) as w:
        model = tiny()
        commit(w, model, 0)
        w.log({"epoch": 1, "agent_steps": 10, "x": 1.0}, 10)
        e2 = commit(w, model, 2)
        write_record(w, e2, suite="select-2p", purpose="select", rate=0.5)
        w.log({"epoch": 2, "agent_steps": 20, "x": 2.0}, 20)
        w.finish("finished")
        gate.set()
    (wb,) = mod.runs
    assert [s for s, _ in wb.history] == [10, 20] and wb.dropped == []
    assert wb.history[1][1]["select/select-2p vs F/score_rate"] == 0.5 and wb.history[1][1]["x"] == 2.0


# ---------------------------------------------------------------- backfill


def test_backfill_reproduces_the_layout_and_is_idempotent(monkeypatch):
    mod = stub_wandb(monkeypatch)
    run = train_run()  # no project set: made without a mirror
    assert mod.runs == []
    result = wm.backfill(run, Settings(project="p"))
    (wb,) = mod.runs
    assert result == {"id": f"exp-v-{run.config_hash[:8]}", "rows": 4, "finals": 1, "through": 30, "resumed": False,
                      "dry_run": False}
    assert wb.kwargs["tags"] == ["backfill", "exp"] and wb.kwargs["notes"] == wm.BACKFILL_NOTES
    assert wb.kwargs["group"] == "exp" and wb.kwargs["name"] == "exp/v"
    assert [s for s, _ in wb.history] == [0, 10, 20, 30]
    assert wb.history[2][1]["select/select-2p vs F/score_rate"] == 0.6 and wb.history[2][1]["losses/entropy"] == 0.5
    assert wb.summary[THROUGH_KEY] == 30 and wb.summary["status"] == "finished"
    assert wb.summary[FINAL_TABLE].data[0][-1] == run.evals("heldout-2p")[0].eval_id
    assert wb.config["status_at_backfill"] == "finished" and wb.config["steps_per_epoch"] == 10
    again = wm.backfill(run, Settings(project="p"))
    assert again["rows"] == 0 and again["resumed"] and mod.runs[1].history == wb.history
    assert wm.backfill(run, Settings(project="p"), dry_run=True) == {"id": wb.kwargs["id"], "rows": 4, "finals": 1,
                                                                       "dry_run": True}


def test_backfill_after_a_live_mirror_adds_only_what_is_new(monkeypatch):
    mod = stub_wandb(monkeypatch)
    monkeypatch.setenv("CATANRL_WANDB_PROJECT", "p")
    run = train_run()
    live = mod.runs[0]
    write_record(run, run.resolve("e2"), suite="heldout-2p", purpose="heldout", rate=0.9, opponents=("F", "F"))
    result = wm.backfill(run, Settings(project="p"))
    assert result["rows"] == 0 and result["finals"] == 2 and result["resumed"]
    assert mod.runs[1].history == live.history and len(mod.runs[1].summary[FINAL_TABLE].data) == 2


# ---------------------------------------------------------------- end to end


def test_tiny_end_to_end_with_the_stub_enabled(monkeypatch):
    mod = stub_wandb(monkeypatch)
    monkeypatch.setenv("CATANRL_WANDB_PROJECT", "p")
    v = load_plan("tiny", variant="2p").variants[0]
    assert run_variant(v, argv=["test"]) == "finished"
    run = load_experiment("tiny/2p")
    (wb,) = mod.runs
    assert wb.kwargs["id"] == f"tiny-2p-{v.config_hash[:8]}" and wb.kwargs["group"] == "tiny"
    assert wb.finished and wb.dropped == []
    rows = dict(wb.history)
    assert [r["epoch"] for r in rows.values()] == [0, 1, 2, 3]
    assert all("select/select-2p vs F/score_rate" in row for row in rows.values())  # a select at every checkpoint
    assert all("losses/entropy" in rows[r["agent_steps"]] for r in run.metrics())
    assert wb.summary[THROUGH_KEY] == run.metrics()[-1]["agent_steps"]
    best = run.aliases()["best"]
    assert wb.summary["best/ref"] == best["ref"] and wb.summary["best/metric"] == best["metric"]
    final = run.evals("heldout-2p")[0]
    assert wb.summary[f"final/heldout-2p/{' vs '.join(final.data['labels'])}"] == final.summary["score_rate"]
    assert wb.summary[FINAL_TABLE].data[0][-1] == final.eval_id and wb.summary["status"] == "finished"


def test_backfill_with_the_real_package_offline(monkeypatch, tmp_path):
    wandb = pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    run = train_run()
    result = wm.backfill(run, Settings(project="catanrl-v2-test"))
    assert result["rows"] == 4 and result["finals"] == 1 and result["through"] == 30
    offline = list((tmp_path / "wandb").glob("wandb/offline-run-*"))
    assert len(offline) == 1 and offline[0].name.endswith(result["id"])
    assert wandb.run is None  # finished
