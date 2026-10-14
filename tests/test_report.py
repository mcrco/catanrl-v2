"""Test 13: report gates, supersedes over the training lineage, dagger rows,
--final, the lockfile and wildcards."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest
import torch
import yaml
from test_results import FakeHarness

from catanrl import provenance
from catanrl.config import Variant, config_hash, resolve
from catanrl.envs.encoding import action_size, dims_for, row_layout
from catanrl.eval import results
from catanrl.models import build_model
from catanrl.report import ReportError, report_data
from catanrl.runs import RunWriter

HELDOUT = {"name": "heldout-2p", "boards": 2}


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Runs in a temp dir; provenance points at a temp repo with commits c1 -> c2."""
    monkeypatch.setenv("CATANRL_RUNS_DIR", str(tmp_path / "runs"))
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    heads = []
    for i in range(2):
        (repo / "f").write_text(str(i))
        git(repo, "add", "f")
        git(repo, "commit", "-qm", f"c{i}")
        heads.append(git(repo, "rev-parse", "HEAD"))
    state = SimpleNamespace(head=heads[0], dirty=False)

    def capture(*, argv=None, kind="eval", harness="native"):
        return {"git": {"root": str(repo), "submodules": {}, "repos": {"catanrl-v2": {
                    "path": str(repo), "head": state.head, "dirty": state.dirty, "untracked": [], "branch": "main",
                    "diff_sha256": None}}},
                "code": {"pinned": "x"}, "python": {"uv_lock_sha256": "l"}, "engine": {}, "run": {}}

    monkeypatch.setattr(provenance, "capture", capture)
    monkeypatch.setattr(provenance, "repo_root", lambda: repo)
    monkeypatch.setattr(provenance, "code_id", lambda kind="eval", harness="native": {"pinned": "x"})
    monkeypatch.setattr(results, "play_schedule", FakeHarness())
    return SimpleNamespace(tmp=tmp_path, heads=heads, state=state)


def tiny(level="private", seed=0):
    torch.manual_seed(seed)
    dims = dims_for(["base-2p"])
    return build_model({"preset": "graph-tiny"}, dims=dims, levels=level, num_actions=action_size(dims),
                       layout=row_layout(dims, level))


def make_run(name, *, level="private", init=None, status="finished"):
    raw = {"seed": 1, "env": {"scenarios": [{"game": "base-2p"}]}, "model": {"preset": "graph-tiny"},
           "obs": {"level": level}}
    if init:
        raw["init_from"] = {"ckpt": init}
    cfg = resolve(raw, experiment="exp", variant=name)
    with RunWriter.open(Variant("exp", name, None, {}, cfg, config_hash(cfg)), argv=["t"]) as w:
        refs = []
        for epoch in (0, 2):
            refs.append(w.save_checkpoint(tiny(level, seed=epoch), epoch=epoch, agent_steps=epoch))
            w.save_trainer_state({"epoch": epoch})
        w.set_alias("best", refs[0], record=SimpleNamespace(eval_id="sel", summary={"score_rate": 0.5}))
        w.finish(status)
    return refs


def write_spec(env, cite, supersedes=()):
    path = env.tmp / "report" / "results.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump({"supersedes": list(supersedes), "cite": cite}))
    return path


def cite(spec="POLICY:exp/a@best", suite=HELDOUT, **extra):
    return {"lineup": [spec, "F"], "suite": suite, **extra}


def test_gates(env):
    make_run("a")
    make_run("full", level="full")
    with pytest.raises(ReportError, match="catanrl eval POLICY:ckpt=exp/a@e0#"):
        report_data(write_spec(env, {"x": cite()}))
    out = report_data(write_spec(env, {"x": cite()}), run_missing=True)
    entry = yaml.safe_load(out.read_text())["entries"]["x"]
    assert entry["n"] == 4 and entry["flags"] == [] and entry["refs"][0].startswith("exp/a@e0#")
    with pytest.raises(ReportError, match="select suite"):
        report_data(write_spec(env, {"x": cite(suite={"name": "select-2p", "boards": 2})}), run_missing=True)
    with pytest.raises(ReportError, match="hidden information"):
        report_data(write_spec(env, {"y": cite("POLICY:exp/full@best")}), run_missing=True)
    report_data(write_spec(env, {"y": cite("POLICY:exp/full@best", oracle_ok=True)}), run_missing=True)
    results.play_schedule.forfeit = "s0b0r0"
    with pytest.raises(ReportError, match="invalid"):
        report_data(write_spec(env, {"z": cite("POLICY:exp/a@e2")}), run_missing=True)


def test_supersedes_walks_the_lineage(env):
    c1, c2 = env.heads
    make_run("parent")  # trained at c1
    env.state.head = c2
    make_run("child", init="exp/parent@best")  # trained at c2, warm-started from c1
    spec = write_spec(env, {"x": cite("POLICY:exp/child@best")}, supersedes=[c1])
    report_data(spec, run_missing=True)
    with pytest.raises(ReportError, match="before"):
        report_data(write_spec(env, {"x": cite("POLICY:exp/child@best")}, supersedes=[c2]), run_missing=True)


def test_dagger_rows_and_final(env):
    make_run("a")
    env.state.dirty = True
    out = report_data(write_spec(env, {"x": cite()}), run_missing=True)
    assert yaml.safe_load(out.read_text())["entries"]["x"]["flags"] == ["dirty"]
    with pytest.raises(ReportError, match="--final refuses dirty"):
        report_data(write_spec(env, {"x": cite()}), final=True)


def test_lockfile_and_wildcards(env):
    refs = make_run("a")
    make_run("b.g1")
    make_run("c.g1", status="time_limit")
    spec = write_spec(env, {"x": cite(), "w": cite("POLICY:exp/*.g1@best", group_by="seed")})
    out = report_data(spec, run_missing=True)
    data = yaml.safe_load(out.read_text())
    assert [r["refs"][0].split("@")[0] for r in data["entries"]["w"]["rows"]] == ["exp/b.g1"]
    assert data["entries"]["w"]["groups"] == {"1": {"mean": pytest.approx(data["entries"]["w"]["rows"][0]["rate"]),
                                                    "sd": None, "n": 1}}
    from catanrl.runs import _append_alias

    _append_alias(env.tmp / "runs" / "exp" / "a", "best", refs[1], by="trainer", eval_id="sel2")
    with pytest.raises(ReportError, match="resolve differently"):
        report_data(spec, run_missing=True)
    report_data(spec, run_missing=True, update=True)


def test_paired_entries(env):
    make_run("a")
    spec = write_spec(env, {"a0": cite("POLICY:exp/a@e0"), "a2": cite("POLICY:exp/a@e2"),
                            "d": {"paired": ["a0", "a2"]}})
    data = yaml.safe_load(report_data(spec, run_missing=True).read_text())
    assert data["entries"]["d"]["games"] == 4
