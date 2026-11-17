"""tools/modal_train.py, the parts that need no Modal: staging a revision, the cost arithmetic,
pull's no-overwrite rule, the supervisor's copy of a run directory and the status arithmetic."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("modal")
spec = importlib.util.spec_from_file_location(
    "modal_train", Path(__file__).resolve().parents[1] / "tools" / "modal_train.py")
mt = importlib.util.module_from_spec(spec)
sys.modules["modal_train"] = mt
spec.loader.exec_module(mt)


def git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
                          env=env).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """Two branches with a worktree each, like this repository: main, and wt/other one commit ahead."""
    main = tmp_path / "main"
    main.mkdir()
    git(main, "init", "--quiet", "-b", "main")
    (main / "configs").mkdir()
    (main / "configs" / "a.yaml").write_text("name: a\ntrain: {}\n")
    (main / "pyproject.toml").write_text("[project]\nname='x'\n")
    (main / "uv.lock").write_text("lock 1\n")
    git(main, "add", "-A")
    git(main, "commit", "--quiet", "-m", "one")
    other = tmp_path / "other"
    git(main, "worktree", "add", "--quiet", "-b", "wt/other", str(other))
    (other / "configs" / "b.yaml").write_text("sweep: {players: [2, 4]}\n")
    (other / "uv.lock").write_text("lock 2\n")
    git(other, "add", "-A")
    git(other, "commit", "--quiet", "-m", "two")
    (other / "dirty.txt").write_text("never shipped\n")           # the working tree is not the revision
    (other / "configs" / "b.yaml").write_text("edited, uncommitted\n")
    return main


def test_stage_is_exactly_the_commit_of_another_branch(repo, tmp_path):
    sha = mt.resolve_rev("wt/other", repo)
    assert sha == git(repo, "rev-parse", "wt/other") != git(repo, "rev-parse", "HEAD")
    stage = mt.stage_revision(sha, tmp_path / "cache", repo)
    src = stage / "src"
    assert git(src, "rev-parse", "HEAD") == sha
    assert git(src, "status", "--porcelain") == ""
    assert git(src, "rev-list", "--count", "HEAD") == "1"          # depth 1
    assert (src / "configs" / "b.yaml").read_text() == "sweep: {players: [2, 4]}\n"
    assert not (src / "dirty.txt").exists()
    assert (stage / "deps" / "uv.lock").read_text() == "lock 2\n"
    assert sorted(p.name for p in (stage / "deps").iterdir()) == ["pyproject.toml", "uv.lock"]
    assert git(src, "remote") == ""
    # every directory git needs holds a file: Modal's upload drops empty directories
    assert any(p.is_file() for p in (src / ".git" / "refs").rglob("*"))
    assert any(p.is_file() for p in (src / ".git" / "objects").rglob("*"))
    # the second call is the cached directory, byte for byte (the image's hash rests on it)
    before = {p: p.stat().st_mtime_ns for p in stage.rglob("*") if p.is_file()}
    assert mt.stage_revision(sha, tmp_path / "cache", repo) == stage
    assert before == {p: p.stat().st_mtime_ns for p in stage.rglob("*") if p.is_file()}


def test_stage_head_and_bad_revisions(repo, tmp_path):
    head = mt.resolve_rev("HEAD", repo)
    stage = mt.stage_revision(head, tmp_path / "cache", repo)
    assert not (stage / "src" / "configs" / "b.yaml").exists()
    assert (stage / "deps" / "uv.lock").read_text() == "lock 1\n"
    with pytest.raises(mt.ToolError):
        mt.resolve_rev("no-such-branch", repo)


def test_config_is_read_from_the_revision(repo):
    head, other = mt.resolve_rev("HEAD", repo), mt.resolve_rev("wt/other", repo)
    assert mt.config_at(head, "configs/a.yaml", repo) == ("configs/a.yaml", "a", False)
    assert mt.config_at(other, "configs/b.yaml", repo) == ("configs/b.yaml", "b", True)
    assert mt.config_at(head, str(repo / "configs" / "a.yaml"), repo)[0] == "configs/a.yaml"
    with pytest.raises(mt.ToolError):
        mt.config_at(head, "configs/b.yaml", repo)                  # only on the other branch


def test_cost_arithmetic():
    l40s = mt.usd_per_s("L40S", 8, 32)
    assert l40s == pytest.approx(0.000542 + 8 * 0.0000131 + 32 * 0.00000222)
    assert l40s * 3600 == pytest.approx(2.584, abs=0.001)
    assert mt.usd_per_s("A10G", 8, 32) * 3600 == pytest.approx(1.735, abs=0.001)
    assert mt.usd_per_s(None, 2, 4) * 3600 == pytest.approx(0.126, abs=0.001)
    assert mt.usd_per_s("X9", 1, 1, gpu_usd_per_h=3.6) == pytest.approx(0.001 + 0.0000131 + 0.00000222)
    with pytest.raises(mt.ToolError):
        mt.usd_per_s("X9", 1, 1)
    line = mt.cost_line("L40S", 8, 32, 12, l40s)
    assert "$2.584/h" in line and "$31.01" in line


def test_segment_plan():
    # a 6-minute budget: the call supervises all of it, the trainer's limit leaves a grace
    seg, limit_h = mt.segment_plan(360, 0, 360 + mt.MARGIN_S, None)
    assert seg == 360 and limit_h * 3600 == pytest.approx(324)
    # 30 hours: the first call is capped by Modal's 24 h, the second gets the rest
    seg, limit_h = mt.segment_plan(30 * 3600, 0, mt.DAY_S, None)
    assert seg == mt.DAY_S - mt.MARGIN_S and limit_h * 3600 == pytest.approx(seg - 600)
    seg2, _ = mt.segment_plan(30 * 3600, seg, mt.DAY_S, None)
    assert seg2 == pytest.approx(30 * 3600 - seg)
    assert mt.segment_plan(3600, 3590, 3750, None)[0] == 10
    assert mt.segment_plan(3600, 0, 3750, 60)[0] == 60


def fake_remote(files: dict[str, bytes], mtime: int = 1):
    listing = [(rel, len(data), mtime) for rel, data in sorted(files.items())]
    return listing, lambda rel, f: f.write(files[rel])


def test_pull_never_overwrites_silently(tmp_path):
    dest, manifest = tmp_path / "runs" / "e" / "v", tmp_path / "runs" / "_modal" / "e" / "v.pull.json"
    remote = {"config.json": b"{}", "run.json": b'{"status": "running"}', "metrics.jsonl": b'{"epoch": 1}\n',
              "checkpoints/e000001.pt": b"weights-1", ".lock": b"", "aliases.jsonl": b'{"name": "best"}\n'}
    report = mt.pull_files(*fake_remote(remote), dest, manifest)
    assert sorted(report["new"]) == sorted(k for k in remote if k != ".lock")
    assert not (dest / ".lock").exists()
    assert mt.pull_files(*fake_remote(remote), dest, manifest)["same"] == sorted(k for k in remote if k != ".lock")

    # the Volume moves on; one local file was edited by hand, one checkpoint differs locally
    remote["run.json"] = b'{"status": "finished"}'
    remote["metrics.jsonl"] += b'{"epoch": 2}\n'
    remote["checkpoints/e000002.pt"] = b"weights-2"
    remote["aliases.jsonl"] += b'{"name": "best", "epoch": 2}\n'
    (dest / "aliases.jsonl").write_bytes(b'{"name": "best"}\n{"name": "mine"}\n')     # a local pin
    (dest / "checkpoints" / "e000001.pt").write_bytes(b"other weights")
    report = mt.pull_files(*fake_remote(remote, mtime=2), dest, manifest)
    assert report["new"] == ["checkpoints/e000002.pt"]
    assert report["extended"] == ["metrics.jsonl"]
    assert report["updated"] == ["run.json"]
    assert sorted(report["conflict"]) == ["aliases.jsonl", "checkpoints/e000001.pt"]
    assert (dest / "aliases.jsonl").read_bytes().endswith(b'{"name": "mine"}\n')
    assert (dest / "checkpoints" / "e000001.pt").read_bytes() == b"other weights"
    assert (dest / "run.json").read_bytes() == remote["run.json"]
    assert (dest / "metrics.jsonl").read_bytes() == remote["metrics.jsonl"]
    assert not list(dest.rglob("*.modalpull"))

    # the conflicts stay conflicts until --force, which names them
    again = mt.pull_files(*fake_remote(remote, mtime=2), dest, manifest)
    assert sorted(again["conflict"]) == ["aliases.jsonl", "checkpoints/e000001.pt"] and not again["updated"]
    forced = mt.pull_files(*fake_remote(remote, mtime=2), dest, manifest, force=True)
    assert sorted(forced["forced"]) == ["aliases.jsonl", "checkpoints/e000001.pt"]
    assert (dest / "checkpoints" / "e000001.pt").read_bytes() == b"weights-1"


def test_pull_keeps_a_local_file_it_never_wrote(tmp_path):
    dest, manifest = tmp_path / "e" / "v", tmp_path / "_modal" / "v.pull.json"
    dest.mkdir(parents=True)
    (dest / "run.json").write_bytes(b"the local run")
    (dest / "metrics.jsonl").write_bytes(b'{"epoch": 9}\n')            # not a prefix of the Volume's
    (dest / "config.json").write_bytes(b"{}")
    remote = {"run.json": b"the Volume's run", "metrics.jsonl": b'{"epoch": 1}\n{"epoch": 2}\n', "config.json": b"{}"}
    report = mt.pull_files(*fake_remote(remote), dest, manifest)
    assert sorted(report["conflict"]) == ["metrics.jsonl", "run.json"] and report["same"] == ["config.json"]
    assert (dest / "run.json").read_bytes() == b"the local run"


def test_pull_refuses_a_foreign_run_directory(tmp_path, monkeypatch):
    (tmp_path / "e" / "v").mkdir(parents=True)
    monkeypatch.setattr(mt, "vol_list", lambda path: [("runs/e/v/config.json", 2, 1)])
    with pytest.raises(mt.ToolError, match="not made by this tool"):
        mt.pull("e/v", tmp_path)


def make_run(path: Path, epoch: int, rows: int) -> None:
    (path / "checkpoints").mkdir(parents=True, exist_ok=True)
    for e in range(1, epoch + 1):
        (path / "checkpoints" / f"e{e:06d}.pt").write_bytes(b"w%d" % e)
    (path / "trainer_state.pt").write_bytes(b"state %d" % epoch)
    (path / "config.json").write_text(json.dumps({"config": {"train": {"total_timesteps": 1000}}}))
    (path / "run.json").write_text(json.dumps({
        "status": "running", "trainer_state": {"epoch": epoch},
        "checkpoints": [{"epoch": e, "file": f"e{e:06d}.pt", "weights_sha256": f"{e:064x}"} for e in range(1, epoch + 1)]}))
    with open(path / "metrics.jsonl", "w") as f:
        for e in range(1, rows + 1):
            f.write(json.dumps({"epoch": e, "agent_steps": 100 * e, "throughput/agent_steps_per_s": 50.0}) + "\n")
    (path / ".lock").write_text("")


def test_sync_run_copies_changes_cuts_torn_rows_and_follows_removals(tmp_path):
    src, dst, seen = tmp_path / "local", tmp_path / "vol", {}
    make_run(src, epoch=2, rows=2)
    with open(src / "metrics.jsonl", "a") as f:
        f.write('{"epoch": 3, "agent_st')                             # a row being written
    (src / ".trainer_state.123.tmp").write_bytes(b"half")
    assert mt.sync_run(src, dst, seen) == 6
    assert sorted(p.relative_to(dst).as_posix() for p in dst.rglob("*") if p.is_file()) == [
        "checkpoints/e000001.pt", "checkpoints/e000002.pt", "config.json", "metrics.jsonl", "run.json",
        "trainer_state.pt"]
    assert (dst / "metrics.jsonl").read_text().count("\n") == 2 and (dst / "metrics.jsonl").read_text().endswith("}\n")
    assert mt.sync_run(src, dst, seen) == 0                           # nothing changed
    (src / "checkpoints" / "e000003.pt").write_bytes(b"w3")
    (src / "trainer_state.pt").write_bytes(b"state 3")
    record = json.loads((src / "run.json").read_text())
    record["trainer_state"]["epoch"] = 3
    (src / "run.json").write_text(json.dumps(record))
    with open(src / "metrics.jsonl", "a") as f:
        f.write('eps": 300}\n')
    (src / "checkpoints" / "e000001.pt").unlink()                     # set aside or pruned by the trainer
    assert mt.sync_run(src, dst, seen) == 4                           # e3, metrics, trainer state, run.json
    assert (dst / "metrics.jsonl").read_text().count("\n") == 3
    assert not (dst / "checkpoints" / "e000001.pt").exists()
    assert (dst / "trainer_state.pt").read_bytes() == b"state 3"
    assert json.loads((dst / "run.json").read_text())["trainer_state"]["epoch"] == 3
    assert not list(dst.rglob("*.part"))
    # the way back (a new container): the same files, and a record that copies nothing up again
    back = tmp_path / "container"
    (dst / "checkpoints" / "e000009.pt.part").write_bytes(b"a copy that was cut")
    assert mt.restore_run(dst, back) == 6                             # e2, e3, config, metrics, run.json, state
    assert {p.relative_to(back).as_posix(): p.read_bytes() for p in back.rglob("*") if p.is_file()} == {
        p.relative_to(dst).as_posix(): p.read_bytes() for p in dst.rglob("*") if p.is_file() and p.suffix != ".part"}
    assert mt.sync_run(back, dst, mt._signatures(back)) == 0


def test_pull_cut_short_leaves_a_manifest(tmp_path):
    listing = [("a.txt", 1, 1), ("b.txt", 1, 1)]

    def fetch(rel, f):
        if rel == "b.txt":
            raise KeyboardInterrupt
        f.write(b"a")

    manifest = tmp_path / "_modal" / "e" / "v.pull.json"
    with pytest.raises(KeyboardInterrupt):
        mt.pull_files(listing, fetch, tmp_path / "e" / "v", manifest, workers=1)
    assert "a.txt" in json.loads(manifest.read_text())                # the next pull goes on from here
    report = mt.pull_files(listing, lambda rel, f: f.write(b"a"), tmp_path / "e" / "v", manifest)
    assert (report["same"], report["new"]) == (["a.txt"], ["b.txt"])


def test_progress_and_status(tmp_path):
    run = tmp_path / "e" / "v"
    make_run(run, epoch=2, rows=4)
    p = mt.progress_of(run)
    assert (p["status"], p["commit_epoch"], p["checkpoints"], p["epoch"], p["agent_steps"]) == ("running", 2, 2, 4, 400)
    assert p["target"] == ["agent_steps", 1000] and p["trainer"] == {"agent_steps_per_s": 50.0}
    samples = [[1000.0, 2, 200, 1], [1100.0, 4, 400, 1]]
    rate = mt.wall_rate(samples, 1)
    assert rate == (pytest.approx(0.02), pytest.approx(2.0))
    assert mt.eta_seconds(p, rate) == pytest.approx(300)
    assert mt.wall_rate(samples, 2) is None                           # a new segment has no rate yet
    launch = {"app_id": "ap-1", "rev": "a" * 40, "launch_id": "L", "budget_s": 7200.0,
              "usd_per_s": mt.usd_per_s("L40S", 8, 32), "shape": {"gpu": "L40S", "cores": 8, "GiB": 32}}
    state = {"launch_id": "L", "used_s": 3600.0, "heartbeat": 1100.0, "phase": "training", "segments": 1,
             "progress": p, "samples": samples}
    text = mt.status_text("e/v", launch, state, True, "ephemeral", 1160.0)
    assert "epoch 4, 0.00M agent steps of 0.0M (40.0%)" in text
    assert "ETA 5.0 min" in text and "cost so far $2.58 at $2.584/h" in text and "heartbeat 1.0 min ago" in text
    assert "no heartbeat" in mt.status_text("e/v", launch, {"launch_id": "old"}, True, "ephemeral", 0.0)


def test_ref_files(tmp_path):
    run = tmp_path / "e" / "v"
    make_run(run, epoch=3, rows=3)
    (run / "aliases.jsonl").write_text(json.dumps({"name": "best", "epoch": 1}) + "\n"
                                       + json.dumps({"name": "best", "epoch": 2}) + "\n")
    assert mt.ref_files("e/v@e3", tmp_path) == ("e/v", ["config.json", "run.json", "checkpoints/e000003.pt", "aliases.jsonl"])
    assert mt.ref_files("e/v", tmp_path)[1][2] == "checkpoints/e000002.pt"        # @best: the last row
    assert mt.ref_files("e/v@latest", tmp_path)[1][2] == "checkpoints/e000003.pt"
    assert mt.ref_files(f"e/v@e1#{'0' * 12}", tmp_path)[1][2] == "checkpoints/e000001.pt"
    for bad in ("e/v@e9", "e/v@e1#ffff", "nope/v@e1", "e"):
        with pytest.raises(mt.ToolError):
            mt.ref_files(bad, tmp_path)


def test_guard_trips_on_a_scenario_whose_games_are_all_cut(tmp_path):
    guard = {"share": 0.9, "epochs": 20}
    two, four = "environment/base-2p-fr/truncated", "environment/base-4p/truncated"
    healthy = [{"epoch": i, two: 0.02, four: 0.0, "environment/truncated": 0.01, "losses/entropy": 0.3,
                "losses/approx_kl": 0.03, "losses/value_loss": 0.5} for i in range(30)]
    h = mt.health_of(healthy)
    assert (h["cut"], h["cut_rows"]) == (0.02, 0) and mt.guard_reason(h, guard) is None
    assert h["losses"] == {"entropy": 0.3, "KL": 0.03, "value loss": 0.5}
    # the 2p games collapse first: the overall share stays low, the scenario's own does not
    rows = healthy + [{"epoch": 30 + i, two: 1.0, four: 0.0, "environment/truncated": 0.3} for i in range(19)]
    h = mt.health_of(rows)
    assert (h["cut"], h["cut_rows"], h["cut_of"]) == (1.0, 19, "base-2p-fr") and mt.guard_reason(h, guard) is None
    rows.append({"epoch": 49, four: 0.0})                             # an epoch that ended no 2p game: no count, no break
    assert mt.health_of(rows)["cut_rows"] == 19
    rows.append({"epoch": 50, two: 0.95, four: 0.0})
    h = mt.health_of(rows)
    assert h["cut_rows"] == 20 and "base-2p-fr games were cut at the turn limit" in mt.guard_reason(h, guard)
    assert mt.guard_reason(h, None) is None                           # --no-guard
    assert h["losses"]["entropy"] == 0.3                              # the newest row that has one
    rows.append({"epoch": 51, two: 0.5, four: 0.0})                   # one epoch of finished games ends the count
    assert mt.health_of(rows)["cut_rows"] == 0
    # a trainer that logs no scenario (AlphaZero): all games
    az = [{"epoch": i, "environment/truncated": 1.0, "train/policy_loss": 0.6, "train/value_loss": 0.5} for i in range(20)]
    h = mt.health_of(az)
    assert (h["cut_rows"], h["cut_of"]) == (20, None) and "of the games were cut" in mt.guard_reason(h, guard)
    assert mt.health_of([]) == {} and mt.guard_reason({}, guard) is None and mt.health_text({}) is None
    # progress_of carries it, status prints it, and a guard stop is spelled out
    run = tmp_path / "e" / "v"
    make_run(run, epoch=2, rows=4)
    with open(run / "metrics.jsonl", "a") as f:
        f.writelines(json.dumps({**row, "agent_steps": 100 * row["epoch"]}) + "\n" for row in rows[:-1])
    p = mt.progress_of(run)
    assert p["health"]["cut_rows"] == 20 and p["epoch"] == 50
    launch = {"app_id": "ap-1", "rev": "a" * 40, "launch_id": "L", "budget_s": 7200.0,
              "usd_per_s": mt.usd_per_s("L40S", 8, 32), "shape": {"gpu": "L40S", "cores": 8, "GiB": 32}}
    state = {"launch_id": "L", "used_s": 3600.0, "heartbeat": 1100.0, "phase": "guard_stopped", "segments": 1,
             "progress": p, "samples": [], "guard_stop": mt.guard_reason(p["health"], guard)}
    text = mt.status_text("e/v", launch, state, False, "stopped", 1160.0)
    assert "health: 95.0% of the games of base-2p-fr cut at the turn limit (20 epochs in a row" in text
    assert "STOPPED BY THE GUARD: at least 90% of the base-2p-fr games" in text
    assert "stops the run once 90%" in mt.guard_line(guard) and "off" in mt.guard_line(None)


FAKE_TRAINER = r'''#!/usr/bin/env python3
"""Stands in for `catanrl run`: every game is cut at the turn limit; SIGTERM ends the "epoch"."""
import json, os, signal, sys, time
run = os.path.join(os.environ["CATANRL_RUNS_DIR"], "e", "v")
os.makedirs(run, exist_ok=True)
stop = []
signal.signal(signal.SIGTERM, lambda *a: stop.append(1))
def state(status, epoch):
    with open(os.path.join(run, "run.json.part"), "w") as f:
        json.dump({"status": status, "trainer_state": {"epoch": epoch}, "checkpoints": []}, f)
    os.replace(os.path.join(run, "run.json.part"), os.path.join(run, "run.json"))
with open(os.path.join(run, "config.json"), "w") as f:
    json.dump({"config": {"train": {"total_timesteps": 10**9}}}, f)
epoch = 0
while not stop and epoch < 600:
    epoch += 1
    with open(os.path.join(run, "metrics.jsonl"), "a") as f:
        f.write(json.dumps({"epoch": epoch, "agent_steps": 100 * epoch, "environment/base-2p-fr/truncated": CUT,
                            "environment/truncated": CUT, "losses/entropy": 0.01}) + "\n")
    state("running", epoch)
    time.sleep(0.1)
state("interrupted" if stop else "finished", epoch)
print("fake trainer out at epoch", epoch, flush=True)
'''


@pytest.mark.parametrize("cut, guard, phase", [(1.0, {"share": 0.9, "epochs": 5}, "guard_stopped"),
                                               (1.0, None, "finished"),
                                               (0.5, {"share": 0.9, "epochs": 5}, "finished")])
def test_supervisor_stops_a_run_whose_games_are_all_cut(tmp_path, monkeypatch, cut, guard, phase):
    class Vol:
        def commit(self): pass
        def reload(self): pass
    remote = tmp_path / "remote"
    (remote / ".venv" / "bin").mkdir(parents=True)
    trainer = remote / ".venv" / "bin" / "catanrl"
    trainer.write_text(FAKE_TRAINER.replace("CUT", repr(cut)).replace("epoch < 600", "epoch < 60" if phase == "finished" else "epoch < 600"))
    trainer.chmod(0o755)
    for name, value in (("REMOTE", str(remote)), ("VOL", str(tmp_path / "vol")), ("LOCAL_RUNS", str(tmp_path / "runs")),
                        ("LOGS", str(tmp_path / "logs")), ("volume", Vol())):
        monkeypatch.setattr(mt, name, value)
    spec = {"run": "e/v", "config": "c.yaml", "variant": "v", "rev": "a" * 40, "sets": [], "ships": [],
            "launch_id": "L", "budget_s": 600.0, "timeout_s": 750, "segment_s": None, "sync_s": 1.0, "min_sync_s": 0.0,
            "guard": guard}
    out = mt._supervise(spec)
    assert out["phase"] == phase
    state = json.loads((tmp_path / "vol" / "_modal" / "e" / "v.state.json").read_text())
    kept = json.loads((tmp_path / "vol" / "runs" / "e" / "v" / "run.json").read_text())
    if phase == "guard_stopped":
        assert "base-2p-fr games were cut at the turn limit" in state["guard_stop"]
        assert kept["status"] == "interrupted" and kept["trainer_state"]["epoch"] < 600   # stopped early, resumable
    else:
        assert "guard_stop" not in state and kept["status"] == "finished"
