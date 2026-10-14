"""Test 8 (second half) and CLI smoke: Ctrl-C on `catanrl eval` exits nonzero."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time


def cli(*args, env=None, **kwargs):
    return subprocess.Popen([sys.executable, "-m", "catanrl.cli", *args], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, **kwargs)


def test_ctrl_c_on_eval_exits_nonzero(tmp_path):
    env = {**os.environ, "CATANRL_RUNS_DIR": str(tmp_path / "runs")}
    proc = cli("eval", "R", "R", "--suite", "heldout-2p", "--threads", "1", "--num-envs", "8", env=env)
    time.sleep(8)
    proc.send_signal(signal.SIGINT)
    out, err = proc.communicate(timeout=120)
    assert proc.returncode != 0, out + err
    assert not list((tmp_path / "runs").glob("_evals/heldout-2p/*.json"))  # no record from a partial eval


def test_validate_and_ls(tmp_path):
    env = {**os.environ, "CATANRL_RUNS_DIR": str(tmp_path / "runs")}
    proc = cli("validate", "tiny", env=env)
    out, err = proc.communicate(timeout=300)
    assert proc.returncode == 0 and out.strip() == "ok", err[-2000:]
    proc = cli("ls", env=env)
    out, err = proc.communicate(timeout=300)
    assert proc.returncode == 0, err[-2000:]
    proc = cli("pin", "nope/x@1", "ref", env=env)
    out, err = proc.communicate(timeout=300)
    assert proc.returncode == 2 and "error: RunError" in err


def test_results_and_verify_find_records_under_evals(monkeypatch, capsys):
    from test_results import SUITE, FakeHarness

    from catanrl.cli import main
    from catanrl.eval import results

    monkeypatch.setattr(results, "play_schedule", FakeHarness())
    record = results.evaluate(["R", "F"], SUITE)
    assert (record.path.parent / "provenance" / f"{record.eval_id}.start0.json").exists()
    assert main(["results"]) == 0 and record.eval_id in capsys.readouterr().out
    assert main(["verify", record.eval_id, "--games", "2"]) == 0
    assert '"identical": 2' in capsys.readouterr().out
