"""The launch path: signals reach the trainer through the runner, and a
refused launch does not hang."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest
import torch

from catanrl import canon
from catanrl.runs import load_experiment


def wait_for(predicate, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while not predicate():
        if time.time() > deadline:
            raise TimeoutError
        time.sleep(0.2)


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_a_signal_to_the_runner_stops_the_trainer_gracefully(tmp_path, sig):
    """`catanrl run` forwards the signal; the child writes an off-interval checkpoint."""
    runs = tmp_path / "runs"
    env = {**os.environ, "CATANRL_RUNS_DIR": str(runs)}
    argv = [sys.executable, "-m", "catanrl.cli", "run", "tiny", "--only", "2p", "--set", "train.total_timesteps=500000",
            "--set", "train.checkpoint_interval=1000"]
    proc = subprocess.Popen(argv, env=env, cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    metrics = runs / "tiny" / "2p" / "metrics.jsonl"
    try:
        wait_for(lambda: len(canon.read_jsonl(metrics)) >= 2)
        proc.send_signal(sig)
        out, _ = proc.communicate(timeout=120)
    finally:
        proc.kill()
    assert proc.returncode != 0, out[-3000:]
    run = load_experiment(runs / "tiny" / "2p")
    assert run.status == "interrupted", out[-3000:]
    assert run.checkpoints()[-1].epoch >= 2  # the off-interval checkpoint (the interval is 1000)


def test_a_refused_ppo_resume_closes_the_trainer(monkeypatch):
    from run_helpers import train
    from train_configs import tiny_ppo

    from catanrl.train import puffer

    config = tiny_ppo(epochs=2)
    writer = train(puffer.train_ppo, config, "v", stop_after=1)
    state = torch.load(writer.path / "trainer_state.pt", weights_only=False)
    torch.save({**state, "map_pool": "other"}, writer.path / "trainer_state.pt")
    closed = []
    original = puffer.CatanPuffeRL.close
    monkeypatch.setattr(puffer.CatanPuffeRL, "close", lambda self: (closed.append(1), original(self)))
    with pytest.raises(RuntimeError, match="map pool"):
        train(puffer.train_ppo, config, "v")
    assert closed  # PuffeRL's utilization thread is stopped, so the process can exit
