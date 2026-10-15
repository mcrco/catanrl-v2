"""The launch path: signals reach the trainer through the runner, and a
refused launch does not hang."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

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


def _fake_children(monkeypatch, plan, status="finished", epoch=0):
    """run_child stand-in: records the variant and leaves a run with a `best` (at ``epoch``) in ``status``."""
    from types import SimpleNamespace

    from catanrl import runner
    from catanrl.envs.encoding import action_size, dims_for, row_layout
    from catanrl.models import build_model
    from catanrl.runs import RunWriter

    launched = []
    dims = dims_for(["base-2p"])

    def child(argv):
        name = argv[argv.index("--variant") + 1]
        launched.append(name)
        with RunWriter.open(plan.variant(name), argv=["fake"]) as w:
            model = build_model({"preset": "graph-tiny"}, dims=dims, levels="private",
                                num_actions=action_size(dims), layout=row_layout(dims, "private"))
            ref = w.save_checkpoint(model, epoch=epoch, agent_steps=0)
            w.save_trainer_state({"epoch": epoch})
            w.set_alias("best", ref, record=SimpleNamespace(eval_id="x", summary={"score_rate": 0.5}))
            w.finish(status)
        return 0, False

    monkeypatch.setattr(runner, "run_child", child)
    monkeypatch.setattr(runner, "_missing_finals", lambda run, v: False)
    return launched


def test_every_launch_path_checks_the_stored_config(monkeypatch):
    from catanrl import runner
    from catanrl.config import ConfigConflict, load_plan

    plan = load_plan("tiny", variant="2p", overrides=[])
    launched = _fake_children(monkeypatch, plan)
    runner.run_plan(plan)
    assert launched == ["2p"]
    runner.run_plan(plan)  # finished, nothing missing: skipped
    assert launched == ["2p"]
    changed = load_plan("tiny", variant="2p", overrides=["seed=99"])
    for next_ in (False, True):
        with pytest.raises(ConfigConflict, match="seed: 7 -> 99"):
            runner.run_plan(changed, next_=next_)


def test_next_leaves_stopped_runs_and_moves_on(monkeypatch):
    from catanrl import runner
    from catanrl.config import load_plan

    plan = load_plan("tiny", overrides=[])
    launched = _fake_children(monkeypatch, plan, status="time_limit")
    runner.run_plan(plan, next_=True)
    assert launched == ["2p", "4p"]  # each once: a time-limited run waits for an explicit `catanrl run`


def test_a_later_stage_runs_alone_once_its_parents_are_done(monkeypatch):
    from catanrl import runner
    from catanrl.config import load_plan

    path = "tests/fixtures/experiments/fx-lineage.yaml"
    full = load_plan(path, only="graph-tiny-s42.*", overrides=[])
    launched = _fake_children(monkeypatch, full, epoch=3000)  # g1 starts from ppo@3000
    for name in ("graph-tiny-s42.ppo", "graph-tiny-s42.g1"):
        runner.run_plan(load_plan(path, variant=name, overrides=[]))
    assert launched == ["graph-tiny-s42.ppo", "graph-tiny-s42.g1"]


def test_yaml_outside_the_repo_is_refused(tmp_path):
    from catanrl import runner
    from catanrl.config import ConfigError, load_plan

    source = tmp_path / "tiny.yaml"
    source.write_text(Path("configs/experiments/tiny.yaml").read_text())
    with pytest.raises(ConfigError, match="outside"):
        runner.run_plan(load_plan(source))


def test_a_stored_variant_takes_set_through_the_config_check():
    from catanrl.cli import main
    from catanrl.config import load_plan
    from catanrl.runs import RunWriter

    v = load_plan("tiny", variant="2p").variants[0]
    with RunWriter.open(v, argv=["t"]) as w:
        w.finish("interrupted")
    assert main(["run", "tiny/2p", "--set", "seed=99"]) == 2  # ConfigConflict, not silently ignored
    assert main(["run", "tiny/2p", "--smoke"]) != 0
