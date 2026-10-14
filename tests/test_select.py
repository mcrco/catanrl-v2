"""The select eval across a resume: ``best`` never moves to a worse checkpoint."""

from __future__ import annotations

import pytest
from run_helpers import train
from train_configs import tiny_ppo

from catanrl.train import select as select_module
from catanrl.train.imitation import IMITATION_KEYS, train_imitation
from catanrl.train.puffer import train_ppo

METRICS = {0: 0.1, 1: 0.2, 2: 0.9, 3: 0.5, 4: 0.3}


@pytest.fixture
def fake_evals(monkeypatch):
    """Select evals scored by epoch; records are 'cached' per checkpoint path."""
    records = {}

    class Record:
        def __init__(self, metric):
            self.summary = {"score_rate": metric}

    def fake_select_eval(select, ref, eval_runtime):
        records[str(ref)] = Record(METRICS[ref.epoch])
        return records[str(ref)], METRICS[ref.epoch]

    monkeypatch.setattr(select_module, "select_eval", fake_select_eval)
    return records


SELECT = {"suite": "select-2p", "lineup": ["POLICY:ckpt=${self}", "F"], "every": 1}


def best_epoch(writer):
    return writer.aliases()["best"]["epoch"]


def ppo_config():
    return {**tiny_ppo(epochs=4, checkpoint_interval=1), "eval": {"select": SELECT}}


def dagger_config():
    return {**tiny_ppo(), "train": {**IMITATION_KEYS, "iterations": 4, "steps_per_iteration": 8,
                                    "checkpoint_interval": 1, "minibatch_size": 64},
            "eval": {"select": SELECT}}


@pytest.mark.parametrize("trainer, config", [(train_ppo, ppo_config), (train_imitation, dagger_config)])
def test_best_survives_a_resume(tmp_path, fake_evals, trainer, config):
    straight = train(trainer, config(), "straight")
    assert best_epoch(straight) == 2

    stopped = train(trainer, config(), "run", stop_after=2)
    assert best_epoch(stopped) == 2  # e2's select ran after its trainer state was written
    resumed = train(trainer, config(), "run")
    assert best_epoch(resumed) == 2  # e3 (0.5) must not take over from e2 (0.9)


def test_best_moves_after_a_crash_between_record_and_alias(tmp_path, fake_evals, monkeypatch):
    """e2's select record exists but its alias was never written: the resume moves best to e2."""
    from catanrl.runs import RunWriter

    real = RunWriter.set_alias

    def crash_at_e2(self, name, ref, **kwargs):
        if ref.epoch == 2:
            raise KeyboardInterrupt  # killed between the select record and the alias
        real(self, name, ref, **kwargs)

    from catanrl.eval import results

    # Records are found like real cached ones: e2's exists although its alias does not.
    monkeypatch.setattr(results, "find", lambda lineup, suite, **kw: fake_evals.get(lineup[0].split("ckpt=")[1]))
    monkeypatch.setattr(RunWriter, "set_alias", crash_at_e2)
    with pytest.raises(KeyboardInterrupt):
        train(train_ppo, ppo_config(), "run")
    monkeypatch.setattr(RunWriter, "set_alias", real)
    resumed = train(train_ppo, ppo_config(), "run")
    assert best_epoch(resumed) == 2
