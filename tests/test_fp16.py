"""float16 mixed precision for PPO (``train.precision: float16``, :mod:`catanrl.train.puffer`)
and ``train.stop_epoch``, on the CPU (CPU autocast and ``GradScaler("cpu")``; nothing here
needs CUDA: the device rule's CUDA side is checked with a stubbed capability).

The config rules; the device rules; PuffeRL kept at float32 with its own autocast off; the
GradScaler step (exact against a float32 step, a non-finite gradient skipped with the scale
halved, autocast off around the optimizer so Muon's Newton-Schulz stays float32); the policy's
autocast scoped to the network's calls (graph and lstm); tiny float16 trainings: metrics, a
forced overflow, the scaler's state through a resume, no autocast at checkpoints and evals; and
stop_epoch: finished on the whole budget's schedule, not resumed by a relaunch.
"""

from __future__ import annotations

import copy
import functools
import math

import pytest
import torch
from run_helpers import train
from train_configs import tiny_ppo

from catanrl.config import ConfigError, Variant, config_hash, resolve
from catanrl.train import puffer
from catanrl.train.puffer import amp_dtype, puffer_config, scaled_step, train_ppo
from catanrl.train.puffer_policy import PufferPolicy


def resolved(**train_keys) -> dict:
    return resolve(tiny_ppo(**train_keys), experiment="test", variant="v")


# ---------------------------------------------------------------- config


def test_float16_needs_amp_and_known_precisions():
    with pytest.raises(ConfigError, match="amp: true"):
        resolved(precision="float16", amp=False)
    with pytest.raises(ConfigError, match="train.precision must be one of"):
        resolved(precision="float8", amp=True)
    half = resolved(precision="float16", amp=True)
    assert (half["train"]["precision"], half["train"]["amp"]) == ("float16", True)
    for precision, amp in (("float32", False), ("float32", True), ("bfloat16", False), ("bfloat16", True)):
        resolved(precision=precision, amp=amp)  # as before float16 existed


def test_stop_epoch_is_optional_and_hashed():
    plain = resolved()
    assert "stop_epoch" not in plain["train"]
    unset = resolved(stop_epoch=None)
    assert config_hash(unset) == config_hash(plain) and unset["train"] == plain["train"]
    stopped = resolved(stop_epoch=3)
    assert stopped["train"]["stop_epoch"] == 3 and config_hash(stopped) != config_hash(plain)
    for bad in (0, -1, 2.5, True, "3"):
        with pytest.raises(ConfigError, match="train.stop_epoch"):
            resolved(stop_epoch=bad)
    imitation = {**tiny_ppo(), "train": {"algo": "imitation", "stop_epoch": 3}}
    with pytest.raises(ConfigError, match="unknown key 'train.stop_epoch'"):
        resolve(imitation, experiment="test", variant="v")
    with pytest.raises(ConfigError, match="did you mean 'stop_epoch'"):
        resolved(stop_epochs=3)


# ---------------------------------------------------------------- device rules and PuffeRL's config


def test_amp_dtype_device_rules(monkeypatch):
    half = {"precision": "float16", "amp": True}
    assert amp_dtype({"precision": "float32", "amp": False}, "cuda") is None
    assert amp_dtype({"precision": "bfloat16", "amp": True}, "cuda") is None  # PuffeRL's own, untouched
    assert amp_dtype(half, "cpu") is torch.float16
    with pytest.raises(ValueError, match="amp: true"):
        amp_dtype({"precision": "float16", "amp": False}, "cpu")
    with pytest.raises(ValueError, match="CUDA or the CPU"):
        amp_dtype(half, "mps")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "NVIDIA GeForce GTX 1080")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (6, 1))
    with pytest.raises(ValueError, match="sm_61"):
        amp_dtype(half, "cuda")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (7, 5))  # this box's 2070 SUPER
    assert amp_dtype(half, "cuda:0") is torch.float16


def test_pufferl_stays_float32_with_its_autocast_off():
    train = dict(resolved(precision="float16", amp=True)["train"])
    config = puffer_config(train, {"device": "cpu"}, rows=16, seed=1, data_dir="/tmp")
    assert (config["precision"], config["amp"]) == ("float32", False)
    for precision, amp in (("float32", False), ("bfloat16", True)):
        train = dict(resolved(precision=precision, amp=amp)["train"])
        config = puffer_config(train, {"device": "cpu"}, rows=16, seed=1, data_dir="/tmp")
        assert (config["precision"], config["amp"]) == (precision, amp)


# ---------------------------------------------------------------- the GradScaler step


def _net(seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(12, 24), torch.nn.ReLU(), torch.nn.Linear(24, 6))


def _loss(net: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    return net(x).pow(2).mean()


def test_scaled_step_equals_the_float32_step_with_accumulation_and_clip():
    """Two scaled micro-batches, unscaled before the clip: exactly the float32 step (the scale
    is a power of two, so scaling and unscaling are exact)."""
    x = torch.randn(2, 32, 12, generator=torch.Generator().manual_seed(1))
    ref, net = _net(), _net()
    ref_opt, opt = torch.optim.Adam(ref.parameters(), lr=1e-2), torch.optim.Adam(net.parameters(), lr=1e-2)
    for micro in x:
        _loss(ref, micro).backward()
    norm = torch.nn.utils.clip_grad_norm_(ref.parameters(), 0.05)
    assert norm > 0.05  # the clip binds
    ref_opt.step()
    scaler = torch.amp.GradScaler("cpu", init_scale=1024.0)
    for micro in x:
        scaler.scale(_loss(net, micro)).backward()
    assert scaled_step(scaler, opt, net.parameters(), 0.05, "cpu")
    assert all(torch.equal(a, b) for a, b in zip(ref.parameters(), net.parameters()))
    assert scaler.get_scale() == 1024.0 and all(p.grad is None for p in net.parameters())


def test_a_non_finite_gradient_skips_the_step_and_halves_the_scale():
    net = _net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    before = copy.deepcopy(net.state_dict())
    scaler = torch.amp.GradScaler("cpu", init_scale=1024.0)
    x = torch.randn(32, 12)
    scaler.scale(_loss(net, x)).backward()
    net[0].weight.grad[0, 0] = float("inf")  # an fp16 overflow in the backward
    assert not scaled_step(scaler, opt, net.parameters(), 1.0, "cpu")
    assert all(torch.equal(v, before[k]) for k, v in net.state_dict().items())  # the step was skipped
    assert not opt.state and scaler.get_scale() == 512.0
    assert all(p.grad is None for p in net.parameters())
    scaler.scale(_loss(net, x)).backward()  # the next one steps at the backed-off scale
    assert scaled_step(scaler, opt, net.parameters(), 1.0, "cpu")
    assert scaler.get_scale() == 512.0 and int(opt.state[net[0].weight]["step"]) == 1


def test_the_optimizer_steps_outside_autocast():
    """scaled_step turns autocast off around the optimizer, whatever surrounds it: Muon's
    Newton-Schulz (heavyball runs it in the gradient's dtype) stays float32 on float32
    gradients, and equals the plain float32 step exactly. The same step inside float16
    autocast (what PuffeRL's never-exited autocast would do) differs."""
    heavyball = pytest.importorskip("heavyball")
    seen = []

    def muon(net):
        opt = heavyball.ForeachMuon(net.parameters(), lr=0.01, betas=(0.95, 0.999), eps=1e-12,
                                    heavyball_momentum=True)
        opt.register_step_pre_hook(lambda o, args, kwargs: seen.append(
            (torch.is_autocast_enabled("cpu"), {p.grad.dtype for g in o.param_groups for p in g["params"]})))
        return opt

    x = torch.randn(64, 12, generator=torch.Generator().manual_seed(2))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(heavyball.utils, "compile_mode", None)  # eager: CPU autocast reaches its matmuls
        ref = _net()
        ref_opt = muon(ref)
        _loss(ref, x).backward()
        ref_opt.step()
        net, scaler = _net(), torch.amp.GradScaler("cpu", init_scale=256.0)
        opt = muon(net)
        scaler.scale(_loss(net, x)).backward()
        with torch.autocast("cpu", dtype=torch.float16):
            assert scaled_step(scaler, opt, net.parameters(), 1e9, "cpu")
        leaked = _net()
        leaked_opt = muon(leaked)
        _loss(leaked, x).backward()
        with torch.autocast("cpu", dtype=torch.float16):
            leaked_opt.step()
    assert seen[1] == (False, {torch.float32})
    assert all(torch.equal(a, b) for a, b in zip(ref.parameters(), net.parameters()))
    assert not all(torch.equal(a, b) for a, b in zip(ref.parameters(), leaked.parameters()))


# ---------------------------------------------------------------- the policy's autocast


def _spy_autocast(monkeypatch, obj, name):
    """Record (autocast on, its dtype) at every call of ``obj.name``."""
    calls = []
    original = getattr(obj, name)

    def spy(*args, **kwargs):
        calls.append((torch.is_autocast_enabled("cpu"), torch.get_autocast_dtype("cpu")))
        return original(*args, **kwargs)

    monkeypatch.setattr(obj, name, spy)
    return calls


def test_the_graph_policy_autocasts_only_the_network(monkeypatch):
    from test_models import BASE_2P, encode, model_for, played_games

    model = model_for("graph-tiny")
    obs = encode(played_games(), BASE_2P)
    rows = obs.shape[0] - obs.shape[0] % 2
    obs = obs[:rows]
    exact = PufferPolicy(copy.deepcopy(model), num_rows=rows)
    with pytest.raises(ValueError, match="amp_dtype"):
        PufferPolicy(model, num_rows=rows, amp_dtype=torch.bfloat16)
    calls = _spy_autocast(monkeypatch, model, "heads")
    half = PufferPolicy(model, num_rows=rows, amp_dtype=torch.float16)
    with torch.no_grad():
        want_logits, want_values = exact.forward_eval(obs)
        logits, values = half.forward_eval(obs)
        train_logits, batch = half(obs)
    assert calls == [(True, torch.float16)] * 2 and not torch.is_autocast_enabled("cpu")
    for out in (logits, values, train_logits, batch.logits):
        assert out.dtype == torch.float32 and torch.isfinite(out).all()
    legal = want_logits > -1e8
    assert torch.equal(logits > -1e8, legal) and (logits[~legal] == -1e9).all()  # masked in float32, not -inf
    assert torch.allclose(logits[legal], want_logits[legal], atol=1e-3)
    assert torch.allclose(values, want_values, atol=1e-3)


def test_the_lstm_policy_autocasts_its_calls_and_keeps_float32_memory(monkeypatch):
    from test_lstm import _rollout, _segments
    from test_models import model_for

    torch.manual_seed(4)
    model = model_for("lstm-tiny")
    calls = _spy_autocast(monkeypatch, model, "remember")
    policy = PufferPolicy(model, num_rows=4, amp_dtype=torch.float16)
    obs, terminals = _segments(model, segments=4, horizon=3, seed=5)
    policy.begin_rollout()
    rollout = _rollout(policy, obs, terminals)
    assert policy._memory[0].dtype == policy._memory[1].dtype == torch.float32
    policy.begin_rollout()
    logits, batch = policy(obs, {"idx": torch.arange(4), "terminals": terminals})
    assert calls and all(on and dtype == torch.float16 for on, dtype in calls)
    assert not torch.is_autocast_enabled("cpu")
    assert rollout.dtype == logits.dtype == batch.logits.dtype == torch.float32
    legal = rollout.reshape(12, -1) > -1e8
    assert torch.equal(logits > -1e8, legal)
    assert torch.allclose(logits[legal], rollout.reshape(12, -1)[legal], atol=1e-2)


# ---------------------------------------------------------------- tiny float16 trainings


def half(**train_keys) -> dict:
    return tiny_ppo(precision="float16", amp=True, **train_keys)


def small_scaler(monkeypatch, init_scale: float = 256.0) -> None:
    """A low initial scale, so a tiny CPU run never overflows by itself."""
    monkeypatch.setattr(torch.amp, "GradScaler", functools.partial(torch.amp.GradScaler, init_scale=init_scale))


def poison(monkeypatch, at: int) -> None:
    """An inf in a gradient after the ``at``-th micro-batch backward (1-based, over the test)."""
    calls = []
    original = puffer.CatanPuffeRL.backward

    def backward(self, loss):
        original(self, loss)
        calls.append(1)
        if len(calls) == at:
            next(p for p in self.policy.parameters() if p.grad is not None).grad.view(-1)[0] = float("inf")

    monkeypatch.setattr(puffer.CatanPuffeRL, "backward", backward)


def test_float16_training_logs_and_leaves_no_autocast(monkeypatch):
    checkpoints = []
    original = puffer.CatanPuffeRL.save_checkpoint

    def save_checkpoint(self):  # where the select and monitor evals run
        if not self._in_train:
            checkpoints.append(torch.is_autocast_enabled("cpu"))
        return original(self)

    monkeypatch.setattr(puffer.CatanPuffeRL, "save_checkpoint", save_checkpoint)
    writer = train(train_ppo, half(epochs=4), "half")
    assert writer.status == "finished" and checkpoints and not any(checkpoints)
    assert not torch.is_autocast_enabled("cpu")
    rows = writer.rows
    assert [row["epoch"] for row in rows] == [1, 2, 3, 4]
    for row in rows:
        assert row["amp/optimizer_steps"] == 2 and row["amp/skipped_steps"] >= 0 and row["amp/scale"] > 0
        assert all(isinstance(v, float) and math.isfinite(v)
                   for k, v in row.items() if k.startswith("losses/") and k != "losses/explained_variance")
        assert 0.0 <= row["amp/probe_kl"] < 1e-3 and row["amp/probe_steps"] > 0
    effective = writer._run_json()["launches"][-1]["effective"]
    assert (effective["precision"], effective["amp"]) == ("float16", True)
    assert effective["grad_scaler"]["init_scale"] == 65536.0
    assert writer.load_trainer_state()["grad_scaler"]["scale"] == rows[-1]["amp/scale"]


def test_float32_training_has_no_amp_metrics_or_scaler_state():
    writer = train(train_ppo, tiny_ppo(epochs=2), "plain")
    assert not any(k.startswith("amp/") for row in writer.rows for k in row)
    assert "grad_scaler" not in writer.load_trainer_state()
    assert writer._run_json()["launches"][-1]["effective"]["precision"] == "float32"


def test_a_forced_overflow_is_one_skipped_step(monkeypatch):
    small_scaler(monkeypatch)
    poison(monkeypatch, at=3)  # epoch 2's first of its two optimizer steps (one micro-batch each)
    writer = train(train_ppo, half(epochs=4), "overflow")
    rows = writer.rows
    assert [row["amp/skipped_steps"] for row in rows] == [0, 1, 0, 0]
    assert [row["amp/skipped_total"] for row in rows] == [0, 1, 1, 1]
    assert [row["amp/scale"] for row in rows] == [256.0, 128.0, 128.0, 128.0]
    state = writer.load_trainer_state()
    assert {int(s["step"]) for s in state["optimizer"]["state"].values()} == {7}  # 8 steps, 1 skipped
    assert state["amp_skipped_total"] == 1


def test_the_scaler_state_survives_a_resume(monkeypatch):
    small_scaler(monkeypatch)
    poison(monkeypatch, at=3)
    stopped = train(train_ppo, half(epochs=4), "resumed", stop_after=2)
    assert stopped.status == "interrupted"
    saved = stopped.load_trainer_state()
    assert saved["epoch"] == 2 and saved["grad_scaler"]["scale"] == 128.0 and saved["amp_skipped_total"] == 1
    restored = {}
    original = puffer.CatanPuffeRL.restore

    def spy(self, state):
        original(self, state)
        restored.update(scaler=self.scaler.state_dict(), skipped=self.amp_skipped_total)

    monkeypatch.setattr(puffer.CatanPuffeRL, "restore", spy)
    resumed = train(train_ppo, half(epochs=4), "resumed")
    assert resumed.status == "finished" and resumed.mode == "resume"
    assert restored == {"scaler": saved["grad_scaler"], "skipped": 1}
    assert [(row["epoch"], row["amp/scale"], row["amp/skipped_total"]) for row in resumed.rows] == [
        (3, 128.0, 1), (4, 128.0, 1)]


# ---------------------------------------------------------------- stop_epoch


def test_stop_epoch_finishes_on_the_whole_budgets_schedule():
    from catanrl.runner import run_variant

    config = tiny_ppo(epochs=6, anneal_lr=True, min_lr_ratio=0.1)  # checkpoints every 2 epochs
    full = train(train_ppo, config, "full")
    config["train"]["stop_epoch"] = 3
    stopped = train(train_ppo, config, "stopped")
    assert stopped.status == "finished"
    assert [ref.epoch for ref in stopped.saved] == [0, 2, 3]  # e3 off the interval
    assert [row["epoch"] for row in stopped.rows] == [1, 2, 3]
    lr = {row["epoch"]: row["learning_rate"] for row in full.rows}
    assert {row["epoch"]: row["learning_rate"] for row in stopped.rows} == {e: lr[e] for e in (1, 2, 3)}
    assert stopped.load_trainer_state()["scheduler"]["T_max"] == full.load_trainer_state()["scheduler"]["T_max"] == 6
    weights = {ref.epoch: ref.weights_sha256 for ref in full.saved}
    assert {ref.epoch: ref.weights_sha256 for ref in stopped.saved if ref.epoch != 3} == {0: weights[0], 2: weights[2]}
    # A relaunch reopens the finished run without training it.
    cfg = resolve(config, experiment="test", variant="stopped")
    assert run_variant(Variant("test", "stopped", None, {}, cfg, config_hash(cfg)), argv=["pytest"]) == "finished"
    run = stopped._run_json()
    assert run["launches"][-1]["mode"] == "done" and len(stopped.metrics()) == 3


def test_stop_epoch_past_the_budget_is_refused():
    with pytest.raises(ValueError, match="past the budget's 4 epochs"):
        train(train_ppo, tiny_ppo(epochs=4, stop_epoch=5), "late")
