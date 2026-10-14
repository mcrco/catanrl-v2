"""AlphaZero: targets, playout caps, the replay buffer, and runs through RunWriter."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from catanrl import canon
from catanrl.config import Variant, config_hash, resolve
from catanrl.envs.puffer import plan_env
from catanrl.runs import RunWriter, load_experiment
from catanrl.train.alphazero import (
    ALPHAZERO_KEYS,
    ReplayBuffer,
    SelfPlay,
    _Pending,
    temperature,
    train_alphazero,
    value_targets,
)
from catanrl.train.puffer import build_policy_model


@pytest.fixture(autouse=True)
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CATANRL_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.chdir(tmp_path)


TRAIN = {**ALPHAZERO_KEYS, "iterations": 2, "steps_per_iteration": 24, "sims": 4, "fast_sims": 2,
         "full_search_prob": 0.5, "min_replay": 1, "updates_per_iteration": 3, "minibatch_size": 16,
         "replay_capacity": 500, "checkpoint_interval": 1, "search_rows": 16}
ENV = {"scenarios": [{"game": "base-2p", "maps": 4, "turn_limit": 8}], "num_envs": 4}  # short games


def variant(name="t", seed=3, eval=None, init_from=None, **train) -> Variant:
    raw = {"seed": seed, "env": ENV, "obs": {"level": "private"}, "model": {"preset": "graph-tiny"},
           "train": {**TRAIN, **train}, "runtime": {"device": "cpu", "threads": 2, "determinism": "strict"},
           **({"eval": eval} if eval else {}), **({"init_from": init_from} if init_from else {})}
    cfg = resolve(raw, experiment="az", variant=name)
    return Variant("az", name, None, {}, cfg, config_hash(cfg))


def selfplay(**train):
    cfg = {"model": {"preset": "graph-tiny"}, "obs": {"level": "private"}}
    plan = plan_env(ENV, pool_seed=1)
    model = build_policy_model(cfg, plan.dims).eval()
    play = SelfPlay(plan, {**TRAIN, **train}, model=model, num_envs=4, threads=2, seed=5, device="cpu", mode="strict")
    replay = ReplayBuffer(5000, play.vec.observation_size, play.vec.action_size, play.players)
    return play, replay, model


def test_value_targets_are_seat_relative():
    targets, present = value_targets(winner=2, seat=1, num_players=3, max_players=4, truncation="share")
    assert targets.tolist() == [0, 1, 0, 0] and present.tolist() == [1, 1, 1, 0]  # slot 1 = seat 2
    targets, _ = value_targets(winner=0, seat=2, num_players=3, max_players=4, truncation="share")
    assert targets.tolist() == [0, 1, 0, 0]  # seat 0 is one place after seat 2
    shared, _ = value_targets(winner=None, seat=0, num_players=4, max_players=6, truncation="share")
    assert shared.tolist() == [0.25] * 4 + [0, 0]
    zero, _ = value_targets(winner=None, seat=0, num_players=2, max_players=2, truncation="zero")
    assert zero.tolist() == [0, 0]


def test_temperature_decays_over_the_first_moves():
    train = {"temperature_start": 1.0, "temperature_end": 0.2, "temperature_moves": 4}
    assert [temperature(train, ply) for ply in range(6)] == pytest.approx([1.0, 0.8, 0.6, 0.4, 0.2, 0.2])
    assert temperature({**train, "temperature_moves": 0}, 0) == 0.2


def test_finished_games_file_their_positions_with_outcomes():
    play, replay, _ = selfplay()
    b, players = play.buffers, play.players
    obs = np.ones(play.vec.observation_size, np.float16)
    policy = np.zeros(play.vec.action_size, np.float16)
    play.pending[0] = [_Pending(obs, policy, seat=0, num_players=2), _Pending(obs, None, seat=1, num_players=2)]
    play.pending[1] = [_Pending(obs, None, seat=0, num_players=2)]
    b["terminals"][:] = 0
    b["rewards"][:] = 0
    b["terminals"][0 : 2 * players] = 1          # envs 0 and 1 end
    b["rewards"][1] = 1.0                         # env 0: seat 1 wins; env 1: truncated
    play._finish(replay)
    assert replay.size == 3 and play.games == 2
    assert replay.value[:3].tolist() == [[0, 1], [1, 0], [0.5, 0.5]]
    assert replay.has_policy[:3].tolist() == [True, False, False]
    assert play.pending[0] == [] and play.ply[0] == 0


def test_policy_targets_cover_legal_actions_and_the_playout_cap_is_kept():
    play, replay, model = selfplay(full_search_prob=0.5)
    stats = {"leaves": 0.0, "full": 0.0, "depth": 0.0, "max_depth": 0.0}
    rng = np.random.default_rng(0)
    for step in range(60):
        play.step(0, step, rng, replay, stats)
    pending = sum(len(rows) for rows in play.pending)
    assert replay.size > 0 and replay.size + pending == play.decisions
    with_policy = int(replay.has_policy[: replay.size].sum()) + sum(
        r.policy is not None for rows in play.pending for r in rows)
    assert with_policy == stats["full"] and 0.3 < stats["full"] / play.decisions < 0.7
    rows = replay.obs[: replay.size].astype(np.float32)
    masks = rows[:, model.mask_offset : model.mask_offset + model.num_actions] > 0
    targets = replay.policy[: replay.size].astype(np.float32)
    has = replay.has_policy[: replay.size]
    assert np.allclose(targets[has].sum(1), 1.0, atol=2e-3)
    assert np.all(targets[~masks] == 0) and np.all(targets[~has] == 0)


def test_replay_buffer_is_a_fifo_and_round_trips():
    replay = ReplayBuffer(5, obs_size=2, num_actions=3, max_players=2)
    for start in (0, 3, 6):
        n = 3
        replay.add(np.arange(start, start + n)[:, None].repeat(2, 1), np.zeros((n, 3)), np.ones(n, bool),
                   np.zeros((n, 2)), np.ones((n, 2)))
    assert replay.size == 5 and replay.added == 9
    state = replay.state()
    assert state["obs"][:, 0].tolist() == [4, 5, 6, 7, 8]  # oldest first
    other = ReplayBuffer(5, obs_size=2, num_actions=3, max_players=2)
    other.restore(state)
    assert sorted(other.obs[:, 0].tolist()) == [4, 5, 6, 7, 8] and other.added == 9
    small = ReplayBuffer(3, obs_size=2, num_actions=3, max_players=2)
    small.restore(state)  # a smaller capacity keeps the newest
    assert sorted(small.obs[:, 0].tolist()) == [6, 7, 8]


def run(v: Variant) -> RunWriter:
    with RunWriter.open(v, argv=["test"]) as writer:
        train_alphazero(writer.variant_for_training, writer)
    return writer


@pytest.mark.train  # about 10 s with its select evals (the resume test below runs RunWriter by default)
def test_a_tiny_run_trains_checkpoints_and_selects(monkeypatch):
    v = variant(eval={"select": {"suite": {"name": "select-2p", "boards": 1}, "lineup": ["POLICY:ckpt=${self}", "F"],
                                 "every": 2}})
    writer = run(v)
    run_ = load_experiment("az/t")
    assert writer.status == "finished" and [c.epoch for c in run_.checkpoints()] == [0, 1, 2]
    metrics = run_.metrics()
    assert [m["epoch"] for m in metrics] == [1, 2] and metrics[-1]["replay"] > 0
    assert "train/policy_loss" in metrics[-1] and metrics[-1]["selfplay/leaves_per_s"] > 0
    assert "best" in run_.aliases()
    net = run_.build_policy("latest")
    assert net.dims_agnostic


def test_resume_restores_the_replay_buffer_and_counters(monkeypatch):
    v = variant()
    real = RunWriter.should_stop

    def stop_after_one(self):
        if self.launch == 0 and len(canon.read_jsonl(self.path / "metrics.jsonl")) >= 1:
            self.stop_reason = "interrupted"
            return True
        return real(self)

    monkeypatch.setattr(RunWriter, "should_stop", stop_after_one)
    assert run(v).status == "interrupted"
    path = load_experiment("az/t").path
    saved = torch.load(path / "trainer_state.pt", weights_only=False)
    assert saved["epoch"] == 1 and saved["replay"]["size"] > 0

    import catanrl.train.alphazero as az

    restored = {}
    original = az.ReplayBuffer.restore

    def spy(self, state):
        original(self, state)
        restored["replay"] = self.state()

    monkeypatch.setattr(az.ReplayBuffer, "restore", spy)
    assert run(v).status == "finished"
    for key in ("size", "added"):
        assert restored["replay"][key] == saved["replay"][key]
    for key in ("obs", "policy", "has_policy", "value", "present"):
        assert torch.equal(restored["replay"][key], saved["replay"][key])
    final = torch.load(path / "trainer_state.pt", weights_only=False)
    assert final["epoch"] == 2 and final["decisions"] > saved["decisions"] and final["replay"]["added"] > saved["replay"]["added"]


def test_a_warm_start_selects_against_its_start(monkeypatch):
    """From a network, AlphaZero's default select plays each checkpoint against that network
    (pinned when the run is created); e0, the start's weights, is best first, and best moves
    only above every earlier checkpoint."""
    from catanrl.train import select as select_module

    run(variant("start"))
    start = load_experiment("az/start").resolve("latest")
    rates = {0: 0.5, 1: 0.45, 2: 0.55, 3: 0.52}
    played = []

    def fake_select_eval(select, ref, eval_runtime):
        played.append(select_module.lineup_for(select, ref))
        return SimpleNamespace(eval_id=f"x{ref.epoch}", summary={"score_rate": rates[ref.epoch]}), rates[ref.epoch]

    monkeypatch.setattr(select_module, "select_eval", fake_select_eval)
    warm = variant("warm", init_from={"ckpt": "az/start@latest"}, iterations=3)
    assert warm.config["eval"]["select"]["lineup"] == ["POLICY:ckpt=${self}", "POLICY:ckpt=${init}"]
    run(warm)
    warm_run = load_experiment("az/warm")
    ckpts = warm_run.checkpoints()
    assert ckpts[0].weights_sha256 == start.weights_sha256
    assert {tuple(lineup) for lineup in played} == {(f"POLICY:ckpt={c}", f"POLICY:ckpt={start}") for c in ckpts}
    best = [row for row in canon.read_jsonl(warm_run.path / "aliases.jsonl") if row["name"] == "best"]
    assert [(row["epoch"], row["metric"]) for row in best] == [(0, 0.5), (2, 0.55)]


@pytest.mark.train
def test_the_start_scores_half_against_itself():
    """The real select eval: e0 against the start is a mirror (exactly 50%), and best is e0 or above 50%."""
    run(variant("start"))
    start = str(load_experiment("az/start").resolve("latest"))
    run(variant("warm", init_from={"ckpt": "az/start@latest"},
                eval={"select": {"suite": {"name": "select-2p", "boards": 2}}}))
    warm_run = load_experiment("az/warm")
    records = {r.data["refs"][0]: r for r in warm_run.evals("select-2p")}
    assert sorted(records) == sorted(str(c) for c in warm_run.checkpoints())
    assert all(r.data["refs"][1] == start and all(r.data["fair"]) for r in records.values())
    assert records[str(warm_run.resolve(0))].summary["score_rate"] == 0.5
    best = warm_run.aliases()["best"]
    assert best["epoch"] == 0 or best["metric"] > 0.5


@pytest.mark.train
def test_self_play_and_learning_are_deterministic():
    shas = []
    for name in ("a", "b"):
        run(variant(name))
        shas.append([c.weights_sha256 for c in load_experiment(f"az/{name}").checkpoints()])
    assert shas[0] == shas[1]
    run(variant("c", seed=4))
    assert load_experiment("az/c").checkpoints()[-1].weights_sha256 != shas[0][-1]
