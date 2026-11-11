"""AlphaZero: targets, playout caps, the replay buffer, and runs through RunWriter."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import cppanatron as cpp
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from catanrl import canon
from catanrl.config import Variant, config_hash, resolve
from catanrl.envs.encoding import encoder_dims
from catanrl.envs.puffer import plan_env
from catanrl.runs import RunWriter, load_experiment
from catanrl.search import BatchEvaluator, run_pools
from catanrl.train.alphazero import (
    ALPHAZERO_KEYS,
    PROBE_ROWS,
    Monitors,
    ReplayBuffer,
    RootOutputs,
    Rounds,
    SelfPlay,
    _Pending,
    check_train,
    gumbel_target,
    pruned_visits,
    temperature,
    train_alphazero,
    value_targets,
)
from catanrl.train.puffer import build_policy_model


@pytest.fixture(autouse=True)
def runs_dir(request, tmp_path, monkeypatch):
    if request.node.get_closest_marker("runs") is None:  # those read the real one (conftest)
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


def selfplay(level="private", env=ENV, **train):
    cfg = {"model": {"preset": "graph-tiny"}, "obs": {"level": level}}
    plan = plan_env(env, pool_seed=1)
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


def run(v: Variant, **kwargs) -> RunWriter:
    with RunWriter.open(v, argv=["test"]) as writer:
        train_alphazero(writer.variant_for_training, writer, **kwargs)
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


# ---------------------------------------------------------------- leaves per round, recorded moves, kernels


@pytest.mark.train
def test_the_defaults_train_the_weights_from_before_the_keys():
    """pending, fast_pending, record_fast and kernels at their defaults: the checkpoints this tiny
    run wrote before the keys existed (main e57b88a; they move with the torch build)."""
    run(variant())
    shas = [c.weights_sha256[:12] for c in load_experiment("az/t").checkpoints()]
    assert shas == ["4bd9521fd942", "337abc40ae6b", "2bfb0524faff"]


def test_record_fast_false_stores_only_the_full_searches():
    play, replay, _ = selfplay(full_search_prob=0.5, record_fast=False)
    stats = {"leaves": 0.0, "full": 0.0, "depth": 0.0, "max_depth": 0.0}
    rng = np.random.default_rng(0)
    for step in range(60):
        play.step(0, step, rng, replay, stats)
    waiting = [r for rows in play.pending for r in rows]
    assert 0 < replay.size and replay.added + len(waiting) == stats["full"] < play.decisions  # every move counts
    assert play.games > 0 and replay.has_policy[: replay.size].all() and all(r.policy is not None for r in waiting)
    assert np.allclose(replay.policy[: replay.size].astype(np.float32).sum(1), 1.0, atol=2e-3)


@pytest.mark.train
def test_a_run_that_records_only_full_searches_learns_and_logs(capsys):
    run(variant(record_fast=False))
    run_ = load_experiment("az/t")
    state = torch.load(run_.path / "trainer_state.pt", weights_only=False)
    last = run_.metrics()[-1]
    assert state["replay"]["has_policy"].all() and 0 < last["positions"] == state["replay"]["added"] < last["decisions"]
    assert last["games"] > 0 and last["replay"] == state["replay"]["size"]
    assert "train/policy_loss" in last and "oos/value_bce" in last and "search/overrule" in last  # the monitors too
    assert last["selfplay/rounds"] == 5.0  # the root and 4 sims (the fast searches end after 3 rounds)
    line = [row for row in capsys.readouterr().out.splitlines() if row.startswith("az/t: iteration 2/2: ")]
    assert len(line) == 1 and f"{last['games']} games (+" in line[0] and f"{last['positions']} positions (+" in line[0]
    assert all(part in line[0] for part in (" games/h, ", " leaves/s, ", " s, ETA 0.0 h"))


def test_pending_cuts_the_rounds_and_keeps_the_visit_counts():
    """One leaf a search per round: the root, then ``sims`` rounds. With ``pending`` leaves a round
    the same visits take a fraction of the rounds, the target is still the visit distribution and
    nothing is left in flight."""
    rounds = {}
    for pending in (1, 8):
        play = selfplay(root="puct", sims=48, pending=pending, fast_pending=pending)[0]
        assert play.full_pool.config.max_pending == play.fast_pool.config.max_pending == pending
        b, pool = play.buffers, play.full_pool
        acting = np.flatnonzero(b["acting"])
        sids = [pool.add(play.vec.game(int(b["env_index"][row])), seed=int(row)) for row in acting]
        evaluator = Rounds(play.evaluator)
        run_pools([pool], evaluator, batch_rows=[len(sids) * pending])
        rounds[pending] = evaluator.count
        assert pool.pending == 0
        for sid in sids:
            visits = np.asarray(pool.visits(sid), np.float64)
            assert visits.sum() == 48 and visits[pool.select(sid, 0.0)] == visits.max()
            assert np.allclose(pool.policy_target(sid), visits / 48, atol=1e-6)
    assert rounds[1] == 49 and rounds[8] <= 12
    play, replay, _ = selfplay(sims=48, fast_sims=12, pending=8, fast_pending=4)
    stats = {"leaves": 0.0, "full": 0.0, "depth": 0.0, "max_depth": 0.0}
    play.step(0, 0, np.random.default_rng(0), replay, stats)  # both pools, a call of the evaluator per round
    assert stats["rounds"] <= 12 and stats["leaves"] >= 13 * play.decisions


def test_fast_searches_get_no_dirichlet_noise():
    play = selfplay(root="puct", dirichlet_noise=True)[0]
    assert play.full_pool.config.dirichlet_alpha > 0 and play.fast_pool.config.dirichlet_alpha == 0
    quiet = selfplay(root="puct", dirichlet_noise=False)[0]
    assert quiet.full_pool.config.dirichlet_alpha == 0 and repr(quiet.fast_pool.config) == repr(play.fast_pool.config)


class Copies(BatchEvaluator):
    """The fused kernels' contract on the CPU: built from a copy of the weights (FusedGraph's
    buffers), so it evaluates the weights of the moment it was built."""

    def __init__(self, model, *, kernels, **kwargs):
        super().__init__(copy.deepcopy(model) if kernels == "fused" else model, **kwargs)


def test_fused_kernels_follow_the_weights(monkeypatch):
    """The evaluator is rebuilt after every learn(): self-play searches with the weights the
    module has (the same run as ``kernels: torch``). Left on its first weights, the check before
    the next iteration's self-play stops the run."""
    import catanrl.train.alphazero as az

    monkeypatch.setattr(az, "BatchEvaluator", Copies)
    run(variant("torch"))
    run(variant("fused", kernels="fused"))
    shas = [[c.weights_sha256 for c in load_experiment(f"az/{name}").checkpoints()] for name in ("torch", "fused")]
    assert shas[0] == shas[1] and len(set(shas[0])) == 3
    monkeypatch.setattr(az.SelfPlay, "refresh", lambda self: None)
    with pytest.raises(RuntimeError, match="stale"):
        run(variant("stale", kernels="fused", learning_rate=0.01))


# ---------------------------------------------------------------- monitors


@pytest.mark.parametrize("target", [{}, {"policy_target": "gumbel", "target_c_scale": 0.01},
                                    {"policy_target": "pruned", "root": "puct", "sims": 16}],
                         ids=["engine", "gumbel", "pruned"])
def test_monitors_are_logging_only(target):
    """With and without the monitors: the same replay rows and targets, weights, optimizer,
    RNG and logged training values, bit for bit; with a rebuilt or pruned target too, whose
    monitors also read the searches' own targets."""
    run(variant("on", **target))
    run(variant("off", **target), monitors=False)
    on, off = load_experiment("az/on"), load_experiment("az/off")
    a = torch.load(on.path / "trainer_state.pt", weights_only=False)
    b = torch.load(off.path / "trainer_state.pt", weights_only=False)
    assert "monitors" in a and "monitors" not in b
    assert a["replay"]["size"] > 0 and a["replay"]["added"] == b["replay"]["added"]
    for key in ("obs", "policy", "has_policy", "value", "present"):
        assert torch.equal(a["replay"][key], b["replay"][key])
    assert all(torch.equal(value, b["model"][key]) for key, value in a["model"].items())
    for param, state in a["optimizer"]["state"].items():
        assert all(torch.equal(torch.as_tensor(v), torch.as_tensor(b["optimizer"]["state"][param][k]))
                   for k, v in state.items())
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"]) and a["rng"]["python"] == b["rng"]["python"]
    assert [c.weights_sha256 for c in on.checkpoints()] == [c.weights_sha256 for c in off.checkpoints()]
    added = ("oos/", "search/overrule", "search/target_entropy", "search/net_entropy", "search/kl_target_net",
             "target/", "drift/", "throughput/", "selfplay/decisions_per_s", "selfplay/leaves_per_s",
             "selfplay/games_per_hour")
    for x, y in zip(on.metrics(), off.metrics(), strict=True):
        assert "train/policy_loss" in x and "drift/kl_init" in x and "search/overrule" in x
        assert ("target/entropy" in x) == ("target/kl_net" in x) == bool(target)
        assert ("target/pruned" in x) == (target.get("policy_target") == "pruned")
        assert {k: v for k, v in x.items() if not k.startswith(added)} == {
            k: v for k, v in y.items() if not k.startswith(added)}


def test_the_value_health_reads_only_the_newest_rows_out_of_sample():
    play, filled, model = selfplay()
    rng = np.random.default_rng(0)
    stats = {"leaves": 0.0, "full": 0.0, "depth": 0.0, "max_depth": 0.0}
    for step in range(40):
        play.step(0, step, rng, filled, stats)
    assert filled.size >= 30
    replay = ReplayBuffer(25, play.vec.observation_size, play.vec.action_size, play.players)
    obs = filled.obs[:30]
    present = np.ones((30, 2), np.float32)
    old = np.ones((18, 2), np.float32)                     # seat-sums of 2: wrong if read
    new = np.eye(2, dtype=np.float32)[np.arange(12) % 2]   # winners alternate
    replay.add(obs[:18], np.zeros((18, replay.shapes[1])), np.zeros(18, bool), old, present[:18])
    replay.add(obs[18:], np.zeros((12, replay.shapes[1])), np.zeros(12, bool), new, present[18:])  # wraps around
    monitors = Monitors(model, device="cpu", rows=5)
    logs = monitors.before_learn(replay, 12)
    with torch.no_grad():
        logits = model.value_logits(torch.from_numpy(obs[18:].astype(np.float32)))[0]
    target = torch.from_numpy(new)
    p = torch.sigmoid(logits)
    assert logs["oos/rows"] == 12 and logs["oos/target_sum_2p"] == 1.0
    assert logs["oos/value_bce"] == pytest.approx(float(F.binary_cross_entropy_with_logits(logits, target)), rel=1e-5)
    assert logs["oos/value_brier"] == pytest.approx(float(((p - target) ** 2).mean()), rel=1e-5)
    assert logs["oos/value_sum_2p"] == pytest.approx(float(p.sum(1).mean()), rel=1e-5)
    assert logs["oos/sum_gap_2p"] == pytest.approx(abs(float(p.sum(1).mean()) - 1.0), rel=1e-4, abs=1e-6)
    assert not any(k.startswith("search/") for k in logs)  # no searches seen
    assert monitors.before_learn(replay, 0) == {}


def test_the_overrule_rate_compares_targets_with_the_raw_argmax():
    play, _, model = selfplay()
    b = play.buffers
    rows = b["observations"][np.flatnonzero(b["acting"])]
    rows = np.concatenate([rows, rows])
    n, mask = len(rows), slice(model.mask_offset, model.mask_offset + model.num_actions)
    with torch.no_grad():
        logp = F.log_softmax(model.policy_logits(torch.from_numpy(rows)).float(), dim=1)
    best = logp.argmax(1).numpy()
    second = logp.clone()
    second[torch.arange(n), best] = -1e30
    other = second.argmax(1).numpy()
    assert all(rows[i, mask][other[i]] > 0 for i in range(n))  # a legal alternative
    chosen = np.where(np.arange(n) < n // 2, best, other)  # the second half is overruled
    targets = np.zeros((n, model.num_actions), np.float16)
    targets[np.arange(n), chosen] = 1.0
    forced = rows[:1].copy()
    forced[0, mask] = 0
    forced[0, model.mask_offset + int(best[0])] = 1.0  # one legal action: left out
    forced_target = np.zeros((1, model.num_actions), np.float16)
    forced_target[0, int(other[0])] = 1.0
    monitors = Monitors(model, device="cpu", rows=3)
    monitors.searched(rows[: n // 2], targets[: n // 2])  # two steps
    monitors.searched(np.concatenate([rows[n // 2 :], forced]), np.concatenate([targets[n // 2 :], forced_target]))
    logs = monitors.before_learn(ReplayBuffer(5, rows.shape[1], model.num_actions, 2), 0)
    legal = logp > -1e8
    net_entropy = -(logp.exp() * torch.where(legal, logp, torch.zeros_like(logp))).sum(1).mean()
    assert logs["search/overrule_rows"] == n and logs["search/overrule"] == 0.5
    assert logs["search/target_entropy"] == pytest.approx(0.0, abs=1e-6)
    assert logs["search/net_entropy"] == pytest.approx(float(net_entropy), rel=1e-5)
    assert logs["search/kl_target_net"] == pytest.approx(float(-logp[torch.arange(n), chosen].mean()), rel=1e-5)
    assert not any(k.startswith("target/") for k in logs)
    # policy_target: gumbel hands the rebuilt targets too: target/* reads them, search/* still the searches'.
    rebuilt = np.zeros((n, model.num_actions), np.float16)
    rebuilt[np.arange(n), other] = 1.0
    both = Monitors(model, device="cpu", rows=3)
    both.searched(np.concatenate([rows, forced]), np.concatenate([targets, forced_target]),
                  np.concatenate([rebuilt, forced_target]))
    logged = both.before_learn(ReplayBuffer(5, rows.shape[1], model.num_actions, 2), 0)
    assert {k: v for k, v in logged.items() if k.startswith("search/")} == pytest.approx(logs, rel=1e-6)
    assert logged["target/overrule"] == 1.0 and logged["target/entropy"] == pytest.approx(0.0, abs=1e-6)
    assert logged["target/kl_net"] == pytest.approx(float(-logp[torch.arange(n), other].mean()), rel=1e-5)
    assert "target/pruned" not in logged
    # policy_target: pruned hands each search's pruned share too: target/pruned is their mean (forced rows left out).
    both.begin()
    shares = np.linspace(0.0, 0.5, n + 1)
    both.searched(np.concatenate([rows, forced]), np.concatenate([targets, forced_target]),
                  np.concatenate([rebuilt, forced_target]), shares)
    pruned = both.before_learn(ReplayBuffer(5, rows.shape[1], model.num_actions, 2), 0)
    assert pruned["target/pruned"] == pytest.approx(shares[:n].mean())
    assert {k: v for k, v in pruned.items() if k != "target/pruned"} == pytest.approx(logged, rel=1e-6)
    monitors.begin()
    assert monitors.before_learn(ReplayBuffer(5, rows.shape[1], model.num_actions, 2), 0) == {}
    both.begin()
    assert both.before_learn(ReplayBuffer(5, rows.shape[1], model.num_actions, 2), 0) == {}


def test_an_unchanged_network_has_not_drifted_and_the_probe_phases_read_the_rows():
    run(variant(min_replay=10**9))  # no update ever
    metrics = load_experiment("az/t").metrics()
    assert [m["drift/kl_init"] for m in metrics] == [0.0, 0.0]
    assert [m["drift/agree_init"] for m in metrics] == [1.0, 1.0]
    assert all(m["search/overrule_rows"] > 0 and 0.0 <= m["search/overrule"] <= 1.0 for m in metrics)
    assert any(m.get("oos/rows", 0) > 0 for m in metrics) and "train/policy_loss" not in metrics[-1]
    assert all(m["throughput/monitor_s"] > 0 for m in metrics)
    state = torch.load(load_experiment("az/t").path / "trainer_state.pt", weights_only=True)["monitors"]
    rows, init = state["probe_obs"].numpy().astype(np.float32), state["probe_init_logp"]
    assert 0 < len(rows) <= PROBE_ROWS and init.shape == (len(rows), 314) and metrics[0]["drift/rows"] == len(rows)
    plan = plan_env(ENV, pool_seed=1)
    names = np.array(cpp.GraphActionSpace(encoder_dims(plan.dims)).names())
    model = build_policy_model({"model": {"preset": "graph-tiny"}, "obs": {"level": "private"}}, plan.dims)
    legal = rows[:, model.mask_offset : model.mask_offset + model.num_actions] > 0
    assert np.all(legal.sum(1) > 1)  # non-forced decisions only
    phases = Monitors(model, device="cpu", rows=8)
    phases.restore(state)
    allowed = {"robber": ("ROBBER@",), "discard": ("DISCARD_",), "setup": ("SETTLEMENT@", "ROAD@"),
               "build": ("SETTLEMENT@", "CITY@", "ROAD@", "END_TURN", "BUY_", "PLAY_", "MARITIME_", "MONOPOLY_",
                         "YEAR_OF_PLENTY_")}
    for phase, prefixes in allowed.items():
        for row in np.flatnonzero(phases.phases[phase].numpy()):
            assert all(name.startswith(prefixes) for name in names[legal[row]]), (phase, names[legal[row]])
    assert phases.phases["setup"].any() and phases.phases["robber"].any() and phases.phases["build"].any()
    assert metrics[0]["drift/rows_setup"] == int(phases.phases["setup"].sum())


def test_the_drift_probe_is_fixed_for_the_run_and_survives_a_resume(monkeypatch):
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
    saved = torch.load(path / "trainer_state.pt", weights_only=True)["monitors"]
    captures = []
    monkeypatch.setattr(Monitors, "capture", lambda self, *a, **k: captures.append(1))
    assert run(v).status == "finished" and not captures  # the resume restores, never redraws
    final = torch.load(path / "trainer_state.pt", weights_only=True)
    assert final["epoch"] == 2
    for key in ("probe_obs", "probe_init_logp"):
        assert torch.equal(final["monitors"][key], saved[key])
    metrics = load_experiment("az/t").metrics()
    assert [m["launch"] for m in metrics] == [0, 1]
    assert metrics[1]["drift/rows"] == metrics[0]["drift/rows"] == len(saved["probe_obs"])
    assert metrics[1]["drift/kl_init"] > 0.0  # against the start, not the resumed weights


# ---------------------------------------------------------------- rebuilt policy targets

TARGET_KEYS = ("target_c_visit", "target_c_scale", "target_rescale")


def searched_roots(play: SelfPlay, steps: int, rng: np.random.Generator) -> list[dict]:
    """Every acting row's full-budget search over ``steps`` VecEnv steps of ``play`` (moves sampled
    from the search at temperature 1): the engine's target and what gumbel_target reads."""
    b, pool, out = play.buffers, play.full_pool, []
    for _ in range(steps):
        acting = np.flatnonzero(b["acting"])
        roots = RootOutputs(pool, play.mask)
        sids = []
        for row in acting:
            env, seat = int(b["env_index"][row]), int(b["seat"][row])
            sids.append(pool.add(play.vec.game(env), seed=int(rng.integers(2**62)),
                                 tracker=play.vec.tracker(env, seat) if play.dets else None))
            roots.expect(sids[-1])
        run_pools([roots], play.evaluator, batch_rows=[len(acting)])
        for row, sid in zip(acting, sids):
            legal, logits, value = roots.root(sid)
            out.append({"engine": pool.policy_target(sid).astype(np.float64), "logits": logits, "value": value,
                        "legal": legal & (b["observations"][row, play.mask] > 0), "visits": pool.visits(sid),
                        "q": pool.q(sid), "row": b["observations"][row].copy()})
            b["actions"][row] = pool.select(sid, 1.0)
            pool.remove(sid)
        play.vec.step()
    return out


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def test_the_target_constants_default_to_the_engines_and_never_reach_the_search():
    from catanrl.config import train_defaults

    assert train_defaults("alphazero") == ALPHAZERO_KEYS  # configs/train/alphazero.yaml
    engine = cpp.SearchConfig()
    assert tuple(ALPHAZERO_KEYS[k] for k in TARGET_KEYS) == (
        engine.gumbel_c_visit, engine.gumbel_c_scale, engine.gumbel_rescale_values)
    assert ALPHAZERO_KEYS["policy_target"] == "engine"
    same = repr(selfplay(dets=1)[0].full_pool.config)
    soft = selfplay(dets=1, policy_target="gumbel", target_c_visit=0.0, target_c_scale=0.01, target_rescale=False,
                    target_prior_temperature=10.0)[0]
    assert repr(soft.full_pool.config) == same and "gumbel_c_scale=0.1," in same


def test_check_train_refuses_targets_it_cannot_build():
    assert check_train(ALPHAZERO_KEYS)["policy_target"] == "engine"
    for dets in (0, 1):
        pruned = {**ALPHAZERO_KEYS, "policy_target": "pruned", "root": "puct", "dets": dets}
        assert check_train(pruned)["policy_target"] == "pruned"
    for bad, match in (({"policy_target": "visits"}, "engine, gumbel or pruned"),
                       ({"policy_target": "gumbel", "root": "puct"}, "Gumbel root's improved policy; a PUCT"),
                       ({"policy_target": "pruned"}, "PUCT root's visits"),  # the default root is gumbel
                       ({"policy_target": "pruned", "root": "puct", "dets": 2}, "dets 0 or 1"),
                       ({"policy_target": "pruned", "root": "puct", "target_c_scale": 0.01},
                        "set policy_target: gumbel"),
                       ({"policy_target": "gumbel", "target_c_scale": -0.1}, "non-negative"),
                       ({"policy_target": "gumbel", "target_prior_temperature": 0.0}, "positive"),
                       ({"pending": 0}, "leaves per network round"), ({"fast_pending": 0}, "leaves per network round"),
                       ({"kernels": "cuda"}, "torch or fused"),
                       ({"target_c_scale": 0.01}, "set policy_target: gumbel"),  # silently unused otherwise
                       ({"target_prior_temperature": 10.0}, "set policy_target: gumbel")):
        with pytest.raises(ValueError, match=match):
            check_train({**ALPHAZERO_KEYS, **bad})


@pytest.mark.parametrize("level,dets,game", [("private", 0, "base-2p"), ("private", 1, "base-2p"),
                                             ("full", 3, "base-4p")])
def test_gumbel_target_at_the_engine_constants_is_the_engines(level, dets, game):
    """Rebuilt from a searched root's statistics and the network's outputs there, at the engine's
    constants the target is the engine's but for its float32 rounding: on the true game, on one
    tracker world, and on three worlds of a full-information network (whose rows, so priors and
    values, differ per world); roots with unvisited actions (v_mix) and more than gumbel_m included.
    A gentler sigma gives a distribution over the same legal actions."""
    env = {"scenarios": [{"game": game, "maps": 4, "turn_limit": 30}], "num_envs": 4}
    play, _, _ = selfplay(level=level, env=env, dets=dets, sims=24, full_search_prob=1.0)
    roots = searched_roots(play, 20, np.random.default_rng(dets))
    assert any(r["legal"].sum() > 16 and (r["visits"][r["legal"]] == 0).any() for r in roots)
    if dets > 1:
        assert any(len(r["logits"]) > 1 and not np.array_equal(r["logits"][0], r["logits"][1]) for r in roots)
    worst = 0.0
    for r in roots:
        target = gumbel_target(r["logits"], r["legal"], r["visits"], r["q"], r["value"])
        assert np.all(r["engine"][~r["legal"]] == 0) and np.all(target[~r["legal"]] == 0)
        worst = max(worst, float(np.abs(target - r["engine"]).max()))
        for sigma in ({"rescale": False}, {"c_scale": 0.01}, {"c_visit": 0.0, "c_scale": 0.0},
                      {"rescale": False, "c_scale": 0.15, "prior_temperature": 10.0}):
            soft = gumbel_target(r["logits"], r["legal"], r["visits"], r["q"], r["value"], **sigma)
            assert np.all(soft[~r["legal"]] == 0) and np.all(soft[r["legal"]] > 0)
            assert soft.sum() == pytest.approx(1.0, abs=1e-12)
        # A prior temperature is the (float64) logits divided by it, v_mix's prior weights included; T = 1 is exact.
        tempered = gumbel_target(r["logits"], r["legal"], r["visits"], r["q"], r["value"], prior_temperature=10.0)
        divided = r["logits"].astype(np.float64) / 10.0
        assert np.array_equal(tempered, gumbel_target(divided, r["legal"], r["visits"], r["q"], r["value"]))
        assert np.array_equal(target, gumbel_target(r["logits"], r["legal"], r["visits"], r["q"], r["value"],
                                                    prior_temperature=1.0))
    assert worst <= 1e-6


def test_a_gentler_sigma_is_less_concentrated_at_two_moves():
    """Both moves visited: the engine's min-max rescaled sigma puts the better one ~110 logits up at
    2048 simulations, whatever the gap in Q (here 0.03); raw Q (no rescale) or a smaller scale does not."""
    legal = np.zeros(6, dtype=bool)
    legal[[1, 4]] = True
    logits = np.array([-1e9, 0.3, -1e9, -1e9, -0.2, -1e9])
    visits = np.array([0, 1000, 0, 0, 1048, 0])
    q = np.where(legal, np.array([0.0, 0.52, 0.0, 0.0, 0.55, 0.0]), np.nan)
    engine = gumbel_target(logits, legal, visits, q, 0.5)
    assert engine[1] < 1e-40  # exp(0.3 - (-0.2 + 109.8))
    for sigma, expected in (({"rescale": False}, 1 / (1 + np.exp(0.5 - 109.8 * 0.03))),
                            ({"c_scale": 0.001}, 1 / (1 + np.exp(0.5 - 1.098)))):
        soft = gumbel_target(logits, legal, visits, q, 0.5, **sigma)
        assert soft[4] == pytest.approx(expected, rel=1e-12) and soft[1] + soft[4] == pytest.approx(1.0)
        assert np.all(soft[~legal] == 0) and entropy(soft) > entropy(engine)
    # One move visited, one not: the unvisited one is completed by v_mix, (0.5 + 1000 * 0.52) / 1001.
    mixed = (0.5 + 1000 * 0.52) / 1001
    raw = gumbel_target(logits, legal, np.array([0, 1000, 0, 0, 0, 0]), np.where(visits == 1000, 0.52, np.nan), 0.5,
                        rescale=False)
    assert np.log(raw[1] / raw[4]) == pytest.approx(0.5 + 105.0 * (0.52 - mixed), rel=1e-9)
    # A prior temperature T divides the prior's log-odds: a 9-nat prior lead (the knight over a roll, say)
    # is 0.9 at T = 10, under the same sigma.
    sharp = np.array([-1e9, 4.5, -1e9, -1e9, -4.5, -1e9])
    for T in (1.0, 10.0):
        soft = gumbel_target(sharp, legal, visits, q, 0.5, rescale=False, c_scale=0.15, prior_temperature=T)
        assert np.log(soft[1] / soft[4]) == pytest.approx(9.0 / T - (50 + 1048) * 0.15 * 0.03, rel=1e-9)


def test_a_rebuilt_target_changes_the_targets_and_nothing_else():
    """policy_target: gumbel leaves the searches and their moves as the engine target's run has them,
    so (no updates here, which read the targets) every replay row, value target and game: at the
    engine's constants its targets are the engine's (to float16), and with a gentler sigma they are
    softer distributions over the same legal actions, which target/entropy reads while
    search/target_entropy keeps reading the searches' own targets."""
    common = {"dets": 1, "full_search_prob": 1.0, "min_replay": 10**9, "iterations": 1}
    names = {"engine": {}, "same": {"policy_target": "gumbel"},
             "soft": {"policy_target": "gumbel", "target_c_scale": 0.01}}
    for name, train in names.items():
        run(variant(name, **common, **train))
    states = {name: torch.load(load_experiment(f"az/{name}").path / "trainer_state.pt", weights_only=True)
              for name in names}
    engine = states["engine"]
    has = engine["replay"]["has_policy"]
    assert int(has.sum()) > 50
    for name in ("same", "soft"):
        assert (states[name]["games"], states[name]["decisions"]) == (engine["games"], engine["decisions"])
        for key in ("obs", "has_policy", "value", "present"):
            assert torch.equal(states[name]["replay"][key], engine["replay"][key])
    targets = {name: s["replay"]["policy"][has].float() for name, s in states.items()}
    assert torch.allclose(targets["same"], targets["engine"], atol=1e-3)
    model = selfplay()[2]
    legal = engine["replay"]["obs"][has][:, model.mask_offset : model.mask_offset + model.num_actions] > 0
    assert bool((targets["soft"][~legal] == 0).all()) and bool((targets["soft"][legal] > 0).any())
    assert torch.allclose(targets["soft"].sum(1), torch.ones(len(legal)), atol=2e-3)
    metrics = {name: load_experiment(f"az/{name}").metrics() for name in names}
    searched = {name: [m["search/target_entropy"] for m in rows] for name, rows in metrics.items()}
    assert searched["same"] == searched["engine"] and searched["soft"] == searched["engine"]
    assert not any("target/entropy" in m for m in metrics["engine"])
    rebuilt = {name: [m["target/entropy"] for m in metrics[name]] for name in ("same", "soft")}
    assert rebuilt["same"] == pytest.approx(searched["engine"], abs=1e-3)
    assert all(s > e for s, e in zip(rebuilt["soft"], searched["engine"], strict=True))


@pytest.mark.runs
def test_the_rebuilt_target_on_az_2p_2048s_start():
    """az-2p-2048's search (one tracker world, Gumbel m 16; 64 simulations here) on base-2p positions of
    its start's own search play, puffer-graph2/public@e3600: the root's outputs are the network's on
    the row the trainer stores (a public row does not depend on the sampled world), gumbel_target at
    the engine's constants is the engine's target, and a gentler sigma is less concentrated at two
    legal moves, where the engine's target is nearly always one-hot."""
    from catanrl.runs import RunError, load_checkpoint

    try:
        model = load_checkpoint("puffer-graph2/public@e3600#3cd78e6dafac").model("all", "cpu")
    except (RunError, FileNotFoundError):
        pytest.skip("puffer-graph2/public@e3600 is not in the runs directory")
    plan = plan_env({"scenarios": [{"game": "base-2p"}], "num_envs": 8}, pool_seed=1)
    train = {**ALPHAZERO_KEYS, "sims": 64, "dets": 1, "gumbel_m": 16, "dirichlet_noise": False, "search_rows": 64}
    play = SelfPlay(plan, train, model=model, num_envs=8, threads=2, seed=7, device="cpu", mode="strict")
    roots = searched_roots(play, 60, np.random.default_rng(0))
    assert sum(r["legal"].sum() > 16 for r in roots) >= 8
    worst, two = 0.0, []
    for r in roots:
        logits, _ = play.evaluator(r["row"][None])
        assert np.array_equal(r["logits"], logits)
        target = gumbel_target(r["logits"], r["legal"], r["visits"], r["q"], r["value"])
        worst = max(worst, float(np.abs(target - r["engine"]).max()))
        if r["legal"].sum() == 2:
            soft = gumbel_target(r["logits"], r["legal"], r["visits"], r["q"], r["value"], rescale=False)
            two.append((r["engine"].max(), soft.max(), entropy(r["engine"]), entropy(soft)))
    assert worst <= 1e-6
    engine_top, soft_top, engine_entropy, soft_entropy = np.array(two).T
    assert len(two) >= 10 and (engine_top >= 0.99).mean() > (soft_top >= 0.99).mean()
    assert soft_entropy.mean() > engine_entropy.mean()


# ---------------------------------------------------------------- pruned visit targets


def hand_root(weights: dict[int, float], visits: dict[int, int], q: dict[int, float], size: int = 9):
    """A root over ``size`` codec indices whose legal actions are ``weights``' keys, with priors
    proportional to the weights (the logits off them are high, and never read)."""
    legal = np.zeros(size, dtype=bool)
    legal[list(weights)] = True
    logits = np.full(size, 50.0)
    logits[list(weights)] = np.log(list(weights.values()))
    prior = np.zeros(size)
    prior[list(weights)] = np.array(list(weights.values())) / sum(weights.values())
    n = np.zeros(size, dtype=np.uint32)
    n[list(visits)] = list(visits.values())
    values = np.full(size, np.nan)
    values[list(q)] = list(q.values())
    return logits, legal, n, values, prior


def test_pruned_visits_keep_what_the_value_earns_under_the_networks_prior():
    """A hand-made root of 600 visits at c_puct 0.75. Action 0, the most visited, keeps its
    visits and sets the bar S, its PUCT score. Of the others: 1 (no prior to speak of, a worse
    Q: visits only noise explains) loses all; 2 (a real prior, a slightly worse Q) is cut to
    the fewest visits at which its score is no higher than S; 4 (Q above S) keeps all; 5 is
    cut to one visit, so to none; 7 has the one visit it earns and keeps it; 8 is unvisited."""
    c = 0.75
    weights = {0: 0.88, 1: 1e-12, 2: 0.08, 4: 0.015, 5: 0.01, 7: 0.01, 8: 0.005}
    visits = {0: 500, 1: 55, 2: 30, 4: 9, 5: 5, 7: 1}
    q = {0: 0.60, 1: 0.40, 2: 0.58, 4: 0.75, 5: 0.50, 7: 0.50}
    logits, legal, n, values, prior = hand_root(weights, visits, q)
    total = sum(visits.values())
    assert total == 600

    def score(action: int, count: float) -> float:
        return q[action] + c * prior[action] * np.sqrt(total + 1.0) / (1.0 + count)

    bar = score(0, 500)
    kept = pruned_visits(logits, legal, n, values, c_puct=c)
    assert kept.dtype == np.float64 and np.all(kept[~legal] == 0) and kept[8] == 0
    assert kept[0] == 500 and kept[1] == 0
    assert q[4] >= bar and kept[4] == 9
    assert kept[2] == 28 and score(2, 28) <= bar < score(2, 27)
    assert score(5, 1) <= bar < score(5, 0) and kept[5] == 0  # cut from 5 to 1: dropped
    assert score(7, 1) <= bar < score(7, 0) and kept[7] == 1  # not cut
    assert kept.sum() == 500 + 28 + 9 + 1
    target = (kept / kept.sum()).astype(np.float16)
    assert float(target.astype(np.float64).sum()) == pytest.approx(1.0, abs=2e-3) and np.all(target[~legal] == 0)
    assert np.array_equal(kept, pruned_visits(logits[None], legal, n, values, c_puct=c))  # [1, A] logits
    # the cut scales with the exploration constant: none of it, and no visit but the value's is earned
    greedy = pruned_visits(logits, legal, n, values, c_puct=0.0)
    assert greedy[[0, 4]].tolist() == [500, 9] and greedy.sum() == 509


def test_pruned_visits_at_the_edges():
    weights = {1: 0.3, 3: 0.7, 6: 1e-9}
    # one action visited: one-hot
    logits, legal, n, values, _ = hand_root(weights, {6: 40}, {6: 0.2})
    kept = pruned_visits(logits, legal, n, values, c_puct=0.75)
    assert kept.tolist() == [0, 0, 0, 0, 0, 0, 40, 0, 0]
    # no visit at all: zeros, like the engine's target of such a root
    assert not pruned_visits(logits, legal, np.zeros(9, np.uint32), np.full(9, np.nan), c_puct=0.75).any()
    assert not pruned_visits(logits, np.zeros(9, bool), n, values, c_puct=0.75).any()
    # equal Q everywhere: the visits PUCT's own prior term gives are kept, whole
    logits, legal, n, values, _ = hand_root(weights, {1: 30, 3: 70}, {1: 0.5, 3: 0.5})
    assert pruned_visits(logits, legal, n, values, c_puct=0.75).tolist() == n.tolist()
    # a tie in visits goes to the larger prior, which keeps its visits; the other is cut against it:
    # 0.3 / (1 + n) <= 0.7 / 101 first at n = 43
    logits, legal, n, values, _ = hand_root(weights, {1: 100, 3: 100}, {1: 0.5, 3: 0.5})
    assert pruned_visits(logits, legal, n, values, c_puct=0.75).tolist() == [0, 43, 0, 100, 0, 0, 0, 0, 0]
    # equal priors too: the first in the codec's order
    logits, legal, n, values, _ = hand_root({1: 0.5, 3: 0.5}, {1: 100, 3: 100}, {1: 0.5, 3: 0.4})
    kept = pruned_visits(logits, legal, n, values, c_puct=0.75)
    assert kept[1] == 100 and 1 < kept[3] < 100
    # a non-finite logit has no prior: its visits go unless its Q reaches the bar
    logits, legal, n, values, _ = hand_root(weights, {1: 20, 3: 70, 6: 10}, {1: 0.5, 3: 0.5, 6: 0.45})
    logits[6] = -np.inf
    assert pruned_visits(logits, legal, n, values, c_puct=0.75)[6] == 0
    # several root worlds each ran their own PUCT: refused
    with pytest.raises(ValueError, match="one root world"):
        pruned_visits(np.stack([logits, logits]), legal, n, values, c_puct=0.75)


class Sharp(BatchEvaluator):
    """The network with its policy logits scaled up: a saturated prior, like a long-trained PPO
    network's, on which the root's Dirichlet noise decides most of the side visits."""

    def __call__(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits, values = super().__call__(observations)
        return logits * np.float32(4000.0), values


class Seen:
    """SelfPlay's monitor hook: keeps what each step hands it."""

    def __init__(self) -> None:
        self.steps: list[tuple] = []

    def searched(self, obs, targets, rebuilt=None, pruned=None) -> None:
        self.steps.append((obs.copy(), targets, rebuilt, pruned))


def total_variation(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(p.astype(np.float64) - q).sum(-1)


@pytest.mark.parametrize("dets", [0, 1])
def test_a_pruned_target_drops_noise_visits_and_plays_the_same_moves(dets, monkeypatch):
    """Real PUCT searches with Dirichlet noise from a saturated prior, through SelfPlay.step (true
    game, and one tracker world; several leaves a round). The same seeds under policy_target:
    engine and pruned play the same moves and file the same rows; the pruned targets are
    distributions over the visited legal actions that keep the most visited one, sit closer to
    the network's prior than the visit distributions, and are zero on some actions that the prior
    all but rules out and the noise had the search visit."""
    common = {"root": "puct", "sims": 64, "fast_sims": 8, "pending": 4, "full_search_prob": 0.5, "dets": dets,
              "temperature_moves": 6}
    import catanrl.train.alphazero as az

    monkeypatch.setattr(az, "BatchEvaluator", Sharp)
    plays, seen, boards = {}, {}, {}
    for name in ("engine", "pruned"):
        torch.manual_seed(0)  # the same weights
        play, replay, _ = selfplay(policy_target=name, **common)
        assert play.full_pool.config.dirichlet_alpha > 0 and play.full_pool.config.c_puct == 0.75
        play.monitor = seen[name] = Seen()
        rng, stats = np.random.default_rng(1), {"leaves": 0.0, "full": 0.0, "depth": 0.0, "max_depth": 0.0}
        boards[name] = []
        for step in range(30):
            play.step(0, step, rng, replay, stats)
            boards[name].append(play.buffers["observations"].copy())
        plays[name] = (play, replay)
    for a, b in zip(boards["engine"], boards["pruned"], strict=True):
        assert np.array_equal(a, b)  # the same moves: the next rows are the same
    (play, replay), other = plays["pruned"], plays["engine"][1]
    assert replay.size == other.size > 0 and play.games == plays["engine"][0].games
    for key in ("obs", "has_policy", "value", "present"):
        assert np.array_equal(getattr(replay, key)[: replay.size], getattr(other, key)[: other.size])
    assert all(r is None and p is None for _, _, r, p in seen["engine"].steps)
    obs, raw, pruned, share = (np.concatenate(part) for part in zip(*seen["pruned"].steps, strict=True))
    for (_, engine_raw, _, _), (_, pruned_raw, _, _) in zip(seen["engine"].steps, seen["pruned"].steps, strict=True):
        assert np.array_equal(engine_raw, pruned_raw)  # search/* read the same targets either way
    assert len(obs) > 30
    legal = obs[:, play.mask] > 0
    logits, _ = play.evaluator(obs)
    z = np.where(legal, logits.astype(np.float64), -np.inf)
    prior = np.exp(z - z.max(1, keepdims=True))
    prior /= prior.sum(1, keepdims=True)
    raw, pruned = raw.astype(np.float64), pruned.astype(np.float64)
    assert np.all(pruned[~legal] == 0) and np.allclose(pruned.sum(1), 1.0, atol=2e-3)
    assert np.all(pruned[raw == 0] == 0)  # nothing is added
    rows = np.arange(len(raw))
    top = np.where(raw == raw.max(1, keepdims=True), prior, -1.0).argmax(1)  # most visited, then the larger prior
    assert np.all(pruned[rows, top] >= raw[rows, top])  # keeps its visits
    assert np.all((share >= 0) & (share < 1)) and share.mean() > 0.05
    assert total_variation(pruned, prior).mean() < 0.5 * total_variation(raw, prior).mean()
    noise_only = (raw > 0) & (prior < 1e-6)
    assert noise_only.any() and (pruned[noise_only] == 0).mean() > 0.5
    assert np.all((pruned[noise_only] == 0) | (raw[noise_only] > 0))


def test_a_pruned_target_changes_the_targets_and_nothing_else(monkeypatch):
    """policy_target: pruned leaves the searches and their moves as the engine target's run has them
    (no updates here, which read the targets): every replay row, value target and game. Its targets
    are distributions inside the visit distributions' supports; target/* read them and search/* keep
    reading the searches' own."""
    import catanrl.train.alphazero as az

    monkeypatch.setattr(az, "BatchEvaluator", Sharp)
    common = {"root": "puct", "sims": 32, "pending": 4, "full_search_prob": 1.0, "min_replay": 10**9, "iterations": 1}
    for name in ("engine", "pruned"):
        run(variant(name, policy_target=name, **common))
    states = {name: torch.load(load_experiment(f"az/{name}").path / "trainer_state.pt", weights_only=True)
              for name in ("engine", "pruned")}
    engine, pruned = states["engine"], states["pruned"]
    has = engine["replay"]["has_policy"]
    assert int(has.sum()) > 50 and (pruned["games"], pruned["decisions"]) == (engine["games"], engine["decisions"])
    for key in ("obs", "has_policy", "value", "present"):
        assert torch.equal(pruned["replay"][key], engine["replay"][key])
    visits, kept = engine["replay"]["policy"][has].float(), pruned["replay"]["policy"][has].float()
    assert torch.allclose(kept.sum(1), torch.ones(len(kept)), atol=2e-3) and bool((kept[visits == 0] == 0).all())
    assert not torch.equal(kept, visits)
    metrics = {name: load_experiment(f"az/{name}").metrics() for name in ("engine", "pruned")}
    assert [m["search/target_entropy"] for m in metrics["pruned"]] == [
        m["search/target_entropy"] for m in metrics["engine"]]
    assert [m["search/kl_target_net"] for m in metrics["pruned"]] == [
        m["search/kl_target_net"] for m in metrics["engine"]]
    assert not any(k.startswith("target/") for m in metrics["engine"] for k in m)
    for m in metrics["pruned"]:
        assert 0.0 < m["target/pruned"] < 1.0 and m["target/entropy"] != m["search/target_entropy"]
        assert 0.0 <= m["target/overrule"] <= 1.0 and m["target/kl_net"] >= 0.0
