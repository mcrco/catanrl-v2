"""The recurrent graph network (class ``lstm``): build and stateless forward,
state threading in training against the rollout recursion, checkpoints,
the native POLICY agent's memory streams (invariant rows, equal to the
reference stepper), the refusals, and a three-epoch CPU training."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from catanatron.registry import SpecError
from eval_helpers import agent, make_suite, schedule, tiny_model
from reference_stepper import play_reference
from run_helpers import train
from test_models import BASE_2P, WIDE_4P, _names, encode, model_for, played_games
from train_configs import tiny_ppo

from catanrl.agents import resolve_agent
from catanrl.checkpoint import Checkpoint, save
from catanrl.envs.encoding import dims_for
from catanrl.eval.harness import play_schedule
from catanrl.train import puffer
from catanrl.train.puffer_policy import PufferPolicy


def lstm_checkpoint(tmp_path, model=None):
    model = model if model is not None else tiny_model(["base-2p"], preset="lstm-tiny", seed=5)
    config = {"env": {"scenarios": [{"game": "base-2p"}]}, "model": model.spec}
    return save(tmp_path / "e000000.pt", model, run=None, epoch=0, agent_steps=0, config=config, code={})


# ---------------------------------------------------------------- the model


def test_build_and_stateless_forward_is_the_zero_state():
    model = model_for("lstm-tiny")
    assert model.recurrent and model.dims_agnostic and model.hidden_size == 8
    assert model.parts == {"actor": ["net.trunk.", "net.cell.", "net.inject.", "net.policy."],
                           "critic": ["net.value."]}
    obs = encode(played_games(), BASE_2P)
    logits, values = model(obs)
    mask = obs[:, model.mask_offset : model.mask_offset + model.num_actions]
    assert torch.all((logits > -1e8) == (mask > 0))
    assert torch.all((values > 0) & (values < 1))
    h, c = model.initial_state(obs.shape[0])
    assert torch.all(h == 0) and torch.all(c == 0)
    step_logits, value_logits, present, h1, c1 = model.step(obs, h, c)
    assert torch.equal(step_logits, logits)
    assert torch.allclose(torch.sigmoid(value_logits) * present, values)
    assert h1.shape == (obs.shape[0], 8) and not torch.all(h1 == 0)
    # Memory changes the outputs: the same positions from a stepped state.
    again, _, _, _, _ = model.step(obs, h1, c1)
    legal = logits > -1e8
    assert torch.equal(legal, again > -1e8) and not torch.allclose(logits[legal], again[legal])


def test_lstm_network_is_dims_agnostic():
    small = model_for("lstm-tiny", BASE_2P, seed=3)
    large = model_for("lstm-tiny", WIDE_4P, seed=99)
    large.load_state_dict(small.state_dict())
    played = played_games(count=8)
    obs_s, obs_l = encode(played, BASE_2P), encode(played, WIDE_4P)
    h, c = small.initial_state(len(played))
    logits_s, values_s, _, h_s, c_s = small.step(obs_s, h, c)
    logits_l, values_l, _, h_l, c_l = large.step(obs_l, h, c)
    names_l = {name: i for i, name in enumerate(_names(WIDE_4P))}
    columns = [names_l[name] for name in _names(BASE_2P)]
    assert torch.allclose(logits_s, logits_l[:, columns], atol=1e-4)
    assert torch.allclose(values_s, values_l[:, :2], atol=1e-5)
    assert torch.allclose(h_s, h_l, atol=1e-5) and torch.allclose(c_s, c_l, atol=1e-5)


def test_lstm_preset_refuses_a_separate_critic_level():
    from catanrl.envs.encoding import action_size, row_layout
    from catanrl.models import build_model

    layout = row_layout(BASE_2P, "private", "full")
    with pytest.raises(ValueError, match="critic level"):
        build_model({"preset": "lstm-tiny"}, dims=BASE_2P, levels={"level": "private", "critic_level": "full"},
                    num_actions=action_size(BASE_2P), layout=layout)


# ---------------------------------------------------------------- state threading


def _segments(model, segments: int, horizon: int, seed: int = 0):
    """Rows [segments, horizon, D] of played positions where some rows wait
    (mask = the no-op only), plus terminals [segments, horizon] marking new games."""
    rng = np.random.default_rng(seed)
    played = played_games(count=segments * horizon, plies=120, seed=seed)
    obs = encode(played, model.dims).reshape(segments, horizon, -1).clone()
    acting = torch.from_numpy(rng.random((segments, horizon)) < 0.6)
    acting[:, 0] = True
    offset, noop = model.mask_offset, model.noop
    for s in range(segments):
        for t in range(horizon):
            if not acting[s, t]:
                obs[s, t, offset : offset + model.num_actions] = 0.0
                obs[s, t, offset + noop] = 1.0
    terminals = torch.from_numpy(rng.random((segments, horizon)) < 0.2).float()
    terminals[:, 0] = 0.0
    assert torch.equal(model.acting(obs.reshape(-1, obs.shape[-1])).reshape(segments, horizon), acting)
    return obs, terminals


def _rollout(policy: PufferPolicy, obs: torch.Tensor, terminals: torch.Tensor) -> torch.Tensor:
    """forward_eval step by step: the logits of every step."""
    out = []
    with torch.no_grad():
        for t in range(obs.shape[1]):
            logits, _ = policy.forward_eval(obs[:, t], {"done": terminals[:, t].bool()})
            out.append(logits)
    return torch.stack(out, dim=1)


def test_training_forward_equals_the_rollout_recursion():
    """The training forward (segments x horizon, stored initial states, resets at
    terminals, cell steps on acting rows) gives the rollout's logits and memory."""
    torch.manual_seed(1)
    model = model_for("lstm-tiny")
    policy = PufferPolicy(model, num_rows=4)
    obs, terminals = _segments(model, segments=4, horizon=6)
    # The first epoch: every row starts from zero.
    policy.begin_rollout()
    expected = _rollout(policy, obs, terminals)
    after_rollout = tuple(m.clone() for m in policy._memory)
    logits, batch = policy(obs, {"idx": torch.arange(4), "terminals": terminals})
    assert torch.allclose(logits.reshape(4, 6, -1), expected, atol=1e-5)
    assert batch.index.tolist() == torch.nonzero(model.acting(obs.reshape(24, -1))).squeeze(1).tolist()
    # A later epoch starts from the stored memory, and a minibatch is a subset of segments.
    policy.begin_rollout()
    assert all(torch.equal(a, b) for a, b in zip(policy._memory_start, after_rollout))
    obs2, terminals2 = _segments(model, segments=4, horizon=6, seed=7)
    expected2 = _rollout(policy, obs2, terminals2)
    idx = torch.tensor([3, 1])
    logits2, _ = policy(obs2[idx], {"idx": idx, "terminals": terminals2[idx]})
    assert torch.allclose(logits2.reshape(2, 6, -1), expected2[idx], atol=1e-5)
    # A reset where a game starts: that step's memory input is the zero state.
    h0 = policy._memory_start[0][idx]
    assert not torch.all(h0 == 0)
    with pytest.raises(ValueError):
        policy(obs2.reshape(24, -1), None)
    with pytest.raises(ValueError, match="idx and terminals"):
        policy(obs2, {"action": None, "lstm_h": None, "lstm_c": None})


def test_rollout_memory_resets_on_done_and_steps_on_acting_rows():
    torch.manual_seed(2)
    model = model_for("lstm-tiny")
    policy = PufferPolicy(model, num_rows=4)
    obs, _ = _segments(model, segments=4, horizon=2, seed=3)
    with torch.no_grad():
        policy.forward_eval(obs[:, 0], {"done": torch.zeros(4, dtype=torch.bool)})
        h_after_first = policy._memory[0].clone()
        assert torch.all(h_after_first != 0)
        acting = model.acting(obs[:, 1])
        policy.forward_eval(obs[:, 1], {"done": torch.tensor([True, False, False, False])})
        h = policy._memory[0]
    assert not acting[0] or not torch.equal(h[0], h_after_first[0])
    waiting = torch.nonzero(~acting).squeeze(1).tolist()
    for row in waiting:
        if row != 0:
            assert torch.equal(h[row], h_after_first[row])  # waiting rows keep their memory
        else:
            assert torch.all(h[row] == 0)  # unless their game ended


# ---------------------------------------------------------------- checkpoints and agents


def test_checkpoint_round_trip_and_recurrent_flag(tmp_path):
    model = tiny_model(["base-2p"], preset="lstm-tiny", seed=5).eval()
    ref = lstm_checkpoint(tmp_path, model)
    ckpt = Checkpoint.load(ref.path)
    assert ckpt.recurrent and ckpt.dims_agnostic
    loaded = ckpt.model("all")
    obs = encode(played_games(count=4), model.dims)
    h, c = model.initial_state(4)
    with torch.no_grad():
        for a, b in zip(model.step(obs, h, c), loaded.step(obs, h, c)):
            assert torch.equal(a, b)
    big = ckpt.model(dims=dims_for(["base-4p"]))
    assert big.dims == dims_for(["base-4p"]) and big.recurrent


def test_mcts_belief_and_the_bridges_refuse_a_recurrent_network(tmp_path):
    model = tiny_model(["base-2p"], preset="lstm-tiny", seed=5)
    for key, params in (("MCTS", {"sims": 4}), ("BELIEF", {"worlds": 2})):
        with pytest.raises(ValueError, match="recurrent"):
            agent(key, model, **params)
    path = lstm_checkpoint(tmp_path, model).path
    for spec in (f"MCTS:ckpt={path}:sims=4", f"BELIEF:ckpt={path}:worlds=2"):
        with pytest.raises(SpecError, match="recurrent"):
            resolve_agent(spec)
    for harness in ("catanatron", "server"):
        with pytest.raises(SpecError, match="native harness"):
            resolve_agent(f"POLICY:ckpt={path}", harness=harness)
    resolved = resolve_agent(f"POLICY:ckpt={path}")
    assert resolved.fair and resolved.native(device="cpu", infer_rows=4).needs_memory
    from catanatron.models.player import Color

    with pytest.raises(RuntimeError, match="decision history"):
        resolved.player(Color.RED)._agent()
    from catanrl.bridge.agent import BridgeError, ViewAgent

    with pytest.raises(BridgeError, match="decision history"):
        ViewAgent(resolved)


def test_policy_agent_decides_from_its_memory_streams():
    model = tiny_model(["base-2p"], preset="lstm-tiny", seed=5)
    policy = agent("POLICY", model, infer_rows=4)
    played = played_games(count=3, plies=150, seed=4)
    rows = np.stack([policy.encode(g, g.current_player_index) for g in played])
    masks = rows[:, -policy.action_space.size:]
    seeds = np.zeros(3, dtype=np.uint64)
    with pytest.raises(ValueError, match="streams"):
        policy.act(rows, masks, seeds)
    streams, fresh = np.array([0, 1, 2]), np.array([True, True, True])
    first = policy.act(rows, masks, seeds, streams=streams, first=fresh)
    assert np.array_equal(first, policy.act(rows, masks, seeds, streams=streams, first=fresh))
    zero = model.initial_state(3)
    with torch.no_grad():
        logits = model.step(torch.from_numpy(rows), *zero)[0].numpy()
    assert np.array_equal(first, np.where(masks > 0, logits, -np.inf).argmax(1))
    # The memory after that decision is kept per stream and used at the next decision.
    second = policy.act(rows, masks, seeds, streams=streams, first=~fresh)
    h, c = (np.stack([policy._memory[s][i] for s in (0, 1, 2)]) for i in (0, 1))
    with torch.no_grad():
        _, _, _, h1, c1 = model.step(torch.from_numpy(rows), *zero)
        logits = model.step(torch.from_numpy(rows), h1, c1)[0].numpy()
    assert np.array_equal(second, np.where(masks > 0, logits, -np.inf).argmax(1))
    with pytest.raises(ValueError, match="no memory"):
        policy.act(rows, masks, seeds, streams=np.array([5, 6, 7]), first=~fresh)
    with pytest.raises(ValueError, match="once per batch"):
        policy.act(rows, masks, seeds, streams=np.array([0, 0, 1]), first=fresh)


SUITE = make_suite(({"game": "base-2p"}, {"game": "base-3p"}), turn_limit=160)
GAMES = schedule(SUITE, 3, boards=3)  # 3 x 2p rotations + 3 x 3p rotations = 15 games
MODEL = tiny_model(["base-2p", "base-3p"], preset="lstm-tiny", seed=11)


def _lineup(**params):
    return [agent("POLICY", MODEL, infer_rows=8, **params), agent("F"), agent("R")]


def _play(players, games=GAMES, **kwargs):
    rows = []
    play_schedule(players, games, suite=SUITE, on_game=rows.append, **kwargs)
    return {row["game_id"]: row for row in rows}


@pytest.mark.parametrize("sample", [False, True])
def test_eval_rows_are_invariant_and_match_the_reference(sample):
    """Test 21 for a recurrent POLICY: rows do not depend on num_envs, threads,
    workers or order, and equal the Python reference stepper's."""
    params = {"sample": True} if sample else {}
    reference = _play(_lineup(**params), num_envs=16, threads=1)
    assert sorted(reference) == sorted(g.game_id for g in GAMES)
    assert any(row["decisions"][0] > 1 for row in reference.values())  # memory was in play
    for num_envs, threads in ((1, 1), (5, 4)):
        assert _play(_lineup(**params), num_envs=num_envs, threads=threads) == reference
    half = len(GAMES) // 2
    split = {**_play(_lineup(**params), GAMES[:half], num_envs=3), **_play(_lineup(**params), GAMES[half:], num_envs=7)}
    assert split == reference
    assert _play(_lineup(**params), list(reversed(GAMES)), num_envs=4) == reference
    stepped = {row["game_id"]: row for row in play_reference(_lineup(**params), GAMES, suite=SUITE)}
    assert stepped == reference


# ---------------------------------------------------------------- training


def test_ppo_trains_three_epochs_through_the_recurrent_path(monkeypatch):
    trainers = []
    original_init = puffer.CatanPuffeRL.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        trainers.append(self)

    monkeypatch.setattr(puffer.CatanPuffeRL, "__init__", init)
    config = tiny_ppo(seed=3, epochs=3)
    config["model"] = {"preset": "lstm-tiny"}
    writer = train(puffer.train_ppo, config, "lstm")
    assert writer.status == "finished"
    assert [ref.epoch for ref in writer.saved] == [0, 2, 3]
    assert len(writer.rows) == 3 and all(np.isfinite(row["losses/policy_loss"]) for row in writer.rows)
    trainer = trainers[0]
    policy = trainer.uncompiled_policy
    assert trainer.config["use_rnn"] is True and policy.recurrent and policy.hidden_size == 8
    assert trainer.observations.dim() == 3  # segments x horizon x row
    assert policy._memory[0].shape == (trainer.vecenv.num_agents, 8)
    # The memory carried across epochs: the next epoch's segments start from it, not from zero.
    assert policy._memory_start is not None and not torch.all(policy._memory_start[0] == 0)
    # The checkpoint is the stateless network: it loads and self-tests without the memory.
    loaded = writer.checkpoint("latest").model("all")
    assert loaded.recurrent and set(loaded.state_dict()) == set(trainer.model.state_dict())


def test_tiny_experiment_end_to_end_with_lstm(tmp_path, monkeypatch):
    """tests/test_e2e_tiny.py's row with the lstm-tiny preset: three PPO epochs,
    select evals of the recurrent POLICY at every checkpoint, best, final."""
    import catanrl
    from catanrl.config import load_plan
    from catanrl.eval import results
    from catanrl.runner import final_evals, run_variant

    monkeypatch.setenv("CATANRL_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.chdir(tmp_path)
    v = load_plan("tiny", variant="2p", overrides=["model.preset=lstm-tiny"]).variants[0]
    assert (v.config["model"]["preset"], v.config["model"]["class"]) == ("lstm-tiny", "lstm")
    assert run_variant(v, argv=["test"]) == "finished"
    run = catanrl.load_experiment("tiny/2p")
    assert [c.epoch for c in run.checkpoints()] == [0, 1, 2, 3]
    best = run.aliases()["best"]
    record = results.EvalRecord.load(next(run.path.glob(f"evals/*/{best['eval_id']}.json")))
    assert record.key["suite"]["purpose"] == "select" and record.key["lineup"][0]["checkpoint"]["model"]["class"] == "lstm"
    net = run.build_policy("best")
    assert net.recurrent and net.catanrl_ref == best["ref"]
    native = resolve_agent(f"POLICY:tiny/2p@best").native(device="cpu", infer_rows=4)
    assert native.needs_memory
    finals = final_evals(run, v)
    assert len(finals) == 1 and finals[0].data["refs"][0] == best["ref"]
