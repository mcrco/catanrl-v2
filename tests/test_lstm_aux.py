"""The auxiliary hidden-information head on the recurrent network (``model.args.aux``
on the lstm class, ``train.aux_coef``): the head's shape and its silence on the other
parts' initial weights, fairness (neither a head nor the memory reads the critic
block), the memory's reach into the predictions and the aux loss's reach into the
cell through the training forward, the knob at 0 (the plain recurrent forward,
bitwise) and above 0 (losses/aux and the per-feature errors logged), and a
checkpoint with the head self-testing and playing through the harness and
``catanrl eval`` without the block. The rollout and inference never run the head
(a call counter pins it) and the batch evaluator's ``aux`` mode refuses the
recurrent network. The graph head's own tests are tests/test_aux_head.py, whose
helpers these reuse."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from run_helpers import train
from test_aux_head import declared_positions, rows_with_block
from test_models import BASE_2P, BASE_4P
from train_configs import tiny_ppo

from catanrl.agents import resolve_agent
from catanrl.checkpoint import Checkpoint, ModelDrift
from catanrl.envs.encoding import HIDDEN_FEATURES, action_size, row_layout
from catanrl.models import LstmModel, build_model
from catanrl.models.graph import GraphAuxHead
from catanrl.search.evaluator import BatchEvaluator
from catanrl.train import puffer
from catanrl.train.puffer_policy import PufferPolicy

AUX = {"preset": "lstm-tiny", "args": {"aux": True}}
PLAIN = {"preset": "lstm-tiny"}
GRAPH_AUX = {"preset": "graph-tiny", "args": {"shared": True, "aux": True}}
PARTS = {"actor": ["net.trunk.", "net.cell.", "net.inject.", "net.policy."], "critic": ["net.value."],
         "aux": ["net.aux."]}


def lstm_model(spec=AUX, dims=BASE_2P, seed=0, critic="full"):
    """A recurrent network at (private, critic); ``critic=None`` builds it without the block."""
    torch.manual_seed(seed)
    return build_model(spec, dims=dims, levels={"level": "private", "critic_level": critic},
                       num_actions=action_size(dims), layout=row_layout(dims, "private", critic)).eval()


# ---------------------------------------------------------------- the head


def test_aux_head_on_the_memory_predicts_the_other_seats():
    """The zero-state predictions are decode_with_aux after one cell step from zero (what
    the checkpoint self-test pins), beside the unchanged policy and value outputs; the
    targets and the present other seats are the graph head's."""
    model = lstm_model(dims=BASE_4P)
    assert isinstance(model, LstmModel) and model.recurrent and model.aux and model.has_critic_block
    assert not model.reads_critic_block and model.parts == PARTS
    positions = declared_positions("base-2p", 3) + declared_positions("base-4p", 3)
    obs = rows_with_block(positions, BASE_4P)
    with torch.no_grad():
        predictions = model.aux_predictions(obs)
        targets, others = model.aux_targets(obs)
        h0, c0 = model.initial_state(6)
        emb, rows = model.embed(obs)
        h1, _ = model.remember(emb.pooled, h0, c0)
        logits, values, present, again = model.decode_with_aux(emb, rows, h1)
        step_logits, step_values, step_present, _, _ = model.step(obs, h0, c0)
    assert predictions.shape == targets.shape == (6, 3, len(HIDDEN_FEATURES)) and others.shape == (6, 3)
    assert torch.equal(predictions, again)
    assert torch.equal(logits, step_logits) and torch.equal(values, step_values)
    assert torch.equal(present, step_present) and torch.equal(others, present[:, 1:])
    assert others.tolist() == [[1.0, 0.0, 0.0]] * 3 + [[1.0, 1.0, 1.0]] * 3        # a 2p game has one other seat
    graph = build_model(GRAPH_AUX, dims=BASE_4P, levels={"level": "private", "critic_level": "full"},
                        num_actions=action_size(BASE_4P), layout=row_layout(BASE_4P, "private", "full"))
    graph_targets, graph_others = graph.aux_targets(obs)
    assert torch.equal(targets, graph_targets) and torch.equal(others, graph_others)
    assert bool((targets[3:] > 0).any(dim=(0, 1)).all())                         # every feature is exercised in 4p
    assert bool((targets[:3, 1:] == 0).all())                                     # absent seats are all zero
    # The evaluator's aux mode (PREDICT's predictor) runs the head on single positions, which
    # carry no memory: refused, not the zero-state predictions in the policy logits' place.
    with pytest.raises(ValueError, match="recurrent"):
        BatchEvaluator(model, device="cpu", infer_rows=4, heads="aux")


def test_aux_head_leaves_the_other_parts_initial_weights_unchanged():
    """Built last: for one seed the trunk, cell, injections, policy and value weights
    equal the plain lstm network's."""
    plain = lstm_model(PLAIN, seed=3, critic=None)
    with_aux = lstm_model(AUX, seed=3)
    state, other = plain.state_dict(), with_aux.state_dict()
    assert set(state) < set(other) and all(k.startswith("net.aux.") for k in set(other) - set(state))
    assert all(torch.equal(state[k], other[k]) for k in state)


def test_no_head_and_not_the_memory_reads_the_critic_block():
    """Policy, value and the stepped memory are identical whatever the block holds
    (fairness); the aux targets change with it; a model built without the block (what
    the eval harness plays) gives the same outputs and predictions."""
    model = lstm_model()
    positions = declared_positions("base-2p", 4)
    rows = rows_with_block(positions, BASE_2P)
    swapped = rows_with_block(positions, BASE_2P, critic="private")               # the block as the private players
    h0, c0 = model.initial_state(4)
    with torch.no_grad():
        logits, values, _present, h, c = model.step(rows, h0, c0)
        logits2, values2, _present2, h2, c2 = model.step(swapped, h0, c0)
        assert all(torch.equal(a, b) for a, b in ((logits, logits2), (values, values2), (h, h2), (c, c2)))
        predictions = model.aux_predictions(rows)
        assert torch.equal(predictions, model.aux_predictions(swapped))
        assert not torch.equal(model.aux_targets(rows)[0], model.aux_targets(swapped)[0])
        plain = lstm_model(AUX, critic=None)
        plain.load_state_dict(model.state_dict())
        assert plain.aux and not plain.has_critic_block and not plain.reads_critic_block
        private = rows[:, : plain.obs_size]
        plain_logits, _, _, plain_h, _ = plain.step(private, h0, c0)
        assert torch.equal(plain_logits, logits) and torch.equal(plain_h, h)
        assert torch.equal(plain.aux_predictions(private), predictions)
        with pytest.raises(ValueError, match="critic block"):
            plain.aux_targets(private)
    # Without the head the block is still refused, and the head does not train without the block.
    with pytest.raises(ValueError, match="critic level"):
        lstm_model(PLAIN)
    with pytest.raises(ValueError, match="aux head"):
        PufferPolicy(lstm_model(PLAIN, critic=None), num_rows=2, aux_coef=0.1)
    with pytest.raises(ValueError, match="critic block"):
        PufferPolicy(plain, num_rows=2, aux_coef=0.1)


# ---------------------------------------------------------------- the memory and the training forward


def _segments_with_block(model, segments: int, horizon: int, seed: int = 0):
    """Rows [segments, horizon, D] of declared positions with the critic block, where some
    rows wait (the mask the no-op only; the first two steps act, so a memory reaches the
    cell's recurrent weights), plus terminals [segments, horizon] with one game boundary."""
    rng = np.random.default_rng(seed)
    obs = rows_with_block(declared_positions("base-2p", segments * horizon, seed=seed), model.dims)
    obs = obs.reshape(segments, horizon, -1).clone()
    acting = torch.from_numpy(rng.random((segments, horizon)) < 0.6)
    acting[:, :2] = True
    offset, noop = model.mask_offset, model.noop
    for s in range(segments):
        for t in range(horizon):
            if not acting[s, t]:
                obs[s, t, offset : offset + model.num_actions] = 0.0
                obs[s, t, offset + noop] = 1.0
    terminals = torch.zeros(segments, horizon)
    terminals[0, horizon - 1] = 1.0
    assert torch.equal(model.acting(obs.reshape(-1, obs.shape[-1])).reshape(segments, horizon), acting)
    return obs, terminals


def test_aux_predictions_read_the_memory_and_the_aux_loss_trains_the_cell(monkeypatch):
    """The predictions change with the memory; the rollout, a step and the stateless
    forward never run the head and the training forward runs it once; through that
    forward the aux loss's gradient reaches the cell's recurrent weights and the trunk,
    and none of the policy or value head; at aux_coef 0 the forward is the plain one,
    bitwise."""
    model = lstm_model(seed=1)
    obs = rows_with_block(declared_positions("base-2p", 4, seed=3), BASE_2P)
    with torch.no_grad():
        emb, rows = model.embed(obs)
        h0, c0 = model.initial_state(4)
        h1, c1 = model.remember(emb.pooled, h0, c0)
        h2, _ = model.remember(emb.pooled, h1, c1)
        first = model.decode_with_aux(emb, rows, h1)[3]
        second = model.decode_with_aux(emb, rows, h2)[3]
    assert first.shape == (4, 1, len(HIDDEN_FEATURES)) and not torch.allclose(first, second)
    calls: list[int] = []
    head_forward = GraphAuxHead.forward

    def counted(head, emb):
        calls.append(emb.pooled.shape[0])
        return head_forward(head, emb)

    monkeypatch.setattr(GraphAuxHead, "forward", counted)
    model.train()
    policy = PufferPolicy(model, num_rows=4, aux_coef=1.0)
    segments, terminals = _segments_with_block(model, segments=4, horizon=3, seed=4)
    with torch.no_grad():
        for t in range(3):                                     # the rollout, step by step
            policy.forward_eval(segments[:, t], {"done": terminals[:, t].bool()})
        model.step(obs, h0, c0)
        model(obs)
    assert calls == []                                         # the rollout and inference never run the head
    state = {"idx": torch.arange(4), "terminals": terminals}
    _logits, batch = policy(segments, state)
    assert calls == [batch.index.numel()]                      # the training forward: once, on the acting rows
    predictions, targets, others = batch.aux
    acting = batch.index.numel()
    assert predictions.shape == targets.shape == (acting, 1, len(HIDDEN_FEATURES)) and others.shape == (acting, 1)
    assert torch.equal(targets, model.aux_targets(segments.reshape(12, -1)[batch.index])[0])
    weighted, terms = policy.aux_loss(batch)
    assert float(terms["aux"]) > 0
    weighted.backward()
    cell = model.net.cell
    assert float(cell.weight_hh.grad.abs().sum()) > 0 and float(cell.weight_ih.grad.abs().sum()) > 0
    assert sum(float(p.grad.abs().sum()) for p in model.net.aux.parameters()) > 0
    assert sum(float(p.grad.abs().sum()) for p in model.net.trunk.parameters()) > 0
    assert all(p.grad is None for head in (model.net.policy, model.net.value) for p in head.parameters())
    model.zero_grad()
    off = PufferPolicy(model, num_rows=4, aux_coef=0.0)
    with torch.no_grad():
        logits_off, batch_off = off(segments, state)
        logits_on, batch_on = policy(segments, state)
    assert batch_off.aux is None and batch_on.aux is not None
    assert torch.equal(logits_off, logits_on) and torch.equal(batch_off.logits, batch_on.logits)
    assert torch.equal(batch_off.index, batch_on.index) and torch.equal(batch_off.present, batch_on.present)


# ---------------------------------------------------------------- training and the checkpoint


def lstm_aux_config(aux_coef: float, **train) -> dict:
    cfg = tiny_ppo(epochs=3, **train)
    cfg["obs"] = {"level": "private", "critic_level": "full"}
    cfg["model"] = AUX
    cfg["train"]["aux_coef"] = aux_coef
    return cfg


def test_ppo_trains_the_head_through_the_recurrent_path_and_the_checkpoint_plays():
    """Three tiny epochs: losses/aux and aux/mse_<feature>, aux/var_<feature> are logged;
    the checkpoint carries the head as its aux part, self-tests it under ``all`` and
    ``aux`` (the actor's self-test does not pin it), and plays as a fair POLICY row."""
    writer = train(puffer.train_ppo, lstm_aux_config(1.0), "lstm-aux")
    assert writer.status == "finished"
    rows = writer.rows
    assert len(rows) == 3 and all(row["losses/aux"] > 0 and np.isfinite(row["losses/policy_loss"]) for row in rows)
    for name in HIDDEN_FEATURES:
        assert f"aux/mse_{name}" in rows[-1] and f"aux/var_{name}" in rows[-1]
    mse = [rows[-1][f"aux/mse_{name}"] for name in HIDDEN_FEATURES]
    assert rows[-1]["losses/aux"] == pytest.approx(sum(mse) / len(mse), rel=0.05)   # the epoch's mean over features
    ref = writer.saved[-1]
    ckpt = Checkpoint.load(ref.path)
    assert ckpt.recurrent and ckpt.header["parts"] == PARTS and ckpt.header["obs"]["critic_level"] == "full"
    assert "aux" in ckpt.header["self_test"]
    loaded = ckpt.model("all")
    assert loaded.aux and loaded.recurrent and loaded.has_critic_block
    head = Checkpoint.load(ref.path).model("aux")
    assert all(torch.equal(head.state_dict()[k], v) for k, v in loaded.state_dict().items())
    original = LstmModel.aux_predictions
    LstmModel.aux_predictions = lambda self, obs: original(self, obs) + 0.01
    try:
        Checkpoint.load(ref.path).model("actor")           # the actor's self-test does not pin the head
        with pytest.raises(ModelDrift, match="aux"):
            Checkpoint.load(ref.path).model("all")
        with pytest.raises(ModelDrift, match="aux"):
            Checkpoint.load(ref.path).model("aux")
    finally:
        LstmModel.aux_predictions = original
    _plays_without_the_block(ref)


def _plays_without_the_block(ref):
    from eval_helpers import make_suite, schedule

    from catanrl.cli import main
    from catanrl.eval.harness import play_schedule

    agent = resolve_agent(f"POLICY:ckpt={ref.path}").native(device="cpu", infer_rows=4)
    assert agent.needs_memory and agent.model.aux and not agent.model.has_critic_block
    assert agent.encoder.size == agent.model.obs_size == row_layout(agent.dims, "private")["action_mask"][0] + \
        row_layout(agent.dims, "private")["action_mask"][1][0]
    suite = make_suite(({"game": "base-2p"},), turn_limit=60)
    rows = []
    play_schedule([agent, resolve_agent("F").native(device="cpu", infer_rows=4)], schedule(suite, 2, boards=2),
                  suite=suite, on_game=rows.append, threads=1, num_envs=2, device="cpu", infer_rows=4)
    assert len(rows) == 4 and all(r["winner"] in (0, 1, None) for r in rows)
    assert any(row["decisions"][0] > 1 for row in rows)                            # memory was in play
    assert main(["eval", f"POLICY:ckpt={ref.path}", "F", "--suite", "heldout-2p", "--boards", "1", "--threads", "1",
                 "--num-envs", "2"]) == 0
