"""The dice input (``obs.dice_state``) in the AlphaZero trainer: its self-play rows, search
leaves and replay carry the dice state; a network warm-started from one without the input
self-plays exactly as that network (the same searches, moves, targets and replay); and a run
logs ``dice/*`` with a probe batch that survives a resume."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from dice_features import DiceColumns

from catanrl import canon
from catanrl import checkpoint as ck
from catanrl.config import Variant, config_hash, resolve
from catanrl.envs.encoding import make_encoder
from catanrl.envs.puffer import plan_env
from catanrl.runs import RunWriter, load_experiment
from catanrl.train.alphazero import ALPHAZERO_KEYS, ReplayBuffer, SelfPlay, train_alphazero
from catanrl.train.dice_probe import STATE_KEY, FairDice
from catanrl.train.puffer import build_policy_model


@pytest.fixture(autouse=True)
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CATANRL_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.chdir(tmp_path)


TRAIN = {**ALPHAZERO_KEYS, "iterations": 2, "steps_per_iteration": 40, "sims": 6, "fast_sims": 2,
         "full_search_prob": 0.5, "min_replay": 1, "updates_per_iteration": 3, "minibatch_size": 16,
         "replay_capacity": 2000, "checkpoint_interval": 1, "search_rows": 16}
ENV = {"scenarios": [{"game": "base-2p-fr", "maps": 4, "turn_limit": 10}], "num_envs": 4}  # balanced dice
STATS = {"leaves": 0.0, "full": 0.0, "depth": 0.0, "max_depth": 0.0}


def network(dice: bool, seed: int = 0):
    torch.manual_seed(seed)
    cfg = {"model": {"preset": "graph-tiny", "args": {"shared": True}},
           "obs": {"level": "full", **({"dice_state": True} if dice else {})}}
    return build_policy_model(cfg, plan_env(ENV, pool_seed=1).dims).eval()


def selfplay(model, groups: int = 1, **train):
    plan = plan_env(ENV, pool_seed=1)
    play = SelfPlay(plan, {**TRAIN, **train}, model=model, num_envs=4, threads=2, seed=5, device="cpu", mode="strict",
                    groups=groups)
    return play, ReplayBuffer(5000, play.vec.observation_size, play.vec.action_size, play.players)


class Spy:
    """Wraps the self-play evaluator: records every leaf batch."""

    def __init__(self, evaluator) -> None:
        self.inner, self.rows = evaluator, []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __call__(self, rows):
        self.rows.append(np.array(rows))
        return self.inner(rows)


@pytest.mark.parametrize("dets", [0, 1])
def test_self_play_rows_leaves_and_replay_carry_the_dice_state(dets):
    model = network(True)
    plain_size = network(False).obs_size
    play, replay = selfplay(model, dets=dets)
    assert play.vec.dice_state and play.vec.observation_size == model.obs_size == plain_size + 127
    assert all(pool.observation_size == model.obs_size for pool in play.pools)
    encoder = make_encoder(play.dims, "full", True)
    columns = DiceColumns(encoder)
    spy = play.evaluator = Spy(play.evaluator)
    rng = np.random.default_rng(0)
    roots, stored = set(), set()
    for step in range(120):
        b = play.buffers
        for row in np.flatnonzero(b["acting"]):
            env, seat = int(b["env_index"][row]), int(b["seat"][row])
            game = play.vec.game(env)
            assert np.array_equal(b["observations"][row], encoder.encode(game, seat))  # the VecEnv's row
            columns.check(game, seat, b["observations"][row])
            roots.add(columns.all(b["observations"][row]).tobytes())
            stored.add(b["observations"][row].astype(np.float16).tobytes())
        play.step(0, step, rng, replay, dict(STATS))
    # the leaves: each row reads one dice state, a tile's features its own number's
    leaves = np.concatenate(spy.rows)
    assert leaves.shape[1] == model.obs_size and len(leaves) > 200
    beyond = 0
    for leaf in leaves[:: max(len(leaves) // 300, 1)]:
        glob = dict(zip(columns.global_dice, columns.glob(leaf)))
        assert abs(sum(glob[f"dice_odds_0_{t}"] for t in range(2, 13)) - 6.0) < 1e-4
        for number, tile in zip(columns.tile_numbers(leaf), columns.tiles(leaf)):
            want = ([glob[f"dice_odds_0_{number}"] * 1.2, glob[f"dice_odds_1_{number}"] * 1.2,
                     glob[f"dice_deck_{number}"], glob[f"dice_recent_{number}"]] if number else [0.0] * 4)
            assert np.allclose(tile, want, atol=1e-5)
        beyond += columns.all(leaf).tobytes() not in roots
    assert beyond > 0  # leaves past a roll: the search's own dice states
    # the replay: the rows the searches were asked on, as float16
    assert replay.size > 20 and replay.obs.shape[1] == model.obs_size
    assert all(row.tobytes() in stored for row in replay.obs[: replay.size])
    draws = columns.g0 + columns.global_names.index("dice_draws_left")
    assert (replay.obs[: replay.size, draws] < 1.0).any()  # used decks among them


@pytest.mark.parametrize("dets, groups", [(0, 1), (1, 1), (0, 2)], ids=["oracle", "dets", "pipeline"])
def test_a_warm_started_network_self_plays_as_its_source(tmp_path, dets, groups):
    """Zero dice weights (init_from.zero_new_inputs): the same leaves' logits and values bit
    for bit, so the same searches, moves and targets; the replay rows differ by the dice
    columns alone. (``groups`` 2: the trainer's default, search/pipeline.py.)"""
    source = network(False, seed=3)
    config = {"env": {"scenarios": [{"game": "base-2p-fr"}]}, "model": source.spec}
    ref = ck.save(tmp_path / "e000000.pt", source, run=None, epoch=0, agent_steps=0, config=config, code={})
    target = network(True, seed=4)
    ck.load_parts(target, {"ckpt": str(ref.path), "parts": ["actor", "critic"], "zero_new_inputs": True})
    (plain, plain_replay), (dice, dice_replay) = (selfplay(net, groups, dets=dets) for net in (source, target))
    rngs = np.random.default_rng(1), np.random.default_rng(1)
    stats = dict(STATS), dict(STATS)
    for step in range(100):
        plain.step(0, step, rngs[0], plain_replay, stats[0])
        dice.step(0, step, rngs[1], dice_replay, stats[1])
        assert np.array_equal(plain.buffers["actions"], dice.buffers["actions"]), step
    assert stats[0] == stats[1] and plain.decisions == dice.decisions and plain.games == dice.games > 0
    assert plain_replay.size == dice_replay.size > 20
    size = plain_replay.size
    for name in ("policy", "has_policy", "value", "present"):
        assert np.array_equal(getattr(plain_replay, name)[:size], getattr(dice_replay, name)[:size]), name
    others = np.setdiff1d(np.arange(target.obs_size), FairDice(target).columns)
    assert len(others) == source.obs_size
    assert np.array_equal(dice_replay.obs[:size][:, others], plain_replay.obs[:size])


def variant(name: str, *, dice: bool, init_from=None, **train) -> Variant:
    raw = {"seed": 3, "env": ENV, "obs": {"level": "full", **({"dice_state": True} if dice else {})},
           "model": {"preset": "graph-tiny", "args": {"shared": True}}, "train": {**TRAIN, **train},
           "runtime": {"device": "cpu", "threads": 2, "determinism": "strict"},
           "eval": {"select": None, "final": []}, **({"init_from": init_from} if init_from else {})}
    cfg = resolve(raw, experiment="azd", variant=name)
    return Variant("azd", name, None, {}, cfg, config_hash(cfg))


def run(v: Variant, monitors: bool = False) -> RunWriter:
    with RunWriter.open(v, argv=["test"]) as writer:
        train_alphazero(writer.variant_for_training, writer, monitors=monitors)
    return writer


def test_a_run_from_a_plain_checkpoint_starts_as_it_logs_the_probe_and_resumes_it(monkeypatch):
    assert run(variant("plain", dice=False, iterations=1)).status == "finished"
    start = load_experiment("azd/plain").resolve("latest")
    init = {"ckpt": "azd/plain@latest", "parts": ["actor", "critic"], "zero_new_inputs": True}
    v = variant("dice", dice=True, init_from=init, iterations=3)
    real = RunWriter.should_stop

    def stop_after_two(self):
        if self.launch == 0 and len(canon.read_jsonl(self.path / "metrics.jsonl")) >= 2:
            self.stop_reason = "interrupted"
            return True
        return real(self)

    monkeypatch.setattr(RunWriter, "should_stop", stop_after_two)
    assert run(v, monitors=True).status == "interrupted"  # the drift probe plays dice rows too
    dice_run = load_experiment("azd/dice")
    e0 = ck.Checkpoint.load(dice_run.checkpoints()[0].path)
    assert e0.dice_state and e0.levels["dice_state"]
    source, warm = ck.Checkpoint.load(start.path).model("all"), e0.model("all")
    theirs, mine = source.state_dict(), warm.state_dict()
    assert all(torch.equal(mine[k], theirs[k]) for k in theirs)
    assert sorted(set(mine) - set(theirs)) == sorted(ck.dice_input_keys(warm)) and len(set(mine) - set(theirs)) == 2
    saved = torch.load(dice_run.path / "trainer_state.pt", weights_only=False)
    rows = saved[STATE_KEY]
    assert rows.dtype == torch.float32 and rows.shape[1] == warm.obs_size and 0 < len(rows) <= 512
    assert saved["replay"]["obs"].shape[1] == warm.obs_size
    # at e0 the true and the fair rows read alike (zero weights), bit for bit
    with torch.no_grad():
        (logits, values), (fair_logits, fair_values) = warm(rows), warm(FairDice(warm).rows(rows))
        plain_logits, plain_values = source(rows[:, np.setdiff1d(np.arange(warm.obs_size), FairDice(warm).columns)])
    assert torch.equal(logits, fair_logits) and torch.equal(values, fair_values)
    assert torch.equal(logits, plain_logits) and torch.equal(values, plain_values)
    assert not torch.equal(rows, FairDice(warm).rows(rows))  # though the rows differ

    assert run(v, monitors=True).status == "finished"  # the resume
    final = torch.load(dice_run.path / "trainer_state.pt", weights_only=False)
    assert final["epoch"] == 3 and torch.equal(final[STATE_KEY], rows)  # the same batch
    metrics = load_experiment("azd/dice").metrics()
    assert [m["launch"] for m in metrics] == [0, 0, 1]
    assert all("dice/weight_norm" in m and "dice/weight_norm_global" in m and "dice/weight_norm_tiles" in m
               for m in metrics)
    probed = [m for m in metrics if "dice/probe_kl" in m]
    assert probed and probed[-1] is metrics[-1] and all(m["dice/probe_rows"] == len(rows) for m in probed)
    learned = [m for m in metrics if "train/policy_loss" in m]
    assert learned and all("dice/probe_kl" in m for m in learned)
    assert 0.0 < learned[0]["dice/weight_norm"] and metrics[-1]["dice/weight_norm_tiles"] > 0.0
    assert all(np.isfinite(m["dice/probe_kl"]) and m["dice/probe_value_gap"] > 0.0 for m in probed)
    assert all(m["drift/rows"] > 0 and "oos/value_bce" in m for m in metrics)  # the monitors, on dice rows
    assert metrics[0]["drift/kl_init"] > 0.0
    assert not any(key.startswith("dice/") for m in load_experiment("azd/plain").metrics() for key in m)
