"""The offline distillation gate (catanrl.train.distill): positions, searches, the dataset,
the targets, the fit and its anchors, and runs through RunWriter (collect, resume, fit)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from catanrl import canon, checkpoint, provenance
from catanrl.config import ConfigError, Variant, check_distill, config_hash, resolve, train_defaults
from catanrl.envs.puffer import plan_env
from catanrl.runs import RunWriter, load_experiment
from catanrl.search import BatchEvaluator, make_pool, search_config
from catanrl.train import distill
from catanrl.train.alphazero import gumbel_target
from catanrl.train.puffer import build_policy_model

ENV = {"scenarios": [{"game": "base-2p-fr", "maps": 4, "turn_limit": 10}], "num_envs": 3}  # short games
TRAIN = {"games": 5, "positions_per_game": 6, "sims": 12, "gumbel_m": 4, "search_rows": 8, "updates": 6,
         "minibatch_size": 16, "log_interval": 2, "checkpoint_interval": 4, "holdout_fraction": 0.2}
ENGINE = {"target_c_visit": 50.0, "target_c_scale": 0.1, "target_rescale": True, "target_prior_temperature": 1.0}
CANOPY = {"policy_target": "gumbel", "target_c_scale": 1.0, "target_rescale": False}


def raw(train: dict, init: str | None = None) -> dict:
    return {"seed": 7, "env": ENV, "obs": {"level": "full"}, "model": {"preset": "graph-tiny", "args": {"shared": True}},
            "train": {"algo": "distill", **TRAIN, **train},
            "runtime": {"device": "cpu", "threads": 2, "determinism": "strict"},
            **({"init_from": {"ckpt": init, "parts": ["actor", "critic"]}} if init else {})}


def variant(name: str, init: str, **train) -> Variant:
    cfg = resolve(raw(train, init), experiment="gate", variant=name)
    return Variant("gate", name, None, {}, cfg, config_hash(cfg))


def start_model(seed: int = 3):
    cfg = resolve(raw({"phase": "collect"}), experiment="gate", variant="start")
    plan = plan_env(cfg["env"], pool_seed=1)
    torch.manual_seed(seed)
    return cfg, plan, build_policy_model(cfg, plan.dims).eval()


@pytest.fixture
def start(tmp_path) -> str:
    """A shared-trunk graph-tiny network at the full level, saved as the gate's start (a .pt path)."""
    cfg, _, model = start_model()
    ref = checkpoint.save(tmp_path / "start.pt", model, run=None, epoch=0, agent_steps=0, config=cfg,
                          code=provenance.code_id("eval"))
    return str(ref.path)


@pytest.fixture(scope="module")
def searched():
    """Four games of the start's raw self-play, six positions each, searched (no runs directory)."""
    _, plan, model = start_model()
    evaluator = BatchEvaluator(model, device="cpu", infer_rows=8, mode="strict")
    kept, winner, decisions, seeds = distill.play_games(plan, model, evaluator, games=4, keep=6, seed=11, threads=2)
    params = {"sims": 12, "oracle": True, "root": "gumbel", "c_puct": 0.75, "gumbel_m": 4, "noise": False}
    pool = make_pool(plan.dims, model.levels["level"], search_config(params, max_players=2, threads=2),
                     critic_level=model.levels["critic_level"])
    mask = slice(model.mask_offset, model.mask_offset + model.num_actions)
    positions = [(env, item) for env, items in enumerate(kept) for item in items]
    arrays, leaves = distill.search_positions(pool, evaluator, positions, rows=8, seed=5, mask=mask)
    arrays.update({"decisions": decisions[arrays["game"]].astype(np.int32),
                   "winner": winner[arrays["game"]].astype(np.int8), "game_seed": seeds[arrays["game"]]})
    return {"model": model, "evaluator": evaluator, "kept": kept, "decisions": decisions, "arrays": arrays,
            "leaves": leaves, "mask": mask}


def run(v: Variant) -> RunWriter:
    with RunWriter.open(v, argv=["test"]) as writer:
        distill.train_distill(writer.variant_for_training, writer)
    return writer


# ---------------------------------------------------------------- config


def test_the_defaults_are_the_trainers_keys():
    assert train_defaults("distill") == distill.DISTILL_KEYS  # configs/train/distill.yaml
    engine = {k: distill.DISTILL_KEYS[k] for k in ENGINE}
    assert engine == ENGINE  # the engine's own sigma: gumbel_target's defaults and alphazero's
    from catanrl.train.alphazero import ALPHAZERO_KEYS

    assert all(ALPHAZERO_KEYS[k] == v for k, v in ENGINE.items())


def test_check_distill_refuses_what_it_cannot_build():
    base = dict(distill.DISTILL_KEYS)
    check_distill({**base, "phase": "collect"})
    check_distill({**base, "dataset": "gate/positions", **CANOPY, "target_anchor": 0.5})
    for bad, match in (({"phase": "fit"}, "train.dataset"),
                       ({"phase": "collect", "dataset": "gate/positions"}, "phase: fit"),
                       ({"phase": "search"}, "collect or fit"),
                       ({"phase": "collect", "dets": 1}, "true game"),
                       ({"phase": "collect", "kernels": "cuda"}, "torch or fused"),
                       ({"dataset": "x/y", "target_c_scale": 1.0}, "set policy_target: gumbel"),
                       ({"dataset": "x/y", "policy_target": "gumbel", "root": "puct"}, "Gumbel root"),
                       ({"dataset": "x/y", "target_anchor": 1.5}, r"\[0, 1\]"),
                       ({"dataset": "x/y", "holdout_fraction": 1.0}, "fraction"),
                       ({"dataset": "x/y", "updates": 0}, "positive")):
        with pytest.raises(ConfigError, match=match):
            check_distill({**base, **bad})
    with pytest.raises(ConfigError, match="train.dataset"):  # at resolution, before a run dir exists
        resolve(raw({}), experiment="gate", variant="fit")


# ---------------------------------------------------------------- positions and searches


def test_each_game_keeps_a_uniform_sample_of_its_decisions(searched):
    kept, decisions, mask = searched["kept"], searched["decisions"], searched["mask"]
    for items, total in zip(kept, decisions, strict=True):
        assert len(items) == min(6, int(total)) and total > 6
        picked = [item.decision for item in items]
        assert picked == sorted(set(picked)) and picked[-1] < total
        for item in items:
            assert (item.row[mask] > 0).sum() > 1 and item.seat in (0, 1)
    # Over many games every decision index is equally likely to be kept: a reservoir, not the first six.
    _, plan, model = start_model()
    evaluator = BatchEvaluator(model, device="cpu", infer_rows=8, mode="strict")
    many, _, _, _ = distill.play_games(plan, model, evaluator, games=8, keep=3, seed=2, threads=2)
    late = sum(item.decision >= 3 for items in many for item in items)
    assert late > 12  # 24 kept; with ~30+ decisions a game, nearly all should be past the first three


def test_searched_positions_keep_what_gumbel_target_reads(searched):
    """Per position: legal actions in the row, the engine's target over them, Q unvisited exactly
    where visits are 0, and the start's own logits and values at the root (the stored row through
    the same strict evaluator gives them back exactly)."""
    a, evaluator, mask = searched["arrays"], searched["evaluator"], searched["mask"]
    offsets = distill.offsets_of(a["num_legal"])
    assert len(a["num_legal"]) == sum(len(items) for items in searched["kept"]) and offsets[-1] == len(a["actions"])
    assert searched["leaves"] == len(a["num_legal"]) * 13  # the root, then 12 simulations
    logits, values = evaluator(a["obs"])
    for i in range(len(a["num_legal"])):
        span = slice(offsets[i], offsets[i + 1])
        legal = np.flatnonzero(a["obs"][i, mask] > 0)
        assert np.array_equal(a["actions"][span], legal) and len(legal) > 1
        assert a["engine"][span].sum() == pytest.approx(1.0, abs=1e-5)
        assert np.array_equal(np.isnan(a["q"][span]), a["visits"][span] == 0)
        assert a["visits"][span].sum() == 12
        assert np.array_equal(a["prior_logits"][span], logits[i, legal])
    assert np.array_equal(a["start_value"], values)
    assert np.array_equal(a["root_value"], np.clip(values[:, 0], 0.0, 1.0))
    assert np.all(a["search_value"] >= 0) and np.all(a["depth"] > 0)


def test_the_engine_target_rebuilds_from_the_stored_statistics(searched):
    """gumbel_target at the engine's constants, from what the dataset keeps over the legal actions
    only, is the engine's own target to float32 rounding (alphazero.py's check, offline)."""
    a = searched["arrays"]
    rebuilt = distill.policy_targets(a, {"policy_target": "gumbel", **ENGINE, "target_anchor": 0.0})
    assert np.abs(rebuilt - a["engine"]).max() <= 1e-6
    stored = distill.policy_targets(a, {"policy_target": "engine", **ENGINE, "target_anchor": 0.0})
    offsets = distill.offsets_of(a["num_legal"])
    assert np.allclose(np.add.reduceat(stored, offsets[:-1]), 1.0, atol=1e-12)
    assert np.abs(stored - a["engine"]).max() <= 1e-6


def test_the_canopy_target_is_the_probes_and_the_compact_rebuild_is_the_dense_one(searched):
    """canopy: Q unrescaled (clamped to [0, 1]), c_scale 1, no prior temperature, the engine's
    c_visit; built per legal action it equals gumbel_target over the codec's whole index space."""
    a, model = searched["arrays"], searched["model"]
    canopy = distill.policy_targets(a, {**distill.DISTILL_KEYS, **CANOPY})
    offsets = distill.offsets_of(a["num_legal"])
    num_actions = model.num_actions
    for i in range(len(a["num_legal"])):
        span = slice(offsets[i], offsets[i + 1])
        legal = np.zeros(num_actions, dtype=bool)
        legal[a["actions"][span]] = True
        logits = np.full(num_actions, -1e9)
        logits[legal] = a["prior_logits"][span]
        visits = np.zeros(num_actions)
        visits[legal] = a["visits"][span]
        q = np.full(num_actions, np.nan)
        q[legal] = a["q"][span]
        full = gumbel_target(logits, legal, visits, q, float(a["root_value"][i]), c_visit=50.0, c_scale=1.0,
                             rescale=False, prior_temperature=1.0)
        assert np.array_equal(full[legal], canopy[span])


def test_the_targets_on_two_moves():
    """Both moves visited, Q 0.52 and 0.55 after 1000 and 1048 visits: the engine's target puts
    the better one ~110 logits up whatever the gap, the canopy target (50 + 1048) x 1 x 0.03, and
    canopy-anchor is half of it and half the start's own policy."""
    a = {"num_legal": np.array([2]), "actions": np.array([1, 4], dtype=np.int16),
         "prior_logits": np.array([0.3, -0.2], dtype=np.float32), "visits": np.array([1000, 1048], dtype=np.uint32),
         "q": np.array([0.52, 0.55]), "root_value": np.array([0.5], dtype=np.float32),
         "engine": np.array([0.0, 1.0], dtype=np.float32)}
    rebuilt = distill.policy_targets(a, {"policy_target": "gumbel", **ENGINE, "target_anchor": 0.0})
    assert rebuilt[0] < 1e-40
    canopy = distill.policy_targets(a, {**ENGINE, **CANOPY, "target_anchor": 0.0})
    assert np.log(canopy[1] / canopy[0]) == pytest.approx(-0.5 + 1098 * 0.03, rel=1e-9)
    prior = a["prior_logits"].astype(np.float64)
    start = np.exp(prior - np.logaddexp(*prior))
    mixed = distill.policy_targets(a, {**ENGINE, **CANOPY, "target_anchor": 0.5})
    assert np.allclose(mixed, 0.5 * canopy + 0.5 * start, rtol=0, atol=1e-15)
    # Cross-entropy to the mix is half the cross-entropy to canopy plus half that to the start
    # (a KL anchor to the start, up to the start's entropy).
    logp = np.log(np.array([0.7, 0.3]))
    assert -(mixed * logp).sum() == pytest.approx(0.5 * -(canopy * logp).sum() + 0.5 * -(start * logp).sum())


def test_target_stats_read_the_overrule_and_one_hot_rates(searched):
    a = searched["arrays"]
    engine = distill.policy_targets(a, {"policy_target": "engine", **ENGINE, "target_anchor": 0.0})
    stats = distill.target_stats(engine, a)
    assert 0.0 <= stats["target/overrule"] <= 1.0 and 0.0 <= stats["target/top_ge_099"] <= 1.0
    anchored = distill.target_stats(distill.policy_targets(a, {**ENGINE, **CANOPY, "target_anchor": 1.0}), a)
    assert anchored["target/overrule"] == 0.0  # the start's own policy agrees with itself
    assert anchored["target/entropy"] == pytest.approx(anchored["target/start_entropy"])


# ---------------------------------------------------------------- the dataset


def test_the_dataset_round_trips_and_chunks_merge_in_order(searched, tmp_path):
    a = searched["arrays"]
    sha = distill.save_arrays(tmp_path / "c.npz", a)
    assert sha == canon.sha256_file(tmp_path / "c.npz")
    back = distill.load_arrays(tmp_path / "c.npz")
    assert set(back) == set(distill.POSITION_ARRAYS) | set(distill.EDGE_ARRAYS)
    for name, values in a.items():
        assert back[name].dtype == values.dtype and np.array_equal(back[name], values, equal_nan=values.dtype.kind == "f")
    assert distill.save_arrays(tmp_path / "again.npz", back) == sha  # the same arrays, the same bytes
    cut = int(np.searchsorted(a["game"], 2))  # games 0-1 and 2-3 as two chunks
    edge_cut = int(distill.offsets_of(a["num_legal"])[cut])
    first = {k: (v[:edge_cut] if k in distill.EDGE_ARRAYS else v[:cut]) for k, v in a.items()}
    second = {k: (v[edge_cut:] if k in distill.EDGE_ARRAYS else v[cut:]) for k, v in a.items()}
    merged = distill.merge_chunks([first, second])
    for name, values in a.items():
        assert np.array_equal(merged[name], values, equal_nan=values.dtype.kind == "f")


def test_games_not_positions_are_held_out():
    games = np.repeat(np.arange(50), 20)
    held = distill.holdout_games(games, 0.1, seed=42)
    assert len(held) == 5 and np.array_equal(held, distill.holdout_games(games, 0.1, seed=42))
    assert not np.array_equal(held, distill.holdout_games(games, 0.1, seed=43))
    mask = np.isin(games, held)
    assert mask.sum() == 100 and not set(games[mask]) & set(games[~mask])
    assert len(distill.holdout_games(np.arange(3), 0.1, seed=1)) == 1  # at least one game when there are two
    assert len(distill.holdout_games(np.arange(3), 0.0, seed=1)) == 0


def test_outcomes_are_seat_relative():
    out = distill.outcome_targets(np.array([0, 1, 1, -1]), np.array([0, 0, 1, 1]), 2)
    assert out.tolist() == [[1, 0], [0, 1], [1, 0], [0.5, 0.5]]


# ---------------------------------------------------------------- the fit's anchors


def test_the_start_is_a_fixed_point_of_its_own_anchors(searched):
    """At the start, with its own policy as the target (target_anchor 1) and its own values as the
    value anchor, the loss has no gradient: the anchors hold the network where it is."""
    model, a = searched["model"], searched["arrays"]
    net = start_model()[2]
    net.load_state_dict(model.state_dict())
    target = distill.policy_targets(a, {**ENGINE, **CANOPY, "target_anchor": 1.0})
    obs = torch.from_numpy(a["obs"])
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.0)
    step = distill.fit_step(net, optimizer, obs, torch.from_numpy(distill.dense(target, a, net.num_actions)),
                            torch.from_numpy(a["start_value"]), {"value_coef": 1.0, "max_grad_norm": 1.0})
    assert step["grad_norm"] < 1e-5 and abs(step["value_kl"]) < 1e-6


def test_the_value_anchor_pulls_a_drifted_critic_back(searched):
    """With the value head moved off the start, steps on the anchored loss (policy target: the
    start's own policy) bring its values back toward the start's."""
    model, a = searched["model"], searched["arrays"]
    net = start_model()[2]
    net.load_state_dict(model.state_dict())
    with torch.no_grad():
        for name, param in net.named_parameters():
            if name.startswith(tuple(net.parts["critic"])) and not name.startswith(tuple(net.parts["actor"])):
                param.add_(0.3 * torch.randn_like(param))
    obs = torch.from_numpy(a["obs"])
    target = torch.from_numpy(distill.dense(distill.policy_targets(a, {**ENGINE, **CANOPY, "target_anchor": 1.0}),
                                            a, net.num_actions))
    start_value = torch.from_numpy(a["start_value"])
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=0.0)
    kls = [distill.fit_step(net, optimizer, obs, target, start_value, {"value_coef": 1.0, "max_grad_norm": 1.0})
           ["value_kl"] for _ in range(30)]
    assert kls[0] > 1e-3 and kls[-1] < 0.3 * kls[0]


# ---------------------------------------------------------------- runs


def test_collect_writes_a_checked_dataset_and_a_fit_trains_the_matched_dose(start, capsys):
    assert run(variant("positions", start, phase="collect")).status == "finished"
    positions = load_experiment("gate/positions")
    data = distill.Dataset.load("gate/positions")
    meta, a = data.meta, data.arrays
    searched = [line.split(" positions, ")[0] for line in capsys.readouterr().out.splitlines() if ": search: " in line]
    assert searched == [f"gate/positions: chunk {i + 1}/2: search: {c['positions']}/{c['positions']}"
                        for i, c in enumerate(meta["chunks"])]  # the progress line at each search's end
    assert meta["sha256"] == canon.sha256_file(positions.path / distill.DATASET)
    assert not list((positions.path / distill.CHUNKS).glob("*.npz"))  # merged, then removed
    assert meta["games"] == 5 and len(meta["chunks"]) == 2 and len(data) == sum(c["positions"] for c in meta["chunks"])
    assert sorted(np.unique(a["game"]).tolist()) == [0, 1, 2, 3, 4]
    assert meta["spec"]["start"] == checkpoint.Checkpoint.load(start).ref.weights_sha256
    rows = positions.metrics()
    assert [r["epoch"] for r in rows] == [1, 2] and rows[-1]["data/sha256"] == meta["sha256"]
    assert rows[-1]["agent_steps"] == len(data) and rows[0]["collect/leaves_per_s"] > 0

    fits = {"engine": {}, "canopy": CANOPY, "canopy-anchor": {**CANOPY, "target_anchor": 0.5}}
    for name, target in fits.items():
        assert run(variant(name, start, dataset="gate/positions", **target)).status == "finished"
    start_sha = checkpoint.Checkpoint.load(start).ref.weights_sha256
    for name in fits:
        fit = load_experiment(f"gate/{name}")
        ckpts = fit.checkpoints()
        assert [c.epoch for c in ckpts] == [0, 4, 6] and ckpts[0].weights_sha256 == start_sha
        assert ckpts[-1].agent_steps == 6 * 16
        state = torch.load(fit.path / "trainer_state.pt", weights_only=False)
        assert {int(s["step"]) for s in state["optimizer"]["state"].values()} == {6}  # exactly `updates` steps
        rows = fit.metrics()
        assert [r["epoch"] for r in rows] == [0, 2, 4, 6] and [r["agent_steps"] for r in rows] == [0, 32, 64, 96]
        first = rows[0]
        assert first["data/sha256"] == meta["sha256"] and first["data/heldout_games"] == 1
        assert first["data/heldout_positions"] + first["data/train_positions"] == len(data)
        assert first["heldout/kl_start"] < 1e-6 and first["heldout/agree_start"] == 1.0
        assert first["heldout/value_drift"] < 1e-6 and abs(first["heldout/value_kl"]) < 1e-6
        assert "target/overrule" in first and "train/policy_loss" in rows[-1]
        assert fit.build_policy("latest").num_actions == meta["num_actions"]  # loads as a normal checkpoint
    assert len({load_experiment(f"gate/{n}").checkpoints()[-1].weights_sha256 for n in fits}) == 3


def test_a_stopped_collect_resumes_at_its_next_chunk_and_makes_the_same_dataset(start, monkeypatch):
    run(variant("straight", start, phase="collect"))
    straight = distill.Dataset.load("gate/straight").meta["sha256"]

    real = RunWriter.should_stop

    def stop_after_one(self):
        if self.launch == 0 and len(canon.read_jsonl(self.path / "metrics.jsonl")) >= 1:
            self.stop_reason = "interrupted"
            return True
        return real(self)

    monkeypatch.setattr(RunWriter, "should_stop", stop_after_one)
    assert run(variant("stopped", start, phase="collect")).status == "interrupted"
    path = load_experiment("gate/stopped").path
    assert [p.name for p in sorted((path / distill.CHUNKS).glob("*.npz"))] == ["c00000.npz"]
    played = []
    original = distill.play_games

    def spy(*args, **kwargs):
        played.append(kwargs["games"])
        return original(*args, **kwargs)

    monkeypatch.setattr(distill, "play_games", spy)
    assert run(variant("stopped", start, phase="collect")).status == "finished"
    assert played == [2]  # only the second chunk's games
    assert distill.Dataset.load("gate/stopped").meta["sha256"] == straight


def test_a_fit_refuses_a_dataset_of_another_start(start, tmp_path):
    run(variant("positions", start, phase="collect"))
    cfg, _, model = start_model(seed=9)
    other = checkpoint.save(tmp_path / "other.pt", model, run=None, epoch=0, agent_steps=0, config=cfg,
                            code=provenance.code_id("eval"))
    with pytest.raises(RuntimeError, match="another start"):
        run(variant("engine", str(other.path), dataset="gate/positions"))
