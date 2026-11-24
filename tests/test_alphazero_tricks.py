"""Four optional pieces of the AlphaZero trainer after KataGo (catanrl.train.alphazero's
docstring): a lower-noise value target (``value_search_blend``), replay sampling by policy
surprise (``surprise_weight``), a soft policy head (``model.args.soft_policy``,
``soft_policy_coef``) and smaller searches in decided games (``decided_value``).

* unset, the keys and the model arg leave specs and config hashes as they were, and each
  refusal of ``check_alphazero_kata`` at resolution and in the trainer;
* the pure functions: the search value, the policy surprise, the sampling distribution, the
  soft target and the blend's ramp;
* the replay's search labels: adds, the saved state, a state from before the labels;
* the soft policy head is built after every other part, nothing that plays reads it, and a
  warm start from a checkpoint without it makes it a copy of the policy head;
* self-play that records the labels, or that watches for decided games without treating them
  differently, is self-play without (the same moves, targets and rng draws); a decided game's
  searches take their own full-search probability and fast budget, and its fast rows are thinned
  with them under ``record_fast_prob``;
* learning: the blended value target, the weighted draws, the soft loss; a tiny run with all
  four on logs them and resumes;
* ``init_from`` carries the outcome head as the part ``outcome`` (a continued run keeps the
  head its source trained), beside the soft head's copy; configs that name only actor and
  critic keep their hashes and their fresh head.
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from test_alphazero_dice import ENV, STATS, TRAIN, run
from test_dice_input import (
    DIMS,
    assert_plays_alike,
    first_projections,
    real_rows,
    saved,
    trained_scale,
)
from test_outcome_head import same

import catanrl.train.alphazero as az
from catanrl import canon
from catanrl import checkpoint as ck
from catanrl.config import TRAIN_OPTIONAL, ConfigError, Variant, config_hash, resolve
from catanrl.envs.encoding import action_size, row_layout
from catanrl.envs.puffer import plan_env
from catanrl.models import build_model, resolve_spec
from catanrl.runs import RunWriter, load_experiment
from catanrl.train.alphazero import (
    ALPHAZERO_OPTIONAL,
    MIN_LOG_PRIOR,
    SEARCH_LABELS,
    ReplayBuffer,
    SelfPlay,
    _Pending,
    check_train,
    learn,
    policy_surprise,
    search_labels,
    search_value,
    soft_policy_target,
    surprise_probabilities,
    value_blend,
)
from catanrl.train.puffer import build_policy_model

#: The four pieces' keys at their unset values.
UNSET = {"value_search_blend": 0.0, "value_blend_iterations": 0, "surprise_weight": 0.0, "surprise_cap": 0.0,
         "soft_policy_coef": 0.0, "soft_policy_temperature": 4.0, "decided_value": 0.0, "decided_lookback": 0,
         "decided_full_search_prob": 0.0, "decided_fast_sims": 0}
#: The bundle's search in small: a PUCT root, pruned targets, trees kept between decisions.
PUCT = {"root": "puct", "policy_target": "pruned", "reuse": True}


@pytest.fixture(autouse=True)
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CATANRL_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.chdir(tmp_path)


# ---------------------------------------------------------------- configs


def raw(*, soft=None, outcome: bool = False, dice: bool = True, env=ENV, init_from=None, **train):
    args = {"shared": True, **({} if soft is None else {"soft_policy": soft}), **({"outcome": True} if outcome else {})}
    return {"seed": 3, "env": env, "obs": {"level": "full", **({"dice_state": True} if dice else {})},
            "model": {"preset": "graph-tiny", "args": args}, "train": {**TRAIN, **train},
            "runtime": {"device": "cpu", "threads": 2, "determinism": "strict"},
            "eval": {"select": None, "final": []}, **({"init_from": init_from} if init_from else {})}


def variant(name: str, **kwargs) -> Variant:
    cfg = resolve(raw(**kwargs), experiment="azt", variant=name)
    return Variant("azt", name, None, {}, cfg, config_hash(cfg))


def test_unset_keys_keep_specs_and_config_hashes():
    assert ALPHAZERO_OPTIONAL == TRAIN_OPTIONAL["alphazero"] and UNSET.items() <= ALPHAZERO_OPTIONAL.items()
    base = resolve_spec({"preset": "graph-s", "args": {"shared": True}})
    assert "soft_policy" not in base["args"]
    assert resolve_spec({"preset": "graph-s", "args": {"shared": True, "soft_policy": False}}) == base
    on = resolve_spec({"preset": "graph-s", "args": {"shared": True, "soft_policy": True}})
    assert on["args"] == {**base["args"], "soft_policy": True} and resolve_spec(on) == on
    with pytest.raises(ValueError, match="soft_policy"):
        resolve_spec({"preset": "graph-s", "args": {"soft_policy": 1}})
    plain = variant("a")
    unset = variant("a", soft=False, **UNSET)
    assert plain.config_hash == unset.config_hash and plain.config == unset.config
    assert not set(UNSET) & set(plain.config["train"]) and "soft_policy" not in plain.config["model"]["args"]
    assert variant("a", **{key: None for key in UNSET}).config_hash == plain.config_hash
    on = variant("a", soft=True, **PUCT, value_search_blend=1, value_blend_iterations=200, surprise_weight=0.5,
                 surprise_cap=10, soft_policy_coef=4, soft_policy_temperature=2, decided_value=0.9,
                 decided_lookback=3, decided_full_search_prob=0.05, decided_fast_sims=1)
    train = on.config["train"]
    assert on.config["model"]["args"]["soft_policy"] is True
    assert {key: train[key] for key in UNSET} == {
        "value_search_blend": 1.0, "value_blend_iterations": 200, "surprise_weight": 0.5, "surprise_cap": 10.0,
        "soft_policy_coef": 4.0, "soft_policy_temperature": 2.0, "decided_value": 0.9, "decided_lookback": 3,
        "decided_full_search_prob": 0.05, "decided_fast_sims": 1}
    assert all(isinstance(train[key], type(value)) for key, value in UNSET.items())
    # each piece alone moves the hash; a key at its unset value beside it does not
    hashes = {plain.config_hash, on.config_hash}
    for kwargs in ({"value_search_blend": 0.5}, {"surprise_weight": 0.5}, {"soft": True, "soft_policy_coef": 1.0},
                   {"decided_value": 0.9, "decided_lookback": 3}):
        one = variant("a", **kwargs)
        assert one.config_hash not in hashes
        assert variant("a", **kwargs, soft_policy_temperature=4, decided_fast_sims=0).config_hash == one.config_hash
        hashes.add(one.config_hash)


REFUSALS = [
    ({"value_search_blend": 1.5}, "value_search_blend"),
    ({"value_search_blend": -0.1}, "value_search_blend"),
    ({"value_search_blend": True}, "value_search_blend"),
    ({"value_blend_iterations": 10}, "value_blend_iterations"),
    ({"value_search_blend": 0.5, "value_blend_iterations": -1}, "value_blend_iterations"),
    ({"value_search_blend": 0.5, "value_blend_iterations": 2.5}, "value_blend_iterations"),
    ({"surprise_weight": 1.0}, "surprise_weight"),
    ({"surprise_weight": "half"}, "surprise_weight"),
    ({"surprise_cap": 10.0}, "surprise_cap"),
    ({"surprise_weight": 0.5, "surprise_cap": 1.0}, "surprise_cap"),
    ({"soft_policy_coef": -1.0}, "soft_policy_coef"),
    ({"soft_policy_temperature": 2.0}, "soft_policy_temperature"),
    ({"soft_policy_coef": 1.0, "soft_policy_temperature": 0.0}, "soft_policy_temperature"),
    ({"decided_value": 0.5, "decided_lookback": 3}, "decided_value"),
    ({"decided_value": 1.0, "decided_lookback": 3}, "decided_value"),
    ({"decided_value": 0.9}, "decided_lookback"),
    ({"decided_lookback": 3}, "decided_value"),
    ({"decided_full_search_prob": 0.1}, "decided_value"),
    ({"decided_fast_sims": 1}, "decided_value"),
    ({"decided_value": 0.9, "decided_lookback": 3, "decided_full_search_prob": 1.5}, "decided_full_search_prob"),
    ({"decided_value": 0.9, "decided_lookback": 3, "decided_fast_sims": 1}, "reuse"),
]


@pytest.mark.parametrize("train, match", REFUSALS)
def test_the_keys_refusals(train, match):
    soft = train.get("soft_policy_coef", 0.0) > 0
    with pytest.raises(ConfigError, match=match):
        resolve(raw(soft=soft, **train), experiment="azt", variant="bad")
    with pytest.raises(ValueError, match=match):  # and in the trainer, on a config that skipped resolution
        check_train({**TRAIN, **train})


def test_a_decided_fast_budget_needs_reuse_unless_it_is_the_fast_one():
    decided = {"decided_value": 0.9, "decided_lookback": 3}
    assert check_train({**TRAIN, **decided, "decided_fast_sims": TRAIN["fast_sims"]})["decided_fast_sims"] == 2
    assert check_train({**TRAIN, **PUCT, **decided, "decided_fast_sims": 1})["decided_fast_sims"] == 1


@pytest.mark.parametrize("kwargs, match", [
    ({"soft_policy_coef": 0.5}, "set model.args.soft_policy"),
    ({"soft": True}, "soft_policy_coef"),
])
def test_the_soft_head_and_its_coefficient_come_together(kwargs, match):
    with pytest.raises(ConfigError, match=match):
        resolve(raw(**kwargs), experiment="azt", variant="bad")
    ppo = {**raw(soft=True), "train": {"algo": "ppo"}}
    with pytest.raises(ConfigError, match="trains with train.algo alphazero"):
        resolve(ppo, experiment="azt", variant="bad")


FOUR = {"scenarios": [{"game": "base-4p", "maps": 4, "turn_limit": 10}], "num_envs": 4}


@pytest.mark.parametrize("train", [{"value_search_blend": 0.5}, {"decided_value": 0.9, "decided_lookback": 3}])
def test_the_search_value_is_for_two_player_games(train):
    key = next(iter(train))
    with pytest.raises(ConfigError, match=f"{key} reads the search's value for both seats of a 2-player game"):
        resolve(raw(dice=False, env=FOUR, **train), experiment="azt", variant="bad")
    plan = plan_env(FOUR, pool_seed=1)
    cfg = {"model": {"preset": "graph-tiny", "args": {"shared": True}}, "obs": {"level": "full"}}
    model = build_policy_model(cfg, plan.dims).eval()
    with pytest.raises(ValueError, match="2-player game"):
        SelfPlay(plan, {**TRAIN, **train}, model=model, num_envs=4, threads=2, seed=5, device="cpu", mode="strict")
    resolve(raw(dice=False, env=FOUR, surprise_weight=0.5), experiment="azt", variant="fine")  # no seat's value read


# ---------------------------------------------------------------- the pure functions


def test_the_search_value_is_q_averaged_by_the_kept_visits():
    nan = float("nan")
    assert search_value(np.array([0.0, 3.0, 1.0, 0.0]), np.array([nan, 0.8, 0.4, 0.9])) == pytest.approx(0.7)
    assert search_value(np.array([5.0, 3.0, 0.0]), np.array([nan, 0.25, 0.9])) == pytest.approx(0.25)  # no value: out
    assert search_value(np.array([2, 2]), np.array([1.0, 0.0])) == pytest.approx(0.5)
    assert search_value(np.zeros(3), np.array([0.5, 0.5, 0.5])) is None
    assert search_value(np.array([1.0, 0.0]), np.array([nan, 0.5])) is None
    assert search_value(np.array([1.0]), np.array([1.0 + 1e-6])) == 1.0  # float32 rounding stays in [0, 1]
    # pruned visits weigh it: the visits the noise forced on a bad move drop out
    visits, kept, q = np.array([40.0, 8.0]), np.array([40.0, 0.0]), np.array([0.6, 0.1])
    assert search_value(visits, q) == pytest.approx(24.8 / 48) and search_value(kept, q) == pytest.approx(0.6)


def test_policy_surprise_is_the_kl_from_the_clean_prior_to_the_target():
    prior = np.array([0.5, 0.25, 0.25, 0.0])
    with np.errstate(divide="ignore"):
        logp = np.log(prior)
    assert policy_surprise(prior, logp) == pytest.approx(0.0, abs=1e-12)
    target = np.array([0.0, 1.0, 0.0, 0.0])
    assert policy_surprise(target, logp) == pytest.approx(np.log(4.0))
    assert policy_surprise(8.0 * target, logp) == pytest.approx(np.log(4.0))  # normalized
    mixed = np.array([0.25, 0.75, 0.0, 0.0])
    assert policy_surprise(mixed, logp) == pytest.approx(0.25 * np.log(0.25 / 0.5) + 0.75 * np.log(0.75 / 0.25))
    assert policy_surprise(mixed.astype(np.float16), logp[None]) == pytest.approx(policy_surprise(mixed, logp))
    # worlds: the mean prior ([0.25, 0.25, 0.5, 0] and the one above: [0.375, 0.25, 0.375, 0])
    with np.errstate(divide="ignore"):
        worlds = np.log(np.array([[0.25, 0.25, 0.5, 0.0], prior]))
    assert policy_surprise(mixed, worlds) == pytest.approx(0.25 * np.log(0.25 / 0.375) + 0.75 * np.log(3.0))
    # mass where the prior has none: the floor, not infinity
    assert policy_surprise(np.array([0.0, 0.0, 0.0, 1.0]), logp) == pytest.approx(-MIN_LOG_PRIOR)
    assert policy_surprise(np.zeros(4), logp) == 0.0


def test_the_sampling_distribution_mixes_uniform_and_surprise():
    p = surprise_probabilities(np.array([0.0, 1.0, 3.0], np.float32), 0.5)
    assert p.tolist() == pytest.approx([1 / 6, 1 / 6 + 1 / 8, 1 / 6 + 3 / 8]) and p.sum() == pytest.approx(1.0)
    assert p.dtype == np.float64
    assert surprise_probabilities(np.array([0.0, 1.0, 3.0]), 0.0).tolist() == pytest.approx([1 / 3] * 3)
    # the cap: surprise beyond cap x the mean (20 here) counts as that
    c = np.array([1.0, 1.0, 1.0, 1.0, 96.0])
    assert surprise_probabilities(c, 0.5, 2.0).tolist() == pytest.approx(
        (0.1 + 0.5 * np.array([1, 1, 1, 1, 40]) / 44).tolist())
    assert surprise_probabilities(c, 0.5, 0.0)[4] == pytest.approx(0.1 + 0.5 * 96 / 100)
    assert surprise_probabilities(c, 0.5, 10.0)[4] == pytest.approx(0.1 + 0.5 * 96 / 100)  # a cap above everything
    # not known (rows restored from a state without it): the mean of the known ones
    nan = float("nan")
    assert surprise_probabilities(np.array([nan, 1.0, 3.0, nan]), 0.5).tolist() == pytest.approx(
        (0.125 + 0.5 * np.array([2, 1, 3, 2]) / 8).tolist())
    assert surprise_probabilities(np.array([nan, nan]), 0.9).tolist() == [0.5, 0.5]
    assert surprise_probabilities(np.zeros(4), 0.9).tolist() == [0.25] * 4


def test_the_soft_target_is_the_target_to_the_power_one_over_t_on_the_legal_actions():
    target = torch.tensor([[0.9, 0.1, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    legal = torch.tensor([[True, True, True, False], [True, True, False, False]])
    soft = soft_policy_target(target, legal, 4.0)
    want = torch.tensor([[(0.9 + 1e-7) ** 0.25, (0.1 + 1e-7) ** 0.25, 1e-7 ** 0.25, 0.0],
                         [1e-7 ** 0.25, (1.0 + 1e-7) ** 0.25, 0.0, 0.0]])
    assert torch.allclose(soft, want / want.sum(1, keepdim=True), atol=1e-6)
    assert torch.allclose(soft.sum(1), torch.ones(2)) and not soft[~legal].any() and (soft[legal] > 0).all()
    assert soft[0, 2] / soft[0, 0] == pytest.approx((1e-7 / 0.9) ** 0.25, rel=1e-3)  # a legal action never searched
    assert torch.allclose(soft_policy_target(target, legal, 1.0), target, atol=1e-6)
    entropy = lambda p: float(-(p * p.clamp_min(1e-12).log()).sum(1)[0])
    assert entropy(soft) > entropy(soft_policy_target(target, legal, 2.0)) > entropy(target)


def test_the_blend_ramps_over_the_runs_iterations():
    train = {"value_search_blend": 0.5, "value_blend_iterations": 4}
    assert [value_blend(train, i) for i in (0, 1, 2, 4, 9)] == pytest.approx([0.0, 0.125, 0.25, 0.5, 0.5])
    assert value_blend({"value_search_blend": 0.5}, 0) == 0.5 and value_blend({}, 7) == 0.0


def test_search_labels_are_seat_relative():
    rows = [_Pending(None, None, 0, 2, 0, 0.8, 1.5), _Pending(None, None, 1, 2), _Pending(None, None, 1, 2, 0, 0.25, 0.0)]
    labels = search_labels(rows, 2)
    assert set(labels) == set(SEARCH_LABELS)
    assert labels["search_value"].tolist() == [[pytest.approx(0.8), pytest.approx(0.2)], [0.0, 0.0], [0.25, 0.75]]
    assert labels["has_search_value"].tolist() == [True, False, True] and labels["surprise"].tolist() == [1.5, 0.0, 0.0]
    assert labels["search_value"].dtype == labels["surprise"].dtype == np.float32
    assert labels["has_search_value"].dtype == bool


# ---------------------------------------------------------------- the replay


def positions(rng, count):
    return (rng.random((count, 10), dtype=np.float32).astype(np.float16),
            rng.random((count, 5), dtype=np.float32).astype(np.float16), rng.random(count) < 0.5,
            rng.random((count, 2), dtype=np.float32), np.ones((count, 2), dtype=np.float32))


def labels_of(rng, count):
    value = rng.random(count, dtype=np.float32)
    return {"search_value": np.stack((value, 1.0 - value), 1), "has_search_value": rng.random(count) < 0.6,
            "surprise": rng.random(count, dtype=np.float32) * 3.0}


def test_the_replay_holds_the_search_labels_and_restores_a_state_without_them():
    import io

    rng = np.random.default_rng(0)
    plain, labelled = ReplayBuffer(8, 10, 5, 2), ReplayBuffer(8, 10, 5, 2, search=True)
    with pytest.raises(ValueError, match="search labels"):
        labelled.add(*positions(rng, 3))
    with pytest.raises(ValueError, match="search labels"):
        plain.add(*positions(rng, 3), None, labels_of(rng, 3))
    assert plain.names == ("obs", "policy", "has_policy", "value", "present") and plain.surprise is None
    assert labelled.names[-3:] == SEARCH_LABELS and not set(SEARCH_LABELS) & set(plain.state())
    first, second = labels_of(rng, 5), labels_of(rng, 6)
    labelled.add(*positions(rng, 5), None, first)
    labelled.add(*positions(rng, 6), search=second)  # wraps: 11 positions through 8 slots
    state = labelled.state()
    newest = {name: np.concatenate((first[name], second[name]))[-8:] for name in SEARCH_LABELS}
    for name in SEARCH_LABELS:
        assert np.array_equal(state[name].numpy(), newest[name]), name
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    again = ReplayBuffer(8, 10, 5, 2, search=True)
    again.restore(torch.load(buffer, weights_only=True))
    assert again.size == 8 and all(np.array_equal(getattr(again, name), state[name].numpy()) for name in again.names)
    assert again.search_value.dtype == again.surprise.dtype == np.float32 and again.has_search_value.dtype == bool
    batch = again.batch(np.arange(8), "cpu")
    assert batch["search_value"].dtype == torch.float32 and batch["has_search_value"].dtype == torch.bool
    assert torch.equal(batch["search_value"], state["search_value"]) and "surprise" not in batch
    plain.add(*positions(rng, 3))
    assert set(plain.batch(np.arange(3), "cpu")) == {"obs", "policy", "has_policy", "value", "present"}
    # with the outcome labels too: both sets, the outcome's first
    both = ReplayBuffer(8, 10, 5, 2, outcome=True, search=True)
    assert both.names[5:] == az.OUTCOME_LABELS + SEARCH_LABELS
    # a state from before the labels: no position has a search value, and the surprise of those
    # with a policy target is not known (NaN), which the sampling reads as the buffer's mean
    old = ReplayBuffer(8, 10, 5, 2, search=True)
    old.restore({k: v for k, v in state.items() if k not in SEARCH_LABELS})
    has = state["has_policy"].numpy()
    assert old.size == 8 and not old.has_search_value.any() and not old.search_value.any()
    assert np.array_equal(old.obs, again.obs) and 0 < has.sum() < 8
    assert np.isnan(old.surprise[has]).all() and not old.surprise[~has].any()
    fresh = labels_of(rng, 2)
    old.add(*positions(rng, 2), search=fresh)  # new positions take the oldest slots, with their labels
    assert np.array_equal(old.surprise[:2], fresh["surprise"])
    p = surprise_probabilities(old.surprise[: old.size], 0.5)
    assert np.isfinite(p).all() and p.sum() == pytest.approx(1.0)
    # the other way: a state with the labels into a buffer without them
    bare = ReplayBuffer(8, 10, 5, 2)
    bare.restore(state)
    assert bare.size == 8 and bare.surprise is None and np.array_equal(bare.value, again.value)


# ---------------------------------------------------------------- the soft policy head


def model_of(*, soft: bool = True, outcome: bool = False, dice: bool = True, seed: int = 0,
             preset: str = "graph-tiny", shared: bool = True):
    torch.manual_seed(seed)
    levels = {"level": "full", **({"dice_state": True} if dice else {})}
    args = {"shared": shared, **({"outcome": True} if outcome else {}), **({"soft_policy": True} if soft else {})}
    return build_model({"preset": preset, "args": args}, dims=DIMS, levels=levels,
                       num_actions=action_size(DIMS), layout=row_layout(DIMS, "full", None, dice)).eval()


@pytest.mark.parametrize("preset, outcome", [("graph-tiny", False), ("graph-tiny", True), ("graph-s", True),
                                             ("graph-m", False)])
def test_the_soft_head_is_built_last_and_changes_no_other_weight_or_output(preset, outcome):
    with_head = model_of(preset=preset, outcome=outcome, seed=3)
    plain = model_of(soft=False, preset=preset, outcome=outcome, seed=3)
    mine, theirs = with_head.state_dict(), plain.state_dict()
    assert list(mine)[: len(theirs)] == list(theirs)
    assert all(torch.equal(mine[k], theirs[k]) for k in theirs)
    new = [k for k in mine if k not in theirs]
    assert new and all(k.startswith("soft_policy_head.") for k in new)
    assert [k.removeprefix("soft_") for k in new] == [k for k in theirs if k.startswith("policy_head.")]
    assert all(mine[k].shape == mine[k.removeprefix("soft_")].shape for k in new)
    assert with_head.soft_policy and not plain.soft_policy
    assert with_head.parts == {**plain.parts, "soft_policy": ["soft_policy_head."]}
    rows = real_rows(True)
    with torch.no_grad():
        assert_plays_alike(with_head, plain, rows, rows)
        heads = with_head.heads(rows)
        logits, value_logits, present, extras = with_head.training_heads(rows)
    assert all(torch.equal(a, b) for a, b in zip(heads, (logits, value_logits, present)))
    assert set(extras) == {"soft_policy"} | ({"final_vp", "turns_left"} if outcome else set())
    soft = extras["soft_policy"]
    legal = rows[:, with_head.mask_offset : with_head.mask_offset + with_head.num_actions] > 0
    assert soft.shape == logits.shape and (soft[~legal] < -1e8).all() and (soft[legal] > -1e8).all()
    assert not torch.equal(soft, logits)  # a head of its own
    if outcome:
        with torch.no_grad():
            both = with_head.heads_with_outcome(rows)
        assert torch.equal(both[3], extras["final_vp"]) and torch.equal(both[4], extras["turns_left"])
    with pytest.raises(ValueError, match="no outcome head"):
        model_of(soft=False).training_heads(rows)
    with pytest.raises(ValueError, match="shared trunk"):
        model_of(shared=False)


@pytest.mark.parametrize("dice", [False, True], ids=["same-inputs", "new-dice-input"])
@pytest.mark.parametrize("preset", ["graph-tiny", "graph-s"])
def test_a_warm_start_makes_the_soft_head_a_copy_of_the_policy_head(tmp_path, preset, dice):
    """init_from with the actor part from a checkpoint without the head: every weight of the
    source, and the soft head the policy head's, so its logits start as the policy's."""
    source = trained_scale(model_of(soft=False, dice=False, preset=preset, seed=3))
    ref = saved(tmp_path, source)
    target = model_of(dice=dice, preset=preset, seed=4)
    init = {"ckpt": str(ref.path), "parts": ["actor", "critic"], **({"zero_new_inputs": True} if dice else {})}
    ck.load_parts(target, init)
    theirs, mine = source.state_dict(), target.state_dict()
    soft = [k for k in mine if k.startswith("soft_policy_head.")]
    assert sorted(set(mine) - set(theirs)) == sorted([*(ck.dice_input_keys(target) if dice else []), *soft])
    assert all(torch.equal(mine[k], theirs[k]) for k in theirs)
    assert soft and all(torch.equal(mine[k], theirs[k.removeprefix("soft_")]) for k in soft)
    assert all(mine[k].data_ptr() != mine[k.removeprefix("soft_")].data_ptr() for k in soft)  # copies
    rows_source, rows = real_rows(False), real_rows(dice)
    assert_plays_alike(source, target, rows_source, rows)
    with torch.no_grad():
        theirs_heads, heads = source.heads(rows_source), target.heads(rows)
        logits, _, _, extras = target.training_heads(rows)
    assert all(torch.equal(a, b) for a, b in zip(theirs_heads, heads))
    assert torch.equal(extras["soft_policy"], logits) and torch.equal(logits, heads[0])
    assert float(logits[logits > -1e8].abs().max()) > 1.0  # not the near-zero logits of a fresh network
    # as saved: the checkpoint carries the head, and plays as the source
    again = ck.Checkpoint.load(saved(tmp_path / "warm", target).path)
    loaded = again.model("all")
    assert loaded.soft_policy and again.header["model"]["args"]["soft_policy"] is True
    assert all(torch.equal(loaded.state_dict()[k], mine[k]) for k in mine)
    assert_plays_alike(source, again.model("actor"), rows_source, rows)


def test_a_warm_start_from_a_checkpoint_with_the_head_takes_its_head(tmp_path):
    source = trained_scale(model_of(seed=3))
    target = model_of(seed=4)
    ck.load_parts(target, {"ckpt": str(saved(tmp_path, source).path), "parts": ["actor", "critic"]})
    theirs, mine = source.state_dict(), target.state_dict()
    assert all(torch.equal(mine[k], theirs[k]) for k in theirs) and list(mine) == list(theirs)
    soft = [k for k in mine if k.startswith("soft_policy_head.")]
    assert soft and any(not torch.equal(mine[k], mine[k.removeprefix("soft_")]) for k in soft)  # not the copy
    # and into a network without the head: the source's other weights, as any part load
    bare = model_of(soft=False, seed=5)
    ck.load_parts(bare, {"ckpt": str(saved(tmp_path, source).path), "parts": ["actor", "critic"]})
    assert all(torch.equal(v, theirs[k]) for k, v in bare.state_dict().items())
    rows = real_rows(True)
    assert_plays_alike(source, bare, rows, rows)


# ---------------------------------------------------------------- init_from's outcome part

OUTCOME_INIT = ["actor", "critic", "outcome"]
#: Config hashes from before ``outcome`` was a part init_from could name (variant("a", ...) below).
OLD_INIT_HASHES = {"default": "04dfc939f06e1d9c", "actor-critic": "414e3d2f46e7541b",
                   "outcome-net": "9f01a6a7058dfd6e", "per-part": "f6d10142f2a8b392"}


@pytest.mark.parametrize("soft", [False, True], ids=["plain", "soft-head"])
def test_the_outcome_part_carries_the_outcome_head(tmp_path, soft):
    """parts: [actor, critic, outcome] from a checkpoint with the head: every weight of the
    source, its outcome head included, so the network plays and predicts outcomes as the source;
    a soft head beside it is still the copy of the policy head."""
    source = trained_scale(model_of(soft=False, outcome=True, seed=3))
    path = str(saved(tmp_path, source).path)
    target = model_of(soft=soft, outcome=True, seed=4)
    refs = ck.load_parts(target, {"ckpt": path, "parts": OUTCOME_INIT})
    assert [str(ref) for ref in refs] == [f"{path}#{refs[0].weights_sha256[:12]}"] * 3
    theirs, mine = source.state_dict(), target.state_dict()
    head = [k for k in theirs if k.startswith("outcome_head.")]
    soft_keys = [k for k in mine if k.startswith("soft_policy_head.")]
    assert head and all(torch.equal(mine[k], theirs[k]) for k in theirs)
    assert sorted(set(mine) - set(theirs)) == sorted(soft_keys) and bool(soft_keys) == soft
    rows = real_rows(True)
    assert_plays_alike(source, target, rows, rows)
    with torch.no_grad():
        assert same(source.heads(rows), target.heads(rows))
        assert same(source.heads_with_outcome(rows), target.heads_with_outcome(rows))
        logits, _, _, extras = target.training_heads(rows)
    assert same(source.heads_with_outcome(rows)[3:], (extras["final_vp"], extras["turns_left"]))
    if soft:
        assert all(torch.equal(mine[k], theirs[k.removeprefix("soft_")]) for k in soft_keys)
        assert torch.equal(extras["soft_policy"], logits)
    # without the part, as before it existed: the source's actor and critic, and a head of the seed's
    fresh = model_of(soft=soft, outcome=True, seed=4)
    seeded = {k: v.clone() for k, v in fresh.state_dict().items()}
    ck.load_parts(fresh, {"ckpt": path, "parts": ["actor", "critic"]})
    state = fresh.state_dict()
    assert all(torch.equal(state[k], seeded[k]) for k in head) and not same([state[k] for k in head],
                                                                           [theirs[k] for k in head])
    assert all(torch.equal(state[k], theirs[k]) for k in theirs if k not in head)
    # the head alone, from another checkpoint than the actor's and the critic's (one ref per part)
    other = trained_scale(model_of(soft=False, outcome=True, seed=8))
    mixed = model_of(soft=soft, outcome=True, seed=4)
    ck.load_parts(mixed, {"ckpt": {"actor": path, "critic": path,
                                   "outcome": str(saved(tmp_path / "other", other).path)}})
    state = mixed.state_dict()
    assert all(torch.equal(state[k], (other if k in head else source).state_dict()[k]) for k in theirs)


def test_the_outcome_part_needs_the_head_on_both_sides(tmp_path):
    with_head = str(saved(tmp_path / "with", trained_scale(model_of(soft=False, outcome=True, seed=3))).path)
    without = str(saved(tmp_path / "without", trained_scale(model_of(soft=False, seed=3))).path)
    with pytest.raises(ck.CheckpointError, match="the checkpoint has no outcome head"):
        ck.load_parts(model_of(outcome=True, seed=4), {"ckpt": without, "parts": OUTCOME_INIT})
    with pytest.raises(ck.CheckpointError, match=r"which this network does not have \(model.args.outcome\)"):
        ck.load_parts(model_of(seed=4), {"ckpt": with_head, "parts": OUTCOME_INIT})
    # a refused load changes no weight
    target = model_of(outcome=True, seed=4)
    before = {k: v.clone() for k, v in target.state_dict().items()}
    with pytest.raises(ck.CheckpointError):
        ck.load_parts(target, {"ckpt": without, "parts": OUTCOME_INIT})
    assert same(before, dict(target.state_dict()))
    # at resolution: the part with a network without the head, and parts that are none of the three
    init = {"ckpt": "azt/src@e1#0123456789ab", "parts": OUTCOME_INIT}
    with pytest.raises(ConfigError, match="init_from's outcome part loads the outcome head"):
        variant("a", init_from=init)
    with pytest.raises(ConfigError, match="init_from.parts must be a non-empty subset"):
        variant("a", outcome=True, outcome_vp_coef=0.5, init_from={**init, "parts": ["actor", "aux"]})
    with pytest.raises(ConfigError, match="init_from.ckpt per part takes keys"):
        variant("a", init_from={"ckpt": {"actor": init["ckpt"], "soft_policy": init["ckpt"]}})
    with pytest.raises(ConfigError, match="init_from's outcome part loads the outcome head"):
        variant("a", init_from={"ckpt": dict.fromkeys(OUTCOME_INIT, init["ckpt"])})


def test_configs_without_the_outcome_part_keep_their_hashes():
    ref = "azt/src@e1#0123456789ab"
    head = {"outcome": True, "outcome_vp_coef": 0.5}
    old = {"default": variant("a", init_from={"ckpt": ref}),
           "actor-critic": variant("a", init_from={"ckpt": ref, "parts": ["actor", "critic"]}),
           "outcome-net": variant("a", **head, init_from={"ckpt": ref, "parts": ["actor", "critic"]}),
           "per-part": variant("a", init_from={"ckpt": {"actor": ref, "critic": ref}})}
    assert {name: v.config_hash for name, v in old.items()} == OLD_INIT_HASHES
    assert old["default"].config["init_from"] == {"ckpt": ref}  # no part is written in
    carried = variant("a", **head, init_from={"ckpt": ref, "parts": OUTCOME_INIT})
    assert carried.config["init_from"] == {"ckpt": ref, "parts": OUTCOME_INIT}
    assert carried.config_hash not in OLD_INIT_HASHES.values()
    per_part = variant("a", **head, init_from={"ckpt": dict.fromkeys(OUTCOME_INIT, ref)})
    assert per_part.config_hash not in {carried.config_hash, *OLD_INIT_HASHES.values()}


def test_a_run_continues_its_source_outcome_head(monkeypatch):
    """A run with the outcome head, then one from its checkpoint with parts: [actor, critic,
    outcome] and the soft head: the run pins the three parts and starts from the source's
    weights, outcome head included."""
    head = {"outcome": True, "outcome_vp_coef": 0.5, "outcome_turns_coef": 0.5}
    assert run(variant("src", **head, iterations=1)).status == "finished"
    start = load_experiment("azt/src").resolve("latest")
    v = variant("carry", soft=True, **head, soft_policy_coef=1.0, iterations=1,
                init_from={"ckpt": "azt/src@latest", "parts": OUTCOME_INIT})
    assert run(v).status == "finished"
    carried = load_experiment("azt/carry")
    assert carried.manifest["resolved"]["init_from"] == dict.fromkeys(OUTCOME_INIT, str(start))
    source = ck.Checkpoint.load(start.path).model("all")
    warm = ck.Checkpoint.load(carried.checkpoints()[0].path).model("all")
    theirs, mine = source.state_dict(), warm.state_dict()
    assert any(k.startswith("outcome_head.") for k in theirs) and all(torch.equal(mine[k], theirs[k]) for k in theirs)
    rows = real_rows(True)
    with torch.no_grad():
        assert same(source.heads_with_outcome(rows), warm.heads_with_outcome(rows))
        logits, _, _, extras = warm.training_heads(rows)
    assert torch.equal(extras["soft_policy"], logits)


def test_the_fused_kernels_take_the_soft_head_network_and_fold_the_same_weights(monkeypatch):
    from catanrl.models import graph_kernels

    monkeypatch.setattr(graph_kernels, "kernels", lambda: SimpleNamespace(hidden=96))
    full = trained_scale(model_of(preset="graph-s", outcome=True, seed=7))
    bare = model_of(soft=False, preset="graph-s", seed=57)
    bare.load_state_dict({k: v for k, v in full.state_dict().items()
                          if not k.startswith(("soft_policy_head.", "outcome_head."))})
    a, b = graph_kernels.FusedGraph(full), graph_kernels.FusedGraph(bare)
    assert same(vars(a), vars(b))
    rows = real_rows(True)
    with torch.no_grad():
        (glob_a, us_a), (glob_b, us_b) = first_projections(a, rows), first_projections(b, rows)
    assert torch.equal(glob_a, glob_b) and all(torch.equal(x, y) for x, y in zip(us_a, us_b))


def test_the_batch_evaluator_answers_as_without_the_soft_head():
    from catanrl.search import BatchEvaluator

    full = trained_scale(model_of(seed=7))
    bare = model_of(soft=False, seed=57)
    bare.load_state_dict({k: v for k, v in full.state_dict().items() if not k.startswith("soft_policy_head.")})
    rows = real_rows(True).numpy()
    a, b = (BatchEvaluator(net, device="cpu", infer_rows=16, mode="strict", kernels="torch")(rows)
            for net in (full, bare))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


# ---------------------------------------------------------------- self-play


def network(soft: bool = False, outcome: bool = False, seed: int = 0):
    torch.manual_seed(seed)
    args = {"shared": True, **({"outcome": True} if outcome else {}), **({"soft_policy": True} if soft else {})}
    cfg = {"model": {"preset": "graph-tiny", "args": args}, "obs": {"level": "full", "dice_state": True}}
    return build_policy_model(cfg, plan_env(ENV, pool_seed=1).dims).eval()


def selfplay(model, **train):
    plan = plan_env(ENV, pool_seed=1)
    play = SelfPlay(plan, check_train({**TRAIN, **train}), model=model, num_envs=4, threads=2, seed=5, device="cpu",
                    mode="strict", groups=1)
    replay = ReplayBuffer(5000, play.vec.observation_size, play.vec.action_size, play.players,
                          outcome=play.outcome, search=play.search)
    return play, replay


def played(play, replay, steps: int = 100, seed: int = 1):
    """``steps`` of self-play: each step's actions, and the statistics."""
    rng, stats, actions = np.random.default_rng(seed), dict(STATS), []
    for step in range(steps):
        play.step(0, step, rng, replay, stats)
        actions.append(play.buffers["actions"].copy())
    return actions, stats, rng


BASE = ("obs", "policy", "has_policy", "value", "present")


def stats_full(run) -> int:
    return int(run[2][1]["full"])


def assert_same_selfplay(a, b):
    (play_a, replay_a, (actions_a, stats_a, rng_a)), (play_b, replay_b, (actions_b, stats_b, rng_b)) = a, b
    assert all(np.array_equal(x, y) for x, y in zip(actions_a, actions_b))
    assert play_a.decisions == play_b.decisions and play_a.games == play_b.games > 0
    assert all(stats_a[key] == stats_b[key] for key in stats_a.keys() & stats_b.keys())
    size = replay_a.size
    assert size == replay_b.size > 20
    for name in BASE:
        assert np.array_equal(getattr(replay_a, name)[:size], getattr(replay_b, name)[:size]), name
    assert rng_a.random() == rng_b.random()  # the same draws from the caller's generator


@pytest.mark.parametrize("search", [{}, PUCT, {"root": "puct", "dets": 1}], ids=["gumbel", "pruned-reuse", "puct-dets"])
def test_self_play_that_records_the_search_labels_is_self_play_without(search, monkeypatch):
    """The value and the surprise are read off searches that ran anyway: the same moves, targets
    and draws; the labels are those of the full searches."""
    weighed = []
    real_value, real_pruned = az.search_value, az.pruned_visits

    def value_spy(weights, q):
        weighed.append(weights)
        return real_value(weights, q)

    kept = []

    def pruned_spy(*args, **kwargs):
        kept.append(real_pruned(*args, **kwargs))
        return kept[-1]

    model = network(seed=3)
    off = selfplay(model, **search)
    off = (*off, played(*off))
    assert not off[0].search and not off[1].search and off[1].surprise is None
    monkeypatch.setattr(az, "search_value", value_spy)
    monkeypatch.setattr(az, "pruned_visits", pruned_spy)
    on = selfplay(model, **search, value_search_blend=0.5, surprise_weight=0.5)
    on = (*on, played(*on))
    assert_same_selfplay(off, on)
    play, replay = on[:2]
    assert play.search and replay.search and len(weighed) == stats_full(on)
    size = replay.size
    has = replay.has_policy[:size]
    assert 0 < has.sum() < size  # record_fast: the fast searches' positions too, without labels
    assert np.array_equal(replay.has_search_value[:size], has)
    values = replay.search_value[:size]
    assert np.allclose(values[has].sum(1), 1.0, atol=1e-6) and (values[has] >= 0).all() and not values[~has].any()
    assert len(np.unique(values[has][:, 0])) > 5
    surprise = replay.surprise[:size]
    assert (surprise >= 0).all() and np.isfinite(surprise).all() and not surprise[~has].any()
    assert (surprise[has] > 0).mean() > 0.5 and surprise.max() < 20.0
    if search.get("policy_target") == "pruned":  # weighed by the very visits the policy target keeps
        assert len(weighed) == len(kept) > 0 and all(a is b for a, b in zip(weighed, kept))


def test_the_recorded_labels_are_the_searches_own():
    """Each full search's value and surprise, recomputed from its tree as the search ended: the
    root's action values by the pruned visits, and the KL from the root's prior to the pruned
    target."""
    play, replay = selfplay(network(seed=3), **PUCT, record_fast=False, value_search_blend=1.0, surprise_weight=0.5)
    pool, search, finish = play.full_pool, play._search, play._finish
    full, checked = {}, []

    def tracked(pool, env, is_full, seed, stats):
        full[env] = is_full
        return search(pool, env, is_full, seed, stats)

    def snapshot(replay):  # every search as it ended, before the finished games' trees go
        for env, (_, sid, _, _) in play.trees.items():
            if not full[env]:
                continue
            row = play.pending[env][-1]  # record_fast false: the full searches' positions alone
            visits, q = np.asarray(pool.visits(sid), np.float64), np.asarray(pool.q(sid), np.float64)
            log_priors = az.root_log_priors(pool, sid)
            kept = az.pruned_visits(log_priors, row.obs.astype(np.float32)[play.mask] > 0, visits, q,
                                    c_puct=pool.config.c_puct)
            assert row.search_value == pytest.approx(search_value(kept, q)) and 0.0 <= row.search_value <= 1.0
            assert row.surprise == pytest.approx(policy_surprise(kept, log_priors), abs=5e-3)  # a float16 target
            checked.append(row.surprise)
        finish(replay)

    play._search, play._finish = tracked, snapshot
    rng, stats = np.random.default_rng(1), dict(STATS)
    for step in range(60):
        play.step(0, step, rng, replay, stats)
    size = replay.size
    assert len(checked) == stats["full"] > 30 and size > 20 and max(checked) > 0.01
    assert replay.has_search_value[:size].all() and replay.has_policy[:size].all()


def test_a_game_is_decided_by_its_last_searches():
    play, _ = selfplay(network(), decided_value=0.9, decided_lookback=3)
    assert play.decided == 0.9 and all(d.maxlen == 3 and not d for d in play.leaders)
    assert not play._decided(0)
    for leaders, want in (([(0, 0.95), (0, 0.92)], False),  # too few searches so far
                          ([(0, 0.95), (0, 0.92), (0, 0.9)], True),
                          ([(1, 0.95), (1, 0.92), (1, 0.99)], True),
                          ([(0, 0.95), (1, 0.92), (0, 0.9)], False),  # the leader changed
                          ([(0, 0.95), (0, 0.89), (0, 0.99)], False),  # one below the bar
                          ([(1, 0.5), (0, 0.95), (0, 0.92), (0, 0.91)], True)):  # the last three count
        play.leaders[1] = deque(leaders, maxlen=3)
        assert play._decided(1) is want, leaders
    off, _ = selfplay(network())
    off.leaders[1] = deque([(0, 0.99)], maxlen=1)
    assert not off.decided and not off._decided(1)


@pytest.mark.parametrize("search", [{}, PUCT], ids=["gumbel", "pruned-reuse"])
def test_watching_for_decided_games_alone_changes_no_search(search):
    """decided_value with the decided games' searches as everyone's (the same full-search
    probability, fast_sims): the same draws against the same threshold, the same self-play."""
    model = network(seed=3)
    off = selfplay(model, **search)
    off = (*off, played(*off))
    on = selfplay(model, **search, decided_value=0.5001, decided_lookback=1,
                  decided_full_search_prob=TRAIN["full_search_prob"])
    on = (*on, played(*on))
    assert_same_selfplay(off, on)
    stats = on[2][1]
    assert 0 < stats["decided"] <= on[0].decisions and stats["simulations"] > 0
    assert "decided" not in off[2][1] and "simulations" not in off[2][1]


def test_decided_games_are_searched_less(monkeypatch):
    """A bar every search clears (0.5001, one search back): from a game's second decision on it
    is decided, never searched in full (probability 0) and its fast searches run 1 simulation."""
    model = network(seed=3)
    play, replay = selfplay(model, **PUCT, sims=24, fast_sims=6, decided_value=0.5001, decided_lookback=1,
                            decided_full_search_prob=0.0, decided_fast_sims=1)
    assert play.options[1]["simulations"] == 6 and play.decided_options == {**play.options[1], "simulations": 1}
    pool, search, started = play.full_pool, play._search, {}

    def tracked(pool, env, full, seed, stats):
        decided = play._decided(env)
        sid = search(pool, env, full, seed, stats)
        started[env] = (full, decided, int(pool.visits(sid).sum()), len(play.leaders[env]))
        return sid

    finish, ended = play._finish, {}

    def snapshot(replay):
        ended.clear()
        for env, (_, sid, _, _) in play.trees.items():
            ended[env] = (int(pool.visits(sid).sum()), int(pool.stats(sid).simulations), list(play.leaders[env]))
        finish(replay)

    play._search, play._finish = tracked, snapshot
    rng, stats, rows, over = np.random.default_rng(1), dict(STATS), [], set(range(4))
    b = play.buffers
    for step in range(80):
        acting = np.flatnonzero(b["acting"])
        envs, seats = b["env_index"][acting].tolist(), b["seat"][acting].tolist()
        play.step(0, step, rng, replay, stats)
        for env, seat in zip(envs, seats):
            full, decided, carried, searches = started[env]
            total, simulations, leaders = ended[env]
            assert (searches == 0) == (env in over)  # a new game starts with no search behind it
            cap = 24 if full else 1 if decided else 6
            assert total == max(carried, cap) and simulations == total - carried
            (leader, value), = leaders  # this search's, the one the next decision reads
            assert 0.5 <= value <= 1.0 and leader in (seat, 1 - seat)
            rows.append((full, decided, searches, simulations))
        over = set(np.flatnonzero(b["terminals"].reshape(-1, play.players)[:, 0]).tolist())
    full, decided, searches, simulations = np.array(rows, dtype=np.float64).T
    assert play.games > 4 and (searches == 0).sum() >= play.games
    assert not decided[searches == 0].any()  # a game's first decision: nothing to go by yet
    assert decided.mean() > 0.6 and not full[decided == 1].any()  # decided games: no full searches
    assert 0 < full[decided == 0].sum() and stats["full"] == full.sum()
    assert stats["decided"] == decided.sum() and stats["simulations"] == simulations.sum()
    assert simulations[(decided == 1)].max() <= 1 < simulations[(decided == 0) & (full == 0)].max()
    assert all(not leaders for env, leaders in enumerate(play.leaders) if env in over)  # reset at a game's end


def test_record_fast_prob_thins_a_decided_games_fast_rows_as_its_full_searches():
    """record_fast_prob below 1 stores a fast-searched move with that probability, and a decided
    game's with it times decided_full_search_prob / full_search_prob (a quarter of it here): a
    second draw per acting row, after the full searches'."""
    play, replay = selfplay(network(seed=3), **PUCT, decided_value=0.5001, decided_lookback=1,
                            decided_full_search_prob=0.125, record_fast_prob=0.8)
    rng, twin, stats = np.random.default_rng(1), np.random.default_rng(1), dict(STATS)
    b, rows, stored = play.buffers, [], 0
    for step in range(200):
        decided = np.array([play._decided(int(env)) for env in b["env_index"][np.flatnonzero(b["acting"])]], dtype=bool)
        full = twin.random(len(decided)) < np.where(decided, 0.125, TRAIN["full_search_prob"])
        kept = twin.random(len(decided)) < np.where(decided, 0.2, 0.8)
        play.step(0, step, rng, replay, stats)
        stored += int((full | kept).sum())
        assert replay.added + sum(len(waiting) for waiting in play.pending) == stored
        rows += zip(decided, full, kept)
    assert rng.random() == twin.random()  # the two draws a step, no other
    decided, full, kept = np.array(rows).T
    assert stats["full"] == full.sum() and stats["recorded_fast"] == (kept & ~full).sum()
    assert stats["decided"] == decided.sum() and play.games > 4
    fast = ~full
    assert (fast & ~decided).sum() >= 30 and (fast & decided).sum() >= 300
    assert kept[fast & ~decided].mean() > 0.6 and 0.13 < kept[fast & decided].mean() < 0.27  # 0.8 and 0.2


# ---------------------------------------------------------------- learning


def filled(model, steps: int = 100, **train):
    play, replay = selfplay(model, **train)
    played(play, replay, steps)
    return play, replay


def test_unset_learning_draws_and_trains_as_before():
    """A buffer with the search labels, read by a trainer with every key unset: the same
    minibatches (rng.integers), the same loss and the same weights as a buffer without."""
    results = []
    for search in ({}, {"value_search_blend": 0.5, "surprise_weight": 0.5}):
        net = network(seed=2)
        _, replay = filled(net, **search)
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
        rng = np.random.default_rng(4)
        stats = learn(net, optimizer, replay, TRAIN, "cpu", rng, 7)
        results.append((stats, {k: v.clone() for k, v in net.state_dict().items()}, rng.random()))
    (plain, weights, draw), (labelled, weights_too, draw_too) = results
    assert plain == labelled and draw == draw_too and all(torch.equal(weights[k], weights_too[k]) for k in weights)
    assert not any(key.startswith(("surprise", "soft", "search_value")) or key == "value_blend" for key in plain)


def test_the_value_target_blends_the_search_value_into_the_game_result(monkeypatch):
    net = network(seed=2)
    _, replay = filled(net, value_search_blend=0.5)
    size = replay.size
    has = replay.has_search_value[:size]
    assert 0 < has.sum() < size
    seen = []
    real = F.binary_cross_entropy_with_logits

    def bce(logits, target, **kwargs):
        seen.append(target.clone())
        return real(logits, target, **kwargs)

    batches = []
    real_batch = ReplayBuffer.batch
    monkeypatch.setattr(ReplayBuffer, "batch", lambda self, index, device: batches.append(
        real_batch(self, index, device)) or batches[-1])
    monkeypatch.setattr(az.F, "binary_cross_entropy_with_logits", bce)
    train = {**TRAIN, "value_search_blend": 0.5, "value_blend_iterations": 4, "updates_per_iteration": 2}
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    for iteration, weight in ((0, 0.0), (2, 0.25), (4, 0.5), (40, 0.5)):
        seen.clear(), batches.clear()
        stats = learn(net, optimizer, replay, train, "cpu", np.random.default_rng(iteration), iteration)
        assert stats["value_blend"] == pytest.approx(weight) and len(seen) == len(batches) == 2
        gaps = []
        for target, batch in zip(seen, batches):
            searched = batch["has_search_value"]
            assert 0 < searched.sum() < len(searched)
            want = torch.where(searched[:, None], (1 - weight) * batch["value"] + weight * batch["search_value"],
                               batch["value"])
            assert torch.allclose(target, want, atol=1e-7) and torch.equal(target[~searched], batch["value"][~searched])
            assert (weight == 0) == torch.equal(target, batch["value"])
            gaps.append(float((batch["search_value"][searched][:, 0] - batch["value"][searched][:, 0]).abs().mean()))
        assert stats["search_value_gap"] == pytest.approx(np.mean(gaps), rel=1e-5) and stats["search_value_gap"] > 0
        # the seat sums keep reading the game result: every game here is cut, a half to each seat
        assert stats["target_sum_2p"] == pytest.approx(1.0)
    plain = ReplayBuffer(100, replay.shapes[0], replay.shapes[1], replay.shapes[2])
    with pytest.raises(ValueError, match="search labels"):
        learn(net, optimizer, plain, train, "cpu", np.random.default_rng(0))


def test_surprise_weighting_draws_the_surprising_positions_more_often(monkeypatch):
    net = network(seed=2)
    _, replay = filled(net, surprise_weight=0.5)
    size = replay.size
    replay.surprise[:size] = 0.0
    replay.surprise[:4] = [3.0, 1.0, 0.0, 4.0]  # half of the draws go to rows 0, 1 and 3
    drawn = []
    real_batch = ReplayBuffer.batch
    monkeypatch.setattr(ReplayBuffer, "batch", lambda self, index, device: drawn.append(np.array(index))
                        or real_batch(self, index, device))
    train = {**TRAIN, "surprise_weight": 0.5, "updates_per_iteration": 40, "minibatch_size": 256}
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)
    rng, twin = np.random.default_rng(3), np.random.default_rng(3)
    stats = learn(net, optimizer, replay, train, "cpu", rng)
    index = np.concatenate(drawn)
    assert len(drawn) == 40 and index.min() >= 0 and index.max() < size
    share = np.bincount(index, minlength=size) / len(index)
    p = surprise_probabilities(replay.surprise[:size], 0.5)
    assert share[:4].tolist() == pytest.approx(p[:4].tolist(), abs=0.02) and p[0] == pytest.approx(0.5 / size + 3 / 16)
    assert share[4:].max() < 3 * 0.5 / size + 0.01
    # the draws: rng.random through the cumulative distribution, one call a minibatch
    want = np.searchsorted(np.cumsum(p), twin.random(256) * np.cumsum(p)[-1], side="right")
    assert np.array_equal(drawn[0], np.minimum(want, size - 1))
    assert stats["surprise_sampled_mean"] == pytest.approx(float((p[index] * size).mean()))
    assert stats["surprise_sampled_max"] == pytest.approx(float(p[3] * size)) and stats["surprise_sampled_mean"] > 5
    assert stats["surprise_ess"] == pytest.approx(1.0 / (size * (p ** 2).sum())) and stats["surprise_ess"] < 0.2
    assert stats["surprise_mean"] == pytest.approx(8.0 / size)
    # a cap: 2 x the mean leaves the three rows level
    drawn.clear()
    capped = learn(net, optimizer, replay, {**train, "surprise_cap": 2.0}, "cpu", np.random.default_rng(3))
    assert capped["surprise_sampled_max"] == pytest.approx(size * (0.5 / size + 0.5 / 3))
    assert capped["surprise_ess"] > stats["surprise_ess"]
    # no surprise in the buffer at all: uniform
    replay.surprise[:size] = 0.0
    flat = learn(net, optimizer, replay, train, "cpu", np.random.default_rng(3))
    assert flat["surprise_ess"] == pytest.approx(1.0) and flat["surprise_sampled_max"] == pytest.approx(1.0)


def test_the_soft_head_learns_the_soft_target():
    net = network(soft=True, seed=4)
    _, replay = filled(net)
    train = {**TRAIN, "soft_policy_coef": 1.0, "updates_per_iteration": 20, "minibatch_size": 32}
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-3)
    rng = np.random.default_rng(1)
    first = learn(net, optimizer, replay, train, "cpu", rng)
    for _ in range(5):
        last = learn(net, optimizer, replay, train, "cpu", rng)
    assert np.isfinite(first["soft_policy_loss"]) and last["soft_policy_loss"] < first["soft_policy_loss"]
    # the soft target is flatter than the policy target, and the loss is at least its entropy
    assert first["soft_target_entropy"] > first["target_entropy"] > 0
    assert last["soft_policy_loss"] >= last["soft_target_entropy"] - 1e-3
    sharper = learn(net, optimizer, replay, {**train, "soft_policy_temperature": 1.5}, "cpu", rng)
    assert sharper["target_entropy"] < sharper["soft_target_entropy"] < first["soft_target_entropy"]
    # the loss by hand, on the minibatch the next call draws
    batch = replay.batch(np.random.default_rng(9).integers(0, replay.size, 32), "cpu")
    has = batch["has_policy"]
    with torch.no_grad():
        extras = net.training_heads(batch["obs"])[3]
    target = batch["policy"][has] / batch["policy"][has].sum(1, keepdim=True)
    legal = batch["obs"][has][:, net.mask_offset : net.mask_offset + net.num_actions] > 0
    soft = soft_policy_target(target, legal, 4.0)
    assert bool(has.any()) and torch.allclose(soft.sum(1), torch.ones(int(has.sum())), atol=1e-5)
    assert not soft[~legal].any() and (extras["soft_policy"][has][~legal] < -1e8).all()
    by_hand = -(soft * F.log_softmax(extras["soft_policy"], dim=1)[has]).sum(1).mean()
    one = learn(net, optimizer, replay, {**train, "updates_per_iteration": 1}, "cpu", np.random.default_rng(9))
    assert one["soft_policy_loss"] == pytest.approx(float(by_hand), rel=1e-5)
    assert one["soft_target_entropy"] == pytest.approx(float(-(soft * soft.clamp_min(1e-12).log()).sum(1).mean()),
                                                       rel=1e-5)
    # at a coefficient of 0 nothing trains the head
    frozen = network(soft=True, seed=2)
    before = {k: v.clone() for k, v in frozen.state_dict().items()}
    learn(frozen, torch.optim.AdamW(frozen.parameters(), lr=3e-3, weight_decay=0.0), replay,
          {**TRAIN, "updates_per_iteration": 2}, "cpu", np.random.default_rng(1))
    after = frozen.state_dict()
    assert all(torch.equal(after[k], before[k]) for k in before if k.startswith("soft_policy_head."))
    assert any(not torch.equal(after[k], before[k]) for k in before if k.startswith("policy_head."))
    # without the head the same call has no such statistics
    bare = network(seed=2)
    plain = learn(bare, torch.optim.AdamW(bare.parameters(), lr=1e-3), replay, TRAIN, "cpu", rng)
    assert not any(key.startswith("soft") for key in plain)


# ---------------------------------------------------------------- a run


ALL = {**PUCT, "value_search_blend": 0.5, "value_blend_iterations": 2, "surprise_weight": 0.5, "surprise_cap": 10.0,
       "soft_policy_coef": 1.0, "soft_policy_temperature": 4.0, "decided_value": 0.5001, "decided_lookback": 2,
       "decided_full_search_prob": 0.1, "decided_fast_sims": 1}


def test_the_trainer_wants_the_head_and_its_coefficient_together():
    v = variant("a", soft=True, soft_policy_coef=1.0)
    stripped = {**v.config, "train": {k: x for k, x in v.config["train"].items() if k != "soft_policy_coef"}}
    bad = Variant("azt", "a", None, {}, stripped, config_hash(stripped))
    with pytest.raises(ValueError, match="soft_policy_coef weighs the soft policy head's loss"):
        run(bad)


def test_a_run_with_all_four_logs_them_and_resumes(monkeypatch):
    """A run without any of them, then one with all four (and the outcome head) from its
    checkpoint: the soft head starts as the policy head, the replay carries the search labels,
    and the blend's ramp goes on across the resume."""
    assert run(variant("plain", dice=False, iterations=1)).status == "finished"
    start = load_experiment("azt/plain").resolve("latest")
    init = {"ckpt": "azt/plain@latest", "parts": ["actor", "critic"], "zero_new_inputs": True}
    v = variant("all", soft=True, outcome=True, init_from=init, iterations=3, outcome_vp_coef=0.5,
                outcome_turns_coef=0.5, **ALL)
    real = RunWriter.should_stop

    def stop_after_two(self):
        if self.launch == 0 and len(canon.read_jsonl(self.path / "metrics.jsonl")) >= 2:
            self.stop_reason = "interrupted"
            return True
        return real(self)

    monkeypatch.setattr(RunWriter, "should_stop", stop_after_two)
    assert run(v, monitors=True).status == "interrupted"
    tricks = load_experiment("azt/all")
    e0 = ck.Checkpoint.load(tricks.checkpoints()[0].path)
    source, warm = ck.Checkpoint.load(start.path).model("all"), e0.model("all")
    theirs, mine = source.state_dict(), warm.state_dict()
    assert warm.soft_policy and warm.outcome and all(torch.equal(mine[k], theirs[k]) for k in theirs)
    soft = [k for k in mine if k.startswith("soft_policy_head.")]
    assert soft and all(torch.equal(mine[k], theirs[k.removeprefix("soft_")]) for k in soft)
    state = torch.load(tricks.path / "trainer_state.pt", weights_only=False)
    replay = state["replay"]
    assert all(name in replay for name in SEARCH_LABELS) and len(replay["surprise"]) == len(replay["obs"]) > 0
    assert replay["search_value"].dtype == replay["surprise"].dtype == torch.float32
    assert bool(replay["has_search_value"].any()) and float(replay["surprise"].max()) > 0.0

    assert run(v, monitors=True).status == "finished"  # the resume
    metrics = load_experiment("azt/all").metrics()
    assert [m["launch"] for m in metrics] == [0, 0, 1]
    learned = [m for m in metrics if "train/policy_loss" in m]
    assert len(learned) == 3
    assert [m["train/value_blend"] for m in learned] == pytest.approx([0.0, 0.25, 0.5])  # iterations 0, 1, 2
    for m in learned:
        for key in ("search_value_gap", "surprise_sampled_mean", "surprise_sampled_max", "surprise_ess",
                    "surprise_mean", "soft_policy_loss", "soft_target_entropy", "outcome_vp_loss"):
            assert np.isfinite(m[f"train/{key}"]), key
        assert 0.0 < m["train/surprise_ess"] <= 1.0 and m["train/surprise_sampled_max"] >= 1.0
        assert m["train/soft_policy_loss"] > 0.0 and m["train/soft_target_entropy"] > 0.0
    for m in metrics:
        assert 0.0 <= m["selfplay/decided"] <= 1.0 and m["selfplay/sims_per_decision"] > 0.0
    assert max(m["selfplay/decided"] for m in metrics) > 0.3
    final = ck.Checkpoint.load(load_experiment("azt/all").resolve("latest").path).model("all")
    moved = [k for k in soft if not torch.equal(final.state_dict()[k], mine[k])]
    assert moved and any(not torch.equal(final.state_dict()[k], final.state_dict()[k.removeprefix("soft_")])
                         for k in soft)  # the head trains, away from the policy head
    names = ("train/value_blend", "train/surprise_ess", "train/soft_policy_loss", "selfplay/decided",
             "selfplay/sims_per_decision", "train/search_value_gap")
    assert not any(key in m for m in load_experiment("azt/plain").metrics() for key in names)
