"""The PREDICT agent (DESIGN 2.1, 6.5): a public/full policy on the private row
with the other seats' hidden features filled in by a private network's aux head.

The claim it rests on, pinned first: a row at ``public`` or ``full`` is the
private row with exactly the hidden features that level shows, of the other
present seats, filled in (bitwise elsewhere). Then the fill's grid (the
encoder's scale, totals from the game's rules), the oracle equivalence (true
hidden features in the aux head's place give the full POLICY's actions), the
harness (VecEnv rows equal the reference stepper's, invariant to ``num_envs``
and threads), specs (canonical form, both refs pinned, refusals, fairness,
deferred foreign runs), the Catanatron player and ``catanrl eval`` records."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from catanatron.models.player import Color
from catanatron.registry import REGISTRY, SpecError
from eval_helpers import agent, make_suite, schedule
from reference_stepper import play_reference
from test_aux_head import declared_positions
from test_models import BASE_2P, BASE_4P, played_games

from catanrl import games as game_configs
from catanrl.agents import PredictParams, resolve_agent
from catanrl.agents.native import PredictAgent
from catanrl.agents.spec import PredictPlayer, check_spec_syntax, spec_ckpts
from catanrl.checkpoint import _position_game, save
from catanrl.envs.encoding import (
    HIDDEN_FEATURES,
    action_size,
    dims_for,
    hidden_feature_indices,
    make_encoder,
    row_layout,
    shown_hidden_features,
)
from catanrl.eval.harness import play_schedule
from catanrl.models import GraphModel, build_model
from catanrl.models.rows import feature_names

LEVELS = ("private", "public", "full")


def net(level: str, dims, seed: int = 0, *, aux: bool = False, critic: str | None = None):
    """A tiny shared graph network at ``level`` (``aux`` adds the head, its targets at ``critic``)."""
    torch.manual_seed(seed)
    spec = {"preset": "graph-tiny", "args": {"shared": True, "aux": aux}}
    levels = {"level": level, "critic_level": critic or level}
    return build_model(spec, dims=dims, levels=levels, num_actions=action_size(dims),
                       layout=row_layout(dims, level, critic)).eval()


def decisions(names=("base-2p", "base-3p"), count: int = 3, seed: int = 0):
    """Games at a decision (two or more playable actions), assorted plies."""
    rng = np.random.default_rng(seed)
    out = []
    for name in names:
        for i in range(count):
            game = game_configs.load_game(name).cppanatron_game(seed + i, seed + i)
            for _ in range(int(rng.integers(0, 90))):
                if game.winning_color is not None:
                    break
                actions = game.playable_actions
                game.execute(actions[int(rng.integers(len(actions)))])
            while len(game.playable_actions) < 2:
                game.execute(game.playable_actions[0])
            out.append(game)
    return out


def phase_games(name: str, seed: int = 3) -> list:
    """Snapshots of random games in every phase: fresh and mid initial placement, play turns,
    the robber prompt, a discard, road building, late game."""
    config = game_configs.load_game(name)
    out, flags = [], {"robber": 0, "discard": 0, "road": 0}
    rng = np.random.default_rng(seed)
    for i in range(8):
        game = config.cppanatron_game(map_seed=seed + i, seed=seed + i)
        out.append(game.copy())
        for ply in range(260):
            if game.winning_color is not None:
                break
            if ply in (3, 40, 150):
                out.append(game.copy())
            for flag, hit in (("robber", game.is_moving_robber), ("discard", game.is_discarding),
                              ("road", game.is_road_building)):
                if hit and flags[flag] < 3:
                    flags[flag] += 1
                    out.append(game.copy())
            actions = game.playable_actions
            game.execute(actions[int(rng.integers(len(actions)))])
    assert flags["robber"] and flags["discard"], flags
    return out


def players_block(rows: np.ndarray, dims):
    offset, (players, width) = row_layout(dims, "private")["players"]
    return rows[:, offset : offset + players * width].reshape(len(rows), players, width)


def hidden_truth(full_rows: np.ndarray, dims) -> np.ndarray:
    """[B, P-1, 15]: the other slots' hidden features as a full row holds them."""
    names = feature_names(dims, "full")["players"]
    return players_block(full_rows, dims)[:, 1:][:, :, hidden_feature_indices(names)]


# ---------------------------------------------------------------- the claim


@pytest.mark.parametrize(("name", "dims"), [("base-2p", BASE_2P), ("base-4p", BASE_4P)])
def test_a_level_row_is_the_private_row_plus_the_hidden_features_it_shows(name, dims):
    """For every seat of games in several phases (initial placement, mid-game, robber and
    discard prompts, declared hands and dev cards), the rows at ``public`` and ``full`` differ
    from the private row only at the hidden features that level shows, in the other present
    seats' slots: so ``private + fill == level`` exactly. The layouts are the same at every
    level, and the set of differing features is exactly ``shown_hidden_features``."""
    games = [_position_game(p) for p in declared_positions(name)] + phase_games(name) + played_games(name)
    assert any(g.is_initial_build_phase for g in games) and any(g.is_discarding for g in games)
    encoders = {level: make_encoder(dims, level) for level in LEVELS}
    assert all(encoders[level].layout() == encoders["private"].layout() for level in LEVELS)
    names = feature_names(dims, "private")["players"]
    assert all(feature_names(dims, level) == feature_names(dims, "private") for level in LEVELS)
    offset, (players, width) = row_layout(dims, "private")["players"]
    hidden = hidden_feature_indices(names)
    present_at = names.index("present")
    rows = {level: np.stack([encoders[level].encode(g, seat) for g in games for seat in range(g.num_players)])
            for level in LEVELS}
    private = rows["private"]
    for level in ("public", "full"):
        shown = shown_hidden_features(level)
        assert shown == (tuple(f for f in HIDDEN_FEATURES if f.startswith("resource_")) if level == "public"
                         else HIDDEN_FEATURES)
        columns = [hidden[k] for k, feature in enumerate(HIDDEN_FEATURES) if feature in shown]
        other = rows[level]
        block_private, block_other = players_block(private, dims), players_block(other, dims)
        # Every differing column of the whole row is a shown hidden feature of another present seat.
        differing = private != other
        allowed = np.zeros_like(differing)
        present = block_private[:, :, present_at] > 0
        for slot in range(1, players):
            for column in columns:
                allowed[:, offset + slot * width + column] = present[:, slot]
        assert not (differing & ~allowed).any(), "a level changes more than the hidden features"
        # The claim is not vacuous: every shown feature differs for some other seat, nothing else does.
        seen = {names[i] for i in np.flatnonzero((block_private[:, 1:] != block_other[:, 1:]).any(axis=(0, 1)))}
        assert seen == set(shown)
        assert np.array_equal(block_private[:, 0], block_other[:, 0])            # the own seat
        assert np.array_equal(private[:, : offset], other[:, : offset])          # the global block
        assert np.array_equal(private[:, offset + players * width :], other[:, offset + players * width :])
        # private + fill == level, bitwise, filling only present other seats.
        filled = private.copy()
        block = players_block(filled, dims)
        for slot in range(1, players):
            for column in columns:
                block[:, slot, column] = np.where(present[:, slot], block_other[:, slot, column], block[:, slot, column])
        assert np.array_equal(filled, other)
    assert shown_hidden_features("private") == ()


# ---------------------------------------------------------------- the grid


def test_round_snaps_predictions_onto_the_encoders_grid():
    """The hidden features are counts over the rules' totals (bank per resource, deck per kind,
    points to win; the engine's double ratio cast to float32) and 0/1 flags: declared hands
    land on that grid, a prediction within half a step of the truth rounds back to it
    bitwise, and out-of-range values clamp. The totals come from each game's rules."""
    import cppanatron as cpp

    rules = cpp.GameRules(resource_bank=(5, 7, 9, 11, 13), development_deck=(3, 1, 1, 1, 2), vps_to_win=12)
    variant = cpp.Game(2, rules=rules, seed=4)
    standard = game_configs.load_game("base-2p").cppanatron_game(5, 5)
    for game in (variant, standard):
        while game.is_initial_build_phase:
            game.execute(game.playable_actions[0])
    dims = BASE_2P
    encoder = make_encoder(dims, "full")
    hands, decks = [[2, 1, 0, 1, 3], [0, 5, 1, 1, 0]], [[1, 1, 0, 1, 0], [2, 0, 1, 0, 0]]
    truths = []
    for game, hand, deck in zip((variant, standard), hands, decks):
        other = (game.current_player_index + 1) % 2
        game.set_resources(other, hand)
        game.set_development_cards(other, deck)
        row = encoder.encode(game, game.current_player_index)[None]
        truth = hidden_truth(row, dims)[0, 0]                                   # slot 1's 15 features
        totals = [game.vps_to_win, *game.rules.resource_bank, *game.rules.development_deck]
        counts = [game.player(other).actual_victory_points, *hand, *deck]
        expected = [np.float32(np.float64(c) / np.float64(t)) for c, t in zip(counts, totals)]
        expected += [1.0 if n > 0 else 0.0 for n in deck[:4]]                   # playable this turn
        assert truth.tolist() == expected, (truth.tolist(), expected)
        assert game.rules.resource_bank == tuple(rules.resource_bank) or game is standard
        truths.append((game, truth))
    games = [g for g, _ in truths]
    truth = np.stack([t for _, t in truths])[:, None, :]                        # [2, 1, 15]
    assert np.array_equal(PredictAgent.round_predictions(truth, games), truth)
    rng = np.random.default_rng(0)
    for game_index, game in enumerate(games):
        totals = np.array([game.vps_to_win, *game.rules.resource_bank, *game.rules.development_deck], float)
        step = np.concatenate([0.49 / totals, [0.49] * 4])
        noisy = truth.copy()
        noisy[game_index, 0] += rng.uniform(-1, 1, 15) * step
        rounded = PredictAgent.round_predictions(noisy.astype(np.float32), games)
        assert np.array_equal(rounded[game_index], truth[game_index])
    wild = np.array([[[1.7, -0.3, 2.0, 0.5, -1.0, 0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.49, 0.5, 2.0, -5.0]]], np.float32)
    rounded = PredictAgent.round_predictions(wild, [standard])[0, 0]
    assert rounded[0] == 1.0 and rounded[1] == 0.0 and rounded[2] == 1.0 and rounded[4] == 0.0
    assert rounded[3] == np.float32(np.rint(0.5 * 19) / 19) and rounded[5] == np.float32(19 / 19)
    assert rounded[11:].tolist() == [0.0, 1.0, 1.0, 0.0]
    with pytest.raises(ValueError, match="rules"):
        agent("PREDICT", net("full", dims), predictor=net("private", dims, aux=True, critic="full"),
              round=True).policy_rows(encoder.encode(standard, 0)[None])


# ---------------------------------------------------------------- oracle equivalence


def _oracle(predict_agent: PredictAgent, private: np.ndarray, truth: np.ndarray, monkeypatch, calls: list):
    """The predictor's aux head answers with the true hidden features of each private row (the
    rows reach it one at a time, or padded with copies, so they are looked up by content)."""
    by_row = {private[i].tobytes(): torch.from_numpy(np.ascontiguousarray(truth[i], dtype=np.float32))
              for i in range(len(private))}

    def oracle(obs):
        calls.append(len(obs))
        return torch.stack([by_row[row.cpu().numpy().tobytes()] for row in obs])

    monkeypatch.setattr(predict_agent.predictor_evaluator.model, "aux_predictions", oracle)


@pytest.mark.parametrize("level", ["full", "public"])
def test_oracle_predictions_reproduce_the_level_policy(level, monkeypatch):
    """With the aux head's output replaced by the true hidden features, PREDICT builds exactly
    the policy's level row (a 2p game with an absent seat, 3p games) and chooses what the
    POLICY agent of the same network chooses on that row: argmax, sampled, and with round=true
    (the truth is on the grid). This proves the fill and the plumbing; only the predictor's
    accuracy is left to the experiment."""
    dims = dims_for(["base-2p", "base-3p"])
    policy = net(level, dims, seed=1)
    predictor = net("private", dims, seed=2, aux=True, critic="full")
    games = decisions()
    private = np.stack([make_encoder(dims, "private").encode(g, g.current_player_index) for g in games])
    target = np.stack([make_encoder(dims, level).encode(g, g.current_player_index) for g in games])
    truth = hidden_truth(np.stack([make_encoder(dims, "full").encode(g, g.current_player_index) for g in games]),
                         dims)
    masks = private[:, -action_size(dims):]
    seeds = np.arange(len(games), dtype=np.uint64) + 11
    for params in ({}, {"round": True}, {"sample": True, "temperature": 0.7}):
        predict = agent("PREDICT", policy, predictor=predictor, infer_rows=4, **params)
        assert predict.level == "private" and predict.policy_level == level and predict.needs_games
        assert not predict.needs_trackers and predict.model.obs_size == predict.encoder.size
        # Before the oracle: a random predictor changes only the shown hidden features of present seats.
        differing = predict.policy_rows(private, games) != private
        offset, players, width = predict.players_span
        allowed = np.zeros_like(differing)
        present = players_block(private, dims)[:, :, predict.present_index] > 0
        for slot in range(1, players):
            for k in predict.fill_features:
                allowed[:, offset + slot * width + predict.hidden_index[k]] = present[:, slot]
        assert not (differing & ~allowed).any() and differing.any()
        calls: list[int] = []
        _oracle(predict, private, truth, monkeypatch, calls)
        filled = predict.policy_rows(private, games)
        assert np.array_equal(filled, target) and sum(calls) == len(games)
        reference = agent("POLICY", policy, infer_rows=4, **{k: v for k, v in params.items() if k != "round"})
        chosen = predict.act(private, masks, seeds, games=games)
        assert chosen.tolist() == reference.act(target, masks, seeds).tolist()
        assert all(masks[i, c] > 0 for i, c in enumerate(chosen))
    with pytest.raises(ValueError, match="games"):
        predict.act(private, masks, seeds)


def test_present_seats_only_and_the_other_slot_untouched():
    """In a 2p game at 3-seat dims the absent slot keeps its zeros whatever the head predicts,
    and the own slot is never written."""
    dims = dims_for(["base-2p", "base-3p"])
    predict = agent("PREDICT", net("full", dims), predictor=net("private", dims, aux=True, critic="full"))
    game = decisions(("base-2p",), count=1)[0]
    private = predict.encode(game, game.current_player_index)[None]
    predict.predict = lambda rows: np.full((1, 2, 15), 0.5, np.float32)
    filled = predict.policy_rows(private, [game])
    block, before = players_block(filled, dims)[0], players_block(private, dims)[0]
    assert np.array_equal(block[0], before[0]) and np.array_equal(block[2], before[2]) and not block[2].any()
    assert (block[1, predict.hidden_index] == 0.5).all()


# ---------------------------------------------------------------- the harness


def test_vecenv_rows_equal_the_reference_and_are_invariant():
    """PREDICT through play_schedule gives exactly the reference stepper's rows (which encodes
    its own private rows), whether VecEnv encodes at its level (PREDICT first) or the agent
    re-encodes (a full POLICY first), for any num_envs and threads; round=true plays too."""
    dims = dims_for(["base-2p", "base-3p"])
    policy = net("full", dims, seed=3)
    predictor = net("private", dims, seed=4, aux=True, critic="full")
    suite = make_suite(({"game": "base-2p"}, {"game": "base-3p"}), turn_limit=120)
    games = schedule(suite, 2, boards=2)

    def play(lineup, **kwargs):
        rows = []
        play_schedule(lineup, games, suite=suite, fill=agent("F"), on_game=rows.append, device="cpu", **kwargs)
        return {r["game_id"]: r for r in rows}

    def predict(**params):
        return agent("PREDICT", policy, predictor=predictor, infer_rows=8, **params)

    first = play([predict(), agent("F")], num_envs=4, threads=2)
    assert sorted(first) == sorted(g.game_id for g in games) and len(first) == 10
    reference = {r["game_id"]: r for r in play_reference([predict(), agent("F")], games, suite=suite, fill=agent("F"))}
    assert first == reference
    assert play([predict(), agent("F")], num_envs=1, threads=1) == first
    second = play([agent("POLICY", policy, infer_rows=8), predict()], num_envs=3, threads=2)
    assert second == {r["game_id"]: r for r in play_reference([agent("POLICY", policy, infer_rows=8), predict()],
                                                              games, suite=suite, fill=agent("F"))}
    rounded = play([predict(round=True), agent("F")], num_envs=2, threads=1)
    assert sorted(rounded) == sorted(first)


# ---------------------------------------------------------------- specs


@pytest.fixture
def ckpts(tmp_path, monkeypatch):
    """Free-standing checkpoints per kind, plus runs exp/pol (a full policy, best e3) and exp/aux
    (a private aux network, best e2) in a private runs dir."""
    from types import SimpleNamespace

    from catanrl.config import Variant, config_hash, resolve
    from catanrl.runs import RunWriter

    monkeypatch.setenv("CATANRL_RUNS_DIR", str(tmp_path / "runs"))
    dims = BASE_2P
    nets = {
        "full": net("full", dims, 1), "public": net("public", dims, 2), "private": net("private", dims, 3),
        "aux": net("private", dims, 4, aux=True, critic="full"), "aux-full": net("full", dims, 5, aux=True),
    }
    paths = {name: save(tmp_path / f"{name}.pt", model, run=None, epoch=0, agent_steps=0,
                        config={"model": model.spec}, code={}).path for name, model in nets.items()}
    refs = {}
    for variant, (level, critic, aux) in {"pol": ("full", None, False), "aux": ("private", "full", True)}.items():
        raw = {"seed": 1, "env": {"scenarios": [{"game": "base-2p"}]}, "obs": {"level": level, "critic_level": critic},
               "model": {"preset": "graph-tiny", "args": {"shared": True, "aux": aux}}}
        cfg = resolve(raw, experiment="exp", variant=variant)
        with RunWriter.open(Variant("exp", variant, None, {}, cfg, config_hash(cfg)), argv=["t"]) as w:
            w.save_checkpoint(net(level, dims, 6, aux=aux, critic=critic), epoch=0, agent_steps=0)
            w.save_trainer_state({"epoch": 0})
            best = 3 if variant == "pol" else 2
            ref = w.save_checkpoint(net(level, dims, 7 + best, aux=aux, critic=critic), epoch=best, agent_steps=30)
            w.save_trainer_state({"epoch": best})
            w.set_alias("best", ref, record=SimpleNamespace(eval_id="x", summary={"score_rate": 0.5}))
            w.finish("finished")
        refs[variant] = ref
    return {"paths": paths, "refs": refs}


def test_specs_pin_both_checkpoints_and_round_trip(ckpts):
    pol, aux = ckpts["refs"]["pol"], ckpts["refs"]["aux"]
    resolved = resolve_agent("PREDICT:ckpt=exp/pol:predictor=exp/aux")
    assert resolved.canonical == (f"PREDICT:ckpt=exp/pol@e3#{pol.weights_sha256[:12]}:predictor=exp/aux@e2#"
                                  f"{aux.weights_sha256[:12]}:round=false:sample=false:temperature=1.0"
                                  ":allow_drift=false")
    assert resolve_agent(resolved.canonical).canonical == resolved.canonical
    assert resolved.fair and str(resolved.checkpoint) == str(pol) and str(resolved.predictor) == str(aux)
    assert resolved.checkpoint.path == pol.path and resolved.predictor.path == aux.path
    assert resolved.checkpoints == (resolved.checkpoint, resolved.predictor)
    assert resolved.label == "PREDICT:exp/pol@e3+exp/aux@e2"
    assert isinstance(resolved.params, PredictParams)
    assert "PREDICT" in REGISTRY
    key = resolved.key_dict()
    assert key["checkpoint"]["weights_sha256"] == pol.weights_sha256
    assert key["predictor"]["weights_sha256"] == aux.weights_sha256 and key["predictor"]["levels"]["level"] == "private"
    assert key["params"] == {"round": False, "sample": False, "temperature": 1.0, "allow_drift": False}
    other = resolve_agent("PREDICT:ckpt=exp/pol@best:predictor=exp/aux@e0:device=cpu")
    assert other.key_dict() != key and "device" not in other.canonical and other.predictor.epoch == 0
    assert resolve_agent("PREDICT:ckpt=exp/pol:predictor=exp/aux:round=true").key_dict()["params"]["round"] is True
    pinned = resolve_agent(f"PREDICT:ckpt=exp/pol@e3#{pol.weights_sha256[:8]}:predictor=exp/aux@best#"
                           f"{aux.weights_sha256[:8]}")
    assert pinned.canonical == resolved.canonical
    assert spec_ckpts("PREDICT:ckpt=exp/pol@best:predictor=exp/aux@best:round=true") == ["exp/pol@best", "exp/aux@best"]
    assert spec_ckpts("BELIEF:ckpt=exp/pol") == ["exp/pol"] and spec_ckpts("F") == []
    check_spec_syntax("PREDICT:ckpt=exp/pol@best:predictor=${params.predictor}")  # a plan's unexpanded value
    check_spec_syntax("PREDICT:ckpt=${run}@best:predictor=exp/aux@best")
    with pytest.raises(SpecError):
        check_spec_syntax("PREDICT:ckpt=exp/pol@best:predictor=exp/aux@best:worlds=3")
    with pytest.raises(SpecError, match="predictor"):
        check_spec_syntax("PREDICT:ckpt=exp/pol@best")
    public = resolve_agent(f"PREDICT:ckpt={ckpts['paths']['public']}:predictor=exp/aux")
    assert public.fair and public.checkpoint.run is None and public.canonical.startswith("PREDICT:ckpt=/")


def test_bad_pairs_and_harnesses_are_refused(ckpts, monkeypatch):
    paths = ckpts["paths"]
    for spec, reason in (
        (f"PREDICT:ckpt=exp/pol:predictor={paths['private']}", "no aux head"),
        (f"PREDICT:ckpt=exp/pol:predictor={paths['aux-full']}", "reads full rows"),
        (f"PREDICT:ckpt={paths['private']}:predictor=exp/aux", "private actor"),
        ("PREDICT:ckpt=exp/pol", "needs a predictor"),
        ("PREDICT:ckpt=exp/pol:predictor=", "needs a predictor"),
        ("PREDICT:ckpt=exp/pol:predictor=exp/aux:round=maybe", "round"),
        ("PREDICT:ckpt=exp/pol:predictor=exp/nope", "nope"),
        ("PREDICT:ckpt=exp/pol:predictor=exp/aux@e9", "e9"),
    ):
        with pytest.raises(SpecError, match=reason):
            resolve_agent(spec)
    with pytest.raises(SpecError, match="server"):
        resolve_agent("PREDICT:ckpt=exp/pol:predictor=exp/aux", harness="server")
    assert isinstance(resolve_agent("PREDICT:ckpt=exp/pol:predictor=exp/aux", harness="catanatron").player(Color.RED),
                      PredictPlayer)
    # The predictor is self-tested on its aux part: a drifted head is refused at resolution.
    from catanrl.agents import spec as spec_module

    original = GraphModel.aux_predictions
    spec_module._tested_checkpoint.cache_clear()
    GraphModel.aux_predictions = lambda self, obs: original(self, obs) + 0.01
    try:
        with pytest.raises(SpecError, match="aux"):
            resolve_agent("PREDICT:ckpt=exp/pol:predictor=exp/aux@e0")
    finally:
        GraphModel.aux_predictions = original
        spec_module._tested_checkpoint.cache_clear()
    resolve_agent("PREDICT:ckpt=exp/pol:predictor=exp/aux@e0")


def test_foreign_runs_of_both_checkpoints_defer_the_final(ckpts, tmp_path):
    """A final's PREDICT naming runs of other experiments is pinned at creation only when both
    are usable; otherwise it waits (Unbound names the missing one), like a BELIEF's control."""
    from catanrl import runs
    from catanrl.config import Variant, config_hash, resolve
    from catanrl.runs import RunWriter, bind_refs, bound_lineup, load_experiment

    pol, aux = ckpts["refs"]["pol"], ckpts["refs"]["aux"]
    assert runs.spec_runs("PREDICT:ckpt=exp/pol@best:predictor=ctl/main@best") == ["exp/pol", "ctl/main"]
    assert runs.spec_runs("PREDICT:ckpt=exp/pol@best:predictor=${params.p}") == []
    assert runs.foreign_runs("PREDICT:ckpt=exp/pol@best:predictor=ctl/main@best", "exp") == ["ctl/main"]
    waiting = "PREDICT:ckpt=exp/pol@best:predictor=ctl/main@best"
    ready = "PREDICT:ckpt=exp/pol@best:predictor=exp/aux@best"
    ev = {"final": [{"suite": {"name": "heldout-2p", "boards": 1}, "lineup": [waiting, "F"]},
                    {"suite": {"name": "heldout-2p", "boards": 1}, "lineup": [ready, "F"]}]}
    cfg = resolve({"kind": "eval", "eval": ev}, experiment="other", variant="v")
    bound = bind_refs(cfg, "other/v")["specs"]
    assert waiting not in bound and bound[ready].startswith(f"PREDICT:ckpt={pol}:predictor={aux}")
    with RunWriter.open(Variant("other", "v", None, {}, cfg, config_hash(cfg)), argv=["t"]) as w:
        w.finish("finished")
    run = load_experiment("other/v")
    assert bound_lineup(run, [ready, "F"], pin=False)[0] == bound[ready]
    with pytest.raises(runs.Unbound, match="waits for ctl/main, which is not there yet"):
        bound_lineup(run, [waiting, "F"], pin=False)


# ---------------------------------------------------------------- the players and catanrl eval


def test_a_whole_catanatron_game_stays_in_sync(ckpts):
    from catanrl.agents import player

    config = game_configs.load_game("base-2p")
    players = [player("PREDICT:ckpt=exp/pol:predictor=exp/aux:round=true", Color.RED), REGISTRY.build("R", Color.BLUE)]
    game = config.catanatron_game(players, map_seed=3, seed=5)
    game.play()
    assert game.state.num_turns > 10 and len(players[0]._mirror["records"]) > 50
    native = players[0]._agent()
    assert isinstance(native, PredictAgent) and native.round and native.predictor.aux


def test_catanrl_eval_records_both_checkpoints_as_fair(ckpts, monkeypatch, capsys):
    """A record written by ``catanrl eval`` (native, then the catanatron harness) marks the
    PREDICT slot fair, pins the policy in ``refs`` and the predictor in ``predictors`` (both in
    the key, so the eval id moves when either does), lands under the policy's run, and is
    cached."""
    from catanrl.cli import main
    from catanrl.eval import results
    from catanrl.runs import runs_root

    pol, aux = ckpts["refs"]["pol"], ckpts["refs"]["aux"]
    spec = "PREDICT:ckpt=exp/pol@best:predictor=exp/aux@best"
    assert main(["eval", spec, "F", "--suite", "heldout-2p", "--boards", "2", "--threads", "1", "--num-envs", "2",
                 "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "PREDICT:exp/pol@e3+exp/aux@e2" in out
    (path,) = (runs_root() / "exp" / "pol" / "evals" / "heldout-2p").glob("*.json")
    record = results.EvalRecord.load(path)
    assert record.data["fair"] == [True, False] and record.summary["games"] == 4
    assert record.data["refs"] == [str(pol), None] and record.data["predictors"] == [str(aux), None]
    assert record.data["specs"][0] == resolve_agent(spec).canonical
    slot = record.key["lineup"][0]
    assert slot["checkpoint"]["weights_sha256"] == pol.weights_sha256
    assert slot["predictor"]["weights_sha256"] == aux.weights_sha256
    other = results.find([spec.replace("exp/aux@best", "exp/aux@e0"), "F"], {"name": "heldout-2p", "boards": 2})
    assert other is None  # another predictor is another record

    def no_play(*args, **kwargs):
        raise AssertionError("a cached eval replayed games")

    play = results.play_schedule
    monkeypatch.setattr(results, "play_schedule", no_play)
    assert main(["eval", spec, "F", "--suite", "heldout-2p", "--boards", "2", "--device", "cpu"]) == 0
    monkeypatch.setattr(results, "play_schedule", play)
    assert main(["eval", spec, "F", "--suite", "heldout-2p", "--boards", "1", "--threads", "1", "--harness",
                 "catanatron"]) == 0
    paths = sorted((runs_root() / "exp" / "pol" / "evals" / "heldout-2p").glob("*.json"))
    assert len(paths) == 2
    catanatron = next(results.EvalRecord.load(p) for p in paths if p != path)
    assert catanatron.key["harness"] == "catanatron" and catanatron.data["fair"] == [True, False]
    assert catanatron.data["predictors"] == [str(aux), None] and catanatron.summary["games"] == 2
