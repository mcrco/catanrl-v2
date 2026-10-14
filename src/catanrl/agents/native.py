"""Native agents: networks playing cppanatron games in batches.

``make_native_agent(resolved, device=..., infer_rows=...)`` turns a resolved
agent spec (``catanrl.agents.spec.ResolvedAgent``) into one of:

* :class:`PolicyAgent` (``POLICY``): the actor's argmax, or a sample at
  ``temperature`` from a per-decision seed (never torch's global RNG);
* :class:`MctsAgent` (``MCTS``): one SearchPool search per decision, all of a
  batch's decisions in lockstep; ``dets = K`` samples K root worlds from the
  deciding seat's hand tracker, ``oracle = true`` searches the true game;
* :class:`BeliefAgent` (``BELIEF``): the policy averaged over K worlds
  sampled from the deciding seat's hand tracker;
* :class:`BotAgent` (``F``, ``R``): marks a seat the harness plays with the
  engine's bots.

The harness drives every agent the same way: ``bind(dims)`` once (graph
networks are rebuilt at the harness's dims, which leaves their outputs
unchanged), then ``act(rows, masks, decision_seeds, games=..., trackers=...)``
per batch of decisions, returning indices into the
``GraphActionSpace`` of those dims. ``rows`` are encoded at ``agent.level``;
agents with ``needs_games`` also get copies of the games, and those with
``needs_trackers`` the deciding seats' hand trackers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import cppanatron as cpp
import numpy as np
import torch

from catanrl.envs.encoding import action_size, check_dims, critic_level_of, make_encoder, row_layout
from catanrl.models import CatanModel, build_model
from catanrl.search.evaluator import BatchEvaluator
from catanrl.search.pool import make_pool, run_searches, search_config


def _derive_seed(root: int, *labels) -> int:
    try:
        from catanrl.seeding import derive_seed
    except ImportError:  # until catanrl.seeding lands: v1's formula
        from catanrl.envs.puffer import _derive_seed as derive_seed
    return derive_seed(root, *labels)


def _param(params: Any, key: str, default: Any = None) -> Any:
    if isinstance(params, Mapping):
        return params.get(key, default)
    return getattr(params, key, default)


class NativeAgent:
    """Base: a network at some dims and level, evaluated in strict padded batches."""

    key = "NATIVE"
    needs_games = False
    needs_trackers = False

    def __init__(self, model: CatanModel, params: Any, *, device: str, infer_rows: int,
                 mode: str = "strict", heads: str = "both") -> None:
        self.params = params
        self.device = device
        self.infer_rows = int(infer_rows)
        self.mode = mode
        self.heads = heads
        self.level = model.levels["level"]
        if critic_level_of(self.level, model.levels["critic_level"]) is not None and heads == "both":
            raise NotImplementedError(
                "searching with a critic at another level needs an engine encoder that writes the "
                "critic block (VecEnv's critic_players) for SearchPool leaves")
        self._weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        self._spec = model.spec
        self._levels = dict(model.levels)
        self.dims_agnostic = model.dims_agnostic
        self._set_model(model)

    def _set_model(self, model: CatanModel) -> None:
        self.model = model
        self.dims = dict(model.dims)
        self.encoder = make_encoder(self.dims, self.level)
        self.action_space = self.encoder.action_space
        self.evaluator = BatchEvaluator(model, device=self.device, infer_rows=self.infer_rows,
                                        mode=self.mode, heads=self.heads)

    def bind(self, dims: Mapping[str, int]) -> None:
        """Play at ``dims``: rebuild a dims-agnostic network there."""
        dims = check_dims(dims)
        if dims == self.dims:
            return
        if not self.dims_agnostic:
            raise ValueError(f"{self.key}: this network was trained at dims {self.dims} and is "
                             f"not dims-agnostic; the games need {dims}")
        critic = critic_level_of(self._levels["level"], self._levels["critic_level"])
        model = build_model(self._spec, dims=dims, levels=self._levels, num_actions=action_size(dims),
                            layout=row_layout(dims, self.level, critic))
        model.load_state_dict(self._weights)
        self._set_model(model)

    def encode(self, game: cpp.Game, seat: int) -> np.ndarray:
        return self.encoder.encode(game, seat)

    def act(self, rows: np.ndarray, masks: np.ndarray, decision_seeds: np.ndarray, *,
            games: Sequence[cpp.Game] | None = None,
            trackers: Sequence[cpp.HandTracker | None] | None = None) -> np.ndarray:
        raise NotImplementedError


def _choose(logits: np.ndarray, masks: np.ndarray, seeds: np.ndarray, *, sample: bool,
            temperature: float) -> np.ndarray:
    """Argmax over legal actions, or a sample from a per-decision seed."""
    logits = np.where(masks > 0, logits.astype(np.float64), -np.inf)
    if not sample:
        return logits.argmax(axis=1)
    out = np.empty(len(logits), dtype=np.int64)
    for i, (row, seed) in enumerate(zip(logits, seeds)):
        scaled = row / max(float(temperature), 1e-8)
        probs = np.exp(scaled - scaled.max())
        probs /= probs.sum()
        out[i] = np.random.default_rng(int(seed)).choice(len(probs), p=probs)
    return out


class PolicyAgent(NativeAgent):
    key = "POLICY"

    def __init__(self, model, params, *, device, infer_rows, mode="strict") -> None:
        super().__init__(model, params, device=device, infer_rows=infer_rows, mode=mode, heads="policy")
        self.sample = bool(_param(params, "sample", False))
        self.temperature = float(_param(params, "temperature", 1.0))

    def act(self, rows, masks, decision_seeds, *, games=None, trackers=None):
        logits, _ = self.evaluator(rows)
        return _choose(logits, masks, decision_seeds, sample=self.sample, temperature=self.temperature)


class BeliefAgent(NativeAgent):
    key = "BELIEF"
    needs_games = True
    needs_trackers = True

    def __init__(self, model, params, *, device, infer_rows, mode="strict") -> None:
        super().__init__(model, params, device=device, infer_rows=infer_rows, mode=mode, heads="policy")
        self.worlds = int(_param(params, "worlds", 16))
        self.aggregate = str(_param(params, "aggregate", "mean_prob"))
        if self.worlds < 1 or self.aggregate not in ("mean_prob", "mean_logit"):
            raise ValueError("BELIEF needs worlds >= 1 and aggregate mean_prob or mean_logit")

    def act(self, rows, masks, decision_seeds, *, games=None, trackers=None):
        if games is None or trackers is None:
            raise ValueError("BELIEF decides from games and hand trackers")
        k = self.worlds
        world_rows = np.empty((len(games) * k, self.encoder.size), dtype=np.float32)
        for i, (game, tracker, seed) in enumerate(zip(games, trackers, decision_seeds)):
            seat = game.current_player_index
            for w in range(k):
                world = cpp.sample_world(game, tracker, seed=_derive_seed(int(seed), "world", w))
                self.encoder.encode_into(world, seat, world_rows[i * k + w])
        logits, _ = self.evaluator(world_rows)
        logits = logits.reshape(len(games), k, -1).astype(np.float64)
        legal = masks[:, None, :] > 0
        logits = np.where(legal, logits, -np.inf)
        if self.aggregate == "mean_logit":
            score = np.where(masks > 0, np.where(legal, logits, 0.0).mean(axis=1), -np.inf)
        else:
            probs = np.exp(logits - logits.max(axis=2, keepdims=True))
            probs /= probs.sum(axis=2, keepdims=True)
            score = np.where(masks > 0, probs.mean(axis=1), -np.inf)
        return score.argmax(axis=1)


class MctsAgent(NativeAgent):
    key = "MCTS"
    needs_games = True

    def __init__(self, model, params, *, device, infer_rows, mode="strict") -> None:
        super().__init__(model, params, device=device, infer_rows=infer_rows, mode=mode, heads="both")
        self.config = search_config(params, max_players=model.num_players)
        self.needs_trackers = self.config.determinizations > 0
        self.temperature = float(_param(params, "temperature", 0.0))
        self._make_pool()

    def _set_model(self, model) -> None:
        super()._set_model(model)
        if hasattr(self, "config"):
            self._make_pool()

    def _make_pool(self) -> None:
        self.config.max_players = int(self.dims["max_players"])
        self.pool = make_pool(self.dims, self.level, self.config)

    def act(self, rows, masks, decision_seeds, *, games=None, trackers=None):
        if games is None:
            raise ValueError("MCTS searches from games")
        if self.needs_trackers and trackers is None:
            raise ValueError("MCTS with dets searches worlds sampled from hand trackers")
        ids = []
        for i, (game, seed) in enumerate(zip(games, decision_seeds)):
            tracker = trackers[i] if self.needs_trackers else None
            ids.append(self.pool.add(game, seed=int(seed), tracker=tracker))
        try:
            run_searches(self.pool, self.evaluator, batch_rows=len(ids) * self.config.max_pending)
            return np.array([self.pool.select(i, self.temperature) for i in ids], dtype=np.int64)
        finally:
            for i in ids:
                self.pool.remove(i)


class BotAgent:
    """F or R: the harness plays these seats with the engine's bots."""

    needs_games = False
    needs_trackers = False

    def __init__(self, key: str, params: Any = None) -> None:
        if key not in ("F", "R"):
            raise ValueError(f"no native bot {key!r}")
        self.key = key
        enemies = str(_param(params, "enemies", "next") or "next")
        self.value_enemies = {"next": cpp.ValueEnemies.NEXT, "all": cpp.ValueEnemies.ALL}[enemies]

    def bind(self, dims) -> None:
        pass


AGENTS = {"POLICY": PolicyAgent, "MCTS": MctsAgent, "BELIEF": BeliefAgent}


def _load_model(resolved: Any, part: str) -> CatanModel:
    from catanrl.checkpoint import Checkpoint

    ckpt = Checkpoint.load(resolved.checkpoint.path, ref=resolved.checkpoint)
    return ckpt.model(part, device="cpu", allow_drift=bool(_param(resolved.params, "allow_drift", False)))


def make_native_agent(resolved: Any, *, device: str, infer_rows: int, mode: str = "strict",
                      model: CatanModel | None = None) -> NativeAgent | BotAgent:
    """The native backend of a resolved agent spec.

    ``model`` overrides loading the network from ``resolved.checkpoint``
    (tests and in-memory agents). ``mode`` is the evaluator's determinism.
    """
    key = resolved.key
    if key in ("F", "R"):
        return BotAgent(key, resolved.params)
    if key not in AGENTS:
        raise ValueError(f"{key} has no native backend")
    if model is None:
        model = _load_model(resolved, "all" if key == "MCTS" else "actor")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return AGENTS[key](model, resolved.params, device=device, infer_rows=infer_rows, mode=mode)
