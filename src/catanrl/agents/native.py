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
* :class:`PredictAgent` (``PREDICT``): a public/full policy on one row, the
  private row with the other seats' hidden features filled in by a private
  network's aux head (DESIGN 2.1);
* :class:`BotAgent` (``F``, ``R``): marks a seat the harness plays with the
  engine's bots.

Agents that sample worlds have a ``tracker_mode`` (``cpp.TrackerMode``, from
the ``tracker`` param): the harness keeps VecEnv's trackers in exact mode when
any agent asks for it, and every agent passes its own mode as the deal, so a
pool agent deals the pool from an exact tracker and an exact agent given a
pool tracker fails loudly rather than silently degrading.

The harness drives every agent the same way: ``bind(dims)`` once (graph
networks are rebuilt at the harness's dims, which leaves their outputs
unchanged), then ``act(rows, masks, decision_seeds, games=..., trackers=...)``
per batch of decisions, returning indices into the
``GraphActionSpace`` of those dims. ``rows`` are encoded at ``agent.level``;
agents with ``needs_games`` also get copies of the games, and those with
``needs_trackers`` the deciding seats' hand trackers. A recurrent POLICY
(``needs_memory``) also gets ``streams`` (an id per row that is stable for
one seat of one game, e.g. ``env * max_players + seat``) and ``first``
(whether this is that seat's first decision of the game): its memory is the
sequence of a seat's own decisions, zero at the first, so rows are the same
whatever env slot a game plays in. MCTS and BELIEF refuse a recurrent
network: a leaf or a sampled world has no decision history of its own.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import cppanatron as cpp
import numpy as np
import torch

from catanrl.envs.encoding import (
    HIDDEN_FEATURES,
    action_size,
    check_dims,
    critic_level_of,
    hidden_feature_indices,
    make_encoder,
    row_layout,
    shown_hidden_features,
)
from catanrl.models import CatanModel, build_model
from catanrl.models.rows import feature_names
from catanrl.search.dedup import DedupEvaluator
from catanrl.search.evaluator import BatchEvaluator
from catanrl.search.pool import make_pool, run_searches, search_config
from catanrl.seeding import derive_seed as _derive_seed


def _param(params: Any, key: str, default: Any = None) -> Any:
    if isinstance(params, Mapping):
        return params.get(key, default)
    return getattr(params, key, default)


def tracker_mode_of(params: Any) -> cpp.TrackerMode:
    """The engine tracker mode of an agent's ``tracker`` param (default pool)."""
    modes = {"pool": cpp.TrackerMode.POOL, "exact": cpp.TrackerMode.EXACT}
    mode = str(_param(params, "tracker", "pool") or "pool")
    if mode not in modes:
        raise ValueError(f"tracker must be pool or exact, not {mode!r}")
    return modes[mode]


class NativeAgent:
    """Base: a network at some dims and level, evaluated in strict padded batches."""

    key = "NATIVE"
    needs_games = False
    needs_trackers = False
    tracker_mode = cpp.TrackerMode.POOL
    needs_memory = False

    def __init__(self, model: CatanModel, params: Any, *, device: str, infer_rows: int,
                 mode: str = "strict", heads: str = "both", reads_level: str | None = None) -> None:
        self.params = params
        self.device = device
        self.infer_rows = int(infer_rows)
        self.mode = mode
        self.heads = heads
        #: The level of the rows the agent reads (``act``'s ``rows``, ``encode``): the
        #: network's, unless the agent builds the network's rows itself (PREDICT).
        self.level = reads_level or model.levels["level"]
        self._weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        self._spec = model.spec
        self._levels = dict(model.levels)
        self.dims_agnostic = model.dims_agnostic
        self.recurrent = bool(getattr(model, "recurrent", False))
        if self.recurrent and not self.needs_memory:
            raise ValueError(f"{self.key} cannot play a recurrent network (class {model.spec['class']}): "
                             "only POLICY keeps a seat's decision history")
        if critic_level_of(self._levels["level"], self._levels["critic_level"]) is not None and (
                heads == "policy" or not model.reads_critic_block):
            # Decisions read only the actor: drop the critic block (it has the players block's
            # width, so the same weights load) and play from actor-level rows. Searches keep it
            # when the critic reads it (their leaves come from an ActorCriticEncoder); a block
            # no head reads (a shared trunk's aux targets) is dropped for searches too.
            self._levels["critic_level"] = self._levels["level"]
            model = self._rebuild(dict(model.dims))
        self._set_model(model)

    def _rebuild(self, dims: dict) -> CatanModel:
        critic = critic_level_of(self._levels["level"], self._levels["critic_level"])
        model = build_model(self._spec, dims=dims, levels=self._levels, num_actions=action_size(dims),
                            layout=row_layout(dims, self._levels["level"], critic))
        model.load_state_dict(self._weights)
        return model

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
        self._set_model(self._rebuild(dims))

    def check_maps(self, maps: Iterable[Any]) -> None:
        """Refuse maps the network would misread at these dims (the cnn's board check)."""
        self.model.check_maps(maps)

    def encode(self, game: cpp.Game, seat: int) -> np.ndarray:
        return self.encoder.encode(game, seat)

    def act(self, rows: np.ndarray, masks: np.ndarray, decision_seeds: np.ndarray, *,
            games: Sequence[cpp.Game] | None = None,
            trackers: Sequence[cpp.HandTracker | None] | None = None,
            streams: np.ndarray | None = None, first: np.ndarray | None = None) -> np.ndarray:
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
        self.needs_memory = bool(getattr(model, "recurrent", False))
        super().__init__(model, params, device=device, infer_rows=infer_rows, mode=mode, heads="policy")
        self.sample = bool(_param(params, "sample", False))
        self.temperature = float(_param(params, "temperature", 1.0))
        #: stream id -> (h, c) after the seat's last decision; entries of finished games are
        #: overwritten at the next game's first decision in that stream.
        self._memory: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def act(self, rows, masks, decision_seeds, *, games=None, trackers=None, streams=None, first=None):
        if not self.needs_memory:
            logits, _ = self.evaluator(rows)
        else:
            logits = self._recall(rows, streams, first)
        return _choose(logits, masks, decision_seeds, sample=self.sample, temperature=self.temperature)

    def _recall(self, rows: np.ndarray, streams, first) -> np.ndarray:
        """Decide from each row's memory and keep the stepped memory."""
        if streams is None or first is None:
            raise ValueError("POLICY with a recurrent network decides from streams and first (its rows' memory)")
        streams = np.asarray(streams, dtype=np.int64).reshape(-1)
        first = np.asarray(first, dtype=bool).reshape(-1)
        if len(streams) != len(rows) or len(first) != len(rows):
            raise ValueError("streams and first need one entry per row")
        if len(set(streams.tolist())) != len(streams):
            raise ValueError("a stream decides once per batch")
        size = self.evaluator.hidden_size
        h = np.zeros((len(rows), size), dtype=np.float32)
        c = np.zeros((len(rows), size), dtype=np.float32)
        for i, (stream, fresh) in enumerate(zip(streams.tolist(), first.tolist())):
            if not fresh:
                if stream not in self._memory:
                    raise ValueError(f"stream {stream} has no memory: its first decision was not marked first")
                h[i], c[i] = self._memory[stream]
        logits, _, h, c = self.evaluator.step(rows, h, c)
        for i, stream in enumerate(streams.tolist()):
            self._memory[stream] = (h[i].copy(), c[i].copy())
        return logits


class BeliefAgent(NativeAgent):
    key = "BELIEF"
    needs_games = True
    needs_trackers = True

    def __init__(self, model, params, *, device, infer_rows, mode="strict") -> None:
        super().__init__(model, params, device=device, infer_rows=infer_rows, mode=mode, heads="policy")
        self.worlds = int(_param(params, "worlds", 16))
        self.aggregate = str(_param(params, "aggregate", "mean_prob"))
        self.tracker_mode = tracker_mode_of(params)
        if self.worlds < 1 or self.aggregate not in ("mean_prob", "mean_logit"):
            raise ValueError("BELIEF needs worlds >= 1 and aggregate mean_prob or mean_logit")

    def act(self, rows, masks, decision_seeds, *, games=None, trackers=None, streams=None, first=None):
        if games is None or trackers is None:
            raise ValueError("BELIEF decides from games and hand trackers")
        k = self.worlds
        world_rows = np.empty((len(games) * k, self.encoder.size), dtype=np.float32)
        for i, (game, tracker, seed) in enumerate(zip(games, trackers, decision_seeds)):
            seat = game.current_player_index
            for w in range(k):
                world = cpp.sample_world(game, tracker, seed=_derive_seed(int(seed), "world", w),
                                         deal=self.tracker_mode)
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


class PredictAgent(NativeAgent):
    """``PREDICT``: the policy (a ``public`` or ``full`` network) decides on the private row of
    the deciding seat with, for every other present seat, the hidden features its level
    shows (:func:`~catanrl.envs.encoding.shown_hidden_features`: the hand per resource for
    ``public``, all 15 for ``full``) overwritten by the predictor's aux head, run on that same
    private row. ``private + fill == <level>`` row for row (pinned by test), so the policy
    sees exactly the row it was trained on, with predictions in place of the truth.

    ``round=true`` snaps each prediction onto the encoder's grid: a count over the rules'
    total (the bank per resource, the deck per kind, the points to win, read from each game's
    rules as the encoder does), the playable flags to 0/1. Flags are rounded independently of
    the counts (a predicted playable card with a predicted count of 0 stays as predicted).
    """

    key = "PREDICT"
    needs_games = True

    def __init__(self, model, params, *, device, infer_rows, mode="strict", predictor: CatanModel | None = None
                 ) -> None:
        if predictor is None:
            raise ValueError("PREDICT needs its predictor network (the aux-head checkpoint)")
        if not getattr(predictor, "aux", False):
            raise ValueError("PREDICT: the predictor has no aux head (model.args.aux)")
        if predictor.levels["level"] != "private":
            raise ValueError(f"PREDICT: the predictor reads {predictor.levels['level']} rows; it must be private")
        if getattr(predictor, "recurrent", False) or getattr(model, "recurrent", False):
            raise ValueError("PREDICT cannot play a recurrent network: a filled row has no decision history")
        self.policy_level = model.levels["level"]
        if self.policy_level == "private":
            raise ValueError("PREDICT: a private policy has nothing to fill in; it plays as POLICY")
        self.sample = bool(_param(params, "sample", False))
        self.temperature = float(_param(params, "temperature", 1.0))
        self.round = bool(_param(params, "round", False))
        self._predictor_weights = {k: v.detach().cpu() for k, v in predictor.state_dict().items()}
        self._predictor_spec = predictor.spec
        if not predictor.dims_agnostic:
            raise ValueError("PREDICT: the predictor must be dims-agnostic (a graph network)")
        #: Which of HIDDEN_FEATURES (by position) the fill writes: those the policy's level shows.
        shown = shown_hidden_features(self.policy_level)
        self.fill_features = [k for k, name in enumerate(HIDDEN_FEATURES) if name in shown]
        super().__init__(model, params, device=device, infer_rows=infer_rows, mode=mode, heads="policy",
                         reads_level="private")

    def _set_model(self, model: CatanModel) -> None:
        super()._set_model(model)
        dims = self.dims
        # The predictor at the policy's dims, on private rows (no critic block: the agent reads
        # actor-level rows; the same weights load), evaluated like the policy.
        predictor = build_model(self._predictor_spec, dims=dims, levels={"level": "private"},
                                num_actions=action_size(dims), layout=row_layout(dims, "private"))
        predictor.load_state_dict(self._predictor_weights)
        self.predictor = predictor
        self.predictor_evaluator = BatchEvaluator(predictor, device=self.device, infer_rows=self.infer_rows,
                                                  mode=self.mode, heads="aux")
        self.num_features = int(predictor.aux_head.num_features)
        if self.num_features != len(HIDDEN_FEATURES):
            raise ValueError(f"PREDICT: the predictor predicts {self.num_features} features, not {len(HIDDEN_FEATURES)}")
        # Where the fill goes in a private row: the players block, each slot's hidden features.
        offset, (players, width) = self.encoder.layout()["players"]
        self.players_span = (int(offset), int(players), int(width))
        names = feature_names(dims, "private")["players"]
        self.hidden_index = hidden_feature_indices(names)
        self.present_index = names.index("present")
        if self.model.obs_size != self.encoder.size:
            raise ValueError("PREDICT: the policy's row and the private row differ in size")

    def predict(self, rows: np.ndarray) -> np.ndarray:
        """The predictor's aux output on private rows: [B, P-1, F] (row k = slot k+1), in the
        encoder's scale."""
        flat, _ = self.predictor_evaluator(rows)
        return flat.reshape(len(rows), self.dims["max_players"] - 1, self.num_features)

    def policy_rows(self, rows: np.ndarray, games: Sequence[cpp.Game] | None = None) -> np.ndarray:
        """The rows the policy decides on: ``rows`` (private) with, for each other present
        seat, the hidden features the policy's level shows replaced by the predictions
        (rounded with ``round``). ``games`` give the rules' totals for rounding."""
        rows = np.ascontiguousarray(rows, dtype=np.float32)
        predicted = self.predict(rows)
        if self.round:
            if games is None:
                raise ValueError("PREDICT with round=true reads each game's rules for the totals")
            predicted = self.round_predictions(predicted, games)
        out = rows.copy()
        offset, players, width = self.players_span
        block = out[:, offset : offset + players * width].reshape(len(out), players, width)
        present = block[:, 1:, self.present_index] > 0                        # [B, P-1]
        for k in self.fill_features:
            column = block[:, 1:, self.hidden_index[k]]
            column[...] = np.where(present, predicted[:, :, k], column)
        return out

    @staticmethod
    def round_predictions(predicted: np.ndarray, games: Sequence[cpp.Game]) -> np.ndarray:
        """Each prediction snapped to the nearest value the encoder can write: ``count /
        total`` (as the engine computes it: a double ratio cast to float32) for the actual
        VP over the game's points to win, the hand per resource over the rules' bank and
        the development cards per kind over the rules' deck; the playable flags to 0/1."""
        out = np.empty_like(predicted)
        for i, game in enumerate(games):
            rules = game.rules
            totals = [int(game.vps_to_win), *(int(n) for n in rules.resource_bank),
                      *(int(n) for n in rules.development_deck)]
            for k, total in enumerate(totals):
                if total <= 0:
                    out[i, :, k] = 0.0
                    continue
                counts = np.clip(np.rint(predicted[i, :, k].astype(np.float64) * total), 0, total)
                out[i, :, k] = (counts / float(total)).astype(np.float32)
            out[i, :, len(totals):] = (predicted[i, :, len(totals):] >= 0.5).astype(np.float32)
        return out

    def act(self, rows, masks, decision_seeds, *, games=None, trackers=None, streams=None, first=None):
        if games is None:
            raise ValueError("PREDICT decides from games (their rules give the fill's totals)")
        if len(games) != len(rows):
            raise ValueError("PREDICT needs one game per row")
        logits, _ = self.evaluator(self.policy_rows(rows, games))
        return _choose(logits, masks, decision_seeds, sample=self.sample, temperature=self.temperature)


class MctsAgent(NativeAgent):
    """``MCTS``: one SearchPool search per decision.

    With ``reuse`` a seat's search outlives its decision: at the stream's next decision
    it is re-rooted at the action played (``SearchPool.advance``, keeping the subtree and
    starting a fresh budget) when the game handed is exactly the previous game plus that
    action (``search_equivalent``: a chance action, another seat's move or a forced move
    in between diverge) and the engine keeps the worlds; otherwise a fresh search starts.
    Rows then depend on the stream's history, so ``reuse`` is keyed and the agent
    ``needs_memory`` (``streams`` and ``first``); a stream's first decision drops its
    live search. Live searches: one per stream (env x seat), each a full search's
    retained subtree (about 12 MB at 1024 sims in 4p), so budget ``num_envs x MCTS
    seats x 12 MB``. ``counters``: ``decisions``, ``fresh``, ``reused``,
    ``dropped_first``, ``refused_diverged`` (the game is not the previous one plus the
    action played) and ``refused_engine`` (``advance`` returned false); ``search_stats``
    sums ``SearchStats`` over every search run.

    ``dedup`` (execution-only, on by default) evaluates byte-identical leaf rows once
    (:class:`~catanrl.search.dedup.DedupEvaluator`, bitwise neutral).
    """

    key = "MCTS"
    needs_games = True

    def __init__(self, model, params, *, device, infer_rows, mode="strict") -> None:
        self.reuse = bool(_param(params, "reuse", False))
        self.dedup = bool(_param(params, "dedup", True))
        super().__init__(model, params, device=device, infer_rows=infer_rows, mode=mode, heads="both")
        # After super().__init__, which refuses a recurrent network while needs_memory is off.
        self.needs_memory = self.reuse
        self.config = search_config(params, max_players=model.num_players)
        self.needs_trackers = self.config.determinizations > 0
        self.tracker_mode = tracker_mode_of(params)
        self.temperature = float(_param(params, "temperature", 0.0))
        self.counters: Counter[str] = Counter()
        self.search_stats: Counter[str] = Counter()
        self._make_pool()

    def _set_model(self, model) -> None:
        super()._set_model(model)
        if self.dedup:
            self.evaluator = DedupEvaluator(self.evaluator)
        if hasattr(self, "config"):
            self._make_pool()

    def _make_pool(self) -> None:
        self.config.max_players = int(self.dims["max_players"])
        self.pool = make_pool(self.dims, self.level, self.config, critic_level=self._levels["critic_level"])
        #: stream -> (search id, the game handed, the index played): a new pool drops them all.
        self._live: dict[int, tuple[int, cpp.Game, int]] = {}

    def act(self, rows, masks, decision_seeds, *, games=None, trackers=None, streams=None, first=None):
        if games is None:
            raise ValueError("MCTS searches from games")
        if self.needs_trackers and trackers is None:
            raise ValueError("MCTS with dets searches worlds sampled from hand trackers")
        if self.reuse:
            streams, first = self._streams(len(games), streams, first)
        ids: list[int | None] = [None] * len(games)
        try:
            if self.reuse:
                # Advance before adding: add() balances lanes by the live searches.
                self._reroot(ids, games, streams, first)
            for i, (game, seed) in enumerate(zip(games, decision_seeds)):
                if ids[i] is None:
                    tracker = trackers[i] if self.needs_trackers else None
                    ids[i] = self.pool.add(game, seed=int(seed), tracker=tracker,
                                           deal=self.tracker_mode if tracker is not None else None)
                    self.counters["fresh"] += 1
            run_searches(self.pool, self.evaluator, batch_rows=len(ids) * self.config.max_pending)
            choices = [int(self.pool.select(sid, self.temperature)) for sid in ids]
            self.counters["decisions"] += len(ids)
            for sid in ids:
                self._record(self.pool.stats(sid))
            if self.reuse:
                for i, stream in enumerate(streams):
                    self._live[stream] = (ids[i], games[i], choices[i])
            else:
                for sid in ids:
                    self.pool.remove(sid)
            return np.array(choices, dtype=np.int64)
        except BaseException:
            # Leave the pool empty whatever failed (the live searches too: their streams are
            # out of step with the pool), so the error is the real one and the agent keeps working.
            if self.pool.pending:
                self.pool.cancel()
            for sid in list(self.pool.ids):
                self.pool.remove(sid)
            self._live.clear()
            raise

    @staticmethod
    def _streams(n: int, streams, first) -> tuple[list[int], list[bool]]:
        if streams is None or first is None:
            raise ValueError("MCTS with reuse=true decides from streams and first (its searches' streams)")
        streams = np.asarray(streams, dtype=np.int64).reshape(-1)
        first = np.asarray(first, dtype=bool).reshape(-1)
        if len(streams) != n or len(first) != n:
            raise ValueError("streams and first need one entry per row")
        if len(set(streams.tolist())) != len(streams):
            raise ValueError("a stream decides once per batch")
        return streams.tolist(), first.tolist()

    def _reroot(self, ids: list, games: Sequence[cpp.Game], streams: list[int], first: list[bool]) -> None:
        """Re-root each stream's live search at the action it played, when the game handed is
        that search's root plus the action and the engine keeps the worlds; drop it otherwise."""
        for i, (game, stream, fresh) in enumerate(zip(games, streams, first)):
            entry = self._live.pop(stream, None)
            if fresh:
                if entry is not None:
                    self.pool.remove(entry[0])
                    self.counters["dropped_first"] += 1
                continue
            if entry is None:
                raise ValueError(f"stream {stream} has no live search: its first decision was not marked first")
            sid, prev, chosen = entry
            expected = prev.copy()
            expected.execute(self.action_space.decode(prev, chosen))
            # The real game evolved by exactly the action played (hands, RNG and options
            # included); whether the sampled worlds survive it is the engine's own refusal.
            same = expected.search_equivalent(game)
            if same and self.pool.advance(sid, chosen):
                ids[i] = sid
                self.counters["reused"] += 1
            else:
                self.counters["refused_engine" if same else "refused_diverged"] += 1
                self.pool.remove(sid)

    def _record(self, stats: cpp.SearchStats) -> None:
        for name in ("nodes", "evaluations", "simulations", "coalesced_outcomes", "max_depth"):
            self.search_stats[name] += int(getattr(stats, name))
        self.search_stats["transpositions"] += int(getattr(stats, "transpositions", 0))


class BotAgent:
    """F or R: the harness plays these seats with the engine's bots."""

    needs_games = False
    needs_trackers = False
    tracker_mode = cpp.TrackerMode.POOL
    needs_memory = False
    recurrent = False

    def __init__(self, key: str, params: Any = None) -> None:
        if key not in ("F", "R"):
            raise ValueError(f"no native bot {key!r}")
        self.key = key
        enemies = str(_param(params, "enemies", "next") or "next")
        self.value_enemies = {"next": cpp.ValueEnemies.NEXT, "all": cpp.ValueEnemies.ALL}[enemies]

    def bind(self, dims) -> None:
        pass

    def check_maps(self, maps) -> None:
        pass


AGENTS = {"POLICY": PolicyAgent, "MCTS": MctsAgent, "BELIEF": BeliefAgent, "PREDICT": PredictAgent}


def _load_model(resolved: Any, part: str, ref: Any = None) -> CatanModel:
    from catanrl.checkpoint import Checkpoint

    ref = resolved.checkpoint if ref is None else ref
    ckpt = Checkpoint.load(ref.path, ref=ref)
    return ckpt.model(part, device="cpu", allow_drift=bool(_param(resolved.params, "allow_drift", False)))


def make_native_agent(resolved: Any, *, device: str, infer_rows: int, mode: str = "strict",
                      model: CatanModel | None = None, predictor: CatanModel | None = None) -> NativeAgent | BotAgent:
    """The native backend of a resolved agent spec.

    ``model`` (and, for PREDICT, ``predictor``) overrides loading the network from
    ``resolved.checkpoint`` (``resolved.predictor``): tests and in-memory agents.
    ``mode`` is the evaluator's determinism.
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
    if key == "PREDICT":
        if predictor is None:
            predictor = _load_model(resolved, "aux", resolved.predictor)
        return PredictAgent(model, resolved.params, device=device, infer_rows=infer_rows, mode=mode,
                            predictor=predictor)
    return AGENTS[key](model, resolved.params, device=device, infer_rows=infer_rows, mode=mode)
