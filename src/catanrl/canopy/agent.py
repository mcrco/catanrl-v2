"""``CANOPY``: released Canopy (cullback/canopy, nexus-v3 iteration 315) as a native agent.

``CANOPY:sims=1600`` resolves (:func:`resolve_canopy`) to a :class:`CanopyResolved` whose
``native()`` is a :class:`CanopyAgent`: the harness drives it like any network agent
(``bind``, ``check_maps``, ``act`` with games, trackers, streams and first flags), and it drives
the Rust bridge processes (:mod:`catanrl.canopy.bridge`).

Per decision the agent serializes the game from its seat's point of view
(:func:`catanrl.canopy.snapshot.snapshot`: the opponent's resources dealt from the seat's
tracker, its development cards as a count), sends every decision of the batch to the bridge
(each game to one process, all processes in parallel), and maps Canopy's choice back to the
engine's legal action with that Canopy id. Canopy proposes from its own legal set, which under
these rules is a subset of the engine's (its pruning drops options, never adds one); a choice
the engine does not offer is played as Canopy's best-scored choice the engine does offer (the
policy the bridge returns), else the engine's first legal index, and counted, logged and kept
in the game's stats as a fallback. Every Canopy-legal action the engine refuses is counted
(``canopy_illegal``) whether or not it was chosen: a rule mismatch seen in play.

Determinism: a decision depends on the snapshot, its seed (``derive_seed(decision seed,
"canopy")``) and the game's saved search, which depends only on the game's earlier decisions
(``reuse``: the search is re-rooted when the moves since the seat's last decision replay
exactly, :func:`~catanrl.canopy.snapshot.replay_chain`; within a turn, or across moves the seat
knows were forced, whatever the opponent's hidden cards). A game stays with one bridge process
(its env's, ``env % procs``), the bridge searches each state with its own seed and its batches do
not mix states, so rows do not depend on ``procs`` (or ``$CATANRL_CANOPY_PROCS``, which overrides
it), ``num_envs`` or ``threads`` (checked by the smoke's replay).

Per game (keyed by the seat's first decision seed) :meth:`CanopyAgent.pop_game` returns the
seat's counts and timings, which :mod:`catanrl.eval.results` writes into the row's ``canopy``
entry (not an outcome: wall times differ between runs). ``reused`` counts decisions whose search
was re-rooted at the position (the bridge's replay matched it), ``reused_visits`` those whose new
root had visits already (``retained_visits`` sums them): a re-rooted search can land on a node
the old search never expanded, and every search gets its full budget either way.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import cppanatron as cpp
import numpy as np
from catanatron.params import ParamsError, build_params
from catanatron.registry import SpecError, parse_spec

from catanrl.agents.native import NativeAgent
from catanrl.agents.spec import ResolvedAgent, _fmt, key_params, keyed_fields
from catanrl.canopy import CANOPY_KEY
from catanrl.canopy.bridge import BridgePool
from catanrl.canopy.release import CHECKPOINT_SHA256, Release, ReleaseError, load_release
from catanrl.canopy.snapshot import (
    BASE_BANK,
    BASE_DECK,
    BASE_PIECES,
    SeatHistory,
    SnapshotError,
    action_name,
    check_rules,
    legal_by_id,
    replay_chain,
    snapshot,
)
from catanrl.canopy.topology import CanopyBoard, MapError
from catanrl.envs.encoding import check_dims, make_encoder
from catanrl.seeding import derive_seed

#: Examples of a kind kept per game (the counts are complete).
EXAMPLES = 5


@dataclass(frozen=True)
class CanopyParams:
    #: Simulations per decision; 0 plays the raw policy's argmax over Canopy's legal actions.
    sims: int = 1600
    c_puct: float = 2.5
    c_visit: float = 50.0
    c_scale: float = 1.0
    dirichlet_alpha: float = 0.05
    #: Root noise weight; 0 (v1's evaluation setting) turns it off. Canopy's tournament used 0.25.
    dirichlet_epsilon: float = 0.0
    #: Re-root the game's saved search at the moves since the seat's last decision when they
    #: replay exactly (Canopy's own tree reuse); false searches every decision afresh.
    reuse: bool = True
    #: Bridge processes (execution-only; ``$CATANRL_CANOPY_PROCS`` overrides it, e.g. for a verify,
    #: which resolves the canonical spec and so would run one).
    procs: int = field(default=1, metadata={"execution": True})
    timeout: float = field(default=3600.0, metadata={"execution": True})


def _procs_override() -> int | None:
    """``$CATANRL_CANOPY_PROCS``: bridge processes for every CANOPY agent of this process (procs is
    execution-only, so no key changes; rows do not depend on it)."""
    value = os.environ.get("CATANRL_CANOPY_PROCS")
    if not value:
        return None
    try:
        procs = int(value)
    except ValueError:
        procs = 0
    if procs < 1:
        raise SpecError(f"CATANRL_CANOPY_PROCS={value!r}: a positive count of bridge processes")
    return procs


class _Spec:
    """``build_params``'s view of the CANOPY key."""

    Params = CanopyParams


def canopy_params(args: Sequence[Any] = (), named: Mapping[str, Any] | None = None) -> CanopyParams:
    try:
        params = build_params(_Spec, list(args), dict(named or {}))
    except (ParamsError, TypeError) as error:
        raise SpecError(f"{CANOPY_KEY}: {error}".replace("_Spec", CANOPY_KEY)) from None
    if params.sims < 0:
        raise SpecError(f"{CANOPY_KEY}: sims counts simulations (0 plays the raw policy)")
    if min(params.c_puct, params.c_visit, params.c_scale, params.dirichlet_alpha) <= 0:
        raise SpecError(f"{CANOPY_KEY}: c_puct, c_visit, c_scale and dirichlet_alpha are positive")
    if not 0.0 <= params.dirichlet_epsilon <= 1.0:
        raise SpecError(f"{CANOPY_KEY}: dirichlet_epsilon is a weight in [0, 1]")
    if params.procs < 1 or params.timeout <= 0:
        raise SpecError(f"{CANOPY_KEY}: procs and timeout are positive")
    return params


@dataclass(frozen=True)
class CanopyResolved(ResolvedAgent):
    """A resolved CANOPY spec: plays natively through the bridge."""

    release: Release | None = field(default=None, compare=False, repr=False)

    def key_dict(self) -> dict:
        """The keyed params, and the release: source commit, patches, binary and weights."""
        return {"key": self.key, "harness": self.harness, "params": key_params(self.params),
                "canopy": self.release.key()}

    def native(self, *, device: str | None = None, infer_rows: int = 256, mode: str = "strict") -> CanopyAgent:
        return CanopyAgent(self.params, self.release)

    def player(self, color):
        raise SpecError(f"{CANOPY_KEY} plays through the native harness only")


def resolve_canopy(spec: str | Mapping[str, Any], *, harness: str = "native", games: Sequence[Any] = ()
                   ) -> CanopyResolved:
    """Check a CANOPY spec, its games and the release; raises ``SpecError``."""
    from catanrl.games import load_game

    key, args, named = parse_spec(spec if isinstance(spec, str) else dict(spec))
    if key != CANOPY_KEY:
        raise SpecError(f"{spec!r} is not a {CANOPY_KEY} spec")
    if harness != "native":
        raise SpecError(f"{CANOPY_KEY} plays through the native harness, not {harness}")
    params = canopy_params(args, named)
    for game in (g if hasattr(g, "num_players") else load_game(g) for g in games):
        supply = (tuple(game.resource_bank), tuple(game.development_deck),
                  (game.roads_per_player, game.settlements_per_player, game.cities_per_player))
        if game.num_players != 2 or supply != (BASE_BANK, BASE_DECK, BASE_PIECES) or game.friendly_robber:
            raise SpecError(f"{CANOPY_KEY}: the released Canopy plays 2 players with the base supply and no "
                            f"friendly robber; game {game.name} differs")
    try:
        release = load_release()
    except ReleaseError as error:
        raise SpecError(f"{CANOPY_KEY}: {error}") from None
    canonical = f"{CANOPY_KEY}:" + ":".join(f"{name}={_fmt(getattr(params, name))}" for name in keyed_fields(params))
    label = f"{CANOPY_KEY}:nexus-v3@315:sims={params.sims}"
    return CanopyResolved(key=CANOPY_KEY, impl="catanrl.canopy", params=params, checkpoint=None, fair=True,
                          label=label, canonical=canonical, harness=harness,
                          details={"release": release.key(), "checkpoint_sha256": CHECKPOINT_SHA256},
                          release=release)


# ---------------------------------------------------------------- the agent


def _new_stats() -> dict[str, Any]:
    return {"decisions": 0, "searched": 0, "raw": 0, "forced": 0, "no_legal": 0, "reused": 0, "reused_visits": 0,
            "retained_visits": 0, "fresh": 0, "reset": 0, "reset_reasons": {}, "refreshed": 0, "fallbacks": 0, "fallback_examples": [], "ambiguous": 0,
            "canopy_illegal": 0, "canopy_illegal_examples": [], "unexpressed": 0, "unexpressed_examples": [],
            "pruned": 0, "trade_prompts": 0, "opponent_unknown_cards": 0, "history_mismatches": 0,
            "root_visits": 0, "bridge_s": 0.0, "wall_s": 0.0}


@dataclass
class _Stream:
    game_id: str
    first_seed: int
    process: int
    history: SeatHistory
    stats: dict
    previous: tuple[cpp.Game, cpp.Action] | None = None


def _example(stats: dict, kind: str, text: str) -> None:
    examples = stats[f"{kind}_examples"]
    if len(examples) < EXAMPLES:
        examples.append(text)


class CanopyAgent(NativeAgent):
    """Released Canopy, playing the harness's decisions through bridge processes."""

    key = CANOPY_KEY
    needs_games = True
    needs_trackers = True
    tracker_mode = cpp.TrackerMode.POOL
    needs_memory = True
    recurrent = False
    dims_agnostic = True

    def __init__(self, params: CanopyParams, release: Release, *, pool_factory: Any = None) -> None:
        # No network of ours: NativeAgent.__init__ is not called; the harness reads level, dims,
        # the encoder (for rows it builds itself) and the flags above.
        procs = _procs_override()
        if procs is not None:
            params = dataclasses.replace(params, procs=procs)
        self.params = params
        self.release = release
        self.level = "public"
        self.dims: dict | None = None
        self.encoder = None
        self.action_space = None
        self._pool_factory = pool_factory or (lambda: BridgePool(
            release, procs=params.procs, sims=params.sims, c_puct=params.c_puct, c_visit=params.c_visit,
            c_scale=params.c_scale, dirichlet_alpha=params.dirichlet_alpha,
            dirichlet_epsilon=params.dirichlet_epsilon, reuse=params.reuse, timeout=params.timeout))
        self._pool: Any = None
        self._streams: dict[int, _Stream] = {}
        self._games: dict[int, dict] = {}
        self.counters: Counter[str] = Counter()

    # ------------------------------------------------------------ harness interface

    def bind(self, dims: Mapping[str, int]) -> None:
        dims = check_dims(dims)
        if dims != self.dims:
            self.dims = dims
            self.encoder = make_encoder(dims, self.level)
            self.action_space = self.encoder.action_space

    def check_maps(self, maps) -> None:
        for catan_map in maps:
            try:
                CanopyBoard.from_map(catan_map)
            except MapError as error:
                raise ValueError(f"{CANOPY_KEY}: {error}") from None

    def encode(self, game: cpp.Game, seat: int) -> np.ndarray:
        return self.encoder.encode(game, seat)

    @property
    def pool(self):
        if self._pool is None:
            self._pool = self._pool_factory()
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def pop_game(self, first_seed: int) -> dict | None:
        """The stats of the game whose seat's first decision had ``first_seed`` (once)."""
        return self._games.pop(int(first_seed), None)

    def _stream(self, stream: int, fresh: bool, seed: int, seat: int, game: cpp.Game) -> _Stream:
        if fresh:
            check_rules(game)
            procs = len(self.pool)
            # The seat's first decision seed names the game (unique per game and seat, and the same
            # whatever env plays it, so the stats rows hold do not depend on num_envs).
            # Streams are env * max_players + seat: spread the envs over the processes (which
            # process searches a game changes no decision).
            process = stream // int(self.dims["max_players"] if self.dims else 1) % procs
            entry = _Stream(game_id=f"{seed:016x}", first_seed=seed, process=process,
                            history=SeatHistory(seat), stats=_new_stats())
            self._streams[stream] = entry
            self._games[seed] = entry.stats
            return entry
        entry = self._streams.get(stream)
        if entry is None:
            raise ValueError(f"{CANOPY_KEY}: stream {stream} has no game: its first decision was not marked first")
        if entry.history.seat != seat:
            raise ValueError(f"{CANOPY_KEY}: stream {stream} is seat {entry.history.seat}, not {seat}")
        return entry

    def act(self, rows, masks, decision_seeds, *, games=None, trackers=None, streams=None, first=None):
        if games is None or trackers is None or streams is None or first is None:
            raise ValueError(f"{CANOPY_KEY} decides from games, hand trackers, streams and first flags")
        start = time.perf_counter()
        n = len(games)
        entries, boards, legal, unexpressed, snaps = [], [], [], [], []
        for i in range(n):
            game, tracker, seed = games[i], trackers[i], int(decision_seeds[i])
            seat = int(game.current_player_index)
            entry = self._stream(int(streams[i]), bool(first[i]), seed, seat, game)
            board = CanopyBoard.from_map(game.board.map)
            tree = None
            if self.params.reuse and entry.previous is not None:
                tree = replay_chain(board, entry.previous[0], entry.previous[1], game, seat)
            entry.history.start(game, tracker)
            if tracker.unknown_count(1 - seat) > 0:
                entry.stats["opponent_unknown_cards"] += 1
            snaps.append(snapshot(game, tracker, board, entry.history, game_id=entry.game_id,
                                  seed=derive_seed(seed, "canopy"), world_seed=derive_seed(seed, "canopy-world"),
                                  tree_actions=tree))
            by_id, cannot = legal_by_id(board, game)
            entries.append(entry)
            boards.append(board)
            legal.append(by_id)
            unexpressed.append(cannot)
        attempts: list[list[tuple]] = [[] for _ in range(n)]
        todo = list(range(n))
        while todo:
            asked = self._ask([entries[i] for i in todo], [snaps[i] for i in todo], [sorted(legal[i]) for i in todo])
            for i, answer in zip(todo, asked):
                attempts[i].append(answer)
            # A re-rooted search keeps Canopy's canonical-ordering cursors, which can leave it only
            # moves the engine refuses (END_TURN in the middle of a road building card): ask once
            # more on the imported position alone, which resets them (counted as ``refreshed``).
            todo = [i for i in todo if len(attempts[i]) == 1 and attempts[i][0][1].get("reuse") == "reused"
                    and attempts[i][0][0] not in legal[i]]
            for i in todo:
                snaps[i] = {**snaps[i], "tree_actions": [], "reuse_tree": False}
        choices = np.empty(n, dtype=np.int64)
        for i in range(n):
            choices[i] = self._play(entries[i], boards[i], games[i], trackers[i], masks[i], legal[i],
                                    unexpressed[i], attempts[i])
        wall = (time.perf_counter() - start) / max(1, n)
        for entry in entries:
            entry.stats["wall_s"] += wall
        self.counters["decisions"] += n
        return choices

    def _ask(self, entries: Sequence[_Stream], snaps: Sequence[dict], legal: Sequence[list[int]]
             ) -> list[tuple[int | None, dict, float]]:
        """Canopy's (choice, details, bridge seconds) per snapshot, each game at its process."""
        procs = len(self.pool)
        payload: list[list[dict]] = [[] for _ in range(procs)]
        offered: list[list[list[int]]] = [[] for _ in range(procs)]
        slots = []
        for entry, snap, ids in zip(entries, snaps, legal):
            slots.append((entry.process, len(payload[entry.process])))
            payload[entry.process].append(snap)
            offered[entry.process].append(ids)
        active: list[list[str]] = [[] for _ in range(procs)]
        for entry in self._streams.values():
            active[entry.process].append(entry.game_id)
        answers = self.pool.decide(payload, active, sims=self.params.sims, legal=offered)
        self.counters["requests"] += sum(1 for states in payload if states)
        out = []
        for process, k in slots:
            answer = answers[process]
            share = float(answer.get("elapsed_ms", 0.0)) / 1000.0 / max(1, len(payload[process]))
            aid = answer["actions"][k]
            out.append((None if aid is None else int(aid), answer["details"][k], share))
        return out

    def _play(self, entry: _Stream, board: CanopyBoard, game: cpp.Game, tracker: cpp.HandTracker,
              mask: np.ndarray, by_id: dict, unexpressed: list, attempts: Sequence[tuple]) -> int:
        stats = entry.stats
        stats["decisions"] += 1
        stats["refreshed"] += len(attempts) - 1
        for _, detail, bridge_s in attempts:
            stats["bridge_s"] += bridge_s
            refused = [a for a in (int(a) for a in detail.get("canopy_legal", [])) if a not in by_id]
            if refused:
                stats["canopy_illegal"] += len(refused)
                _example(stats, "canopy_illegal", f"turn {game.num_turns} {game.current_prompt.name} "
                         f"({detail.get('reuse')}): " + ", ".join(action_name(board, a) for a in refused[:4]))
        aid, detail, _ = attempts[-1]
        mode = str(detail.get("mode"))
        stats[{"search": "searched", "raw": "raw", "forced": "forced"}.get(mode, "no_legal")] += 1
        reuse = str(detail.get("reuse", "fresh"))
        if reuse.startswith("reset"):
            stats["reset"] += 1
            reason = reuse.split(":", 1)[1].split("_at_")[0] if ":" in reuse else reuse
            stats["reset_reasons"][reason] = stats["reset_reasons"].get(reason, 0) + 1
        else:
            stats["reused" if reuse == "reused" else "fresh"] += 1
        before = int(detail.get("root_visits_before", 0) or 0)
        if reuse == "reused" and before > 0:
            # ``reused`` counts re-rooted searches; this, those whose new root kept visits.
            stats["reused_visits"] += 1
            stats["retained_visits"] += before
        stats["root_visits"] += int(detail.get("root_visits", 0) or 0)
        canopy_legal = {int(a) for a in detail.get("canopy_legal", [])}
        stats["pruned"] += sum(1 for a in by_id if a not in canopy_legal)
        if unexpressed:
            stats["unexpressed"] += 1
            _example(stats, "unexpressed", f"turn {game.num_turns}: {unexpressed[0]!r}")
        if aid is not None and aid in by_id:
            options = by_id[aid]
            if len(options) > 1:
                stats["ambiguous"] += 1
            action = options[0]
        else:
            ranked = sorted(detail.get("policy", []), key=lambda pair: (-float(pair[1]), int(pair[0])))
            action = next((by_id[int(a)][0] for a, _ in ranked if int(a) in by_id), None)
            if action is None:
                action = self.action_space.decode(game, int(np.flatnonzero(mask > 0)[0]))
            text = (f"{entry.game_id} turn {game.num_turns} {game.current_prompt.name}: Canopy chose "
                    f"{'nothing' if aid is None else action_name(board, aid)} ({reuse}), which the engine does "
                    f"not offer; played {action!r}")
            print(f"warning: {CANOPY_KEY} fallback: {text}", file=sys.stderr, flush=True)
            stats["fallbacks"] += 1
            _example(stats, "fallback", text)
        index = int(self.action_space.index(game, action))
        if not mask[index] > 0:
            raise RuntimeError(f"{CANOPY_KEY}: {action!r} is index {index}, which the mask does not allow")
        after = game.copy()
        after.execute(action)
        entry.history.played(game, tracker, action, after)
        mismatches = entry.history.mismatches
        if mismatches:
            stats["history_mismatches"] += len(mismatches)
            print(f"warning: {CANOPY_KEY} {entry.game_id}: {'; '.join(mismatches)}", file=sys.stderr, flush=True)
            mismatches.clear()
        entry.previous = (game, action)
        return index

    def respond_trade(self, games, decision_seeds, *, trackers=None, streams=None, first=None):
        """Canopy has no domestic trades: it rejects every offer (counted per game)."""
        from catanrl.agents.trade import default_trade_action

        out = []
        for i, game in enumerate(games):
            if streams is not None and first is not None:
                seat = int(game.current_player_index)
                entry = self._stream(int(streams[i]), bool(first[i]), int(decision_seeds[i]), seat, game)
                entry.stats["trade_prompts"] += 1
                entry.previous = None  # a trade step is not a Canopy move: the next search starts afresh
            out.append(default_trade_action(game))
        return out


def summary_totals(stats: Sequence[Mapping[str, Any]]) -> dict:
    """One slot's Canopy stats over its games: sums, per-decision times, reset reasons."""
    keys = ("decisions", "searched", "raw", "forced", "no_legal", "reused", "reused_visits", "retained_visits", "fresh",
            "reset", "refreshed", "fallbacks",
            "ambiguous", "canopy_illegal", "unexpressed", "pruned", "trade_prompts", "opponent_unknown_cards",
            "history_mismatches", "root_visits", "bridge_s", "wall_s")
    totals = {k: sum(s.get(k, 0) for s in stats) for k in keys}
    reasons: Counter[str] = Counter()
    for s in stats:
        reasons.update(s.get("reset_reasons", {}))
    decisions = totals["decisions"]
    return {"games": len(stats), **totals, "reset_reasons": dict(sorted(reasons.items())),
            "per_decision_s": totals["wall_s"] / decisions if decisions else 0.0,
            "bridge_per_decision_s": totals["bridge_s"] / decisions if decisions else 0.0,
            "per_game_decisions": decisions / len(stats) if stats else 0.0,
            "fallback_examples": [e for s in stats for e in s.get("fallback_examples", [])][:EXAMPLES],
            "canopy_illegal_examples": [e for s in stats for e in s.get("canopy_illegal_examples", [])][:EXAMPLES]}


__all__ = ["CanopyAgent", "CanopyParams", "CanopyResolved", "SnapshotError", "canopy_params", "resolve_canopy",
           "summary_totals"]
