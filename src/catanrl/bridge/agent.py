"""Native agents deciding from one seat's view.

:class:`ViewAgent` wraps a native agent (POLICY, MCTS, BELIEF; see
:mod:`catanrl.agents.native`) as ``decide(view, playable_actions)``, the
callback ``catanatron.server.client.GameClient.play`` and the TUI's
``decide_fns`` take::

    agent = ViewAgent("MCTS:ckpt=runs/x@best:sims=256:dets=4", device="cuda")
    client.on_message = agent.observe          # optional: exact monopolies, the room's supply
    result = await client.play(agent.decide)
    catanatron.tui.run(["human", "nn", "F"], decide_fns={"nn": agent.decide})

Per decision it

1. replays the view's new log entries for the deciding seat
   (:class:`catanrl.bridge.tracking.SeatTracker`: the seat's game with
   placeholder hidden cards, and its hand tracker) and checks the result
   against every public field of the view;
2. masks the network by exactly the offered actions;
3. runs the native agent at batch 1 on that game: POLICY on the seat's row
   (a PRIVATE row reads nothing hidden; a PUBLIC one only in two-player
   games, where the other hand follows from the bank), BELIEF and MCTS on
   worlds the tracker deals (``cpp.sample_world``), never on the
   placeholders;
4. decodes the chosen index, converts it to Catanatron and returns the
   offered action it equals (the object from ``playable_actions``).

An MCTS agent with ``reuse`` keeps a seat's search between its decisions.
The ViewAgent hands it one stream per game slot and seat (``env *
max_players + seat``, the slot a per-game integer) and ``first`` at the
seat's first decision of a game, and sets the agent's ``reuse_check`` to
``"view"``:
the game it hands is replayed from views, so the agent compares what the
seat sees (its private row, the legal actions, the mover, prompt and turn),
not hands and RNG. A forced move (one offered action) is answered without
the agent, so the search it kept is refused as diverged at the seat's next
decision and a fresh one starts. A game's last message (``after``, or
``aborted`` once it had started; :meth:`ViewAgent.observe`) drops its
searches and frees its slot at the next decision; a game whose end was never
seen keeps them until it is evicted, when its slot's next game drops them at
its first decision, as a new game in a harness env would.

Decision seeds are ``catanrl.seeding.derive_seed(game_id, "decide", seat,
index)``, ``index`` being the number of log entries before the decision: a
pure function of the view, so a transcript fixes every decision's seed. (The
native harness counts the seat's own decisions instead, which a view cannot
reproduce after a reconnect or a TUI undo.) ``seed_fn`` overrides it.

Nothing falls back silently. The seat's replay re-anchors on the view's exact
``apply_setup`` snapshot whenever it cannot explain the view (a TUI edit, a
mid-game join whose log it cannot follow); what is left (no position at all,
a position that still differs from the view, a chosen action that was not
offered, which is never mapped to another one, an error) plays the
server's default move or is counted in :attr:`ViewAgent.stats`, and
``strict=True`` (the tests) raises :class:`BridgeError` instead.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import Counter, OrderedDict, deque
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import cppanatron as cpp
import numpy as np
from catanatron.serialization import action_from_json, action_to_json
from cppanatron.catanatron import from_catanatron_action, to_catanatron_action

from catanrl.bridge.position import (
    Position,
    SeatView,
    ViewError,
    color_name,
)
from catanrl.bridge.tracking import SeatTracker
from catanrl.seeding import derive_seed

logger = logging.getLogger(__name__)

#: What a seat plays when the bridge cannot decide (catanatron-server's
#: ``default_action``): roll, else end the turn, else the first offered move.
DEFAULT_PREFERENCE = ("ROLL", "END_TURN")
GAME_MESSAGES = ("before", "decide", "step", "after")
#: A game's end: played out, or voided after it started (``aborted`` carries its id then).
END_MESSAGES = ("after", "aborted")


class BridgeError(RuntimeError):
    """A decision the bridge cannot make exactly (raised with ``strict=True``)."""


class UnfairAgent(ValueError):
    """An agent that would read hidden information from a view's game."""


def native_agent(agent: Any, *, device: str = "cpu", infer_rows: int = 1,
                 mode: str = "strict", memory: bool = False) -> tuple[Any, bool | None, str]:
    """(native agent, resolved ``fair`` if known, label) for an agent spec
    (resolved with ``harness="server"``), a resolved agent or a native agent.
    ``memory``: the caller hands ``act`` the seat's ``streams`` and ``first``
    (as :class:`ViewAgent` does); an agent that needs them (MCTS with
    ``reuse``) is refused otherwise, since it could not decide at all."""
    from catanrl.agents.native import BotAgent, NativeAgent

    if isinstance(agent, str):
        from catanrl.agents.spec import resolve_agent

        agent = resolve_agent(agent, harness="server")
    if isinstance(agent, (NativeAgent, BotAgent)):
        native, fair, label = agent, None, str(getattr(agent, "key", "agent"))
    elif callable(getattr(agent, "native", None)):
        native = agent.native(device=device, infer_rows=infer_rows, mode=mode)
        fair, label = getattr(agent, "fair", None), str(getattr(agent, "label", agent))
    elif hasattr(agent, "key") and hasattr(agent, "params"):
        from catanrl.agents.native import make_native_agent

        native = make_native_agent(agent, device=device, infer_rows=infer_rows, mode=mode)
        fair, label = getattr(agent, "fair", None), str(agent.key)
    else:
        raise TypeError(f"cannot make a native agent of {agent!r}")
    if getattr(native, "needs_memory", False) and not memory:
        raise ValueError(f"{label} keeps a seat's search between its decisions and needs the seat's streams, "
                         "which this caller does not hand it (ViewAgent does)")
    return native, fair, label


def check_fair(native: Any, num_players: int | None = None) -> None:
    """Refuse agents that would read the placeholders (DESIGN 6.5): built-in
    bots, an oracle MCTS, and a POLICY whose actor is ``full``, or ``public``
    beyond two players."""
    from catanrl.agents.native import BeliefAgent, BotAgent, MctsAgent, PolicyAgent

    if isinstance(native, BotAgent):
        raise UnfairAgent(f"{native.key} plays the engine's bot on the true game; on a view use "
                          f"Catanatron's own {native.key} (it samples worlds from the view)")
    if isinstance(native, MctsAgent):
        if native.config.determinizations < 1:
            raise UnfairAgent("MCTS with oracle=true searches the true game; use dets >= 1")
    elif isinstance(native, PolicyAgent):
        level = native.level
        if level == "full" or (level == "public" and num_players not in (None, 2)):
            raise UnfairAgent(f"POLICY with a {level} actor reads hidden cards here; use BELIEF")
    elif not isinstance(native, BeliefAgent):
        raise UnfairAgent(f"unknown agent {native!r}")


def default_choice(playable_actions: Sequence[Any]) -> Any:
    """catanatron-server's ``default_action`` on JSON actions."""
    for kind in DEFAULT_PREFERENCE:
        for action in playable_actions:
            if action[1] == kind:
                return action
    return playable_actions[0]


def action_key(action: Any) -> str:
    """Canonical text of an action or ``[color, type, value]`` triple (a road
    is the same road in either direction)."""
    if not isinstance(action, (list, tuple)) or len(action) != 3 or not isinstance(action[1], str):
        action = action_to_json(action)
    color, kind, value = action
    value = json.loads(json.dumps(value))  # tuples -> lists
    if kind == "BUILD_ROAD":
        value = sorted(value)
    return json.dumps([color, kind, value], sort_keys=True)


class ViewAgent:
    """A native agent playing from a seat's view (see the module doc).

    ``agent``: an agent spec, a resolved agent or a native agent.
    ``supply``: the game's supply when neither the view (only uniform banks
    can be inferred) nor a server ``before`` message (:meth:`observe`) gives
    it. ``seed_fn(view, seat)`` replaces the decision seed. ``strict``
    raises :class:`BridgeError` where a fallback would be counted.
    ``max_games``: games (and seats) followed at once; a reusing MCTS agent
    holds at most one live search per seat of each, and a game whose end
    :meth:`observe` saw leaves at the next decision.
    """

    def __init__(self, agent: Any, *, device: str = "cpu", infer_rows: int = 1, mode: str = "strict",
                 supply: Any = None, strict: bool = False,
                 seed_fn: Callable[[Mapping[str, Any], int], int] | None = None, max_games: int = 8) -> None:
        self.native, fair, self.label = native_agent(agent, device=device, infer_rows=infer_rows, mode=mode,
                                                     memory=True)
        if fair is False:
            raise UnfairAgent(f"{self.label} is not fair on views (DESIGN 6.5)")
        check_fair(self.native)
        if getattr(self.native, "recurrent", False):
            raise BridgeError(f"{self.label}: a recurrent network needs its seat's decision history, which the "
                              "server bridge does not replay")
        #: The agent keeps a seat's search between its decisions (MCTS with ``reuse``): it
        #: is handed the seat's stream and ``first``, and matches the game it is handed as
        #: the seat sees it (the replay's placeholders and RNG are its own).
        self.memory = bool(getattr(self.native, "needs_memory", False))
        if self.memory:
            self.native.reuse_check = "view"
        self.supply = supply
        self.strict = strict
        self.seed_fn = seed_fn
        self.max_games = int(max_games)
        self.stats: Counter[str] = Counter()
        #: (game id, color) -> the seat's tracker, most recent last.
        self.seats: OrderedDict[tuple[Any, str], SeatTracker] = OrderedDict()
        self._games: OrderedDict[Any, dict[str, Any]] = OrderedDict()
        #: Game slots (the streams' env) finished and evicted games gave up, for the next new game.
        self._free_envs: list[int] = []
        self._inbox: deque[dict] = deque()
        self._lock = threading.Lock()

    # ------------------------------------------------------------ the stream

    def observe(self, body: Mapping[str, Any]) -> None:
        """Take in a server message (``GameClient.on_message``): ``before``
        names the room's supply, and every game message shows the hand sizes
        after its step (so monopolies are exact). Cheap and thread-safe: the
        work happens at the next decision. Raises :class:`UnfairAgent` (which
        stops the client) when the room's player count makes the agent
        unfair (a PUBLIC POLICY beyond two players)."""
        if body.get("type") in ("commitments", "before"):
            # Refuse a seat the agent cannot play fairly before the game starts.
            players = (body.get("config") or {}).get("num_players")
            if isinstance(players, int):
                check_fair(self.native, players)
        if body.get("type") in GAME_MESSAGES or body.get("type") in END_MESSAGES:
            self._inbox.append(dict(body))

    def _game(self, game_id: Any) -> dict[str, Any]:
        info = self._games.get(game_id)
        if info is None:
            while len(self._games) >= self.max_games:
                self._free_envs.append(self._games.popitem(last=False)[1]["env"])
            # A new game takes a freed slot: with a reusing agent, its first decision then
            # drops the live searches an evicted game left there (a finished game's went
            # with its last message). "decided": the seats whose decisions reached the agent
            # (its streams).
            env = self._free_envs.pop() if self._free_envs else len(self._games)
            info = self._games[game_id] = {"supply": None, "sizes": {}, "env": env, "decided": set()}
        return info

    def _drain(self) -> None:
        while self._inbox:
            body = self._inbox.popleft()
            if body.get("type") in END_MESSAGES:
                self._forget(body.get("game_id"))
                continue
            info = self._game(body.get("game_id"))
            if body.get("type") == "before":
                info["supply"] = (body.get("config") or {}).get("supply")
            state = body.get("state")
            if not state or "index" not in body:
                continue
            try:
                sizes = SeatView(state, 0).hand_sizes()
            except (KeyError, ViewError):
                continue
            index = int(body["index"])
            info["sizes"][index] = sizes
            for (game_id, _color), tracker in self.seats.items():
                if game_id == body.get("game_id"):
                    tracker.sizes[index] = sizes

    def _forget(self, game_id: Any) -> None:
        """The game is over: drop the agent's searches of it and free its slot now, not
        when it is evicted ``max_games`` games later (a retained search is a full tree
        with its games, 5-20 MB at 1024 sims). Nothing for a game not followed, one
        evicted already, a second seat's copy of the message or a room voided before
        its game (no id)."""
        info = self._games.pop(game_id, None)
        if info is None:
            return
        if self.memory:
            self.native.forget(info["env"])
        self._free_envs.append(info["env"])

    # ------------------------------------------------------------ deciding

    def decide(self, view: Mapping[str, Any], playable_actions: Sequence[Any]) -> Any:
        """One of ``playable_actions`` (the object itself) for ``view``."""
        if not playable_actions:
            raise ValueError("nothing to decide: no playable actions")
        with self._lock:
            self._drain()
            if len(playable_actions) == 1:
                self.stats["forced"] += 1
                if view["current_prompt"] == "MOVE_ROBBER" and (view.get("game") or {}).get("friendly_robber"):
                    # Even a lone friendly-robber option bounds the others' points.
                    tracker = self._tracker(view, playable_actions[0][0])
                    tracker.sync(view)
                    tracker.options([from_catanatron_action(action_from_json(list(playable_actions[0])))])
                return playable_actions[0]
            tracker = self._tracker(view, playable_actions[0][0])
            self.stats["decisions"] += 1
            try:
                position = tracker.sync(view)
                return self._choose(view, playable_actions, tracker, position)
            except (BridgeError, UnfairAgent):
                raise
            except Exception as error:
                if self.strict:
                    raise
                logger.exception("%s could not decide; playing the default move", self.label)
                self.stats[f"fallback:{type(error).__name__}"] += 1
                return default_choice(playable_actions)

    def _tracker(self, view: Mapping[str, Any], color: Any) -> SeatTracker:
        game_id = (view.get("game") or {}).get("id")
        key = (game_id, color_name(color))
        tracker = self.seats.get(key)
        if tracker is None:
            info = self._game(game_id)
            supply = info["supply"] if info["supply"] is not None else self.supply
            tracker = SeatTracker(view, color, supply=supply,
                                  mode=getattr(self.native, "tracker_mode", cpp.TrackerMode.POOL))
            tracker.sizes.update(info["sizes"])
            self.seats[key] = tracker
            while len(self.seats) > self.max_games:
                self.seats.popitem(last=False)
        else:
            self.seats.move_to_end(key)
        return tracker

    def _fallback(self, reason: str, playable_actions: Sequence[Any], detail: str = "") -> Any:
        self.stats[f"fallback:{reason}"] += 1
        if self.strict:
            raise BridgeError(f"{reason}: {detail}")
        logger.warning("%s: %s (%s); playing the default move", self.label, reason, detail)
        return default_choice(playable_actions)

    def _choose(self, view: Mapping[str, Any], playable_actions: Sequence[Any], tracker: SeatTracker,
                position: Position) -> Any:
        seat = tracker.seat
        if position.method == "broken":  # nor could the view be declared (a trade negotiation)
            return self._fallback("no_position", playable_actions, "; ".join(position.mismatches))
        if not position.exact:
            self.stats["inexact"] += 1
            if self.strict:
                raise BridgeError(f"the position differs from the view: {position.mismatches[:3]}")
            logger.warning("%s: the position differs from the view: %s", self.label,
                           "; ".join(position.mismatches[:3]))
        game = position.game
        check_fair(self.native, game.num_players)
        self._bind(game)

        space = self.native.action_space
        offered = [from_catanatron_action(action_from_json(list(a))) for a in playable_actions]
        mask = np.zeros(int(space.size), dtype=np.float32)
        for action in offered:
            try:
                mask[int(space.index(game, action))] = 1.0
            except (IndexError, ValueError):  # outside the action space (a domestic trade)
                self.stats["unindexed_offers"] += 1
        if not mask.any():
            return self._fallback("no_indexable_offer", playable_actions)
        tracker.options(offered)
        game = tracker.game  # the friendly robber may have redealt its placeholders

        row = self.native.encode(game, seat)
        if not np.array_equal(row[-mask.size:], mask):
            self.stats["mask_differs"] += 1
            if self.strict and position.exact:
                raise BridgeError("the game's legal actions differ from the offered ones")
            row = row.copy()
            row[-mask.size:] = mask
        index = self._act(row, mask, self._seed(view, seat), game, tracker)
        chosen = space.decode(game, index)
        key = action_key(action_to_json(to_catanatron_action(chosen)))
        for action in playable_actions:
            if action_key(list(action)) == key:
                return action
        return self._fallback("not_offered", playable_actions, f"{chosen} was not offered")

    def _act(self, row: np.ndarray, mask: np.ndarray, seed: int, game: cpp.Game, tracker: SeatTracker) -> int:
        """The agent's index for the seat's ``game``; a reusing agent decides on the seat's stream."""
        native = self.native
        seeds = np.array([seed], dtype=np.uint64)
        trackers = [tracker.tracker] if getattr(native, "needs_trackers", False) else None
        if not self.memory:
            games = [game] if getattr(native, "needs_games", False) else None
            return int(native.act(row[None], mask[None], seeds, games=games, trackers=trackers)[0])
        info = self._game(tracker.game_id)
        seat = tracker.seat
        # The stream is numbered as the harness numbers them (env * max_players + seat), with
        # the game's slot as the env; first at the seat's first decision of the game to reach
        # the agent (a decision that failed does not count: the agent restarted its streams).
        streams = np.array([info["env"] * int(native.dims["max_players"]) + seat], dtype=np.int64)
        first = np.array([seat not in info["decided"]], dtype=bool)
        try:
            # The agent keeps the game it is handed until the seat's next decision; the
            # replay moves this one on.
            index = native.act(row[None], mask[None], seeds, games=[game.copy()], trackers=trackers,
                               streams=streams, first=first)[0]
        except BaseException:
            self._restart_streams()
            raise
        info["decided"].add(seat)
        return int(index)

    def _restart_streams(self) -> None:
        """The agent dropped every live search and forgot its streams (an error in ``act``,
        a bind to new dims): each seat's next decision is a first again."""
        for info in self._games.values():
            info["decided"].clear()

    def _seed(self, view: Mapping[str, Any], seat: int) -> int:
        if self.seed_fn is not None:
            return int(self.seed_fn(view, seat))
        game_id = str((view.get("game") or {}).get("id") or "game").replace("|", "/")
        return derive_seed(game_id, "decide", seat, len(view["action_records"]))

    def _bind(self, game: cpp.Game) -> None:
        """Play at the agent's dims when the game fits them, else (a graph
        network) at dims that cover both; a network tied to one board (cnn)
        must also recognise the game's map."""
        from catanrl.envs.encoding import covering_dims

        need = covering_dims([game.board.map], game.num_players)
        have = self.native.dims
        if any(need[k] > have[k] for k in need):
            if not getattr(self.native, "dims_agnostic", False):
                raise ViewError(f"{self.label} plays at dims {have}; this game needs {need}")
            self.native.bind({k: max(need[k], have[k]) for k in need})
            self._restart_streams()
        try:
            self.native.check_maps([game.board.map])
        except ValueError as error:
            raise ViewError(f"{self.label}: {error}") from None

    # ------------------------------------------------------------ reporting

    def summary(self) -> dict[str, Any]:
        """Decisions and fallbacks, replay repairs and tracking events, summed
        over the seats this agent played."""
        repairs: Counter[str] = Counter()
        tracking: Counter[str] = Counter()
        for tracker in self.seats.values():
            repairs.update(tracker.replay.repairs)
            tracking.update(tracker.stats)
        out = {"agent": self.label, "decisions": dict(self.stats), "replay_repairs": dict(repairs),
               "tracking": dict(tracking)}
        counters = getattr(self.native, "counters", None)
        if counters is not None:  # MCTS: fresh and reused searches, refusals
            out["search"] = dict(counters)
        return out
