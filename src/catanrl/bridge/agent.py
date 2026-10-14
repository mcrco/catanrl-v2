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

Decision seeds are ``catanrl.seeding.derive_seed(game_id, "decide", seat,
index)``, ``index`` being the number of log entries before the decision: a
pure function of the view, so a transcript fixes every decision's seed. (The
native harness counts the seat's own decisions instead, which a view cannot
reproduce after a reconnect or a TUI undo.) ``seed_fn`` overrides it.

Nothing falls back silently. A seat whose replay is broken (until it
re-anchors at its next turn) decides on the ``apply_setup`` snapshot when
the prompt allows, and otherwise plays the server's default move; a position
that differs from the view is used but counted; a chosen action that was not
offered is never mapped to another one. All of it is counted in
:attr:`ViewAgent.stats`, and ``strict=True`` (the tests) raises
:class:`BridgeError` instead.
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
    Inexpressible,
    Position,
    SeatView,
    ViewError,
    color_name,
    snapshot,
)
from catanrl.bridge.tracking import SeatTracker
from catanrl.seeding import derive_seed

logger = logging.getLogger(__name__)

#: What a seat plays when the bridge cannot decide (catanatron-server's
#: ``default_action``): roll, else end the turn, else the first offered move.
DEFAULT_PREFERENCE = ("ROLL", "END_TURN")
GAME_MESSAGES = ("before", "decide", "step", "after")


class BridgeError(RuntimeError):
    """A decision the bridge cannot make exactly (raised with ``strict=True``)."""


class UnfairAgent(ValueError):
    """An agent that would read hidden information from a view's game."""


def native_agent(agent: Any, *, device: str = "cpu", infer_rows: int = 1,
                 mode: str = "strict") -> tuple[Any, bool | None, str]:
    """(native agent, resolved ``fair`` if known, label) for an agent spec
    (resolved with ``harness="server"``), a resolved agent or a native agent."""
    from catanrl.agents.native import BotAgent, NativeAgent

    if isinstance(agent, str):
        from catanrl.agents.spec import resolve_agent

        agent = resolve_agent(agent, harness="server")
    if isinstance(agent, (NativeAgent, BotAgent)):
        return agent, None, str(getattr(agent, "key", "agent"))
    if callable(getattr(agent, "native", None)):
        native = agent.native(device=device, infer_rows=infer_rows, mode=mode)
        return native, getattr(agent, "fair", None), str(getattr(agent, "label", agent))
    if hasattr(agent, "key") and hasattr(agent, "params"):
        from catanrl.agents.native import make_native_agent

        native = make_native_agent(agent, device=device, infer_rows=infer_rows, mode=mode)
        return native, getattr(agent, "fair", None), str(agent.key)
    raise TypeError(f"cannot make a native agent of {agent!r}")


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
    """

    def __init__(self, agent: Any, *, device: str = "cpu", infer_rows: int = 1, mode: str = "strict",
                 supply: Any = None, strict: bool = False,
                 seed_fn: Callable[[Mapping[str, Any], int], int] | None = None, max_games: int = 8) -> None:
        self.native, fair, self.label = native_agent(agent, device=device, infer_rows=infer_rows, mode=mode)
        if fair is False:
            raise UnfairAgent(f"{self.label} is not fair on views (DESIGN 6.5)")
        check_fair(self.native)
        self.supply = supply
        self.strict = strict
        self.seed_fn = seed_fn
        self.max_games = int(max_games)
        self.stats: Counter[str] = Counter()
        #: (game id, color) -> the seat's tracker, most recent last.
        self.seats: OrderedDict[tuple[Any, str], SeatTracker] = OrderedDict()
        self._games: OrderedDict[Any, dict[str, Any]] = OrderedDict()
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
        if body.get("type") in GAME_MESSAGES:
            self._inbox.append(dict(body))

    def _game(self, game_id: Any) -> dict[str, Any]:
        info = self._games.get(game_id)
        if info is None:
            info = self._games[game_id] = {"supply": None, "sizes": {}}
            while len(self._games) > self.max_games:
                self._games.popitem(last=False)
        return info

    def _drain(self) -> None:
        while self._inbox:
            body = self._inbox.popleft()
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
                    if tracker.sync(view).method == "replay":
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
            tracker = SeatTracker(view, color, supply=supply)
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
        hand_tracker: cpp.HandTracker = tracker.tracker
        if position.method == "broken":
            if self.strict:
                raise BridgeError(f"the replay is broken: {tracker.broken}")
            try:
                position = snapshot(view, color=tracker.color, rules=tracker.rules, catan_map=tracker.catan_map)
            except (Inexpressible, ViewError, ValueError) as error:
                return self._fallback("no_position", playable_actions, f"{tracker.broken}; {error}")
            self.stats["snapshot"] += 1
            hand_tracker = cpp.HandTracker(position.game, seat)
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
        if position.method == "replay":
            tracker.options(offered)
            game = tracker.game  # the friendly robber may have redealt its placeholders
        else:
            hand_tracker.observe_options(game, offered)

        row = self.native.encode(game, seat)
        if not np.array_equal(row[-mask.size:], mask):
            self.stats["mask_differs"] += 1
            if self.strict and position.exact:
                raise BridgeError("the game's legal actions differ from the offered ones")
            row = row.copy()
            row[-mask.size:] = mask
        seed = self._seed(view, seat)
        index = int(self.native.act(
            row[None], mask[None], np.array([seed], dtype=np.uint64),
            games=[game] if getattr(self.native, "needs_games", False) else None,
            trackers=[hand_tracker] if getattr(self.native, "needs_trackers", False) else None)[0])
        chosen = space.decode(game, index)
        key = action_key(action_to_json(to_catanatron_action(chosen)))
        for action in playable_actions:
            if action_key(list(action)) == key:
                return action
        return self._fallback("not_offered", playable_actions, f"{chosen} was not offered")

    def _seed(self, view: Mapping[str, Any], seat: int) -> int:
        if self.seed_fn is not None:
            return int(self.seed_fn(view, seat))
        game_id = str((view.get("game") or {}).get("id") or "game").replace("|", "/")
        return derive_seed(game_id, "decide", seat, len(view["action_records"]))

    def _bind(self, game: cpp.Game) -> None:
        """Play at the agent's dims when the game fits them, else (a graph
        network) at dims that cover both."""
        from catanrl.envs.encoding import covering_dims

        need = covering_dims([game.board.map], game.num_players)
        have = self.native.dims
        if all(need[k] <= have[k] for k in need):
            return
        if not getattr(self.native, "dims_agnostic", False):
            raise ViewError(f"{self.label} plays at dims {have}; this game needs {need}")
        self.native.bind({k: max(need[k], have[k]) for k in need})

    # ------------------------------------------------------------ reporting

    def summary(self) -> dict[str, Any]:
        """Decisions and fallbacks, replay repairs and tracking events, summed
        over the seats this agent played."""
        repairs: Counter[str] = Counter()
        tracking: Counter[str] = Counter()
        for tracker in self.seats.values():
            repairs.update(tracker.replay.repairs)
            tracking.update(tracker.stats)
        return {"agent": self.label, "decisions": dict(self.stats), "replay_repairs": dict(repairs),
                "tracking": dict(tracking)}
