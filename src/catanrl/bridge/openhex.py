"""Our agents on OpenHex (https://open-hex.web.app), a REST arena for Catan bots.

OpenHex runs its own rules engine and speaks its own JSON: cube coordinates
for hexes and vertices, an event stream per game, and actions that differ
from Catanatron's in a few places (a knight names its hex, the robber's
steal is a separate action, discards go in one batch, free roads have their
own action). This module puts an OpenHex game into the bridge's terms:

* :class:`Board`: the board as a Catanatron map (and so a cppanatron one),
  with OpenHex vertices matched to Catanatron nodes by geometry (a vertex is
  the corner its three hexes share);
* :class:`EventLog`: OpenHex events as Catanatron log entries, which
  :class:`catanrl.bridge.replay.Replay` replays for the seat (placeholders
  for what it cannot see) while :func:`~catanrl.bridge.tracking.transition_events`
  feeds the seat's hand tracker, exactly as for a Catanatron view;
* :class:`OpenHexSeat`: one game for one of our seats, which decides with a
  native agent (POLICY, BELIEF, MCTS, a reusing MCTS on the seat's own
  stream) on the replayed game and answers with OpenHex actions;
* :class:`OpenHexBot`: the API client, which joins the queue, plays every
  game it is matched into, and joins again.

Nothing here reads hidden information: OpenHex sends each bot its own view.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cppanatron as cpp
import numpy as np

from catanrl.bridge.position import DEVELOPMENT_CARDS, RESOURCES
from catanrl.bridge.replay import BankShort, Replay
from catanrl.bridge.tracking import points_event, transition_events
from catanrl.seeding import derive_seed

logger = logging.getLogger(__name__)

API = "https://open-hex.web.app/api"
#: OpenHex's public game records (the site reads them for its replays).
FIRESTORE = "https://firestore.googleapis.com/v1/projects/open-hex/databases/(default)/documents/games/{}"
RES = {"wood": "WOOD", "brick": "BRICK", "sheep": "SHEEP", "wheat": "WHEAT", "ore": "ORE"}
RES_NAME = {v: k for k, v in RES.items()}
TERRAIN = {"forest": "WOOD", "hills": "BRICK", "pasture": "SHEEP", "fields": "WHEAT", "mountains": "ORE",
           "desert": None}
CARDS = {"knight": "KNIGHT", "yearOfPlenty": "YEAR_OF_PLENTY", "monopoly": "MONOPOLY",
         "roadBuilding": "ROAD_BUILDING", "victoryPoint": "VICTORY_POINT"}
#: OpenHex's starting supply (the base game's).
BANK = (19, 19, 19, 19, 19)
DECK = (14, 2, 2, 2, 5)
#: Events whose effects the replay derives itself (a roll's production is
#: the exception: the game pays in full past its bank, the engine would not,
#: so the ``resourcesProduced`` after a roll is what the roll pays).
DERIVED = frozenset({"setupResourcesGranted", "setupResourcesCollected", "resourcesProduced",
                     "longestRoadAwarded", "longestRoadClaimed", "largestArmyAwarded", "largestArmyClaimed",
                     "gameWon"})
AWARDS = frozenset({"longestRoadAwarded", "longestRoadClaimed", "largestArmyAwarded", "largestArmyClaimed"})


class OpenHexError(RuntimeError):
    """Something OpenHex sent that the bridge cannot follow."""


# ---------------------------------------------------------------- coordinates


def cube(point: Any) -> tuple[int, int, int]:
    if isinstance(point, Mapping):
        return int(point["x"]), int(point["y"]), int(point["z"])
    x, y, z = point
    return int(x), int(y), int(z)


def field_of(data: Mapping[str, Any], *names: str) -> Any:
    """The first of ``names`` that ``data`` has (OpenHex renamed some fields)."""
    for name in names:
        if name in data and data[name] is not None:
            return data[name]
    raise OpenHexError(f"none of {names} in {data!r}"[:300])


def vertex_hexes(v: tuple[int, int, int]) -> tuple[tuple[int, int, int], ...]:
    """The three hexes meeting at vertex ``v`` (OpenHex's convention)."""
    x, y, z = v
    if x + y + z == 1:
        return (x, y - 1, z), (x - 1, y, z), (x, y, z - 1)
    return (x - 1, y, z - 1), (x, y - 1, z - 1), (x - 1, y - 1, z)


def hex_vertices(h: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    x, y, z = h
    return [(x, y + 1, z), (x + 1, y, z), (x, y, z + 1), (x + 1, y, z + 1), (x, y + 1, z + 1), (x + 1, y + 1, z)]


_SQ3 = math.sqrt(3)
_CORNERS = {"NORTH": (0.0, -1.0), "NORTHEAST": (_SQ3 / 2, -0.5), "SOUTHEAST": (_SQ3 / 2, 0.5),
            "SOUTH": (0.0, 1.0), "SOUTHWEST": (-_SQ3 / 2, 0.5), "NORTHWEST": (-_SQ3 / 2, -0.5)}


def _center(c: Sequence[int]) -> tuple[float, float]:
    """A Catanatron tile's center (pointy-top, unit size)."""
    return _SQ3 * (c[0] + c[2] / 2), 1.5 * c[2]


class Board:
    """An OpenHex board (``tiles`` and ``ports``) as a Catanatron map.

    Hex coordinates carry over as they are (both are cube coordinates with
    the same neighbours); vertices become the Catanatron nodes at the corner
    their three hexes share, and edges the node pairs.
    """

    def __init__(self, board: Mapping[str, Any]) -> None:
        from catanatron.models.coordinate_system import UNIT_VECTORS
        from catanatron.server.maps import parse_map
        from cppanatron.catanatron import map_spec_from_catanatron

        self.doc = {"tiles": list(board["tiles"]), "ports": list(board.get("ports") or [])}
        land: dict[tuple, tuple[str | None, int | None]] = {}
        for tile in board["tiles"]:
            terrain = tile["terrain"]
            if terrain not in TERRAIN:
                raise OpenHexError(f"unknown terrain {terrain!r}")
            land[cube(field_of(tile, "hex", "location"))] = (TERRAIN[terrain], tile.get("number"))
        direction_of = {vector: d.value for d, vector in UNIT_VECTORS.items()}
        ports: dict[tuple, dict] = {}
        for port in board.get("ports") or []:
            a, b = (cube(v) for v in field_of(port, "edge", "vertices"))
            shared = set(vertex_hexes(a)) & set(vertex_hexes(b))
            water = [h for h in shared if h not in land]
            ground = [h for h in shared if h in land]
            if len(water) != 1 or len(ground) != 1:
                raise OpenHexError(f"port {port!r} is not on a coastal edge")
            towards = tuple(g - w for g, w in zip(ground[0], water[0]))
            resource = port.get("resource")
            ports[water[0]] = {"coordinate": list(water[0]), "type": "port", "direction": direction_of[towards],
                               "port_resource": RES[resource] if resource else None}
        tiles = [{"coordinate": list(c), "type": "land", "resource": r, "number": n if r else None}
                 for c, (r, n) in land.items()]
        tiles += list(ports.values())
        coast = {h for c in land for h in (tuple(a + b for a, b in zip(c, u)) for u in UNIT_VECTORS.values())
                 if h not in land and h not in ports}
        tiles += [{"coordinate": list(c), "type": "water"} for c in sorted(coast)]
        self.catan_map = parse_map({"tiles": tiles})
        self.map = cpp.CatanMap.from_spec(map_spec_from_catanatron(self.catan_map))

        corners: dict[int, tuple[float, float]] = {}
        for coordinate, tile in self.catan_map.land_tiles.items():
            cx, cy = _center(coordinate)
            for ref, node in tile.nodes.items():
                dx, dy = _CORNERS[ref.value]
                corners[node] = (cx + dx, cy + dy)
        self.node: dict[tuple, int] = {}
        for h in land:
            for v in hex_vertices(h):
                if v in self.node:
                    continue
                points = [_center(c) for c in vertex_hexes(v)]
                px, py = sum(p[0] for p in points) / 3, sum(p[1] for p in points) / 3
                node = min(corners, key=lambda n: (corners[n][0] - px) ** 2 + (corners[n][1] - py) ** 2)
                if math.dist(corners[node], (px, py)) > 1e-6:
                    raise OpenHexError(f"vertex {v} matches no node")
                self.node[v] = node
        if len(set(self.node.values())) != len(corners):
            raise OpenHexError("the vertices and nodes do not correspond one to one")
        self.vertex = {node: v for v, node in self.node.items()}
        self.robber = cube(board["robber"]) if board.get("robber") else None

    def node_of(self, vertex: Any) -> int:
        return self.node[cube(vertex)]

    def edge_of(self, vertices: Sequence[Any]) -> tuple[int, int]:
        a, b = (self.node_of(v) for v in vertices)
        return (a, b) if a < b else (b, a)

    def vertex_doc(self, node: int) -> dict:
        x, y, z = self.vertex[int(node)]
        return {"x": x, "y": y, "z": z}

    def hex_doc(self, coordinate: Sequence[int]) -> dict:
        x, y, z = (int(c) for c in coordinate)
        return {"x": x, "y": y, "z": z}


def openhex_rules(config: Mapping[str, Any], bank_margin: Sequence[int] = (0, 0, 0, 0, 0)) -> cpp.GameRules:
    """The game's rules; ``bank_margin`` is added to the bank (OpenHex pays
    rolls in full past its bank, which the engine's rule never does)."""
    return cpp.GameRules(
        vps_to_win=int(config.get("victoryPoints", 10)),
        discard_limit=int(config.get("discardLimit", 7)),
        friendly_robber=str(config.get("robberMode", "normal")).lower() == "friendly",
        resource_bank=tuple(b + int(m) for b, m in zip(BANK, bank_margin)), development_deck=DECK,
        roads_per_player=int(config.get("maxRoads", 15)),
        settlements_per_player=int(config.get("maxSettlements", 5)),
        cities_per_player=int(config.get("maxCities", 4)))


def color_of(name: str) -> str:
    return str(name).upper()


def _counts(resources: Mapping[str, int]) -> list[int]:
    return [int(resources.get(RES_NAME[r], 0)) for r in RESOURCES]


# ---------------------------------------------------------------- events as a log


@dataclass
class Entry:
    """A Catanatron log entry, with what the replay needs beside it."""

    record: list
    #: A monopoly's take per seat (OpenHex's ``collected``).
    amounts: list[int] | None = None
    #: A roll's payout per seat (OpenHex's ``production``); None lets the
    #: engine's bank rule pay.
    produced: list[list[int]] | None = None


class EventLog:
    """OpenHex events in, Catanatron log entries out.

    A roll waits for the ``resourcesProduced`` that follows it in the same
    instant (nothing follows when nothing was produced, or on a 7), so the
    replay pays what the game paid. A robber move waits for the steal that
    may follow it (award events can come between); a knight is its card plus
    the robber move it carries. Discards are held until the next other event
    and come out in seat order, one card each, as Catanatron's discard prompt
    takes them.
    """

    def __init__(self, board: Board, colors: Sequence[str]) -> None:
        self.board = board
        self.colors = list(colors)
        self.robber: tuple[str, tuple] | None = None
        self.discards: dict[str, list[str]] = {}
        self.won: str | None = None
        self.roll: Entry | None = None

    def feed(self, event: Mapping[str, Any]) -> list[Entry]:
        kind = event.get("type")
        out: list[Entry] = []
        if kind == "gameWon":
            # The game can end on a knight (Largest Army) before its steal:
            # whatever was pending never happens.
            self.won = color_of(event["player"])
            self.robber, self.discards = None, {}
            return self._flush_roll(None)
        if self.roll is not None:
            out += self._flush_roll(self._production(event) if kind == "resourcesProduced"
                                    else [[0] * 5 for _ in self.colors])
            if kind == "resourcesProduced":
                return out
        if kind != "resourcesDiscarded":
            out += self._flush_discards()
        if self.robber is not None and kind != "resourceStolen" and kind not in AWARDS:
            out += self._flush_robber(None, None)
        color = color_of(event["player"]) if event.get("player") else None
        if kind in ("setupSettlementPlaced", "settlementBuilt"):
            out.append(Entry([[color, "BUILD_SETTLEMENT", self.board.node_of(event["vertex"])], None]))
        elif kind == "cityBuilt":
            out.append(Entry([[color, "BUILD_CITY", self.board.node_of(event["vertex"])], None]))
        elif kind in ("setupRoadPlaced", "roadBuilt", "freeRoadPlaced"):
            edge = self.board.edge_of(field_of(event, "edge", "vertices"))
            out.append(Entry([[color, "BUILD_ROAD", list(edge)], None]))
        elif kind == "diceRolled":
            dice = event.get("dice") or [event["die1"], event["die2"]]
            self.roll = Entry([[color, "ROLL", None], [int(dice[0]), int(dice[1])]])
        elif kind == "robberMoved":
            self.robber = (color, cube(field_of(event, "hex", "location")))
        elif kind == "knightPlayed":
            out.append(Entry([[color, "PLAY_KNIGHT_CARD", None], None]))
            self.robber = (color, cube(field_of(event, "hex", "location")))
        elif kind == "resourceStolen":
            if self.robber is None or self.robber[0] != color:
                raise OpenHexError(f"a steal without a robber move: {event!r}")
            resource = event.get("resource")
            out += self._flush_robber(color_of(event["target"]), RES[resource] if resource else None)
        elif kind == "resourcesDiscarded":
            cards = [r for r, n in zip(RESOURCES, _counts(event["resources"])) for _ in range(n)]
            self.discards.setdefault(color, []).extend(cards)
        elif kind == "devCardBought":
            card = event.get("card")
            out.append(Entry([[color, "BUY_DEVELOPMENT_CARD", None], CARDS[card] if card else None]))
        elif kind == "yearOfPlentyPlayed":
            pair = sorted((RES[r] for r in event["resources"]), key=RESOURCES.index)
            out.append(Entry([[color, "PLAY_YEAR_OF_PLENTY", pair], None]))
        elif kind == "monopolyPlayed":
            collected = {color_of(c): int(n) for c, n in (event.get("collected") or {}).items()}
            amounts = [0 if c == color else collected.get(c, 0) for c in self.colors]
            out.append(Entry([[color, "PLAY_MONOPOLY", RES[event["resource"]]], None], amounts=amounts))
        elif kind == "roadBuildingPlayed":
            out.append(Entry([[color, "PLAY_ROAD_BUILDING", None], None]))
        elif kind == "bankTradeExecuted":
            (give, n), = [(RES[r], int(k)) for r, k in event["offering"].items() if int(k)]
            (take, m), = [(RES[r], int(k)) for r, k in event["requesting"].items() if int(k)]
            if n % m:
                raise OpenHexError(f"an uneven bank trade: {event!r}")
            rate = n // m
            value = [give] * rate + [None] * (4 - rate) + [take]
            out += [Entry([[color, "MARITIME_TRADE", value], None]) for _ in range(m)]
        elif kind == "turnEnded":
            out.append(Entry([[color, "END_TURN", None], None]))
        elif kind == "resourcesProduced":
            logger.warning("openhex: production after its roll was replayed (the events came apart); "
                           "the engine's own payout stands")
        elif kind not in DERIVED:
            logger.warning("openhex: an event the bridge does not know: %s", json.dumps(event)[:200])
        return out

    def _production(self, event: Mapping[str, Any]) -> list[list[int]]:
        amounts = [[0] * 5 for _ in self.colors]
        for color, got in (event.get("production") or {}).items():
            seat = self.colors.index(color_of(color))
            for resource, n in (got or {}).items():
                amounts[seat][RESOURCES.index(RES[resource])] += int(n)
        return amounts

    def _flush_roll(self, produced: list[list[int]] | None) -> list[Entry]:
        if self.roll is None:
            return []
        entry, self.roll = self.roll, None
        entry.produced = produced
        return [entry]

    def _flush_robber(self, victim: str | None, stolen: str | None) -> list[Entry]:
        color, coordinate = self.robber  # type: ignore[misc]
        self.robber = None
        return [Entry([[color, "MOVE_ROBBER", [list(coordinate), victim]], stolen])]

    def settle(self, *, robber: bool = True, discards: bool = True) -> list[Entry]:
        """What is held back, as it stands: a roll whose production has not
        come (the engine's rule pays it), a robber move that stole from
        nobody, discards that are complete. (OpenHex's state says when: the
        events alone cannot, when the next event is ours to make.)"""
        out: list[Entry] = self._flush_roll(None)
        if discards:
            out += self._flush_discards()
        if robber and self.robber is not None:
            out += self._flush_robber(None, None)
        return out

    def _flush_discards(self) -> list[Entry]:
        out = [Entry([[c, "DISCARD_RESOURCE", r], None]) for c in self.colors for r in self.discards.get(c, [])]
        self.discards = {}
        return out


# ---------------------------------------------------------------- one seat of one game


def native_for(spec: Any, *, device: str = "cpu") -> tuple[Any, str]:
    """The native agent of ``spec`` on a seat's view: fair (DESIGN 6.5) and
    not recurrent (the replay keeps no decision history). One that keeps a
    seat's search between its decisions (MCTS with ``reuse``) is handed the
    seat's stream by :class:`OpenHexSeat`, and compares the game it is
    handed as the seat sees it (``reuse_check = "view"``, as the ViewAgent
    sets it: the replay's placeholders and RNG are its own)."""
    from catanrl.bridge.agent import check_fair, native_agent

    native, fair, label = native_agent(spec, device=device, memory=True)
    if fair is False:
        raise ValueError(f"{label} is not fair on a seat's view")
    check_fair(native)
    if getattr(native, "recurrent", False):
        raise ValueError(f"{label}: a recurrent network needs its seat's decision history, which the replay "
                         "does not keep")
    if getattr(native, "needs_memory", False):
        native.reuse_check = "view"
    return native, label


class _Slots:
    """Game slots for a reusing agent's streams (``env * max_players + seat``,
    as the harness numbers them, the slot being the env): one per live seat,
    process-wide, a freed one taken again first (as the ViewAgent recycles
    its slots), so the ids do not grow with the games played."""

    def __init__(self) -> None:
        self._free: list[int] = []
        self._taken = 0
        self._lock = threading.Lock()

    def take(self) -> int:
        with self._lock:
            if self._free:
                return self._free.pop()
            self._taken += 1
            return self._taken - 1

    def give(self, env: int) -> None:
        with self._lock:
            self._free.append(env)


SLOTS = _Slots()


def default_of(legal: Sequence[cpp.Action]) -> cpp.Action:
    """catanatron-server's default move: roll, else end the turn, else the first."""
    for kind in ("ROLL", "END_TURN"):
        for action in legal:
            if action.action_type.name == kind:
                return action
    return legal[0]


@dataclass
class Plan:
    """What the seat is about to send, and what it decided ahead."""

    actions: list[dict]
    #: The robber's victim, for the steal OpenHex asks for next.
    victim: str | None = None


@dataclass
class OpenHexSeat:
    """One of our seats in one OpenHex game.

    A reusing agent (MCTS with ``reuse``) keeps the seat's search between its
    decisions: the seat hands it one stream (a slot of :data:`SLOTS` as the
    env) with ``first`` at its first decision, as the ViewAgent does, and a
    copy of the game, since the replay moves its own on in place. A plan
    decided ahead on a copy (the robber after a knight, each discard of a
    chain after the first) re-roots the search through the agent's view check
    like any own move. A discard chain deals its worlds from a tracker of its
    copy, which knows nothing of the other hands, so its search must not reach
    the decision after it (the robber, when we rolled the seven): the stream
    restarts after the chain, and that decision deals from the seat's tracker.
    The game replayed again on a larger bank (:class:`BankShort`) leaves the
    remembered game stale, so the stream restarts there, as it does after an
    error in the agent (which emptied its pool); :meth:`close` at the game's
    end drops the searches and frees the slot.
    """

    native: Any
    game_id: str
    state: Mapping[str, Any]
    color: str
    stats: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        state = self.state
        self.board = Board(state["board"])
        self.colors = [color_of(c) for c in state["turnOrder"]]
        self.seat = self.colors.index(color_of(self.color))
        #: The agent keeps a seat's search between its decisions: it is handed
        #: this seat's stream and ``first`` (see the class doc).
        self.memory = bool(getattr(self.native, "needs_memory", False))
        self.env = SLOTS.take() if self.memory else None
        #: Whether a decision reached the agent (its stream started).
        self.decided = False
        #: The agent's search counters at the start: this game's share is the difference.
        self._counters = dict(getattr(self.native, "counters", {}))
        #: Added to the bank of the replayed game where OpenHex paid past it.
        self.bank_margin = [0] * 5
        #: The entries replayed so far (replayed again on a larger bank).
        self.entries: list[Entry] = []
        self._build()
        self.log = EventLog(self.board, self.colors)
        self.last_seq = -1
        self.victim: str | None = None
        self.discarded_at: int | None = None
        self.waited = 0
        self.decisions = 0
        self.fallbacks = 0

    def _build(self) -> None:
        """A fresh replay and tracker on the rules, with the bank margin."""
        self.rules = openhex_rules(self.state.get("config") or {}, self.bank_margin)
        self.replay = Replay({"colors": self.colors}, self.seat, self.rules, self.board.map)
        self.tracker = cpp.HandTracker(self.replay.game, self.seat)

    @property
    def bank(self) -> list[int]:
        """OpenHex's bank: the replay's less the margin (below zero where
        OpenHex paid past it)."""
        return [int(b) - m for b, m in zip(self.replay.game.resource_bank, self.bank_margin)]

    # ------------------------------------------------------------ following

    def feed(self, events: Sequence[Mapping[str, Any]]) -> None:
        for item in sorted(events, key=lambda e: e["seq"]):
            if item["seq"] <= self.last_seq:
                continue
            self.last_seq = item["seq"]
            event = item["event"] if "event" in item else item
            for entry in self.log.feed(event):
                self._step(entry)

    def _step(self, entry: Entry) -> None:
        try:
            self._apply(entry)
        except BankShort as short:
            # OpenHex paid a roll its bank could not; the engine pays nobody
            # then. The game replays on a bank that much larger: the hands
            # and the tracker stay exact, the bank features scale by it.
            self.bank_margin = [m + d for m, d in zip(self.bank_margin, short.deficit)]
            self.stats["bank_short"] = self.stats.get("bank_short", 0) + 1
            logger.info("game %s: the bank ran short by %s; replaying on a bank of %s", self.game_id,
                        short.deficit, [b + m for b, m in zip(BANK, self.bank_margin)])
            entries, self.entries = self.entries, []
            self._build()
            self._restart_stream()  # the agent's remembered game is the old bank's
            for past in entries:
                self._apply(past)
            self._apply(entry)

    def _apply(self, entry: Entry) -> None:
        sizes = None
        if entry.amounts is not None:
            game = self.replay.game
            sizes = [sum(game.player(i).resources) for i in range(len(self.colors))]
            total = sum(entry.amounts)
            sizes = [s + total if i == self.colors.index(entry.record[0][0]) else s - entry.amounts[i]
                     for i, s in enumerate(sizes)]
        tr = self.replay.step(entry.record, sizes_after=sizes, produced=entry.produced)
        if self.replay.repairs["production"]:
            self.stats["production_mismatch"] = self.replay.repairs["production"]
        events = transition_events(tr, self.seat, amounts=entry.amounts)
        if self.replay.game.winning_color is None:
            events.append(points_event(tr))
        for event in events:
            self.tracker.apply(event)
        self.entries.append(entry)

    @property
    def over(self) -> bool:
        return self.log.won is not None or self.replay.game.winning_color is not None

    @property
    def search(self) -> dict[str, int]:
        """The agent's search counters over this game (MCTS: searches fresh
        and reused, refusals; empty for an agent that keeps none)."""
        counters = getattr(self.native, "counters", None)
        if counters is None:
            return {}
        return {k: int(v) - self._counters.get(k, 0) for k, v in counters.items()}

    def close(self) -> None:
        """The game is over (or given up): drop the agent's searches of it and
        free its slot, so they do not linger while the bot plays on."""
        if self.env is not None:
            self.native.forget(self.env)
            SLOTS.give(self.env)
            self.env = None

    def _restart_stream(self) -> None:
        """The agent's search of this game is stale (the game replayed again
        on a larger bank), dealt from a tracker that knows less than the
        seat's (a discard chain's) or gone (a bind to new dims): drop it, and
        the seat's next decision is a first again."""
        if self.env is not None:
            self.native.forget(self.env)
        self.decided = False

    def mismatches(self, state: Mapping[str, Any]) -> list[str]:
        """What the replayed game shows differently from OpenHex's state."""
        game = self.replay.game
        out = []
        if game.is_initial_build_phase:
            # OpenHex grants the second settlement's resources after its
            # road, Catanatron at the settlement: hands differ in between.
            return out
        bank = state.get("bank") or state.get("resourceBank")
        if bank and self.bank != _counts(bank):
            out.append(f"bank {tuple(self.bank)} vs {_counts(bank)}")
        for p in state.get("players") or []:
            i = self.colors.index(color_of(p["color"]))
            ours = game.player(i)
            size = p.get("handSize", sum((p.get("hand") or {}).values()))
            if i == self.seat and tuple(ours.resources) != tuple(_counts(p.get("hand") or {})):
                out.append(f"our hand {tuple(ours.resources)} vs {_counts(p.get('hand') or {})}")
            elif sum(ours.resources) != int(size):
                out.append(f"{p['color']} hand size {sum(ours.resources)} vs {size}")
        robber = (state.get("board") or {}).get("robberHex") or (state.get("board") or {}).get("robber")
        if robber and tuple(game.board.robber_coordinate) != cube(robber):
            out.append(f"robber {tuple(game.board.robber_coordinate)} vs {cube(robber)}")
        return out

    # ------------------------------------------------------------ deciding

    def plan(self, state: Mapping[str, Any]) -> Plan | None:
        """The actions to send now, or None when it is not our move."""
        if self.over:
            return None
        for entry in self.log.settle(robber=False, discards=False):
            self._step(entry)
        pending = state.get("pendingAction") or state.get("pending") or {}
        kind = str(pending.get("type", "")).lower() if isinstance(pending, Mapping) else str(pending or "").lower()
        mine = color_of(state.get("currentPlayer") or "") == self.colors[self.seat]
        ours = self.log.robber is not None and self.log.robber[0] == self.colors[self.seat]
        if "discard" in kind:
            # Someone still discards: ours, if the replay says so and it has not gone out.
            game = self.replay.game
            if game.is_discarding and game.remaining_discards(self.seat) > 0 \
                    and self.discarded_at != len(self.replay.entries):
                return self._discard()
            return None
        # Nobody discards any more: what was discarded is complete.
        for entry in self.log.settle(robber=False):
            self._step(entry)
        if "steal" in kind:
            if ours:
                return Plan([{"type": "stealFromPlayer", "target": self._victim()}])
            return None
        if self.log.robber is not None:
            # A robber move with no steal: known complete once OpenHex asks
            # for something else of the mover (or the turn moved on).
            if ours and self.victim is not None and self.waited < 3:
                self.waited += 1  # a steal's event may still be on its way
                return None
            if ours or mine:
                self.waited = 0
                for entry in self.log.settle():
                    self._step(entry)
            else:
                return None
        game = self.replay.game
        if game.is_discarding or game.current_player_index != self.seat or not mine:
            return None
        if "roll" in kind:
            # OpenHex takes nothing before the roll (Catanatron would let a
            # development card go first).
            return Plan([{"type": "rollDice"}])
        return self._decide(game)

    def _victim(self) -> str:
        if self.victim is not None:
            return self.victim.lower()
        others = [c for i, c in enumerate(self.colors) if i != self.seat]
        return others[0].lower()

    def _choose(self, game: cpp.Game, tracker: cpp.HandTracker | None) -> cpp.Action:
        native = self.native
        legal = list(game.playable_actions)
        if len(legal) == 1:
            return legal[0]
        self._bind(game)
        space = native.action_space
        mask = np.zeros(int(space.size), dtype=np.float32)
        for action in legal:
            try:
                mask[int(space.index(game, action))] = 1.0
            except (IndexError, ValueError):
                pass
        if not mask.any():
            return legal[0]
        row = native.encode(game, self.seat).copy()
        row[-mask.size:] = mask
        seed = derive_seed(self.game_id, "decide", self.seat, len(self.replay.entries) + self.decisions)
        self.decisions += 1
        if tracker is None:
            tracker = cpp.HandTracker(game, self.seat)
        try:
            chosen = space.decode(game, self._act(row, mask, seed, game, tracker))
        except (ValueError, RuntimeError) as error:
            # (The last moves of a game whose winner's victory point cards
            # are hidden: the sampler sees a position nobody could be in.)
            self.fallbacks += 1
            logger.warning("game %s: the agent could not decide (%s); playing the default move",
                           self.game_id, str(error)[:120])
            return default_of(legal)
        if not any(chosen == a for a in legal):
            self.fallbacks += 1
            return default_of(legal)
        return chosen

    def _act(self, row: np.ndarray, mask: np.ndarray, seed: int, game: cpp.Game, tracker: cpp.HandTracker) -> int:
        """The agent's index for ``game``; a reusing agent decides on the seat's stream."""
        native = self.native
        seeds = np.array([seed], dtype=np.uint64)
        trackers = [tracker] if getattr(native, "needs_trackers", False) else None
        if not self.memory:
            games = [game] if getattr(native, "needs_games", False) else None
            return int(native.act(row[None], mask[None], seeds, games=games, trackers=trackers)[0])
        streams = np.array([self.env * int(native.dims["max_players"]) + self.seat], dtype=np.int64)
        first = np.array([not self.decided], dtype=bool)
        try:
            # The agent keeps the game it is handed until the seat's next decision: the
            # replay moves its game on in place, and a plan decided ahead moves its copy.
            index = native.act(row[None], mask[None], seeds, games=[game.copy()], trackers=trackers,
                               streams=streams, first=first)[0]
        except BaseException:
            self.decided = False  # the agent emptied its pool and forgot its streams
            raise
        self.decided = True
        return int(index)

    def _bind(self, game: cpp.Game) -> None:
        from catanrl.envs.encoding import covering_dims

        need = covering_dims([game.board.map], game.num_players)
        have = self.native.dims
        if all(need[k] <= have[k] for k in need):
            return
        if not getattr(self.native, "dims_agnostic", False):
            raise OpenHexError(f"the agent plays at dims {have}; this game needs {need}")
        self.native.bind({k: max(need[k], have[k]) for k in need})
        self._restart_stream()

    def _decide(self, game: cpp.Game) -> Plan:
        action = self._choose(game, self.tracker)
        kind = action.action_type.name
        if kind == "PLAY_KNIGHT_CARD":
            # OpenHex takes the knight with its hex: decide the robber now.
            ahead = game.copy()
            ahead.execute(action)
            tracker = self.tracker.copy()
            tracker.apply(cpp.TrackerEvent.play_development_card(self.seat, cpp.DevelopmentCard.KNIGHT))
            tracker.apply(cpp.TrackerEvent.public_victory_points(
                [int(ahead.player(i).victory_points) for i in range(len(self.colors))]))
            robber = self._choose(ahead, tracker)
            coordinate, victim = self._robber(robber)
            self.victim = victim
            return Plan([{"type": "playKnight", "location": self.board.hex_doc(coordinate)}], victim)
        return self._translate(game, action)

    def _robber(self, action: cpp.Action) -> tuple[list, str | None]:
        from catanatron.serialization import action_to_json
        from cppanatron.catanatron import to_catanatron_action

        _, kind, value = action_to_json(to_catanatron_action(action))
        if kind != "MOVE_ROBBER":
            raise OpenHexError(f"expected a robber move, got {kind}")
        return value[0], value[1]

    def _translate(self, game: cpp.Game, action: cpp.Action) -> Plan:
        from catanatron.serialization import action_to_json
        from cppanatron.catanatron import to_catanatron_action

        _, kind, value = action_to_json(to_catanatron_action(action))
        initial = game.is_initial_build_phase
        if kind == "ROLL":
            return Plan([{"type": "rollDice"}])
        if kind == "END_TURN":
            return Plan([{"type": "endTurn"}])
        if kind == "BUILD_SETTLEMENT":
            return Plan([{"type": "placeInitialSettlement" if initial else "buildSettlement",
                          "vertex": self.board.vertex_doc(value)}])
        if kind == "BUILD_CITY":
            return Plan([{"type": "buildCity", "vertex": self.board.vertex_doc(value)}])
        if kind == "BUILD_ROAD":
            vertices = [self.board.vertex_doc(n) for n in value]
            name = "placeInitialRoad" if initial else "placeFreeRoad" if game.is_road_building else "buildRoad"
            return Plan([{"type": name, "vertices": vertices}])
        if kind == "BUY_DEVELOPMENT_CARD":
            return Plan([{"type": "buyDevCard"}])
        if kind == "PLAY_ROAD_BUILDING":
            return Plan([{"type": "playRoadBuilding"}])
        if kind == "PLAY_YEAR_OF_PLENTY":
            return Plan([{"type": "playYearOfPlenty", "resources": [RES_NAME[r] for r in value]}])
        if kind == "PLAY_MONOPOLY":
            return Plan([{"type": "playMonopoly", "resource": RES_NAME[value]}])
        if kind == "MARITIME_TRADE":
            give = [r for r in value[:4] if r]
            return Plan([{"type": "tradeWithBank", "offering": {RES_NAME[give[0]]: len(give)},
                          "requesting": {RES_NAME[value[4]]: 1}}])
        if kind == "MOVE_ROBBER":
            self.victim = value[1]
            return Plan([{"type": "moveRobber", "location": self.board.hex_doc(value[0])}], value[1])
        raise OpenHexError(f"no OpenHex action for {kind} {value!r}")

    def _discard(self) -> Plan:
        """Every card this seat must discard, in one batch: decided one at a
        time on a copy, where seats before ours (Catanatron's order) discard
        placeholder cards first. Each decision deals from a tracker of the
        copy (the seat's own knows nothing of the placeholders), so a reusing
        agent's search of the chain ends with it (see the class doc)."""
        ahead = self.replay.game.copy()
        cards = [0] * 5
        asked = self.decisions
        for _ in range(64):
            if not ahead.is_discarding or ahead.remaining_discards(self.seat) == 0:
                break
            if ahead.current_player_index != self.seat:
                ahead.execute(ahead.playable_actions[0])
                continue
            action = self._choose(ahead, None)
            ahead.execute(action)
            cards[RESOURCES.index(action.value.name if hasattr(action.value, "name") else str(action.value))] += 1
        if self.memory and self.decisions > asked:
            self._restart_stream()
        self.discarded_at = len(self.replay.entries)
        return Plan([{"type": "discardResources",
                      "resources": {RES_NAME[r]: n for r, n in zip(RESOURCES, cards) if n}}])

    def fallback(self, state: Mapping[str, Any]) -> list[dict]:
        """A move OpenHex should take when ours was refused: what its
        ``pending`` asks for, else the default move."""
        self.fallbacks += 1
        pending = state.get("pendingAction") or state.get("pending") or {}
        kind = str(pending.get("type", "")).lower() if isinstance(pending, Mapping) else str(pending or "").lower()
        game = self.replay.game
        if "roll" in kind:
            return [{"type": "rollDice"}]
        if "steal" in kind:
            return [{"type": "stealFromPlayer", "target": self._victim()}]
        if "discard" in kind or game.is_discarding:
            return [self._discard().actions[0]]
        legal = list(game.playable_actions)
        if "robber" in kind:
            robbers = [a for a in legal if a.action_type.name == "MOVE_ROBBER"]
            if robbers:
                return self._translate(game, robbers[0]).actions
        return self._translate(game, default_of(legal)).actions


# ---------------------------------------------------------------- the API client


class OpenHexClient:
    def __init__(self, api_key: str, base: str = API, timeout: float = 30.0) -> None:
        self.base = base.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                        "User-Agent": "catanrl-openhex/1"}
        self.timeout = timeout

    def request(self, method: str, path: str, body: Any = None) -> Any:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, data=data, method=method, headers=self.headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as reply:
                text = reply.read().decode()
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:500]
            raise OpenHexHTTPError(error.code, detail) from None
        return json.loads(text) if text else None

    def active_games(self) -> list:
        games = self.request("GET", "/games?status=active")
        return games.get("games", games) if isinstance(games, Mapping) else list(games or [])

    def state(self, game_id: str) -> dict:
        return self.request("GET", f"/games/{game_id}/state")

    def events(self, game_id: str, after: int) -> list:
        reply = self.request("GET", f"/games/{game_id}/events?after={after}")
        return reply.get("events", []) if isinstance(reply, Mapping) else list(reply or [])

    def act(self, game_id: str, action: dict, duration_ms: int) -> Any:
        reply = self.request("POST", f"/games/{game_id}/actions", {"action": action, "duration_ms": duration_ms})
        # A refused action may come back as 200 with an error body.
        if isinstance(reply, Mapping) and (reply.get("success") is False or reply.get("error")):
            raise OpenHexHTTPError(200, str(reply.get("error") or reply)[:500])
        return reply

    def join(self, mode: str) -> Any:
        return self.request("POST", "/queue/join", {"mode": mode})

    def queue_status(self) -> Any:
        return self.request("GET", "/queue/status")

    def leave(self, mode: str) -> Any:
        return self.request("POST", "/queue/leave", {"mode": mode})


class OpenHexHTTPError(OpenHexError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


class OpenHexBot:
    """One registered bot: keep it queued, play every game it gets."""

    def __init__(self, spec: Any, api_key: str, mode: str, *, base: str = API, device: str = "cpu",
                 state_dir: str | Path = "openhex", poll_s: float = 1.0, name: str | None = None) -> None:
        self.native, self.label = native_for(spec, device=device)
        self.client = OpenHexClient(api_key, base)
        self.mode = mode
        self.poll_s = float(poll_s)
        self.name = name or self.label
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        backoff = 5.0
        while True:
            try:
                games = self.client.active_games()
                if games:
                    for game in games:
                        self.play(str(game.get("id") or game.get("game_id")))
                    backoff = 5.0
                    continue
                self._join()
                time.sleep(3.0)
                backoff = 5.0
            except (OpenHexError, OSError, ValueError) as error:
                logger.warning("%s: %s; trying again in %.0f s", self.name, error, backoff)
                time.sleep(backoff)
                backoff = min(300.0, backoff * 2)
            except Exception:  # noqa: BLE001 - a bot thread must outlive any one game
                # (A replay that cannot follow a game, say.) The game is lost
                # either way: OpenHex aborts it at the deadline, after which it
                # is no longer active and the bot queues again.
                logger.exception("%s: giving up on the current game; trying again in %.0f s", self.name, backoff)
                time.sleep(backoff)
                backoff = min(300.0, backoff * 2)

    def _join(self) -> None:
        try:
            self.client.join(self.mode)
        except OpenHexHTTPError as error:
            if error.status not in (400, 409):  # already queued
                raise

    def board_of(self, game_id: str, state: Mapping[str, Any]) -> dict:
        """The game's board: from the state while OpenHex still sends its
        tiles (kept on disk), else from the copy kept, else from OpenHex's
        public game record (so a restart mid-game can go on)."""
        boards = self.state_dir / "boards"
        boards.mkdir(exist_ok=True)
        path = boards / f"{game_id}.json"
        board = dict(state.get("board") or {})
        if board.get("tiles"):
            path.write_text(json.dumps({"tiles": board["tiles"], "ports": board.get("ports") or []}))
            return board
        if path.exists():
            return {**board, **json.loads(path.read_text())}
        record = json.loads(urllib.request.urlopen(FIRESTORE.format(game_id), timeout=self.client.timeout).read())
        archived = json.loads(record["fields"]["boardJson"]["stringValue"])
        path.write_text(json.dumps({"tiles": archived["tiles"], "ports": archived.get("ports") or []}))
        return {**board, "tiles": archived["tiles"], "ports": archived.get("ports") or []}

    def play(self, game_id: str) -> dict:
        reply = self.client.state(game_id)
        state = {**reply["state"], "board": self.board_of(game_id, reply["state"])}
        seat = OpenHexSeat(self.native, game_id, state, reply["your_color"])
        logger.info("%s: game %s as %s against %s", self.name, game_id, reply["your_color"],
                    [p.get("name") for p in reply["game"].get("players", [])])
        refused = 0
        lagging: tuple[list[str], int] = ([], 0)  # (kinds of difference, polls they have lasted)
        try:
            while True:
                seat.feed(self.client.events(game_id, seat.last_seq))
                reply = self.client.state(game_id)
                info, state = reply["game"], reply["state"]
                if info.get("status") != "active":
                    return self._record(game_id, seat, info)
                try:
                    plan = seat.plan(state)
                except (OpenHexError, ValueError, RuntimeError) as error:
                    logger.warning("%s: game %s: no plan (%s); playing the default move", self.name, game_id,
                                   str(error)[:160])
                    plan = Plan(seat.fallback(state))
                if plan is None:
                    time.sleep(self.poll_s)
                    continue
                problems = seat.mismatches(state)
                # OpenHex's state lags its events (and sometimes steps back): a
                # difference counts once its kind has lasted three polls.
                kinds = [p.split(" ")[0] for p in problems]
                lagging = (kinds, lagging[1] + 1 if kinds == lagging[0] else 1)
                if problems and lagging[1] == 3:
                    seat.stats["mismatches"] = seat.stats.get("mismatches", 0) + 1
                    logger.warning("%s: game %s: the replay differs from OpenHex: %s", self.name, game_id,
                                   "; ".join(problems[:3]))
                started = time.monotonic()
                for action in plan.actions:
                    try:
                        self.client.act(game_id, action, int(1000 * (time.monotonic() - started)))
                        refused = 0
                    except OpenHexHTTPError as error:
                        if error.status >= 500:
                            raise
                        refused += 1
                        logger.warning("%s: game %s: %s refused (%s)", self.name, game_id, action, error.detail[:200])
                        for fallback in seat.fallback(state) if refused <= 3 else []:
                            try:
                                self.client.act(game_id, fallback, 0)
                            except OpenHexHTTPError as again:
                                logger.warning("%s: fallback %s refused too (%s)", self.name, fallback, again.detail[:200])
                        time.sleep(self.poll_s)
                        break
        finally:
            seat.close()  # however the game ended: a reusing agent's searches of it go now

    def _record(self, game_id: str, seat: OpenHexSeat, info: Mapping[str, Any]) -> dict:
        (self.state_dir / "boards" / f"{game_id}.json").unlink(missing_ok=True)
        line = {"time": time.time(), "game_id": game_id, "mode": self.mode, "color": seat.colors[seat.seat],
                "status": info.get("status"), "winner": info.get("winner"),
                "players": [p.get("name") for p in info.get("players", [])],
                "decisions": seat.decisions, "fallbacks": seat.fallbacks, **seat.stats}
        if seat.search:
            line["search"] = seat.search
        logger.info("%s: %s", self.name, json.dumps(line))
        with open(self.state_dir / "games.jsonl", "a") as file:
            file.write(json.dumps(line) + "\n")
        return line
