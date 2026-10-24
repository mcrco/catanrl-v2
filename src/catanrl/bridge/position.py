"""A seat's Catanatron view as a cppanatron game.

A view is Catanatron's ``client_view`` for one seat: the server's ``before``
plus ``decide``/``step`` state reassembled by ``catanatron.server.table.Table``
(the map from ``before``, the redacted log as ``action_records``), or what the
TUI hands a ``decide_fns`` callback. It shows the seat's own hand and cards,
everyone's public state and the unseen development cards (deck plus the
other hands), never the other hands.

This module holds what both constructions of a position share:

* :class:`SeatView`, typed access to a view;
* :func:`game_rules` (the rules, the starting bank and deck inferred from the
  view or taken from a supply) and :func:`shared_map` (one immutable
  ``cpp.CatanMap`` per map document, shared by every game on it);
* :func:`public_mismatches`, the check that a cppanatron game shows exactly
  what the view shows (everything a PRIVATE row reads, and more);
* :func:`snapshot`: the position from one view through ``Game.apply_setup``,
  exact at every prompt Catanatron offers (:data:`PROMPT_COVERAGE`), with
  the other seats' hidden cards filled by a deterministic placeholder deal
  built from public counts only (the hand tracker's sampler replaces them).

:mod:`catanrl.bridge.replay` replays the view's log instead, which also
feeds the seat's hand tracker; the snapshot is where it (re-)anchors when it
cannot explain a view (a TUI edit, a log it cannot follow).

Nothing here reads hidden information: placeholders depend only on the hand
sizes, the bank, the unseen pool and public victory points.
"""

from __future__ import annotations

import json
from collections import OrderedDict, deque
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cppanatron as cpp

RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
DEVELOPMENT_CARDS = ("KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT")
PLAYABLE_CARDS = DEVELOPMENT_CARDS[:-1]
VICTORY_POINT = DEVELOPMENT_CARDS.index("VICTORY_POINT")
#: Supplies by the names catanatron-server room configs use.
SUPPLY_NAMES = ("base", "extension")

#: Which view fields each prompt's ``apply_setup`` declaration carries; every
#: one is exact (``snapshot`` reproduces the view's public state and the
#: seat's cards) except a trade negotiation, which Catanatron never offers.
PROMPT_COVERAGE = {
    "BUILD_INITIAL_SETTLEMENT / BUILD_INITIAL_ROAD": "prompt; placement read from the pieces (newest last)",
    "PLAY_TURN before / after the roll": "has_rolled, has_played_development_card, owned-at-start flags",
    "PLAY_TURN with free roads (Road Building)": "free_roads",
    "DISCARD": "acting_player (the discarding seat), discards left per seat",
    "MOVE_ROBBER (after a 7 or a knight)": "prompt, has_rolled, has_played_development_card",
    "every prompt": ("road networks, cached and published road lengths, the length Longest Road was won "
                     "with, award holders, robber, turns of each seat's last knight and purchase"),
    "DECIDE_TRADE / DECIDE_ACCEPTEES": "not expressible (Inexpressible); never offered by Catanatron",
}


class ViewError(ValueError):
    """A view that is malformed or does not add up."""


class Inexpressible(ValueError):
    """A position ``apply_setup`` cannot express (see :data:`PROMPT_COVERAGE`)."""


# ---------------------------------------------------------------- reading views


def color_name(color: Any) -> str:
    return str(getattr(color, "value", getattr(color, "name", color)))


def seat_of(view: Mapping[str, Any], color: Any = None) -> int:
    """The index (in ``view["colors"]``) of ``color``, or of the seat whose
    hand the view shows."""
    colors = list(view["colors"])
    if color is not None:
        name = color_name(color)
        if name not in colors:
            raise ViewError(f"{name} is not seated in this game: {colors}")
        return colors.index(name)
    visible = [i for i in range(len(colors)) if f"P{i}_WOOD_IN_HAND" in view["player_state"]]
    if len(visible) != 1:
        raise ViewError("cannot tell whose view this is (pass the seat's color)")
    return visible[0]


@dataclass(frozen=True)
class SeatView:
    """One seat's view, with typed accessors (counts in RESOURCES /
    DEVELOPMENT_CARDS order)."""

    doc: Mapping[str, Any]
    seat: int

    @classmethod
    def of(cls, view: Mapping[str, Any], color: Any = None) -> SeatView:
        return cls(view, seat_of(view, color))

    @property
    def colors(self) -> tuple[str, ...]:
        return tuple(self.doc["colors"])

    @property
    def color(self) -> str:
        return self.colors[self.seat]

    @property
    def num_players(self) -> int:
        return len(self.colors)

    def value(self, seat: int, key: str) -> Any:
        return self.doc["player_state"][f"P{seat}_{key}"]

    def visible(self, seat: int) -> bool:
        return f"P{seat}_WOOD_IN_HAND" in self.doc["player_state"]

    def resources(self, seat: int) -> tuple[int, ...]:
        if not self.visible(seat):
            raise ViewError(f"seat {seat}'s hand is hidden in this view")
        return tuple(int(self.value(seat, f"{r}_IN_HAND")) for r in RESOURCES)

    def development_cards(self, seat: int) -> tuple[int, ...]:
        if not self.visible(seat):
            raise ViewError(f"seat {seat}'s cards are hidden in this view")
        return tuple(int(self.value(seat, f"{d}_IN_HAND")) for d in DEVELOPMENT_CARDS)

    def owned_at_start(self, seat: int) -> tuple[bool, ...]:
        return tuple(bool(self.value(seat, f"{d}_OWNED_AT_START")) for d in PLAYABLE_CARDS)

    def hand_size(self, seat: int) -> int:
        if self.visible(seat):
            return sum(self.resources(seat))
        return int(self.value(seat, "NUM_RESOURCES_IN_HAND"))

    def hand_sizes(self) -> tuple[int, ...]:
        return tuple(self.hand_size(i) for i in range(self.num_players))

    def development_card_count(self, seat: int) -> int:
        if self.visible(seat):
            return sum(self.development_cards(seat))
        return int(self.value(seat, "NUM_DEVELOPMENT_CARDS_IN_HAND"))

    def played(self, seat: int) -> tuple[int, ...]:
        return tuple(int(self.value(seat, f"PLAYED_{d}")) for d in DEVELOPMENT_CARDS)

    def victory_points(self, seat: int) -> int:
        """Public points (victory point cards excluded)."""
        return int(self.value(seat, "VICTORY_POINTS"))

    @property
    def bank(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self.doc["resource_freqdeck"])

    @property
    def deck_size(self) -> int:
        if "development_deck_size" in self.doc:
            return int(self.doc["development_deck_size"])
        return sum(self._deck_composition())

    def _deck_composition(self) -> tuple[int, ...]:
        deck = self.doc["development_listdeck"]
        if isinstance(deck, Mapping):
            return tuple(int(deck.get(d, 0)) for d in DEVELOPMENT_CARDS)
        return tuple(sum(1 for card in deck if card == d) for d in DEVELOPMENT_CARDS)

    def unseen_development_cards(self) -> tuple[int, ...]:
        """The deck plus every hidden hand's cards."""
        if "unseen_development_cards" in self.doc:
            unseen = self.doc["unseen_development_cards"]
            return tuple(int(unseen[d]) for d in DEVELOPMENT_CARDS)
        deck = list(self._deck_composition())
        for i in range(self.num_players):
            if not self.visible(i):
                raise ViewError("a view with hidden hands must list the unseen development cards")
        return tuple(deck)

    def starting_deck(self) -> tuple[int, ...]:
        """The rules' development deck: unseen + visible hands + every played card."""
        deck = list(self.unseen_development_cards())
        for i in range(self.num_players):
            played = self.played(i)
            held = self.development_cards(i) if self.visible(i) else (0,) * 5
            for d in range(5):
                deck[d] += held[d] + played[d]
        return tuple(deck)

    def buildings(self, seat: int) -> dict[str, list]:
        return dict(self.doc["buildings_by_color"].get(self.colors[seat], {}))

    def award_holder(self, flag: str) -> int:
        holders = [i for i in range(self.num_players) if self.value(i, flag)]
        return holders[0] if holders else -1


# ---------------------------------------------------------------- rules and maps


def _supply_counts(supply: Any) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, int, int]] | None:
    """(bank, deck, (roads, settlements, cities)) of a supply-like value, or None."""
    if supply is None:
        return None
    if isinstance(supply, Mapping) and "supply" in supply:
        supply = supply["supply"]
    if supply is None:
        return None
    if isinstance(supply, str):
        if supply not in SUPPLY_NAMES:
            raise ViewError(f"unknown supply {supply!r}; expected one of {SUPPLY_NAMES}")
        from catanatron.models.decks import BASE_SUPPLY, EXTENSION_SUPPLY

        supply = BASE_SUPPLY if supply == "base" else EXTENSION_SUPPLY
    if hasattr(supply, "cppanatron_rules"):  # catanrl.games.GameConfig
        supply = supply.cppanatron_rules()
    if isinstance(supply, cpp.GameRules):
        check_fair_dice(supply)
        return (tuple(supply.resource_bank), tuple(supply.development_deck),
                (supply.roads_per_player, supply.settlements_per_player, supply.cities_per_player))
    if hasattr(supply, "resource_bank") and hasattr(supply, "roads"):  # catanatron Supply
        return (tuple(supply.resource_bank), tuple(supply.development_deck),
                (supply.roads, supply.settlements, supply.cities))
    raise ViewError(f"cannot read a supply from {supply!r}")


def check_fair_dice(rules: cpp.GameRules) -> None:
    """Refuse rules a Catanatron game cannot have: Catanatron rolls two fair
    dice, so its views replayed on balanced dice would refuse its rolls."""
    if rules.dice_mode != cpp.DiceMode.RANDOM:
        raise ViewError(f"a Catanatron game rolls two fair dice, not {rules.dice_mode.name.lower()} dice")


def game_rules(view: Mapping[str, Any], supply: Any = None) -> cpp.GameRules:
    """The game's rules from the view.

    Victory points to win, the discard limit and the friendly robber are in
    ``view["game"]``. Pieces per player follow from any seat's pieces left
    plus built. The starting deck is the unseen pool plus the seat's cards
    plus every played card. The starting bank is the bank plus every hand;
    only its total is public, so it is read as uniform unless ``supply``
    (a Catanatron ``Supply``, ``"base"``/``"extension"``, a room config with
    ``supply``, a catanrl ``GameConfig`` or a ``cpp.GameRules``) gives it, in
    which case the view must agree with it.
    """
    sv = SeatView.of(view) if _has_single_perspective(view) else SeatView(view, 0)
    game = view["game"]
    built = sv.buildings(0)
    pieces = (
        int(sv.value(0, "ROADS_AVAILABLE")) + len(built.get("ROAD", [])),
        int(sv.value(0, "SETTLEMENTS_AVAILABLE")) + len(built.get("SETTLEMENT", [])),
        int(sv.value(0, "CITIES_AVAILABLE")) + len(built.get("CITY", [])),
    )
    deck = sv.starting_deck()
    total = sum(sv.bank) + sum(sv.hand_sizes())
    given = _supply_counts(supply)
    if given is None:
        if total % len(RESOURCES):
            raise ViewError(f"{total} resource cards cannot be a uniform bank; pass the game's supply")
        bank = (total // len(RESOURCES),) * len(RESOURCES)
    else:
        bank, given_deck, given_pieces = given
        if tuple(given_deck) != deck or tuple(given_pieces) != pieces or sum(bank) != total:
            raise ViewError(f"the view does not match the supply: deck {deck} vs {tuple(given_deck)}, "
                            f"pieces {pieces} vs {tuple(given_pieces)}, cards {total} vs {sum(bank)}")
    return cpp.GameRules(
        vps_to_win=int(game["vps_to_win"]),
        discard_limit=int(game["discard_limit"]),
        friendly_robber=bool(game["friendly_robber"]),
        resource_bank=tuple(int(x) for x in bank),
        development_deck=tuple(int(x) for x in deck),
        roads_per_player=pieces[0],
        settlements_per_player=pieces[1],
        cities_per_player=pieces[2],
    )


def _has_single_perspective(view: Mapping[str, Any]) -> bool:
    try:
        seat_of(view)
    except ViewError:
        return False
    return True


_MAPS: OrderedDict[str, cpp.CatanMap] = OrderedDict()
_MAP_CACHE_SIZE = 64


def shared_map(map_doc: Mapping[str, Any]) -> cpp.CatanMap:
    """The cppanatron map of a view's ``map`` document (Catanatron's
    ``map_to_json``), built once and shared by every game on it."""
    key = json.dumps(map_doc, sort_keys=True, separators=(",", ":"))
    catan_map = _MAPS.get(key)
    if catan_map is None:
        from catanatron.serialization import map_from_json
        from cppanatron.catanatron import map_spec_from_catanatron

        catan_map = cpp.CatanMap.from_spec(map_spec_from_catanatron(map_from_json(map_doc)))
        _MAPS[key] = catan_map
        while len(_MAPS) > _MAP_CACHE_SIZE:
            _MAPS.popitem(last=False)
    else:
        _MAPS.move_to_end(key)
    return catan_map


def fresh_game(view: Mapping[str, Any], rules: cpp.GameRules, catan_map: cpp.CatanMap,
               seed: int = 0) -> cpp.Game:
    """A new cppanatron game seated like the view (``view["colors"]`` is the
    turn order) on ``catan_map``. ``seed`` drives only chance the caller does
    not force."""
    colors = [cpp.Color[name] for name in view["colors"]]
    return cpp.Game(colors, map=catan_map, rules=rules, seed=int(seed))


# ---------------------------------------------------------------- dealing


def transport(supply: Sequence[int], demand: Mapping[Hashable, int], *,
              lower: Mapping[Hashable, Sequence[int]] | None = None,
              upper: Mapping[Hashable, Sequence[int]] | None = None,
              prefer: Mapping[Hashable, Sequence[int]] | None = None) -> dict[Hashable, list[int]]:
    """Deal ``supply[k]`` cards of each kind into hands of sizes ``demand``.

    Hand ``h`` gets between ``lower[h][k]`` and ``upper[h][k]`` cards of kind
    ``k``. Cards a hand holds in ``prefer`` stay with it where the bounds
    allow, so a repair moves few cards. Deterministic; raises ValueError when
    no deal fits.
    """
    kinds = len(supply)
    hands = list(demand)
    lo = {h: [int(x) for x in (lower or {}).get(h, (0,) * kinds)] for h in hands}
    hi = {h: [int(x) for x in (upper or {}).get(h, (demand[h],) * kinds)] for h in hands}
    if sum(supply) != sum(demand.values()):
        raise ValueError(f"{sum(supply)} cards do not fill hands of {sum(demand.values())}")
    x = {h: list(lo[h]) for h in hands}
    left = [int(supply[k]) - sum(x[h][k] for h in hands) for k in range(kinds)]
    need = {h: int(demand[h]) - sum(x[h]) for h in hands}
    if min(left, default=0) < 0 or min(need.values(), default=0) < 0 or any(
            lo[h][k] > hi[h][k] for h in hands for k in range(kinds)):
        raise ValueError("the lower bounds alone do not fit")
    if prefer is not None:
        for h in hands:
            for k in range(kinds):
                keep = min(int(prefer.get(h, (0,) * kinds)[k]) - x[h][k], left[k], need[h], hi[h][k] - x[h][k])
                if keep > 0:
                    x[h][k] += keep
                    left[k] -= keep
                    need[h] -= keep
    while any(need[h] > 0 for h in hands):
        if not _augment(x, lo, hi, left, need, hands, kinds):
            raise ValueError("no deal fits the bounds")
    return x


def _augment(x, lo, hi, left, need, hands, kinds) -> bool:
    """One card more along a shortest path: a kind with cards left, then
    alternately a hand that may take that kind and a kind that hand may give
    up, ending at a hand with room."""
    parent: dict[tuple[str, Any], tuple[str, Any] | None] = {}
    queue: deque[tuple[str, Any]] = deque()
    for k in range(kinds):
        if left[k] > 0:
            parent[("k", k)] = None
            queue.append(("k", k))
    while queue:
        node = queue.popleft()
        if node[0] == "k":
            k = node[1]
            for h in hands:
                if x[h][k] < hi[h][k] and ("h", h) not in parent:
                    parent[("h", h)] = node
                    if need[h] > 0:
                        _apply_path(x, left, need, parent, ("h", h))
                        return True
                    queue.append(("h", h))
        else:
            h = node[1]
            for k in range(kinds):
                if x[h][k] > lo[h][k] and ("k", k) not in parent:
                    parent[("k", k)] = node
                    queue.append(("k", k))
    return False


def _apply_path(x, left, need, parent, end) -> None:
    need[end[1]] -= 1
    node = end
    while True:
        prev = parent[node]
        if node[0] == "h":
            x[node[1]][prev[1]] += 1  # the hand takes the previous kind
        elif prev is None:
            left[node[1]] -= 1
            return
        else:
            x[prev[1]][node[1]] -= 1  # the previous hand gives this kind up
        node = prev


def victory_point_room(rules: cpp.GameRules, public_points: int, margin: int = 0) -> int:
    """How many victory point cards a seat with ``public_points`` can hold
    without having won (``margin`` points kept free)."""
    return max(0, int(rules.vps_to_win) - 1 - int(public_points) - int(margin))


def placeholder_hands(sv: SeatView, rules: cpp.GameRules) -> tuple[list[tuple], list[tuple]]:
    """Resource hands and development cards for every seat: visible hands as
    they are, hidden ones dealt deterministically from public counts (the
    hidden resources per kind are the starting bank minus the bank minus the
    visible hands; the hidden cards are the unseen pool minus the deck).
    Hidden hands get victory point cards only when the deck cannot take
    them all, and never so many that their seat would have won."""
    n = sv.num_players
    hidden = [i for i in range(n) if not sv.visible(i)]
    resources: list[tuple] = [sv.resources(i) if sv.visible(i) else () for i in range(n)]
    cards: list[tuple] = [sv.development_cards(i) if sv.visible(i) else () for i in range(n)]
    if hidden:
        pool = [int(rules.resource_bank[k]) - sv.bank[k]
                - sum(resources[i][k] for i in range(n) if sv.visible(i)) for k in range(5)]
        deal = transport(pool, {i: sv.hand_size(i) for i in hidden})
        unseen = sv.unseen_development_cards()
        demand: dict[Any, int] = {i: sv.development_card_count(i) for i in hidden}
        demand["deck"] = sv.deck_size
        dev = None
        for margin in (None, 0):
            upper = {i: _card_upper(sv.development_card_count(i), 0 if margin is None else
                                    victory_point_room(rules, sv.victory_points(i), margin)) for i in hidden}
            try:
                dev = transport(unseen, demand, upper=upper)
                break
            except ValueError:
                continue
        if dev is None:
            raise ViewError("no deal of the unseen development cards fits the public counts")
        for i in hidden:
            resources[i] = tuple(deal[i])
            cards[i] = tuple(dev[i])
    return resources, cards


def _card_upper(count: int, victory_points: int) -> list[int]:
    upper = [count] * 5
    upper[VICTORY_POINT] = min(count, victory_points)
    return upper


# ---------------------------------------------------------------- comparing


def public_mismatches(game: cpp.Game, view: Mapping[str, Any], seat: int | None = None) -> list[str]:
    """What ``game`` shows differently from ``view``: the board, the bank, the
    deck size, the prompt and turn state, every seat's public counts, pieces,
    awards and road lengths, and the viewing seat's hand, cards and which of
    them it may play. Empty when the game is what the seat sees (everything
    a PRIVATE row reads is among these)."""
    sv = SeatView(view, seat_of(view) if seat is None else seat)
    out: list[str] = []

    def check(name: str, ours: Any, theirs: Any) -> None:
        if ours != theirs:
            out.append(f"{name}: game {ours!r}, view {theirs!r}")

    n = sv.num_players
    check("colors", tuple(c.name for c in game.colors), sv.colors)
    check("current player", game.current_player_index, int(view["current_player_index"]))
    check("turn owner", game.current_turn_index, int(view["current_turn_index"]))
    check("prompt", game.current_prompt.name, view["current_prompt"])
    check("turns", game.num_turns, int(view["num_turns"]))
    check("initial phase", game.is_initial_build_phase, bool(view["is_initial_build_phase"]))
    check("discarding", game.is_discarding, bool(view["is_discarding"]))
    check("road building", game.is_road_building, bool(view["is_road_building"]))
    if view["is_discarding"]:
        check("discards left", tuple(game.remaining_discards(i) for i in range(n)),
              tuple(int(x) for x in view["discard_counts"]))
    check("bank", tuple(game.resource_bank), sv.bank)
    check("deck", int(game.development_cards_remaining), sv.deck_size)

    board = view["board"]
    check("robber", tuple(game.board.robber_coordinate), tuple(board["robber_coordinate"]))
    ours_buildings = {node: (color.name, building.name) for node, (color, building) in game.board.buildings.items()}
    theirs_buildings = {int(node): (color, kind) for node, color, kind in board["buildings"]}
    check("buildings", ours_buildings, theirs_buildings)
    ours_roads = {tuple(edge): color.name for edge, color in game.board.roads.items()}
    theirs_roads = {tuple(sorted(edge)): color for edge, color in board["roads"]}
    check("roads", ours_roads, theirs_roads)
    holder = game.board.longest_road_color
    check("longest road holder", None if holder is None else holder.name, board["road_color"])

    for i in range(n):
        p = game.player(i)
        who = f"seat {i}"
        check(f"{who} points", p.victory_points, sv.victory_points(i))
        check(f"{who} pieces", (p.roads_available, p.settlements_available, p.cities_available),
              tuple(int(sv.value(i, f"{k}_AVAILABLE")) for k in ("ROADS", "SETTLEMENTS", "CITIES")))
        check(f"{who} awards", (p.has_road, p.has_army),
              (bool(sv.value(i, "HAS_ROAD")), bool(sv.value(i, "HAS_ARMY"))))
        check(f"{who} rolled", p.has_rolled, bool(sv.value(i, "HAS_ROLLED")))
        check(f"{who} played a card", p.has_played_development_card_in_turn,
              bool(sv.value(i, "HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN")))
        check(f"{who} road length", p.longest_road_length, int(sv.value(i, "LONGEST_ROAD_LENGTH")))
        check(f"{who} played cards", tuple(p.played_development_cards), sv.played(i))
        check(f"{who} hand size", sum(p.resources), sv.hand_size(i))
        check(f"{who} card count", sum(p.development_cards), sv.development_card_count(i))
        if sv.visible(i) and i == sv.seat:
            check(f"{who} hand", tuple(p.resources), sv.resources(i))
            check(f"{who} cards", tuple(p.development_cards), sv.development_cards(i))
            check(f"{who} playable cards", tuple(p.development_card_owned_at_start), sv.owned_at_start(i))
            check(f"{who} actual points", p.actual_victory_points, int(sv.value(i, "ACTUAL_VICTORY_POINTS")))
    return out


# ---------------------------------------------------------------- snapshot


@dataclass
class Position:
    """A cppanatron game for one seat of a view."""

    game: cpp.Game
    seat: int
    #: "replay" or "snapshot".
    method: str
    #: :func:`public_mismatches` against the view; empty when exact.
    mismatches: tuple[str, ...] = ()

    @property
    def exact(self) -> bool:
        return not self.mismatches


def card_turns(records: Sequence[Any], colors: Sequence[str]) -> list[tuple[int, int]]:
    """Per seat, the completed turns (END_TURNs before) of its last knight
    and of its last purchase in a log; -1 for never. Both are public."""
    last = [[-1, -1] for _ in colors]
    completed = 0
    for (color, kind, _value), _result in records:
        if kind == "END_TURN":
            completed += 1
        elif kind == "PLAY_KNIGHT_CARD":
            last[colors.index(color)][0] = completed
        elif kind == "BUY_DEVELOPMENT_CARD":
            last[colors.index(color)][1] = completed
    return [(knight, bought) for knight, bought in last]


def snapshot(view: Mapping[str, Any], *, color: Any = None, rules: cpp.GameRules | None = None,
             catan_map: cpp.CatanMap | None = None, supply: Any = None, seed: int = 0) -> Position:
    """The view's position through ``Game.apply_setup``, at any prompt but a
    domestic trade's negotiation (:class:`Inexpressible`; Catanatron offers
    none).

    Everything the view shows goes into the declaration: the pieces (each
    seat's settlements in building order), the prompt with the seat to act,
    cards left to discard, free roads, the seat's own cards and which of
    them it held when its turn began, Catanatron's road networks and cached
    and published road lengths, awards, and from the log the turns of every
    seat's last knight and purchase. Hidden hands are
    :func:`placeholder_hands`. ``mismatches`` is empty when the game shows
    exactly what the view shows.
    """
    sv = SeatView.of(view, color)
    if view.get("is_resolving_trade") or view["current_prompt"] in ("DECIDE_TRADE", "DECIDE_ACCEPTEES"):
        raise Inexpressible("apply_setup cannot express a trade negotiation")
    rules = rules if rules is not None else game_rules(view, supply)
    catan_map = catan_map if catan_map is not None else shared_map(view["map"])
    resources, cards = placeholder_hands(sv, rules)
    board = view["board"]
    turns = card_turns(view.get("action_records", ()), sv.colors)
    discarding = bool(view["is_discarding"])
    players = []
    for i, name in enumerate(sv.colors):
        built = sv.buildings(i)
        cached = board["road_lengths"].get(name)
        components = board["connected_components"].get(name)
        players.append(cpp.PlayerSetup(
            settlements=[int(x) for x in built.get("SETTLEMENT", [])],
            cities=[int(x) for x in built.get("CITY", [])],
            roads=[tuple(sorted(int(x) for x in edge)) for edge in built.get("ROAD", [])],
            resources=tuple(resources[i]), development_cards=tuple(cards[i]),
            played_development_cards=sv.played(i),
            development_card_owned_at_start=sv.owned_at_start(i) if sv.visible(i) else None,
            discards=int(view["discard_counts"][i]) if discarding else 0,
            longest_road_length=int(sv.value(i, "LONGEST_ROAD_LENGTH")),
            cached_road_length=None if cached is None else int(cached),
            road_components=None if components is None else [[int(n) for n in c] for c in components],
            last_knight_completed_turn=turns[i][0],
            last_development_card_bought_completed_turn=turns[i][1]))
    turn = int(view["current_turn_index"])
    acting = int(view["current_player_index"])
    game = fresh_game(view, rules, catan_map, seed)
    game.apply_setup(
        players,
        robber=tuple(int(x) for x in board["robber_coordinate"]),
        current_player=turn,
        acting_player=acting if acting != turn else None,
        has_rolled=bool(sv.value(turn, "HAS_ROLLED")),
        has_played_development_card=bool(sv.value(turn, "HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN")),
        completed_turns=max(0, int(view["num_turns"]) - 2 * (sv.num_players - 1)),
        longest_road=sv.award_holder("HAS_ROAD"),
        largest_army=sv.award_holder("HAS_ARMY"),
        prompt=cpp.ActionPrompt[view["current_prompt"]],
        free_roads=int(view["free_roads_available"]) if view["is_road_building"] else 0,
        longest_road_length=int(board["road_length"]) if board.get("road_color") else None,
    )
    return Position(game, sv.seat, "snapshot", tuple(public_mismatches(game, view, sv.seat)))
