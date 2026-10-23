"""A cppanatron decision as the Canopy bridge's snapshot, and Canopy's actions back.

:func:`snapshot` writes the deciding seat's position in the bridge's JSON shape (v1's fields
plus the v2 extensions: ``hidden_dev_cards``, ``friendly_robber``, ``port_set``, ``seed``).
It reads only what that seat knows:

* the public board, pieces, awards, bank, deck size, played cards and prompt of the game;
* its own hand and development cards (exact), and what it bought and traded this turn
  (:class:`SeatHistory`, kept from its own decisions);
* the opponent's resource hand from a world dealt from the seat's hand tracker
  (``cpp.sample_world``; in 2p the tracker knows the hand exactly, so the deal is the truth
  the seat has seen), and the opponent's unplayed development cards as a count (the tracker's,
  as the bridge requires; the identities are never read).

The true game's hidden cards never reach a snapshot (``tests/test_canopy.py`` redeals them and
compares the bytes).

Canopy's 249 actions (``examples/catan/game/action.rs``): 0-53 settlement at a node, 54-125
road at an edge, 126-179 city, 180 roll, 181 end turn, 182 buy a card, 183 knight, 184 road
building, 185-199 year of plenty (an unordered pair), 200-204 monopoly, 205-223 robber to a tile
(the victim is the opponent when the tile touches them and they hold cards), 224-228 discard one
card, 229-248 maritime trade at the best ratio. :func:`action_id` maps an engine action into
them (None for what Canopy cannot express: player trades, a one-card year of plenty).

``tree_actions`` (re-rooting a saved search) carry chance outcomes inline: dice sum minus 2,
the drawn card's Canopy kind, the stolen resource. :func:`replay_chain` builds them for the
moves between two decisions of the seat that it can replay exactly: its own move and the forced
moves the engine played after it (in the true game, whose random stream the engine shares), with
outcomes the seat observed. A move of the opponent continues a chain only when the seat knows it
was forced (:func:`publicly_forced`), so whether a search is re-rooted never depends on the
opponent's hidden cards. The agent sees the game only at its own decisions (the harness plays
forced moves itself and keeps no action log), so a chain stops at any move the opponent chose:
Canopy's own tournament re-roots through every move, the opponent's included.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import cppanatron as cpp

from catanrl.canopy.topology import RESOURCE_NAMES, RESOURCES, TILE_EDGES, TILE_NODES, CanopyBoard

AT = cpp.ActionType
SETTLEMENT, ROAD, CITY = 0, 54, 126
ROLL, END_TURN, BUY_DEV, KNIGHT, ROAD_BUILDING = 180, 181, 182, 183, 184
YOP, MONOPOLY, ROBBER, DISCARD, MARITIME, NUM_ACTIONS = 185, 200, 205, 224, 229, 249
YOP_OFFSETS = (0, 5, 9, 12, 14)
#: cppanatron's development card order (KNIGHT, YEAR_OF_PLENTY, MONOPOLY, ROAD_BUILDING,
#: VICTORY_POINT) -> Canopy's (Knight, VictoryPoint, RoadBuilding, YearOfPlenty, Monopoly).
DEV_TO_CANOPY = (0, 3, 4, 2, 1)
CANOPY_KNIGHT, CANOPY_VP, CANOPY_RB, CANOPY_YOP, CANOPY_MONO = range(5)
#: The only rules the released network and the bridge know (base game supply, 2 players).
BASE_BANK, BASE_DECK, BASE_PIECES = (19,) * 5, (14, 2, 2, 2, 5), (15, 5, 4)


class SnapshotError(ValueError):
    """A position the bridge's snapshot cannot express."""


def canopy_devs(counts: Sequence[int]) -> list[int]:
    out = [0] * 5
    for i, c in enumerate(counts):
        out[DEV_TO_CANOPY[i]] += int(c)
    return out


def _resource(r: Any) -> int:
    return RESOURCES.index(r)


def action_id(board: CanopyBoard, action: cpp.Action) -> int | None:
    """Canopy's id of an engine action, or None if Canopy cannot express it."""
    kind, value = action.action_type, action.value
    if kind == AT.BUILD_SETTLEMENT:
        return SETTLEMENT + board.node_to_canopy[int(value)]
    if kind == AT.BUILD_ROAD:
        a, b = (int(v) for v in value)
        return ROAD + board.edge_to_canopy[(a, b) if a < b else (b, a)]
    if kind == AT.BUILD_CITY:
        return CITY + board.node_to_canopy[int(value)]
    fixed = {AT.ROLL: ROLL, AT.END_TURN: END_TURN, AT.BUY_DEVELOPMENT_CARD: BUY_DEV, AT.PLAY_KNIGHT_CARD: KNIGHT,
             AT.PLAY_ROAD_BUILDING: ROAD_BUILDING}
    if kind in fixed:
        return fixed[kind]
    if kind == AT.PLAY_YEAR_OF_PLENTY:
        if len(value) != 2:
            return None
        first, second = sorted(_resource(r) for r in value)
        return YOP + YOP_OFFSETS[first] + second - first
    if kind == AT.PLAY_MONOPOLY:
        return MONOPOLY + _resource(value)
    if kind == AT.MOVE_ROBBER:
        return ROBBER + board.coord_to_tile[tuple(int(c) for c in value[0])]
    if kind == AT.DISCARD_RESOURCE:
        return DISCARD + _resource(value)
    if kind == AT.MARITIME_TRADE:
        give, receive = _resource(value[0]), _resource(value[-1])
        return MARITIME + give * 4 + (receive if receive < give else receive - 1)
    return None  # player trades


def action_name(board: CanopyBoard, aid: int) -> str:
    """A readable name of a Canopy action (logs and mismatch records)."""
    if aid < ROAD:
        return f"settlement@{board.canopy_to_node[aid]}"
    if aid < CITY:
        return f"road@{board.canopy_to_edge[aid - ROAD]}"
    if aid < ROLL:
        return f"city@{board.canopy_to_node[aid - CITY]}"
    fixed = {ROLL: "roll", END_TURN: "end_turn", BUY_DEV: "buy_dev", KNIGHT: "knight", ROAD_BUILDING: "road_building"}
    if aid in fixed:
        return fixed[aid]
    if aid < MONOPOLY:
        k = aid - YOP
        first = max(i for i, o in enumerate(YOP_OFFSETS) if o <= k)
        return f"year_of_plenty({RESOURCE_NAMES[first]},{RESOURCE_NAMES[first + k - YOP_OFFSETS[first]]})"
    if aid < ROBBER:
        return f"monopoly({RESOURCE_NAMES[aid - MONOPOLY]})"
    if aid < DISCARD:
        return f"robber@{board.tile_coords[aid - ROBBER]}"
    if aid < MARITIME:
        return f"discard({RESOURCE_NAMES[aid - DISCARD]})"
    give, k = divmod(aid - MARITIME, 4)
    return f"maritime({RESOURCE_NAMES[give]}->{RESOURCE_NAMES[k if k < give else k + 1]})"


def legal_by_id(board: CanopyBoard, game: cpp.Game) -> tuple[dict[int, list[cpp.Action]], list[cpp.Action]]:
    """The engine's legal actions by Canopy id, and those Canopy cannot express."""
    by_id: dict[int, list[cpp.Action]] = {}
    unexpressed = []
    for action in game.playable_actions:
        aid = action_id(board, action)
        if aid is None:
            unexpressed.append(action)
        else:
            by_id.setdefault(aid, []).append(action)
    return by_id, unexpressed


def check_rules(game: cpp.Game) -> None:
    """Refuse a game whose rules the bridge would misstate (it pins the base supply and two
    players; Canopy's friendly robber is another rule than the engine's)."""
    rules = game.rules
    supply = (tuple(int(n) for n in rules.resource_bank), tuple(int(n) for n in rules.development_deck),
              (int(rules.roads_per_player), int(rules.settlements_per_player), int(rules.cities_per_player)))
    if game.num_players != 2:
        raise SnapshotError(f"Canopy plays two players, not {game.num_players}")
    if supply != (BASE_BANK, BASE_DECK, BASE_PIECES):
        raise SnapshotError(f"Canopy plays the base supply (bank 19, deck 14/2/2/2/5, pieces 15/5/4), not {supply}")
    if bool(rules.friendly_robber):
        raise SnapshotError("the engine's friendly robber is not Canopy's (FRIENDLY_ROBBER_VP 3); refused")


# ---------------------------------------------------------------- the seat's own history


@dataclass
class Observed:
    """Public values right after the seat's last move (for ``tested_non_knight``)."""

    turn: int
    owner: int
    dev_count: tuple[int, int]
    played: tuple[tuple[int, ...], tuple[int, ...]]
    opponent_bought: int          # the opponent's purchases this turn so far, when it is the owner
    own_bought: int               # the seat's purchases this turn
    robber: tuple[int, int, int]
    buildings: tuple[frozenset, frozenset]


@dataclass
class SeatHistory:
    """What a seat remembers between its decisions in one game: its purchases and trades this
    turn (exact: they are its own moves) and Canopy's ``tested_non_knight`` per seat, the count of
    a seat's old unplayed cards seen held through a turn it ended blocked by the robber
    (``game/mod.rs`` ``apply_end_turn``): kept from the seat's decisions, so a stretch of several
    turns without one (a turn of forced moves) is approximated with the values at its end (the
    robber where it is now, not where it was when each skipped turn ended). Against Canopy's rule
    applied to the full action log of 30 random-play base-2p games, the opponent's value sent
    differed at 290 of 11,203 decisions (2.6%; 281 too low, 9 too high). Exact needs the moves
    between the seat's decisions, which the harness does not hand an agent."""

    seat: int
    turn: int = -1
    bought: list[int] = field(default_factory=lambda: [0] * 5)
    traded: bool = False
    tested: list[int] = field(default_factory=lambda: [0, 0])
    last: Observed | None = None
    mismatches: list[str] = field(default_factory=list)

    def start(self, game: cpp.Game, tracker: cpp.HandTracker) -> None:
        """Bring the history to ``game`` (the seat's decision now)."""
        if int(game.num_turns) != self.turn:
            self.turn, self.bought, self.traded = int(game.num_turns), [0] * 5, False
        own = int(tracker.bought_this_turn(self.seat))
        if sum(self.bought) != own:
            self.mismatches.append(f"turn {self.turn}: {sum(self.bought)} purchases remembered, tracker {own}")
            self.bought = [min(b, h) for b, h in zip(self.bought, canopy_devs(game.player(self.seat).development_cards))]
        self._tested(game, tracker)

    def _tested(self, game: cpp.Game, tracker: cpp.HandTracker) -> None:
        last = self.last
        if last is None:
            return
        me, opp = self.seat, 1 - self.seat
        played = canopy_devs(game.player(opp).played_development_cards)
        if played[CANOPY_KNIGHT] > last.played[opp][CANOPY_KNIGHT]:
            self.tested[opp] = 0
        other = sum(played[k] - last.played[opp][k] for k in (CANOPY_RB, CANOPY_YOP, CANOPY_MONO))
        count = int(tracker.development_card_count(opp))
        if other > 0:
            self.tested[opp] = min(self.tested[opp], count)
        robber = tuple(int(c) for c in game.board.robber_coordinate)
        for t in range(last.turn, int(game.num_turns)):
            owner = last.owner if (t - last.turn) % 2 == 0 else 1 - last.owner
            if owner == opp:
                plays = sum(played) - sum(last.played[opp])
                bought = (last.opponent_bought if last.owner == opp and t == last.turn else 0) + max(
                    0, count - last.dev_count[opp] + plays)
                if _blocks(game, robber, _buildings(game, opp)):
                    self._raise(opp, count - bought)
            elif t == last.turn:  # the seat's own turn ended after its last move (by forced moves)
                if _blocks(game, last.robber, last.buildings[me]):
                    self._raise(me, last.dev_count[me] - last.own_bought)
            elif _blocks(game, robber, _buildings(game, me)):  # a turn of forced moves only: no purchases
                self._raise(me, sum(game.player(me).development_cards))

    def _raise(self, seat: int, eligible: int) -> None:
        if eligible > 0:
            self.tested[seat] = max(self.tested[seat], int(eligible))

    def played(self, game: cpp.Game, tracker: cpp.HandTracker, action: cpp.Action, after: cpp.Game) -> None:
        """The seat chose ``action`` in ``game``; ``after`` is ``game`` with it played (the
        engine's random stream, so a purchase shows the card drawn)."""
        me = self.seat
        kind = action.action_type
        if kind == AT.BUY_DEVELOPMENT_CARD:
            drawn = [a - b for a, b in zip(canopy_devs(after.player(me).development_cards),
                                           canopy_devs(game.player(me).development_cards))]
            if sorted(drawn) == [0, 0, 0, 0, 1]:
                self.bought[drawn.index(1)] += 1
            else:
                self.mismatches.append(f"turn {self.turn}: purchase changed the hand by {drawn}")
        elif kind == AT.MARITIME_TRADE:
            self.traded = True
        elif kind == AT.PLAY_KNIGHT_CARD:
            self.tested[me] = 0
        elif kind in (AT.PLAY_ROAD_BUILDING, AT.PLAY_YEAR_OF_PLENTY, AT.PLAY_MONOPOLY):
            self.tested[me] = min(self.tested[me], sum(after.player(me).development_cards))
        robber = tuple(int(c) for c in game.board.robber_coordinate)
        if kind == AT.END_TURN and _blocks(game, robber, _buildings(game, me)):
            # Canopy's apply_end_turn, with the values before the turn passes.
            self._raise(me, sum(game.player(me).development_cards) - sum(self.bought))
        opp = 1 - me
        owner = int(after.current_turn_index)
        self.last = Observed(
            turn=int(after.num_turns), owner=owner,
            dev_count=(0, 0) if after.winning_player_index is not None else tuple(
                sum(after.player(me).development_cards) if s == me else int(tracker.development_card_count(opp))
                for s in range(2)),
            played=tuple(tuple(canopy_devs(after.player(s).played_development_cards)) for s in range(2)),
            opponent_bought=int(tracker.bought_this_turn(opp)) if owner == opp else 0,
            own_bought=sum(self.bought) if kind != AT.END_TURN else 0,
            robber=tuple(int(c) for c in after.board.robber_coordinate),
            buildings=(_buildings(after, 0), _buildings(after, 1)))
        if kind == AT.END_TURN:
            # The turn passed: the opponent's purchases count from zero, its count from here.
            self.last.opponent_bought = 0


def _buildings(game: cpp.Game, seat: int) -> frozenset:
    player = game.player(seat)
    return frozenset(int(n) for n in (*player.settlements, *player.cities))


def _blocks(game: cpp.Game, robber: tuple[int, int, int], buildings: frozenset) -> bool:
    """Whether the robber sits on a producing tile touching ``buildings``."""
    tile = game.board.map.land_tiles.get(robber)
    return tile is not None and tile.resource is not None and any(int(n) in buildings for n in tile.nodes)


# ---------------------------------------------------------------- replaying moves exactly


def _dice(before: cpp.Game, after: cpp.Game) -> int | None:
    """The roll that took ``before`` to ``after`` (as the seat sees it: hands, bank, prompt).
    Several sums can leave the same position (no production); any of them is that position."""
    def seen(g: cpp.Game) -> tuple:
        return (tuple(tuple(g.player(s).resources) for s in range(g.num_players)), tuple(g.resource_bank),
                g.current_prompt, g.current_player_index,
                tuple(int(g.remaining_discards(s)) for s in range(g.num_players)))

    target = seen(after)
    roll = next(a for a in before.playable_actions if a.action_type == AT.ROLL)
    for total in range(2, 13):
        trial = before.copy()
        first = min(6, total - 1)
        trial.execute(roll, dice=[first, total - first])
        if seen(trial) == target:
            return total
    return None


def chain_step(board: CanopyBoard, game: cpp.Game, action: cpp.Action, seat: int) -> list[int] | None:
    """Play ``action`` in ``game`` (in place, from its own random stream) and return its Canopy
    ids with the outcome the seat observes; None if Canopy cannot replay it exactly (an
    expression it lacks, the opponent's purchase, whose card the seat does not see)."""
    aid = action_id(board, action)
    if aid is None:
        return None
    kind = action.action_type
    actor = list(game.colors).index(action.color)
    if kind == AT.BUY_DEVELOPMENT_CARD and actor != seat:
        return None
    before = game.copy() if kind in (AT.ROLL, AT.BUY_DEVELOPMENT_CARD, AT.MOVE_ROBBER) else None
    game.execute(action)
    ids = [aid]
    if kind == AT.ROLL:
        total = _dice(before, game)
        if total is None:
            return None
        ids.append(total - 2)
    elif kind == AT.BUY_DEVELOPMENT_CARD:
        drawn = [a - b for a, b in zip(canopy_devs(game.player(seat).development_cards),
                                       canopy_devs(before.player(seat).development_cards))]
        if sorted(drawn) != [0, 0, 0, 0, 1]:
            return None
        ids.append(drawn.index(1))
    elif kind == AT.MOVE_ROBBER and action.value[1] is not None:
        gained = [a - b for a, b in zip(game.player(actor).resources, before.player(actor).resources)]
        if sorted(gained) != [0, 0, 0, 0, 1]:
            return None
        ids.append(gained.index(1))
    return ids


def publicly_forced(game: cpp.Game, seat: int) -> cpp.Action | None:
    """The move ``game`` forces, if ``seat`` would know it is forced whatever the hidden cards are;
    else None. The seat's own legal set is its to see. Another seat's depends on its unplayed
    development cards only through playing one, so its single legal move counts as forced only when
    it could not play a card: it holds none (a count everyone sees), it played one this turn, or the
    prompt allows none (a discard, a robber move, the initial placement)."""
    legal = list(game.playable_actions)
    if len(legal) != 1:
        return None
    actor = int(game.current_player_index)
    if actor != seat:
        player = game.player(actor)
        no_card_choice = (game.is_initial_build_phase or game.current_prompt != cpp.ActionPrompt.PLAY_TURN
                          or sum(int(n) for n in player.development_cards) == 0
                          or bool(player.has_played_development_card_in_turn))
        if not no_card_choice:
            return None
    return legal[0]


def replay_chain(board: CanopyBoard, previous: cpp.Game, chosen: cpp.Action, game: cpp.Game, seat: int,
                 *, limit: int = 64) -> list[int] | None:
    """The Canopy ids that take ``previous`` (the seat's last decision) to ``game`` (its decision
    now): its chosen move, then the moves the engine forced (:func:`publicly_forced`: forced
    whatever the opponent's hidden cards, so whether a chain continues is the seat's to know); None
    when the game moved otherwise (another seat decided, or might have, or a move Canopy cannot
    replay)."""
    x = previous.copy()
    ids: list[int] = []
    for _ in range(limit):
        step = chain_step(board, x, chosen, seat)
        if step is None:
            return None
        ids += step
        if x.search_equivalent(game):
            return ids
        if x.winning_player_index is not None:
            return None
        chosen = publicly_forced(x, seat)
        if chosen is None:
            return None
    return None


# ---------------------------------------------------------------- the snapshot


def _phase(game: cpp.Game, seat: int) -> dict[str, Any]:
    prompt = game.current_prompt
    if prompt == cpp.ActionPrompt.BUILD_INITIAL_SETTLEMENT:
        return {"kind": "place_settlement"}
    if prompt == cpp.ActionPrompt.BUILD_INITIAL_ROAD:
        return {"kind": "place_road"}
    if prompt == cpp.ActionPrompt.DISCARD:
        return {"kind": "discard", "remaining": int(game.remaining_discards(seat)),
                "roller": int(game.current_turn_index)}
    if prompt == cpp.ActionPrompt.MOVE_ROBBER:
        return {"kind": "move_robber"}
    if prompt == cpp.ActionPrompt.PLAY_TURN:
        if game.is_road_building:
            return {"kind": "road_building", "roads_left": int(game.free_roads_available)}
        return {"kind": "main" if game.player(seat).has_rolled else "pre_roll"}
    raise SnapshotError(f"Canopy has no phase for the {prompt} prompt")


def _setup(game: cpp.Game, board: CanopyBoard) -> tuple[int, int | None]:
    """``setup_count`` (initial settlements placed; 4 after) and ``last_setup_node``: as Canopy
    keeps them (``apply_place_settlement`` sets the node, ``apply_place_road`` clears it), the
    settlement the initial road must touch while one is due, else None. Placement is in snake
    order 0, 1, 1, 0 and each seat's settlements are listed oldest first."""
    if not game.is_initial_build_phase:
        return 4, None
    placed = [list(game.player(s).settlements) for s in range(2)]
    count = sum(len(p) for p in placed)
    if count == 0 or game.current_prompt != cpp.ActionPrompt.BUILD_INITIAL_ROAD:
        return count, None
    order = (0, 1, 1, 0)
    owner = order[count - 1]
    nth = order[:count].count(owner) - 1
    return count, board.node_to_canopy[int(placed[owner][nth])]


def _trade_ratios(game: cpp.Game, seat: int) -> list[int]:
    ports = game.board.player_port_resources(game.colors[seat])
    ratios = [3 if None in ports else 4] * 5
    for resource in ports:
        if resource is not None:
            ratios[_resource(resource)] = 2
    return ratios


def snapshot(game: cpp.Game, tracker: cpp.HandTracker, board: CanopyBoard, history: SeatHistory, *,
             game_id: str, seed: int, world_seed: int, tree_actions: Sequence[int] | None = None) -> dict[str, Any]:
    """The bridge snapshot of ``game`` for its deciding seat (``history.seat``), from what that
    seat knows (see the module doc)."""
    me = int(game.current_player_index)
    if me != history.seat or int(tracker.observer) != me:
        raise SnapshotError(f"seat {history.seat} is not deciding (seat {me} is) or the tracker is another's")
    world = cpp.sample_world(game, tracker, seed=int(world_seed), deal=cpp.TrackerMode.POOL)
    players = []
    for s in range(2):
        p = game.player(s)
        settlements, cities = list(p.settlements), list(p.cities)
        own = s == me
        players.append({
            "hand": [int(n) for n in (p.resources if own else world.player(s).resources)],
            "dev_cards": canopy_devs(p.development_cards) if own else [0] * 5,
            "dev_cards_bought_this_turn": list(history.bought) if own else [0] * 5,
            "dev_cards_played": [0 if k == CANOPY_VP else n for k, n in enumerate(canopy_devs(p.played_development_cards))],
            "knights_played": int(p.played_development_cards[0]),
            "roads_placed": BASE_PIECES[0] - int(p.roads_available),
            "settlements_left": int(p.settlements_available),
            "cities_left": int(p.cities_available),
            "roads_left": int(p.roads_available),
            "has_played_dev_card_this_turn": bool(p.has_played_development_card_in_turn),
            "has_traded_this_turn": bool(history.traded) if own else False,
            "building_vps": len(settlements) + 2 * len(cities),
            "trade_ratios": _trade_ratios(game, s),
            "tested_non_knight": int(history.tested[s]),
            "hidden_dev_cards": 0 if own else int(tracker.development_card_count(s)),
            "hidden_dev_cards_bought_this_turn": 0 if own or int(game.current_turn_index) != s
            else int(tracker.bought_this_turn(s)),
        })
    holder = game.board.longest_road_color
    longest = None
    if holder is not None:
        h = list(game.colors).index(holder)
        longest = {"player": h, "length": int(game.player(h).longest_road_length)}
    army = next(({"player": s, "length": int(game.player(s).played_development_cards[0])}
                 for s in range(2) if game.player(s).has_army), None)
    setup_count, last_setup = _setup(game, board)
    owner = int(game.current_turn_index)
    return {
        "game_id": game_id,
        "tree_actions": [] if tree_actions is None else [int(a) for a in tree_actions],
        "reuse_tree": tree_actions is not None,
        "terrains": list(board.terrains),
        "numbers": list(board.numbers),
        "ports": list(board.ports),
        "port_set": board.port_set,
        "tile_nodes": [list(t) for t in TILE_NODES],
        "tile_edges": [list(t) for t in TILE_EDGES],
        "players": players,
        "settlements": [[board.node_to_canopy[int(n)] for n in game.player(s).settlements] for s in range(2)],
        "cities": [[board.node_to_canopy[int(n)] for n in game.player(s).cities] for s in range(2)],
        "roads": [[board.edge_to_canopy[(min(int(a), int(b)), max(int(a), int(b)))] for a, b in game.player(s).roads]
                  for s in range(2)],
        "bank": [int(n) for n in world.resource_bank],
        "dev_deck_total": int(game.development_cards_remaining),
        "robber": board.coord_to_tile[tuple(int(c) for c in game.board.robber_coordinate)],
        "current_player": me,
        "phase": _phase(game, me),
        # Canopy counts END_TURNs only (none in the initial placement): the engine's completed turns.
        "turn_number": int(game.completed_turns),
        # The turn owner has not rolled; false during the initial placement, as Canopy keeps it.
        "pre_roll": not game.is_initial_build_phase and not bool(game.player(owner).has_rolled),
        "setup_count": setup_count,
        "last_setup_node": last_setup,
        "longest_road": longest,
        "largest_army": army,
        "vp_limit": int(game.vps_to_win),
        "discard_threshold": int(game.discard_limit),
        "settlements_at_turn_start": [board.node_to_canopy[int(n)] for n in game.player(me).settlements],
        "roads_placed_at_turn_start": BASE_PIECES[0] - int(game.player(me).roads_available),
        "friendly_robber": bool(game.friendly_robber),
        "seed": int(seed) & (2**64 - 1),
    }
