"""The game as text for a language model (a port of catanllm's
``util/game_convert.py``, attribution: Marco Yang's catanllm, SP 25 CS 159).

``render_state(game, color)`` is what the seat ``color`` may know, and
nothing else. Tiles are named by id everywhere (``Tile 7 (Wheat 5)``: the
state, the robber, the robber actions and the history), never by cube
coordinate, and a node is annotated with what it touches (``node_text``: each
adjacent tile's resource and number, the pips, a port). Per seat: the
observer's own hand and development cards (those bought this turn marked:
they cannot be played yet); for an opponent its visible victory points, the
number of unplayed development cards (never their kinds, so never a hidden
victory-point card), the cards it has played, and its resource hand: exactly
in a 2-player game, where every resource movement is public (``exact_hands``:
production follows the public dice and buildings; builds, bank and port
trades, discards and development card plays are announced; a steal is seen by
its thief and its victim, one of whom is always the observer; the fork offers
no player-to-player trade; ``tests/test_llm_agent.py`` replays games with a
tracker that sees only that and pins the equality), as a count with three or
more players. catanllm's renderer read every seat's development cards from
the observer's own slot (a bug that showed the wrong counts).

The phase line says what the decision is (setup placement, roll, discard,
robber, free roads); a pending discard is stated with the cards still to go
(the fork prompts one ``Discard`` action per card). The static board (tiles,
ports, node topology) is in the system prompt, not here. ``render_events`` is
the recent history from ``state.action_records`` as the observer saw it: a
bought development card is named only to its buyer, a stolen card only to the
thief and the victim. ``render_actions`` numbers the playable actions as the
model must answer them, each annotated with what choosing it needs (what a
settlement spot or a road's far end produces, what a robber tile holds and
who sits on it, the trade ratio, the cards a Monopoly takes).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from catanatron.models.enums import (
    CITY,
    DEVELOPMENT_CARDS,
    RESOURCES,
    SETTLEMENT,
    ActionPrompt,
    ActionType,
)
from catanatron.models.player import Color
from catanatron.state_functions import (
    get_actual_victory_points,
    get_dev_cards_in_hand,
    get_largest_army,
    get_longest_road_color,
    get_longest_road_length,
    get_played_dev_cards,
    get_player_freqdeck,
    get_visible_victory_points,
    player_key,
    player_num_dev_cards,
    player_num_resource_cards,
)

RESOURCE_LABELS = ("Wood", "Brick", "Sheep", "Wheat", "Ore")
LABELS = dict(zip(RESOURCES, RESOURCE_LABELS))
CARD_LABELS = {"KNIGHT": "Knight", "YEAR_OF_PLENTY": "Year of Plenty", "MONOPOLY": "Monopoly",
               "ROAD_BUILDING": "Road Building", "VICTORY_POINT": "Victory Point"}
PLAYED_CARDS = ("KNIGHT", "MONOPOLY", "ROAD_BUILDING", "YEAR_OF_PLENTY")
#: The ways (of 36) two dice roll each number: a tile's expected production in 36 rolls.
PIPS = {n: 6 - abs(7 - n) for n in range(2, 13) if n != 7}
#: How far (in roads still to build) the state text's open settlement spots reach.
REACH = 2


def label(resource: Any) -> str:
    return LABELS.get(resource, str(resource))


def seat(color: Color, observer: Color | None) -> str:
    """A seat as the observer reads it: ``you`` or its color."""
    return "you" if color == observer else color.name


def whose(color: Color, observer: Color | None) -> str:
    """``your`` or ``BLUE's``."""
    return "your" if color == observer else f"{color.name}'s"


def exact_hands(state: Any) -> bool:
    """Whether every resource hand is public to every seat: two players (module docstring)."""
    return len(state.colors) == 2


def tile_label(tile: Any) -> str:
    """``Wheat 5``; the desert has no number."""
    return "Desert" if tile.resource is None else f"{label(tile.resource)} {tile.number}"


def tile_text(tile: Any) -> str:
    """``Tile 7 (Wheat 5)``."""
    return f"Tile {tile.id} ({tile_label(tile)})"


def node_ports(catan_map: Any) -> dict[int, str]:
    """Node -> its port (``3:1 port``, ``2:1 Wheat port``)."""
    out = {}
    for resource, nodes in catan_map.port_nodes.items():
        for node in nodes:
            out[node] = "3:1 port" if resource is None else f"2:1 {label(resource)} port"
    return out


def node_text(catan_map: Any, node: int, *, robber: Any = None, ports: dict[int, str] | None = None) -> str:
    """What ``node`` touches: ``Wheat 6, Ore 10, Wood 3; 12 pips; 3:1 port`` (pips: the ways of
    36 its tiles' numbers roll; the tile under the ``robber``, a land tile, is marked)."""
    parts, pips = [], 0
    for tile in sorted(catan_map.adjacent_tiles.get(node, []), key=lambda t: t.id):
        if tile.resource is None:
            parts.append("Desert")
            continue
        pips += PIPS[tile.number]
        parts.append(f"{label(tile.resource)} {tile.number}" + (" (robber)" if tile is robber else ""))
    text = f"{', '.join(parts) or 'no land'}; {pips} pip{'' if pips == 1 else 's'}"
    port = (ports if ports is not None else node_ports(catan_map)).get(node)
    return f"{text}; {port}" if port else text


def node_pips(catan_map: Any, node: int) -> int:
    return sum(PIPS[t.number] for t in catan_map.adjacent_tiles.get(node, []) if t.number is not None)


def buildings_on(board: Any, tile: Any, observer: Color) -> list[str]:
    """``BLUE's settlement at 12`` for each building on ``tile``'s corners."""
    out = []
    for node in sorted(tile.nodes.values()):
        if node in board.buildings:
            color, kind = board.buildings[node]
            out.append(f"{whose(color, observer)} {kind.lower()} at {node}")
    return out


def network(board: Any, color: Color) -> set[int]:
    """The nodes ``color``'s roads and buildings reach (where its roads and settlements can grow)."""
    components = board.connected_components[color]
    nodes = set().union(*components) if components else set()
    return {n for n in nodes if not board.is_enemy_node(n, color)}


def road_distances(board: Any, color: Color, limit: int = REACH) -> dict[int, int]:
    """Node -> the roads ``color`` must still build to reach it (at most ``limit``): breadth-first
    from its network over edges without an opponent's road, never through an opponent's building."""
    dist = {n: 0 for n in network(board, color)}
    frontier = list(dist)
    for d in range(1, limit + 1):
        step = []
        for node in frontier:
            if board.is_enemy_node(node, color):
                continue
            for other in board.map.graph.neighbors(node):
                if other in dist or board.is_enemy_road((node, other), color):
                    continue
                dist[other] = d
                step.append(other)
        frontier = step
    return dist


def current_turn(state: Any) -> int | None:
    """The main-phase turn (1, 2, ...), or None in the setup (its first ``4 * players`` records:
    two settlements and two roads each); a turn ends with End Turn."""
    records, setup = state.action_records, 4 * len(state.colors)
    if len(records) < setup:
        return None
    return 1 + sum(1 for r in records[setup:] if r.action.action_type == ActionType.END_TURN)


def bought_this_turn(state: Any, color: Color) -> dict[str, int]:
    """Development cards ``color`` bought since its turn began (not playable until its next turn)."""
    out: dict[str, int] = {}
    for record in reversed(state.action_records):
        if record.action.action_type == ActionType.END_TURN:
            break
        if record.action.action_type == ActionType.BUY_DEVELOPMENT_CARD and record.action.color == color:
            out[record.result] = out.get(record.result, 0) + 1
    return out


# ---------------------------------------------------------------- the state


def phase(game: Any, observer: Color) -> str:
    """What the seat to act is deciding, in words."""
    state = game.state
    mover = state.current_color()
    if mover != observer:
        return f"waiting for {mover.name}"
    prompt = state.current_prompt
    if prompt == ActionPrompt.BUILD_INITIAL_SETTLEMENT:
        if state.buildings_by_color[mover][SETTLEMENT]:
            return ("setup: place your second settlement (anywhere the distance rule allows); it pays you one card "
                    "of each resource tile it touches right away")
        return "setup: place your first settlement (anywhere the distance rule allows)"
    if prompt == ActionPrompt.BUILD_INITIAL_ROAD:
        return "setup: place a road touching the settlement you just placed"
    if prompt == ActionPrompt.DISCARD:
        remaining = int(state.discard_counts[state.color_to_index[observer]])  # the fork counts it down
        return (f"a 7 was rolled: discard {remaining} more of your {player_num_resource_cards(state, observer)} "
                "resource cards (each Discard action removes one card)")
    if prompt == ActionPrompt.MOVE_ROBBER:
        records = state.action_records
        knight = bool(records) and records[-1].action.action_type == ActionType.PLAY_KNIGHT_CARD
        return f"move the robber ({'you played a Knight' if knight else 'a 7 was rolled'}) and steal one card"
    if prompt in (ActionPrompt.DECIDE_TRADE, ActionPrompt.DECIDE_ACCEPTEES):
        return "answer a trade offer"
    if state.is_road_building:
        return f"Road Building: place free road {3 - int(state.free_roads_available)} of 2"
    if not state.player_state[f"{player_key(state, observer)}_HAS_ROLLED"]:
        return "your turn, before rolling: roll the dice (you may first play a development card)"
    return ("your turn, after rolling: build, buy a development card, trade with the bank or a port, play a "
            "development card, or end your turn")


def player_status(state: Any, color: Color, observer: Color) -> str:
    """One seat's block: everything for the observer itself, public facts for the others (an
    opponent's exact hand only when :func:`exact_hands`)."""
    key = player_key(state, color)
    freqdeck = get_player_freqdeck(state, color)
    detail = ", ".join(f"{lab} {n}" for lab, n in zip(RESOURCE_LABELS, freqdeck))
    played = ", ".join(f"{CARD_LABELS[c]} {n}" for c in PLAYED_CARDS if (n := get_played_dev_cards(state, color, c)))
    lines = []
    if color == observer:
        actual, visible = get_actual_victory_points(state, color), get_visible_victory_points(state, color)
        lines.append(f"{color.name} (you): {actual} VP" + (
            f" ({visible} visible to the others; the rest are your Victory Point cards)" if actual != visible else ""))
        lines.append(f"  hand ({sum(freqdeck)} cards): {detail}")
        new = bought_this_turn(state, color)
        cards = []
        for card in DEVELOPMENT_CARDS:
            if n := get_dev_cards_in_hand(state, color, card):
                fresh = 0 if card == "VICTORY_POINT" else new.get(card, 0)
                cards.append(f"{CARD_LABELS[card]} {n}" + (
                    f" ({fresh} bought this turn: playable from your next turn)" if fresh else ""))
        lines.append(f"  development cards: {', '.join(cards) or 'none'}; played: {played or 'none'}")
    else:
        lines.append(f"{color.name}: {get_visible_victory_points(state, color)} VP visible (its unplayed development "
                     "cards may hide Victory Point cards)")
        if exact_hands(state):
            lines.append(f"  hand ({sum(freqdeck)} cards): {detail}")
        else:
            lines.append(f"  hand: {sum(freqdeck)} cards (which kinds is hidden from you)")
        lines.append(f"  development cards: {player_num_dev_cards(state, color)} unplayed (hidden); "
                     f"played: {played or 'none'}")
    board = state.board
    buildings = []
    for node in sorted(n for n, (c, _) in board.buildings.items() if c == color):
        kind = "city" if board.buildings[node][1] == CITY else "settlement"
        buildings.append(f"{kind} at {node} ({node_text(board.map, node)})")
    lines.append(f"  buildings: {'; '.join(buildings) or 'none'}")
    roads = sorted((a, b) for (a, b), c in board.roads.items() if c == color and a < b)
    lines.append(f"  roads ({len(roads)}, longest {get_longest_road_length(state, color)}): "
                 f"{', '.join(f'{a}-{b}' for a, b in roads) or 'none'}")
    ports = sorted(board.get_player_port_resources(color), key=lambda r: (r is not None, r or ""))
    trade = ", ".join("3:1" if r is None else f"2:1 {label(r)}" for r in ports)
    lines.append(f"  pieces left: {state.player_state[f'{key}_SETTLEMENTS_AVAILABLE']} settlements, "
                 f"{state.player_state[f'{key}_CITIES_AVAILABLE']} cities, "
                 f"{state.player_state[f'{key}_ROADS_AVAILABLE']} roads; ports: {trade or 'none'}")
    return "\n".join(lines)


def open_spots(game: Any, observer: Color, *, limit: int = 10) -> list[str]:
    """Settlement spots the distance rule leaves open within :data:`REACH` roads of the observer's
    network, nearest then richest first (nothing during the setup, whose actions list every legal
    spot). Which ones are legal right now is the action list's business."""
    state, board = game.state, game.state.board
    if state.is_initial_build_phase:
        return []
    dist = road_distances(board, observer)
    spots = sorted((d, -node_pips(board.map, n), n) for n, d in dist.items() if n in board.board_buildable_ids)
    ports = node_ports(board.map)
    robber = board.map.land_tiles[board.robber_coordinate]
    head = (f"Open settlement spots within {REACH} roads of your network (the distance rule allows them; "
            f"{len(board.board_buildable_ids)} are open on the whole board):")
    if not spots:
        return [f"{head} none"]
    out = [head]
    for d, _, node in spots[:limit]:
        reach = "on your network" if d == 0 else f"{d} more road{'s' if d > 1 else ''}"
        out.append(f"  node {node} ({reach}): {node_text(board.map, node, robber=robber, ports=ports)}")
    return out


def render_state(game: Any, observer: Color) -> str:
    """The seat ``observer``'s view of ``game`` now (the static board is in the system prompt)."""
    state = game.state
    board = state.board
    turn = current_turn(state)
    when = "Setup phase" if turn is None else f"Turn {turn}"
    out = ["=== GAME STATE ===", f"{when}; you are {observer.name}; to act: {phase(game, observer)}.",
           f"Victory points to win: {game.vps_to_win}", "", "=== PLAYERS ==="]
    out += [player_status(state, color, observer) for color in state.colors]
    road_color = get_longest_road_color(state)
    army_color, knights = get_largest_army(state)
    road = f"{road_color.name} ({get_longest_road_length(state, road_color)} roads)" if road_color else "nobody"
    army = f"{army_color.name} ({knights} knights)" if army_color else "nobody"
    out.append(f"Longest Road (2 VP, 5+ roads): {road}; Largest Army (2 VP, 3+ knights): {army}")
    bank = ", ".join(f"{lab} {n}" for lab, n in zip(RESOURCE_LABELS, state.resource_freqdeck))
    out.append(f"Bank: {bank}; development cards left: {len(state.development_listdeck)}")
    robber = board.map.land_tiles[board.robber_coordinate]
    sitting = buildings_on(board, robber, observer)
    out.append(f"Robber: on {tile_text(robber)}, which produces nothing while it stays there"
               + (f"; buildings on it: {', '.join(sitting)}" if sitting else ""))
    spots = open_spots(game, observer)
    if spots:
        out += ["", *spots]
    return "\n".join(out)


# ---------------------------------------------------------------- the history


def _cards(resources: Sequence[Any]) -> str:
    counts: dict[Any, int] = {}
    for r in resources:
        counts[r] = counts.get(r, 0) + 1
    return ", ".join(f"{n} {label(r)}" for r, n in counts.items())


def _bundle(freqdeck: Sequence[int]) -> str:
    return ", ".join(f"{n} {lab}" for lab, n in zip(RESOURCE_LABELS, freqdeck) if n) or "nothing"


def event_text(game: Any, record: Any, observer: Color) -> str | None:
    """One action record as ``observer`` saw it (None for End Turn, a turn boundary): a bought
    development card is named only to its buyer, a stolen resource only to the thief and the
    victim (in a 2-player game the observer is always one of them)."""
    action, result = record.action, record.result
    kind, value = action.action_type, action.value
    who = seat(action.color, observer)
    catan_map = game.state.board.map
    if kind == ActionType.ROLL:
        return f"{who} rolled {sum(result)}"
    if kind == ActionType.BUILD_SETTLEMENT:
        return f"{who} built a settlement at {value} ({node_text(catan_map, value)})"
    if kind == ActionType.BUILD_CITY:
        return f"{who} upgraded {value} to a city"
    if kind == ActionType.BUILD_ROAD:
        return f"{who} built a road {value[0]}-{value[1]}"
    if kind == ActionType.BUY_DEVELOPMENT_CARD:  # the record's value and result are the card: the buyer's only
        return f"{who} bought a development card" + (f" ({CARD_LABELS[result]})" if action.color == observer else "")
    if kind == ActionType.PLAY_KNIGHT_CARD:
        return f"{who} played a Knight"
    if kind == ActionType.PLAY_YEAR_OF_PLENTY:
        return f"{who} played Year of Plenty for {_cards(value)}"
    if kind == ActionType.PLAY_MONOPOLY:
        return f"{who} played Monopoly on {label(value)}"
    if kind == ActionType.PLAY_ROAD_BUILDING:
        return f"{who} played Road Building"
    if kind == ActionType.DISCARD_RESOURCE:
        return f"{who} discarded {_cards([result])}"
    if kind == ActionType.MOVE_ROBBER:
        coordinate, victim = value
        text = f"{who} moved the robber to {tile_text(catan_map.land_tiles[coordinate])}"
        if victim is None:
            return f"{text} and stole nothing"
        seen = observer in (action.color, victim)  # the thief and the victim know the card
        stolen = f"1 {label(result)}" if seen else "a card"
        return f"{text} and stole {stolen} from {seat(victim, observer)}"
    if kind == ActionType.MARITIME_TRADE:
        given = [r for r in value[:-1] if r is not None]
        return f"{who} traded {len(given)} {label(given[0])} for 1 {label(value[-1])} ({len(given)}:1)"
    if kind == ActionType.END_TURN:
        return None
    if kind == ActionType.OFFER_TRADE:
        return f"{who} offered {_bundle(value[:5])} for {_bundle(value[5:10])}"
    if kind in (ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE):
        return f"{who} {'accepted' if kind == ActionType.ACCEPT_TRADE else 'rejected'} the offer"
    if kind == ActionType.CONFIRM_TRADE:
        return f"{who} traded {_bundle(value[:5])} for {_bundle(value[5:10])} with {seat(value[10], observer)}"
    if kind == ActionType.CANCEL_TRADE:
        return f"{who} cancelled the offer"
    return f"{who}: {kind.name} {value}"


def render_events(game: Any, observer: Color, limit: int) -> str:
    """The last ``limit`` events as ``observer`` saw them (:func:`event_text`), oldest first, one
    line per turn; End Turns are boundaries, not events, and a run of one seat's discards is one
    event. Empty when ``limit`` is 0 or nothing has happened."""
    state = game.state
    records, n = state.action_records, len(state.colors)
    events: list[list[Any]] = []  # [turn, text, discarded resources of a discard run or None]
    turn = None
    for index, record in enumerate(records):
        if index == 4 * n:
            turn = 1
        action = record.action
        if action.action_type == ActionType.END_TURN:
            turn = (turn or 0) + 1
            continue
        previous = records[index - 1].action if index else None
        if (action.action_type == ActionType.DISCARD_RESOURCE and events and events[-1][2] is not None
                and previous is not None and previous.action_type == ActionType.DISCARD_RESOURCE
                and previous.color == action.color):
            events[-1][2].append(record.result)
            events[-1][1] = f"{seat(action.color, observer)} discarded {_cards(events[-1][2])}"
            continue
        text = event_text(game, record, observer)
        if text is not None:
            events.append([turn, text, [record.result] if action.action_type == ActionType.DISCARD_RESOURCE else None])
    events = events[-limit:] if limit > 0 else []
    if not events:
        return ""
    lines: list[str] = []
    current: object = object()
    for turn, text, _ in events:
        if turn == current:
            lines[-1] += f"; {text}"
            continue
        current = turn
        lines.append(f"Setup: {text}" if turn is None else
                     f"Turn {turn} ({seat(state.colors[(turn - 1) % n], observer)}): {text}")
    return f"=== RECENT EVENTS (the last {len(events)}, oldest first) ===\n" + "\n".join(lines)


# ---------------------------------------------------------------- the actions


def describe_action(game: Any, action: Any, observer: Color | None = None) -> str:
    """One playable action in words (catanllm's ``format_playable_actions`` entry, on the fork's
    values) with what choosing it needs; ``observer`` is the acting seat unless given."""
    observer = action.color if observer is None else observer
    state = game.state
    board = state.board
    catan_map = board.map
    kind, value = action.action_type, action.value
    if kind == ActionType.ROLL:
        return "Roll the dice"
    if kind == ActionType.END_TURN:
        return "End your turn"
    if kind == ActionType.BUILD_SETTLEMENT:
        verb = "Place your settlement" if state.is_initial_build_phase else "Build a settlement"
        return f"{verb} at node {value} ({node_text(catan_map, value)})"
    if kind == ActionType.BUILD_CITY:
        return f"Upgrade your settlement at node {value} to a city (doubles its production: {node_text(catan_map, value)})"
    if kind == ActionType.BUILD_ROAD:
        a, b = value
        reached = network(board, action.color)
        new = [n for n in (a, b) if n not in reached]
        far = new[0] if new else b
        if state.is_initial_build_phase:
            verb = "Place a road"
        elif state.is_road_building:
            verb = f"Build a free road (Road Building: {int(state.free_roads_available)} free left)"
        else:
            verb = "Build a road"
        if far in board.buildings:
            what = f"{whose(board.buildings[far][0], observer)} {board.buildings[far][1].lower()} there"
        elif far in board.board_buildable_ids:
            what = "open for a settlement"
        else:
            what = "too close to a building for a settlement"
        return f"{verb} {a}-{b} (to node {far}: {node_text(catan_map, far)}; {what}" + \
            ("; links two parts of your network)" if not new else ")")
    if kind == ActionType.BUY_DEVELOPMENT_CARD:
        return f"Buy a development card ({len(state.development_listdeck)} left in the deck)"
    if kind == ActionType.PLAY_KNIGHT_CARD:
        return "Play a Knight (then move the robber and steal one card; counts toward Largest Army)"
    if kind == ActionType.PLAY_YEAR_OF_PLENTY:
        return f"Play Year of Plenty: take {_cards(value)} from the bank"
    if kind == ActionType.PLAY_MONOPOLY:
        if exact_hands(state):
            taken = sum(player_num_resource_cards(state, c, value) for c in state.colors if c != action.color)
            return f"Play Monopoly on {label(value)} (takes the {taken} {label(value)} your opponent holds)"
        return f"Play Monopoly on {label(value)} (takes every {label(value)} the others hold)"
    if kind == ActionType.PLAY_ROAD_BUILDING:
        return "Play Road Building (then build up to 2 roads for free)"
    if kind == ActionType.DISCARD_RESOURCE:
        return f"Discard 1 {label(value)} (you hold {player_num_resource_cards(state, action.color, value)})"
    if kind == ActionType.MOVE_ROBBER:
        coordinate, victim = value
        tile = catan_map.land_tiles[coordinate]
        sitting = buildings_on(board, tile, observer)
        steal = (f"steal 1 random card from {victim.name} ({player_num_resource_cards(state, victim)} in hand)"
                 if victim is not None else "steal nothing")
        return f"Move the robber to Tile {tile.id} ({tile_label(tile)}; {', '.join(sitting) or 'no buildings'}) and {steal}"
    if kind == ActionType.MARITIME_TRADE:
        given = [r for r in value[:-1] if r is not None]
        ratio = len(given)
        how = {4: "4:1 with the bank", 3: "3:1 at your port"}.get(ratio, f"2:1 at your {label(given[0])} port")
        return f"Trade {ratio} {label(given[0])} for 1 {label(value[-1])} ({how})"
    if kind == ActionType.OFFER_TRADE:
        return f"Offer {_bundle(value[:5])} for {_bundle(value[5:10])}"
    if kind == ActionType.ACCEPT_TRADE:  # the responder gives what was asked and gets what was offered
        return f"Accept the offer: you give {_bundle(value[5:10])} and get {_bundle(value[:5])}"
    if kind == ActionType.REJECT_TRADE:
        return f"Reject the offer of {_bundle(value[:5])} for your {_bundle(value[5:10])}"
    if kind == ActionType.CONFIRM_TRADE:
        return f"Trade with {value[10].name}: you give {_bundle(value[:5])} and get {_bundle(value[5:10])}"
    if kind == ActionType.CANCEL_TRADE:
        return "Cancel your trade offer"
    return f"Action: {kind.name} {value}"


def render_actions(game: Any, playable_actions: Sequence[Any], observer: Color | None = None) -> str:
    out = ["=== AVAILABLE ACTIONS ==="]
    out += [f"{i}: {describe_action(game, a, observer)}" for i, a in enumerate(playable_actions)]
    return "\n".join(out)


def tile_numbers(game: Any) -> dict[int, int]:
    """Tile id -> number, for tests."""
    return {t.id: t.number for t in game.state.board.map.land_tiles.values() if t.number is not None}


__all__ = ["CARD_LABELS", "LABELS", "PIPS", "RESOURCES", "describe_action", "event_text", "exact_hands", "label",
           "node_ports", "node_text", "open_spots", "phase", "player_status", "render_actions", "render_events",
           "render_state", "tile_label", "tile_numbers", "tile_text"]
