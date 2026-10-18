"""The game as text for a language model (a port of catanllm's
``util/game_convert.py``, attribution: Marco Yang's catanllm, SP 25 CS 159).

``render_state(game, color)`` is what the seat ``color`` may know, and nothing
else: its own hand and development cards; for every other seat only the
hand size, the number of development cards held, the cards it has played,
its buildings and its visible victory points (never its resource kinds, its
unplayed cards or its hidden victory-point cards). catanllm's renderer
showed only hand sizes for opponents too, but read every seat's development
cards from the observer's own slot (a bug that showed the wrong counts);
``tests/test_llm_agent.py`` pins the redaction. A pending discard (a 7 was
rolled; the fork prompts one ``Discard one X`` action per card) is stated
with the cards still to go. ``render_actions`` numbers the playable actions
as the model must answer them.
"""

from __future__ import annotations

from typing import Any

from catanatron.models.enums import DEVELOPMENT_CARDS, RESOURCES, ActionType
from catanatron.models.map import PORT_DIRECTION_TO_NODEREFS, Port
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
    player_num_dev_cards,
    player_num_resource_cards,
)

RESOURCE_LABELS = ("Wood", "Brick", "Sheep", "Wheat", "Ore")
PLAYED_CARDS = ("KNIGHT", "MONOPOLY", "ROAD_BUILDING", "YEAR_OF_PLENTY")


def player_status(state: Any, color: Color, observer: Color) -> str:
    """One seat's line: everything for the observer itself, public facts for the others."""
    marker = ">>> " if color == observer else "    "
    played = ", ".join(f"{card}: {n}" for card in PLAYED_CARDS if (n := get_played_dev_cards(state, color, card)))
    extras = []
    if get_longest_road_color(state) == color:
        extras.append(f"holds Longest Road ({get_longest_road_length(state, color)})")
    army_color, knights = get_largest_army(state)
    if army_color == color:
        extras.append(f"holds Largest Army ({knights} knights)")
    hand = player_num_resource_cards(state, color)
    if color == observer:
        freqdeck = get_player_freqdeck(state, color)
        detail = ", ".join(f"{label}: {n}" for label, n in zip(RESOURCE_LABELS, freqdeck))
        cards = ", ".join(f"{card}: {n}" for card in DEVELOPMENT_CARDS if (n := get_dev_cards_in_hand(state, color, card)))
        line = (f"{marker}{color.name} (you): {get_actual_victory_points(state, color)} VP, {hand} resources ({detail}), "
                f"dev cards in hand: {cards or 'none'}")
    else:
        line = (f"{marker}{color.name}: {get_visible_victory_points(state, color)} VP (visible), {hand} resource "
                f"cards (kinds unknown to you), {player_num_dev_cards(state, color)} dev cards in hand (unknown)")
    line += f", played dev cards: {played or 'none'}"
    if extras:
        line += " | " + ", ".join(extras)
    return line


def render_state(game: Any, observer: Color) -> str:
    """catanllm's ``game_to_natural_language`` (its non-verbose form) from ``observer``'s seat."""
    state = game.state
    board = state.board
    catan_map = board.map
    out = ["=== CATAN GAME STATE ===", f"Turn: {state.num_turns}", f"Current player: {state.current_color().name}",
           f"You are: {observer.name}", f"Victory points to win: {game.vps_to_win}", "", "=== PLAYER STATUS ==="]
    out += [player_status(state, color, observer) for color in state.colors]
    remaining = int(state.discard_counts[state.color_to_index[observer]])  # the fork counts a pending discard down
    if remaining > 0:
        out.append(f"You must discard {remaining} more of your {player_num_resource_cards(state, observer)} resource "
                   "cards (a 7 was rolled); each Discard action removes one card.")
    out += ["", "=== BOARD LAYOUT ===", f"Robber position: {tuple(board.robber_coordinate)}", "", "--- HEXAGON TILES ---"]
    for _, tile in sorted(catan_map.land_tiles.items()):
        if tile.resource is None:
            out.append(f"Tile {tile.id}: DESERT (no resources)")
        else:
            out.append(f"Tile {tile.id}: {tile.resource} with number {tile.number}")
    out += ["", "--- PORTS ---"]
    ports = [t for _, t in sorted(catan_map.tiles.items()) if isinstance(t, Port)]
    for tile in ports:
        kind = "3:1 Generic" if tile.resource is None else f"2:1 {tile.resource}"
        a, b = PORT_DIRECTION_TO_NODEREFS[tile.direction]
        for node in sorted((tile.nodes[a], tile.nodes[b])):
            out.append(f"Port at {node}: {kind}")
    if not ports:
        out.append("This map has no ports.")
    out += ["", "--- ROADS (Edge Connections) ---"]
    roads = sorted((edge, color) for edge, color in board.roads.items() if edge[0] < edge[1])
    out += [f"Road: {color.name} connecting nodes {a} <-> {b}" for (a, b), color in roads] or ["No roads built yet"]
    out += ["", "--- POSSIBLE BUILDING LOCATIONS ---"]
    empty = [n for n in sorted(catan_map.land_nodes) if n not in board.buildings]
    out += [f"Available nodes for settlements: {len(empty)} nodes", f"Empty nodes: {empty}"]
    free_edges = sorted(tuple(sorted(e)) for e in catan_map.graph.edges() if tuple(sorted(e)) not in board.roads)
    out.append(f"Available road connections: {len(free_edges)} possible roads")
    if len(free_edges) <= 20:
        out.append(f"Available road edges: {free_edges}")
    out += ["", "=== BUILDINGS SUMMARY ==="]
    for color in state.colors:
        settlements = sorted(n for n, (c, kind) in board.buildings.items() if c == color and kind == "SETTLEMENT")
        cities = sorted(n for n, (c, kind) in board.buildings.items() if c == color and kind == "CITY")
        roads_count = sum(1 for (a, b), c in board.roads.items() if c == color and a < b)
        out.append(f"{color.name}: {len(settlements)} settlements{f' at nodes {settlements}' if settlements else ''}, "
                   f"{len(cities)} cities{f' at nodes {cities}' if cities else ''}, {roads_count} roads")
    out.append("")
    return "\n".join(out)


def describe_action(action: Any) -> str:
    """One playable action in words (catanllm's ``format_playable_actions`` entry, on the fork's values)."""
    kind, value = action.action_type, action.value
    if kind == ActionType.ROLL:
        return "Roll dice"
    if kind == ActionType.END_TURN:
        return "End turn"
    if kind == ActionType.BUILD_ROAD:
        return f"Build road connecting nodes {value[0]} <-> {value[1]}"
    if kind == ActionType.BUILD_SETTLEMENT:
        return f"Build settlement at node {value}"
    if kind == ActionType.BUILD_CITY:
        return f"Build city at node {value}"
    if kind == ActionType.BUY_DEVELOPMENT_CARD:
        return "Buy development card"
    if kind == ActionType.PLAY_KNIGHT_CARD:
        return "Play knight card (then move the robber)"
    if kind == ActionType.PLAY_YEAR_OF_PLENTY:
        return f"Play Year of Plenty card for {' and '.join(str(r) for r in value)}"
    if kind == ActionType.PLAY_ROAD_BUILDING:
        return "Play Road Building card"
    if kind == ActionType.PLAY_MONOPOLY:
        return f"Play Monopoly card for {value}"
    if kind == ActionType.DISCARD_RESOURCE:
        return f"Discard one {value}"
    if kind == ActionType.MOVE_ROBBER:
        coordinate, victim = value[0], value[1]
        who = f"steal from {victim.name}" if victim is not None else "steal from no one"
        return f"Move robber to tile {tuple(coordinate)}, {who}"
    if kind == ActionType.MARITIME_TRADE:
        give, get = value[0], value[4]
        ratio = 4 - sum(1 for v in value[1:4] if v is None)
        return f"Maritime trade: give {ratio} {give} for 1 {get} ({ratio}:1 port trade)"
    return f"Action: {kind.name} {value}"


def render_actions(playable_actions: Any) -> str:
    out = ["=== AVAILABLE ACTIONS ==="]
    out += [f"{i}: {describe_action(a)}" for i, a in enumerate(playable_actions)]
    return "\n".join(out)


def tile_numbers(game: Any) -> dict[int, int]:
    """Tile id -> number, for tests."""
    return {t.id: t.number for t in game.state.board.map.land_tiles.values() if t.number is not None}


__all__ = ["RESOURCES", "describe_action", "player_status", "render_actions", "render_state", "tile_numbers"]
