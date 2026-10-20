"""OpenHex games through the bridge: a recorded game's events (redacted for
each seat, as OpenHex redacts them for a bot) replay to its final state, the
seat's tracker stays exact on its own hand, and at every point where the
seat is to move the agent's plan is an OpenHex action."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("catanatron")

from bridge_games import tiny_agent

from catanrl.bridge.openhex import Board, EventLog, OpenHexSeat, color_of, native_for
from catanrl.bridge.replay import ReplayError

FIXTURE = Path(__file__).parent.parent / "fixtures" / "openhex" / "game-004a8250.json"
#: A ladder game (aborted at turn 95) whose turn 89 produced 7 wood with 5 in
#: the bank: OpenHex paid in full, the engine's rule would pay nobody.
OVERPAID = Path(__file__).parent.parent / "fixtures" / "openhex" / "game-881a71bc.json"
RESOURCE_NAMES = ("wood", "brick", "sheep", "wheat", "ore")
DIMS = {"max_players": 2, "max_tiles": 19, "max_nodes": 54, "max_edges": 72}


def _game():
    return json.loads(FIXTURE.read_text())


def _redacted(events, color):
    me = color_of(color)
    out = []
    for item in events:
        event = dict(item["event"])
        if event["type"] == "devCardBought" and color_of(event["player"]) != me:
            event.pop("card", None)
        if event["type"] == "resourceStolen" and me not in (color_of(event["player"]), color_of(event["target"])):
            event.pop("resource", None)
        out.append({"seq": item["seq"], "event": event})
    return out


def test_the_board_maps_every_vertex_to_a_node_and_back():
    board = Board(_game()["board"])
    assert len(board.node) == 54 and len(board.vertex) == 54
    assert len(board.catan_map.land_tiles) == 19
    assert sum(len(nodes) for nodes in board.catan_map.port_nodes.values()) == 18  # 9 ports, 2 nodes each
    for vertex, node in board.node.items():
        assert board.node_of(board.vertex_doc(node)) == node and board.vertex[node] == vertex
    # Every port's two vertices are the port's own nodes.
    for port in board.doc["ports"]:
        nodes = {board.node_of(v) for v in port["vertices"]}
        resource = port.get("resource")
        key = {"wood": "WOOD", "brick": "BRICK", "sheep": "SHEEP", "wheat": "WHEAT", "ore": "ORE"}.get(resource)
        assert nodes <= board.catan_map.port_nodes[key]


class _Never:
    """An agent that must not be asked: the replay alone is under test."""

    dims = {}
    action_space = None


@pytest.mark.parametrize("color", ["red", "blue"])
def test_a_recorded_game_replays_to_its_final_state(color):
    game = _game()
    seat = OpenHexSeat(_Never(), "fixture", game, color)
    seat.feed(_redacted(game["events"], color))
    final = game["final"]
    assert seat.over and seat.log.won == color_of(final["winner"])
    cpp_game = seat.replay.game
    bank = final["bank"]
    assert tuple(cpp_game.resource_bank) == tuple(bank[r] for r in ("wood", "brick", "sheep", "wheat", "ore"))
    for player in final["players"]:
        i = seat.colors.index(color_of(player["color"]))
        hand = tuple(player["hand"].get(r, 0) for r in ("wood", "brick", "sheep", "wheat", "ore"))
        assert sum(cpp_game.player(i).resources) == sum(hand)
        if i == seat.seat:
            assert tuple(cpp_game.player(i).resources) == hand
            assert tuple(seat.tracker.known_resources(i)) == hand


def _hand(player):
    return tuple(player["hand"].get(r, 0) for r in RESOURCE_NAMES)


def _assert_state(seat, bank, players):
    assert seat.bank == [bank[r] for r in RESOURCE_NAMES]
    for player in players:
        i = seat.colors.index(color_of(player["color"]))
        assert sum(seat.replay.game.player(i).resources) == sum(_hand(player))
        if i == seat.seat:
            assert tuple(seat.replay.game.player(i).resources) == _hand(player)
            assert tuple(seat.tracker.known_resources(i)) == _hand(player)


@pytest.mark.parametrize("color", ["blue", "red"])
def test_a_roll_the_bank_cannot_cover_pays_what_the_game_paid(color):
    # Turn 89 (seq 374): 7 wood produced with 5 in the bank. OpenHex paid in
    # full; the engine's rule pays nobody. The replay starts over on a bank
    # two cards larger and follows the game exactly (the checkpoint is the
    # public record's state after seq 379).
    game = json.loads(OVERPAID.read_text())
    checkpoint = game["checkpoints"][0]
    seat = OpenHexSeat(_Never(), "fixture", game, color)
    seat.feed(_redacted([e for e in game["events"] if e["seq"] <= checkpoint["seq"]], color))
    assert seat.bank_margin == [2, 0, 0, 0, 0] and seat.stats["bank_short"] == 1
    assert tuple(seat.rules.resource_bank) == (21, 19, 19, 19, 19)
    assert "production_mismatch" not in seat.stats
    _assert_state(seat, checkpoint["bank"], checkpoint["players"])


@pytest.mark.parametrize("color", ["blue", "red"])
def test_a_roll_the_game_withheld_is_the_replays_known_limit(color):
    # Seq 380, five rolls later: OpenHex withheld blue's 3 wood (its bank
    # held 2, less than blue's share) where the engine, two cards richer,
    # pays. The engine's bank cannot be set from Python, so the replay keeps
    # the payout, counts the difference, and goes wrong at the next 7: blue,
    # three phantom cards over the limit, is asked to discard before red.
    game = json.loads(OVERPAID.read_text())
    seat = OpenHexSeat(_Never(), "fixture", game, color)
    with pytest.raises(ReplayError, match="DISCARD_RESOURCE"):
        seat.feed(_redacted(game["events"], color))
    assert seat.stats["bank_short"] == 1 and seat.stats["production_mismatch"] == 1
    assert seat.replay.applied == 320


def test_the_agent_plans_an_openhex_action_wherever_the_seat_moves():
    game = _game()
    native, _ = native_for(tiny_agent("POLICY", "private", DIMS))
    kinds = set()
    for color in game["turnOrder"]:
        seat = OpenHexSeat(native, "fixture", game, color)
        me = color_of(color)
        for item in _redacted(game["events"], color):
            seat.feed([item])
            if seat.over:
                break
            pending = ({"type": "steal"} if seat.log.robber and seat.log.robber[0] == me
                       else {"type": "discard"} if seat.replay.game.is_discarding else None)
            current = seat.colors[seat.replay.game.current_player_index].lower()
            plan = seat.plan({"pendingAction": pending, "players": [], "board": {}, "currentPlayer": current})
            if plan is None:
                continue
            for action in plan.actions:
                assert isinstance(action.get("type"), str)
                kinds.add(action["type"])
                if "vertex" in action:
                    assert set(action["vertex"]) == {"x", "y", "z"}
                if "vertices" in action:
                    assert len(action["vertices"]) == 2
            seat.discarded_at = None
        assert seat.decisions > 20
    assert {"rollDice", "endTurn", "moveRobber", "stealFromPlayer", "buildRoad", "discardResources"} <= kinds
