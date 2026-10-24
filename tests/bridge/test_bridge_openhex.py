"""OpenHex games through the bridge: a recorded game's events (redacted for
each seat, as OpenHex redacts them for a bot) replay to its final state, the
seat's tracker stays exact on its own hand, and at every point where the
seat is to move the agent's plan is an OpenHex action. A reusing MCTS agent
(``reuse=true``) decides on the seat's own stream: its search is re-rooted
along the record and along a plan decided ahead, restarts where the game is
replayed again on a larger bank or the agent fails, and goes with the game at
``close``; the plain agent's plans are pinned."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cppanatron as cpp
import pytest

pytest.importorskip("catanatron")

from bridge_games import tiny_agent
from eval_helpers import agent as eval_agent
from eval_helpers import tiny_model as eval_model

from catanrl.bridge.openhex import Board, OpenHexSeat, color_of, native_for
from catanrl.bridge.replay import ReplayError

FIXTURE = Path(__file__).parent.parent / "fixtures" / "openhex" / "game-004a8250.json"
#: A ladder game (aborted at turn 95) whose turn 89 produced 7 wood with 5 in
#: the bank: OpenHex paid in full, the engine's rule would pay nobody.
OVERPAID = Path(__file__).parent.parent / "fixtures" / "openhex" / "game-881a71bc.json"
DATA = Path(__file__).parent.parent / "data"
RESOURCE_NAMES = ("wood", "brick", "sheep", "wheat", "ore")
DIMS = {"max_players": 2, "max_tiles": 19, "max_nodes": 54, "max_edges": 72}


def _fair(record):
    """A recorded game as it was played: both records say ``diceMode: balanced``
    but rolled fair dice (tests/test_dice_reference.py), so the tests here, which
    are not about dice, replay them on fair dice (test_bridge_openhex_dice.py
    replays them as recorded)."""
    return {**record, "config": {**record["config"], "diceMode": "random"}}


def _game():
    return _fair(json.loads(FIXTURE.read_text()))


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
    game = _fair(json.loads(OVERPAID.read_text()))
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
    game = _fair(json.loads(OVERPAID.read_text()))
    seat = OpenHexSeat(_Never(), "fixture", game, color)
    with pytest.raises(ReplayError, match="DISCARD_RESOURCE"):
        seat.feed(_redacted(game["events"], color))
    assert seat.stats["bank_short"] == 1 and seat.stats["production_mismatch"] == 1
    assert seat.replay.applied == 320


# ---------------------------------------------------------------- deciding along a record


def _record(seat, game, events=None, log=None):
    """Drive ``seat`` along the record as OpenHex drives a bot (a poll of
    events, then a plan on a state saying what is pending; a roll and its
    production come in one poll): ``(seq, plan, acts)`` per plan, ``acts``
    the entries ``log`` (see :func:`_logged`) gained deciding it."""
    me = seat.colors[seat.seat]
    polls: list[list] = []
    for item in _redacted(game["events"] if events is None else events, me):
        if polls and polls[-1][-1]["event"]["type"] == "diceRolled" and item["event"]["type"] == "resourcesProduced":
            polls[-1].append(item)
        else:
            polls.append([item])
    for poll in polls:
        seat.feed(poll)
        if seat.over:
            return
        pending = ({"type": "steal"} if seat.log.robber and seat.log.robber[0] == me
                   else {"type": "discard"} if seat.replay.game.is_discarding else None)
        current = seat.colors[seat.replay.game.current_player_index].lower()
        before = len(log) if log is not None else 0
        plan = seat.plan({"pendingAction": pending, "players": [], "board": {}, "currentPlayer": current})
        if plan is not None:
            seat.discarded_at = None
            yield poll[-1]["seq"], plan, log[before:] if log is not None else []


def _logged(native):
    """Wrap ``native.act`` to log each decision: the game's prompt, whether
    the seat marked it first, whether the agent raised, and how the agent's
    counters moved (a reuse or a refusal is counted before the search runs)."""
    log: list[dict] = []
    real = native.act

    def act(rows, masks, seeds, **kwargs):
        before = Counter(native.counters)
        entry = {"prompt": kwargs["games"][0].current_prompt.name, "first": bool(kwargs["first"][0]), "raised": False}
        log.append(entry)
        try:
            return real(rows, masks, seeds, **kwargs)
        except Exception:
            entry["raised"] = True
            raise
        finally:
            entry.update({k: v - before[k] for k, v in native.counters.items() if v != before[k]})

    native.act = act
    return log


def _mcts(**params):
    return native_for(tiny_agent("MCTS", "full", DIMS, sims=8, dets=1, **params))[0]


def test_the_agent_plans_an_openhex_action_wherever_the_seat_moves():
    game = _game()
    native, _ = native_for(tiny_agent("POLICY", "private", DIMS))
    kinds = set()
    for color in game["turnOrder"]:
        seat = OpenHexSeat(native, "fixture", game, color)
        for _seq, plan, _acts in _record(seat, game):
            for action in plan.actions:
                assert isinstance(action.get("type"), str)
                kinds.add(action["type"])
                if "vertex" in action:
                    assert set(action["vertex"]) == {"x", "y", "z"}
                if "vertices" in action:
                    assert len(action["vertices"]) == 2
        assert seat.decisions > 20 and seat.env is None and not seat.memory
    assert {"rollDice", "endTurn", "moveRobber", "stealFromPlayer", "buildRoad", "discardResources"} <= kinds


PINNED = json.loads((DATA / "openhex_mcts_plans.json").read_text())


@pytest.mark.parametrize("color", sorted(PINNED))
def test_mcts_plans_without_reuse_are_pinned(color):
    """The seat's MCTS plans (sims 8, dets 1, no reuse) along the record,
    recorded before streams reached the seat: the plain path is what it was."""
    game = _game()
    seat = OpenHexSeat(_mcts(), "fixture", game, color)
    assert seat.env is None and not seat.memory
    plans = [{"seq": seq, "actions": plan.actions, "victim": plan.victim} for seq, plan, _ in _record(seat, game)]
    assert plans == PINNED[color]


@pytest.mark.parametrize("color", ["blue", "red"])
def test_a_reusing_search_follows_the_seats_own_moves(color):
    """The reusing agent along the record: the seat streams its decisions
    (first once, the agent handed copies), the search is re-rooted wherever
    the record played the seat's own choice next and along a plan decided
    ahead on a copy: each discard of a chain after the first, and the robber
    after a knight (the friendly robber reads public points, so no world's
    robber options differ and the engine re-roots there). A discard chain's
    search ends with it (``dropped_over``), so the decision after it is a
    first. The plans fall at the same points as without reuse, every choice
    is legal (a fallback only where the agent raised, as the plain agent does
    at the game's last moves), and the counters add up."""
    game = _game()
    native = _mcts(reuse=True)
    assert native.needs_memory and native.reuse_check == "view"
    log = _logged(native)
    seat = OpenHexSeat(native, "fixture", game, color)
    assert seat.memory and seat.env is not None and not seat.decided
    plans = list(_record(seat, game, log=log))
    assert [(seq, len(plan.actions)) for seq, plan, _ in plans] == \
        [(p["seq"], len(p["actions"])) for p in PINNED[color]]
    c = native.counters
    raised = sum(a["raised"] for a in log)
    assert len(log) == seat.decisions and seat.fallbacks == raised
    assert c["reused"] > 0 and c["fresh"] + c["reused"] == len(log) and c["decisions"] == len(log) - raised
    chains = [acts for _seq, plan, acts in plans if plan.actions[0]["type"] == "discardResources"]
    assert c["dropped_first"] == 0 and c["dropped_over"] == len(chains)
    # First at the game's first decision and after each decision the agent raised on
    # or that ended a discard chain, never else.
    ends = {id(acts[-1]) for acts in chains}
    assert [a["first"] for a in log] == \
        [True] + [log[i - 1]["raised"] or id(log[i - 1]) in ends for i in range(1, len(log))]
    assert c["reused"] + c["refused_diverged"] + c["refused_engine"] == sum(not a["first"] for a in log)
    knights = discards = reused_knights = 0
    for _seq, plan, acts in plans:
        kind = plan.actions[0]["type"]
        if kind == "playKnight":
            assert [a["prompt"] for a in acts] == ["PLAY_TURN", "MOVE_ROBBER"]
            # (The agent raises at the game's last moves: no search there to re-root.)
            assert acts[1].get("reused") == 1 or acts[1]["raised"]
            knights += 1
            reused_knights += acts[1].get("reused", 0)
        elif kind == "discardResources":
            assert len(acts) >= 2 and all(a["prompt"] == "DISCARD" for a in acts)
            assert all(a.get("reused") == 1 for a in acts[1:])
            discards += 1
    assert knights > 0 and reused_knights > 0 and discards == (1 if color == "blue" else 0)
    assert len(native.pool) == 1 and set(native._live) == {seat.env * 2 + seat.seat}
    assert all(entry[1] is not seat.replay.game for entry in native._live.values())
    assert seat.search == dict(c)
    seat.close()


def test_the_robber_after_our_discard_chain_is_searched_from_the_seats_tracker():
    """Blue rolls a seven at seq 162 and alone discards; OpenHex applies exactly the
    plan and asks Blue for the robber (the normal robber, where the engine keeps a
    tree there). The chain was decided on a copy, its worlds dealt from a tracker of
    the copy, which knows nothing of Red's hand, while the seat's tracker knows Red's
    cards: the robber decision is a fresh search dealt from the seat's tracker, not
    the chain's search re-rooted (before, it was, and the robber's hex and victim were
    picked in worlds that contradict what the seat knows: on 4p, where the copy's
    tracker does not pin the other hands)."""
    game = _game()
    game = {**game, "config": {**game["config"], "robberMode": "normal"}}
    native = _mcts(reuse=True)
    log = _logged(native)
    seat = OpenHexSeat(native, "fixture", game, "blue")
    for item in _redacted(game["events"], "blue"):
        if item["seq"] > 162:
            break
        seat.feed([item])

    def state(pending):
        return {"pendingAction": {"type": pending}, "players": [], "board": {}, "currentPlayer": "blue"}

    [discard] = seat.plan(state("discard")).actions
    assert discard["type"] == "discardResources" and sum(discard["resources"].values()) >= 2
    assert len(log) >= 2 and all(a["prompt"] == "DISCARD" for a in log)
    assert log[0]["first"] and all(a.get("reused") == 1 for a in log[1:])
    assert native.counters["dropped_over"] == 1 and not native._live and not seat.decided
    seat.feed([{"seq": 163, "event": {"type": "resourcesDiscarded", "player": "blue",
                                      "resources": discard["resources"]}}])
    red = seat.colors.index(color_of("red"))
    known = list(seat.tracker.known_resources(red))
    assert sum(known) > 0 and sum(cpp.HandTracker(seat.replay.game, seat.seat).known_resources(red)) == 0
    chain = len(log)
    plan = seat.plan(state("moveRobber"))
    assert plan.actions[0]["type"] == "moveRobber"
    [robber] = log[chain:]
    assert robber["prompt"] == "MOVE_ROBBER" and robber["first"] and robber.get("fresh") == 1
    assert not robber.get("reused") and not robber["raised"]
    seat.close()


def test_the_robber_after_a_knight_reuses_the_knights_search():
    """The record under the normal robber (its moves are legal there too):
    the robber decided ahead on a copy after a knight re-roots the knight's
    search, since no world's options depend on hidden cards any more."""
    game = _game()
    game = {**game, "config": {**game["config"], "robberMode": "normal"}}
    native = _mcts(reuse=True)
    log = _logged(native)
    seat = OpenHexSeat(native, "fixture", game, "red")
    knights = [acts for _seq, plan, acts in _record(seat, game, log=log) if plan.actions[0]["type"] == "playKnight"]
    assert len(knights) > 10 and seat.fallbacks == sum(a["raised"] for a in log)
    for acts in knights:
        assert [a["prompt"] for a in acts] == ["PLAY_TURN", "MOVE_ROBBER"]
        # (The agent raises at the game's last moves, as above: no search there to re-root.)
        assert acts[1].get("reused") == 1 or acts[1]["raised"]
    assert native.counters["reused"] >= len(knights) - seat.fallbacks > 10
    seat.close()


@pytest.mark.parametrize("color", ["blue", "red"])
def test_a_bank_short_re_replay_restarts_the_stream(color):
    """The game replayed again on a larger bank (seq 374's roll, fed with its
    production): the agent's remembered game is the old bank's, so the seat
    drops its search (``dropped_over``) and its next decision is a first
    again (nothing to drop there, no stream error); the game goes on. (Blue's
    discard chains before it end their searches alike.)"""
    game = _fair(json.loads(OVERPAID.read_text()))
    checkpoint = game["checkpoints"][0]
    native = _mcts(reuse=True)
    log = _logged(native)
    seat = OpenHexSeat(native, "fixture", game, color)
    events = [e for e in game["events"] if e["seq"] <= checkpoint["seq"]]
    plans = list(_record(seat, game, events=events, log=log))
    assert seat.stats["bank_short"] == 1 and seat.bank_margin == [2, 0, 0, 0, 0]
    assert seat.fallbacks == 0 and not any(a["raised"] for a in log)
    c = native.counters
    chains = [acts for _seq, plan, acts in plans if plan.actions[0]["type"] == "discardResources" and acts]
    assert len(chains) == (2 if color == "blue" else 0)
    assert c["dropped_over"] == 1 + len(chains) and c["dropped_first"] == 0
    assert c["fresh"] + c["reused"] == c["decisions"] == len(log) and c["reused"] > 0
    firsts = [seq for seq, _plan, acts in plans if any(a["first"] for a in acts)]
    assert len(firsts) == 2 + len(chains) and firsts[0] == plans[0][0]
    assert 374 <= firsts[-1] <= checkpoint["seq"]
    assert seat.decided and len(native.pool) == 1 and set(native._live) == {seat.env * 2 + seat.seat}
    _assert_state(seat, checkpoint["bank"], checkpoint["players"])
    seat.close()


def test_a_finished_game_frees_its_searches_and_slot():
    """``close`` (the bot's, however the game ended): the agent drops the
    seat's live search (``dropped_over``, as Blue's discard chain did before)
    and the slot is free for the next game, whose first decision then has
    nothing to drop; closing again does nothing."""
    game = _game()
    native = _mcts(reuse=True)
    seat = OpenHexSeat(native, "fixture", game, "blue")
    env = seat.env
    chains = sum(plan.actions[0]["type"] == "discardResources" for _seq, plan, _acts in _record(seat, game))
    assert chains == native.counters["dropped_over"] == 1
    stream = env * 2 + seat.seat
    assert len(native.pool) == 1 and set(native._live) == {stream} and stream in native._started
    seat.close()
    assert seat.env is None and len(native.pool) == 0 and not native._live and not native._started
    assert native.counters["dropped_over"] == 2
    seat.close()
    assert native.counters["dropped_over"] == 2
    again = OpenHexSeat(native, "again", game, "red")
    assert again.env == env
    plans = _record(again, game)
    while not again.decided:
        next(plans)
    assert native.counters["dropped_first"] == 0 and len(native.pool) == 1 and set(native._live) == {env * 2 + 1}
    again.close()
    assert len(native.pool) == 0 and native.counters["dropped_over"] == 3


def test_an_error_in_the_agent_restarts_the_stream():
    """An act that fails empties the agent's pool and forgets its streams:
    the seat counts the fallback (the default move), and its next decision is
    a first again."""
    class Failing:
        def __init__(self, inner):
            self.obs_size, self.num_actions, self.num_players = inner.obs_size, inner.num_actions, inner.num_players

        def __call__(self, rows):
            raise RuntimeError("evaluator failed")

    game = _game()
    native = _mcts(reuse=True)
    log = _logged(native)
    seat = OpenHexSeat(native, "fixture", game, "blue")
    plans = _record(seat, game, log=log)
    while native.counters["decisions"] < 3:
        next(plans)
    assert seat.decided and native._live
    healthy = native.evaluator
    native.evaluator = Failing(healthy)
    while not log[-1]["raised"]:
        next(plans)
    assert seat.fallbacks == 1 and not seat.decided and len(native.pool) == 0 and not native._live
    native.evaluator = healthy
    while not seat.decided:
        next(plans)
    assert log[-1]["first"] and len(native.pool) == 1 and seat.fallbacks == 1
    assert native.counters["dropped_first"] == 0 and native.counters["decisions"] == len(log) - 1
    seat.close()


def test_native_for_streams_a_reusing_agent_and_refuses_a_recurrent_one():
    """``native_for``: a reusing MCTS is allowed (the seat streams its
    decisions) and compares games as the seat sees them; a plain one keeps
    the exact check; a recurrent POLICY stays refused (the replay keeps no
    decision history)."""
    reusing = tiny_agent("MCTS", "full", DIMS, sims=8, dets=1, reuse=True)
    assert native_for(reusing)[0] is reusing and reusing.reuse_check == "view"
    plain = tiny_agent("MCTS", "full", DIMS, sims=8, dets=1)
    assert native_for(plain)[0] is plain and plain.reuse_check == "exact"
    recurrent = eval_agent("POLICY", eval_model(["base-2p"], preset="lstm-tiny"))
    assert recurrent.needs_memory
    with pytest.raises(ValueError, match="recurrent"):
        native_for(recurrent)
