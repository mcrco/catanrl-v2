"""The OpenHex seat and the trades between players: following them, and making them.

The two games here were played on OpenHex's own engine (open-hex ``57e47d0``,
the rules of 2026-11-22: its balanced dice, friendly robber, short bank, bank
trades and player trading) by seats of this bridge that traded freely, with
every event recorded: ``sim-1v1-trades.json`` (1v1 rules, stopped between two
moves after 138 rolls) and ``sim-4p-trades.json`` (a whole 4p game). Each
``final`` is OpenHex's state after the last event, hands included.
"""

from __future__ import annotations

import functools
import json
from collections import Counter
from pathlib import Path

import pytest
from bridge_games import tiny_agent
from test_bridge_openhex import RESOURCE_NAMES, _Never, _redacted

from catanrl.agents.trade import Trade
from catanrl.bridge.openhex import EventLog, OpenHexError, OpenHexSeat, color_of, native_for
from catanrl.bridge.trading import TradeJudge

FIXTURES = Path(__file__).parent.parent / "fixtures" / "openhex"


@functools.cache
def _native():
    """The seat's network (tiny): what binds the engine's game for a judge or a move."""
    return native_for(tiny_agent("POLICY", "private", {"max_players": 4, "max_tiles": 19, "max_nodes": 54,
                                                       "max_edges": 72}))[0]


def _recorded(name):
    return json.loads((FIXTURES / name).read_text())


def _kinds(game):
    return Counter(item["event"]["type"] for item in game["events"])


@pytest.mark.parametrize("name", ["sim-1v1-trades.json", "sim-4p-trades.json"])
def test_a_game_with_trades_replays_to_openhexs_state_for_every_seat(name):
    """Every seat follows the whole game from its own events: the trades between
    players (counter-offers among them), OpenHex's dice, robber and bank."""
    game = _recorded(name)
    kinds = _kinds(game)
    assert kinds["tradeExecuted"] > 100 and kinds["knightPlayed"] > 5 and kinds["bankTradeExecuted"] > 10
    executed = [e["event"] for e in game["events"] if e["event"]["type"] == "tradeExecuted"]
    if name == "sim-1v1-trades.json":
        assert game["config"]["diceMode"] == "balanced" and kinds["diceRolled"] > 100
    final = game["final"]
    for color in game["turnOrder"]:
        seat = OpenHexSeat(_Never(), "fixture", game, color)
        seat.feed(_redacted(game["events"], color))
        assert not seat.fair_dice and not seat.offers
        assert seat.mismatches({**final, "players": [
            p if p["color"] == color else {**p, "hand": {}, "handSize": sum(p["hand"].values())}
            for p in final["players"]]}) == []
        for player in final["players"]:
            i = seat.colors.index(color_of(player["color"]))
            hand = tuple(player["hand"].get(r, 0) for r in RESOURCE_NAMES)
            assert sum(seat.replay.game.player(i).resources) == sum(hand)
            if i == seat.seat:
                assert tuple(seat.replay.game.player(i).resources) == hand
                assert tuple(seat.tracker.known_resources(i)) == hand
        mine = sum(color in (e["player"], e["partner"]) for e in executed)
        assert seat.stats.get("trades", 0) == mine
    if name == "sim-4p-trades.json":
        assert kinds["tradeRejected"] > 100 and kinds["tradeCancelled"] > 0


def test_a_bank_trade_of_several_cards_and_kinds_is_one_engine_trade_a_card():
    game = _recorded("sim-4p-trades.json")
    colors = [color_of(c) for c in game["turnOrder"]]
    seat = OpenHexSeat(_Never(), "fixture", game, game["turnOrder"][0])
    red = colors[0]
    log = EventLog(seat.board, colors, lambda _color: [4, 3, 2, 4, 4])

    def records(offering, requesting):
        entries = log.feed({"type": "bankTradeExecuted", "player": red.lower(), "offering": offering,
                            "requesting": requesting})
        entries += log.settle()
        return [e.record[0][2] for e in entries]

    assert records({"wood": 4}, {"ore": 1}) == [["WOOD", "WOOD", "WOOD", "WOOD", "ORE"]]
    assert records({"wood": 8}, {"wheat": 1, "ore": 1}) == [["WOOD"] * 4 + ["WHEAT"], ["WOOD"] * 4 + ["ORE"]]
    # Several kinds: each at the player's own rate (a 3:1 brick here, a sheep port).
    assert records({"brick": 3, "sheep": 4}, {"ore": 3}) == [
        ["BRICK", "BRICK", "BRICK", None, "ORE"], ["SHEEP", "SHEEP", None, None, "ORE"],
        ["SHEEP", "SHEEP", None, None, "ORE"]]
    with pytest.raises(OpenHexError):
        records({"brick": 3, "sheep": 3}, {"ore": 2})


class _Judge:
    """A judge that says what the test tells it to."""

    def __init__(self, answer=True, offer=None, partner=None):
        self.says, self.offer, self.partner, self.asked = answer, offer, partner, []

    def answer(self, game, seat, tracker, seed):
        self.asked.append(("answer", game.current_player_index))
        return self.says

    def take(self, game, seat, partner, trade):
        self.asked.append(("take", partner, trade))
        return self.says

    def propose(self, game, seat, tracker, seed, key):
        offer, self.offer = self.offer, None
        return offer

    def confirm(self, game, seat):
        return self.partner


def _at_an_offer():
    """(game, the first plain offer's event, its index): OpenHex's events up to
    that offer, when its proposer has rolled and nothing is pending."""
    game = _recorded("sim-4p-trades.json")
    index, event = next((i, e["event"]) for i, e in enumerate(game["events"])
                        if e["event"]["type"] == "tradeProposed" and e["event"].get("counterTo") is None)
    return game, event, index


def _seat(game, color, index, judge, **kwargs):
    seat = OpenHexSeat(_native(), "fixture", game, color, judge=judge, **kwargs)
    seat.feed(_redacted(game["events"][:index], color))
    return seat


def _state(current):
    return {"currentPlayer": current, "pendingAction": None}


def test_a_seat_answers_the_current_players_offer_once_as_its_judge_says():
    game, event, index = _at_an_offer()
    proposer = event["player"]
    other = next(c for c in game["turnOrder"] if c != proposer)
    for says, kind in ((True, "acceptTrade"), (False, "rejectTrade")):
        judge = _Judge(says)
        seat = _seat(game, other, index, judge)
        assert seat.plan(_state(proposer)) is None  # no offer stands
        seat.feed([game["events"][index]])
        plan = seat.plan(_state(proposer))
        assert plan.trade and plan.actions == [{"type": kind, "tradeId": event["tradeId"]}]
        # The judge saw the game at this seat's answer to that offer.
        assert judge.asked == [("answer", seat.seat)]
        assert seat.plan(_state(proposer)) is None  # answered once
    # Without a judge the seat follows the trading and answers nothing.
    seat = _seat(game, other, index + 1, None)
    assert seat.plan(_state(proposer)) is None and list(seat.offers) == [event["tradeId"]]


def test_the_real_judge_answers_an_offer_on_the_seats_own_network():
    game, event, index = _at_an_offer()
    proposer = event["player"]
    other = next(c for c in game["turnOrder"] if c != proposer)
    native = _native()
    seat = OpenHexSeat(native, "fixture", game, other, judge=TradeJudge(native, dets=2))
    seat.feed(_redacted(game["events"][:index + 1], other))
    plan = seat.plan(_state(proposer))
    assert [a["type"] for a in plan.actions] in (["acceptTrade"], ["rejectTrade"])
    assert seat.judge.stats["answers"] == 1


def test_the_current_player_offers_waits_for_the_answers_and_confirms_or_cancels():
    game, event, index = _at_an_offer()
    proposer = event["player"]
    others = [c for c in game["turnOrder"] if c != proposer]
    give = tuple(event["offering"].get(r, 0) for r in RESOURCE_NAMES)
    get = tuple(event["requesting"].get(r, 0) for r in RESOURCE_NAMES)
    judge = _Judge(offer=Trade(give, get))
    seat = _seat(game, proposer, index, judge, trade_wait=1000.0)
    plan = seat.plan(_state(proposer))
    assert plan.trade and plan.actions == [{"type": "proposeTrade", "offering": event["offering"],
                                            "requesting": event["requesting"]}]
    assert seat.plan(_state(proposer)).actions == []  # sent: its event is on its way
    seat.sent(plan.actions[0], {"success": True, "events": [game["events"][index]]})
    offer = seat.offers[event["tradeId"]]
    assert (offer.proposer, offer.give, offer.get) == (color_of(proposer), give, get)
    assert seat.plan(_state(proposer)).actions == []  # waiting on the answers
    seat.feed([game["events"][index]])  # (the same event from the feed changes nothing)
    assert list(seat.offers) == [event["tradeId"]]
    seq = game["events"][index]["seq"]
    answers = [{"seq": seq + 1 + i, "event": {"type": "tradeAccepted" if i == 0 else "tradeRejected",
                                              "player": c, "tradeId": event["tradeId"]}}
               for i, c in enumerate(others)]
    seat.feed(answers[:1])
    assert seat.plan(_state(proposer)).actions == []  # one seat has not answered, and there is time
    seat.feed(answers[1:])
    judge.partner = seat.colors.index(color_of(others[0]))
    confirm = {"type": "confirmTrade", "tradeId": event["tradeId"], "partner": others[0]}
    assert seat.plan(_state(proposer)).actions == [confirm]
    # OpenHex refuses it (the acceptee's cards went elsewhere): nobody else accepted, so the offer is cancelled.
    seat.refused(confirm)
    cancel = {"type": "cancelTrade", "tradeId": event["tradeId"]}
    assert seat.plan(_state(proposer)).actions == [cancel]
    seat.feed([{"seq": seq + 10, "event": {"type": "tradeCancelled", "player": proposer,
                                           "tradeId": event["tradeId"], "reason": "cancelled"}}])
    assert not seat.offers


def test_an_offer_nobody_accepts_is_cancelled_when_the_wait_is_over():
    game, event, index = _at_an_offer()
    proposer = event["player"]
    give = tuple(event["offering"].get(r, 0) for r in RESOURCE_NAMES)
    get = tuple(event["requesting"].get(r, 0) for r in RESOURCE_NAMES)
    seat = _seat(game, proposer, index, _Judge(offer=Trade(give, get)), trade_wait=0.0)
    plan = seat.plan(_state(proposer))
    seat.sent(plan.actions[0], {"success": True, "events": [game["events"][index]]})
    assert seat.plan(_state(proposer)).actions == [{"type": "cancelTrade", "tradeId": event["tradeId"]}]


def test_the_current_player_takes_or_rejects_a_counter_offer():
    game, event, index = _at_an_offer()
    proposer = event["player"]
    other = next(c for c in game["turnOrder"] if c != proposer)
    counter = {"seq": game["events"][index]["seq"] + 1, "event": {
        "type": "tradeProposed", "player": other, "tradeId": event["tradeId"] + 1, "counterTo": event["tradeId"],
        "offering": event["requesting"], "requesting": event["offering"]}}
    give = tuple(event["offering"].get(r, 0) for r in RESOURCE_NAMES)
    get = tuple(event["requesting"].get(r, 0) for r in RESOURCE_NAMES)
    for says, kind in ((True, "confirmTrade"), (False, "rejectTrade")):
        judge = _Judge(says)
        seat = _seat(game, proposer, index + 1, judge)
        seat.feed([counter])
        plan = seat.plan(_state(proposer))
        assert plan.trade and plan.actions[0] == {"type": kind, "tradeId": event["tradeId"] + 1}
        # The judge weighed it from our side: we give what the counter-offer asks.
        assert judge.asked == [("take", seat.colors.index(color_of(other)), Trade(give, get))]


def test_the_trading_ends_within_its_budget_and_the_move_follows(monkeypatch):
    game, event, index = _at_an_offer()
    proposer = event["player"]
    clock = [50.0]
    monkeypatch.setattr("catanrl.bridge.openhex.time.monotonic", lambda: clock[0])
    native = _native()
    give = tuple(event["offering"].get(r, 0) for r in RESOURCE_NAMES)
    get = tuple(event["requesting"].get(r, 0) for r in RESOURCE_NAMES)
    judge = _Judge(offer=Trade(give, get))
    seat = OpenHexSeat(native, "fixture", game, proposer, judge=judge, trade_wait=0.0, trade_budget=20.0)
    seat.feed(_redacted(game["events"][:index], proposer))
    plan = seat.plan(_state(proposer))
    seat.sent(plan.actions[0], {"success": True, "events": [game["events"][index]]})
    clock[0] += 30.0
    plan = seat.plan(_state(proposer))  # the wait is over: cancelled
    assert plan.actions == [{"type": "cancelTrade", "tradeId": event["tradeId"]}]
    seat.feed([{"seq": game["events"][index]["seq"] + 1, "event": {
        "type": "tradeCancelled", "player": proposer, "tradeId": event["tradeId"], "reason": "cancelled"}}])
    judge.offer = Trade(give, get)  # the judge would offer again, but the budget is spent
    plan = seat.plan(_state(proposer))
    assert not plan.trade and plan.actions and plan.actions[0]["type"] not in ("proposeTrade", "cancelTrade")
    # OpenHex's deadline did not restart for the trading: the move started where the trading did.
    assert seat.move_started == 50.0 and seat.trading_since is None


def test_the_bot_tells_the_seat_what_openhex_said_and_a_refused_trade_action_costs_no_move(monkeypatch):
    """The reply of an action carries its events (the seat reads its offer's id
    there); a trade action OpenHex refuses is no failed move: the seat is told,
    and no default move is played in its place."""
    from test_bridge_openhex import _idle_bot, _TimedClient

    from catanrl.bridge.openhex import OpenHexHTTPError, Plan

    plans = [Plan([{"type": "proposeTrade"}], trade=True), Plan([]), Plan([{"type": "confirmTrade"}], trade=True),
             Plan([{"type": "endTurn"}])]
    told = []

    class Seat:
        last_seq, stats = 0, {}

        def feed(self, _events):
            pass

        def check(self, _state):
            return []

        def plan(self, _state):
            return plans.pop(0) if plans else None

        def sent(self, action, reply):
            told.append(("sent", action["type"], reply))

        def refused(self, action):
            told.append(("refused", action["type"]))

        def fallback(self, _state):
            raise AssertionError("a refused trade action is not answered with a default move")

        def close(self):
            pass

    class Client(_TimedClient):
        def state(self, _game_id):
            self.polls += 1
            status = "active" if self.polls <= 6 else "completed"
            return {"state": {}, "game": {"status": status, "players": []}, "your_color": "red"}

        def act(self, _game_id, action, duration_ms):
            self.acted.append(action["type"])
            if action["type"] == "confirmTrade":
                raise OpenHexHTTPError(400, "Partner can no longer afford the trade")
            return {"success": True, "events": [{"seq": 7, "event": {"type": "x"}}]}

    clock = [0.0]
    monkeypatch.setattr("catanrl.bridge.openhex.time.sleep", lambda _s: None)
    monkeypatch.setattr("catanrl.bridge.openhex.OpenHexSeat", lambda *_a, **_k: Seat())
    bot = _idle_bot()
    bot.client = Client(clock)
    bot.native = None
    bot.board_of = lambda _game_id, _state: {}
    bot._record = lambda *_a: {}
    bot.play("g")
    assert bot.client.acted == ["proposeTrade", "confirmTrade", "endTurn"]
    assert [t[:2] for t in told] == [("sent", "proposeTrade"), ("refused", "confirmTrade"), ("sent", "endTurn")]
    assert told[0][2]["events"][0]["seq"] == 7
