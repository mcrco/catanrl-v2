"""Domestic trades through the server bridge: in a catanatron-server room
with ``trades``, our bots (a ViewAgent with a TradeJudge) answer a person's
offers, confirm acceptees and offer trades of their own, and every seat
still decides exactly and finishes with a verified transcript."""

from __future__ import annotations

import asyncio
import random

import pytest

pytest.importorskip("websockets")
pytest.importorskip("cryptography")

from bridge_games import tiny_agent
from catanatron.models.player import Color
from catanatron.server.bot import ViewBot
from catanatron.server.client import GameClient, create_room
from catanatron.server.verify import verify_transcript
from test_bridge_server import BASE_DIMS, _clean, _run

from catanrl.bridge.agent import ViewAgent
from catanrl.bridge.server_bot import play_bot, room_url

KINDS = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")


def _person(url: str, seed: int):
    """A seat played by Catanatron's F from its view that also trades as a
    person might: offers a random one-for-one when it may, accepts every
    other offer it can pay, confirms the first acceptee."""
    client = GameClient(url, name="person", transcript_dir=None)
    rng = random.Random(seed)
    bots: dict = {}
    counts = {"offers": 0, "accepts": 0}

    def decide(view, playable):
        kinds = {a[1] for a in playable}
        if "CONFIRM_TRADE" in kinds:
            return next(a for a in playable if a[1] == "CONFIRM_TRADE")
        if "REJECT_TRADE" in kinds:
            accept = [a for a in playable if a[1] == "ACCEPT_TRADE"]
            if accept and rng.random() < 0.5:
                counts["accepts"] += 1
                return accept[0]
            return next(a for a in playable if a[1] == "REJECT_TRADE")
        color = client.table.color
        if view.get("offer_trade") and rng.random() < 0.6:
            seat = view["colors"].index(color)
            hand = [view["player_state"][f"P{seat}_{k}_IN_HAND"] for k in KINDS]
            held = [i for i, n in enumerate(hand) if n]
            give = rng.choice(held)
            ask = rng.choice([i for i in range(5) if i != give])
            value = [0] * 10
            value[give], value[5 + ask] = 1, 1
            counts["offers"] += 1
            return [color, "OFFER_TRADE", value]
        bot = bots.get(color) or bots.setdefault(color, ViewBot("F", Color[color], seed=1))
        return bot.decide(view, playable, client.table.supply)

    return client, decide, counts


def test_bots_answer_confirm_and_offer_trades_in_a_room_with_trades(tmp_path):
    dims = {"max_players": 3, **BASE_DIMS}
    # Zero margins: the untrained network's tiny value differences decide, so trades happen often.
    agent = ViewAgent(tiny_agent("POLICY", "private", dims),
                      trades={"margin": 0.0, "accept_margin": 0.0, "dets": 2, "offers": 2})

    async def scenario(base):
        room = await create_room(base, {"num_players": 3, "vps_to_win": 7, "trades": True}, [None] * 3)
        url = room_url(base, room["id"])
        person, decide, counts = _person(url, seed=5)
        results = await asyncio.gather(
            play_bot(agent, url, seat=0, transcript_dir=str(tmp_path)),
            play_bot(agent, url, seat=1, transcript_dir=str(tmp_path)),
            person.play(decide))
        return results, person, counts

    results, person, counts = _run(scenario)
    for result in results[:2]:
        _clean(result)
    assert verify_transcript(person.transcript()).ok
    stats = agent.judge.stats
    assert counts["offers"] > 0 and stats["answers"] > 0, (counts, stats)
    assert stats["fronts"] > 0, stats
    kinds = [entry[0][1] for entry in person.table.log]
    assert "OFFER_TRADE" in kinds and "CONFIRM_TRADE" in kinds, (counts, stats)
    assert any(entry[0][1] == "OFFER_TRADE" and entry[0][0] != person.table.color for entry in person.table.log), \
        ("no bot offered", stats)


def test_without_a_judge_the_bot_rejects_and_cancels():
    """A ViewAgent without trades still plays a room with trades: it rejects
    every offer (and so never reaches a confirm) and never offers."""
    dims = {"max_players": 3, **BASE_DIMS}
    agent = ViewAgent(tiny_agent("POLICY", "private", dims))

    async def scenario(base):
        room = await create_room(base, {"num_players": 3, "vps_to_win": 6, "trades": True}, [None] * 3)
        url = room_url(base, room["id"])
        person, decide, counts = _person(url, seed=7)
        results = await asyncio.gather(
            play_bot(agent, url, seat=0, transcript_dir=None),
            play_bot(agent, url, seat=1, transcript_dir=None),
            person.play(decide))
        return results, person, counts

    results, person, counts = _run(scenario)
    for result in results[:2]:
        _clean(result)
    bots = {r.game.color for r in results[:2]}
    log = [entry[0] for entry in person.table.log]
    assert counts["offers"] > 0
    assert not [a for a in log if a[0] in bots and a[1] in ("OFFER_TRADE", "ACCEPT_TRADE", "CONFIRM_TRADE")]
    assert agent.stats["trade_defaults"] > 0
