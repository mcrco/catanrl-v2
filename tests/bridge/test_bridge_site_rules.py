"""The house's 1v1 under OpenHex's rules on catanatron-server (base-2p-fr:
15 points, discard above 9, ``robber_mode: friendly_public`` and ``dice_mode:
balanced``): every seat's replay of its views stays exactly the server's
game, its hand tracker equals one fed by the true transitions, the robber
moves and the balanced dice's odds the replay offers are the game's, an
edited view re-anchors on a snapshot that declares the dice and a knight's
award, and our agents play whole verified games on a real server."""

from __future__ import annotations

import asyncio
import json
import os
import random
from collections import Counter
from pathlib import Path

import cppanatron as cpp
import pytest
from bridge_games import lockstep, tiny_agent
from cppanatron.catanatron import from_catanatron_action

from catanrl import games
from catanrl.bridge.agent import ViewAgent
from catanrl.bridge.position import dice_json, game_rules, shared_map, snapshot, view_balanced_dice
from catanrl.bridge.replay import DiceRefused, Replay
from catanrl.bridge.tracking import SeatTracker

GAME = "base-2p-fr"
GAMES = int(os.environ.get("CATANRL_BRIDGE_GAMES", "1")) + 3
#: The room config of the house's 1v1 table under OpenHex's rules.
SITE = {"num_players": 2, "vps_to_win": 15, "discard_limit": 9, "robber_mode": "friendly_public",
        "dice_mode": "balanced"}


def test_a_site_view_reads_as_base_2p_fr():
    ls = lockstep(GAME, 0, site=True)
    for _ in range(30):
        ls.step(ls.choose())
    view = ls.view(0)
    config = games.load_game(GAME)
    assert view["game"]["robber_mode"] == "friendly_public" and view["game"]["dice_mode"] == "balanced"
    assert game_rules(view) == game_rules(view, config) == config.cppanatron_rules()
    assert game_rules(view, "base") == config.cppanatron_rules()


def _play(index: int) -> Counter:
    """One random game; both seats' trackers sync after every move."""
    ls = lockstep(GAME, index, site=True)
    stats: Counter = Counter()
    bridged = [SeatTracker(view) for view in ls.views()]
    native = [cpp.HandTracker(ls.native, seat) for seat in range(2)]
    for _ in range(3000):
        if ls.over:
            break
        action = ls.choose()
        before = ls.native.copy()
        ls.step(action)
        stats["rolls"] += action.action_type.name == "ROLL"
        for seat in range(2):
            native[seat].observe(before, from_catanatron_action(action), ls.native)
        over = ls.truth.winning_color() is not None
        views = ls.views()
        truth = ls.truth.state
        for seat, tracker in enumerate(bridged):
            tracker.sync(views[seat], game_over=over)
            assert tracker.position.exact, tracker.mismatches
            assert tracker.tracker == native[seat]
            game = tracker.game
            # The replay's balanced dice are the game's, and so are the odds a search plans on.
            assert dice_json(game.balanced_dice, ls.colors) == views[seat]["balanced_dice"]
            if not over:
                roller = game.next_roller
                assert game.roll_probabilities(roller) == pytest.approx(
                    truth.balanced_dice.sum_probabilities(roller), abs=1e-12)
            if not over and ls.current_seat == seat:
                offered = [from_catanatron_action(a) for a in ls.truth.playable_actions]
                tracker.options(offered)
                assert sorted(map(repr, game.playable_actions)) == sorted(map(repr, offered))
                if game.current_prompt == cpp.ActionPrompt.MOVE_ROBBER:
                    stats["robber_prompts"] += 1
                    tiles = {tuple(a.value[0]) for a in offered}
                    stats["shielded"] += len(tiles) < len(truth.board.map.land_tiles) - 1
                    stats["after_an_award"] += game.largest_army_taken_from is not None
    assert ls.over and ls.truth.winning_color() is not None
    for tracker in bridged:
        stats.update(tracker.stats)
    return stats


def test_seat_replays_follow_whole_games_exactly():
    stats = Counter()
    for index in range(GAMES):
        stats += _play(index)
    assert stats["rolls"] > 4 * 23 and stats["robber_prompts"] > 50 and stats["shielded"] > 10
    assert not stats["broken"] and not stats["reanchored"] and not stats["mismatched_syncs"]
    assert not stats["tracker_resets"] and not stats["restarts"]


def _knight_award():
    """A site game at the robber prompt of a knight that took Largest Army."""
    for index in range(40):
        ls = lockstep(GAME, index, site=True)
        for _ in range(3000):
            if ls.over:
                break
            ls.step(ls.choose())
            if ls.truth.state.largest_army_taken_from is not None:
                return ls
    raise AssertionError("no knight took Largest Army")


def test_a_snapshot_declares_the_dice_and_a_knights_award():
    ls = _knight_award()
    seat = ls.current_seat
    view = ls.view(seat)
    assert view["largest_army_taken_from"] is not None
    position = snapshot(view)
    assert position.exact, position.mismatches
    game = position.game
    assert game.largest_army_taken_from == ls.native.largest_army_taken_from is not None
    assert game.balanced_dice == ls.native.balanced_dice == view_balanced_dice({**view, "balanced_dice": None})
    assert sorted(map(repr, game.playable_actions)) == sorted(map(repr, ls.native.playable_actions))
    # A view the log does not explain (an edit) re-anchors there, exactly.
    agent = ViewAgent(tiny_agent("POLICY", "private", ls.dims), strict=True)
    playable = ls.playable_json()
    edited = {**view, "action_records": view["action_records"][:-1]}
    choice = agent.decide(edited, playable)
    assert any(choice is offered for offered in playable)
    tracker = next(iter(agent.seats.values()))
    assert tracker.stats["reanchored"] == 1 and tracker.position.exact


def test_a_replay_refuses_dice_the_balanced_deck_no_longer_holds():
    ls = lockstep(GAME, 1, site=True)
    while sum(r[0][1] == "ROLL" for r in ls.view(0)["action_records"]) < 3:
        ls.step(ls.choose())
    view = ls.view(0)
    records = [list(r) for r in view["action_records"]]
    rolls = [i for i, r in enumerate(records) if r[0][1] == "ROLL"]
    # The third roll shows the first roll's card again: the deck drew it already.
    records[rolls[2]] = [records[rolls[2]][0], records[rolls[0]][1]]
    replay = Replay(view, 0, game_rules(view), shared_map(view["map"]))
    for entry in records[:rolls[2]]:
        replay.step(entry)
    with pytest.raises(DiceRefused, match="no longer holds it"):
        replay.step(records[rolls[2]])
    assert replay.applied == rolls[2]


# ---------------------------------------------------------------- a real server

TIMEOUT = 900


def _run(scenario):
    pytest.importorskip("websockets")
    pytest.importorskip("cryptography")
    from catanatron.server.app import GameServer, run_server

    async def bounded():
        server = GameServer()
        ws_server = None
        rng = random.Random()
        for _ in range(50):
            try:
                port = rng.randrange(30000, 40000)
                ws_server = await run_server(server, "127.0.0.1", port)
                break
            except OSError:
                continue
        base = f"ws://127.0.0.1:{port}"
        try:
            return await asyncio.wait_for(scenario(base), TIMEOUT)
        finally:
            server.shutdown()
            ws_server.close()
            await ws_server.wait_closed()

    return asyncio.run(bounded())


def test_belief_and_mcts_seats_play_a_verified_site_game(tmp_path):
    """Our agents in both seats of one site-rules room: every decision exact
    (no fallback, nothing inexact, the mask the game's), the replays never
    broken, both transcripts verified."""
    from catanatron.server.client import create_room
    from catanatron.server.verify import verify_transcript

    from catanrl.bridge.server_bot import play_bot, room_url

    dims = {"max_players": 2, "max_tiles": 19, "max_nodes": 54, "max_edges": 72}
    belief = ViewAgent(tiny_agent("BELIEF", "full", dims, worlds=2))
    mcts = ViewAgent(tiny_agent("MCTS", "full", dims, sims=8, dets=1))

    async def scenario(base):
        room = await create_room(base, SITE, [None, None])
        url = room_url(base, room["id"])
        return await asyncio.gather(play_bot(belief, url, seat=0, transcript_dir=str(tmp_path)),
                                    play_bot(mcts, url, seat=1, transcript_dir=str(tmp_path)))

    results = _run(scenario)
    for result in results:
        assert result.verified and result.game.aborted is None and result.winning_color is not None
        decisions = result.agent["decisions"]
        assert decisions.get("decisions", 0) >= 3
        assert not [k for k in decisions if k.startswith("fallback") or k in ("inexact", "mask_differs")], decisions
        tracking = result.agent["tracking"]
        assert not tracking.get("broken") and not tracking.get("tracker_resets") \
            and not tracking.get("mismatched_syncs") and not tracking.get("reanchored")
    paths = sorted(Path(tmp_path).glob("*.json"))
    assert len(paths) == 2
    for path in paths:
        transcript = json.loads(path.read_text())
        report = verify_transcript(transcript)
        assert report.ok, report.errors
        joined = next(m["body"] for m in transcript["messages"] if m["body"]["type"] == "joined")
        assert joined["room"]["config"]["dice_mode"] == "balanced"
