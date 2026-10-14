"""End to end: our bots take seats in a real catanatron-server room (started
in this process on a free localhost port), play from their views only, and
finish with verified transcripts."""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path

import pytest

pytest.importorskip("websockets")
pytest.importorskip("cryptography")

from bridge_games import tiny_agent  # noqa: E402
from catanatron.server.app import GameServer, run_server  # noqa: E402
from catanatron.server.client import create_room  # noqa: E402
from catanatron.server.verify import verify_transcript  # noqa: E402

from catanrl.bridge.agent import ViewAgent  # noqa: E402
from catanrl.bridge.server_bot import play_bot, room_url  # noqa: E402

BASE_DIMS = {"max_tiles": 19, "max_nodes": 54, "max_edges": 72}
TIMEOUT = 600


async def _serve():
    """A GameServer on a free localhost port in 30000-39999."""
    server = GameServer()
    rng = random.Random()
    for _ in range(50):
        port = rng.randrange(30000, 40000)
        try:
            ws_server = await run_server(server, "127.0.0.1", port)
        except OSError:
            continue
        return server, ws_server, f"ws://127.0.0.1:{port}"
    raise RuntimeError("no free port in 30000-39999")


def _run(scenario):
    async def bounded():
        server, ws_server, base = await _serve()
        try:
            return await asyncio.wait_for(scenario(base), TIMEOUT)
        finally:
            server.shutdown()
            ws_server.close()
            await ws_server.wait_closed()

    return asyncio.run(bounded())


def _clean(result) -> None:
    """A finished, verified seat whose agent decided every move exactly."""
    assert result.verified, getattr(result.game.report, "errors", None)
    assert result.game.aborted is None and result.winning_color is not None
    decisions = result.agent["decisions"]
    assert decisions.get("decisions", 0) >= 10
    assert not [k for k in decisions if k.startswith("fallback") or k in ("inexact", "mask_differs")], decisions
    tracking = result.agent["tracking"]
    assert not tracking.get("broken") and not tracking.get("tracker_resets") and not tracking.get("mismatched_syncs")


def test_policy_and_mcts_bots_play_a_server_game_against_f(tmp_path):
    dims = {"max_players": 3, **BASE_DIMS}
    policy = ViewAgent(tiny_agent("POLICY", "private", dims))
    mcts = ViewAgent(tiny_agent("MCTS", "full", dims, sims=8, dets=1))

    async def scenario(base):
        room = await create_room(base, {"num_players": 3, "vps_to_win": 6}, [None, None, "F"])
        url = room_url(base, room["id"])
        return await asyncio.gather(
            play_bot(policy, url, seat=0, transcript_dir=str(tmp_path)),
            play_bot(mcts, url, seat=1, transcript_dir=str(tmp_path)))

    results = _run(scenario)
    for result in results:
        _clean(result)
    assert results[0].winning_color == results[1].winning_color
    # The saved transcripts verify on their own, and against each other.
    paths = sorted(Path(tmp_path).glob("*.json"))
    assert len(paths) == 2
    for path in paths:
        report = verify_transcript(json.loads(path.read_text()))
        assert report.ok, report.errors


def test_a_belief_bot_creates_its_room_and_plays_it(tmp_path):
    dims = {"max_players": 2, **BASE_DIMS}
    belief = ViewAgent(tiny_agent("BELIEF", "full", dims, worlds=2))

    async def scenario(base):
        return await play_bot(belief, base, transcript_dir=str(tmp_path),
                              new_room={"config": {"num_players": 2, "vps_to_win": 5}, "seats": [None, "F"]})

    result = _run(scenario)
    _clean(result)
    assert result.room_url.startswith("ws://127.0.0.1:3") and "/room/" in result.room_url
    assert result.summary()["verified"] is True


def test_tui_decide_fns_play_a_local_game():
    """The same agents as TUI ``decide_fns``: they see views only when their
    seat decides (headless LocalSession, as the TUI's own tests drive it)."""
    from catanatron.tui.local import LocalSession
    from catanatron.tui.positions import GameSettings

    from catanrl.bridge.server_bot import tui_decide_fns

    dims = {"max_players": 3, **BASE_DIMS}
    agents = {"search": ViewAgent(tiny_agent("MCTS", "full", dims, sims=8, dets=1)),
              "policy": ViewAgent(tiny_agent("POLICY", "private", dims))}
    session = LocalSession.new(["search", "F", "policy"], settings=GameSettings(num_players=3, vps_to_win=8),
                               seed=3, decide_fns=tui_decide_fns(agents))
    session.run()
    assert session.over and session.winner is not None
    for agent in agents.values():
        summary = agent.summary()
        assert summary["decisions"].get("decisions", 0) >= 5
        assert not [k for k in summary["decisions"] if k.startswith("fallback") or k == "inexact"], summary
        assert not summary["tracking"].get("broken"), summary
