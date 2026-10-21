"""The house keeps a room open per table on a house-mode server: when a
person sits and the game starts, the next room opens at once; rooms in play
count against the table's limit; every finished game is recorded. A table
with a map generator opens each room on a fresh board, and a generator that
fails, or draws boards the server would refuse, fails before any room opens."""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("websockets")
pytest.importorskip("cryptography")

from bridge_games import tiny_agent
from catanatron.server.app import GameServer, run_server
from catanatron.server.client import GameClient, list_rooms

from catanrl.bridge.agent import ViewAgent
from catanrl.bridge.house import House, Table, load_tables
from catanrl.bridge.server_bot import room_url

TOKEN = "house-token-for-tests-0123456789"


def _waiting(rooms, players):
    return [r for r in rooms if r["status"] == "waiting" and r["config"]["num_players"] == players
            and len(r["house_seats"]) == players - 1 and all(r["seats"][i]["connected"] for i in r["house_seats"])]


async def _until(predicate, timeout=60):
    async with asyncio.timeout(timeout):
        while not (result := await predicate()):
            await asyncio.sleep(0.05)
        return result


def test_the_house_keeps_one_seat_open_and_respects_its_limit(tmp_path, monkeypatch):
    dims = {"max_players": 3, "max_tiles": 19, "max_nodes": 54, "max_edges": 72}
    agent = ViewAgent(tiny_agent("POLICY", "private", dims))
    table = Table("tiny-3p", agent, {"name": "3p vs tiny", "config": {"num_players": 3, "vps_to_win": 5}},
                  max_games=2)

    async def scenario():
        server = GameServer(house_token=TOKEN)
        ws_server = await run_server(server, "127.0.0.1", 0)
        base = f"ws://127.0.0.1:{ws_server.sockets[0].getsockname()[1]}"
        monkeypatch.delenv("CATANATRON_HOUSE_TOKEN", raising=False)
        house = asyncio.create_task(House(base, [table], state_dir=tmp_path, think_s=0, house_token=TOKEN).run())
        try:
            first = (await _until(lambda: _ready(base)))[0]
            # The person shows no token.
            person = GameClient(room_url(base, first["id"]), name="person", transcript_dir=None)
            game = asyncio.create_task(person.play(lambda view, actions: actions[0]))
            second = (await _until(lambda: _ready(base, other_than=first["id"])))[0]
            # The table's limit is two rooms: with both taken, no third opens.
            other = GameClient(room_url(base, second["id"]), name="person2", transcript_dir=None)
            game2 = asyncio.create_task(other.play(lambda view, actions: actions[0]))
            await _until(lambda: _playing(base, 2))
            await asyncio.sleep(0.5)
            assert not _waiting(await list_rooms(base), 3)
            results = await asyncio.wait_for(asyncio.gather(game, game2), 300)
            # A finished game frees its slot: a room opens again.
            await _until(lambda: _ready(base))
            return results
        finally:
            house.cancel()
            await asyncio.gather(house, return_exceptions=True)
            server.shutdown()
            ws_server.close()
            await ws_server.wait_closed()

    async def _ready(base, other_than=None):
        return [r for r in _waiting(await list_rooms(base), 3) if r["id"] != other_than]

    async def _playing(base, n):
        return sum(r["status"] == "playing" for r in await list_rooms(base)) >= n

    results = asyncio.run(scenario())
    assert all(r.winning_color for r in results)
    lines = [json.loads(line) for line in (tmp_path / "games.jsonl").read_text().splitlines()]
    assert len(lines) == 2 and all(line["verified"] and len(line["house_colors"]) == 2 for line in lines)
    # Two seats' transcripts per game (and the room open at shutdown's, of no game).
    saved = [p for p in (tmp_path / "transcripts" / "tiny-3p").glob("*.json") if "-nogame-" not in p.name]
    assert len(saved) == 4


def test_a_table_with_a_map_generator_opens_each_room_on_a_fresh_board(tmp_path, monkeypatch):
    dims = {"max_players": 2, "max_tiles": 30, "max_nodes": 122, "max_edges": 151}
    agent = ViewAgent(tiny_agent("POLICY", "private", dims))
    table = Table("blob-1v1", agent, {"name": "1v1 island", "config": {"num_players": 2, "vps_to_win": 5}},
                  max_games=2, map={"type": "blob", "land_tiles": [15, 20]})

    async def scenario():
        server = GameServer(house_token=TOKEN)
        ws_server = await run_server(server, "127.0.0.1", 0)
        base = f"ws://127.0.0.1:{ws_server.sockets[0].getsockname()[1]}"
        monkeypatch.delenv("CATANATRON_HOUSE_TOKEN", raising=False)
        house = asyncio.create_task(House(base, [table], state_dir=tmp_path, think_s=0, house_token=TOKEN).run())
        seen, rooms, results = set(), [], []
        try:
            for _ in range(2):
                room = (await _until(lambda: _ready(base, seen)))[0]
                seen.add(room["id"])
                rooms.append(room)
                person = GameClient(room_url(base, room["id"]), name="person", transcript_dir=None)
                results.append(await asyncio.wait_for(person.play(lambda view, actions: actions[0]), 300))
            return rooms, results
        finally:
            house.cancel()
            await asyncio.gather(house, return_exceptions=True)
            server.shutdown()
            ws_server.close()
            await ws_server.wait_closed()

    async def _ready(base, seen):
        return [r for r in _waiting(await list_rooms(base), 2) if r["id"] not in seen]

    rooms, results = asyncio.run(scenario())
    boards = [r["config"]["map"] for r in rooms]
    assert all(15 <= sum(t["type"] == "land" for t in board["tiles"]) <= 20 for board in boards)
    assert json.dumps(boards[0], sort_keys=True) != json.dumps(boards[1], sort_keys=True)
    assert all(r.winning_color and r.verified for r in results)
    lines = [json.loads(line) for line in (tmp_path / "games.jsonl").read_text().splitlines()]
    assert len(lines) == 2 and all(line["verified"] for line in lines)


def _tiny_table(name, players, generator):
    dims = {"max_players": players, "max_tiles": 30, "max_nodes": 122, "max_edges": 151}
    agent = ViewAgent(tiny_agent("POLICY", "private", dims))
    return Table(name, agent, {"name": name, "config": {"num_players": players}}, map=generator)


def test_an_unknown_map_type_fails_in_maps_generate_before_any_room_opens(tmp_path):
    table = _tiny_table("bad", 2, {"type": "nonsense"})
    with pytest.raises(ValueError, match="table bad: .*unknown map type"):
        asyncio.run(House("ws://127.0.0.1:9", [table], state_dir=tmp_path, think_s=0, house_token=TOKEN).run())


def test_a_generator_drawing_boards_the_server_refuses_fails_before_any_room_opens(tmp_path):
    from catanatron.server.config import GameConfig

    # 5-30 land tiles for 4 players: board 0 is fine, but some small boards cannot seat
    # everyone, so the startup check draws several boards, not only one.
    table = _tiny_table("small", 4, {"type": "blob", "land_tiles": [5, 30]})
    GameConfig.from_json(table.room_config(0))
    with pytest.raises(ValueError, match=r"table small: .*seed \d+: .*settlement spots"):
        asyncio.run(House("ws://127.0.0.1:9", [table], state_dir=tmp_path, think_s=0, house_token=TOKEN).run())


def test_load_tables_reads_the_map_key(tmp_path):
    path = tmp_path / "house.yaml"
    path.write_text("server: ws://x\ntables:\n"
                    "  - {name: t, spec: POLICY:x, map: {type: blob, land_tiles: [15, 30]},\n"
                    "     room: {name: t, config: {num_players: 4}}}\n")
    _, (table,) = load_tables(path)
    assert table.map == {"type": "blob", "land_tiles": [15, 30]}
    config = table.room_config(7)
    assert config["num_players"] == 4 and "tiles" in config["map"]
    assert table.room_config(7) == config and table.room_config(8) != config
