"""The house: keep tables open on a catanatron-server for people to play our bots.

A *table* is one kind of game (a room config and an agent spec). For each
table the house keeps exactly one room waiting, with our bots in every seat
but one, so a person who arrives always finds a free seat; when a person
sits and the game starts, the house opens the next room at once. Rooms in
play count against the table's ``max_games`` (the waiting one included), so
the house never runs more games than the server and this machine can carry::

    tables = [Table("ppo-4p", "POLICY:ckpt=ppo-4p/main@best",
                    {"name": "4p vs ppo-4p", "config": {"num_players": 4}}, max_games=4)]
    asyncio.run(House("wss://catan.marcoya.ng", tables, state_dir="/var/lib/catanrl").run())

Every seat of a table is played by one shared :class:`ViewAgent` (one copy of
the network, a tracker per game and seat), each seat with its own client,
key and verified transcript. On a server in house mode the house shows its
token (``CATANATRON_HOUSE_TOKEN``), which is what lets it open rooms there
and keeps people to one seat per room.

Bots answer after at least ``think_s`` (half that for a forced move), so the
person can follow the game.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from catanrl.bridge.agent import ViewAgent
from catanrl.bridge.server_bot import room_url

logger = logging.getLogger(__name__)

#: Boards of a table's ``map`` generator the server must accept at startup (about 5-15 ms each).
MAP_CHECK_SEEDS = 16


@dataclass
class Table:
    """One kind of game the house keeps open."""

    name: str
    #: An agent spec (or a ViewAgent, playing every seat of the table).
    spec: Any
    #: ``{"name": ..., "config": {...}}``: the room to open (every seat open).
    room: Mapping[str, Any]
    #: Rooms at once, the waiting one included.
    max_games: int = 4
    #: Seats our bots take (default: all but one).
    bots: int | None = None
    #: A display name for the bots' seats (default: the table's name).
    label: str | None = None
    #: A ``catanrl.maps`` generator (e.g. ``{"type": "blob", "land_tiles": [15, 30]}``): every
    #: room gets a fresh board from it, sent as the room config's explicit ``map``. None: the
    #: room config's own ``map`` or ``map_type``.
    map: Mapping[str, Any] | None = None

    def room_config(self, seed: int) -> dict:
        """The config of the next room: ``room.config``, with a board drawn by ``seed``."""
        config = dict(self.room.get("config") or {})
        if self.map is not None:
            from catanrl import maps

            config["map"] = maps.generate(self.map, seed)
        return config

    @property
    def num_players(self) -> int:
        return int(dict(self.room.get("config") or {}).get("num_players", 4))

    @property
    def bot_seats(self) -> int:
        return self.num_players - 1 if self.bots is None else int(self.bots)


@dataclass
class TableState:
    table: Table
    agent: ViewAgent | None = None
    games: set = field(default_factory=set)
    failures: int = 0
    stats: dict = field(default_factory=lambda: {"opened": 0, "played": 0, "void": 0, "house_won": 0})


class House:
    def __init__(self, url: str, tables: Sequence[Table], *, state_dir: str | os.PathLike = "house",
                 server_key: str | None = None, think_s: float = 0.8, device: str = "cpu",
                 keep_transcripts: int = 3000, house_token: str | None = None) -> None:
        self.url = url
        self.states = [TableState(t) for t in tables]
        self.state_dir = Path(state_dir)
        self.server_key = server_key
        self.think_s = float(think_s)
        self.device = device
        self.keep_transcripts = int(keep_transcripts)
        #: Shown to the server (None: CATANATRON_HOUSE_TOKEN).
        self.house_token = house_token

    # ------------------------------------------------------------ running

    async def run(self) -> None:
        for state in self.states:
            t = state.table
            if not 1 <= t.bot_seats < t.num_players:
                raise ValueError(f"table {t.name}: {t.bot_seats} bot seats of {t.num_players}")
            if t.map is not None:  # a generator the server would refuse fails here, not per room
                from catanatron.server.config import GameConfig

                for seed in range(MAP_CHECK_SEEDS):  # several boards: a land_tiles range has many sizes
                    try:
                        GameConfig.from_json(t.room_config(seed))
                    except (KeyError, TypeError, ValueError) as error:
                        message = f"table {t.name}: map {dict(t.map)}, seed {seed}: {error}"
                        raise ValueError(message) from error
            # One network per table, loaded once, shared by every seat.
            state.agent = t.spec if isinstance(t.spec, ViewAgent) else ViewAgent(
                t.spec, device=self.device, max_games=8 * t.max_games * t.bot_seats)
            logger.info("table %s: %s, %d bot seats, up to %d games", t.name, state.agent.label, t.bot_seats,
                        t.max_games)
        await asyncio.gather(*(self._keep_open(state) for state in self.states))

    async def _keep_open(self, state: TableState) -> None:
        """Keep one room of this table waiting, while there is room for it."""
        table = state.table
        while True:
            if len(state.games) >= table.max_games:
                await asyncio.wait(state.games, return_when=asyncio.FIRST_COMPLETED)
                continue
            started = asyncio.Event()
            game = asyncio.create_task(self._game(state, started))
            state.games.add(game)
            game.add_done_callback(state.games.discard)
            waiter = asyncio.create_task(started.wait())
            await asyncio.wait({game, waiter}, return_when=asyncio.FIRST_COMPLETED)
            waiter.cancel()
            if started.is_set():
                state.failures = 0
                continue
            # The room ended without a game: the server went away, refused
            # us, or a seat could not be taken. Try again, more slowly.
            state.failures += 1
            delay = min(60.0, 2.0 ** min(state.failures, 6))
            logger.warning("table %s: no game (%s); opening again in %.0f s", table.name,
                           _error_of(game), delay)
            await asyncio.sleep(delay)

    async def _game(self, state: TableState, started: asyncio.Event) -> None:
        from catanatron.server.client import GameClient, create_room

        table = state.table
        assert state.agent is not None
        agent = state.agent
        config = table.room_config(secrets.randbits(63))  # a fresh board per room, given a map
        room = await create_room(self.url, config, [None] * table.num_players,
                                 name=str(table.room.get("name", table.name)), house_token=self.house_token)
        state.stats["opened"] += 1
        url = room_url(self.url, room["id"])
        transcripts = self.state_dir / "transcripts" / table.name
        transcripts.mkdir(parents=True, exist_ok=True)

        def on_message(body: dict) -> None:
            if body.get("type") == "commitments":
                started.set()  # every seat is taken: a person sat
            agent.observe(body)

        clients = []
        for index in range(table.bot_seats):
            client = GameClient(url, name=table.label or table.name, server_key=self.server_key,
                                key_path=str(self.state_dir / f"{table.name}-{index}.key"),
                                transcript_dir=str(transcripts), house_token=self.house_token)
            client.on_message = on_message
            clients.append(client)
        logger.info("table %s: room %s open", table.name, room["id"])
        tasks = [asyncio.create_task(client.play(self._decide_fn(agent))) for client in clients]
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            raise
        for task in pending:  # a seat failed: the room cannot be played
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        errors = [t.exception() for t in done if not t.cancelled() and t.exception() is not None]
        if errors:
            raise errors[0]
        self._record(state, room, [t.result() for t in tasks])
        self._prune(transcripts)

    def _decide_fn(self, agent: ViewAgent):
        think = self.think_s

        async def decide(view, playable_actions):
            start = time.monotonic()
            action = await asyncio.to_thread(agent.decide, view, playable_actions)
            pause = (think if len(playable_actions) > 1 else think / 2) - (time.monotonic() - start)
            if pause > 0:
                await asyncio.sleep(pause)
            return action

        return decide

    # ------------------------------------------------------------ records

    def _record(self, state: TableState, room: Mapping[str, Any], results: list) -> None:
        first = results[0]
        if first.game_id is None:
            return  # void before any game (the server went away): no game to record
        ours = {r.color for r in results}
        if first.aborted:
            state.stats["void"] += 1
        else:
            state.stats["played"] += 1
            state.stats["house_won"] += int(first.winning_color in ours)
        line = {
            "time": time.time(), "table": state.table.name, "room_id": room["id"], "game_id": first.game_id,
            "house_colors": sorted(c for c in ours if c), "winning_color": first.winning_color,
            "aborted": (first.aborted or {}).get("reason"), "verified": all(r.verified for r in results),
        }
        logger.info("table %s: %s", state.table.name, json.dumps(line))
        with open(self.state_dir / "games.jsonl", "a") as file:
            file.write(json.dumps(line) + "\n")

    def _prune(self, directory: Path) -> None:
        files = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime)
        for path in files[: max(0, len(files) - self.keep_transcripts)]:
            path.unlink(missing_ok=True)


def _error_of(task: asyncio.Task) -> str:
    if task.cancelled():
        return "cancelled"
    error = task.exception()
    return f"{type(error).__name__}: {error}"[:200] if error else "ended"


def load_tables(path: str | os.PathLike) -> tuple[dict, list[Table]]:
    """(settings, tables) from a YAML file: ``server``, ``server_key``,
    ``think_s`` and a list of ``tables`` (``name``, ``spec``, ``room``,
    ``max_games``, optional ``bots``, ``label`` and ``map``)."""
    import yaml

    data = yaml.safe_load(Path(path).read_text())
    tables = [Table(**entry) for entry in data.pop("tables")]
    return data, tables
