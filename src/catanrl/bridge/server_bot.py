"""Our agents as catanatron-server seats and TUI players.

:func:`run_bot` takes a seat in a catanatron-server room (see its PROTOCOL.md) with a
:class:`~catanrl.bridge.agent.ViewAgent` and plays it to the end, through
``catanatron.server.client.GameClient``, which does the protocol's fairness
work: it commits to and reveals its seed (signing the commitments' digest),
signs every move bound to those commitments, checks
every message's signature and hash chain and the server's receipts, saves
the transcript and verifies it. Every server message also feeds the
agent (:meth:`ViewAgent.observe`), so the room's supply and every step's hand
sizes are known::

    result = run_bot("MCTS:ckpt=runs/x@best:sims=256:dets=4", "ws://host:8765/room/ROOM")
    result = run_bot("POLICY:ckpt=runs/x@best", "ws://host:8765",
                     new_room={"config": {"num_players": 4}, "seats": [None, "F", "F", "F"]})
    result.verified, result.summary()

:func:`tui_decide_fns` makes ``decide_fns`` for ``catanatron.tui.run``, and
:func:`play_tui` opens a local TUI game in which seats may be our model specs
(:data:`catanrl.agents.spec.MODEL_KEYS`: ``POLICY:...``, ``MCTS:...``,
``BELIEF:...``, played through :class:`ViewAgent`; a ``PREDICT:...`` seat is
routed the same way and refused by its server resolution, as on the server)
next to people and Catanatron's registry bots.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from catanrl.bridge.agent import ViewAgent


@dataclass
class BotResult:
    """A finished server seat: the client's result and the agent's counts."""

    #: ``catanatron.server.client.GameResult`` (winner, transcript path, verified, report).
    game: Any
    agent: dict[str, Any] = field(default_factory=dict)
    room_url: str = ""

    @property
    def verified(self) -> bool:
        return bool(self.game.verified)

    @property
    def winning_color(self) -> str | None:
        return self.game.winning_color

    def summary(self) -> dict[str, Any]:
        return {**self.game.summary(), "room_url": self.room_url, "agent": self.agent}


def room_url(url: str, room_id: str) -> str:
    """``ws://HOST:PORT/room/ROOM_ID`` on the server of ``url``."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, f"/room/{room_id}", "", ""))


def _room_request(new_room: Mapping[str, Any]) -> tuple[dict, list | None, str]:
    """(config, seats, name) from ``{"config", "seats", "name"}`` or a bare config."""
    if "config" in new_room or "seats" in new_room:
        return dict(new_room.get("config") or {}), new_room.get("seats"), str(new_room.get("name", ""))
    return dict(new_room), None, ""


async def play_bot(agent: Any, url: str, *, new_room: Mapping[str, Any] | None = None, seat: int | None = None,
                   name: str | None = None, transcript_dir: str | None = "transcripts", verify: bool = True,
                   device: str = "cpu", infer_rows: int = 1, server_key: str | None = None,
                   key_path: str | None = None, supply: Any = None) -> BotResult:
    """Play one seat of a room with ``agent`` (a spec, a resolved or native
    agent, or a :class:`ViewAgent`); see :func:`run_bot`."""
    from catanatron.server.client import GameClient, create_room

    view_agent = agent if isinstance(agent, ViewAgent) else ViewAgent(
        agent, device=device, infer_rows=infer_rows, supply=supply)
    if new_room is not None:
        config, seats, room_name = _room_request(new_room)
        room = await create_room(url, config, seats, name=room_name)
        url = room_url(url, room["id"])
    client = GameClient(url, name=name or view_agent.label, seat=seat, transcript_dir=transcript_dir,
                        verify=verify, server_key=server_key, key_path=key_path)
    client.on_message = view_agent.observe
    result = await client.play(view_agent.decide)
    return BotResult(game=result, agent=view_agent.summary(), room_url=url)


def run_bot(agent: Any, url: str, *, new_room: Mapping[str, Any] | None = None, seat: int | None = None,
            transcript_dir: str | None = "transcripts", **kwargs: Any) -> BotResult:
    """Play a catanatron-server seat to the end; returns the verified result.

    ``url`` is a room (``ws://HOST:PORT/room/ROOM_ID``), or the server when
    ``new_room`` creates one first: ``{"config": {...room config...},
    "seats": [None, "F", ...], "name": ...}`` (``None`` seats are open; this
    bot takes ``seat``, or the first open one) or a bare room config. The
    transcript is saved under ``transcript_dir`` and verified
    (``verify=False`` skips that). Other keywords go to :func:`play_bot`
    (``name``, ``device``, ``infer_rows``, ``server_key``, ``key_path``,
    ``supply``, ``verify``).
    """
    return asyncio.run(play_bot(agent, url, new_room=new_room, seat=seat, transcript_dir=transcript_dir,
                                **kwargs))


def is_model_spec(spec: str) -> bool:
    """Whether ``spec`` names one of our model keys (``catanrl.agents.spec.MODEL_KEYS``,
    the one enumeration), which the bridge plays through :class:`ViewAgent`; the
    rest are Catanatron registry bots."""
    from catanrl.agents.spec import MODEL_KEYS

    return isinstance(spec, str) and spec.split(":", 1)[0].strip().upper() in MODEL_KEYS


def tui_decide_fns(agents: Mapping[str, Any] | Sequence[str], *, device: str = "cpu", infer_rows: int = 1,
                   supply: Any = None) -> dict[str, Any]:
    """``decide_fns`` for ``catanatron.tui.run``: a name per agent (a spec, a
    resolved or native agent, or a ViewAgent), or a list of specs named by
    themselves (so a seat can be written as its spec)::

        catanatron.tui.run(["human", "nn", "F"], decide_fns=tui_decide_fns({"nn": "POLICY:ckpt=..."}))
        catanatron.tui.run(["human", spec, "F"], decide_fns=tui_decide_fns([spec]))

    The TUI hands a callback views only when its seat decides, so monopolies
    between two decisions are solved from hand sizes (see
    :mod:`catanrl.bridge.tracking`). ``supply`` is the game's supply when its
    bank is not uniform.
    """
    named = dict(agents) if isinstance(agents, Mapping) else {spec: spec for spec in agents}
    out = {}
    for label, agent in named.items():
        view_agent = agent if isinstance(agent, ViewAgent) else ViewAgent(
            agent, device=device, infer_rows=infer_rows, supply=supply)
        out[label] = view_agent.decide
    return out


def play_tui(seats: Sequence[str], *, device: str = "cpu", infer_rows: int = 1, config: Any = None,
             seed: int | None = None, **kwargs: Any) -> Any:
    """Open a local TUI game whose seats (RED, BLUE, ... in order) are
    ``human``, Catanatron registry specs, ``oracle:SPEC``, or our model specs
    (every key in ``catanrl.agents.spec.MODEL_KEYS``, played through
    :class:`ViewAgent` from their views; ``PREDICT`` is refused there with a
    :class:`~catanatron.registry.SpecError` before the TUI opens, as on the
    server). Other keywords go to ``catanatron.tui.run``; returns its session.
    """
    from catanatron.tui import run

    ours = [spec for spec in seats if is_model_spec(spec)]
    supply = (config or {}).get("supply") if isinstance(config, Mapping) else None
    decide_fns = tui_decide_fns(ours, device=device, infer_rows=infer_rows, supply=supply) if ours else None
    return run(list(seats), decide_fns=decide_fns, config=config, seed=seed, **kwargs)
