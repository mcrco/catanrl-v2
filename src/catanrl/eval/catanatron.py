"""The Catanatron harness: play a schedule of games with Catanatron ``Player``s.

``play_catanatron(lineup, games, ...)`` plays every ``ScheduledGame`` as a
Catanatron game on the same map and game seed the native harness would use
(``GameConfig.catanatron_game``), with the schedule's seat order forced (a
Catanatron game shuffles its seats from the game RNG; the reseating happens
before any action, as ``cppanatron.catanatron.reference_game`` does) and the
suite's turn limit (fair dice only: a game with ``dice_mode`` balanced is
refused before anything is played). Each seat is ``agent.player(color)``: POLICY, MCTS,
BELIEF and PREDICT through their cppanatron mirror, Catanatron's own bots,
and ``LLM`` players, which exist only here. Rows have the native harness's shape
(``game_id``, ``seats``, ``winner``, ``winner_seat``, ``turns``,
``truncated``, ``stuck``, ``scores``, ``forfeits``, ``decisions``) and, when
a seat is an LLM, ``llm``: that player's per-game stats per lineup slot
(``None`` for other slots), which :func:`catanrl.eval.results.summarize`
totals and ``verify`` ignores (wall time is not an outcome).

Games are independent, so ``threads`` of them run at once (a thread pool: a
model call waits on the network and releases the GIL); rows are delivered
as games finish. Catanatron's bots draw from the game's RNG, so a row depends
on the game seed and the seat order, not on ``threads``; a real model is not
deterministic at all. ``preflight`` builds every agent's player once before
the first game and calls its ``preflight()`` when it has one (the LLM's model
client, a POLICY's checkpoint), so a missing API key or SDK fails before
anything is played.

When a game raises (or on KeyboardInterrupt), the other running games stop
at their next tick (one more model call at most; their rows are not
delivered), games not started are cancelled, the workers are joined, the
rows of games that still finished are delivered, and the error is re-raised:
nothing keeps playing (and paying) in the background, and a resume replays
only what was not recorded.
"""

from __future__ import annotations

import concurrent.futures
import os
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from catanatron.models.actions import generate_playable_actions
from catanatron.models.player import Color

from catanrl import canon
from catanrl.eval.harness import _row, _scenarios

#: Seat k of a game is this color (Catanatron's base colors, then the extension's).
SEAT_COLORS = (Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE, Color.GREEN, Color.BROWN)
#: Set to a directory to save every LLM exchange: <dir>/<game_id>.s<slot>.jsonl.
TRANSCRIPTS_ENV = "CATANRL_LLM_TRANSCRIPTS"


def reseat(game: Any, players: Sequence[Any]) -> None:
    """Make ``players`` the turn order of a fresh Catanatron game (before any action)."""
    state = game.state
    if state.action_records:
        raise RuntimeError("seats can only be set before the first action")
    state.players = list(players)
    state.colors = tuple(p.color for p in players)
    state.color_to_index = {color: i for i, color in enumerate(state.colors)}
    state.current_player_index = 0
    state.current_turn_index = 0
    game.playable_actions = generate_playable_actions(state)


class GameStopped(RuntimeError):
    """A game abandoned at a tick because the harness is stopping; it has no row."""


def play_game(scheduled: Any, scenario: Any, agents: Sequence[Any], *, transcripts: Path | None = None,
              stop: threading.Event | None = None) -> dict:
    """One scheduled game with ``agents`` (resolved, slot order; the fill last); its row.
    ``stop`` set between ticks abandons the game (:class:`GameStopped`)."""
    n = scenario.game.num_players
    if len(scheduled.seats) != n:
        raise ValueError(f"{scheduled.game_id}: seats do not match its scenario's players")
    players = [agents[int(slot)].player(SEAT_COLORS[seat]) for seat, slot in enumerate(scheduled.seats)]
    for player in players:
        if hasattr(player, "keep_log"):
            player.keep_log = transcripts is not None
    game = scenario.game.catanatron_game(players, map_seed=int(scheduled.map_seed), seed=int(scheduled.game_seed))
    reseat(game, players)
    decisions = [0] * n
    while game.winning_color() is None and game.state.num_turns < scenario.turn_limit:
        if stop is not None and stop.is_set():
            raise GameStopped(f"{scheduled.game_id} was stopped before it finished")
        if len(game.playable_actions) > 1:
            decisions[game.state.current_player_index] += 1
        game.play_tick()
    for observer in game.observers:
        observer.after(game)
    winner = game.winning_color()
    winner_seat = None if winner is None else list(game.state.colors).index(winner)
    row = _row(scheduled, num_slots=len(agents), winner_seat=winner_seat, turns=int(game.state.num_turns),
               truncated=winner is None, stuck=False, decisions=decisions)
    stats: list[dict | None] = [None] * len(agents)
    for seat, player in enumerate(players):
        if hasattr(player, "stats") and player.KEY == "LLM":
            stats[int(scheduled.seats[seat])] = dict(player.stats)
            if transcripts is not None and player.log:
                path = transcripts / f"{scheduled.game_id}.s{int(scheduled.seats[seat])}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                for entry in player.log:
                    canon.append_jsonl(path, entry)
    if any(s is not None for s in stats):
        row["llm"] = stats
    return row


def preflight(agents: Sequence[Any]) -> None:
    """Build each agent's player once and let it load what it needs (``player.preflight()``:
    the LLM's model client, a POLICY's checkpoint): keys, SDKs and refs fail here, not mid-eval."""
    for agent in agents:
        player = agent.player(Color.RED)
        if hasattr(player, "preflight"):
            player.preflight()


def play_catanatron(
    lineup: Sequence[Any],
    games: Sequence[Any],
    *,
    fill: Any = None,
    suite: Any = None,
    scenarios: Sequence[Any] | None = None,
    turn_limit: int | None = None,
    threads: int = 1,
    on_game: Callable[[dict], None],
    transcripts: Path | None = None,
) -> None:
    """Play ``games`` (ScheduledGame) with ``lineup`` (resolved agents for the catanatron
    harness, slot order; ``fill`` plays slot ``len(lineup)``), ``threads`` at a time;
    ``on_game(row)`` per finished game, from the calling thread. ``transcripts`` (a
    directory; default from ``$CATANRL_LLM_TRANSCRIPTS``) saves every LLM exchange."""
    table = _scenarios(suite, scenarios, turn_limit)
    unfair = sorted({f"{s.game.name} ({s.game.dice_mode} dice)" for s in table if s.game.dice_mode != "random"})
    if unfair:
        raise ValueError(f"the catanatron harness rolls two fair dice, not {unfair}: play them in the native harness")
    agents = [*lineup, *([fill] if fill is not None else [])]
    for scheduled in games:
        if max(scheduled.seats) >= len(agents):
            raise ValueError(f"{scheduled.game_id}: a seat's slot has no agent (fill missing?)")
    if transcripts is None and os.environ.get(TRANSCRIPTS_ENV):
        transcripts = Path(os.environ[TRANSCRIPTS_ENV])
    preflight(agents)
    threads = max(1, int(threads))
    if threads == 1 or len(games) <= 1:
        for scheduled in games:
            on_game(play_game(scheduled, table[scheduled.scenario], agents, transcripts=transcripts))
        return
    stop = threading.Event()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=threads)
    futures = [pool.submit(play_game, g, table[g.scenario], agents, transcripts=transcripts, stop=stop) for g in games]
    delivered: set[concurrent.futures.Future] = set()
    try:
        for future in concurrent.futures.as_completed(futures):
            delivered.add(future)
            on_game(future.result())
    except BaseException:
        stop.set()  # running games stop at their next tick: no further model call
        for future in futures:
            future.cancel()  # games not started never start
        pool.shutdown(wait=True)
        for future in futures:  # games that still finished are kept: a resume replays only the rest
            if future not in delivered and not future.cancelled() and future.exception() is None:
                delivered.add(future)
                on_game(future.result())
        raise
    pool.shutdown(wait=True)
