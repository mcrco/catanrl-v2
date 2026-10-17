"""Native evaluation: play a schedule of exact games on cppanatron.

``play_schedule(lineup, games, ...)`` plays every ``ScheduledGame`` (map
seed, game seed, bot seed, lineup slot per seat) and calls ``on_game(row)``
once per finished game. Rows depend only on the schedule and the agents, not
on ``num_envs``, ``threads`` or the order games finish in:

* every game is built from its own seeds and played alone;
* decision seeds are ``derive_seed(game_seed, "decide", rotation, seat, ply)``
  with ``ply`` the seat's own decision count;
* networks evaluate in strict mode: every forward is padded to
  ``infer_rows`` rows, so a row's outputs do not depend on its batch;
* the dims (and so every network's shapes) come from the scenarios'
  generator bounds, not from the games being played.

Games run on cppanatron's VecEnv in schedule mode (engine E1): bots and
forced moves play inside the engine on its threads, each game's random bots
draw from its own stream (E2), and with an MCTS (``dets``) or BELIEF agent in
the lineup the VecEnv keeps a hand tracker per learner seat, fed every
transition (E6), which the agents read through ``tracker(env, seat)``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cppanatron as cpp
import numpy as np

from catanrl import games as game_configs
from catanrl.agents.native import BotAgent, NativeAgent, make_native_agent
from catanrl.envs.encoding import dims_for, encoder_dims, info_level
from catanrl.seeding import derive_seed as _derive_seed

# ---------------------------------------------------------------- schedule inputs


@dataclass
class _Scenario:
    game: game_configs.GameConfig
    turn_limit: int


def _scenarios(suite: Any, scenarios: Sequence[Any] | None, turn_limit: int | None) -> list[_Scenario]:
    if suite is not None:
        scenarios = scenarios if scenarios is not None else suite.scenarios
        turn_limit = turn_limit if turn_limit is not None else suite.turn_limit
    if scenarios is None or turn_limit is None:
        raise ValueError("play_schedule needs the suite (or its scenarios and turn_limit)")
    out = []
    for entry in scenarios:
        if isinstance(entry, Mapping) and "game" in entry:
            limit = int(entry.get("turn_limit", turn_limit))
            entry = entry["game"]
        else:
            limit = int(turn_limit)
        game = entry if isinstance(entry, game_configs.GameConfig) else game_configs.load_game(entry)
        out.append(_Scenario(game, limit))
    return out


def _agents(lineup: Sequence[Any], fill: Any, *, device: str, infer_rows: int, mode: str) -> list[Any]:
    """One agent per lineup slot, then the fill agent (if any)."""
    built = []
    for entry in [*lineup, *([fill] if fill is not None else [])]:
        if isinstance(entry, (NativeAgent, BotAgent)):
            built.append(entry)
        else:
            built.append(make_native_agent(entry, device=device, infer_rows=infer_rows, mode=mode))
    return built


def _row(scheduled: Any, *, num_slots: int, winner_seat: int | None, turns: int, truncated: bool,
         stuck: bool, decisions: Sequence[int]) -> dict[str, Any]:
    seats = [int(s) for s in scheduled.seats]
    n = len(seats)
    scores = [0.0] * num_slots
    if winner_seat is not None:
        scores[seats[winner_seat]] += 1.0
    elif truncated:
        for slot in seats:
            scores[slot] += 1.0 / n
    return {
        "game_id": scheduled.game_id,
        "seats": seats,
        "winner": None if winner_seat is None else seats[winner_seat],
        "winner_seat": winner_seat,
        "turns": int(turns),
        "truncated": bool(truncated),
        "stuck": bool(stuck),
        "scores": scores,
        "forfeits": [0] * num_slots,
        "decisions": [int(d) for d in decisions],
    }


def settlement_capacity(catan_map: cpp.CatanMap) -> int:
    """Settlements the distance rule lets a map hold: the engine's greedy count
    (place on the open node with the fewest open neighbours), which VecEnv uses
    to refuse maps that cannot seat every player's two initial settlements."""
    nodes = sorted(catan_map.land_nodes)
    open_nodes = set(nodes)
    neighbors = {node: list(catan_map.neighbors(node)) for node in nodes}
    placed = 0
    while True:
        best, fewest = None, 0
        for node in nodes:
            if node in open_nodes:
                count = sum(n in open_nodes for n in neighbors[node])
                if best is None or count < fewest:
                    best, fewest = node, count
        if best is None:
            return placed
        placed += 1
        open_nodes.discard(best)
        open_nodes.difference_update(neighbors[best])


def _check_games(games, scenarios: list[_Scenario], maps: _Maps, dims: Mapping[str, int]) -> None:
    """Refuse, before playing anything, a game the VecEnv would refuse mid-eval."""
    encoder = encoder_dims(dims)
    for scheduled in games:
        scenario = scenarios[scheduled.scenario]
        n = scenario.game.num_players
        if len(scheduled.seats) != n:
            raise ValueError(f"{scheduled.game_id}: seats do not match its scenario's players")
        catan_map = maps.get(scheduled.scenario, scheduled.map_seed)
        if not encoder.fits(catan_map, n):
            raise ValueError(f"{scheduled.game_id}: its map does not fit the suite's dims {dict(dims)}")
        if settlement_capacity(catan_map) < 2 * n:
            raise ValueError(f"{scheduled.game_id}: its map cannot seat {n} players' initial settlements")


class _Maps:
    def __init__(self, scenarios: list[_Scenario]) -> None:
        self.scenarios = scenarios
        self.cache: dict[tuple[int, int], cpp.CatanMap] = {}

    def get(self, scenario: int, map_seed: int) -> cpp.CatanMap:
        key = (int(scenario), int(map_seed))
        if key not in self.cache:
            spec = self.scenarios[scenario].game.map_spec(int(map_seed))
            self.cache[key] = cpp.CatanMap.from_spec(cpp.MapSpec.from_dict(spec))
        return self.cache[key]

    def built(self) -> list[cpp.CatanMap]:
        """Every map built so far (all of a schedule's, after ``_check_games``)."""
        return list(self.cache.values())


# ---------------------------------------------------------------- VecEnv schedule mode


def _play_vecenv(agents, games, scenarios, maps, dims, *, num_envs, threads, on_game) -> None:
    networks = [a for a in agents if isinstance(a, NativeAgent)]
    level = networks[0].level if networks else "private"
    enemies = {a.value_enemies for a in agents if isinstance(a, BotAgent) and a.key == "F"}
    if len(enemies) > 1:
        raise ValueError("every F of a lineup must count the same enemies")
    value_enemies = enemies.pop() if enemies else cpp.ValueEnemies.NEXT
    # Schedule mode passes every game's map; a scenario's own map only has to fit the dims
    # (without one the engine checks a BASE board, which small-map suites do not cover).
    first_seed = {}
    for scheduled in games:
        first_seed.setdefault(int(scheduled.scenario), int(scheduled.map_seed))
    cpp_scenarios = [
        cpp.Scenario(num_players=s.game.num_players, rules=s.game.cppanatron_rules(),
                     maps=[maps.get(i, first_seed.get(i, 0))], turn_limit=s.turn_limit,
                     value_enemies=value_enemies)
        for i, s in enumerate(scenarios)
    ]
    num_envs = min(int(num_envs), max(1, len(games)))
    trackers = any(agent.needs_trackers for agent in agents)
    # Exact trackers can deal the pool too, and every agent asks for its own deal.
    exact = any(agent.needs_trackers and agent.tracker_mode == cpp.TrackerMode.EXACT for agent in agents)
    vec = cpp.VecEnv(cpp_scenarios, num_envs=num_envs, dims=encoder_dims(dims), level=info_level(level),
                     seed=0, num_threads=int(threads), auto_reset=False, trackers=trackers,
                     tracker_mode=cpp.TrackerMode.EXACT if exact else cpp.TrackerMode.POOL)
    buffers = vec.allocate()
    vec.reset()
    size = int(vec.action_size)
    queue = list(games)
    queue.reverse()
    idle = list(range(num_envs - 1, -1, -1))
    running: dict[int, tuple[Any, list[int]]] = {}
    num_slots = len(agents)

    def seat_kind(agent) -> str:
        return agent.key if isinstance(agent, BotAgent) else "learner"

    def collect() -> None:
        for done in vec.pop_finished():
            env = int(done["env"])
            scheduled, _decisions = running.pop(env)
            winner = int(done["winner"])
            on_game(_row(scheduled, num_slots=num_slots, winner_seat=None if winner < 0 else winner,
                         turns=int(done["turns"]), truncated=bool(done["truncated"]),
                         stuck=bool(done["stuck"]), decisions=[int(d) for d in done["decisions"]]))
            idle.append(env)

    while True:
        starts = []
        while queue and idle:
            env, scheduled = idle.pop(), queue.pop()
            running[env] = (scheduled, [0] * len(scheduled.seats))
            starts.append({
                "env": env, "scenario": int(scheduled.scenario),
                "map": maps.get(scheduled.scenario, scheduled.map_seed),
                "game_seed": int(scheduled.game_seed), "bot_seed": int(scheduled.bot_seed),
                "seats": [seat_kind(agents[int(slot)]) for slot in scheduled.seats],
                "lineup_slots": [int(slot) for slot in scheduled.seats],
            })
        if starts:
            vec.start_games(starts)
        collect()
        acting = np.flatnonzero(buffers["acting"])
        if len(acting) == 0:
            if not queue and not running:
                break
            if not starts:
                raise RuntimeError("VecEnv schedule mode stalled with games running")
            continue
        slots = buffers["lineup_slot"][acting]
        for slot in np.unique(slots):
            rows_idx = acting[slots == slot]
            agent = agents[int(slot)]
            envs = buffers["env_index"][rows_idx]
            seats = buffers["seat"][rows_idx]
            need_games = agent.needs_games or agent.level != level
            games_list = [vec.game(int(e)) for e in envs] if need_games else None
            if agent.level != level:
                rows = np.stack([agent.encode(g, int(s)) for g, s in zip(games_list, seats)])
            else:
                rows = buffers["observations"][rows_idx]
            seeds = []
            for e, s in zip(envs, seats):
                scheduled, decisions = running[int(e)]
                seeds.append(_derive_seed(int(scheduled.game_seed), "decide", int(scheduled.rotation), int(s),
                                          decisions[int(s)]))
                decisions[int(s)] += 1
            seat_trackers = ([vec.tracker(int(e), int(s)) for e, s in zip(envs, seats)]
                             if agent.needs_trackers else None)
            choices = agent.act(rows, rows[:, -size:], np.array(seeds, dtype=np.uint64),
                                games=games_list if agent.needs_games else None, trackers=seat_trackers)
            buffers["actions"][rows_idx] = np.asarray(choices, dtype=np.int32)
        vec.step()
        collect()


# ---------------------------------------------------------------- entry point


def play_schedule(
    lineup: Sequence[Any],
    games: Sequence[Any],
    *,
    fill: Any = None,
    device: str = "auto",
    num_envs: int = 256,
    threads: int = 1,
    infer_rows: int = 256,
    on_game: Callable[[dict], None],
    suite: Any = None,
    scenarios: Sequence[Any] | None = None,
    turn_limit: int | None = None,
) -> None:
    """Play ``games`` (ScheduledGame) with ``lineup`` (resolved agents, slot order).

    ``fill`` plays the seats whose slot is ``len(lineup)``. ``suite`` (or
    ``scenarios`` and ``turn_limit``) gives each ``ScheduledGame.scenario``
    its game config; a suite's ``determinism`` (``strict`` by default, ``fast``
    for dev suites) sets how networks batch. ``on_game(row)`` gets ``game_id``, ``seats`` (slot per
    seat), ``winner`` (slot or None), ``winner_seat``, ``turns``,
    ``truncated``, ``stuck``, ``scores`` (per slot: 1 for a win, 1/N per
    seat for a truncation), ``forfeits`` (per slot) and ``decisions`` (per
    seat).
    """
    if int(num_envs) < 1 or int(threads) < 1:
        raise ValueError(f"num_envs and threads must be positive, not {num_envs} and {threads}")
    table = _scenarios(suite, scenarios, turn_limit)
    mode = str(getattr(suite, "determinism", "strict"))
    agents = _agents(lineup, fill, device=device, infer_rows=infer_rows, mode=mode)
    dims = dims_for([s.game for s in table])
    for agent in agents:
        agent.bind(dims)
    for scheduled in games:
        if max(scheduled.seats) >= len(agents):
            raise ValueError(f"{scheduled.game_id}: a seat's slot has no agent (fill missing?)")
    maps = _Maps(table)
    _check_games(games, table, maps, dims)
    for agent in agents:  # a network tied to one board (cnn) refuses maps laid out otherwise
        agent.check_maps(maps.built())
    _play_vecenv(agents, list(games), table, maps, dims, num_envs=num_envs, threads=threads, on_game=on_game)
