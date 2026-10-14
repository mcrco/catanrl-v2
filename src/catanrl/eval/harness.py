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

Two backends play the games:

* ``vecenv``: cppanatron's VecEnv in schedule mode (engine E1); bots and
  forced moves run inside the engine on its threads. Used unless an agent
  needs hand trackers.
* ``stepper``: the games stepped one action at a time in Python, so every
  transition can feed the hand trackers of MCTS (``dets``) and BELIEF agents.
  It plays F with the engine's ``value_action`` and R with the engine's bot
  stream (mt19937_64 seeded by the bot seed, Lemire's bounded draw), so both
  backends play the same games.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import cppanatron as cpp
import numpy as np

from catanrl import games as game_configs
from catanrl.agents.native import BotAgent, NativeAgent, make_native_agent
from catanrl.envs.encoding import dims_for, encoder_dims, info_level

MASK64 = (1 << 64) - 1


def _derive_seed(root: int, *labels) -> int:
    try:
        from catanrl.seeding import derive_seed
    except ImportError:
        from catanrl.envs.puffer import _derive_seed as derive_seed
    return derive_seed(root, *labels)


# ---------------------------------------------------------------- engine bot stream


class MT19937_64:
    """std::mt19937_64, bit for bit."""

    def __init__(self, seed: int) -> None:
        mt = [0] * 312
        mt[0] = seed & MASK64
        for i in range(1, 312):
            mt[i] = (6364136223846793005 * (mt[i - 1] ^ (mt[i - 1] >> 62)) + i) & MASK64
        self.mt = mt
        self.index = 312

    def _twist(self) -> None:
        mt = self.mt
        for i in range(312):
            x = (mt[i] & 0xFFFFFFFF80000000) | (mt[(i + 1) % 312] & 0x7FFFFFFF)
            shifted = x >> 1
            if x & 1:
                shifted ^= 0xB5026F5AA96619E9
            mt[i] = mt[(i + 156) % 312] ^ shifted
        self.index = 0

    def __call__(self) -> int:
        if self.index >= 312:
            self._twist()
        y = self.mt[self.index]
        self.index += 1
        y ^= (y >> 29) & 0x5555555555555555
        y ^= (y << 17) & 0x71D67FFFEDA60000
        y ^= (y << 37) & 0xFFF7EEE000000000
        y ^= y >> 43
        return y & MASK64


def below(rng: MT19937_64, bound: int) -> int:
    """cppanatron's ``random::below``: Lemire's unbiased draw in [0, bound)."""
    product = rng() * bound
    low = product & MASK64
    if low < bound:
        threshold = ((1 << 64) - bound) % bound
        while low < threshold:
            product = rng() * bound
            low = product & MASK64
    return product >> 64


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


# ---------------------------------------------------------------- the stepper


@dataclass
class _Table:
    scheduled: Any
    scenario: _Scenario
    game: cpp.Game
    agents: list[Any]                 # per seat
    rng: MT19937_64
    trackers: dict[int, cpp.HandTracker] = field(default_factory=dict)
    decisions: list[int] = field(default_factory=list)
    stuck: bool = False

    def execute(self, action: cpp.Action) -> None:
        if self.trackers:
            before = self.game.copy()
            self.game.execute(action)
            for tracker in self.trackers.values():
                tracker.observe(before, action, self.game)
        else:
            self.game.execute(action)

    def advance(self) -> bool:
        """Play bots and forced moves; False once the game is over."""
        game = self.game
        while True:
            if game.winning_color is not None or game.num_turns >= self.scenario.turn_limit:
                return False
            actions = game.playable_actions
            if not actions:
                self.stuck = True
                return False
            agent = self.agents[game.current_player_index]
            if isinstance(agent, BotAgent):
                if agent.key == "R":
                    self.execute(actions[below(self.rng, len(actions))])
                else:
                    self.execute(cpp.value_action(game, enemies=agent.value_enemies))
            elif len(actions) == 1:
                self.execute(actions[0])
            else:
                return True


def _play_stepper(agents, games, scenarios, maps, *, num_envs, on_game) -> None:
    num_slots = len(agents)
    queue = list(games)
    queue.reverse()
    active: list[_Table] = []

    def start(scheduled) -> None:
        scenario = scenarios[scheduled.scenario]
        game = cpp.Game(scenario.game.num_players, map=maps.get(scheduled.scenario, scheduled.map_seed),
                        rules=scenario.game.cppanatron_rules(), seed=int(scheduled.game_seed))
        seat_agents = [agents[int(slot)] for slot in scheduled.seats]
        table = _Table(scheduled, scenario, game, seat_agents, MT19937_64(int(scheduled.bot_seed)),
                       decisions=[0] * len(seat_agents))
        for seat, agent in enumerate(seat_agents):
            if agent.needs_trackers:
                table.trackers[seat] = cpp.HandTracker(game, seat)
        if table.advance():
            active.append(table)
        else:
            finish(table)

    def finish(table: _Table) -> None:
        game = table.game
        winner = game.winning_player_index if game.winning_color is not None else None
        on_game(_row(table.scheduled, num_slots=num_slots, winner_seat=winner, turns=game.num_turns,
                     truncated=winner is None, stuck=table.stuck, decisions=table.decisions))

    while queue or active:
        while queue and len(active) < num_envs:
            start(queue.pop())
        by_agent: dict[int, list[_Table]] = defaultdict(list)
        for table in active:
            agent = table.agents[table.game.current_player_index]
            by_agent[id(agent)].append(table)
        still: list[_Table] = []
        for tables in by_agent.values():
            agent = tables[0].agents[tables[0].game.current_player_index]
            seats = [t.game.current_player_index for t in tables]
            rows = np.stack([agent.encode(t.game, s) for t, s in zip(tables, seats)])
            masks = rows[:, -agent.action_space.size:]
            seeds = np.array([
                _derive_seed(int(t.scheduled.game_seed), "decide", int(t.scheduled.rotation), s, t.decisions[s])
                for t, s in zip(tables, seats)], dtype=np.uint64)
            games_arg = [t.game for t in tables] if agent.needs_games else None
            trackers = [t.trackers[s] for t, s in zip(tables, seats)] if agent.needs_trackers else None
            choices = agent.act(rows, masks, seeds, games=games_arg, trackers=trackers)
            for table, seat, index in zip(tables, seats, choices):
                table.decisions[seat] += 1
                table.execute(agent.action_space.decode(table.game, int(index)))
                if table.advance():
                    still.append(table)
                else:
                    finish(table)
        active = still


# ---------------------------------------------------------------- VecEnv schedule mode


def _play_vecenv(agents, games, scenarios, maps, dims, *, num_envs, threads, on_game) -> None:
    if not hasattr(cpp.VecEnv, "start_games"):
        raise RuntimeError("this cppanatron has no VecEnv schedule mode (E1); rebuild the engine")
    networks = [a for a in agents if isinstance(a, NativeAgent)]
    level = networks[0].level if networks else "private"
    enemies = {a.value_enemies for a in agents if isinstance(a, BotAgent) and a.key == "F"}
    if len(enemies) > 1:
        raise ValueError("every F of a lineup must count the same enemies")
    value_enemies = enemies.pop() if enemies else cpp.ValueEnemies.NEXT
    cpp_scenarios = [
        cpp.Scenario(num_players=s.game.num_players, rules=s.game.cppanatron_rules(),
                     turn_limit=s.turn_limit, value_enemies=value_enemies)
        for s in scenarios
    ]
    num_envs = max(1, min(int(num_envs), len(games)))
    vec = cpp.VecEnv(cpp_scenarios, num_envs=num_envs, dims=encoder_dims(dims), level=info_level(level),
                     seed=0, num_threads=int(threads), auto_reset=False)
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
            choices = agent.act(rows, rows[:, -size:], np.array(seeds, dtype=np.uint64),
                                games=games_list if agent.needs_games else None)
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
    backend: str = "auto",
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
    table = _scenarios(suite, scenarios, turn_limit)
    mode = str(getattr(suite, "determinism", "strict"))
    agents = _agents(lineup, fill, device=device, infer_rows=infer_rows, mode=mode)
    dims = dims_for([s.game for s in table])
    for agent in agents:
        agent.bind(dims)
    for scheduled in games:
        if len(scheduled.seats) != table[scheduled.scenario].game.num_players:
            raise ValueError(f"{scheduled.game_id}: seats do not match its scenario's players")
        if max(scheduled.seats) >= len(agents):
            raise ValueError(f"{scheduled.game_id}: a seat's slot has no agent (fill missing?)")
    if backend == "auto":
        backend = "stepper" if any(a.needs_trackers for a in agents) else "vecenv"
    maps = _Maps(table)
    if backend == "stepper":
        _play_stepper(agents, list(games), table, maps, num_envs=int(num_envs), on_game=on_game)
    elif backend == "vecenv":
        if any(a.needs_trackers for a in agents):
            raise ValueError("hand trackers need the stepper backend")
        _play_vecenv(agents, list(games), table, maps, dims, num_envs=num_envs, threads=threads,
                     on_game=on_game)
    else:
        raise ValueError(f"backend must be auto, vecenv or stepper, not {backend!r}")
