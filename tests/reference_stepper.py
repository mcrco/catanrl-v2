"""A reference for the eval harness: scheduled games stepped one action at a
time in Python, independently of VecEnv's schedule mode.

It plays F with the engine's ``value_action``, R with a bit-exact copy of the
engine's bot stream (std::mt19937_64 seeded by the game's bot seed and
cppanatron's ``random::below``, Lemire's bounded draw), forced moves without
asking, and feeds hand trackers with ``HandTracker.observe`` on every
transition. Tests check that ``play_schedule`` gives exactly its rows.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import cppanatron as cpp
import numpy as np

from catanrl.agents.native import BotAgent
from catanrl.envs.encoding import dims_for
from catanrl.eval import harness
from catanrl.seeding import derive_seed

MASK64 = (1 << 64) - 1


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


@dataclass
class _Table:
    scheduled: Any
    turn_limit: int
    game: cpp.Game
    agents: list[Any]
    rng: MT19937_64
    index: int = 0
    trackers: dict[int, cpp.HandTracker] = field(default_factory=dict)
    decisions: list[int] = field(default_factory=list)
    stuck: bool = False

    def execute(self, action) -> None:
        before = self.game.copy() if self.trackers else None
        self.game.execute(action)
        for tracker in self.trackers.values():
            tracker.observe(before, action, self.game)

    def advance(self) -> bool:
        game = self.game
        while True:
            if game.winning_color is not None or game.num_turns >= self.turn_limit:
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


def play_reference(lineup, games, *, suite, fill=None) -> list[dict]:
    """The rows play_schedule must give, sorted by game_id."""
    table = harness._scenarios(suite, None, None)
    agents = [*lineup, *([fill] if fill is not None else [])]
    dims = dims_for([s.game for s in table])
    for agent in agents:
        agent.bind(dims)
    maps = harness._Maps(table)
    rows, active = [], []

    def finish(t: _Table) -> None:
        game = t.game
        winner = game.winning_player_index if game.winning_color is not None else None
        rows.append(harness._row(t.scheduled, num_slots=len(agents), winner_seat=winner, turns=game.num_turns,
                                 truncated=winner is None, stuck=t.stuck, decisions=t.decisions))

    for index, scheduled in enumerate(games):
        scenario = table[scheduled.scenario]
        game = cpp.Game(scenario.game.num_players, map=maps.get(scheduled.scenario, scheduled.map_seed),
                        rules=scenario.game.cppanatron_rules(), seed=int(scheduled.game_seed))
        seat_agents = [agents[int(slot)] for slot in scheduled.seats]
        t = _Table(scheduled, scenario.turn_limit, game, seat_agents, MT19937_64(int(scheduled.bot_seed)),
                   index=index, decisions=[0] * len(seat_agents))
        for seat, agent in enumerate(seat_agents):
            if agent.needs_trackers:
                t.trackers[seat] = cpp.HandTracker(game, seat)
        (active.append if t.advance() else finish)(t)
    while active:
        groups: dict[int, list[_Table]] = defaultdict(list)
        for t in active:
            groups[id(t.agents[t.game.current_player_index])].append(t)
        still = []
        for tables in groups.values():
            agent = tables[0].agents[tables[0].game.current_player_index]
            seats = [t.game.current_player_index for t in tables]
            obs = np.stack([agent.encode(t.game, s) for t, s in zip(tables, seats)])
            seeds = np.array([derive_seed(int(t.scheduled.game_seed), "decide", int(t.scheduled.rotation), s,
                                          t.decisions[s]) for t, s in zip(tables, seats)], dtype=np.uint64)
            memory = {}
            if getattr(agent, "needs_memory", False):  # one stream per seat of a game
                memory = {"streams": np.array([t.index * dims["max_players"] + s for t, s in zip(tables, seats)]),
                          "first": np.array([t.decisions[s] == 0 for t, s in zip(tables, seats)])}
            choices = agent.act(obs, obs[:, -agent.action_space.size:], seeds,
                                games=[t.game for t in tables] if agent.needs_games else None,
                                trackers=[t.trackers[s] for t, s in zip(tables, seats)]
                                if agent.needs_trackers else None, **memory)
            for t, seat, index in zip(tables, seats, choices):
                t.decisions[seat] += 1
                t.execute(agent.action_space.decode(t.game, int(index)))
                (still.append if t.advance() else finish)(t)
        active = still
    return sorted(rows, key=lambda row: row["game_id"])
