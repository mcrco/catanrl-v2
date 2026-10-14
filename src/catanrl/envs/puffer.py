"""PufferLib environment over ``cppanatron.VecEnv``.

One :class:`CatanPufferEnv` holds the whole VecEnv: its ``num_agents`` is the
VecEnv's rows (``env * max_players + seat``) and PufferLib's buffers are bound
to the VecEnv zero-copy, so ``step`` copies the actions in, releases the GIL
in C++ and returns. Parallelism comes from the VecEnv's threads. It runs
through ``pufferlib.vector.Serial`` with one env (:func:`make_vecenv`).

Scenarios come from the experiment config's ``env.scenarios`` (see
:func:`scenario_specs`); every map is pre-generated into a seeded pool
(:func:`build_scenarios`), which is recorded and hashed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import cppanatron as cpp
import gymnasium
import numpy as np
import pufferlib
import pufferlib.vector

from catanrl import games
from catanrl.envs.encoding import (
    check_dims,
    critic_level_of,
    dims_for,
    encoder_dims,
    info_level,
    row_layout,
)

SCENARIO_KEYS = {"game", "weight", "seats", "shuffle_seats", "turn_limit", "maps", "value_enemies", "name"}
DEFAULT_POOL = 1024
DEFAULT_TURN_LIMIT = 500


@dataclass
class ScenarioSpec:
    """One ``env.scenarios`` entry, resolved."""

    name: str
    game: games.GameConfig
    weight: float = 1.0
    #: One seat kind per seat: "learner", "F" or "R"; default all learners.
    seats: tuple[str, ...] = ()
    shuffle_seats: bool = True
    turn_limit: int = DEFAULT_TURN_LIMIT
    #: Size of the pre-generated map pool.
    maps: int = DEFAULT_POOL
    value_enemies: str = "next"
    map_seeds: list[int] = field(default_factory=list)
    map_hashes: list[str] = field(default_factory=list)


def _derive_seed(root: int, *labels) -> int:
    try:
        from catanrl.seeding import derive_seed
    except ImportError:  # until catanrl.seeding lands: the v1 formula
        payload = "|".join([str(root), *(str(label) for label in labels)])
        return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (2**63 - 1)
    return derive_seed(root, *labels)


def scenario_specs(entries: Sequence[Mapping[str, Any] | str]) -> list[ScenarioSpec]:
    """Read ``env.scenarios``: game name/path/dict, weight, seats, map pool size, ..."""
    specs = []
    names: dict[str, int] = {}
    for entry in entries:
        if isinstance(entry, str):
            entry = {"game": entry}
        unknown = set(entry) - SCENARIO_KEYS
        if unknown:
            raise ValueError(f"unknown scenario keys {sorted(unknown)}; expected {sorted(SCENARIO_KEYS)}")
        game = games.load_game(entry["game"])
        seats = tuple(str(s) for s in entry.get("seats") or ())
        if seats and len(seats) != game.num_players:
            raise ValueError(f"{game.name}: {len(seats)} seats for {game.num_players} players")
        if seats and "learner" not in seats:
            raise ValueError(f"{game.name}: a training scenario needs a learner seat")
        name = str(entry.get("name") or game.name)
        names[name] = names.get(name, 0) + 1
        if names[name] > 1:
            name = f"{name}.{names[name] - 1}"
        specs.append(
            ScenarioSpec(
                name=name,
                game=game,
                weight=float(entry.get("weight", 1.0)),
                seats=seats,
                shuffle_seats=bool(entry.get("shuffle_seats", True)),
                turn_limit=int(entry.get("turn_limit", DEFAULT_TURN_LIMIT)),
                maps=int(entry.get("maps", DEFAULT_POOL)),
                value_enemies=str(entry.get("value_enemies", "next")),
            )
        )
    if not specs:
        raise ValueError("env.scenarios is empty")
    return specs


def map_hash(spec: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def build_scenarios(specs: Sequence[ScenarioSpec], pool_seed: int) -> tuple[list[cpp.Scenario], dict[str, Any]]:
    """cppanatron scenarios with pre-generated map pools, and the pool's record.

    Map ``j`` of scenario ``i`` has seed ``derive_seed(pool_seed, i, j)``.
    The record lists every map seed and hash and the pool's digest.
    """
    scenarios = []
    record: dict[str, Any] = {"seed": int(pool_seed), "scenarios": []}
    for i, spec in enumerate(specs):
        seeds = [_derive_seed(pool_seed, i, j) for j in range(spec.maps)]
        specs_json = [spec.game.map_spec(seed) for seed in seeds]
        spec.map_seeds = seeds
        spec.map_hashes = [map_hash(s) for s in specs_json]
        catan_maps = [cpp.CatanMap.from_spec(cpp.MapSpec.from_dict(s)) for s in specs_json]
        enemies = {"next": cpp.ValueEnemies.NEXT, "all": cpp.ValueEnemies.ALL}[spec.value_enemies]
        scenarios.append(
            cpp.Scenario(
                num_players=spec.game.num_players,
                rules=spec.game.cppanatron_rules(),
                maps=catan_maps,
                seats=list(spec.seats) or None,
                shuffle_seats=spec.shuffle_seats,
                turn_limit=spec.turn_limit,
                weight=spec.weight,
                value_enemies=enemies,
            )
        )
        record["scenarios"].append(
            {"name": spec.name, "game": spec.game.to_dict(), "weight": spec.weight,
             "seats": list(spec.seats), "maps": [[s, h] for s, h in zip(seeds, spec.map_hashes)]}
        )
    record["hash"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:16]
    return scenarios, record


class CatanPufferEnv(pufferlib.PufferEnv):
    """The VecEnv's rows as PufferLib agents.

    Buffers: observations float32 [rows, obs], actions int32, rewards float32,
    terminals/truncations bool (PufferLib's) are bound zero-copy; the VecEnv's
    acting/learner/seat/env_index/num_players arrays are this env's own. Every
    ``log_interval`` steps the episodes finished since the last pop are
    reported as one info dict (nothing when none finished).
    """

    def __init__(
        self,
        scenarios: Sequence[cpp.Scenario],
        *,
        names: Sequence[str],
        num_envs: int,
        dims: Mapping[str, int],
        level: str = "private",
        critic_level: str | None = None,
        threads: int = 1,
        vp_reward: float = 0.0,
        log_interval: int = 32,
        buf=None,
        seed: int | None = 0,
    ) -> None:
        dims = check_dims(dims)
        kwargs: dict[str, Any] = {}
        if critic_level_of(level, critic_level) is not None:
            kwargs["critic_level"] = info_level(critic_level)  # E5
        self.vec = cpp.VecEnv(
            list(scenarios), num_envs=num_envs, dims=encoder_dims(dims), level=info_level(level),
            seed=0 if seed is None else int(seed), num_threads=threads, vp_reward=vp_reward, **kwargs,
        )
        if hasattr(self.vec, "layout"):
            expected = row_layout(dims, level, critic_level_of(level, critic_level))
            actual = {name: (int(off), tuple(int(x) for x in shape))
                      for name, (off, shape) in self.vec.layout().items()}
            if actual != expected:
                raise RuntimeError("VecEnv rows do not match catanrl's row layout")
        self.dims = dims
        self.max_players = dims["max_players"]
        self.num_games = num_envs
        self.num_agents = int(self.vec.num_rows)
        self.scenario_names = list(names)
        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(int(self.vec.observation_size),), dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Discrete(int(self.vec.action_size))
        self.log_interval = int(log_interval)
        self.render_mode = None
        super().__init__(buf)
        rows = self.num_agents
        self.acting = np.zeros(rows, dtype=np.uint8)
        self.learner = np.zeros(rows, dtype=np.uint8)
        self.seat = np.zeros(rows, dtype=np.int32)
        self.env_index = np.zeros(rows, dtype=np.int32)
        self.num_players = np.zeros(num_envs, dtype=np.int32)
        self.vec.bind(
            observations=self.observations, actions=self.actions, rewards=self.rewards,
            terminals=self.terminals, truncations=self.truncations, acting=self.acting,
            learner=self.learner, seat=self.seat, env_index=self.env_index, num_players=self.num_players,
        )
        self.tick = 0
        self.decisions = 0

    def reset(self, seed=None):
        self.vec.reset(None if seed is None else int(seed))
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        if actions is not self.actions:
            self.actions[:] = actions
        self.decisions += int(self.acting.sum())
        self.vec.step()
        self.tick += 1
        info = []
        if self.tick % self.log_interval == 0:
            stats = self.episode_stats()
            if stats:
                info.append(stats)
        return self.observations, self.rewards, self.terminals, self.truncations, info

    def episode_stats(self) -> dict[str, float]:
        """Flat episode means since the last pop, overall and per scenario."""
        stats = self.vec.pop_stats()
        if not stats["episodes"]:
            return {}
        out = _bucket(stats, "")
        for name, bucket in zip(self.scenario_names, stats["scenarios"]):
            if bucket["episodes"]:
                out.update(_bucket(bucket, f"{name}/"))
        return out

    def close(self):
        pass


def _bucket(bucket: Mapping[str, Any], prefix: str) -> dict[str, float]:
    out = {f"{prefix}{key}": float(bucket[key])
           for key in ("episodes", "truncated", "length", "decisions", "learner_win", "stuck")}
    for seat, rate in enumerate(bucket["seat_win"]):
        out[f"{prefix}seat_win_{seat}"] = float(rate)
    return out


@dataclass
class EnvPlan:
    """Everything a training env needs, built once per launch."""

    specs: list[ScenarioSpec]
    scenarios: list[cpp.Scenario]
    dims: dict[str, int]
    pool: dict[str, Any]


def plan_env(env_cfg: Mapping[str, Any], *, pool_seed: int) -> EnvPlan:
    """Scenarios, map pools and dims from an ``env`` config section."""
    specs = scenario_specs(env_cfg["scenarios"])
    dims_cfg = env_cfg.get("dims", "auto")
    if dims_cfg in (None, "auto"):
        dims = dims_for([s.game for s in specs])
    else:
        dims = check_dims(dims_cfg)
    scenarios, pool = build_scenarios(specs, pool_seed)
    return EnvPlan(specs=specs, scenarios=scenarios, dims=dims, pool=pool)


def make_vecenv(plan: EnvPlan, *, num_envs: int, level: str, critic_level: str | None, threads: int,
                seed: int, vp_reward: float = 0.0, log_interval: int = 32):
    """A ``pufferlib.vector.Serial`` over one :class:`CatanPufferEnv`."""
    kwargs = {
        "scenarios": plan.scenarios, "names": [s.name for s in plan.specs], "num_envs": num_envs, "dims": plan.dims,
        "level": level, "critic_level": critic_level, "threads": threads, "vp_reward": vp_reward,
        "log_interval": log_interval,
    }
    return pufferlib.vector.make(
        CatanPufferEnv, env_kwargs=kwargs, num_envs=1, backend=pufferlib.vector.Serial, seed=seed
    )
