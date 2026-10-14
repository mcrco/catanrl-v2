"""Eval suites: boards, not lineups (DESIGN 6.6).

A suite (``configs/suites/<name>.yaml``: ``purpose``, ``seed``,
``scenarios``, ``boards``, ``turn_limit``, optional ``determinism``) is
played by any lineup. ``Suite.schedule(n)`` is pure:

* board ``i`` of a scenario gets ``derive_seed(seed, purpose, scenario_hash, i)``
  (the scenario hash ignores the game's name), and its ``map`` and ``game``
  children are shared by every rotation;
* rotation ``r`` seats ``lineup[(seat + r) % N]`` (slots past the lineup are
  the ``fill`` slot), so every board is played from every seat;
* the quota is exact and a larger suite extends a smaller one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

import yaml

from catanrl.canon import digest
from catanrl.games import CONFIG_DIR, load_game
from catanrl.seeding import derive_seed

PURPOSES = ("select", "heldout", "dev")
SUITE_KEYS = {"name", "purpose", "seed", "boards", "scenarios", "turn_limit", "determinism"}


@dataclass(frozen=True)
class ScheduledGame:
    game_id: str
    scenario: int
    board: int
    rotation: int
    num_players: int
    map_seed: int
    game_seed: int
    bot_seed: int
    map_hash: str
    seats: tuple[int, ...]


def map_hash(spec: Mapping[str, Any]) -> str:
    return digest(spec)


@cache
def _map_hash(game_key: str, map_seed: int) -> str:
    import json

    return map_hash(load_game(json.loads(game_key)).map_spec(map_seed))


@dataclass(frozen=True)
class Suite:
    name: str
    purpose: Literal["select", "heldout", "dev"]
    seed: int
    boards: int
    scenarios: tuple[dict, ...]
    turn_limit: int
    determinism: Literal["strict", "fast"] = "strict"

    def __post_init__(self) -> None:
        if self.purpose not in PURPOSES:
            raise ValueError(f"suite {self.name}: purpose must be one of {PURPOSES}")
        if self.determinism not in ("strict", "fast"):
            raise ValueError(f"suite {self.name}: determinism must be strict or fast")
        if self.determinism == "fast" and self.purpose != "dev":
            raise ValueError(f"suite {self.name}: only dev suites may run fast")
        if self.boards < 1 or not self.scenarios:
            raise ValueError(f"suite {self.name} needs boards and scenarios")

    @classmethod
    def from_key(cls, name: str, key: Mapping[str, Any]) -> Suite:
        """The suite an eval key describes (its games as keyed, not as their YAML now reads)."""
        scenarios = tuple({"game": {**s["game"], "name": f"{name}-{i}"}, "turn_limit": s["turn_limit"]}
                          for i, s in enumerate(key["scenarios"]))
        return cls(name=name, purpose=key["purpose"], seed=key["seed"], boards=key["boards"], scenarios=scenarios,
                   turn_limit=key["turn_limit"], determinism=key["determinism"])

    def games(self) -> list[Any]:
        return [load_game(s["game"]) for s in self.scenarios]

    def _scenario(self, i: int) -> dict:
        """Scenario ``i`` without names: the resolved game and its turn limit."""
        entry = self.scenarios[i]
        game = load_game(entry["game"]).to_dict()
        game.pop("name")
        return {"game": game, "turn_limit": int(entry.get("turn_limit", self.turn_limit))}

    def scenario_hash(self, i: int) -> str:
        return digest(self._scenario(i))

    def schedule(self, lineup_size: int, *, fill: bool = False) -> list[ScheduledGame]:
        import json

        out = []
        for s, game in enumerate(self.games()):
            n = game.num_players
            if lineup_size < n and not fill:
                raise ValueError(f"suite {self.name}: {n}-player games need {n} agents or a fill agent")
            scenario_hash = self.scenario_hash(s)
            game_key = json.dumps(self._scenario(s)["game"], sort_keys=True)
            for i in range(self.boards):
                board = derive_seed(self.seed, self.purpose, scenario_hash, i)
                map_seed, game_seed = derive_seed(board, "map"), derive_seed(board, "game")
                mhash = _map_hash(game_key, map_seed)
                for r in range(n):
                    seats = tuple(min(lineup_size, (seat + r) % n) for seat in range(n))
                    out.append(ScheduledGame(game_id=f"s{s}b{i}r{r}", scenario=s, board=i, rotation=r, num_players=n,
                                             map_seed=map_seed, game_seed=game_seed,
                                             bot_seed=derive_seed(board, "bots", r), map_hash=mhash, seats=seats))
        seated = {slot for g in out for slot in g.seats}
        if not set(range(lineup_size)) <= seated:
            raise ValueError(f"suite {self.name}: a lineup of {lineup_size} has slots no game seats")
        return out

    def schedule_hash(self, lineup_size: int, *, fill: bool = False) -> str:
        return digest([[g.game_id, g.map_seed, g.game_seed, g.bot_seed, g.map_hash, list(g.seats)]
                       for g in self.schedule(lineup_size, fill=fill)])

    def key(self, lineup_size: int, *, fill: bool = False) -> dict:
        """What of the suite decides outcomes (no names), schedule hash included."""
        return {"seed": self.seed, "purpose": self.purpose, "boards": self.boards, "turn_limit": self.turn_limit,
                "determinism": self.determinism,
                "scenarios": [self._scenario(i) for i in range(len(self.scenarios))],
                "schedule_hash": self.schedule_hash(lineup_size, fill=fill)}


def load_suite(name_or_dict: str | Mapping[str, Any] | Suite) -> Suite:
    """A suite by name (``configs/suites``), path, or dict; ``{name, boards}``
    takes the first ``boards`` boards of a named suite."""
    if isinstance(name_or_dict, Suite):
        return name_or_dict
    if isinstance(name_or_dict, Mapping):
        data = dict(name_or_dict)
        if "purpose" not in data and "name" in data:
            base = load_suite(data.pop("name"))
            boards = int(data.pop("boards", base.boards))
            if data:
                raise ValueError(f"a named suite takes only boards, not {sorted(data)}")
            if boards > base.boards:
                raise ValueError(f"{base.name} has {base.boards} boards, not {boards}")
            return Suite(**{**base.__dict__, "boards": boards})
        name = str(data.pop("name", "inline"))
    else:
        path = Path(name_or_dict)
        if not path.suffix:
            path = CONFIG_DIR / "suites" / f"{name_or_dict}.yaml"
        if not path.exists():
            known = sorted(p.stem for p in (CONFIG_DIR / "suites").glob("*.yaml"))
            raise ValueError(f"unknown suite {name_or_dict!r}; known: {known}")
        data = yaml.safe_load(path.read_text())
        name = path.stem
    unknown = set(data) - SUITE_KEYS
    if unknown:
        raise ValueError(f"suite {name}: unknown keys {sorted(unknown)}")
    scenarios = tuple({"game": s} if isinstance(s, str) else dict(s) for s in data["scenarios"])
    return Suite(name=name, purpose=data["purpose"], seed=int(data["seed"]), boards=int(data["boards"]),
                 scenarios=scenarios, turn_limit=int(data.get("turn_limit", 500)),
                 determinism=data.get("determinism", "strict"))
