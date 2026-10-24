"""Game configs: player count, map generator and rules, loaded from YAML.

A config names only what differs from the base game::

    name: extension-6p
    num_players: 6
    map: {type: extension}              # any catanrl.maps generator
    victory_points_to_win: 10
    discard_limit: 7
    friendly_robber: false
    dice_mode: random                   # or balanced: OpenHex's 1v1 dice
    resource_bank: 24                   # or {wood: 24, brick: 24, ...}
    development_deck: {knight: 20, year_of_plenty: 3, monopoly: 3,
                       road_building: 3, victory_point: 5}
    pieces: {roads: 15, settlements: 5, cities: 4}

:class:`GameConfig` builds cppanatron rules and maps and Catanatron games, and
:meth:`GameConfig.to_dict` gives the YAML shape back so runs record exactly
what they played. ``dice_mode`` enters the YAML shape only when it is not
``random``, so every game written before it hashes as it did.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import yaml

from catanrl import maps

if TYPE_CHECKING:
    import cppanatron

MAX_PLAYERS = 6
#: How rolls are drawn: two fair dice, or OpenHex's balanced dice (``cppanatron.BalancedDice``:
#: a 36-card deck with recency and seven balancing, all of it public).
DICE_MODES = ("random", "balanced")
RESOURCE_NAMES = ("wood", "brick", "sheep", "wheat", "ore")
DEVELOPMENT_CARD_NAMES = ("knight", "year_of_plenty", "monopoly", "road_building", "victory_point")
PIECE_FIELDS = {"roads": "roads_per_player", "settlements": "settlements_per_player",
                "cities": "cities_per_player"}
CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@dataclass(frozen=True)
class GameConfig:
    """One kind of game. Defaults are the base game's rules on the BASE map."""

    name: str
    num_players: int
    map: Mapping[str, Any] = field(default_factory=lambda: {"type": "BASE"})
    victory_points_to_win: int = 10
    discard_limit: int = 7
    friendly_robber: bool = False
    #: One of DICE_MODES; ``random`` (the default) is left out of :meth:`to_dict`.
    dice_mode: str = "random"
    #: Starting bank in RESOURCE_NAMES order.
    resource_bank: tuple[int, ...] = (19, 19, 19, 19, 19)
    #: Starting deck in DEVELOPMENT_CARD_NAMES order.
    development_deck: tuple[int, ...] = (14, 2, 2, 2, 5)
    roads_per_player: int = 15
    settlements_per_player: int = 5
    cities_per_player: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "map", dict(self.map))
        object.__setattr__(self, "resource_bank", tuple(self.resource_bank))
        object.__setattr__(self, "development_deck", tuple(self.development_deck))
        if not 2 <= self.num_players <= MAX_PLAYERS:
            raise ValueError(f"num_players must be between 2 and {MAX_PLAYERS}")
        if self.dice_mode not in DICE_MODES:
            raise ValueError(f"dice_mode must be one of {DICE_MODES}, not {self.dice_mode!r}")
        if self.victory_points_to_win < 1 or self.discard_limit < 0:
            raise ValueError("victory_points_to_win must be positive, discard_limit non-negative")
        for name in ("resource_bank", "development_deck"):
            counts = getattr(self, name)
            if len(counts) != 5 or any(not isinstance(c, int) or c < 0 for c in counts):
                raise ValueError(f"{name} needs five non-negative integer counts")
        if self.roads_per_player < 2 or self.settlements_per_player < 2 or self.cities_per_player < 0:
            raise ValueError("players need at least two roads and two settlements")
        if "type" not in self.map:
            raise ValueError("map needs a type (see catanrl.maps.generate)")

    # ------------------------------------------------------------ engines

    def map_spec(self, seed: int) -> maps.MapSpec:
        return maps.generate(self.map, seed)

    def cppanatron_rules(self) -> "cppanatron.GameRules":
        import cppanatron as cpp

        return cpp.GameRules(
            vps_to_win=self.victory_points_to_win,
            discard_limit=self.discard_limit,
            friendly_robber=self.friendly_robber,
            resource_bank=self.resource_bank,
            development_deck=self.development_deck,
            roads_per_player=self.roads_per_player,
            settlements_per_player=self.settlements_per_player,
            cities_per_player=self.cities_per_player,
            dice_mode=cpp.DiceMode[self.dice_mode.upper()],
        )

    def cppanatron_game(self, map_seed: int, seed: int) -> "cppanatron.Game":
        import cppanatron as cpp

        catan_map = cpp.CatanMap.from_spec(cpp.MapSpec.from_dict(self.map_spec(map_seed)))
        return cpp.Game(self.num_players, map=catan_map, rules=self.cppanatron_rules(), seed=seed)

    def catanatron_game(self, players, map_seed: int, seed: int):
        """A Catanatron game on the same map as :meth:`cppanatron_game` (fair dice only:
        Catanatron has no balanced dice, so such a game is refused)."""
        if self.dice_mode != "random":
            raise ValueError(f"game {self.name}: Catanatron rolls two fair dice, not {self.dice_mode} dice")
        from catanatron.game import Game
        from catanatron.models.decks import Supply
        from cppanatron.catanatron import catanatron_map_from_spec

        return Game(
            players,
            seed=seed,
            catan_map=catanatron_map_from_spec(self.map_spec(map_seed)),
            vps_to_win=self.victory_points_to_win,
            discard_limit=self.discard_limit,
            friendly_robber=self.friendly_robber,
            supply=Supply(
                resource_bank=self.resource_bank,
                development_deck=self.development_deck,
                roads=self.roads_per_player,
                settlements=self.settlements_per_player,
                cities=self.cities_per_player,
            ),
        )

    # ------------------------------------------------------------ YAML

    def to_dict(self) -> dict[str, Any]:
        out = {
            "name": self.name,
            "num_players": self.num_players,
            "map": dict(self.map),
            "victory_points_to_win": self.victory_points_to_win,
            "discard_limit": self.discard_limit,
            "friendly_robber": self.friendly_robber,
            "resource_bank": dict(zip(RESOURCE_NAMES, self.resource_bank)),
            "development_deck": dict(zip(DEVELOPMENT_CARD_NAMES, self.development_deck)),
            "pieces": {key: getattr(self, attr) for key, attr in PIECE_FIELDS.items()},
        }
        if self.dice_mode != "random":
            # Only when set: a game (and so a scenario, suite or config hash) without it is unchanged.
            out["dice_mode"] = self.dice_mode
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, default_name: str = "game") -> GameConfig:
        known = {f.name for f in fields(cls)} | {"pieces"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown game config keys: {sorted(unknown)}")
        values = {k: v for k, v in data.items() if k not in ("pieces", "resource_bank",
                                                             "development_deck")}
        values.setdefault("name", default_name)
        if "resource_bank" in data:
            values["resource_bank"] = _counts(data["resource_bank"], RESOURCE_NAMES)
        if "development_deck" in data:
            values["development_deck"] = _counts(data["development_deck"], DEVELOPMENT_CARD_NAMES,
                                                 uniform=False)
        for key, value in (data.get("pieces") or {}).items():
            if key not in PIECE_FIELDS:
                raise ValueError(f"unknown piece {key!r}; expected {sorted(PIECE_FIELDS)}")
            values[PIECE_FIELDS[key]] = value
        return cls(**values)


def _counts(value: Any, names: tuple[str, ...], *, uniform: bool = True) -> tuple[int, ...]:
    if isinstance(value, int) and uniform:
        return (value,) * len(names)
    if isinstance(value, Mapping):
        unknown = set(value) - set(names)
        if unknown or set(value) != set(names):
            raise ValueError(f"expected exactly the keys {names}, got {sorted(value)}")
        return tuple(value[name] for name in names)
    if isinstance(value, (list, tuple)) and len(value) == len(names):
        return tuple(value)
    raise ValueError(f"expected counts for {names}, got {value!r}")


def load_game(name_or_path: str | Path | Mapping[str, Any]) -> GameConfig:
    """A game config by inline dict, path, or name under configs/games/."""
    if isinstance(name_or_path, Mapping):
        return GameConfig.from_dict(name_or_path)
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / "games" / f"{name_or_path}.yaml"
    with open(path) as f:
        return GameConfig.from_dict(yaml.safe_load(f), default_name=path.stem)
