"""The dice input's features (``obs.dice_state``) recomputed from a game's ``BalancedDice``
binding with the formulas of :data:`catanrl.envs.encoding.DICE_STATE`, for the tests that
check rows wherever one is built."""

from __future__ import annotations

import cppanatron as cpp
import numpy as np

TOTALS = list(range(2, 13))
WAYS = [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]


def global_names(seats: int) -> list[str]:
    return ([f"dice_odds_{k}_{t}" for k in range(seats) for t in TOTALS]
            + [f"dice_deck_{t}" for t in TOTALS] + [f"dice_recent_{t}" for t in TOTALS]
            + ["dice_draws_left"] + [f"dice_sevens_{k}" for k in range(seats)]
            + [f"dice_seven_share_{k}" for k in range(seats)] + [f"dice_streak_{k}" for k in range(seats)])


def tile_names(seats: int) -> list[str]:
    return [f"dice_odds_{k}" for k in range(seats)] + ["dice_deck", "dice_recent"]


def expected(game: cpp.Game, observer: int, seats: int) -> tuple[dict, dict]:
    """The dice features by name for ``observer``: the global block's, and per total a tile's.
    From the game's BalancedDice (a fresh one under random dice)."""
    n = game.num_players
    dice = game.balanced_dice if game.balanced_dice is not None else cpp.BalancedDice.fresh(n)
    f32 = np.float32
    counts, recent = dice.counts, list(dice.recent)
    sevens, total_sevens = list(dice.sevens), sum(dice.sevens)
    glob, tiles = {}, {t: {} for t in TOTALS}
    for k in range(seats):
        seat = (observer + k) % n
        odds = game.roll_probabilities(seat) if k < n else [0.0] * 11
        for i, t in enumerate(TOTALS):
            glob[f"dice_odds_{k}_{t}"] = f32(odds[i] * 6.0)
            tiles[t][f"dice_odds_{k}"] = f32(odds[i] * (36.0 / 5.0))
        glob[f"dice_sevens_{k}"] = f32(min(sevens[seat] / 10.0, 1.0)) if k < n else f32(0)
        glob[f"dice_seven_share_{k}"] = f32(sevens[seat] / total_sevens) if k < n and total_sevens else f32(0)
        owner = k < n and dice.streak_player == seat
        glob[f"dice_streak_{k}"] = min(f32(dice.streak_count / 5.0), f32(1)) if owner else f32(0)
    for i, t in enumerate(TOTALS):
        glob[f"dice_deck_{t}"] = tiles[t]["dice_deck"] = f32(counts[i] / WAYS[i])
        glob[f"dice_recent_{t}"] = tiles[t]["dice_recent"] = min(f32(recent.count(t) / 3.0), f32(1))
    glob["dice_draws_left"] = f32((len(dice.drawable) - 13) / 23.0)
    return glob, tiles


class DiceColumns:
    """Where a row of ``encoder`` (built with the dice input) holds its dice features."""

    def __init__(self, encoder: cpp.GraphEncoder) -> None:
        layout = encoder.layout()
        self.seats = int(layout["players"][1][0])
        self.global_names, self.tile_feature_names = list(encoder.feature_names("global")), list(
            encoder.feature_names("tiles"))
        self.g0 = int(layout["global"][0])
        self.t0, (self.num_tiles, self.tile_width) = int(layout["tiles"][0]), layout["tiles"][1]
        self.global_dice = [n for n in self.global_names if n.startswith("dice_")]
        self.tile_dice = [n for n in self.tile_feature_names if n.startswith("dice_")]
        assert self.global_dice == global_names(self.seats) == self.global_names[-len(self.global_dice):]
        assert self.tile_dice == tile_names(self.seats) == self.tile_feature_names[-len(self.tile_dice):]
        self.numbers = [(int(name.removeprefix("number_")), i) for i, name in enumerate(self.tile_feature_names)
                        if name.startswith("number_")]

    def glob(self, row: np.ndarray) -> np.ndarray:
        """The global dice features of a row, in name order."""
        first = self.g0 + len(self.global_names) - len(self.global_dice)
        return np.asarray(row[first: first + len(self.global_dice)])

    def tiles(self, row: np.ndarray) -> np.ndarray:
        """[tiles, P + 2]: each tile slot's dice features."""
        block = np.asarray(row[self.t0: self.t0 + self.num_tiles * self.tile_width])
        return block.reshape(self.num_tiles, self.tile_width)[:, -len(self.tile_dice):]

    def tile_numbers(self, row: np.ndarray) -> list[int]:
        """Each tile slot's number (0: the desert or a padded slot)."""
        block = np.asarray(row[self.t0: self.t0 + self.num_tiles * self.tile_width])
        block = block.reshape(self.num_tiles, self.tile_width)
        return [next((n for n, i in self.numbers if tile[i]), 0) for tile in block]

    def all(self, row: np.ndarray) -> np.ndarray:
        """Every dice feature of a row: the global ones, then each tile's."""
        return np.concatenate([self.glob(row), self.tiles(row).ravel()])

    def want(self, game: cpp.Game, observer: int, row: np.ndarray) -> np.ndarray:
        """What :meth:`all` must read for ``observer`` in ``game`` (the tiles' numbers from the row)."""
        glob, tiles = expected(game, observer, self.seats)
        out = [glob[name] for name in self.global_dice]
        for number in self.tile_numbers(row):
            out += [tiles[number][name] if number else np.float32(0) for name in self.tile_dice]
        return np.array(out, dtype=np.float32)

    def check(self, game: cpp.Game, observer: int, row: np.ndarray) -> None:
        """The row's dice features are ``game``'s for ``observer``, float32 for float32."""
        got, want = self.all(row), self.want(game, observer, row)
        assert np.array_equal(got, want), [(i, g, w) for i, (g, w) in enumerate(zip(got, want)) if g != w][:5]
