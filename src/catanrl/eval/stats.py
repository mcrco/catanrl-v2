"""Statistics for win rates: Wilson intervals and exact tests.

The protocol is v1's: every board is played from every seat, and agents
compared on the same boards are tested on their discordant games.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Z_95 = 1.959963984540054


@dataclass(frozen=True)
class Proportion:
    successes: float
    trials: int

    @property
    def rate(self) -> float:
        return self.successes / self.trials if self.trials else float("nan")

    def wilson(self, z: float = Z_95) -> tuple[float, float]:
        return wilson_interval(self.successes, self.trials, z)

    def __str__(self) -> str:
        low, high = self.wilson()
        return f"{100 * self.rate:.2f}% [{100 * low:.2f}, {100 * high:.2f}] (n={self.trials})"


def wilson_interval(successes: float, trials: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval; draws can be counted as half successes."""
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return (max(0.0, center - half), min(1.0, center + half))


def binomial_two_sided(successes: int, trials: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test (sum of outcomes no likelier than the observed)."""
    if trials == 0:
        return 1.0
    pmf = [math.comb(trials, k) * p**k * (1 - p) ** (trials - k) for k in range(trials + 1)]
    observed = pmf[successes]
    return min(1.0, sum(q for q in pmf if q <= observed * (1 + 1e-9)))


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Exact McNemar test on discordant pairs: A won and B lost, and vice versa."""
    return binomial_two_sided(only_a, only_a + only_b, 0.5)


def paired_comparison(wins_a: list[bool], wins_b: list[bool]) -> dict[str, float]:
    """Two agents on the same games: their rates, the delta and McNemar's p."""
    if len(wins_a) != len(wins_b):
        raise ValueError("paired results need the same games")
    only_a = sum(a and not b for a, b in zip(wins_a, wins_b))
    only_b = sum(b and not a for a, b in zip(wins_a, wins_b))
    n = len(wins_a)
    return {
        "games": n,
        "rate_a": sum(wins_a) / n if n else float("nan"),
        "rate_b": sum(wins_b) / n if n else float("nan"),
        "delta": (only_a - only_b) / n if n else float("nan"),
        "only_a": only_a,
        "only_b": only_b,
        "p_value": mcnemar_exact(only_a, only_b),
    }
