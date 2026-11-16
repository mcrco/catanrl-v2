"""How far OpenHex's balanced dice move the next roll's odds from two fair dice.

Plays the dice alone (no board, no agents): ``--games`` 1v1 games of ``--rolls`` rolls, the seats
alternating, each total drawn by ``BalancedDice.sum_probabilities`` and applied with
``card_for_total``. Prints the roller's P(7) and P(8) percentiles, P(7) by sevens imbalance and
after the other seat's 7, P(8) right after an 8, the total-variation distance from fair, and how
often some total is impossible. The numbers behind the dice input (``obs.dice_state``,
configs/experiments/az-2p-dice.yaml)::

    .venv/bin/python tools/dice_odds_sim.py --games 5000 --rolls 80
"""

from __future__ import annotations

import argparse
import random

import numpy as np

import cppanatron as cpp

FAIR = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]) / 36


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, default=5000)
    parser.add_argument("--rolls", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    p7, p8, tv, impossible = [], [], [], []
    after_other_7, other, ahead2, behind2, after_8 = [], [], [], [], []
    for _ in range(args.games):
        dice = cpp.BalancedDice.fresh(2)
        previous, sevens = None, [0, 0]
        for t in range(args.rolls):
            roller = t % 2
            p = np.array(dice.sum_probabilities(roller))
            p7.append(p[5])
            p8.append(p[6])
            tv.append(0.5 * np.abs(p - FAIR).sum())
            impossible.append(bool((p == 0).any()))
            (after_other_7 if previous == 7 else other).append(p[5])
            if sevens[roller] - sevens[1 - roller] >= 2:
                ahead2.append(p[5])
            if sevens[1 - roller] - sevens[roller] >= 2:
                behind2.append(p[5])
            if previous == 8:
                after_8.append(p[6])
            total = rng.choices(range(2, 13), weights=p)[0]
            dice = dice.apply(dice.card_for_total(total), roller)
            sevens[roller] += total == 7
            previous = total

    def pct(values):
        return np.percentile(values, [5, 25, 50, 75, 95]).round(3).tolist()

    def mean(values):
        return f"{np.mean(values):.3f} (n = {len(values)})"

    print(f"roller's P(7), percentiles 5/25/50/75/95: {pct(p7)}  (fair {FAIR[5]:.3f})")
    print(f"roller's P(8), percentiles 5/25/50/75/95: {pct(p8)}  (fair {FAIR[6]:.3f})")
    print(f"P(7), roller 2+ sevens ahead: {mean(ahead2)}; 2+ behind: {mean(behind2)}")
    print(f"P(7) right after the other seat's 7: {mean(after_other_7)}; otherwise: {mean(other)}")
    print(f"P(8) right after an 8: {mean(after_8)}")
    print(f"total-variation distance from fair: mean {np.mean(tv):.3f}, 95th {np.percentile(tv, 95):.3f}")
    print(f"rolls with some total impossible: {np.mean(impossible):.3f}")


if __name__ == "__main__":
    main()
