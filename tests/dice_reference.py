"""OpenHex's balanced dice: an exact, dependency-free reference.

Written from a reading of OpenHex's engine (``firebase/functions/openhex/dice.py``,
``engine.py``'s ``_roll`` / ``_exec_roll_dice`` / ``_advance_setup``, ``state.py``); nothing
of OpenHex is imported. OpenHex's code is authoritative where its SPEC.md disagrees (the SPEC's
"draw 25, then reshuffle" deck is not what the code does).

The model, per roll by ``roller`` (the seat whose turn it is):

1. **Reshuffle.** If at most 13 cards remain, the deck becomes the full 36 (d1, d2) cards
   again. Only the deck resets: the recent totals, the sevens per seat and the streak carry
   over. The deck is also refilled when the main phase starts; no roll happens before, so a
   fresh :class:`BalancedDice` stands for that state. A deck yields 23 draws (36 down to 13).
2. **Weights per sum** k in 2..12 over the (possibly refilled) deck of T cards holding
   ``c_k`` cards of sum k: ``w_k = c_k / T * max(0, 1 - 0.34 * a_k)``, ``a_k`` the
   appearances of k among the last 5 totals (3 appearances zero it, 2 leave 0.32, 1 leaves
   0.66).
3. **Sevens** (``roller`` not None, ``num_players >= 2`` and ``w_7 > 0``): ``w_7`` is
   multiplied by ``clamp(imbalance + streak, 0, 2)``. ``streak = 0.4 * count``, negated when
   the roller owns the streak; ``imbalance = 1 + (1/n - s/S) / (1/n)`` once the sevens rolled
   so far ``S >= n`` (``s`` the roller's), else 1.
4. **Draw.** Every remaining card of sum k weighs ``w_k / c_k``: the sum comes up with
   probability ``w_k / sum(w)``, and then each of its remaining cards equally. If every weight
   is zero, a card is drawn uniformly (unreachable: at least 14 cards span at least three
   sums, and recency can zero only one of them, the seven multiplier only the 7).
5. **Update.** The card leaves the deck; its total joins the last-5 window (oldest dropped);
   a 7 counts for the roller and extends the roller's streak (a 7 by another seat starts a
   new streak of 1); any other total clears the streak.

The deck order never matters (a card's chance depends only on its sum), so the per-sum counts
of the remaining cards are a sufficient statistic for every future total; the faces only name
which card left.

Probabilities are exact (:class:`fractions.Fraction`, with 0.34 and 0.4 as decimals). OpenHex
computes in floating point, which differs by rounding only.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction

Card = tuple[int, int]

#: The 36 cards, one per ordered pair of faces.
FULL_DECK: tuple[Card, ...] = tuple((a, b) for a in range(1, 7) for b in range(1, 7))
SUMS: tuple[int, ...] = tuple(range(2, 13))
#: A draw first refills the deck when at most this many cards remain.
RESHUFFLE_AT = 13
#: Totals remembered for the recency penalty.
WINDOW = 5
RECENCY_PENALTY = Fraction(34, 100)
STREAK_STEP = Fraction(4, 10)
MULTIPLIER_MIN = Fraction(0)
MULTIPLIER_MAX = Fraction(2)


def card_sum(card: Card) -> int:
    return card[0] + card[1]


def sum_counts(cards: Iterable[Card]) -> dict[int, int]:
    """Cards per sum, every sum 2..12 present (zero when none is left)."""
    counts = Counter(card_sum(c) for c in cards)
    return {k: counts.get(k, 0) for k in SUMS}


def distribution(counts: Mapping[int, int], weights: Mapping[int, Fraction]) -> dict[int, Fraction]:
    """P(sum) from the cards per sum and the weights per sum (step 4): ``w_k / sum(w)``, or
    uniform over the cards (``c_k / T``) when every weight is zero."""
    total_w = sum(weights.values(), Fraction(0))
    if total_w > 0:
        return {k: weights[k] / total_w for k in SUMS}
    total = sum(counts.values())
    return {k: Fraction(counts[k], total) for k in SUMS}


@dataclass(frozen=True)
class BalancedDice:
    """The dice state between two rolls; everything in it is public (rolls are public).

    ``sevens`` has one entry per seat (seats are turn-order indices); ``num_players`` is
    OpenHex's ``config.num_players`` (the seats when unset). ``streak_player`` owns the current
    run of sevens, ``streak_count`` long (None and 0 when there is none).
    """

    sevens: tuple[int, ...] = (0, 0)
    deck: frozenset[Card] = field(default=frozenset(FULL_DECK))
    recent: tuple[int, ...] = ()
    streak_player: int | None = None
    streak_count: int = 0
    num_players: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "deck", frozenset(self.deck))
        object.__setattr__(self, "recent", tuple(self.recent))
        object.__setattr__(self, "sevens", tuple(self.sevens))
        if not self.deck <= frozenset(FULL_DECK):
            raise ValueError(f"cards outside the 36-card deck: {sorted(self.deck - frozenset(FULL_DECK))}")
        if len(self.recent) > WINDOW or any(t not in SUMS for t in self.recent):
            raise ValueError(f"recent holds at most {WINDOW} totals in 2..12: {self.recent}")
        if not self.sevens or any(s < 0 for s in self.sevens):
            raise ValueError(f"sevens needs one non-negative count per seat: {self.sevens}")
        if (self.streak_player is None) != (self.streak_count == 0) or self.streak_count < 0:
            raise ValueError("a streak has an owner exactly when its count is positive")
        if self.streak_player is not None and not 0 <= self.streak_player < len(self.sevens):
            raise ValueError(f"streak owner {self.streak_player} is not a seat")
        if self.num_players is not None and self.num_players < 1:
            raise ValueError("num_players must be positive")

    @classmethod
    def new(cls, seats: int, num_players: int | None = None) -> BalancedDice:
        """The state at the start of the main phase: a full deck and no history."""
        return cls(sevens=(0,) * seats, num_players=num_players)

    @property
    def n(self) -> int:
        return self.num_players if self.num_players is not None else len(self.sevens)

    def drawable(self) -> frozenset[Card]:
        """The deck the next draw takes from: the full deck if it is refilled first (step 1)."""
        return frozenset(FULL_DECK) if len(self.deck) <= RESHUFFLE_AT else self.deck

    def counts(self) -> dict[int, int]:
        """Cards per sum in :meth:`drawable`."""
        return sum_counts(self.drawable())

    def seven_multiplier(self, roller: int | None) -> Fraction:
        """The factor on the 7's weight for ``roller`` (step 3); 1 when it does not apply."""
        if roller is None or self.n < 2:
            return Fraction(1)
        streak = Fraction(0)
        if self.streak_count > 0 and self.streak_player is not None:
            direction = -1 if self.streak_player == roller else 1
            streak = STREAK_STEP * self.streak_count * direction
        total = sum(self.sevens)
        if total >= self.n:
            ideal = Fraction(1, self.n)
            share = Fraction(self.sevens[roller], total)
            imbalance = 1 + (ideal - share) / ideal
        else:
            imbalance = Fraction(1)
        return max(MULTIPLIER_MIN, min(MULTIPLIER_MAX, imbalance + streak))

    def sum_weights(self, roller: int | None) -> dict[int, Fraction]:
        """``w_k`` for every sum (steps 2-3)."""
        counts = self.counts()
        total = sum(counts.values())
        weights = {}
        for k in SUMS:
            if counts[k] == 0:
                weights[k] = Fraction(0)
                continue
            penalty = 1 - RECENCY_PENALTY * self.recent.count(k)
            weights[k] = Fraction(counts[k], total) * max(Fraction(0), penalty)
        if roller is not None and self.n >= 2 and weights[7] > 0:
            weights[7] *= self.seven_multiplier(roller)
        return weights

    def sum_probabilities(self, roller: int | None) -> dict[int, Fraction]:
        """P(next total = k) for every k in 2..12 when ``roller`` rolls."""
        return distribution(self.counts(), self.sum_weights(roller))

    def card_probabilities(self, roller: int | None) -> dict[Card, Fraction]:
        """P(next card = c) for each of the 36 cards (zero for a card not in the deck)."""
        counts = self.counts()
        by_sum = self.sum_probabilities(roller)
        drawable = self.drawable()
        return {c: by_sum[card_sum(c)] / counts[card_sum(c)] if c in drawable else Fraction(0)
                for c in FULL_DECK}

    def apply(self, card: Card, roller: int | None) -> BalancedDice:
        """The state after ``roller`` draws ``card`` (steps 1 and 5)."""
        card = (int(card[0]), int(card[1]))
        deck = self.drawable()
        if card not in deck:
            raise ValueError(f"{card} is not in the deck ({len(deck)} cards left)")
        total = card_sum(card)
        sevens, streak_player, streak_count = self.sevens, self.streak_player, self.streak_count
        if roller is not None:
            if not 0 <= roller < len(self.sevens):
                raise ValueError(f"roller {roller} is not a seat")
            if total == 7:
                sevens = tuple(s + (seat == roller) for seat, s in enumerate(sevens))
                if streak_player == roller:
                    streak_count += 1
                else:
                    streak_player, streak_count = roller, 1
            else:
                streak_player, streak_count = None, 0
        return BalancedDice(sevens=sevens, deck=deck - {card},
                            recent=(self.recent + (total,))[-WINDOW:], streak_player=streak_player,
                            streak_count=streak_count, num_players=self.num_players)
