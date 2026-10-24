"""cppanatron's balanced dice against the exact reference (tests/dice_reference.py).

Random games under OpenHex's 1v1 rules (and 3 and 4 seats) are played with the
engine's balanced dice, drawn by the engine or replayed from the reference's
own odds. At every roll the reference, fed only the cards drawn, must hold the
engine's whole state, and the engine's odds must equal the reference's exact
fractions for every seat; one state's draws must follow them.
"""

from __future__ import annotations

import random
from collections import Counter
from fractions import Fraction

import pytest
from dice_reference import FULL_DECK, SUMS, BalancedDice, card_sum

cpp = pytest.importorskip("cppanatron")


def rules(players: int) -> cpp.GameRules:
    if players == 2:  # OpenHex's 1v1
        return cpp.GameRules(vps_to_win=15, discard_limit=9, friendly_robber=True,
                             dice_mode=cpp.DiceMode.BALANCED)
    return cpp.GameRules(dice_mode=cpp.DiceMode.BALANCED)


def as_reference(dice) -> BalancedDice:
    return BalancedDice(sevens=dice.sevens, deck=frozenset(dice.deck), recent=dice.recent,
                        streak_player=dice.streak_player, streak_count=dice.streak_count)


def check_state(game, reference: BalancedDice, players: int) -> None:
    engine = game.balanced_dice
    assert as_reference(engine) == reference
    assert frozenset(engine.drawable) == reference.drawable()
    assert engine.counts == tuple(reference.counts()[k] for k in SUMS)
    for seat in range(players):
        numerator, denominator = engine.seven_multiplier(seat)
        assert Fraction(numerator, denominator) == reference.seven_multiplier(seat)
        exact = reference.sum_probabilities(seat)
        for k, probability in zip(SUMS, engine.sum_probabilities(seat)):
            assert abs(probability - float(exact[k])) <= 1e-12
            assert (probability == 0) == (exact[k] == 0)
    assert game.roll_probabilities(game.next_roller) == engine.sum_probabilities(game.next_roller)


def drawn_card(before, after) -> tuple[int, int]:
    (card,) = frozenset(before.drawable) - frozenset(after.deck)
    return card


def play(players: int, seed: int, replay: bool, limit: int = 4000) -> int:
    """One random game; returns the rolls checked."""
    game = cpp.Game(players, seed=seed, rules=rules(players))
    reference = BalancedDice.new(players)
    rng = random.Random(seed)
    rolls = 0
    for _ in range(limit):
        if game.winning_color is not None:
            break
        action = rng.choice(game.playable_actions)
        if action.action_type != cpp.ActionType.ROLL:
            game.execute(action)
            continue
        roller = game.current_player_index
        assert game.next_roller == roller
        check_state(game, reference, players)
        before = game.balanced_dice
        if replay:
            odds = reference.card_probabilities(roller)
            cards = [c for c in FULL_DECK if odds[c] > 0]
            card = rng.choices(cards, weights=[float(odds[c]) for c in cards])[0]
            game.execute(action, dice=card)
        else:
            game.execute(action)
            card = drawn_card(before, game.balanced_dice)
        assert reference.sum_probabilities(roller)[card_sum(card)] > 0
        reference = reference.apply(card, roller)
        rolls += 1
    check_state(game, reference, players)
    return rolls


@pytest.mark.parametrize("players,games", [(2, 16), (3, 6), (4, 6)])
def test_engine_odds_equal_the_reference(players, games):
    # About 8700 + 1700 + 1800 rolls, each checked for every seat.
    rolls = sum(play(players, seed, replay=seed % 2 == 1) for seed in range(games))
    assert rolls >= 250 * games


def test_draws_follow_the_reference_odds():
    """20000 draws from one state (reseeded by redealing the unchanged hands)."""
    # A state with every rule in play: the window holds one 7, the opponent's 7 just
    # started a streak, and the 7s so far bring in the imbalance term.
    game = cpp.Game(2, seed=4, rules=rules(2))
    rng = random.Random(4)
    while True:
        actions = game.playable_actions
        if game.current_prompt == cpp.ActionPrompt.PLAY_TURN and any(
                a.action_type == cpp.ActionType.ROLL for a in actions):
            dice = game.balanced_dice
            if dice.streak_count == 1 and dice.recent.count(7) == 1 and sum(dice.sevens) >= 2:
                break
        game.execute(rng.choice(actions))
        assert game.winning_color is None
    roller = game.current_player_index
    reference = as_reference(game.balanced_dice)
    odds = reference.card_probabilities(roller)
    hands = [cpp.HiddenHand(resources=p.resources, development_cards=p.development_cards,
                            development_card_owned_at_start=p.development_card_owned_at_start)
             for p in game.players]
    roll = next(a for a in game.playable_actions if a.action_type == cpp.ActionType.ROLL)
    draws = 20000
    cards: Counter = Counter()
    for seed in range(draws):
        trial = game.copy()
        trial.redeal_hidden_cards(hands, seed=seed)
        before = trial.balanced_dice
        trial.execute(roll)
        cards[drawn_card(before, trial.balanced_dice)] += 1
    assert all(odds[c] > 0 for c in cards), "a card without a chance was drawn"
    # Chi-square over the cards with a chance (df = cards - 1).
    support = [c for c in FULL_DECK if odds[c] > 0]
    chi2 = sum((cards[c] - draws * float(odds[c])) ** 2 / (draws * float(odds[c])) for c in support)
    df = len(support) - 1
    assert chi2 < df + 5 * (2 * df) ** 0.5, f"chi2 {chi2:.1f} on {df} degrees of freedom"
    totals = Counter(card_sum(c) for c in cards.elements())
    for k, p in reference.sum_probabilities(roller).items():
        sd = (draws * float(p) * (1 - float(p))) ** 0.5
        assert abs(totals[k] - draws * float(p)) <= 5 * sd + 1
