"""The balanced-dice reference (tests/dice_reference.py) on hand-computed cases, and what it
says of the two recorded OpenHex 1v1 games (fair dice despite their config)."""

from __future__ import annotations

import json
from fractions import Fraction as F
from pathlib import Path

import pytest
from dice_reference import FULL_DECK, SUMS, BalancedDice, card_sum, distribution, sum_counts

FIXTURES = Path(__file__).parent / "fixtures" / "openhex"
TWO_D6 = {k: F(6 - abs(7 - k), 36) for k in SUMS}


def deck_without(*cards):
    return frozenset(FULL_DECK) - set(cards)


def check_consistent(state, roller):
    by_sum = state.sum_probabilities(roller)
    by_card = state.card_probabilities(roller)
    assert sum(by_sum.values()) == 1 and sum(by_card.values()) == 1
    for k in SUMS:
        assert sum(p for c, p in by_card.items() if card_sum(c) == k) == by_sum[k]
    return by_sum, by_card


def test_a_fresh_deck_is_two_fair_dice():
    state = BalancedDice.new(2)
    for roller in (0, 1, None):
        by_sum, by_card = check_consistent(state, roller)
        assert by_sum == TWO_D6
        assert set(by_card.values()) == {F(1, 36)}


def test_after_an_opening_seven_the_other_seat_gets_the_streak_boost():
    # Seat 0 rolls (3, 4): 35 cards, five 7s left, recent (7,), sevens (1, 0), streak (0, 1).
    state = BalancedDice.new(2).apply((3, 4), 0)
    assert len(state.deck) == 35 and state.recent == (7,) and state.sevens == (1, 0)
    assert (state.streak_player, state.streak_count) == (0, 1)
    # Seat 1: recency 1 - 0.34 = 33/50; streak +0.4, 1 seven < 2 players so imbalance 1: x 7/5.
    assert state.seven_multiplier(1) == F(7, 5)
    weights = state.sum_weights(1)
    assert weights[7] == F(5, 35) * F(33, 50) * F(7, 5) == F(33, 250)
    assert weights[2] == F(1, 35) and weights[6] == F(5, 35)
    # sum(w) = 30/35 + 33/250 = 1731/1750.
    by_sum, by_card = check_consistent(state, 1)
    assert by_sum[7] == F(231, 1731)
    assert by_sum[2] == F(50, 1731) and by_sum[6] == F(250, 1731)
    assert by_card[(1, 6)] == F(231, 1731 * 5) and by_card[(3, 4)] == 0
    # The streak's owner rolling again (not a normal turn order): 1 - 0.4 = 3/5.
    assert state.seven_multiplier(0) == F(3, 5)
    assert state.sum_probabilities(0)[7] == F(99, 1599)


def test_imbalance_once_every_seat_could_have_a_seven():
    state = BalancedDice(sevens=(3, 1))
    # Seat 0 rolled 3/4 of the sevens: 1 + (1/2 - 3/4) / (1/2) = 1/2; seat 1: 3/2.
    assert state.seven_multiplier(0) == F(1, 2) and state.seven_multiplier(1) == F(3, 2)
    assert state.sum_probabilities(0)[7] == F(3, 33)   # (6/36 * 1/2) / (30/36 + 3/36)
    assert state.sum_probabilities(1)[7] == F(9, 39)   # (6/36 * 3/2) / (30/36 + 9/36)
    assert state.sum_probabilities(0)[8] == F(5, 33)
    # Fewer sevens than players: no imbalance term.
    assert BalancedDice(sevens=(1, 0)).seven_multiplier(0) == 1
    # Four players: 2 sevens < 4, then 4 sevens with the roller holding 2: 1 + (1/4 - 1/2) * 4 = 0.
    assert BalancedDice(sevens=(2, 0, 0, 0)).seven_multiplier(0) == 1
    assert BalancedDice(sevens=(2, 1, 1, 0)).seven_multiplier(0) == 0
    assert BalancedDice(sevens=(2, 1, 1, 0)).seven_multiplier(3) == 2
    # num_players is OpenHex's config value, not the seats.
    assert BalancedDice(sevens=(1, 1), num_players=4).seven_multiplier(0) == 1


def test_the_multiplier_is_clamped_to_zero_and_two():
    # No sevens of the roller's yet (imbalance 2) and the other seat's streak (+0.4): 2.4 -> 2.
    high = BalancedDice(sevens=(0, 2), streak_player=1, streak_count=1)
    assert high.seven_multiplier(0) == 2
    assert high.sum_probabilities(0)[7] == F(12, 42)
    # Every seven the roller's (imbalance 0) and its own streak (-0.4): -0.4 -> 0.
    low = BalancedDice(sevens=(4, 0), streak_player=0, streak_count=1)
    assert low.seven_multiplier(0) == 0
    by_sum, _ = check_consistent(low, 0)
    assert by_sum[7] == 0 and by_sum[8] == F(5, 30)
    # A longer streak (possible only if a seat rolls twice running): 0.4 per seven.
    assert BalancedDice(sevens=(2, 0), streak_player=0, streak_count=2).seven_multiplier(1) == 2
    assert BalancedDice(sevens=(1, 0), streak_player=0, streak_count=1).seven_multiplier(1) == F(7, 5)


def test_recency_penalty():
    state = BalancedDice(recent=(6, 6, 6, 8, 8))
    weights = state.sum_weights(0)
    assert weights[6] == 0                       # 1 - 3 * 0.34 < 0
    assert weights[8] == F(5, 36) * F(8, 25)     # 1 - 2 * 0.34 = 0.32
    # sum(w) = 26/36 + 2/45 = 23/30.
    by_sum, _ = check_consistent(state, 0)
    assert by_sum[8] == F(4, 69) and by_sum[2] == F(5, 138) and by_sum[6] == 0
    assert by_sum[7] == F(6, 36) / F(23, 30)
    # One appearance: 0.66.
    assert BalancedDice(recent=(2,)).sum_weights(0)[2] == F(1, 36) * F(33, 50)


def test_a_seven_without_weight_skips_the_multiplier():
    sevens = [c for c in FULL_DECK if card_sum(c) == 7]
    state = BalancedDice(sevens=(0, 3), deck=deck_without(*sevens), streak_player=1, streak_count=1)
    by_sum, _ = check_consistent(state, 0)
    assert by_sum[7] == 0 and by_sum[6] == F(5, 30)


def test_a_sum_with_no_card_left_never_comes():
    state = BalancedDice(deck=deck_without((1, 1), (6, 6), (2, 5)))
    by_sum, by_card = check_consistent(state, 0)
    assert by_sum[2] == by_sum[12] == 0
    assert by_sum[7] == F(5, 33)
    assert by_card[(2, 5)] == 0 and by_card[(5, 2)] == F(1, 33)


def test_reshuffle_refills_only_the_deck():
    # 14 cards left: no refill, the draw leaves 13.
    fourteen = frozenset(FULL_DECK[:14])
    state = BalancedDice(sevens=(1, 2), deck=fourteen, recent=(7, 7, 4, 5, 9),
                         streak_player=1, streak_count=1)
    assert state.drawable() == fourteen
    after = state.apply((1, 2), 0)
    assert len(after.deck) == 13 and after.recent == (7, 4, 5, 9, 3)
    assert (after.streak_player, after.streak_count) == (None, 0) and after.sevens == (1, 2)
    # 13 left: the next draw sees the full deck, with the history kept.
    assert after.drawable() == frozenset(FULL_DECK)
    assert after.counts() == sum_counts(FULL_DECK)
    assert after.card_probabilities(1)[(1, 2)] > 0
    again = after.apply((1, 2), 1)
    assert len(again.deck) == 35 and (1, 2) not in again.deck
    assert again.recent == (4, 5, 9, 3, 3) and again.sevens == (1, 2)


def test_a_deck_lasts_twenty_three_draws():
    state, sizes = BalancedDice.new(2), []
    order = [c for c in FULL_DECK if card_sum(c) != 7] + [c for c in FULL_DECK if card_sum(c) == 7]
    for i in range(50):
        card = next(c for c in order if c in state.drawable())
        state = state.apply(card, i % 2)
        sizes.append(len(state.deck))
    assert sizes[:23] == list(range(35, 12, -1))
    assert sizes[23:46] == list(range(35, 12, -1)) and sizes[46] == 35


def test_window_and_streak_updates():
    state = BalancedDice.new(3)
    for card, roller in [((1, 6), 0), ((2, 5), 0), ((3, 3), 1), ((4, 3), 2), ((6, 1), 1), ((5, 2), 1)]:
        state = state.apply(card, roller)
    assert state.recent == (7, 6, 7, 7, 7)
    assert state.sevens == (2, 2, 1) and (state.streak_player, state.streak_count) == (1, 2)
    state = state.apply((2, 2), 0)
    assert state.recent == (6, 7, 7, 7, 4) and (state.streak_player, state.streak_count) == (None, 0)
    # Five 7s drawn of six, three in the window: the 7 is out whatever the multiplier.
    assert state.sum_probabilities(2)[7] == 0
    # A roll with no roller (OpenHex's current_player None) moves the window only.
    unowned = BalancedDice.new(2).apply((3, 4), None)
    assert unowned.recent == (7,) and unowned.sevens == (0, 0) and unowned.streak_count == 0
    assert unowned.sum_probabilities(None)[7] == F(5, 35) * F(33, 50) / (F(30, 35) + F(5, 35) * F(33, 50))


def test_only_per_sum_counts_matter():
    # Decks with the same cards per sum give the same totals; the card chances follow the faces.
    a = BalancedDice(sevens=(2, 1), deck=deck_without((1, 2), (3, 4), (2, 2)), recent=(3, 7, 4))
    b = BalancedDice(sevens=(2, 1), deck=deck_without((2, 1), (5, 2), (1, 3)), recent=(3, 7, 4))
    for roller in (0, 1):
        assert a.sum_probabilities(roller) == b.sum_probabilities(roller)
    assert a.card_probabilities(0)[(1, 2)] == 0 and b.card_probabilities(0)[(1, 2)] > 0


def test_all_zero_weights_fall_back_to_uniform_cards():
    counts = sum_counts(FULL_DECK[:20])
    assert distribution(counts, {k: F(0) for k in SUMS}) == {k: F(counts[k], 20) for k in SUMS}
    # Unreachable in play: the most adversarial deck a draw sees (14 cards on three sums, the
    # 6 thrice in the window, the 8 twice, the 7 multiplied by zero) keeps the 8.
    deck = [c for c in FULL_DECK if card_sum(c) in (6, 7)] + [(2, 6), (3, 5), (4, 4)]
    state = BalancedDice(sevens=(4, 0), deck=deck, recent=(6, 6, 6, 8, 8), streak_player=0, streak_count=1)
    assert state.sum_probabilities(0) == {k: F(int(k == 8)) for k in SUMS}


def test_apply_refuses_a_card_not_in_the_deck():
    state = BalancedDice(deck=deck_without((3, 4)))
    with pytest.raises(ValueError):
        state.apply((3, 4), 0)
    with pytest.raises(ValueError):
        BalancedDice(recent=(1,))
    with pytest.raises(ValueError):
        BalancedDice(streak_player=0)


def openhex_rolls(record):
    """(roller seat, card) of every roll of an OpenHex game record, seats in turn order."""
    seats = list(record["turnOrder"])
    return [(seats.index(e["event"]["player"]), (e["event"]["die1"], e["event"]["die2"]))
            for e in sorted(record["events"], key=lambda e: e["seq"]) if e["event"]["type"] == "diceRolled"]


def first_impossible_roll(record):
    """Replay a balanced game's rolls through the reference: the index of the first roll it
    gives probability zero, or None if every roll was possible."""
    state = BalancedDice.new(len(record["turnOrder"]), record["config"]["numPlayers"])
    for i, (roller, card) in enumerate(openhex_rolls(record)):
        if state.card_probabilities(roller)[card] == 0:
            return i
        state = state.apply(card, roller)
    return None


@pytest.mark.parametrize("name, index", [("game-004a8250.json", 10), ("game-881a71bc.json", 14)])
def test_the_recorded_openhex_games_rolled_fair_dice(name, index):
    """The two recorded 1v1 games (fixtures added 2026-10-17 and -19) say diceMode balanced,
    but each draws a card its deck had already given (004a8250 draws (4, 3) twice running,
    881a71bc's 15th roll repeats its first, (6, 5)), and 881a71bc rolls 9, 9, 9, 9 (the
    recency penalty forbids a third 9 in five): they were played while OpenHex lost the dice
    state between actions (its json_codec: a state that went through JSON "came back rolling
    independent dice"), on fair dice. They cannot validate this reference, and a replay that
    refuses impossible balanced rolls must not be fed them as balanced games. The seats still
    alternate from turnOrder[0]."""
    record = json.loads((FIXTURES / name).read_text())
    assert record["config"]["diceMode"] == "balanced" and record["config"]["numPlayers"] == 2
    rolls = openhex_rolls(record)
    assert [roller for roller, _ in rolls] == [i % 2 for i in range(len(rolls))]
    assert first_impossible_roll(record) == index
