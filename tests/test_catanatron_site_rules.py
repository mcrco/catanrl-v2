"""Catanatron's OpenHex rules (``robber_mode="friendly_public"``,
``dice_mode="balanced"``: what catanatron-server's 1v1 rooms can play) against
cppanatron's (``friendly_robber`` and ``DiceMode.BALANCED``), which they must
equal exactly.

Random games run in lockstep on both engines (cppanatron forced to
Catanatron's dice, cards and steals): at every step both offer the same
actions, robber moves included (the shields read public points only, and a
knight's tile the points before its Largest Army award), both hold the same
balanced dice with the same integer weights for every roller, and both mark
the same seat a knight took Largest Army from. Random roll histories, and
random dice states no game need reach, give both engines' ``BalancedDice``
the same weights too.
"""

from __future__ import annotations

import dataclasses
import random
from collections import Counter

import cppanatron as cpp
import pytest
from catanatron.game import TURNS_LIMIT, Game
from catanatron.models.decks import Supply
from catanatron.models.dice import CARDS, BalancedDice, card_bit
from catanatron.models.enums import ActionPrompt, ActionType
from catanatron.models.player import Color, SimplePlayer
from cppanatron.catanatron import catanatron_map_from_spec, from_catanatron_action

from catanrl import games


def _engines(config: games.GameConfig, index: int):
    """(Catanatron, cppanatron) games of ``config`` on the same board and seating."""
    spec = config.map_spec(1000 + index)
    players = [SimplePlayer(c) for c in list(Color)[: config.num_players]]
    truth = Game(players, seed=index, catan_map=catanatron_map_from_spec(spec),
                 vps_to_win=config.victory_points_to_win, discard_limit=config.discard_limit,
                 robber_mode="friendly_public", dice_mode=config.dice_mode,
                 supply=Supply(resource_bank=config.resource_bank, development_deck=config.development_deck))
    native = cpp.Game([cpp.Color[c.value] for c in truth.state.colors],
                      map=cpp.CatanMap.from_spec(cpp.MapSpec.from_dict(spec)),
                      rules=config.cppanatron_rules(), seed=index)
    return truth, native


def _dice(truth) -> tuple:
    d = truth.state.balanced_dice
    return (tuple(card for card in CARDS if d.deck >> card_bit(card) & 1), tuple(d.recent), tuple(d.sevens),
            d.streak_player, d.streak_count)


def _native_dice(native) -> tuple:
    d = native.balanced_dice
    return tuple(sorted(d.deck)), tuple(d.recent), tuple(d.sevens), d.streak_player, d.streak_count


def _post_award_tiles(truth) -> set:
    """The robber tiles under the points after the knight's award (Catanatron's
    own order): where it differs, the order of the knight and its award mattered."""
    state = truth.state
    taken, state.largest_army_taken_from = state.largest_army_taken_from, None
    try:
        from catanatron.models.actions import robber_possibilities

        return {a.value[0] for a in robber_possibilities(state, state.current_color())}
    finally:
        state.largest_army_taken_from = taken


def _lockstep(config: games.GameConfig, index: int, stats: Counter) -> None:
    truth, native = _engines(config, index)
    rng = random.Random(index)
    seats = config.num_players
    while truth.winning_color() is None and truth.state.num_turns < TURNS_LIMIT:
        ours = sorted(repr(from_catanatron_action(a)) for a in truth.playable_actions)
        assert ours == sorted(map(repr, native.playable_actions)), (config.name, index)
        taken = truth.state.largest_army_taken_from
        assert native.largest_army_taken_from == (None if taken is None else truth.state.colors.index(taken))
        if truth.state.current_prompt == ActionPrompt.MOVE_ROBBER:
            stats["robber_prompts"] += 1
            tiles = {a.value[0] for a in truth.playable_actions}
            stats["shielded"] += len(tiles) < len(truth.state.board.map.land_tiles) - 1
            if taken is not None:
                stats["after_an_award"] += 1
                stats["the_order_mattered"] += _post_award_tiles(truth) != tiles
        if config.dice_mode == "balanced":
            assert _dice(truth) == _native_dice(native)
            for roller in [None, *range(seats)]:
                assert tuple(truth.state.balanced_dice.sum_weights(roller)) == \
                    tuple(native.balanced_dice.sum_weights(roller))
            assert tuple(truth.state.balanced_dice.sum_probabilities(native.next_roller)) == \
                pytest.approx(native.roll_probabilities(native.next_roller), abs=1e-15)
        actions = list(truth.playable_actions)
        busy = [a for a in actions if a.action_type not in (ActionType.END_TURN, ActionType.MARITIME_TRADE)]
        action = rng.choice(busy) if busy and rng.random() < 0.8 else rng.choice(actions)
        record = truth.execute(action)
        forced = {}
        if action.action_type == ActionType.ROLL:
            forced["dice"] = tuple(record.result)
            stats["rolls"] += 1
        elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
            forced["development_card"] = cpp.DevelopmentCard[record.result]
        elif action.action_type == ActionType.MOVE_ROBBER and record.result is not None:
            forced["stolen_resource"] = cpp.Resource[record.result]
        native.execute(from_catanatron_action(action), **forced)
    assert native.winning_color == (None if truth.winning_color() is None else cpp.Color[truth.winning_color().value])
    stats["games"] += 1
    stats["won"] += truth.winning_color() is not None


def test_openhex_1v1_plays_alike_in_both_engines():
    """base-2p-fr (OpenHex's 1v1): 60 random games (about 30,000 rolls and 5,500 robber prompts)."""
    config = games.load_game("base-2p-fr")
    assert config.friendly_robber and config.dice_mode == "balanced"
    stats: Counter = Counter()
    for index in range(60):
        _lockstep(config, index, stats)
    assert stats["rolls"] > 20_000 and stats["robber_prompts"] > 3000 and stats["shielded"] > 1000
    # Knights that took Largest Army, some where the points after the award would have shielded otherwise.
    assert stats["after_an_award"] > 10 and stats["the_order_mattered"] > 3, stats


@pytest.mark.parametrize("name, dice_mode", [("base-3p", "balanced"), ("base-4p", "random")])
def test_the_friendly_robber_plays_alike_with_more_seats(name, dice_mode):
    """Several opponents, each shielding its own tiles; balanced dice with three seats."""
    config = dataclasses.replace(games.load_game(name), friendly_robber=True, dice_mode=dice_mode)
    stats: Counter = Counter()
    for index in range(40):
        _lockstep(config, index, stats)
    assert stats["robber_prompts"] > 500 and stats["shielded"] > 50


def _random_history(rng: random.Random, seats: int, rolls: int):
    """(Catanatron, cppanatron) dice after the same random rolls, each a card
    the deck holds (sevens kept likely, so the streak and imbalance terms move)."""
    ours, theirs = BalancedDice.fresh(seats), cpp.BalancedDice.fresh(seats)
    roller = 0
    for _ in range(rolls):
        cards = [c for c in CARDS if ours.can_draw(c)]
        sevens = [c for c in cards if sum(c) == 7]
        card = rng.choice(sevens if sevens and rng.random() < 0.3 else cards)
        who = rng.choice([roller, None]) if rng.random() < 0.05 else roller
        ours.apply(card, who)
        theirs = theirs.apply(card, who)
        if rng.random() < 0.7 or sum(card) != 7:
            roller = (roller + 1) % seats
    return ours, theirs


def test_balanced_dice_weights_agree_after_the_same_rolls():
    rng = random.Random(7)
    checked = 0
    for seats in range(1, 7):
        for _ in range(40):
            ours, theirs = _random_history(rng, seats, rng.randrange(0, 80))
            assert ours.remaining == theirs.remaining and ours.counts() == list(theirs.counts)
            assert ours.recent == list(theirs.recent) and ours.sevens == list(theirs.sevens)
            assert (ours.streak_player, ours.streak_count) == (theirs.streak_player, theirs.streak_count)
            for roller in [None, *range(seats)]:
                assert ours.seven_multiplier(roller) == tuple(theirs.seven_multiplier(roller))
                assert ours.sum_weights(roller) == list(theirs.sum_weights(roller))
                checked += 1
            assert all(ours.can_draw(c) == theirs.can_draw(c) for c in CARDS)
    assert checked > 1000


def test_balanced_dice_weights_agree_on_arbitrary_states():
    """States no game need reach (any deck, window, sevens and streak)."""
    rng = random.Random(11)
    for _ in range(3000):
        seats = rng.randrange(1, 7)
        cards = rng.sample(CARDS, rng.randrange(0, 37))
        recent = [rng.randrange(2, 13) for _ in range(rng.randrange(0, 6))]
        sevens = [rng.randrange(0, 6) for _ in range(seats)]
        player = rng.choice([None, *range(seats)])
        count = 0 if player is None else rng.randrange(1, 5)
        ours = BalancedDice(seats, sum(1 << card_bit(c) for c in cards), recent, sevens, player, count)
        theirs = cpp.BalancedDice(sevens=sevens, deck=cards, recent=recent, streak_player=player,
                                  streak_count=count)
        for roller in [None, *range(seats)]:
            assert ours.sum_weights(roller) == list(theirs.sum_weights(roller))


def test_the_catanatron_players_mirror_follows_a_site_game(monkeypatch):
    """``CatanrlPlayer`` (POLICY/MCTS/BELIEF seated in a Catanatron game) mirrors the game on
    cppanatron with ``friendly_robber`` from the state, which is True under ``friendly_public``:
    its legal moves stay the game's, robber moves included. (Its mirror rolls fair dice, the
    rolls forced to the game's: only an MCTS seat's chance odds would differ.)"""
    from types import SimpleNamespace

    from catanrl.agents.spec import CatanrlPlayer

    monkeypatch.setattr(CatanrlPlayer, "_fit", staticmethod(lambda agent, native: None))
    config = games.load_game("base-2p-fr")
    stats: Counter = Counter()
    for index in range(5):
        truth, _ = _engines(config, index)
        players = {color: CatanrlPlayer(color) for color in truth.state.colors}
        agent = SimpleNamespace(needs_trackers=False)
        rng = random.Random(index)
        while truth.winning_color() is None and truth.state.num_turns < TURNS_LIMIT:
            mirror = players[truth.state.current_color()]._sync(truth, agent)["game"]
            assert mirror.rules.friendly_robber
            assert sorted(repr(from_catanatron_action(a)) for a in truth.playable_actions) == \
                sorted(map(repr, mirror.playable_actions))
            stats["robber_prompts"] += truth.state.current_prompt == ActionPrompt.MOVE_ROBBER
            truth.execute(rng.choice(list(truth.playable_actions)))
    assert stats["robber_prompts"] > 50
