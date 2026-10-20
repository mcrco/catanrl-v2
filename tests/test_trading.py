"""The trade primitives (catanrl.agents.trade): the mover-side menu, the stand-in
partner hand, the engine routes a TRADE agent reads its critic on (the proposer's
post-row clone, the responder's route on a dealt world, the rewind to the
pre-offer state), the trade entries of ``playable_actions``, and the fact the
proposer's shortcut rests on: a private row shows a partner's hand as a size."""

from __future__ import annotations

import cppanatron as cpp
import numpy as np
import pytest

from catanrl import checkpoint as ck
from catanrl import games
from catanrl.agents.trade import (
    SHAPES,
    Trade,
    acceptees_of,
    apply_trade,
    can_pay,
    cancel_action,
    confirm_action,
    default_trade_action,
    enumerate_trades,
    mover_menu,
    offer_action,
    offer_of,
    post_row_clone,
    respond_route,
    response,
    rewind,
    standin_hand,
    trade_by_hands,
    tradeable,
)
from catanrl.envs.encoding import dims_for, make_encoder

MOVER_HAND = [2, 1, 0, 0, 0]  # wood, brick, sheep, wheat, ore
PARTNER_HAND = [0, 0, 1, 2, 0]
WOOD_FOR_SHEEP = Trade((1, 0, 0, 0, 0), (0, 0, 1, 0, 0))
WOOD_BRICK_FOR_WHEAT = Trade((1, 1, 0, 0, 0), (0, 0, 0, 1, 0))
DECIDE_TRADE = cpp.ActionPrompt.DECIDE_TRADE
DECIDE_ACCEPTEES = cpp.ActionPrompt.DECIDE_ACCEPTEES
PLAY_TURN = cpp.ActionPrompt.PLAY_TURN


def _packaged():
    """tests/test_trade_oracle.py's 4p position, the mover's PLAY_TURN after the roll:
    the mover (seat 3, so the engine prompts it on its own offer, last) holds
    MOVER_HAND and every other seat PARTNER_HAND. Returns it with its encoder."""
    pos = next(p for p in ck.packaged_positions() if p["game"]["num_players"] == 4
               and p["options"]["prompt"] == "PLAY_TURN" and p["options"]["has_rolled"])
    game = ck._position_game(pos)
    mover = game.current_player_index
    assert mover == 3
    game.set_resources(mover, MOVER_HAND)
    for seat in range(4):
        if seat != mover:
            game.set_resources(seat, PARTNER_HAND)
    return game, mover, make_encoder(dims_for([pos["game"]]), "private")


def _played(count=6, seed=0):
    """Random-play 4p games stopped at a tradeable moment with the mover holding at
    least two cards, every seat's hand tracker fed from the start (the pattern of
    tests/test_search_invariance.py). Returns (game, mover, trackers) triples."""
    config = games.load_game("base-4p")
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        game = config.cppanatron_game(map_seed=i, seed=100 + i)
        trackers = [cpp.HandTracker(game, seat) for seat in range(4)]
        warmup = int(rng.integers(40, 200))
        for step in range(2000):
            mover = game.current_player_index
            if step >= warmup and tradeable(game, mover) and sum(game.player(mover).resources) >= 2:
                out.append((game, mover, trackers))
                break
            actions = game.playable_actions
            if game.winning_color is not None or not actions:
                break
            action = actions[int(rng.integers(len(actions)))]
            before = game.copy()
            game.execute(action)
            for tracker in trackers:
                tracker.observe(before, action, game)
    assert len(out) == count
    return out


ENCODER = make_encoder(dims_for(["base-4p"]), "private")


def _payable_trade(game, mover):
    """The first 1:1 trade of the mover's menu some partner can pay, with that partner."""
    hands = [tuple(p.resources) for p in game.players]
    for partner in range(game.num_players):
        if partner == mover:
            continue
        for trade in mover_menu(hands[mover], sum(hands[partner]), shapes=((1, 1),)):
            if can_pay(hands[partner], trade.get):
                return partner, trade
    raise AssertionError("no partner can pay any 1:1 trade")


def _at_prompt(game, mover, trade, responder, accepted=()):
    """A clone mid-negotiation at ``responder``'s DECIDE_TRADE, the seats in ``accepted``
    (asked before it) having accepted and the others rejected."""
    g = game.copy()
    g.execute(offer_action(game, mover, trade))
    while g.current_player_index != responder:
        g.execute(response(g, accept=g.current_player_index in accepted))
    assert g.current_prompt == DECIDE_TRADE
    return g


# ---------------------------------------------------------------- menu and stand-in


def test_mover_menu_counts_per_shape_and_prunes_by_partner_size():
    menu = mover_menu(MOVER_HAND, 10)
    # 1:1 wood|brick > 4 other kinds; 2:1 {wood,wood} > 4, {wood,brick} > 3; 1:2 wood|brick > the 10 pairs of 4 kinds
    assert [sum(1 for t in menu if t.shape == s) for s in ("1:1", "2:1", "1:2")] == [8, 7, 20]
    assert [t.shape for t in menu] == ["1:1"] * 8 + ["2:1"] * 7 + ["1:2"] * 20  # shape-major, as enumerate_trades
    for t in menu:
        assert can_pay(MOVER_HAND, t.give) and not any(a and b for a, b in zip(t.give, t.get))
    assert {t.label for t in menu if t.shape == "2:1"} >= {"2wood>sheep", "wood+brick>wheat"}
    # The ask must fit in the partner's public hand: 1:2 goes at size 1, everything at size 0.
    assert [t.shape for t in mover_menu(MOVER_HAND, 1)] == ["1:1"] * 8 + ["2:1"] * 7
    assert mover_menu(MOVER_HAND, 0) == [] and mover_menu([0] * 5, 10) == []
    assert mover_menu(MOVER_HAND, 10, shapes=((2, 1),)) == [t for t in menu if t.shape == "2:1"]
    assert mover_menu(MOVER_HAND, 10, shapes=SHAPES) == menu


def test_mover_menu_is_enumerate_trades_without_the_partner_hand_filter():
    """The oracle's list (both hands known) is the menu filtered by the partner's true
    hand, in the same order: the harness enumerates a superset that needs no hand."""
    menu = mover_menu(MOVER_HAND, sum(PARTNER_HAND))
    assert [t for t in menu if can_pay(PARTNER_HAND, t.get)] == enumerate_trades(MOVER_HAND, PARTNER_HAND)
    assert len(menu) == 15 + 20 and len(enumerate_trades(MOVER_HAND, PARTNER_HAND)) == 12
    assert mover_menu([2] * 5, 2) == enumerate_trades([2] * 5, [2] * 5)  # 120: a hand of twos pays any ask


def test_standin_hand_holds_the_ask_at_the_public_size():
    ask = (0, 0, 1, 2, 0)
    assert standin_hand(5, ask) == (2, 0, 1, 2, 0) and standin_hand(3, ask) == ask
    assert standin_hand(0, (0,) * 5) == (0,) * 5
    for size in (5, 3):
        hand = standin_hand(size, ask)
        assert sum(hand) == size and can_pay(hand, ask)
    with pytest.raises(ValueError):
        standin_hand(2, ask)


# ---------------------------------------------------------------- the proposer's read


@pytest.mark.parametrize("rank", [0, 1, 2], ids=["partner-first", "partner-middle", "partner-last"])
def test_post_row_clone_is_the_true_hand_route_when_the_partner_can_pay(rank):
    game, mover, encoder = _packaged()
    partner = [seat for seat in range(4) if seat != mover][rank]
    before = [tuple(p.resources) for p in game.players]
    for trade in (WOOD_FOR_SHEEP, WOOD_BRICK_FOR_WHEAT):
        clone = post_row_clone(game, mover, partner, trade)
        true = apply_trade(game, mover, partner, trade)
        assert np.array_equal(encoder.encode(clone, mover), encoder.encode(true, mover))
        # The clone is the stand-in hand traded by two overwrites: the same game, RNG and all.
        standin = game.copy()
        standin.set_resources(partner, standin_hand(sum(PARTNER_HAND), trade.get))
        assert clone.search_equivalent(trade_by_hands(standin, mover, partner, trade))
        assert (clone.current_prompt, clone.current_player_index) == (PLAY_TURN, mover)
        assert list(clone.players[mover].resources) == [h - g + r for h, g, r in zip(MOVER_HAND, trade.give, trade.get)]
        assert sum(clone.players[partner].resources) == sum(PARTNER_HAND) - sum(trade.get) + sum(trade.give)
        assert list(clone.resource_bank) == list(game.resource_bank)
        for seat in range(4):
            if seat not in (mover, partner):
                assert tuple(clone.players[seat].resources) == before[seat]
    assert [tuple(p.resources) for p in game.players] == before  # the original is untouched


def test_post_row_clone_never_reads_the_partner_hand():
    """Two true partner hands of one size give the same clone, whether or not the
    partner can pay; the ask only has to fit the public size."""
    game, mover, _ = _packaged()
    partner = (mover + 1) % 4
    cannot_pay = game.copy()
    cannot_pay.set_resources(partner, [3, 0, 0, 0, 0])  # no sheep, no wheat; still 3 cards
    for trade in (WOOD_FOR_SHEEP, WOOD_BRICK_FOR_WHEAT):
        with pytest.raises(ValueError):
            apply_trade(cannot_pay, mover, partner, trade)
        assert post_row_clone(cannot_pay, mover, partner, trade).search_equivalent(
            post_row_clone(game, mover, partner, trade))
    too_small = game.copy()
    too_small.set_resources(partner, [0, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        post_row_clone(too_small, mover, partner, WOOD_FOR_SHEEP)
    with pytest.raises(ValueError):
        post_row_clone(game, mover, mover, WOOD_FOR_SHEEP)


def test_post_row_clone_matches_the_true_route_over_random_play_menus():
    for game, mover, _ in _played():
        hands = [tuple(p.resources) for p in game.players]
        for partner in range(4):
            if partner == mover:
                continue
            for trade in mover_menu(hands[mover], sum(hands[partner])):
                clone = post_row_clone(game, mover, partner, trade)
                if can_pay(hands[partner], trade.get):
                    assert np.array_equal(ENCODER.encode(clone, mover),
                                          ENCODER.encode(apply_trade(game, mover, partner, trade), mover))
                assert (clone.current_prompt, clone.current_player_index) == (PLAY_TURN, mover)


# ---------------------------------------------------------------- rewind and the responder's route


def test_rewind_is_the_pre_offer_game_from_every_prompt():
    game, mover, _ = _packaged()
    others = [seat for seat in range(4) if seat != mover]
    g = game.copy()
    g.execute(offer_action(game, mover, WOOD_FOR_SHEEP))
    prompts = []
    while g.current_prompt == DECIDE_TRADE:
        prompts.append(g.current_player_index)
        assert rewind(g).search_equivalent(game)
        assert g.current_prompt == DECIDE_TRADE  # rewind works on a copy
        g.execute(response(g, accept=g.current_player_index == others[0]))
    assert prompts == [0, 1, 2, 3]  # the engine asks a proposer at seat > 0 about its own offer
    assert g.current_prompt == DECIDE_ACCEPTEES and acceptees_of(g) == [others[0]]
    assert rewind(g).search_equivalent(game)
    assert rewind(game).search_equivalent(game)  # nothing to undo


def test_rewind_of_a_sampled_world_is_a_world_of_the_pre_offer_game():
    """The responder's path: a world dealt from its tracker mid-negotiation (prompt and
    offer kept), rewound, is the world the same seed deals from the pre-offer game,
    since no trade step moves a card or a tracker constraint."""
    for game, mover, trackers in _played(count=4, seed=1):
        partner, trade = _payable_trade(game, mover)
        mid = game.copy()
        mid.execute(offer_action(game, mover, trade))
        while mid.current_prompt == DECIDE_TRADE:
            responder = mid.current_player_index
            for seed in (1, 2, 3):
                world = cpp.sample_world(mid, trackers[responder], seed=seed)
                assert world.current_prompt == DECIDE_TRADE and offer_of(world) == (trade.give, trade.get, mover)
                assert rewind(world).search_equivalent(cpp.sample_world(game, trackers[responder], seed=seed))
            before = mid.copy()
            action = response(mid, accept=responder == partner)
            mid.execute(action)
            for tracker in trackers:
                tracker.observe(before, action, mid)
        assert mid.current_prompt == DECIDE_ACCEPTEES
        world = cpp.sample_world(mid, trackers[mover], seed=4)
        assert rewind(world).search_equivalent(cpp.sample_world(game, trackers[mover], seed=4))


@pytest.mark.parametrize("rank", [0, 1, 2], ids=["responder-first", "responder-middle", "responder-last"])
def test_respond_route_ends_at_play_turn_with_the_two_hands_swapped(rank):
    game, mover, _ = _packaged()
    others = [seat for seat in range(4) if seat != mover]
    responder = others[rank]
    trade = WOOD_FOR_SHEEP
    expected = trade_by_hands(game, mover, responder, trade)
    # From its own prompt, from its prompt after an earlier acceptance, and from the
    # proposer's DECIDE_ACCEPTEES after it accepted: the same post-trade game.
    worlds = [_at_prompt(game, mover, trade, responder)]
    if rank > 0:
        worlds.append(_at_prompt(game, mover, trade, responder, accepted=(others[0],)))
    accepted = worlds[0].copy()
    while accepted.current_prompt == DECIDE_TRADE:
        accepted.execute(response(accepted, accept=accepted.current_player_index == responder))
    assert accepted.current_prompt == DECIDE_ACCEPTEES
    worlds.append(accepted)
    for world in worlds:
        post = respond_route(world, mover, responder, trade)
        assert (post.current_prompt, post.current_player_index) == (PLAY_TURN, mover)
        assert post.search_equivalent(expected)
        assert list(post.players[mover].resources) == [h - g + r for h, g, r in zip(MOVER_HAND, trade.give, trade.get)]
        assert list(post.players[responder].resources) == [h - r + g for h, g, r in zip(PARTNER_HAND, trade.give, trade.get)]
        assert all(tuple(post.players[s].resources) == tuple(PARTNER_HAND) for s in others if s != responder)
        assert list(post.resource_bank) == list(game.resource_bank)


def test_respond_route_on_a_dealt_world_needs_a_proposer_who_can_pay():
    """On a sampled world the responder's own hand is exact and the proposer's is dealt:
    the route holds when the dealt hand covers the offer, and raises otherwise (the
    agent keeps only worlds in which the proposer can pay)."""
    routed = 0
    for game, mover, trackers in _played(count=6, seed=2):
        partner, trade = _payable_trade(game, mover)
        mid = _at_prompt(game, mover, trade, partner)
        for seed in range(6):
            world = cpp.sample_world(mid, trackers[partner], seed=seed)
            assert tuple(world.players[partner].resources) == tuple(game.players[partner].resources)
            pre = rewind(world)
            if can_pay(world.players[mover].resources, trade.give):
                post = respond_route(world, mover, partner, trade)
                assert (post.current_prompt, post.current_player_index) == (PLAY_TURN, mover)
                assert post.search_equivalent(trade_by_hands(pre, mover, partner, trade))
                routed += 1
            else:
                with pytest.raises(ValueError):
                    respond_route(world, mover, partner, trade)
    assert routed > 0


# ---------------------------------------------------------------- the trade entries


def test_trade_entries_name_the_offer_the_responses_and_the_acceptees():
    game, mover, _ = _packaged()
    colors = list(game.colors)
    trade = WOOD_FOR_SHEEP  # every other seat holds a sheep; the mover does not
    offer = offer_action(game, mover, trade)
    assert offer == cpp.Action(colors[mover], cpp.ActionType.OFFER_TRADE, (1, 0, 0, 0, 0, 0, 0, 1, 0, 0))
    assert offer.value == (1, 0, 0, 0, 0, 0, 0, 1, 0, 0)
    for read in (offer_of, acceptees_of, cancel_action, default_trade_action):
        with pytest.raises(ValueError):
            read(game)  # no negotiation at PLAY_TURN
    with pytest.raises(ValueError):
        response(game, accept=False)
    with pytest.raises(ValueError):
        confirm_action(game, 0)

    g = game.copy()
    g.execute(offer)
    while g.current_prompt == DECIDE_TRADE:
        seat = g.current_player_index
        assert offer_of(g) == (trade.give, trade.get, mover)
        reject = response(g, accept=False)
        assert (reject.action_type, reject.color, reject.value) == (
            cpp.ActionType.REJECT_TRADE, colors[seat], (1, 0, 0, 0, 0, 0, 0, 1, 0, 0, mover))
        assert default_trade_action(g) == reject and reject == g.playable_actions[0]
        for read in (acceptees_of, cancel_action):
            with pytest.raises(ValueError):
                read(g)
        with pytest.raises(ValueError):
            confirm_action(g, seat)
        if seat == mover:
            with pytest.raises(ValueError):
                response(g, accept=True)  # the proposer cannot pay its own asking
            g.execute(reject)
            continue
        accept = response(g, accept=True)
        assert (accept.action_type, accept.color, accept.value) == (cpp.ActionType.ACCEPT_TRADE, colors[seat], reject.value)
        g.execute(accept if seat in (0, 2) else reject)

    assert (g.current_prompt, g.current_player_index) == (DECIDE_ACCEPTEES, mover)
    assert acceptees_of(g) == [0, 2] and offer_of(g) == (trade.give, trade.get, mover)
    cancel = cancel_action(g)
    assert (cancel.action_type, cancel.color, cancel.value) == (cpp.ActionType.CANCEL_TRADE, colors[mover], None)
    assert default_trade_action(g) == cancel and cancel == g.playable_actions[0]
    confirm = confirm_action(g, 2)
    assert confirm == confirm_action(g, colors[2]) and confirm in g.playable_actions
    assert (confirm.action_type, confirm.color, confirm.value) == (
        cpp.ActionType.CONFIRM_TRADE, colors[mover], (1, 0, 0, 0, 0, 0, 0, 1, 0, 0, colors[2]))
    with pytest.raises(ValueError):
        confirm_action(g, 1)  # rejected
    with pytest.raises(ValueError):
        response(g, accept=False)
    confirmed = g.copy()
    confirmed.execute(confirm)
    assert confirmed.search_equivalent(trade_by_hands(game, mover, 2, trade))
    cancelled = g.copy()
    cancelled.execute(cancel)
    assert cancelled.search_equivalent(game)


# ---------------------------------------------------------------- the private row and partner hands


def test_private_row_shows_a_partner_hand_as_its_size_only():
    """Fact 3, the proposer's shortcut: the mover's private row is the same for any
    partner hand of one size, and a size change moves hand_size and, across the
    discard limit, over_discard_limit, nothing else."""
    game, mover, encoder = _packaged()
    partner = (mover + 2) % 4
    offset, (_, width) = encoder.layout()["players"]
    names = list(encoder.feature_names("players"))
    slot = (partner - mover) % 4
    feature = {name: offset + slot * width + names.index(name) for name in ("hand_size", "over_discard_limit")}

    def row(hand, level_encoder=encoder):
        g = game.copy()
        g.set_resources(partner, hand)
        return level_encoder.encode(g, mover)

    same_size = row([0, 0, 1, 2, 0])
    assert np.array_equal(same_size, row([3, 0, 0, 0, 0])) and np.array_equal(same_size, row([0, 0, 0, 0, 3]))
    assert np.flatnonzero(same_size != row([1, 1, 1, 1, 0])).tolist() == [feature["hand_size"]]
    assert game.discard_limit == 7
    assert np.flatnonzero(same_size != row([2, 2, 2, 2, 0])).tolist() == sorted(feature.values())
    public = make_encoder(encoder.dims, "public")
    assert not np.array_equal(row([0, 0, 1, 2, 0], public), row([3, 0, 0, 0, 0], public))  # the control
