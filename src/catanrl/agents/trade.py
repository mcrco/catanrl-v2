"""Trade primitives: the bundles a seat can offer and the engine routes that
build the rows a critic-driven trade harness reads.

Domestic trades stay outside the learned action space (``GraphActionSpace``
never lists them, DESIGN 1.2). A harness agent plays them through
``cppanatron.Game`` copies instead and reads its critic on the mover's
post-trade PLAY_TURN row, the row the critic trained on. Everything here is
pure Python over game copies:

* the enumeration: :func:`enumerate_trades` (the oracle's, both hands known)
  and :func:`mover_menu` (the fair menu, which reads nothing of the partner's
  hand but its public size);
* the engine routes: :func:`apply_trade` with the true hands,
  :func:`post_row_clone` with a size-preserving stand-in partner hand (the
  proposer's read), :func:`respond_route` from a responder's prompt on a
  dealt world, :func:`rewind` back to the pre-offer state;
* the trade entries of ``playable_actions`` (:func:`offer_action`,
  :func:`response`, :func:`confirm_action`, :func:`cancel_action`,
  :func:`default_trade_action`) and what they carry (:func:`offer_of`,
  :func:`acceptees_of`): ``Game`` has no accessor for the offer in progress,
  so the live trade is read from those entries' values.

``tools/trade_oracle.py`` imports the enumeration and :func:`apply_trade`
from here, so its records keep their meaning (same enumeration order, same
engine route). This module is hashed outcome code (``agents/`` is in
``catanrl.provenance.OUTCOME``): a change to the enumeration moves every new
eval id.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import cppanatron as cpp

#: Resource kinds in the engine's hand order.
RESOURCES = ("wood", "brick", "sheep", "wheat", "ore")
#: (cards given, cards received) of the trades enumerated.
SHAPES = ((1, 1), (2, 1), (1, 2))
SHAPE_NAMES = tuple(f"{give}:{get}" for give, get in SHAPES)
Counts = tuple[int, int, int, int, int]


# ---------------------------------------------------------------- trades


@dataclass(frozen=True)
class Trade:
    """What the mover gives and gets, as counts per resource kind."""

    give: Counts
    get: Counts

    @property
    def shape(self) -> str:
        return f"{sum(self.give)}:{sum(self.get)}"

    @property
    def label(self) -> str:
        return f"{_side(self.give)}>{_side(self.get)}"


def _side(counts: Sequence[int]) -> str:
    return "+".join(f"{count if count > 1 else ''}{name}" for name, count in zip(RESOURCES, counts) if count)


def _bundles(size: int) -> list[Counts]:
    """Every multiset of ``size`` cards, as counts (5 singles, 15 pairs)."""
    out = []
    for combo in itertools.combinations_with_replacement(range(len(RESOURCES)), size):
        counts = [0] * len(RESOURCES)
        for kind in combo:
            counts[kind] += 1
        out.append(tuple(counts))
    return out


def can_pay(hand: Sequence[int], counts: Sequence[int]) -> bool:
    return all(have >= need for have, need in zip(hand, counts))


def enumerate_trades(mover_hand: Sequence[int], partner_hand: Sequence[int]) -> list[Trade]:
    """Every 1:1, 2:1 and 1:2 trade both hands can pay, give and get disjoint in kind
    (giving wood and brick for wood is a gift of brick, not a trade): at most 120."""
    out = []
    for give_size, get_size in SHAPES:
        for give in _bundles(give_size):
            if not can_pay(mover_hand, give):
                continue
            for get in _bundles(get_size):
                if can_pay(partner_hand, get) and not any(a and b for a, b in zip(give, get)):
                    out.append(Trade(give, get))
    return out


def mover_menu(hand: Sequence[int], partner_size: int,
               shapes: Sequence[tuple[int, int]] = SHAPES) -> list[Trade]:
    """Every trade of ``shapes`` the mover can pay, give and get disjoint in kind, whose
    ask fits in the partner's public hand size: the fair menu, which knows nothing of
    the partner's hand but its size. The order is :func:`enumerate_trades`'s (shape,
    give bundle, get bundle), of which the menu is a superset for the same hands."""
    out = []
    for give_size, get_size in shapes:
        if get_size > partner_size:
            continue
        for give in _bundles(give_size):
            if not can_pay(hand, give):
                continue
            for get in _bundles(get_size):
                if not any(a and b for a, b in zip(give, get)):
                    out.append(Trade(give, get))
    return out


def standin_hand(size: int, ask: Sequence[int]) -> Counts:
    """A hand of ``size`` cards holding ``ask``: the ask plus filler of the first kind.
    At private level the mover's row shows a partner's hand as its size only, so any
    hand of the public size serves for the mover's post-trade row."""
    filler = size - sum(ask)
    if filler < 0:
        raise ValueError(f"a hand of {size} cards cannot hold {_side(ask)}")
    hand = list(ask)
    hand[0] += filler
    return tuple(hand)


# ---------------------------------------------------------------- the trade entries of playable_actions


def _entries(game: cpp.Game, kind: cpp.ActionType) -> list[cpp.Action]:
    return [a for a in game.playable_actions if a.action_type == kind]


def _entry(game: cpp.Game, kind: cpp.ActionType) -> cpp.Action:
    entries = _entries(game, kind)
    if not entries:
        raise ValueError(f"{kind.name} is not playable at {game.current_prompt.name}")
    return entries[0]


def offer_action(game: cpp.Game, seat: int, trade: Trade) -> cpp.Action:
    """``seat``'s OFFER_TRADE of ``trade`` (ten counts: offered, then asked). The engine
    takes it at the seat's own PLAY_TURN after the roll without listing it."""
    return cpp.Action(game.colors[seat], cpp.ActionType.OFFER_TRADE, (*trade.give, *trade.get))


def response(game: cpp.Game, accept: bool) -> cpp.Action:
    """The ACCEPT_TRADE (``accept``) or REJECT_TRADE entry of the responder's DECIDE_TRADE.
    ACCEPT is listed only when the responder can pay the asking."""
    return _entry(game, cpp.ActionType.ACCEPT_TRADE if accept else cpp.ActionType.REJECT_TRADE)


def confirm_action(game: cpp.Game, partner: int | cpp.Color) -> cpp.Action:
    """The CONFIRM_TRADE entry naming ``partner`` (a seat or its Color) at the proposer's
    DECIDE_ACCEPTEES; a confirm's value ends with the acceptee's Color."""
    color = partner if isinstance(partner, cpp.Color) else game.colors[partner]
    for action in _entries(game, cpp.ActionType.CONFIRM_TRADE):
        if action.value[-1] == color:
            return action
    raise ValueError(f"no confirm names {color.name} at {game.current_prompt.name}")


def cancel_action(game: cpp.Game) -> cpp.Action:
    """The CANCEL_TRADE entry at the proposer's DECIDE_ACCEPTEES."""
    return _entry(game, cpp.ActionType.CANCEL_TRADE)


def default_trade_action(game: cpp.Game) -> cpp.Action:
    """What a seat with no trade judgement plays: REJECT at DECIDE_TRADE, CANCEL at
    DECIDE_ACCEPTEES (the first listed entry, as the catanatron bridge answers)."""
    if game.current_prompt == cpp.ActionPrompt.DECIDE_TRADE:
        return response(game, accept=False)
    if game.current_prompt == cpp.ActionPrompt.DECIDE_ACCEPTEES:
        return cancel_action(game)
    raise ValueError(f"{game.current_prompt.name} is not a trade prompt")


def offer_of(game: cpp.Game) -> tuple[Counts, Counts, int]:
    """The offer in progress as ``(give, get, proposer seat)``, from the proposer's side.
    At DECIDE_TRADE the REJECT entry carries it (ten counts and the offering seat); at
    DECIDE_ACCEPTEES a CONFIRM entry does (ten counts and an acceptee) and the proposer
    is the seat to move."""
    if game.current_prompt == cpp.ActionPrompt.DECIDE_TRADE:
        value = response(game, accept=False).value
        proposer = int(value[10])
    elif game.current_prompt == cpp.ActionPrompt.DECIDE_ACCEPTEES:
        value = _entry(game, cpp.ActionType.CONFIRM_TRADE).value
        proposer = game.current_player_index
    else:
        raise ValueError(f"no offer is open at {game.current_prompt.name}")
    return tuple(int(c) for c in value[:5]), tuple(int(c) for c in value[5:10]), proposer


def acceptees_of(game: cpp.Game) -> list[int]:
    """The seats that accepted the offer, in seat order: the CONFIRM entries' colours at
    the proposer's DECIDE_ACCEPTEES (reached only when somebody accepted)."""
    if game.current_prompt != cpp.ActionPrompt.DECIDE_ACCEPTEES:
        raise ValueError(f"no acceptees at {game.current_prompt.name}")
    colors = list(game.colors)
    return [colors.index(action.value[-1]) for action in _entries(game, cpp.ActionType.CONFIRM_TRADE)]


# ---------------------------------------------------------------- engine routes


def tradeable(game: cpp.Game, seat: int) -> bool:
    """The moments a domestic trade could be proposed: the seat's own PLAY_TURN
    decisions after the roll (not while placing a road building card's roads)."""
    return (game.current_player_index == seat and game.current_turn_index == seat
            and game.current_prompt == cpp.ActionPrompt.PLAY_TURN and game.player(seat).has_rolled
            and game.free_roads_available == 0 and game.winning_color is None)


def apply_trade(game: cpp.Game, mover: int, partner: int, trade: Trade) -> cpp.Game:
    """A copy of ``game`` after ``trade``, through the engine's own actions: the mover
    offers, the partner accepts, every other seat rejects, the mover confirms the
    partner. The engine checks an offer's form, not the offerer's hand, and a
    confirmation debits blindly, so both hands are checked here."""
    if not tradeable(game, mover):
        raise ValueError("a trade is proposed by the seat to move, during its turn, after the roll")
    if mover == partner:
        raise ValueError("a seat cannot trade with itself")
    if not can_pay(game.players[mover].resources, trade.give):
        raise ValueError(f"the mover cannot pay {trade.label}")
    if not can_pay(game.players[partner].resources, trade.get):
        raise ValueError(f"the partner cannot pay {trade.label}")
    post = game.copy()
    colors = list(post.colors)
    post.execute(cpp.Action(colors[mover], cpp.ActionType.OFFER_TRADE, (*trade.give, *trade.get)))
    while post.current_prompt == cpp.ActionPrompt.DECIDE_TRADE:
        kind = cpp.ActionType.ACCEPT_TRADE if post.current_player_index == partner else cpp.ActionType.REJECT_TRADE
        post.execute(next(a for a in post.playable_actions if a.action_type == kind))
    confirms = [a for a in post.playable_actions if a.action_type == cpp.ActionType.CONFIRM_TRADE]
    if len(confirms) != 1 or confirms[0].value[-1] != colors[partner]:
        raise RuntimeError(f"the engine did not offer to confirm {trade.label} with seat {partner}")
    post.execute(confirms[0])
    return post


def trade_by_hands(game: cpp.Game, mover: int, partner: int, trade: Trade) -> cpp.Game:
    """The reference: the same trade as two hand overwrites (``Game.set_resources``
    leaves the bank alone, as a domestic trade does). Tests compare the routes."""
    post = game.copy()
    for seat, out, inn in ((mover, trade.give, trade.get), (partner, trade.get, trade.give)):
        hand = [have - gone + got for have, gone, got in zip(post.players[seat].resources, out, inn)]
        post.set_resources(seat, hand)
    return post


def post_row_clone(game: cpp.Game, mover: int, partner: int, trade: Trade) -> cpp.Game:
    """The mover's post-trade state for a critic read, with the partner's hand replaced
    by a stand-in of its public size that holds the ask (bank untouched) before the
    engine route of :func:`apply_trade`. Applied unconditionally: whether the partner
    can really pay is never read, so the row is a function of public state and the
    mover's own hand (at private level, bit for bit the true post-trade row when the
    partner can pay)."""
    clone = game.copy()
    clone.set_resources(partner, standin_hand(sum(game.player(partner).resources), trade.get))
    return apply_trade(clone, mover, partner, trade)


def rewind(game: cpp.Game) -> cpp.Game:
    """A copy of ``game`` with its negotiation undone: every remaining responder rejects
    and the proposer cancels whatever was accepted. No hand moves and no trade handler
    touches the RNG, so the result is ``search_equivalent`` to the pre-offer game, the
    proposer's PLAY_TURN after the roll; a game not mid-negotiation comes back as is."""
    pre = game.copy()
    while pre.current_prompt == cpp.ActionPrompt.DECIDE_TRADE:
        pre.execute(response(pre, accept=False))
    if pre.current_prompt == cpp.ActionPrompt.DECIDE_ACCEPTEES:
        pre.execute(cancel_action(pre))
    return pre


def respond_route(world: cpp.Game, proposer: int, responder: int, trade: Trade) -> cpp.Game:
    """The responder's view of accepting: ``world`` (a game or a dealt world at any point
    of the negotiation) rewound to the pre-offer state, then the engine route with the
    responder accepting and everyone else rejecting, confirmed on the responder. Ends at
    the proposer's PLAY_TURN with the two hands swapped."""
    return apply_trade(rewind(world), proposer, responder, trade)
