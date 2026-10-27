"""Replay a seat's view of a Catanatron game on cppanatron.

A view's ``action_records`` is the game's whole log, redacted for the seat:
every action with its dice, the cards the seat bought and the steals it was
party to; other seats' purchases and the steals between them show no card.
:class:`Replay` executes the log on a fresh cppanatron game seated and mapped
like the view (or from an ``apply_setup`` snapshot of an earlier view),
forcing every outcome the log shows. Both engines follow the same rules
(cppanatron's differential tests), so the replay reaches exactly the view's
public state at every prompt, and every transition it passes through is
what the seat's hand tracker learns from (:mod:`catanrl.bridge.tracking`).
The friendly robber: the engine plays OpenHex's rule (public points, so no
placeholder changes the legal robber moves, and a knight's tile judged on
the points before its Largest Army award), which is Catanatron's
``robber_mode="friendly_public"`` (catanatron-server's room config
``robber_mode: friendly_public``) exactly. Catanatron's own friendly robber
(``friendly_robber``, which catanatron-server refuses) counts hidden victory
point cards and awards first, and a game under it may rob a seat the replay
shields (the step then fails like any move the replay cannot follow;
tests/bridge/bridge_games.py re-rules it as OpenHex's for the tests).

The other seats' hidden cards are placeholders. They start as the fresh
game's (empty hands) and change only by public moves and by placeholder
outcomes the seat cannot see: a hidden steal takes a card the victim's
placeholder hand holds, a hidden purchase draws a card the placeholder deck
holds (not a victory point card while another kind is left). When a public
move needs cards a placeholder hand lacks (a build, a discard, a trade, a
played card, a monopoly's exact take), the hidden cards of the other seats
are re-dealt first (``Game.redeal_hidden_cards``) with the fewest changes:
per resource the cards those seats hold together
(public, from the bank) and every hand size stay as they are, and the unseen
development cards stay the deck plus their hands. No placeholder ever comes
from hidden information, and none can make a seat win the replay that has
not won the game.

A roll pays by the engine's rule: nothing of a resource the bank cannot pay
everyone. A game that pays otherwise (OpenHex pays in full unless one
player's share alone exceeds the bank, and its bank goes negative) tells
:meth:`Replay.step` the public payout (``produced``); where the engine paid
nothing for want of bank, the step raises :class:`BankShort` with the cards
short, and the caller replays the game on rules whose bank is that much
larger (the hands and the tracker then stay exact; the engine's bank is the
game's plus the margin). Any other difference (the engine paying where the
game withheld, or a board the replay has wrong) is logged and counted in
``repairs["production"]``; the engine's payout stands, since its bank cannot
be set from here.

Under balanced dice (``GameRules.dice_mode``, OpenHex's 1v1 rules, and
Catanatron's ``dice_mode="balanced"``) every roll replayed updates the game's
public dice state, as the game's own draw did. A roll that state cannot give
(a card its deck no longer holds, or a total with no chance) raises
:class:`DiceRefused` before anything changes: the game did not roll the dice
the replay follows.

Monopoly takes are public (every hand size is visible), but a log entry does
not carry them: :meth:`Replay.step` takes the hand sizes after the entry when
the caller has them (``sizes_after``, from the view after that step) and
otherwise plays the monopoly on the placeholders and marks the transition
``exact_amounts=False``; :meth:`Replay.resize` then moves placeholder cards
to the sizes a later view shows.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cppanatron as cpp
from catanatron.serialization import action_from_json
from cppanatron.catanatron import from_catanatron_action

from catanrl.bridge.position import (
    DEVELOPMENT_CARDS,
    RESOURCES,
    VICTORY_POINT,
    fresh_game,
    transport,
    victory_point_room,
)

logger = logging.getLogger(__name__)

COSTS = {
    "BUILD_ROAD": (1, 1, 0, 0, 0),
    "BUILD_SETTLEMENT": (1, 1, 1, 1, 0),
    "BUILD_CITY": (0, 0, 0, 2, 3),
    "BUY_DEVELOPMENT_CARD": (0, 0, 1, 1, 1),
}
PLAYED_CARD = {
    "PLAY_KNIGHT_CARD": "KNIGHT",
    "PLAY_YEAR_OF_PLENTY": "YEAR_OF_PLENTY",
    "PLAY_MONOPOLY": "MONOPOLY",
    "PLAY_ROAD_BUILDING": "ROAD_BUILDING",
}
#: Public points one move can add to its player (a settlement that also
#: takes Longest Road): placeholder victory point cards stay this far from a
#: win where the unseen cards allow.
POINT_MARGIN = 3
#: Moves that can add public points (buildings, Longest Road, Largest Army).
POINT_MOVES = frozenset({"BUILD_SETTLEMENT", "BUILD_CITY", "BUILD_ROAD", "PLAY_KNIGHT_CARD"})


class ReplayError(RuntimeError):
    """The log cannot be replayed (it does not belong to this game, or no
    placeholder deal makes a move playable)."""


class BankShort(ReplayError):
    """A roll the game paid in full that the engine's bank could not: the
    replay needs rules with ``deficit`` more of each resource in the bank."""

    def __init__(self, deficit: Sequence[int]) -> None:
        super().__init__(f"the game paid a roll its bank could not cover by {list(deficit)}")
        self.deficit = [int(d) for d in deficit]


class DiceRefused(ReplayError):
    """A roll the replay's balanced dice cannot give: the game's dice state is
    not the one replayed (see the module doc)."""

    def __init__(self, index: int, dice: Sequence[int], reason: str) -> None:
        super().__init__(f"entry {index}: balanced dice cannot roll {tuple(dice)}: {reason}")
        self.index = int(index)
        self.dice = tuple(int(d) for d in dice)
        self.reason = reason


def _cards(kind_counts: Mapping[int, int]) -> list[int]:
    out = [0] * 5
    for k, v in kind_counts.items():
        out[k] += v
    return out


@dataclass(frozen=True)
class Counts:
    """Counts of one moment of a replay. ``resources`` of hidden seats are
    placeholders: only their differences across a public move are public."""

    resources: tuple[tuple[int, ...], ...]
    development_cards: tuple[int, ...]
    played: tuple[tuple[int, ...], ...]
    points: tuple[int, ...]
    bank: tuple[int, ...]
    deck: int

    @classmethod
    def of(cls, game: cpp.Game) -> Counts:
        players = game.players
        return cls(
            resources=tuple(tuple(p.resources) for p in players),
            development_cards=tuple(sum(p.development_cards) for p in players),
            played=tuple(tuple(p.played_development_cards) for p in players),
            points=tuple(p.victory_points for p in players),
            bank=tuple(game.resource_bank),
            deck=int(game.development_cards_remaining),
        )

    @property
    def sizes(self) -> tuple[int, ...]:
        return tuple(sum(r) for r in self.resources)


@dataclass(frozen=True)
class Transition:
    """One replayed log entry."""

    index: int
    entry: Any
    player: int
    kind: str
    action: cpp.Action
    before: Counts
    after: Counts
    #: The robber's victim (seat), if any.
    victim: int | None = None
    #: The stolen resource, when the seat was thief or victim.
    stolen: str | None = None
    #: The bought card, when the seat bought it.
    card: str | None = None
    #: The monopolized resource.
    resource: str | None = None
    #: Whether a monopoly's takes came from public hand sizes (always True
    #: for other moves).
    exact_amounts: bool = True
    #: A confirmed domestic trade's partner (seat).
    partner: int | None = None


class Replay:
    """A view's log replayed on cppanatron for seat ``seat`` (see the module doc).

    ``game`` is the position after :attr:`applied` entries: exactly the
    view's public state and the seat's own cards, placeholders elsewhere.
    """

    def __init__(self, view: Mapping[str, Any], seat: int, rules: cpp.GameRules, catan_map: cpp.CatanMap,
                 *, seed: int = 0, game: cpp.Game | None = None, entries: Sequence[Any] = ()) -> None:
        """A replay from the start of the game, or (``game``, ``entries``)
        from a position reached otherwise (an ``apply_setup`` snapshot)
        after those log entries."""
        self.colors = tuple(view["colors"])
        self.seat = int(seat)
        self.rules = rules
        self.game = game if game is not None else fresh_game(view, rules, catan_map, seed)
        self.others = tuple(s for s in range(len(self.colors)) if s != self.seat)
        self.entries: list[Any] = list(entries)
        #: Placeholder repairs by kind.
        self.repairs: Counter[str] = Counter()

    @property
    def applied(self) -> int:
        return len(self.entries)

    def follows(self, records: Sequence[Any]) -> bool:
        """Whether ``records`` extends the log replayed so far."""
        n = len(self.entries)
        return len(records) >= n and list(records[:n]) == self.entries

    # ------------------------------------------------------------ stepping

    def step(self, entry: Any, *, sizes_after: Sequence[int] | None = None,
             produced: Sequence[Sequence[int]] | None = None) -> Transition:
        """Replay one log entry; ``sizes_after`` are the public hand sizes
        after it (used, and checked, for monopolies); ``produced`` is what a
        roll paid each seat (public), when the game's bank rule may differ
        from the engine's."""
        index = len(self.entries)
        try:
            (color, kind, value), result = entry
            player = self.colors.index(color)
        except (TypeError, ValueError) as error:
            raise ReplayError(f"log entry {index} is not [action, result]: {entry!r}") from error
        mine = player == self.seat
        forced: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        if kind in ("ROLL", "BUY_DEVELOPMENT_CARD"):
            # The record shows the outcome in the action; the move itself has none.
            action = cpp.Action(cpp.Color[color], cpp.ActionType[kind], None)
        else:
            action = from_catanatron_action(action_from_json([color, kind, value]))

        if kind == "ROLL":
            forced["dice"] = tuple(int(d) for d in result)
            self._check_roll(index, player, forced["dice"])
        elif kind == "BUY_DEVELOPMENT_CARD":
            if not mine:
                self._ensure_resources(player, COSTS[kind])
            if result is not None:
                if not mine:
                    raise ReplayError(f"entry {index}: the log shows another seat's card")
                extra["card"] = result
                self._ensure_in_deck(DEVELOPMENT_CARDS.index(result))
                card = result
            else:
                if mine:
                    raise ReplayError(f"entry {index}: the seat's own purchase shows no card")
                card = self._placeholder_card(player)
            forced["development_card"] = cpp.DevelopmentCard[card]
        elif kind == "MOVE_ROBBER":
            victim_color = value[1]
            if victim_color is not None:
                victim = self.colors.index(victim_color)
                extra["victim"] = victim
                if result is not None:
                    if self.seat not in (player, victim):
                        raise ReplayError(f"entry {index}: the log shows a steal between other seats")
                    extra["stolen"] = result
                    if victim != self.seat:
                        self._ensure_resources(victim, _cards({RESOURCES.index(result): 1}))
                    forced["stolen_resource"] = cpp.Resource[result]
                else:
                    if self.seat in (player, victim):
                        raise ReplayError(f"entry {index}: the seat's own steal shows no card")
                    forced["stolen_resource"] = cpp.Resource[self._placeholder_steal(victim)]
        elif kind == "PLAY_MONOPOLY":
            extra["resource"] = value
            if not mine:
                self._ensure_playable(player, DEVELOPMENT_CARDS.index("MONOPOLY"))
            extra["exact_amounts"] = self._prepare_monopoly(player, RESOURCES.index(value), sizes_after)
        else:
            if kind == "CONFIRM_TRADE":
                extra["partner"] = self.colors.index(value[10])
            if not mine:
                self._prepare(player, kind, value)

        if kind in POINT_MOVES:
            self._guard_points(action, forced)
        before = Counts.of(self.game)
        try:
            self.game.execute(action, **forced)
        except (ValueError, RuntimeError) as error:
            raise ReplayError(f"entry {index} {entry!r} is not playable in the replay: {error}") from None
        if kind == "ROLL" and produced is not None:
            self._check_production(index, before, produced)
        after = Counts.of(self.game)
        if sizes_after is not None and tuple(int(s) for s in sizes_after) != after.sizes:
            raise ReplayError(f"entry {index} {entry!r}: hand sizes {after.sizes}, the view shows "
                              f"{tuple(sizes_after)}")
        self.entries.append(entry)
        return Transition(index=index, entry=entry, player=player, kind=kind, action=action,
                          before=before, after=after, **extra)

    def _check_roll(self, index: int, player: int, dice: tuple[int, ...]) -> None:
        """Balanced dice: refuse (:class:`DiceRefused`) a roll the public dice
        state cannot give ``player``; random dice take any faces."""
        game = self.game
        if game.dice_mode != cpp.DiceMode.BALANCED:
            return
        state = game.balanced_dice
        if len(dice) != 2 or not all(1 <= d <= 6 for d in dice):
            raise ReplayError(f"entry {index}: a roll of {dice!r}")
        if not state.can_draw(dice):
            raise DiceRefused(index, dice, f"the deck ({state.remaining} cards left) no longer holds it")
        if game.roll_probabilities(player)[sum(dice) - 2] <= 0.0:
            raise DiceRefused(index, dice, f"a total of {sum(dice)} has no chance (recent totals {state.recent}, "
                                           f"sevens {state.sevens})")

    def _prepare(self, player: int, kind: str, value: Any) -> None:
        """Make another seat's public move playable on the placeholders."""
        game = self.game
        if kind == "BUILD_ROAD":
            if not (game.is_initial_build_phase or game.is_road_building):
                self._ensure_resources(player, COSTS[kind])
        elif kind == "BUILD_SETTLEMENT":
            if not game.is_initial_build_phase:
                self._ensure_resources(player, COSTS[kind])
        elif kind == "BUILD_CITY":
            self._ensure_resources(player, COSTS[kind])
        elif kind == "MARITIME_TRADE":
            given = Counter(r for r in value[:-1] if r is not None)
            self._ensure_resources(player, _cards({RESOURCES.index(r): n for r, n in given.items()}))
        elif kind == "DISCARD_RESOURCE":
            self._ensure_resources(player, _cards({RESOURCES.index(value): 1}))
        elif kind in PLAYED_CARD:
            self._ensure_playable(player, DEVELOPMENT_CARDS.index(PLAYED_CARD[kind]))
        elif kind == "CONFIRM_TRADE":
            self._ensure_resources(player, [int(x) for x in value[:5]])
            partner = self.colors.index(value[10])
            if partner != self.seat:
                self._ensure_resources(partner, [int(x) for x in value[5:10]])
        elif kind == "ACCEPT_TRADE":
            self._ensure_resources(player, [int(x) for x in value[5:10]])

    def _check_production(self, index: int, before: Counts, produced: Sequence[Sequence[int]]) -> None:
        """Compare what the roll paid in the replay with what it paid in the
        game: :class:`BankShort` where the engine's bank rule paid nothing
        the game paid; a warning for any other difference (a board or robber
        the replay has wrong, which the caller's mismatch checks report)."""
        n = len(self.colors)
        engine = [[int(self.game.player(s).resources[k]) - int(before.resources[s][k]) for k in range(5)]
                  for s in range(n)]
        wanted = [[int(x) for x in produced[s]] for s in range(n)]
        if engine == wanted:
            return
        deficit = [0] * 5
        for k in range(5):
            paid = sum(w[k] for w in wanted)
            if all(e[k] == 0 for e in engine) and paid > int(before.bank[k]):
                deficit[k] = paid - int(before.bank[k])
        if any(deficit):
            raise BankShort(deficit)
        self.repairs["production"] += 1
        logger.warning("entry %d: the roll paid %s in the game and %s in the replay (a bank the game "
                       "withheld, or a board or robber the replay has wrong); the replay's payout stands",
                       index, wanted, engine)

    def _prepare_monopoly(self, player: int, resource: int, sizes_after: Sequence[int] | None) -> bool:
        """Give every other seat exactly the cards of ``resource`` the hand
        sizes after the move say it lost; False when no sizes are known."""
        if sizes_after is None:
            return False
        before = [sum(self.game.player(s).resources) for s in range(len(self.colors))]
        exact: dict[int, dict[int, int]] = {}
        for s in range(len(self.colors)):
            if s == player:
                continue
            taken = before[s] - int(sizes_after[s])
            if s == self.seat:
                if taken != self.game.player(s).resources[resource]:
                    raise ReplayError("a monopoly took a different number of the seat's own cards")
            else:
                exact[s] = {resource: taken}
        self._redeal_resources(exact=exact)
        return True

    # ------------------------------------------------------------ resources

    def _ensure_resources(self, seat: int, cards: Sequence[int]) -> None:
        if seat == self.seat:
            return
        held = self.game.player(seat).resources
        if all(held[k] >= cards[k] for k in range(5)):
            return
        self._redeal_resources(lower={seat: list(cards)})

    def _redeal_resources(self, *, lower: Mapping[int, Sequence[int]] | None = None,
                          exact: Mapping[int, Mapping[int, int]] | None = None,
                          sizes: Mapping[int, int] | None = None) -> None:
        """Re-deal the other seats' resources: same total per resource, sizes
        ``sizes`` (default: unchanged), at least ``lower`` and exactly
        ``exact`` (seat -> {resource: count}); as few cards move as possible."""
        current = {s: list(self.game.player(s).resources) for s in self.others}
        supply = [sum(current[s][k] for s in self.others) for k in range(5)]
        demand = {s: int(sizes[s]) if sizes is not None and s in sizes else sum(current[s]) for s in self.others}
        lo = {s: [0] * 5 for s in self.others}
        hi = {s: [demand[s]] * 5 for s in self.others}
        for s, cards in (lower or {}).items():
            lo[s] = [max(a, int(b)) for a, b in zip(lo[s], cards)]
        for s, counts in (exact or {}).items():
            for k, v in counts.items():
                lo[s][k] = hi[s][k] = int(v)
        try:
            deal = transport(supply, demand, lower=lo, upper=hi, prefer=current)
        except ValueError as error:
            raise ReplayError(f"no deal of the other seats' resources fits: {error}") from None
        changed = {s: deal[s] for s in self.others if deal[s] != current[s]}
        if not changed:
            return
        if sizes is None:
            self._redeal(resources=changed)
        else:  # new hand sizes: set them (the bank stays, as the totals do)
            for s, hand in changed.items():
                self.game.set_resources(s, tuple(hand))
        self.repairs["resources"] += 1

    def resize(self, sizes: Sequence[int]) -> None:
        """Move placeholder cards between the other seats so their hands
        have the public ``sizes`` (after a monopoly whose takes were not
        known)."""
        if int(sizes[self.seat]) != sum(self.game.player(self.seat).resources):
            raise ReplayError("the seat's own hand size differs from the view")
        self._redeal_resources(sizes={s: int(sizes[s]) for s in self.others})

    def _redeal(self, *, resources: Mapping[int, Sequence[int]] | None = None,
                cards: Mapping[int, Sequence[int]] | None = None,
                playable: Mapping[int, int] | None = None) -> None:
        """Give the other seats these resources and development cards
        (``Game.redeal_hidden_cards``; the rest of the unseen cards becomes
        the deck). A hand whose cards change counts every kind it now holds
        as held since its turn began, and ``playable`` (seat -> kind) marks
        one more such kind."""
        hands = []
        for seat in range(len(self.colors)):
            p = self.game.player(seat)
            held = list((cards or {}).get(seat, p.development_cards))
            owned = list(p.development_card_owned_at_start)
            if held != list(p.development_cards):
                owned = [held[d] > 0 for d in range(4)]
            if seat in (playable or {}):
                owned[playable[seat]] = True
            hands.append(cpp.HiddenHand(resources=tuple((resources or {}).get(seat, p.resources)),
                                        development_cards=tuple(held), development_card_owned_at_start=owned))
        try:
            self.game.redeal_hidden_cards(hands, seed=len(self.entries))
        except ValueError as error:
            raise ReplayError(f"the placeholders cannot be re-dealt: {error}") from None

    def _placeholder_steal(self, victim: int) -> str:
        held = self.game.player(victim).resources
        if sum(held) == 0:
            raise ReplayError("a hidden steal from an empty hand")
        return RESOURCES[max(range(5), key=lambda k: (held[k], -k))]

    # ------------------------------------------------------------ development cards

    def deck(self) -> list[int]:
        """The replay's deck by kind: the rules' deck minus every hand and played card."""
        deck = [int(x) for x in self.rules.development_deck]
        for p in self.game.players:
            for d in range(5):
                deck[d] -= p.development_cards[d] + p.played_development_cards[d]
        return deck

    def _room(self, seat: int, margin: int = POINT_MARGIN) -> int:
        return victory_point_room(self.rules, self.game.player(seat).victory_points, margin)

    def _placeholder_card(self, player: int) -> str:
        """The card another seat's hidden purchase draws: the deck's most
        common kind other than victory points, else a victory point card
        when the seat has room for it."""
        deck = self.deck()
        kinds = [d for d in range(5) if d != VICTORY_POINT and deck[d] > 0]
        if kinds:
            return DEVELOPMENT_CARDS[max(kinds, key=lambda d: (deck[d], -d))]
        held = self.game.player(player).development_cards[VICTORY_POINT]
        if deck[VICTORY_POINT] > 0 and held + 1 <= self._room(player, 0):
            return DEVELOPMENT_CARDS[VICTORY_POINT]
        # Move another kind into the deck (a hand takes a victory point card instead).
        for d in range(5):
            if d == VICTORY_POINT:
                continue
            try:
                self._redeal_cards(deck_lower={d: 1}, margin=0)
            except ReplayError:
                continue
            return DEVELOPMENT_CARDS[d]
        if deck[VICTORY_POINT] > 0:
            # Every unseen card that can be in the deck is a victory point
            # card: so was this one, and it wins the game.
            return DEVELOPMENT_CARDS[VICTORY_POINT]
        raise ReplayError("no placeholder card fits another seat's purchase")

    def _ensure_in_deck(self, card: int) -> None:
        if self.deck()[card] < 1:
            self._redeal_cards(deck_lower={card: 1})

    def _ensure_playable(self, seat: int, card: int) -> None:
        if seat == self.seat:
            return
        p = self.game.player(seat)
        if p.development_cards[card] < 1:
            self._redeal_cards(lower={seat: {card: 1}})
            p = self.game.player(seat)
        if not p.development_card_owned_at_start[card]:
            # A placeholder bought this turn: the seat held one before.
            self._redeal(playable={seat: card})
            self.repairs["playable"] += 1

    def _guard_points(self, action: cpp.Action, forced: Mapping[str, Any]) -> None:
        """No move may make a seat win the replay on placeholder victory
        point cards: try a move that adds public points (a build, a knight,
        an award it moves) on a copy, and re-deal first the cards of any
        seat it would make win, unless no deal avoids it (then the move
        wins the game)."""
        if not any(self.game.player(s).development_cards[VICTORY_POINT] for s in self.others):
            return
        trial = self.game.copy()
        try:
            trial.execute(action, **forced)
        except (ValueError, RuntimeError):
            return  # the real move reports it
        caps = {}
        for s in self.others:
            p = trial.player(s)
            room = victory_point_room(self.rules, p.victory_points)
            if p.development_cards[VICTORY_POINT] > room:
                caps[s] = room
        if not caps or self._card_deal(caps=caps, margin=0) is None:
            return  # every deal of the unseen cards makes that seat win: so does the game
        self._redeal_cards(caps=caps)
        self.repairs["points"] += 1

    def _card_deal(self, *, lower: Mapping[int, Mapping[int, int]] | None = None,
                   deck_lower: Mapping[int, int] | None = None, margin: int = POINT_MARGIN,
                   caps: Mapping[int, int] | None = None):
        """A deal of the unseen development cards (the deck plus the other
        hands) keeping every hand size, with at least ``lower`` per hand and
        ``deck_lower`` in the deck, no seat within ``margin`` points of a win
        on victory point cards and at most ``caps`` of them; None if there
        is none. Few cards move."""
        current: dict[Any, list[int]] = {s: list(self.game.player(s).development_cards) for s in self.others}
        current["deck"] = self.deck()
        supply = [sum(current[h][d] for h in current) for d in range(5)]
        demand = {h: sum(cards) for h, cards in current.items()}
        lo: dict[Any, list[int]] = {h: [0] * 5 for h in current}
        for s, counts in (lower or {}).items():
            for d, v in counts.items():
                lo[s][d] = max(lo[s][d], int(v))
        for d, v in (deck_lower or {}).items():
            lo["deck"][d] = max(lo["deck"][d], int(v))
        hi: dict[Any, list[int]] = {h: [demand[h]] * 5 for h in current}
        for s in self.others:
            room = min(self._room(s, margin), int((caps or {}).get(s, demand[s])))
            hi[s][VICTORY_POINT] = max(lo[s][VICTORY_POINT], min(demand[s], room))
        try:
            return transport(supply, demand, lower=lo, upper=hi, prefer=current)
        except ValueError:
            return None

    def _redeal_cards(self, *, lower: Mapping[int, Mapping[int, int]] | None = None,
                      deck_lower: Mapping[int, int] | None = None, margin: int = POINT_MARGIN,
                      caps: Mapping[int, int] | None = None) -> None:
        """Re-deal the unseen development cards to meet ``lower``,
        ``deck_lower`` and ``caps`` (see :meth:`_card_deal`; ``margin``
        points kept free when possible)."""
        for m in sorted({margin, 0}, reverse=True):
            deal = self._card_deal(lower=lower, deck_lower=deck_lower, margin=m, caps=caps)
            if deal is not None:
                break
        else:
            raise ReplayError("no deal of the unseen development cards fits the public moves")
        self._redeal(cards={s: deal[s] for s in self.others})
        self.repairs["cards"] += 1
