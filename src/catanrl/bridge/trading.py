"""Domestic trades for a server seat: answer, confirm and propose by the
seat's own network.

A room with ``trades`` (catanatron-server) lets the seat to move offer a
trade after its roll; every other seat answers, and the offerer confirms one
of those who accepted. :class:`TradeJudge` makes those three decisions for a
:class:`~catanrl.bridge.agent.ViewAgent` seat the way the ``TRADE`` agent
makes them in the native harness (:mod:`catanrl.agents.trade`, whose
primitives it uses), reading the network's own value head (its per-seat win
probabilities, ego order) on the rows the policy reads:

* **answer** (DECIDE_TRADE): deal worlds from the seat's hand tracker in
  which the offerer can pay, and accept when the seat's own win probability,
  read on the offerer's row before and after the trade, gains more than
  ``accept_margin`` points on average;
* **confirm** (DECIDE_ACCEPTEES): take the acceptee whose trade gains the
  seat most, above ``margin``; otherwise cancel;
* **propose** (the seat's own turn, when its decide says it may offer):
  score every bundle of the fair menu (1:1, 2:1, 1:2; what a partner can pay
  is voted by dealt worlds, nothing of its hand read but its size) against
  every partner, keep the bundles that gain both seats more than ``margin``,
  their Pareto front ordered by the seat's own gain, and offer them in turn,
  at most ``offers`` a turn;
* **take** (an offer put to the seat to move alone, as OpenHex's
  counter-offers are): take it when the seat gains more than ``margin``.

Nothing it reads of another seat's hand is more than its public size: the
offerer's and partners' hands come from worlds the tracker deals.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

import cppanatron as cpp
import numpy as np

from catanrl.agents.trade import (
    SHAPES,
    Trade,
    acceptees_of,
    can_pay,
    confirm_action,
    mover_menu,
    offer_of,
    post_row_clone,
    respond_route,
    rewind,
    tradeable,
)
from catanrl.seeding import derive_seed

DECIDE_TRADE = cpp.ActionPrompt.DECIDE_TRADE
DECIDE_ACCEPTEES = cpp.ActionPrompt.DECIDE_ACCEPTEES
#: Prompts of a negotiation, by Catanatron's name.
TRADE_PROMPTS = ("DECIDE_TRADE", "DECIDE_ACCEPTEES")


@dataclass(frozen=True)
class Candidate:
    trade: Trade
    partner: int
    gain: float
    partner_gain: float


@dataclass
class _Turn:
    """A seat's proposals in one turn: the bundles offered (never again this
    turn), how many, and the front read for the hand it was read on."""

    turn: int
    offered: set = field(default_factory=set)
    count: int = 0
    hand: tuple | None = None
    front: list | None = None


class TradeJudge:
    """Trade decisions for the seats a native agent plays (see the module doc).

    ``native``: a native agent with a network (``model``, ``encoder``,
    ``tracker_mode``); margins are in points of win probability.
    """

    def __init__(self, native: Any, *, margin: float = 1.0, accept_margin: float = 1.0, dets: int = 8,
                 min_feasible: float = 0.25, offers: int = 2, shapes: Sequence[tuple[int, int]] = SHAPES,
                 device: str = "cpu", infer_rows: int = 256, max_turns: int = 256) -> None:
        if not hasattr(native, "model") or not hasattr(native, "encoder"):
            raise ValueError(f"{getattr(native, 'key', native)}: trades read the agent's own network, "
                             "and this agent has none")
        if getattr(native, "recurrent", False):
            raise ValueError("trades read stateless rows; a recurrent network's values read its history")
        if margin < 0 or accept_margin < 0 or dets < 1 or offers < 0 or not 0 < min_feasible <= 1:
            raise ValueError("trades need margins >= 0, dets >= 1, offers >= 0 and min_feasible in (0, 1]")
        self.native = native
        self.margin = float(margin) / 100.0
        self.accept_margin = float(accept_margin) / 100.0
        self.dets = int(dets)
        self.min_feasible = float(min_feasible)
        self.offers = int(offers)
        self.shapes = tuple((int(g), int(r)) for g, r in shapes)
        self.device = device
        self.infer_rows = int(infer_rows)
        self.max_turns = int(max_turns)
        self._critic = None
        self._critic_model = None
        self._turns: OrderedDict[Hashable, _Turn] = OrderedDict()
        self.stats = {"answers": 0, "accepts": 0, "confirms": 0, "cancels": 0, "offers": 0, "fronts": 0,
                      "critic_rows": 0}

    # ------------------------------------------------------------ the network

    def _values(self, rows: np.ndarray) -> np.ndarray:
        """Per-seat win probabilities (ego order) of ``rows``."""
        from catanrl.search.evaluator import BatchEvaluator

        model = self.native.model  # rebuilt when the agent rebinds its dims
        if self._critic_model is not model:
            self._critic = BatchEvaluator(model, device=self.device, infer_rows=self.infer_rows, mode="fast",
                                          heads="both")
            self._critic_model = model
        self.stats["critic_rows"] += len(rows)
        return np.asarray(self._critic(rows)[1], dtype=np.float64)

    def _rows(self, games: Sequence[tuple[cpp.Game, int]]) -> np.ndarray:
        encoder = self.native.encoder
        rows = np.empty((len(games), encoder.size), dtype=np.float32)
        for i, (game, seat) in enumerate(games):
            encoder.encode_into(game, seat, rows[i])
        return rows

    def _world(self, game: cpp.Game, tracker: cpp.HandTracker, seed: int, w: int) -> cpp.Game:
        return cpp.sample_world(game, tracker, seed=derive_seed(seed, "world", w),
                                deal=getattr(self.native, "tracker_mode", cpp.TrackerMode.POOL))

    # ------------------------------------------------------------ answering and confirming

    def answer(self, game: cpp.Game, seat: int, tracker: cpp.HandTracker, seed: int) -> bool:
        """Accept the offer in ``game`` (at the seat's DECIDE_TRADE)?"""
        self.stats["answers"] += 1
        give, get, proposer = offer_of(game)
        if proposer == seat or not can_pay(game.player(seat).resources, get):
            return False
        trade = Trade(give, get)
        worlds = []
        for w in range(4 * self.dets):
            world = self._world(game, tracker, seed, w)
            if can_pay(world.player(proposer).resources, give):
                worlds.append(world)
                if len(worlds) == self.dets:
                    break
        if not worlds:
            return False
        pairs = []
        for world in worlds:
            pairs += [(rewind(world), proposer), (respond_route(world, proposer, seat, trade), proposer)]
        values = self._values(self._rows(pairs))
        k = (seat - proposer) % game.num_players
        gain = float(np.mean(values[1::2, k] - values[0::2, k]))
        accept = gain > self.accept_margin
        self.stats["accepts"] += accept
        return accept

    def confirm(self, game: cpp.Game, seat: int) -> int | None:
        """The acceptee to trade with (at the seat's DECIDE_ACCEPTEES), or None to cancel."""
        give, get, _ = offer_of(game)
        n = game.num_players
        acceptees = sorted(acceptees_of(game), key=lambda q: (q - seat) % n)
        posts = []
        for acceptee in acceptees:
            post = game.copy()
            post.execute(confirm_action(game, acceptee))
            posts.append((post, seat))
        values = self._values(self._rows([(rewind(game), seat), *posts]))
        best = None
        for j, acceptee in enumerate(acceptees):  # slot order: a tie keeps the nearer seat
            gain = float(values[1 + j, 0] - values[0, 0])
            if gain > self.margin and (best is None or gain > best[1]):
                best = (acceptee, gain)
        self.stats["confirms" if best else "cancels"] += 1
        return None if best is None else best[0]

    def take(self, game: cpp.Game, seat: int, partner: int, trade: Trade) -> bool:
        """Take ``trade`` (what the seat gives and gets) from ``partner`` at the seat's own
        play turn? An offer made to the seat to move alone (OpenHex's counter-offer): yes
        when the seat's win probability gains more than ``margin``, read as a proposal's is
        (the partner's hand a stand-in of its public size)."""
        self.stats["answers"] += 1
        if not tradeable(game, seat) or partner == seat or not can_pay(game.player(seat).resources, trade.give) \
                or sum(trade.get) > sum(game.player(partner).resources):
            return False
        try:
            post = post_row_clone(game, seat, partner, trade)
        except (ValueError, RuntimeError):  # an offer the engine does not take
            return False
        values = self._values(self._rows([(game, seat), (post, seat)]))
        accept = float(values[1, 0] - values[0, 0]) > self.margin
        self.stats["accepts"] += accept
        return accept

    # ------------------------------------------------------------ proposing

    def propose(self, game: cpp.Game, seat: int, tracker: cpp.HandTracker, seed: int, key: Hashable
                ) -> Trade | None:
        """A bundle to offer now (the seat may offer: its decide said so), or None.
        ``key`` names the seat in its game (the per-turn state)."""
        if self.offers == 0 or not tradeable(game, seat):
            return None
        hand = tuple(int(c) for c in game.player(seat).resources)
        if not sum(hand):
            return None
        turn = self._turn(key, game.num_turns)
        if turn.count >= self.offers:
            return None
        if turn.front is None or turn.hand != hand:
            turn.front, turn.hand = self._front(game, seat, tracker, seed, turn.offered), hand
        offer = next((c for c in turn.front if c.trade not in turn.offered and can_pay(hand, c.trade.give)), None)
        if offer is None:
            return None
        turn.offered.add(offer.trade)
        turn.count += 1
        self.stats["offers"] += 1
        return offer.trade

    def _turn(self, key: Hashable, num_turns: int) -> _Turn:
        turn = self._turns.get(key)
        if turn is None or turn.turn != num_turns:
            turn = self._turns[key] = _Turn(num_turns)
            while len(self._turns) > self.max_turns:
                self._turns.popitem(last=False)
        self._turns.move_to_end(key)
        return turn

    def _front(self, game: cpp.Game, seat: int, tracker: cpp.HandTracker, seed: int, offered: set
               ) -> list[Candidate]:
        """The ordered Pareto front of the bundles gaining both seats more than ``margin``."""
        self.stats["fronts"] += 1
        n = game.num_players
        hand = game.player(seat).resources
        partners = [(seat + k) % n for k in range(1, n)]
        sizes = {q: sum(game.player(q).resources) for q in partners}
        menu = [t for t in mover_menu(hand, max(sizes.values()), self.shapes) if t not in offered]
        if not menu:
            return []
        worlds = [self._world(game, tracker, seed, w) for w in range(self.dets)]
        candidates = []
        for trade in menu:
            for q in partners:
                if sum(trade.get) > sizes[q]:
                    continue
                feasible = sum(can_pay(world.player(q).resources, trade.get) for world in worlds) / len(worlds)
                if feasible >= self.min_feasible:
                    candidates.append((trade, q))
        if not candidates:
            return []
        rows = self._rows([(game, seat)] + [(post_row_clone(game, seat, q, t), seat) for t, q in candidates])
        values = self._values(rows)
        pre = values[0]
        best: dict[Trade, Candidate] = {}
        for (trade, q), post in zip(candidates, values[1:]):
            k = (q - seat) % n
            gain, partner_gain = float(post[0] - pre[0]), float(post[k] - pre[k])
            if gain > self.margin and partner_gain > self.margin:
                current = best.get(trade)
                if current is None or gain > current.gain:
                    best[trade] = Candidate(trade, q, gain, partner_gain)
        pool = list(best.values())
        front = [c for c in pool if not any(
            o.gain >= c.gain and o.partner_gain >= c.partner_gain and (o.gain > c.gain or o.partner_gain > c.partner_gain)
            for o in pool)]
        front.sort(key=lambda c: (-c.gain, -c.partner_gain, sum(c.trade.give)))
        return front


def offer_json(color: str, trade: Trade) -> list:
    """The OFFER_TRADE a seat sends for ``trade`` (Catanatron's JSON)."""
    return [color, "OFFER_TRADE", [int(c) for c in (*trade.give, *trade.get)]]
