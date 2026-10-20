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

:class:`TradeAgent` (``TRADE``) is the harness agent built on them: a POLICY
seat that proposes, answers and confirms domestic trades by a critic's
per-seat win probabilities, read on the proposer's play-turn row only. The
proposer scores every bundle of the fair menu against every partner on
stand-in clones, keeps the Pareto front of (own gain, partner gain) in ego
view and offers it in order over the turn; a responder accepts when its own
mean gain over worlds dealt from its tracker clears ``accept_margin``; the
confirmer takes the acceptee that gains it most. Nothing it reads of another
seat's hand is more than its public size.
"""

from __future__ import annotations

import copy
import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import cppanatron as cpp
import numpy as np

from catanrl.agents.native import NativeAgent, PolicyAgent, _param, tracker_mode_of
from catanrl.agents.spec import trade_shapes
from catanrl.envs.encoding import action_size, critic_level_of, make_encoder, row_layout
from catanrl.models import CatanModel, build_model
from catanrl.search.evaluator import BatchEvaluator
from catanrl.seeding import derive_seed

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


# ---------------------------------------------------------------- the TRADE agent

DECIDE_TRADE = cpp.ActionPrompt.DECIDE_TRADE
DECIDE_ACCEPTEES = cpp.ActionPrompt.DECIDE_ACCEPTEES


@dataclass(frozen=True)
class Candidate:
    """A bundle the proposer scored: its partner (after the collapse over partners, the one
    whose row gains the proposer most), the proposer's and that partner's gain in win
    probability (both read from the proposer's play-turn row) and the fraction of the
    dealt worlds in which the partner could pay."""

    trade: Trade
    partner: int
    gain: float
    partner_gain: float
    feasibility: float


@dataclass(frozen=True)
class TradeDecision:
    """What a TRADE seat did at one forwarded row: ``offer`` or ``policy`` at its play turn,
    ``accept`` / ``reject`` at DECIDE_TRADE, ``confirm`` / ``cancel`` at DECIDE_ACCEPTEES.
    ``gain`` is the seat's own gain behind the choice (the offered bundle's, the mean over
    worlds, the confirmed acceptee's); ``front`` is the ordered front when this row read the
    critic for one (empty when nothing cleared the margins). The audit compares these across
    games of one public state."""

    kind: str
    seat: int
    bundle: Trade | None = None
    partner: int | None = None
    gain: float | None = None
    partner_gain: float | None = None
    front: tuple[Candidate, ...] | None = None
    note: str = ""


@dataclass
class _Stream:
    """A seat's state over one game (a harness stream), reset on its first forwarded row.
    The per-turn fields reset when ``num_turns`` moves: ``rejected_bundles`` holds the
    bundles offered this turn and not confirmed (never re-offered this turn); ``front`` is
    the ordered Pareto front read this turn, offered one bundle per negotiation round and
    dropped by a confirm; ``front_hand`` keys it for ``when=every``."""

    turn: int = -1
    offers_this_turn: int = 0
    rejected_bundles: set[Trade] = field(default_factory=set)
    evaluated: bool = False
    front: list[Candidate] | None = None
    front_hand: Counts | None = None
    policy_started: bool = False


def check_critic(critic: CatanModel) -> None:
    """The critic v1 reads: stateless, ``private``, no separate critic block (the agent
    encodes one row per candidate at the actor's level; a public or full critic would need
    a row per dealt world, a recurrent one the seat's decision history)."""
    if getattr(critic, "recurrent", False):
        raise ValueError(f"TRADE: the critic is a recurrent ({critic.spec['class']}) network; its values read a "
                         "seat's decision history, which the trade rows do not carry")
    levels = critic.levels
    if critic_level_of(levels["level"], levels["critic_level"]) is not None:
        raise ValueError("TRADE: the critic reads a separate critic block; the agent encodes one row per candidate")
    if levels["level"] != "private":
        raise ValueError(f"TRADE: the critic reads {levels['level']} rows; v1 reads the critic on private rows")


class TradeAgent(NativeAgent):
    """``TRADE``: ``model`` (the policy) plays as POLICY does at every row that is not an
    offer or a trade prompt; ``critic`` (``None``: the policy's own value head) judges the
    trades. ``params`` are :class:`catanrl.agents.spec.TradeParams`.

    * ``act`` (the seat's own rows): at a proposal moment (its PLAY_TURN after the roll,
      ``propose``, a card in hand, under ``offers`` this turn) the first eligible row of the
      turn (``when=first``; every row with a new hand for ``every``) reads the critic once on
      the pre row and one post row per (bundle, partner) of :func:`mover_menu` whose partner
      can pay in at least ``min_feasible`` of ``dets`` worlds dealt from the seat's tracker;
      candidates gaining both seats more than ``margin`` are collapsed per bundle to the
      partner gaining the proposer most, Pareto-filtered, ordered by own gain, partner gain,
      fewer cards given, enumeration order, and cached as the turn's front. Each eligible
      row then offers the front's next bundle not yet offered this turn; otherwise the policy
      decides (and only then steps a recurrent policy's memory).
    * ``respond_trade`` (trade prompts): at DECIDE_TRADE it rejects its own offer, else deals
      worlds from its tracker keeping those where the proposer can pay (``dets`` kept or
      ``4 * dets`` drawn; none: REJECT) and accepts iff its own slot of the proposer's row,
      pre to post, gains more than ``accept_margin`` on average; at DECIDE_ACCEPTEES it
      confirms the acceptee whose engine post-confirm row gains it most above ``margin``
      (ties: the lower slot), else cancels and never re-offers the bundle this turn.

    Every critic read is a proposer's play-turn row after the roll: the live game and
    stand-in clones (fact: at private a partner's hand shows as its size), dealt worlds
    rewound and routed, or the engine's post-confirm state. ``oracle=true`` reads the true
    game in place of worlds (unfair); ``audit=true`` repeats every decision on a world of the
    same public state and raises on a divergence. ``stats`` counts what happened;
    ``decisions`` holds the last call's :class:`TradeDecision` per row.
    """

    key = "TRADE"
    needs_games = True
    needs_memory = True

    def __init__(self, model: CatanModel, params: Any, *, device: str, infer_rows: int, mode: str = "strict",
                 critic: CatanModel | None = None) -> None:
        critic = model if critic is None else critic
        check_critic(critic)
        self.oracle = bool(_param(params, "oracle", False))
        self.audit = bool(_param(params, "audit", False))
        if self.audit and self.oracle:
            raise ValueError("TRADE: audit=true checks that no true hand is read; oracle=true reads them by design")
        self.needs_trackers = not self.oracle
        self.tracker_mode = tracker_mode_of(params)
        # Margins are written in points of win probability; the critic's values are probabilities.
        self.margin = float(_param(params, "margin", 0.0)) / 100.0
        self.accept_margin = float(_param(params, "accept_margin", 0.0)) / 100.0
        self.dets = int(_param(params, "dets", 8))
        self.min_feasible = float(_param(params, "min_feasible", 0.25))
        self.offers = int(_param(params, "offers", 3))
        self.when = str(_param(params, "when", "first"))
        self.propose = bool(_param(params, "propose", True))
        shapes = _param(params, "shapes", "11-21-12")
        self.shapes = trade_shapes(shapes) if isinstance(shapes, str) else tuple((int(g), int(r)) for g, r in shapes)
        if self.margin < 0 or self.accept_margin < 0:
            raise ValueError("TRADE: margin and accept_margin are points of win probability (>= 0)")
        if self.dets < 1 or self.offers < 1 or not 0 < self.min_feasible <= 1 or self.when not in ("first", "every"):
            raise ValueError("TRADE needs dets >= 1, offers >= 1, min_feasible in (0, 1] and when first or every")
        self._shared = critic is model
        self._critic_weights = {k: v.detach().cpu() for k, v in critic.state_dict().items()}
        self._critic_spec = critic.spec
        self._critic_levels = dict(critic.levels)
        self._critic_dims = dict(critic.dims)
        self._critic_agnostic = bool(critic.dims_agnostic)
        self.critic_level = critic.levels["level"]
        #: evaluations: proposal reads (a menu built, the critic read when it is not empty);
        #: critic_rows: rows the critic read; the rest count decisions by kind, the responder's
        #: rejects for want of a payable world, and the audit's checks and failures.
        self.stats = {"evaluations": 0, "critic_rows": 0, "offers": 0, "accepts": 0, "rejects": 0, "confirms": 0,
                      "cancels": 0, "no_feasible_world": 0, "audits": 0, "audit_divergences": 0}
        self._streams: dict[int, _Stream] = {}
        self.decisions: list[TradeDecision] = []
        super().__init__(model, params, device=device, infer_rows=infer_rows, mode=mode, heads="policy")

    def _set_model(self, model: CatanModel) -> None:
        self.model = model
        self.dims = dict(model.dims)
        self.encoder = make_encoder(self.dims, self.level)
        self.action_space = self.encoder.action_space
        self.policy = PolicyAgent(model, self.params, device=self.device, infer_rows=self.infer_rows, mode=self.mode)
        self.evaluator = self.policy.evaluator
        if self._shared:
            critic = model
        else:
            if not self._critic_agnostic and self.dims != self._critic_dims:
                raise ValueError(f"TRADE: the critic was trained at dims {self._critic_dims} and is not "
                                 f"dims-agnostic; the games need {self.dims}")
            critic = build_model(self._critic_spec, dims=self.dims, levels=self._critic_levels,
                                 num_actions=action_size(self.dims), layout=row_layout(self.dims, self.critic_level))
            critic.load_state_dict(self._critic_weights)
        self.critic_encoder = make_encoder(self.dims, self.critic_level)
        self.critic = BatchEvaluator(critic, device=self.device, infer_rows=self.infer_rows, mode=self.mode,
                                     heads="both")

    def check_maps(self, maps) -> None:
        self.model.check_maps(maps)
        self.critic.model.check_maps(maps)

    # ------------------------------------------------------------ the harness entry points

    def act(self, rows, masks, decision_seeds, *, games=None, trackers=None, streams=None, first=None
            ) -> list[int | cpp.Action]:
        games, trackers, streams, first = self._inputs(len(rows), games, trackers, streams, first)
        states = [self._stream(stream, fresh) for stream, fresh in zip(streams, first)]
        audit = [copy.deepcopy(state) for state in states] if self.audit else None
        decisions, read = self._propose(games, trackers, decision_seeds, states)
        if audit is not None:
            self._check_audit(self._propose, games, trackers, decision_seeds, audit, decisions)
        self._count(decisions, read)
        choices: list[int | cpp.Action] = [0] * len(rows)
        policy_rows = []
        for i, decision in enumerate(decisions):
            if decision.kind == "offer":
                choices[i] = offer_action(games[i], decision.seat, decision.bundle)
            else:
                policy_rows.append(i)
        if policy_rows:
            idx = np.array(policy_rows)
            fresh = np.array([not states[i].policy_started for i in policy_rows], dtype=bool)
            chosen = self.policy.act(rows[idx], masks[idx], np.asarray(decision_seeds)[idx], streams=streams[idx],
                                     first=fresh)
            for i, choice in zip(policy_rows, chosen):
                choices[i] = int(choice)
                states[i].policy_started = True
        self.decisions = decisions
        return choices

    def respond_trade(self, games, decision_seeds, *, trackers=None, streams=None, first=None) -> list[cpp.Action]:
        games, trackers, streams, first = self._inputs(len(games), games, trackers, streams, first)
        states = [self._stream(stream, fresh) for stream, fresh in zip(streams, first)]
        audit = [copy.deepcopy(state) for state in states] if self.audit else None
        decisions, read = self._respond(games, trackers, decision_seeds, states)
        if audit is not None:
            self._check_audit(self._respond, games, trackers, decision_seeds, audit, decisions)
        self._count(decisions, read)
        self.decisions = decisions
        actions = []
        for game, decision in zip(games, decisions):
            if decision.kind in ("accept", "reject"):
                actions.append(response(game, accept=decision.kind == "accept"))
            elif decision.kind == "confirm":
                actions.append(confirm_action(game, decision.partner))
            else:
                actions.append(cancel_action(game))
        return actions

    # ------------------------------------------------------------ state and bookkeeping

    def _inputs(self, n: int, games, trackers, streams, first):
        if games is None or len(games) != n:
            raise ValueError("TRADE decides from games (one per row)")
        if self.needs_trackers and (trackers is None or len(trackers) != n):
            raise ValueError("TRADE deals worlds from the deciding seats' hand trackers (one per row)")
        if streams is None or first is None:
            raise ValueError("TRADE keeps a state per seat of a game: it decides from streams and first")
        streams = np.asarray(streams, dtype=np.int64).reshape(-1)
        first = np.asarray(first, dtype=bool).reshape(-1)
        if len(streams) != n or len(first) != n:
            raise ValueError("streams and first need one entry per row")
        if len(set(streams.tolist())) != n:
            raise ValueError("a stream decides once per batch")
        return games, trackers, streams, first

    def _stream(self, stream: int, fresh: bool) -> _Stream:
        if fresh:
            self._streams[int(stream)] = _Stream()
        elif int(stream) not in self._streams:
            raise ValueError(f"stream {stream} has no state: its first row was not marked first")
        return self._streams[int(stream)]

    @staticmethod
    def _sync_turn(state: _Stream, game: cpp.Game) -> None:
        """The per-turn fields start afresh when the game's turn count moves (a trade adds no turn)."""
        if state.turn != game.num_turns:
            state.turn = game.num_turns
            state.offers_this_turn = 0
            state.rejected_bundles = set()
            state.evaluated = False
            state.front = None
            state.front_hand = None

    def _count(self, decisions: Sequence[TradeDecision], read: int) -> None:
        stats = self.stats
        stats["critic_rows"] += read
        for decision in decisions:
            stats["evaluations"] += decision.front is not None
            if decision.kind != "policy":
                stats[decision.kind + "s"] += 1
            stats["no_feasible_world"] += decision.note == "no feasible world"

    def _check_audit(self, decide, games, trackers, seeds, states, decisions) -> None:
        """The same decisions on worlds of the games' public state (own hand exact, the others
        dealt by the seat's own tracker): any difference means a true hand was read."""
        worlds = [cpp.sample_world(game, tracker, seed=derive_seed(int(seed), "audit"), deal=self.tracker_mode)
                  for game, tracker, seed in zip(games, trackers, seeds)]
        again, _ = decide(worlds, trackers, seeds, states)
        self.stats["audits"] += len(decisions)
        for game, decision, other in zip(games, decisions, again):
            if decision != other:
                self.stats["audit_divergences"] += 1
                raise RuntimeError(f"TRADE audit: seat {decision.seat} at {game.current_prompt.name} decided "
                                   f"{decision} on the true game but {other} on a world of its public state")

    def _worlds(self, game: cpp.Game, tracker, seed: int, count: int, start: int = 0) -> list[cpp.Game]:
        """``count`` worlds dealt from the seat's tracker (the true game alone with ``oracle``)."""
        if self.oracle:
            return [game]
        return [cpp.sample_world(game, tracker, seed=derive_seed(seed, "world", w), deal=self.tracker_mode)
                for w in range(start, start + count)]

    # ------------------------------------------------------------ the proposer

    def _proposal_moment(self, state: _Stream, game: cpp.Game, seat: int) -> bool:
        return (self.propose and tradeable(game, seat) and sum(game.player(seat).resources) > 0
                and state.offers_this_turn < self.offers)

    def _propose(self, games, trackers, seeds, states) -> tuple[list[TradeDecision], int]:
        """Every row's decision: one critic call for every row that reads a new front."""
        seats = [game.current_player_index for game in games]
        moments = []
        read: set[int] = set()  # rows that read a new front this call (an empty menu reads nothing)
        plans: dict[int, tuple[list[tuple[Trade, int, float]], int]] = {}
        chunks: list[np.ndarray] = []
        rows_so_far = 0
        for i, (game, seat, state) in enumerate(zip(games, seats, states)):
            self._sync_turn(state, game)
            moments.append(self._proposal_moment(state, game, seat))
            if not moments[i]:
                continue
            hand = tuple(int(c) for c in game.player(seat).resources)
            stale = not state.evaluated if self.when == "first" else (state.front is None or state.front_hand != hand)
            if not stale:
                continue
            candidates, rows = self._plan(game, seat, trackers[i] if trackers is not None else None, int(seeds[i]),
                                          state.rejected_bundles)
            state.evaluated, state.front_hand, state.front = True, hand, []
            read.add(i)
            if candidates:
                plans[i] = (candidates, rows_so_far)
                chunks.append(rows)
                rows_so_far += len(rows)
        if plans:
            values = self.critic(np.concatenate(chunks))[1].astype(np.float64)
            for i, (candidates, offset) in plans.items():
                posts = values[offset + 1 : offset + 1 + len(candidates)]
                states[i].front = self._front(candidates, values[offset], posts, seats[i], games[i].num_players)
        decisions = []
        for i, (game, seat, state) in enumerate(zip(games, seats, states)):
            if not moments[i]:
                decisions.append(TradeDecision("policy", seat))
                continue
            front = tuple(state.front) if i in read else None
            hand = game.player(seat).resources
            offer = next((c for c in state.front or () if c.trade not in state.rejected_bundles
                          and can_pay(hand, c.trade.give)), None)
            if offer is None:
                decisions.append(TradeDecision("policy", seat, front=front))
                continue
            state.offers_this_turn += 1
            state.rejected_bundles.add(offer.trade)
            decisions.append(TradeDecision("offer", seat, offer.trade, offer.partner, offer.gain, offer.partner_gain,
                                           front=front))
        return decisions, rows_so_far

    def _plan(self, game: cpp.Game, seat: int, tracker, seed: int, rejected: set[Trade]
              ) -> tuple[list[tuple[Trade, int, float]], np.ndarray | None]:
        """The candidates of a proposal read, ``(bundle, partner, feasibility)`` in menu then
        slot order, and their rows: the seat's pre row first, then one post row per candidate
        on a stand-in clone. Nothing of a partner's hand is read but its size: payability is
        voted by worlds dealt from the seat's tracker."""
        n = game.num_players
        hand = game.player(seat).resources
        partners = [(seat + k) % n for k in range(1, n)]
        sizes = {q: sum(game.player(q).resources) for q in partners}
        menu = [trade for trade in mover_menu(hand, max(sizes.values()), self.shapes) if trade not in rejected]
        if not menu:
            return [], None
        worlds = self._worlds(game, tracker, seed, self.dets)
        candidates = []
        for trade in menu:
            for q in partners:
                if sum(trade.get) > sizes[q]:
                    continue
                feasibility = sum(can_pay(world.player(q).resources, trade.get) for world in worlds) / len(worlds)
                if feasibility >= self.min_feasible:
                    candidates.append((trade, q, feasibility))
        rows = np.empty((1 + len(candidates), self.critic_encoder.size), dtype=np.float32)
        self.critic_encoder.encode_into(game, seat, rows[0])
        for j, (trade, q, _) in enumerate(candidates):
            self.critic_encoder.encode_into(post_row_clone(game, seat, q, trade), seat, rows[1 + j])
        return candidates, rows

    def _front(self, candidates, pre: np.ndarray, posts: np.ndarray, seat: int, n: int) -> list[Candidate]:
        """The ordered Pareto front: per bundle the partner whose row gains the proposer most
        (ties: the lower slot, since partners come in slot order) among candidates gaining
        both seats more than ``margin``; bundles another bundle weakly dominates dropped;
        ordered by own gain, partner gain, fewer cards given, enumeration order."""
        best: dict[Trade, Candidate] = {}
        for (trade, q, feasibility), post in zip(candidates, posts):
            k = (q - seat) % n
            gain, partner_gain = float(post[0] - pre[0]), float(post[k] - pre[k])
            if gain > self.margin and partner_gain > self.margin:
                current = best.get(trade)
                if current is None or gain > current.gain:
                    best[trade] = Candidate(trade, q, gain, partner_gain, feasibility)
        pool = list(best.values())  # insertion order = enumeration order
        front = [c for c in pool if not any(
            o.gain >= c.gain and o.partner_gain >= c.partner_gain and (o.gain > c.gain or o.partner_gain > c.partner_gain)
            for o in pool)]
        front.sort(key=lambda c: (-c.gain, -c.partner_gain, sum(c.trade.give)))
        return front

    # ------------------------------------------------------------ the responder and the confirmer

    def _respond(self, games, trackers, seeds, states) -> tuple[list[TradeDecision], int]:
        """Every trade prompt's decision: one critic call for every row that reads it."""
        decisions: list[TradeDecision | None] = [None] * len(games)
        plans: dict[int, tuple] = {}
        chunks: list[np.ndarray] = []
        rows_so_far = 0
        size = self.critic_encoder.size
        for i, (game, state) in enumerate(zip(games, states)):
            self._sync_turn(state, game)
            seat = game.current_player_index
            if game.current_prompt == DECIDE_TRADE:
                give, get, proposer = offer_of(game)
                trade = Trade(give, get)
                if proposer == seat:
                    decisions[i] = TradeDecision("reject", seat, trade, proposer, note="own offer")
                    continue
                if not can_pay(game.player(seat).resources, get):  # the engine lists REJECT alone: never forwarded
                    decisions[i] = TradeDecision("reject", seat, trade, proposer, note="cannot pay")
                    continue
                worlds = self._payable_worlds(game, trackers[i] if trackers is not None else None, int(seeds[i]),
                                              proposer, give)
                if not worlds:
                    decisions[i] = TradeDecision("reject", seat, trade, proposer, note="no feasible world")
                    continue
                rows = np.empty((2 * len(worlds), size), dtype=np.float32)
                for w, world in enumerate(worlds):
                    self.critic_encoder.encode_into(rewind(world), proposer, rows[2 * w])
                    self.critic_encoder.encode_into(respond_route(world, proposer, seat, trade), proposer, rows[2 * w + 1])
                plans[i] = ("respond", trade, proposer, len(worlds), rows_so_far)
            elif game.current_prompt == DECIDE_ACCEPTEES:
                give, get, _ = offer_of(game)
                trade = Trade(give, get)
                acceptees = sorted(acceptees_of(game), key=lambda c: (c - seat) % game.num_players)
                rows = np.empty((1 + len(acceptees), size), dtype=np.float32)
                self.critic_encoder.encode_into(rewind(game), seat, rows[0])
                for j, acceptee in enumerate(acceptees):
                    post = game.copy()
                    post.execute(confirm_action(game, acceptee))
                    self.critic_encoder.encode_into(post, seat, rows[1 + j])
                plans[i] = ("confirm", trade, acceptees, rows_so_far)
            else:
                raise ValueError(f"{game.current_prompt.name} is not a trade prompt")
            chunks.append(rows)
            rows_so_far += len(rows)
        if plans:
            values = self.critic(np.concatenate(chunks))[1].astype(np.float64)
        for i, plan in plans.items():
            game, state, seat = games[i], states[i], games[i].current_player_index
            if plan[0] == "respond":
                _, trade, proposer, count, offset = plan
                k = (seat - proposer) % game.num_players
                gains = [values[offset + 2 * w + 1, k] - values[offset + 2 * w, k] for w in range(count)]
                gain = float(np.mean(gains))
                kind = "accept" if gain > self.accept_margin else "reject"
                decisions[i] = TradeDecision(kind, seat, trade, proposer, gain)
            else:
                _, trade, acceptees, offset = plan
                best = None
                for j, acceptee in enumerate(acceptees):  # slot order: a tie keeps the lower slot
                    gain = float(values[offset + 1 + j, 0] - values[offset, 0])
                    if gain > self.margin and (best is None or gain > best[1]):
                        best = (acceptee, gain)
                if best is None:
                    state.rejected_bundles.add(trade)
                    decisions[i] = TradeDecision("cancel", seat, trade)
                else:
                    state.rejected_bundles.discard(trade)  # a confirmed bundle may be right again on the new hand
                    state.front = None
                    decisions[i] = TradeDecision("confirm", seat, trade, best[0], best[1])
        return decisions, rows_so_far

    def _payable_worlds(self, game: cpp.Game, tracker, seed: int, proposer: int, give: Counts) -> list[cpp.Game]:
        """Worlds in which the proposer can pay its offer (the offer says it can): ``dets``
        kept out of at most ``4 * dets`` dealt."""
        kept = []
        budget = 1 if self.oracle else 4 * self.dets
        for w in range(budget):
            world = self._worlds(game, tracker, seed, 1, start=w)[0]
            if can_pay(world.player(proposer).resources, give):
                kept.append(world)
                if len(kept) == self.dets:
                    break
        return kept
