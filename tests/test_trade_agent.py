"""The TRADE agent (catanrl.agents.trade.TradeAgent): the proposer against a stub
critic keyed on row bytes (must-offer, must-not-offer, the ego Pareto front and its
ties), the responder and the confirmer, the rows the critic sees over real decisions,
and a recurrent policy inside it. The harness is not involved: ``drive`` steps random
4p games in Python and forwards every acting row the way the harness does, one per
stream, with the harness's seeds, streams and first flags."""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from typing import Any

import cppanatron as cpp
import numpy as np
import pytest
import torch
from catanatron.registry import SpecError
from eval_helpers import agent, tiny_model

from catanrl import checkpoint as ck
from catanrl import games
from catanrl.agents import resolve_agent
from catanrl.agents.trade import (
    Trade,
    TradeAgent,
    TradeDecision,
    _Stream,
    apply_trade,
    can_pay,
    cancel_action,
    confirm_action,
    mover_menu,
    offer_action,
    post_row_clone,
    respond_route,
    response,
    rewind,
    tradeable,
)
from catanrl.checkpoint import save
from catanrl.envs.encoding import dims_for, make_encoder
from catanrl.seeding import derive_seed

MOVER_HAND = [2, 1, 0, 0, 0]  # wood, brick, sheep, wheat, ore
PARTNER_HAND = [0, 0, 1, 2, 0]
WOOD_FOR_SHEEP = Trade((1, 0, 0, 0, 0), (0, 0, 1, 0, 0))
BRICK_FOR_WOOD = Trade((0, 1, 0, 0, 0), (1, 0, 0, 0, 0))
DECIDE_TRADE = cpp.ActionPrompt.DECIDE_TRADE
DECIDE_ACCEPTEES = cpp.ActionPrompt.DECIDE_ACCEPTEES
PLAY_TURN = cpp.ActionPrompt.PLAY_TURN
DIMS = dims_for(["base-4p"])
ENCODER = make_encoder(DIMS, "private")
POLICY = tiny_model(["base-4p"], seed=1)
CRITIC = tiny_model(["base-4p"], seed=2)
BASE = np.linspace(0.40, 0.10, 4).astype(np.float32)  # what every row the stub does not name reads


def trade_agent(**params) -> TradeAgent:
    return agent("TRADE", POLICY, critic=CRITIC, infer_rows=4, **params)


class StubCritic:
    """A critic keyed on row bytes, as tests/test_trade_oracle.py builds one: rows in
    ``table`` read their values, every other row reads ``base``. Records what it saw."""

    def __init__(self, base=BASE):
        self.base = np.asarray(base, dtype=np.float32)
        self.table: dict[bytes, np.ndarray] = {}
        self.calls: list[int] = []
        self.rows: list[np.ndarray] = []

    def set(self, row: np.ndarray, values) -> None:
        self.table[np.ascontiguousarray(row, dtype=np.float32).tobytes()] = np.asarray(values, dtype=np.float32)

    def __call__(self, rows):
        rows = np.ascontiguousarray(rows, dtype=np.float32)
        self.calls.append(len(rows))
        self.rows.extend(rows)
        return np.zeros((len(rows), 1), np.float32), np.stack([self.table.get(r.tobytes(), self.base) for r in rows])


def packaged(mover_hand=MOVER_HAND, partner_hand=PARTNER_HAND):
    """tests/test_trading.py's 4p position: the mover (seat 3, prompted last on its own
    offers) at its PLAY_TURN after the roll with MOVER_HAND, every other seat PARTNER_HAND.
    The hands are declared through apply_setup, which derives the bank from them, so a
    tracker started here deals a consistent pool (the partners' nine cards are three sheep
    and six wheat: an ask of sheep or wheat is feasible, any other never)."""
    pos = copy.deepcopy(next(p for p in ck.packaged_positions() if p["game"]["num_players"] == 4
                             and p["options"]["prompt"] == "PLAY_TURN" and p["options"]["has_rolled"]))
    mover = int(pos["options"]["current_player"])
    for seat, player in enumerate(pos["players"]):
        player["resources"] = list(mover_hand if seat == mover else partner_hand)
    game = ck._position_game(pos)
    assert mover == 3 and tradeable(game, mover)
    return game, mover


def at_prompt(game, mover, trade, responder, accepted=()):
    """A clone at ``responder``'s DECIDE_TRADE on ``trade``, the seats in ``accepted``
    (asked before it) having accepted and the others rejected."""
    g = game.copy()
    g.execute(offer_action(game, mover, trade))
    while g.current_player_index != responder:
        g.execute(response(g, accept=g.current_player_index in accepted))
    assert g.current_prompt == DECIDE_TRADE
    return g


def at_acceptees(game, mover, trade, accepted):
    g = game.copy()
    g.execute(offer_action(game, mover, trade))
    while g.current_prompt == DECIDE_TRADE:
        g.execute(response(g, accept=g.current_player_index in accepted))
    assert g.current_prompt == DECIDE_ACCEPTEES and g.current_player_index == mover
    return g


def act_one(ag, game, seat, *, tracker=None, stream=0, first=True, seed=7):
    """One forwarded row of ``seat``, as the harness forwards it: (choice, decision)."""
    row = ag.encode(game, seat)[None]
    trackers = None if not ag.needs_trackers else [tracker if tracker is not None else cpp.HandTracker(game, seat)]
    choices = ag.act(row, row[:, -ag.action_space.size:], np.array([seed], dtype=np.uint64), games=[game],
                     trackers=trackers, streams=[stream], first=[first])
    return choices[0], ag.decisions[0]


def respond_one(ag, game, *, tracker=None, stream=0, first=True, seed=7):
    seat = game.current_player_index
    trackers = None if not ag.needs_trackers else [tracker if tracker is not None else cpp.HandTracker(game, seat)]
    actions = ag.respond_trade([game], np.array([seed], dtype=np.uint64), trackers=trackers, streams=[stream],
                               first=[first])
    return actions[0], ag.decisions[0]


def core(decision: TradeDecision) -> tuple:
    return decision.kind, decision.seat, decision.bundle, decision.partner


# ---------------------------------------------------------------- the driver: real decisions without the harness


@dataclass
class Forwarded:
    """One acting row the driver forwarded: what the harness hands the agent (a copy of
    the game and of the seat's tracker, the decision seed, the stream, the first flag,
    the row) and what came back."""

    kind: str  # act | trade
    game: cpp.Game
    tracker: cpp.HandTracker
    seat: int
    seed: int
    stream: int
    first: bool
    row: np.ndarray | None
    choice: Any
    decision: TradeDecision
    rows_read: int  # critic rows this row's call read


def drive(ag, *, count=2, seed=0, turn_limit=40) -> list[Forwarded]:
    """Random 4p games with every seat played by ``ag`` and the trackers fed from the start:
    forced moves play alone, every other row is forwarded as the harness forwards it (one
    row per call; strict evaluation makes a batch of one the same as any batch). Returns
    every forwarded row in order, so a fresh agent can replay the games."""
    config = games.load_game("base-4p")
    size = ag.action_space.size
    out = []
    for i in range(count):
        game_seed = 100 + seed + i
        game = config.cppanatron_game(map_seed=seed + i, seed=game_seed)
        trackers = [cpp.HandTracker(game, s, ag.tracker_mode) for s in range(4)]
        plies = [0] * 4
        while game.winning_color is None and game.num_turns < turn_limit:
            actions = game.playable_actions
            if not actions:
                break
            seat = game.current_player_index
            if len(actions) == 1:
                step(game, trackers, actions[0])
                continue
            stream, first = i * 4 + seat, plies[seat] == 0
            decision_seed = derive_seed(game_seed, "decide", 0, seat, plies[seat])
            plies[seat] += 1
            seeds = np.array([decision_seed], dtype=np.uint64)
            given = [trackers[seat]] if ag.needs_trackers else None
            snapshot = (game.copy(), trackers[seat].copy())
            read = ag.stats["critic_rows"]
            if game.current_prompt in (DECIDE_TRADE, DECIDE_ACCEPTEES):
                action = ag.respond_trade([game], seeds, trackers=given, streams=[stream], first=[first])[0]
                out.append(Forwarded("trade", *snapshot, seat, decision_seed, stream, first, None, action,
                                     ag.decisions[0], ag.stats["critic_rows"] - read))
            else:
                row = ag.encode(game, seat)[None]
                choice = ag.act(row, row[:, -size:], seeds, games=[game], trackers=given, streams=[stream],
                                first=[first])[0]
                action = choice if isinstance(choice, cpp.Action) else ag.action_space.decode(game, int(choice))
                out.append(Forwarded("act", *snapshot, seat, decision_seed, stream, first, row[0], choice,
                                     ag.decisions[0], ag.stats["critic_rows"] - read))
            step(game, trackers, action)
    return out


def step(game, trackers, action) -> None:
    before = game.copy()
    game.execute(action)
    for tracker in trackers:
        tracker.observe(before, action, game)


def replay(ag, log: list[Forwarded], variant=lambda f: f.game) -> list[tuple[Any, TradeDecision]]:
    """The log's rows forwarded again to ``ag`` (fresh), each game replaced by
    ``variant(forwarded)``: (choice, decision) per row."""
    out = []
    size = ag.action_space.size
    for f in log:
        game = variant(f)
        seeds = np.array([f.seed], dtype=np.uint64)
        given = [f.tracker] if ag.needs_trackers else None
        if f.kind == "trade":
            choice = ag.respond_trade([game], seeds, trackers=given, streams=[f.stream], first=[f.first])[0]
        else:
            row = ag.encode(game, f.seat)[None]
            choice = ag.act(row, row[:, -size:], seeds, games=[game], trackers=given, streams=[f.stream],
                            first=[f.first])[0]
        out.append((choice, ag.decisions[0]))
    return out


# ---------------------------------------------------------------- construction


def test_the_agent_is_a_policy_with_a_critic_and_per_seat_state():
    ag = trade_agent()
    assert ag.key == "TRADE" and ag.needs_games and ag.needs_trackers and ag.needs_memory
    assert ag.tracker_mode == cpp.TrackerMode.POOL and ag.level == "private" and ag.critic_level == "private"
    assert ag.policy.key == "POLICY" and not ag.policy.needs_memory and ag.evaluator is ag.policy.evaluator
    assert ag.critic.heads == "both" and ag.critic.model is not ag.policy.model
    assert ag.shapes == ((1, 1), (2, 1), (1, 2)) and ag.offers == 3 and ag.margin == 0.0 and ag.dets == 8
    one = agent("TRADE", POLICY, infer_rows=4)  # no critic given: the policy's own value head
    assert one.critic.model is one.policy.model
    oracle = trade_agent(oracle=True, dets=1)
    assert not oracle.needs_trackers and oracle.oracle
    assert trade_agent(tracker="exact").tracker_mode == cpp.TrackerMode.EXACT
    assert trade_agent(shapes="21").shapes == ((2, 1),) and trade_agent(margin=2.0).margin == pytest.approx(0.02)
    with pytest.raises(ValueError, match="audit"):
        trade_agent(oracle=True, dets=1, audit=True)
    with pytest.raises(ValueError, match="private"):
        agent("TRADE", POLICY, critic=tiny_model(["base-4p"], level="full", seed=3), infer_rows=4)
    with pytest.raises(ValueError, match="dets"):
        trade_agent(dets=0)
    game, mover = packaged()
    row = ENCODER.encode(game, mover)[None]
    with pytest.raises(ValueError, match="games"):
        trade_agent().act(row, row[:, -ag.action_space.size:], np.array([1], dtype=np.uint64))
    with pytest.raises(ValueError, match="streams"):
        trade_agent().act(row, row[:, -ag.action_space.size:], np.array([1], dtype=np.uint64), games=[game],
                          trackers=[cpp.HandTracker(game, mover)])
    with pytest.raises(ValueError, match="first"):
        act_one(trade_agent(), game, mover, first=False)  # a stream's first row must say so
    with pytest.raises(ValueError, match="trade prompt"):
        respond_one(trade_agent(), game)


# ---------------------------------------------------------------- the proposer


def test_must_offer_the_one_bundle_the_critic_likes_for_both(monkeypatch):
    """A stub that gains the mover and the slot-1 partner 3 pp on exactly one post row
    (wood>sheep): act returns that OFFER_TRADE, reads the critic once on the pre row plus
    one post row per feasible candidate, and never runs the policy."""
    game, mover = packaged()
    partner = (mover + 1) % 4  # slot 1
    tracker = cpp.HandTracker(game, mover)
    ag = trade_agent(dets=8, min_feasible=0.1)
    stub = StubCritic()
    stub.set(ENCODER.encode(post_row_clone(game, mover, partner, WOOD_FOR_SHEEP), mover), BASE + [0.03, 0.03, 0, 0])
    ag.critic = stub
    monkeypatch.setattr(ag.policy, "act", lambda *a, **k: pytest.fail("the policy decided an offered row"))
    choice, decision = act_one(ag, game, mover, tracker=tracker)
    assert choice == offer_action(game, mover, WOOD_FOR_SHEEP)
    assert choice == cpp.Action(game.colors[mover], cpp.ActionType.OFFER_TRADE, (1, 0, 0, 0, 0, 0, 0, 1, 0, 0))
    assert core(decision) == ("offer", mover, WOOD_FOR_SHEEP, partner)
    assert decision.gain == pytest.approx(0.03, abs=1e-6) and decision.partner_gain == pytest.approx(0.03, abs=1e-6)
    assert len(decision.front) == 1 and decision.front[0].trade == WOOD_FOR_SHEEP
    # Feasibility is the payability vote over the worlds dealt from the mover's own tracker.
    worlds = [cpp.sample_world(game, tracker, seed=derive_seed(7, "world", w), deal=cpp.TrackerMode.POOL)
              for w in range(8)]
    feasible = sum(can_pay(w.player(partner).resources, WOOD_FOR_SHEEP.get) for w in worlds) / 8
    assert decision.front[0].feasibility == feasible >= 0.1
    # One strict call: the pre row (the harness's own row for the seat) then the candidates' post rows.
    candidates = [(t, q) for t in mover_menu(MOVER_HAND, 3) for q in ((mover + k) % 4 for k in (1, 2, 3))
                  if sum(can_pay(w.player(q).resources, t.get) for w in worlds) / 8 >= 0.1]
    assert stub.calls == [1 + len(candidates)] and len(candidates) > 1
    assert np.array_equal(stub.rows[0], ENCODER.encode(game, mover))
    for (t, q), row in zip(candidates, stub.rows[1:]):
        assert np.array_equal(row, ENCODER.encode(post_row_clone(game, mover, q, t), mover))
    assert ag.stats == {"evaluations": 1, "critic_rows": 1 + len(candidates), "offers": 1, "accepts": 0, "rejects": 0,
                        "confirms": 0, "cancels": 0, "no_feasible_world": 0, "audits": 0, "audit_divergences": 0}
    state = ag._streams[0]
    assert state.offers_this_turn == 1 and state.rejected_bundles == {WOOD_FOR_SHEEP} and state.evaluated


def test_must_not_offer_when_the_partner_loses_and_the_policy_decides():
    game, mover = packaged()
    ag = trade_agent(dets=2)
    stub = StubCritic()
    ag.critic = stub
    rows_seen = []
    for t in mover_menu(MOVER_HAND, 3):
        for q in range(4):
            if q != mover:
                row = ENCODER.encode(post_row_clone(game, mover, q, t), mover)
                stub.set(row, BASE + [0.03, -0.03, -0.03, -0.03])
                rows_seen.append(row)
    choice, decision = act_one(ag, game, mover)
    assert isinstance(choice, int) and decision == TradeDecision("policy", mover, front=())
    assert ag.action_space.decode(game, choice) in game.playable_actions
    assert len(stub.calls) == 1 and stub.calls[0] > 1 and ag.stats["offers"] == 0 and ag.stats["evaluations"] == 1
    # The policy's own choice on that row, as POLICY would make it.
    row = ENCODER.encode(game, mover)[None]
    assert choice == int(agent("POLICY", POLICY, infer_rows=4).act(row, row[:, -ag.action_space.size:],
                                                                   np.array([7], dtype=np.uint64))[0])


def test_must_not_offer_without_a_card_or_a_partner_with_one():
    """An empty mover hand or partners holding nothing: no critic call at all."""
    for hands in ("mover", "partners"):
        game, mover = packaged(**{hands.replace("partners", "partner") + "_hand": [0] * 5})
        ag = trade_agent(dets=2)
        stub = StubCritic()
        ag.critic = stub
        choice, decision = act_one(ag, game, mover)
        assert isinstance(choice, int) and stub.calls == [] and decision.kind == "policy"
        assert decision.front == (() if hands == "partners" else None)  # an empty hand is not a proposal moment
        assert ag.stats["critic_rows"] == 0 and ag.stats["offers"] == 0


def test_must_not_offer_again_in_the_turn_past_the_cap_or_the_first_read(monkeypatch):
    """offers=1: the second eligible row of the turn goes to the policy; when=first: a turn
    evaluated once is not read again, whatever the stub would now say."""
    game, mover = packaged()
    partner = (mover + 1) % 4
    liked = ENCODER.encode(post_row_clone(game, mover, partner, WOOD_FOR_SHEEP), mover)
    ag = trade_agent(dets=8, min_feasible=0.1, offers=1)
    stub = StubCritic()
    stub.set(liked, BASE + [0.03, 0.03, 0, 0])
    ag.critic = stub
    choice, decision = act_one(ag, game, mover)
    assert decision.kind == "offer" and len(stub.calls) == 1
    choice, decision = act_one(ag, game, mover, first=False)  # the offer was rejected: the same state again
    assert isinstance(choice, int) and decision == TradeDecision("policy", mover) and len(stub.calls) == 1
    ag = trade_agent(dets=8, min_feasible=0.1)
    stub = StubCritic()
    ag.critic = stub
    choice, decision = act_one(ag, game, mover)
    assert decision == TradeDecision("policy", mover, front=()) and len(stub.calls) == 1
    stub.set(liked, BASE + [0.03, 0.03, 0, 0])
    choice, decision = act_one(ag, game, mover, first=False)
    assert isinstance(choice, int) and decision == TradeDecision("policy", mover) and len(stub.calls) == 1
    # when=every reads again on a new hand only: the same hand reuses the (empty) front.
    every = trade_agent(dets=8, min_feasible=0.1, when="every")
    stub = StubCritic()
    every.critic = stub
    act_one(every, game, mover)
    stub.set(liked, BASE + [0.03, 0.03, 0, 0])
    choice, decision = act_one(every, game, mover, first=False)
    assert isinstance(choice, int) and len(stub.calls) == 1
    richer, _ = packaged(mover_hand=[2, 1, 0, 1, 0])
    stub.set(ENCODER.encode(post_row_clone(richer, mover, partner, WOOD_FOR_SHEEP), mover), BASE + [0.03, 0.03, 0, 0])
    choice, decision = act_one(every, richer, mover, first=False)
    assert core(decision) == ("offer", mover, WOOD_FOR_SHEEP, partner) and len(stub.calls) == 2


def test_must_not_offer_a_bundle_rejected_this_turn():
    game, mover = packaged()
    partner = (mover + 1) % 4
    liked = ENCODER.encode(post_row_clone(game, mover, partner, WOOD_FOR_SHEEP), mover)
    ag = trade_agent(dets=8, min_feasible=0.1)
    stub = StubCritic()
    stub.set(liked, BASE + [0.03, 0.03, 0, 0])
    ag.critic = stub
    # The bundle was offered and rejected earlier this turn: the menu leaves it out, so
    # its post row is never read and nothing else clears the margins.
    ag._streams[0] = _Stream(turn=game.num_turns, rejected_bundles={WOOD_FOR_SHEEP})
    choice, decision = act_one(ag, game, mover, first=False)
    assert isinstance(choice, int) and decision == TradeDecision("policy", mover, front=())
    assert len(stub.calls) == 1 and not any(np.array_equal(r, liked) for r in stub.rows)
    # And after its own offer the agent never repeats the bundle: the front's next member or the policy.
    ag = trade_agent(dets=8, min_feasible=0.1)
    stub = StubCritic()
    stub.set(liked, BASE + [0.03, 0.03, 0, 0])
    ag.critic = stub
    assert act_one(ag, game, mover)[1].kind == "offer"
    choice, decision = act_one(ag, game, mover, first=False)
    assert isinstance(choice, int) and decision.kind == "policy" and ag._streams[0].rejected_bundles == {WOOD_FOR_SHEEP}


def test_the_front_is_ego_pareto_ordered_and_offered_in_turn():
    """Gains in (mover, partner) pp on the mover's row: the partner per bundle is the one
    gaining the mover most (ties: the lower slot), a weakly dominated bundle is dropped, and
    the front is ordered by own gain, partner gain, fewer cards given, enumeration order.
    At private a 1:1 trade leaves every hand size as it was, so its post row is the same
    whoever the partner is: the partners tie on the mover's gain and the lowest slot that
    can pay is offered, reading its own slot of that one row."""
    game, mover = packaged()
    q1, q2 = (mover + 1) % 4, (mover + 2) % 4
    trades = {name: Trade(give, get) for name, give, get in (
        ("wood>sheep", (1, 0, 0, 0, 0), (0, 0, 1, 0, 0)), ("brick>sheep", (0, 1, 0, 0, 0), (0, 0, 1, 0, 0)),
        ("wood>wheat", (1, 0, 0, 0, 0), (0, 0, 0, 1, 0)), ("2wood>sheep", (2, 0, 0, 0, 0), (0, 0, 1, 0, 0)),
        ("wood+brick>wheat", (1, 1, 0, 0, 0), (0, 0, 0, 1, 0)), ("brick>wheat", (0, 1, 0, 0, 0), (0, 0, 0, 1, 0)),
        ("wood+brick>sheep", (1, 1, 0, 0, 0), (0, 0, 1, 0, 0)), ("2wood>wheat", (2, 0, 0, 0, 0), (0, 0, 0, 1, 0)))}
    one_row = ENCODER.encode(post_row_clone(game, mover, q1, trades["wood>sheep"]), mover)
    assert np.array_equal(one_row, ENCODER.encode(post_row_clone(game, mover, q2, trades["wood>sheep"]), mover))
    assert not np.array_equal(ENCODER.encode(post_row_clone(game, mover, q1, trades["2wood>sheep"]), mover),
                              ENCODER.encode(post_row_clone(game, mover, q2, trades["2wood>sheep"]), mover))
    table = {  # (bundle, partner the row is built with) -> (mover gain, gain per slot 1, 2, 3)
        ("wood>sheep", q1): (3, (1, 5, 0)),           # 1:1: every partner ties at 3 and slot 1 is kept, at (3, 1)
        ("brick>sheep", q1): (3, (2, 0, 0)),          # dominates wood>sheep's (3, 1)
        ("wood>wheat", q1): (1, (4, 0, 0)),
        ("brick>wheat", q1): (5, (0.5, 0, 0)),
        ("2wood>sheep", q1): (5, (0.5, 0, 0)), ("wood+brick>wheat", q2): (5, (0, 0.5, 0)),  # 2:1: a row per partner
        ("wood+brick>sheep", q1): (2, (-1, 0, 0)), ("2wood>wheat", q1): (-1, (5, 5, 5))}   # one side loses: out
    ag = trade_agent(dets=4, min_feasible=0.01, offers=5)
    stub = StubCritic()
    for (name, q), (own, others) in table.items():
        values = BASE.copy()
        values[0] += own / 100
        values[1:] += np.array(others) / 100
        stub.set(ENCODER.encode(post_row_clone(game, mover, q, trades[name]), mover), values)
    ag.critic = stub
    tracker = cpp.HandTracker(game, mover)
    choice, decision = act_one(ag, game, mover, tracker=tracker)
    assert choice == offer_action(game, mover, trades["brick>wheat"])
    expected = [("brick>wheat", q1), ("2wood>sheep", q1), ("wood+brick>wheat", q2), ("brick>sheep", q1), ("wood>wheat", q1)]
    assert [(c.trade.label, c.partner) for c in decision.front] == expected
    assert [(round(c.gain * 100, 3), round(c.partner_gain * 100, 3)) for c in decision.front] == [
        (5, 0.5), (5, 0.5), (5, 0.5), (3, 2), (1, 4)]
    assert all(c.feasibility >= 0.01 for c in decision.front)
    # The front is offered in order, one bundle per rejected round, without another read.
    offered = [core(decision)]
    for _ in range(5):
        offered.append(core(act_one(ag, game, mover, tracker=tracker, first=False)[1]))
    assert offered == [("offer", mover, trades[n], q) for n, q in expected] + [("policy", mover, None, None)]
    assert len(stub.calls) == 1 and ag.stats["offers"] == 5
    # offers=3 stops after three rounds of the same front.
    capped = trade_agent(dets=4, min_feasible=0.01)
    capped.critic = stub
    kinds = [act_one(capped, game, mover, tracker=tracker, first=i == 0)[1].kind for i in range(4)]
    assert kinds == ["offer", "offer", "offer", "policy"]


# ---------------------------------------------------------------- the responder and the confirmer


def _responder_rows(game, mover, mid, responder, trade, tracker, seed, dets):
    """The worlds the responder keeps (the proposer can pay) and their (pre, post) rows."""
    worlds = []
    for w in range(4 * dets):
        world = cpp.sample_world(mid, tracker, seed=derive_seed(seed, "world", w), deal=cpp.TrackerMode.POOL)
        if can_pay(world.player(mover).resources, trade.give):
            worlds.append(world)
            if len(worlds) == dets:
                break
    rows = [(ENCODER.encode(rewind(w), mover), ENCODER.encode(respond_route(w, mover, responder, trade), mover))
            for w in worlds]
    return worlds, rows


@pytest.mark.parametrize("rank", [0, 1], ids=["responder-first", "responder-second"])
def test_responder_accepts_iff_its_mean_gain_clears_the_margin(rank):
    game, mover = packaged()
    responder = [s for s in range(4) if s != mover][rank]
    slot = (responder - mover) % 4
    mid = at_prompt(game, mover, WOOD_FOR_SHEEP, responder)
    tracker = cpp.HandTracker(game, responder)  # started at the pre-offer game: the proposer's hand is unknown
    worlds, rows = _responder_rows(game, mover, mid, responder, WOOD_FOR_SHEEP, tracker, 7, dets=3)
    assert 0 < len(worlds) <= 3
    # Two worlds dealing the proposer the same hand give the same private rows: one table entry.
    same_as = [next(j for j, (_, other) in enumerate(rows) if np.array_equal(other, post)) for _, post in rows]
    seen = set()
    for margin, deltas in ((1.0, [0.02] * 3), (3.0, [0.02] * 3), (0.0, [0.03, 0.03, -0.09]), (0.5, [0.03, 0.0, 0.0])):
        effective = [deltas[same_as[w]] for w in range(len(worlds))]
        expected = "accept" if np.mean(effective) > margin / 100 else "reject"
        seen.add(expected)
        ag = trade_agent(dets=3, accept_margin=margin)
        stub = StubCritic()
        for (pre, post), delta in zip(rows, effective):
            stub.set(pre, BASE)
            values = BASE.copy()
            values[slot] += delta
            values[0] -= delta  # the proposer's own slot moves the other way: never read by the responder
            stub.set(post, values)
        ag.critic = stub
        action, decision = respond_one(ag, mid, tracker=tracker)
        assert action == response(mid, accept=expected == "accept")
        assert core(decision) == (expected, responder, WOOD_FOR_SHEEP, mover)
        assert decision.gain == pytest.approx(np.mean(effective), abs=1e-6)
        assert stub.calls == [2 * len(worlds)] and ag.stats[expected + "s"] == 1
        for i, (pre, post) in enumerate(rows):
            assert np.array_equal(stub.rows[2 * i], pre) and np.array_equal(stub.rows[2 * i + 1], post)
    assert seen == {"accept", "reject"}


def test_responder_rejects_its_own_offer_and_an_offer_no_world_can_pay():
    game, mover = packaged()
    ag = trade_agent(dets=3)
    stub = StubCritic()
    ag.critic = stub
    # The mover holds wood, so the engine asks it about its own brick>wood with ACCEPT listed.
    own = at_prompt(game, mover, BRICK_FOR_WOOD, mover)
    assert any(a.action_type == cpp.ActionType.ACCEPT_TRADE for a in own.playable_actions)
    action, decision = respond_one(ag, own, tracker=cpp.HandTracker(game, mover))
    assert action == response(own, accept=False) and decision == TradeDecision("reject", mover, BRICK_FOR_WOOD, mover,
                                                                                note="own offer")
    assert stub.calls == [] and ag.stats["rejects"] == 1 and ag.stats["no_feasible_world"] == 0
    # A proposer whose public hand (one card) cannot cover the two wood it offers: no world is kept.
    poor, _ = packaged(mover_hand=[1, 0, 0, 0, 0])
    responder = (mover + 1) % 4
    two_wood = Trade((2, 0, 0, 0, 0), (0, 0, 1, 0, 0))
    mid = at_prompt(poor, mover, two_wood, responder)
    action, decision = respond_one(ag, mid, tracker=cpp.HandTracker(poor, responder))
    assert action == response(mid, accept=False)
    assert decision == TradeDecision("reject", responder, two_wood, mover, note="no feasible world")
    assert stub.calls == [] and ag.stats["rejects"] == 2 and ag.stats["no_feasible_world"] == 1
    # oracle=true judges on the true game: the true proposer cannot pay either.
    oracle = trade_agent(oracle=True, dets=1)
    oracle.critic = stub
    assert respond_one(oracle, mid)[1].note == "no feasible world"


def test_confirm_takes_the_acceptee_gaining_the_proposer_most_or_cancels():
    """Two acceptees of a 2:1 offer (their post-confirm sizes differ, so the proposer's row
    tells them apart; a 1:1 offer's acceptees share one row and the lower slot is confirmed)."""
    game, mover = packaged()
    accepted = [s for s in range(4) if s != mover][::2]  # seats 0 and 2: slots 1 and 3
    trade = Trade((1, 1, 0, 0, 0), (0, 0, 0, 1, 0))  # wood+brick>wheat
    g = at_acceptees(game, mover, trade, accepted)
    pre = ENCODER.encode(rewind(g), mover)
    posts = {}
    for seat in accepted:
        post = g.copy()
        post.execute(confirm_action(g, seat))
        posts[seat] = ENCODER.encode(post, mover)
    assert not np.array_equal(posts[0], posts[2])
    one = at_acceptees(game, mover, WOOD_FOR_SHEEP, accepted)
    shared = []
    for seat in accepted:
        post = one.copy()
        post.execute(confirm_action(one, seat))
        shared.append(ENCODER.encode(post, mover))
    assert np.array_equal(*shared)
    for own, expected in (({0: 0.01, 2: 0.02}, 2), ({0: 0.02, 2: 0.01}, 0), ({0: 0.02, 2: 0.02}, 0),  # ties: lower slot
                          ({0: -0.01, 2: -0.02}, None), ({0: 0.0, 2: 0.0}, None)):  # strict: 0 does not clear 0
        ag = trade_agent(dets=2)
        stub = StubCritic()
        stub.set(pre, BASE)
        for seat, gain in own.items():
            values = BASE.copy()
            values[0] += gain
            values[(seat - mover) % 4] -= 0.05  # the acceptee's own gain is not the confirmer's concern
            stub.set(posts[seat], values)
        ag.critic = stub
        action, decision = respond_one(ag, g, tracker=cpp.HandTracker(game, mover))
        assert stub.calls == [3]
        assert np.array_equal(stub.rows[0], pre) and np.array_equal(stub.rows[1], posts[0])
        if expected is None:
            assert action == cancel_action(g) and decision == TradeDecision("cancel", mover, trade)
            assert ag._streams[0].rejected_bundles == {trade} and ag.stats["cancels"] == 1
        else:
            assert action == confirm_action(g, expected) and core(decision) == ("confirm", mover, trade, expected)
            assert decision.gain == pytest.approx(own[expected], abs=1e-6) and ag.stats["confirms"] == 1
            assert ag._streams[0].front is None and trade not in ag._streams[0].rejected_bundles
    # A margin the best gain does not clear cancels too.
    ag = trade_agent(dets=2, margin=1.5)
    stub = StubCritic()
    stub.set(pre, BASE)
    stub.set(posts[2], BASE + [0.01, 0, 0, 0])
    ag.critic = stub
    assert respond_one(ag, g, tracker=cpp.HandTracker(game, mover))[1].kind == "cancel"
    # The 1:1 offer: one row for both acceptees, read once each; the tie goes to slot 1 (seat 0).
    ag = trade_agent(dets=2)
    stub = StubCritic()
    stub.set(shared[0], BASE + [0.02, 0, 0, 0])
    ag.critic = stub
    action, decision = respond_one(ag, one, tracker=cpp.HandTracker(game, mover))
    assert action == confirm_action(one, 0) and core(decision) == ("confirm", mover, WOOD_FOR_SHEEP, 0)


# ---------------------------------------------------------------- the rows the critic sees


def _players_block(row):
    offset, (players, width) = ENCODER.layout()["players"]
    return row[offset : offset + players * width].reshape(players, width)


def test_every_critic_row_is_a_proposers_play_turn_row_after_the_roll():
    """Over real decisions (random 4p games, trackers from the start, a tiny critic whose
    values barely move): every row the critic reads has the play_turn prompt flag alone,
    slot 0 acting = turn_owner = has_rolled = 1 (the observer is the proposer, at its
    rolled turn), no no-op bit; a proposal's pre row is the row the harness forwarded; a
    candidate's post row does not depend on the stand-in's filler kind nor on the world."""
    ag = trade_agent(dets=2, shapes="11", sample=True)
    seen = []
    real = ag.critic

    def spy(rows):
        seen.extend(np.array(rows, copy=True))
        return real(rows)

    ag.critic = spy
    log = drive(ag, count=3, seed=3, turn_limit=40)
    kinds = {f.decision.kind for f in log}
    assert {"offer", "policy", "accept", "reject"} <= kinds, kinds
    assert len(log) >= 200 and len(seen) >= 200
    globals_ = ENCODER.feature_names("global")
    players = ENCODER.feature_names("players")
    prompt_flags = [i for i, name in enumerate(globals_) if name.startswith("prompt_")]
    play_turn = globals_.index("prompt_play_turn")
    noop = ENCODER.layout()["action_mask"][0] + ag.action_space.noop
    for row in seen:
        assert row[play_turn] == 1.0 and sum(row[i] for i in prompt_flags) == 1.0
        slot0 = _players_block(row)[0]
        assert slot0[players.index("acting")] == slot0[players.index("turn_owner")] == slot0[players.index("has_rolled")] == 1.0
        assert row[noop] == 0.0
    proposals = [f for f in log if f.kind == "act" and f.decision.front]
    assert len(proposals) >= 5
    assert sum(f.rows_read for f in log) == len(seen)
    offset = 0
    for f in log:
        if f.kind == "act" and f.rows_read:
            assert np.array_equal(seen[offset], f.row)  # the pre row is the VecEnv's own row for that seat
        offset += f.rows_read
    for f in proposals[:5]:
        game, seat = f.game, f.seat
        for c in f.decision.front:
            reference = ENCODER.encode(post_row_clone(game, seat, c.partner, c.trade), seat)
            size = sum(game.player(c.partner).resources)
            for kind in range(5):  # any hand of the public size holding the ask gives the same row
                hand = list(c.trade.get)
                hand[kind] += size - sum(c.trade.get)
                clone = game.copy()
                clone.set_resources(c.partner, hand)
                assert np.array_equal(ENCODER.encode(apply_trade(clone, seat, c.partner, c.trade), seat), reference)
            for k in (1, 2, 3):
                world = cpp.sample_world(game, f.tracker, seed=k, deal=cpp.TrackerMode.POOL)
                assert np.array_equal(ENCODER.encode(post_row_clone(world, seat, c.partner, c.trade), seat), reference)


def test_one_critic_call_per_batch_and_rows_are_batch_independent():
    """Several seats' rows in one act / respond_trade call: one strict critic call for all
    their reads, and the same decisions as one row at a time."""
    ag = trade_agent(dets=2, shapes="11")
    log = drive(ag, count=1, seed=5, turn_limit=24)
    acts = [f for f in log if f.kind == "act" and tradeable(f.game, f.seat) and sum(f.game.player(f.seat).resources)]
    assert len(acts) >= 2
    batch = [dataclasses.replace(f, first=True) for f in acts[:3]]  # fresh streams: each row reads a front
    fresh = trade_agent(dets=2, shapes="11")
    calls = []
    real = fresh.critic

    def spy(rows):
        calls.append(len(rows))
        return real(rows)

    fresh.critic = spy
    rows = np.stack([f.row for f in batch])
    choices = fresh.act(rows, rows[:, -ag.action_space.size:], np.array([f.seed for f in batch], dtype=np.uint64),
                        games=[f.game for f in batch], trackers=[f.tracker for f in batch],
                        streams=[100 + i for i in range(len(batch))], first=[True] * len(batch))
    assert len(calls) == 1 and calls[0] == fresh.stats["critic_rows"] > len(batch)
    together = list(zip(choices, fresh.decisions))
    alone = replay(trade_agent(dets=2, shapes="11"), batch)
    assert together == alone


# ---------------------------------------------------------------- a recurrent policy inside TRADE


LSTM = tiny_model(["base-4p"], preset="lstm-tiny", seed=5)


def lstm_agent(**params):
    return agent("TRADE", LSTM, critic=CRITIC, infer_rows=4, **params)


def test_recurrent_policy_memory_is_stepped_by_policy_rows_only():
    ag = lstm_agent(dets=2, shapes="11", sample=True)
    assert ag.policy.needs_memory and ag.recurrent and ag.needs_memory
    stepped = []
    real = ag.policy.evaluator.step

    def spy(rows, h, c):
        stepped.append(len(rows))
        return real(rows, h, c)

    ag.policy.evaluator.step = spy
    log = drive(ag, count=1, seed=4, turn_limit=36)
    policy_rows = sum(1 for f in log if f.decision.kind == "policy")
    assert sum(stepped) == policy_rows and policy_rows < len(log)
    assert any(f.kind == "trade" for f in log) and any(f.decision.kind == "offer" for f in log)
    # Replayed through a fresh agent: a trade prompt or an offered row leaves the stream's memory alone.
    fresh = lstm_agent(dets=2, shapes="11", sample=True)
    size = fresh.action_space.size
    untouched = 0
    for f in log:
        before = tuple(m.copy() for m in fresh.policy._memory.get(f.stream, ()))
        seeds = np.array([f.seed], dtype=np.uint64)
        if f.kind == "trade":
            fresh.respond_trade([f.game], seeds, trackers=[f.tracker], streams=[f.stream], first=[f.first])
        else:
            row = fresh.encode(f.game, f.seat)[None]
            fresh.act(row, row[:, -size:], seeds, games=[f.game], trackers=[f.tracker], streams=[f.stream],
                      first=[f.first])
        after = tuple(m.copy() for m in fresh.policy._memory.get(f.stream, ()))
        if fresh.decisions[0].kind != "policy":
            assert len(before) == len(after) and all(np.array_equal(a, b) for a, b in zip(before, after))
            untouched += 1
    assert untouched > 0


def test_a_stream_opened_by_a_trade_prompt_then_decides_from_the_zero_state():
    game, mover = packaged()
    responder = (mover + 1) % 4
    ag = lstm_agent(dets=2)
    stub = StubCritic()
    ag.critic = stub
    mid = at_prompt(game, mover, WOOD_FOR_SHEEP, responder)
    decision = respond_one(ag, mid, tracker=cpp.HandTracker(game, responder), stream=responder, first=True)[1]
    assert decision.kind in ("accept", "reject") and responder not in ag.policy._memory
    # The responder's own rolled turn next: END_TURN, its forced roll, then its PLAY_TURN row.
    g = game.copy()
    g.execute(next(a for a in g.playable_actions if a.action_type == cpp.ActionType.END_TURN))
    while len(g.playable_actions) == 1 or g.current_player_index != responder or g.current_prompt != PLAY_TURN:
        g.execute(g.playable_actions[0])
    assert tradeable(g, responder)
    choice, decision = act_one(ag, g, responder, stream=responder, first=False)
    assert decision == TradeDecision("policy", responder, front=()) and responder in ag.policy._memory
    row = ENCODER.encode(g, responder)
    with torch.no_grad():
        logits = LSTM.step(torch.from_numpy(row[None]), *LSTM.initial_state(1))[0].numpy()[0]
    mask = row[-ag.action_space.size:]
    assert choice == int(np.where(mask > 0, logits, -np.inf).argmax())


def test_a_recurrent_critic_is_refused(tmp_path):
    with pytest.raises(ValueError, match="recurrent"):
        agent("TRADE", POLICY, critic=LSTM, infer_rows=4)
    with pytest.raises(ValueError, match="recurrent"):
        agent("TRADE", LSTM, infer_rows=4)  # one network: the critic would be the lstm's value head
    graph = save(tmp_path / "graph.pt", POLICY, run=None, epoch=0, agent_steps=0, config={"model": POLICY.spec},
                 code={}).path
    lstm = save(tmp_path / "lstm.pt", LSTM, run=None, epoch=0, agent_steps=0, config={"model": LSTM.spec}, code={}).path
    with pytest.raises(SpecError, match="recurrent"):
        resolve_agent(f"TRADE:ckpt={graph}:critic={lstm}")
    resolved = resolve_agent(f"TRADE:ckpt={lstm}:critic={graph}")
    assert resolved.fair and resolved.native(device="cpu", infer_rows=4).policy.needs_memory
    assert isinstance(resolve_agent(f"TRADE:ckpt={graph}").native(device="cpu", infer_rows=4), TradeAgent)
