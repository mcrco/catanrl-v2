"""TRADE's fairness (design section 5), as a black-box property: the agent's choices and
scores over real decisions are identical when every true game it is given is replaced by
any game with the same public state (a world dealt from its own tracker under another
seed, or one whose other seats' hands are replaced by hands of the same size);
``audit=true`` checks that at runtime and raises when a true hand is read;
``oracle=true`` reads the true hands, is unfair at resolution and decides otherwise."""

from __future__ import annotations

import cppanatron as cpp
import pytest
from test_trade_agent import POLICY, TradeAgent, drive, replay, trade_agent

from catanrl.agents import resolve_agent
from catanrl.checkpoint import save

PARAMS = {"dets": 2, "shapes": "11", "sample": True}


@pytest.fixture(scope="module")
def log():
    """Every forwarded row of two random 4p games played by one fair TRADE agent."""
    rows = drive(trade_agent(**PARAMS), count=2, seed=11, turn_limit=36)
    kinds = {f.decision.kind for f in rows}
    assert {"offer", "policy", "reject"} <= kinds and ({"accept", "confirm"} & kinds), kinds
    return rows


def world(k: int):
    """The forwarded game replaced by a world of its public state dealt from the seat's own
    tracker under seed ``k`` (its own hand exact, every other hand re-dealt)."""
    return lambda f: cpp.sample_world(f.game, f.tracker, seed=k, deal=cpp.TrackerMode.POOL)


def other_hands_exchanged(f):
    """The forwarded game with one card exchanged between two other seats (a kind one holds
    for a kind the other holds): every hand size, the bank and the hands' totals stay, so the
    public state is the same and the engine still deals worlds of it; the two hidden hands
    differ. Games where no such pair exists come back as they are."""
    game = f.game.copy()
    others = [s for s in range(game.num_players) if s != f.seat]
    hands = {s: list(game.player(s).resources) for s in others}
    for a in others:
        for b in others:
            for i in range(5):
                for j in range(5):
                    if a != b and i != j and hands[a][i] > 0 and hands[b][j] > 0:
                        hands[a][i], hands[a][j] = hands[a][i] - 1, hands[a][j] + 1
                        hands[b][j], hands[b][i] = hands[b][j] - 1, hands[b][i] + 1
                        game.set_resources(a, hands[a])
                        game.set_resources(b, hands[b])
                        return game
    return game


def test_choices_and_scores_do_not_depend_on_any_hidden_hand(log):
    truth = replay(trade_agent(**PARAMS), log)
    assert truth == [(f.choice, f.decision) for f in log]  # the replay reproduces the games' decisions
    for k in (1, 2, 3):
        assert replay(trade_agent(**PARAMS), log, world(k)) == truth
    exchanged = [other_hands_exchanged(f) for f in log]
    assert sum(not g.search_equivalent(f.game) for g, f in zip(exchanged, log)) > len(log) // 2
    assert replay(trade_agent(**PARAMS), log, other_hands_exchanged) == truth
    assert any(f.decision.front for f in log) and any(f.decision.gain is not None for f in log)


def test_audit_passes_on_real_games_and_catches_a_true_hand_read(monkeypatch):
    ag = trade_agent(audit=True, **PARAMS)
    rows = drive(ag, count=1, seed=11, turn_limit=30)
    assert ag.stats["audits"] == len(rows) > 0 and ag.stats["audit_divergences"] == 0
    assert any(f.kind == "trade" for f in rows) and any(f.decision.kind == "offer" for f in rows)
    # An agent that reads the true hands (the true game in place of every dealt world).
    monkeypatch.setattr(TradeAgent, "_worlds", lambda self, game, tracker, seed, count, start=0: [game])
    cheat = trade_agent(audit=True, **PARAMS)
    with pytest.raises(RuntimeError, match="audit"):
        drive(cheat, count=2, seed=11, turn_limit=36)
    assert cheat.stats["audit_divergences"] == 1


def test_oracle_is_unfair_and_decides_from_the_true_hands(log, tmp_path):
    ref = save(tmp_path / "net.pt", POLICY, run=None, epoch=0, agent_steps=0, config={"model": POLICY.spec}, code={})
    assert resolve_agent(f"TRADE:ckpt={ref.path}").fair
    assert not resolve_agent(f"TRADE:ckpt={ref.path}:oracle=true:dets=1").fair
    truth = [(f.choice, f.decision) for f in log]
    oracle = replay(trade_agent(oracle=True, dets=1, shapes="11", sample=True), log)
    assert any(a != b for a, b in zip(oracle, truth))
    assert replay(trade_agent(oracle=True, dets=1, shapes="11", sample=True), log, other_hands_exchanged) != oracle
