"""The native harness with TRADE seats: offers and trade answers played through
``VecEnv.set_action``, trade prompts routed by their empty mask, the plain agents
defaulting at them, the ``trades`` row field, and rows that do not depend on
num_envs, threads, workers or order and equal the reference stepper's."""

from __future__ import annotations

import cppanatron as cpp
import pytest
from eval_helpers import agent, make_suite, schedule, tiny_model
from reference_stepper import play_reference

from catanrl.eval.harness import TRADE_COUNTS, play_schedule

SUITE = make_suite(({"game": "base-3p"}, {"game": "base-4p"}), turn_limit=60)
MODEL = tiny_model(["base-3p", "base-4p"], seed=1)  # TRADE's policy, private
CRITIC = tiny_model(["base-3p", "base-4p"], seed=2)
FULL = tiny_model(["base-3p", "base-4p"], level="full", seed=3)  # the plain agents re-encode their rows
ANSWERS = (cpp.ActionType.REJECT_TRADE, cpp.ActionType.CANCEL_TRADE)


def trade(**params):
    return agent("TRADE", MODEL, critic=CRITIC, infer_rows=8, dets=3, **params)


def plain(key, **params):
    return agent(key, FULL, infer_rows=8, **params)


# Two tables: TRADE against a responder-only TRADE and R (F fills the fourth seat), and TRADE
# against the plain agents, which answer its offers by NativeAgent.respond_trade's default.
CASES = {
    "trade-vs-trade": (lambda: [trade(), trade(propose=False), agent("R")], schedule(SUITE, 3, boards=1), "F"),
    "trade-vs-plain": (lambda: [trade(), plain("POLICY"), plain("MCTS", sims=4), plain("BELIEF", worlds=2)],
                       schedule(SUITE, 4, boards=1), None),
}
REFERENCE: dict[str, dict] = {}


def play(players, games, fill=None, **kwargs):
    rows = []
    play_schedule(players, games, suite=SUITE, on_game=rows.append, fill=None if fill is None else agent(fill),
                  **kwargs)
    return {row["game_id"]: row for row in rows}


def reference(case):
    """The case's rows at (16, 1), played once for every test of the module."""
    lineup, games, fill = CASES[case]
    if case not in REFERENCE:
        REFERENCE[case] = play(lineup(), games, fill, num_envs=16, threads=1)
    return REFERENCE[case]


def counted(player):
    """``player.respond_trade`` wrapped to log its calls (the plain agents' defaults at a trade prompt)."""
    log = []
    inner = player.respond_trade

    def respond_trade(games, seeds, **kwargs):
        actions = inner(games, seeds, **kwargs)
        log.append([a.action_type for a in actions])
        return actions

    player.respond_trade = respond_trade
    return log


@pytest.mark.parametrize("case", sorted(CASES))
def test_trade_lineups_play_to_completion_with_trades(case):
    """Every game plays out; offers come only from the proposer's seats, accepts only from a
    TRADE responder's, and ``trades`` is on the rows of games with a trade step and on no other."""
    games = CASES[case][1]
    rows = reference(case)
    assert sorted(rows) == sorted(g.game_id for g in games)
    for g in games:
        assert rows[g.game_id]["seats"] == list(g.seats)
    assert all(not r["stuck"] for r in rows.values())
    traded = [r for r in rows.values() if "trades" in r]
    assert traded, "the proposer never traded"
    for row in rows.values():
        if "trades" not in row:
            continue
        assert list(row["trades"]) == list(TRADE_COUNTS.values())
        for seat, slot in enumerate(row["seats"]):
            offers, accepts, confirms, cancels = (row["trades"][k][seat] for k in TRADE_COUNTS.values())
            assert offers == 0 or slot == 0  # the only proposer
            assert accepts == 0 or (case == "trade-vs-trade" and slot == 1)  # the only accepting responder
            assert confirms + cancels <= offers and (case == "trade-vs-trade" or confirms == cancels == 0)
            assert row["decisions"][seat] >= offers + accepts + confirms + cancels
        assert any(any(v) for v in row["trades"].values())
    assert sum(r["trades"]["offers"][s] for r in traded for s in range(len(r["seats"]))) > 0
    if case == "trade-vs-trade":
        assert sum(r["trades"]["confirms"][s] for r in traded for s in range(len(r["seats"]))) > 0


def test_plain_agents_default_at_every_trade_prompt():
    """POLICY, MCTS and BELIEF are prompted on the TRADE seat's offers (their rows reach the
    harness with no legal index) and answer with the default REJECT, never raising."""
    lineup, games, _fill = CASES["trade-vs-plain"]
    players = lineup()
    logs = [counted(p) for p in players[1:]]
    rows = play(players, games, num_envs=4, threads=2)
    assert rows == reference("trade-vs-plain")
    for log in logs:
        assert log and all(kind in ANSWERS for kinds in log for kind in kinds)
    offers = sum(r["trades"]["offers"][s] for r in rows.values() if "trades" in r for s in range(len(r["seats"])))
    assert players[0].stats["offers"] == offers > 0 and players[0].stats["accepts"] == 0
    assert players[0].stats["confirms"] == players[0].stats["cancels"] == 0  # nobody accepted


def test_trade_rows_count_what_the_agents_did():
    """A row's per-seat counts are the agents' own counters summed over their seats: the
    proposer's offers, confirms and cancels (every DECIDE_ACCEPTEES answer, so a record's
    cancel rate is the pilot's) and the TRADE responder's accepts; R's accepts happen inside
    the engine and are not a learner's decision."""
    lineup, games, fill = CASES["trade-vs-trade"]
    players = lineup()
    rows = play(players, games, fill, num_envs=4, threads=2)
    assert rows == reference("trade-vs-trade")

    def total(slot: int, kind: str) -> int:
        return sum(r["trades"][kind][seat] for r in rows.values() if "trades" in r
                   for seat, s in enumerate(r["seats"]) if s == slot)

    proposer, responder = players[0], players[1]
    assert total(0, "offers") == proposer.stats["offers"] > 0
    assert total(0, "confirms") == proposer.stats["confirms"] > 0
    assert total(0, "cancels") == proposer.stats["cancels"] > 0
    assert total(1, "accepts") == responder.stats["accepts"] > 0
    assert total(0, "accepts") == proposer.stats["accepts"] == 0  # prompted on its own offers only: REJECT
    assert all(total(slot, kind) == 0 for slot in (1, 2) for kind in ("offers", "confirms", "cancels"))


@pytest.mark.parametrize("case", sorted(CASES))
def test_rows_are_invariant(case):
    lineup, games, fill = CASES[case]
    expected = reference(case)
    for num_envs, threads in ((1, 1), (5, 4)):
        assert play(lineup(), games, fill, num_envs=num_envs, threads=threads) == expected
    half = len(games) // 2
    split = {**play(lineup(), games[:half], fill, num_envs=3), **play(lineup(), games[half:], fill, num_envs=7)}
    assert split == expected
    assert play(lineup(), list(reversed(games)), fill, num_envs=4) == expected


@pytest.mark.parametrize("case", sorted(CASES))
def test_rows_equal_the_reference_stepper(case):
    """The games stepped one action at a time in Python, trade prompts answered by respond_trade
    and offers executed as the cpp.Action act returned, give exactly the harness's rows (the
    trades field and the decision counts, so the harness's ply is the engine's)."""
    lineup, games, fill = CASES[case]
    stepped = play_reference(lineup(), games, suite=SUITE, fill=None if fill is None else agent(fill))
    assert {row["game_id"]: row for row in stepped} == reference(case)


def test_sampled_trade_replays_from_its_decision_seeds():
    games = [g for g in schedule(SUITE, 3, boards=1) if g.scenario == 0]
    lineup = lambda: [trade(sample=True, temperature=1.0), trade(propose=False, sample=True), agent("R")]
    rows = play(lineup(), games, num_envs=1)
    assert any("trades" in r for r in rows.values())
    assert play(lineup(), games, num_envs=3, threads=2) == rows


def test_trade_against_f_reproduces_policy_against_f():
    """F rejects every offer and a rejected offer restores the state, so [TRADE, F, F, F] plays
    [POLICY, F, F, F]'s games with the same network: the outcome fields are equal while the
    TRADE seat's decisions exceed the policy's by its offers and the own-offer prompts it
    answered (a proposer at seat > 0 is asked on its own offer when it can pay the ask)."""
    suite = make_suite(({"game": "base-4p"},), turn_limit=60)
    games = schedule(suite, 4, boards=1)
    proposer = trade()
    traded, policy = [], []
    play_schedule([proposer, agent("F"), agent("F"), agent("F")], games, suite=suite, on_game=traded.append)
    play_schedule([agent("POLICY", MODEL, infer_rows=8), agent("F"), agent("F"), agent("F")], games, suite=suite,
                  on_game=policy.append)
    traded = {r["game_id"]: r for r in traded}
    policy = {r["game_id"]: r for r in policy}
    assert sorted(traded) == sorted(policy) == sorted(g.game_id for g in games)
    extra = 0
    for game_id, row in policy.items():
        other = traded[game_id]
        assert "trades" not in row
        for key in ("seats", "winner", "winner_seat", "turns", "truncated", "stuck", "scores", "forfeits"):
            assert other[key] == row[key], (game_id, key)
        seat = row["seats"].index(0)
        offers = other["trades"]["offers"][seat] if "trades" in other else 0
        assert other["decisions"][seat] - row["decisions"][seat] >= offers
        extra += other["decisions"][seat] - row["decisions"][seat]
        assert [d for s, d in enumerate(other["decisions"]) if s != seat] == [0] * 3
        if "trades" in other:
            assert all(other["trades"][k] == [0] * 4 for k in ("accepts", "confirms", "cancels"))
    stats = proposer.stats
    assert stats["offers"] > 0 and stats["accepts"] == stats["confirms"] == stats["cancels"] == 0
    assert extra == stats["offers"] + stats["rejects"]  # F never offers: every reject is of its own offer
