"""The outcome head (``model.args.outcome``) through the bridges: training only, so an agent
on a network with it decides as the same weights without it, on the same rows: the site
bridge (ViewAgent on catanatron-server views) and the OpenHex bridge (a seat's replayed game)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("catanatron")

from bridge_games import lockstep
from test_bridge_dice_input import _dims
from test_bridge_openhex_dice import _balanced_game, _redacted

from catanrl.agents.native import make_native_agent
from catanrl.bridge.agent import ViewAgent
from catanrl.bridge.openhex import OpenHexSeat
from catanrl.envs.encoding import action_size, row_layout
from catanrl.models import build_model


def agents(dims: dict):
    """POLICY on a private graph-tiny network with the dice input and the outcome head, and on
    the same weights without the head; each with the (rows, choices) it saw and made."""
    out = []
    for outcome in (True, False):
        torch.manual_seed(0 if outcome else 1)
        model = build_model({"preset": "graph-tiny", "args": {"shared": True, **({"outcome": True} if outcome else {})}},
                            dims=dict(dims), levels={"level": "private", "dice_state": True},
                            num_actions=action_size(dims), layout=row_layout(dims, "private", None, True))
        if not outcome:
            model.load_state_dict({k: v for k, v in out[0][0].model.state_dict().items()
                                   if not k.startswith("outcome_head.")})
        agent = make_native_agent(SimpleNamespace(key="POLICY", params={}, checkpoint=None), device="cpu",
                                  infer_rows=1, model=model)
        agent.model = model
        seen = []
        act = agent.act

        def spy(rows, masks, seeds, _act=act, _seen=seen, **kwargs):
            choice = _act(rows, masks, seeds, **kwargs)
            _seen.append((np.array(rows), np.array(choice)))
            return choice

        agent.act = spy
        out.append((agent, seen))
    assert out[0][0].model.outcome and not out[1][0].model.outcome
    return out


def assert_same(seen_a, seen_b, count: int) -> None:
    assert len(seen_a) == len(seen_b) == count
    for (rows_a, choice_a), (rows_b, choice_b) in zip(seen_a, seen_b):
        assert np.array_equal(rows_a, rows_b) and np.array_equal(choice_a, choice_b)


def test_the_site_bridge_decides_as_without_the_head():
    ls = lockstep("base-2p-fr", 3, site=True)
    (with_head, seen_a), (without, seen_b) = agents(ls.dims)
    ours, theirs = ViewAgent(with_head, strict=True), ViewAgent(without, strict=True)
    checked = 0
    while checked < 30 and not ls.over:
        seat = ls.current_seat
        playable = ls.playable_json()
        if len(playable) > 1:
            assert ours.decide(ls.view(seat), playable) == theirs.decide(ls.view(seat), playable)
            checked += 1
        ls.step(ls.choose())
    assert_same(seen_a, seen_b, 30)


@pytest.mark.parametrize("color", ["blue", "red"])
def test_the_openhex_bridge_decides_as_without_the_head(color):
    record, events, _after, _generator = _balanced_game(rolls=40)
    seat = OpenHexSeat(None, "outcome", record, color)
    seat.feed(_redacted(events, color))
    game = seat.replay.game
    (with_head, seen_a), (without, seen_b) = agents(_dims(game))
    play = game.copy()
    while play.current_player_index != seat.seat or len(play.playable_actions) < 2:
        play.execute(play.playable_actions[0])
    for agent in (with_head, without):
        seat.native = agent
        seat._choose(play.copy(), None)
    assert_same(seen_a, seen_b, 1)
