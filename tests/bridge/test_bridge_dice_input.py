"""The dice input (``obs.dice_state``) through the bridges: a network with it decides on rows
whose dice features are the replayed game's, which follows the table's dice.

* The site bridge (ViewAgent on catanatron-server views): on balanced dice (``base-2p-fr``,
  the house's 1v1 under OpenHex's rules) every row the agent is handed carries the true
  game's dice state (each seat's next-roll odds, the seat's own first, the deck, the window,
  the sevens, and each tile's own number's), on fair dice (``base-2p``) a fresh state's.
* The OpenHex bridge: after a balanced game's events (the engine's draws, written as
  OpenHex events) the seat's replay holds the generator's dice state, and the row its
  agent decides on carries that state's features.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("catanatron")

from bridge_games import lockstep
from dice_features import DiceColumns
from test_bridge_openhex_dice import _balanced_game, _redacted

from catanrl.agents.native import make_native_agent
from catanrl.bridge.agent import ViewAgent
from catanrl.bridge.openhex import OpenHexSeat
from catanrl.envs.encoding import FAIR_ODDS, action_size, fresh_dice_features, row_layout
from catanrl.models import build_model


def dice_agent(dims: dict):
    """POLICY on a fresh private graph-tiny network with the dice input (fair on views)."""
    torch.manual_seed(0)
    model = build_model({"preset": "graph-tiny"}, dims=dict(dims), levels={"level": "private", "dice_state": True},
                        num_actions=action_size(dims), layout=row_layout(dims, "private", None, True))
    agent = make_native_agent(SimpleNamespace(key="POLICY", params={}, checkpoint=None), device="cpu",
                              infer_rows=1, model=model)
    assert agent.dice_state and agent.encoder.dice_state
    rows = []
    act = agent.act

    def spy(rows_, masks, seeds, **kwargs):
        rows.extend(np.array(r) for r in rows_)
        return act(rows_, masks, seeds, **kwargs)

    agent.act = spy
    return agent, rows


@pytest.mark.parametrize("name", ["base-2p-fr", "base-2p"])
def test_the_site_bridge_hands_the_true_dice_state(name):
    ls = lockstep(name, 3, site=True)
    agent, rows = dice_agent(ls.dims)
    columns = DiceColumns(agent.encoder)
    fresh = np.array(fresh_dice_features(columns.global_names, 2)[-len(columns.global_dice):], dtype=np.float32)
    ours = ViewAgent(agent, strict=True)
    checked = unfair = aged = 0
    while checked < 40 and not ls.over:
        seat = ls.current_seat
        playable = ls.playable_json()
        if len(playable) > 1:
            before = len(rows)
            ours.decide(ls.view(seat), playable)
            assert len(rows) == before + 1
            columns.check(ls.native, seat, rows[-1])  # every dice feature, global and per tile
            unfair += not np.allclose(ls.native.roll_probabilities(seat), FAIR_ODDS)
            aged += not np.array_equal(columns.glob(rows[-1]), fresh)
            checked += 1
        ls.step(ls.choose())
    assert checked == 40
    assert (unfair > 0) == (aged > 0) == (name == "base-2p-fr")


@pytest.mark.parametrize("color", ["blue", "red"])
def test_the_openhex_bridge_replays_the_dice_its_agent_reads(color):
    record, events, _after, generator = _balanced_game(rolls=40)
    seat = OpenHexSeat(None, "dice", record, color)
    seat.feed(_redacted(events, color))
    game = seat.replay.game
    assert game.balanced_dice == generator.balanced_dice and "dice_desync" not in seat.stats
    agent, rows = dice_agent(_dims(game))
    columns = DiceColumns(agent.encoder)
    seat.native = agent
    me = seat.seat
    row = agent.encode(game, me)
    columns.check(generator, me, row)  # the generator's dice state, the seat's own first
    assert columns.glob(row)[columns.global_dice.index("dice_draws_left")] < 1.0  # 40 rolls in
    # A decision of the seat's: roll on a copy until it has a choice, then the row it is handed.
    play = game.copy()
    while play.current_player_index != me or len(play.playable_actions) < 2:
        play.execute(play.playable_actions[0])
    seat._choose(play, None)
    assert len(rows) == 1 and np.array_equal(rows[0][:-agent.action_space.size],
                                             agent.encode(play, me)[:-agent.action_space.size])
    columns.check(play, me, rows[0])


def _dims(game) -> dict:
    from catanrl.envs.encoding import covering_dims

    return covering_dims([game.board.map], game.num_players)
