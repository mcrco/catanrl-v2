"""Checks train_ppo, train_imitation and the eval harness make before playing."""

from __future__ import annotations

import pytest
from eval_helpers import agent, make_suite, schedule, tiny_model

from catanrl.envs.puffer import plan_env
from catanrl.eval.harness import play_schedule
from catanrl.train.puffer import build_policy_model, check_bindable

HEXAGON = {"type": "hexagon", "radius": 2}


@pytest.mark.parametrize("preset, dims, ok", [
    ("mlp", "pool", False),       # trained at the pool's dims: evals (generator bounds) cannot bind it
    ("mlp", "auto", True),
    ("graph-tiny", "pool", True),  # dims-agnostic
])
def test_networks_must_be_bindable_by_evals(preset, dims, ok):
    plan = plan_env({"scenarios": [{"game": "random-4p", "maps": 8}], "dims": dims}, pool_seed=3)
    model = build_policy_model({"model": {"preset": preset}, "obs": {"level": "private"}}, plan.dims)
    if ok:
        check_bindable(model, plan)
    else:
        with pytest.raises(ValueError, match="dims-agnostic"):
            check_bindable(model, plan)


@pytest.mark.parametrize("game, board, error", [
    ("random-4p", None, "board"),        # other dims: refused at build
    ("hexagon-3p", None, "laid out"),    # BASE dims, another tile order: refused by check_bindable
    ("base-4p", HEXAGON, "laid out"),    # and the reverse
    ("base-4p", None, None),
    ("hexagon-3p", HEXAGON, None),       # the hexagon pools train a hexagon-board cnn
])
def test_cnn_trains_on_its_boards_pools_only(game, board, error):
    """The cnn lays one board on a plane: a pool of other dims is refused at build time and
    a pool with the same dims in another tile order by check_bindable, which would
    otherwise train on scrambled boards with nothing in the harness to show for it."""
    args = {} if board is None else {"board": board}
    cfg = {"model": {"preset": "cnn-tiny", "args": args}, "obs": {"level": "private"}}
    plan = plan_env({"scenarios": [{"game": game, "maps": 4}], "dims": "auto"}, pool_seed=3)
    if error is None:
        check_bindable(build_policy_model(cfg, plan.dims), plan)
    else:
        with pytest.raises(ValueError, match=error):
            check_bindable(build_policy_model(cfg, plan.dims), plan)


def test_eval_harness_refuses_boards_a_cnn_cannot_read():
    """play_schedule checks the suite's maps against every network before playing: a BASE
    cnn on a hexagon suite (the same dims) is refused at bind, not played scrambled, and
    plays the BASE suite at those dims."""
    model = tiny_model(["hexagon-3p"], preset="cnn-tiny")  # the BASE board at hexagon-3p's dims
    lineup = [agent("POLICY", model, infer_rows=8), agent("F"), agent("F")]
    suite = make_suite(({"game": "hexagon-3p"},), turn_limit=40)
    with pytest.raises(ValueError, match="laid out"):
        play_schedule(lineup, schedule(suite, 3, boards=1), suite=suite, on_game=lambda row: None)
    suite = make_suite(({"game": "base-3p"},), turn_limit=40)
    rows = []
    play_schedule(lineup, schedule(suite, 3, boards=1), suite=suite, on_game=rows.append)
    assert len(rows) == 3


def test_catanatron_player_refuses_boards_a_cnn_cannot_read(tmp_path):
    """The Catanatron player binds the game's dims and checks its map when it mirrors a game, as
    the harness and the bridge do: a BASE cnn refuses a hexagon game (the same dims, another tile
    order) and a game of more players than its dims, and plays a BASE game."""
    from cppanatron import parity

    from catanrl.agents import resolve_agent
    from catanrl.checkpoint import save
    from catanrl.games import load_game

    model = tiny_model(["base-2p"], preset="cnn-tiny")
    config = {"env": {"scenarios": [{"game": "base-2p"}]}, "model": model.spec}
    path = save(tmp_path / "e000000.pt", model, run=None, epoch=0, agent_steps=0, config=config, code={}).path
    resolved = resolve_agent(f"POLICY:ckpt={path}", harness="catanatron")

    def catanatron_game(name, players, seed=0):
        return parity.new_game("catanatron", players, seed=seed, map_spec=load_game(name).map_spec(seed))

    for name, players, error in (("hexagon-3p", 2, "laid out"), ("base-4p", 4, "not dims-agnostic")):
        game = catanatron_game(name, players)
        with pytest.raises(ValueError, match=error):
            resolved.player(game.current_color).decide(game.game, game.playable_actions)
    game = catanatron_game("base-2p", 2)
    assert len(game.playable_actions) > 1
    assert resolved.player(game.current_color).decide(game.game, game.playable_actions) in game.playable_actions
