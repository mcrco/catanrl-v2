"""Local stand-ins for catanrl.eval.suites / catanrl.agents.spec in the
learning-side eval tests: a ScheduledGame with the documented fields, a
small schedule (every board in every seat rotation) and resolved-agent
records for make_native_agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from catanrl import games as game_configs
from catanrl.agents.native import make_native_agent
from catanrl.envs.encoding import action_size, dims_for, row_layout
from catanrl.envs.puffer import _derive_seed
from catanrl.models import build_model


@dataclass(frozen=True)
class ScheduledGame:
    game_id: str
    scenario: int
    board: int
    rotation: int
    num_players: int
    map_seed: int
    game_seed: int
    bot_seed: int
    map_hash: str
    seats: tuple[int, ...]


@dataclass(frozen=True)
class Suite:
    scenarios: tuple
    turn_limit: int = 500


@dataclass(frozen=True)
class Resolved:
    key: str
    params: Any = None
    checkpoint: Any = None


def schedule(suite: Suite, lineup_size: int, boards: int, seed: int = 7) -> list[ScheduledGame]:
    out = []
    for scenario, entry in enumerate(suite.scenarios):
        n = game_configs.load_game(entry["game"]).num_players
        for board in range(boards):
            root = _derive_seed(seed, "board", scenario, board)
            for rotation in range(n):
                seats = tuple(min(lineup_size, (seat + rotation) % n)
                              for seat in range(n))
                out.append(ScheduledGame(
                    game_id=f"s{scenario}b{board}r{rotation}", scenario=scenario, board=board,
                    rotation=rotation, num_players=n, map_seed=_derive_seed(root, "map"),
                    game_seed=_derive_seed(root, "game"), bot_seed=_derive_seed(root, "bots", rotation),
                    map_hash="", seats=seats))
    return out


def tiny_model(games=("base-2p",), level="private", seed=0, preset="graph-tiny"):
    torch.manual_seed(seed)
    dims = dims_for(list(games))
    return build_model({"preset": preset}, dims=dims, levels={"level": level},
                       num_actions=action_size(dims), layout=row_layout(dims, level))


def agent(key: str, model=None, infer_rows: int = 16, device: str = "cpu", **params):
    return make_native_agent(Resolved(key, params), device=device, infer_rows=infer_rows, model=model)
