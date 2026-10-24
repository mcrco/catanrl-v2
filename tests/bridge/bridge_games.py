"""Lockstep games for the bridge tests.

A stock Catanatron game is the truth: it gives every seat its ``client_view``
(the redacted document a server or the TUI hands a bot). A cppanatron game
plays the same actions with the same outcomes alongside, so the native
features, hand trackers and agents of a seat can be compared with what the
bridge builds from that seat's views alone.

The friendly robber is the one rule the engines differ on: Catanatron shields
seats under 3 actual points (hidden victory point cards count) after a knight's
Largest Army award, the engine follows OpenHex (at most 2 public points, a
knight's tile on the points before its award). The bridge tests' conftest makes
the truth play the engine's rule for each bridge test
(:func:`openhex_friendly_robber`), and undoes it after.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import cppanatron as cpp
import torch
from catanatron.models.enums import Action, ActionType
from catanatron.models.player import Color, SimplePlayer
from catanatron.serialization import action_to_json, client_view, state_to_json
from cppanatron.catanatron import from_catanatron_action

from catanrl import games
from catanrl.agents.native import make_native_agent
from catanrl.envs.encoding import action_size, covering_dims, row_layout
from catanrl.models import build_model


def openhex_friendly_robber(monkeypatch) -> None:
    """Re-rule stock Catanatron's friendly robber as OpenHex's (open-hex
    engine.py ``_friendly_robber_shields``), through ``monkeypatch`` so it ends
    with the test: a tile is shielded when an opponent of the mover with at
    most 2 visible points has a building on it (Catanatron's own fallback,
    every tile when all are shielded, is OpenHex's), and a knight's tile is
    judged on the points before the knight's Largest Army award (OpenHex
    validates it first): the seat the knight took the award from keeps its 2
    points for that robber move."""
    from catanatron import state_functions
    from catanatron.models import actions
    from catanatron.state_functions import get_visible_victory_points, player_key

    award = state_functions.maintain_largest_army

    def maintain_largest_army(state, color, previous_army_color, previous_army_size):
        award(state, color, previous_army_color, previous_army_size)
        took = (previous_army_color is not None and previous_army_color != color
                and not state.player_state[f"{player_key(state, previous_army_color)}_HAS_ARMY"])
        state.openhex_army_taken_from = previous_army_color if took else None

    def blocks(state, color, action) -> bool:
        coordinate, _ = action.value
        records = state.action_records
        after_knight = bool(records) and records[-1].action.action_type == ActionType.PLAY_KNIGHT_CARD
        taken = getattr(state, "openhex_army_taken_from", None) if after_knight else None
        for node_id in state.board.map.land_tiles[coordinate].nodes.values():
            building = state.board.buildings.get(node_id)
            if building is None or building[0] == color:
                continue
            points = get_visible_victory_points(state, building[0]) + (2 if building[0] == taken else 0)
            if points <= 2:
                return True
        return False

    monkeypatch.setattr(state_functions, "maintain_largest_army", maintain_largest_army)
    monkeypatch.setattr(actions, "_robber_action_blocks_low_vp_enemy", blocks)


DATA = Path(__file__).resolve().parents[1] / "data"
GAMES = ("base-2p", "base-3p", "base-4p", "extension-5p", "extension-6p", "hexagon-3p")
CHANCE = {ActionType.ROLL: "dice", ActionType.BUY_DEVELOPMENT_CARD: "development_card",
          ActionType.MOVE_ROBBER: "stolen_resource"}


@dataclass
class Lockstep:
    config: games.GameConfig
    map_seed: int
    seed: int
    friendly_robber: bool | None = None
    vps_to_win: int | None = None
    truth: object = field(init=False)
    native: cpp.Game = field(init=False)

    def __post_init__(self) -> None:
        overrides = {}
        if self.friendly_robber is not None:
            overrides["friendly_robber"] = self.friendly_robber
        if self.vps_to_win is not None:
            overrides["victory_points_to_win"] = self.vps_to_win
        if overrides:
            data = {**self.config.to_dict(), **overrides}
            self.config = games.GameConfig.from_dict(data)
        players = [SimplePlayer(c) for c in list(Color)[: self.config.num_players]]
        self.truth = self.config.catanatron_game(players, self.map_seed, self.seed)
        spec = cpp.MapSpec.from_dict(self.config.map_spec(self.map_seed))
        colors = [cpp.Color[c.value] for c in self.truth.state.colors]
        self.native = cpp.Game(colors, map=cpp.CatanMap.from_spec(spec), rules=self.config.cppanatron_rules(),
                               seed=self.seed)
        self.rng = random.Random(self.seed)

    @property
    def colors(self) -> list[str]:
        return [c.value for c in self.truth.state.colors]

    @property
    def current_seat(self) -> int:
        return self.truth.state.current_player_index

    @property
    def over(self) -> bool:
        return self.truth.winning_color() is not None or not self.truth.playable_actions

    def view(self, seat: int) -> dict:
        return client_view(state_to_json(self.truth), self.colors[seat])

    def views(self) -> list[dict]:
        """Every seat's view now (one serialization)."""
        doc = state_to_json(self.truth)
        return [client_view(doc, color) for color in self.colors]

    @property
    def dims(self) -> dict:
        return covering_dims([self.native.board.map], len(self.colors))

    def playable_json(self) -> list:
        return [action_to_json(a) for a in self.truth.playable_actions]

    def choose(self) -> Action:
        """A random move that favours building, buying and playing cards
        over ending the turn, so games reach every prompt."""
        actions = list(self.truth.playable_actions)
        busy = [a for a in actions if a.action_type not in (ActionType.END_TURN, ActionType.MARITIME_TRADE)]
        if busy and self.rng.random() < 0.8:
            return self.rng.choice(busy)
        return self.rng.choice(actions)

    def step(self, action: Action) -> None:
        """Play ``action`` on both engines with Catanatron's outcome."""
        record = self.truth.execute(action)
        forced = {}
        kind = action.action_type
        if kind == ActionType.ROLL:
            forced["dice"] = tuple(record.result)
        elif kind == ActionType.BUY_DEVELOPMENT_CARD:
            forced["development_card"] = cpp.DevelopmentCard[record.result]
        elif kind == ActionType.MOVE_ROBBER and record.result is not None:
            forced["stolen_resource"] = cpp.Resource[record.result]
        self.native.execute(from_catanatron_action(action), **forced)


def lockstep(name: str, index: int, **kwargs) -> Lockstep:
    config = games.load_game(name)
    return Lockstep(config, map_seed=1000 + index, seed=index, **kwargs)


def tiny_model(level: str, dims: dict):
    """The committed tiny graph network (tests/data/tiny-graph.pt, as
    test_golden_eval loads it) at ``dims``, reading rows at ``level``: its
    weights fit every level, since the levels differ only in which features
    are zero."""
    saved = torch.load(DATA / "tiny-graph.pt", weights_only=True)
    model = build_model(saved["model"], dims=dict(dims), levels={"level": level, "critic_level": level},
                        num_actions=action_size(dims), layout=row_layout(dims, level))
    model.load_state_dict(saved["state_dict"])
    return model


def tiny_agent(key: str, level: str, dims: dict, **params):
    """A native agent on the tiny network (CPU, strict, batch 1)."""
    resolved = SimpleNamespace(key=key, params=params, checkpoint=None)
    return make_native_agent(resolved, device="cpu", infer_rows=1, model=tiny_model(level, dims))
