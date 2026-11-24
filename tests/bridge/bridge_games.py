"""Lockstep games for the bridge tests.

A stock Catanatron game is the truth: it gives every seat its ``client_view``
(the redacted document a server or the TUI hands a bot). A cppanatron game
plays the same actions with the same outcomes alongside, so the native
features, hand trackers and agents of a seat can be compared with what the
bridge builds from that seat's views alone.

The friendly robber is the one rule the engines differ on: Catanatron's own
(``friendly_robber``, what ``GameConfig.catanatron_game`` builds) shields the
mover's opponents under 3 actual points (hidden victory point cards count) and
opens every tile when all are shielded, the engine follows OpenHex (any seat
with at most 2 public points, the mover included, and no robber move when every
tile but the robber's is shielded). A lockstep with the friendly robber
therefore takes its truth from :func:`site_game`, as a ``site`` lockstep does:
a catanatron-server room's game, with Catanatron's
``robber_mode="friendly_public"``, which is OpenHex's rule, and the config's
dice, balanced ones included.
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
    #: The truth as a catanatron-server room plays the config (:func:`site_game`);
    #: a config with the friendly robber always is.
    site: bool = False
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
        build = site_game if self.site or self.config.friendly_robber else games.GameConfig.catanatron_game
        self.truth = build(self.config, players, self.map_seed, self.seed)
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


def site_game(config: games.GameConfig, players, map_seed: int, seed: int):
    """The Catanatron game a catanatron-server room plays for ``config``: its
    friendly robber is ``robber_mode="friendly_public"`` (the room config's
    ``robber_mode: friendly_public``, OpenHex's rule, as the engine plays it),
    its dice the config's (``dice_mode: balanced`` too)."""
    from catanatron.game import Game
    from catanatron.models.decks import Supply
    from cppanatron.catanatron import catanatron_map_from_spec

    return Game(
        players,
        seed=seed,
        catan_map=catanatron_map_from_spec(config.map_spec(map_seed)),
        vps_to_win=config.victory_points_to_win,
        discard_limit=config.discard_limit,
        robber_mode="friendly_public" if config.friendly_robber else "normal",
        dice_mode=config.dice_mode,
        supply=Supply(resource_bank=config.resource_bank, development_deck=config.development_deck,
                      roads=config.roads_per_player, settlements=config.settlements_per_player,
                      cities=config.cities_per_player),
    )


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
