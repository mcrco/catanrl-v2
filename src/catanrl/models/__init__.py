"""Networks over cppanatron graph rows.

``build_model(spec, dims=..., levels=..., num_actions=..., layout=...)``
builds a :class:`CatanModel` from a model spec, the dict checkpoints store
under ``"model"``::

    {"preset": "graph-s", "class": "graph", "args": {"hidden": 96, ...}}

A spec naming only a preset reads ``configs/models/<preset>.yaml``. Missing
args take the class defaults and unknown args raise; ``model.spec`` is the
fully resolved spec, so storing it rebuilds the identical module.

Every model maps rows to ``(logits [B, A], values [B, P])``: logits with
illegal actions at -1e9, values each seat's win probability in [0, 1] (slot 0
= the row's own seat, absent seats 0). Classes:

* ``graph``: :mod:`catanrl.models.graph` (dims-agnostic). With ``args.aux``
  (a shared trunk) it carries an auxiliary head predicting the other seats'
  hidden features (``aux_predictions``); its training targets are the
  ``critic_players`` block of a row whose ``critic_level`` is above its
  ``level`` (``aux_targets``), which no head reads;
* ``mlp``: :mod:`catanrl.models.mlp` (padded dims only);
* ``cnn``: :mod:`catanrl.models.cnn` (v1's cross-dimensional CNN; the BASE board only);
* ``lstm``: :mod:`catanrl.models.lstm` (dims-agnostic, recurrent: the graph
  trunk with an LSTM over the seat's own decisions). Its stateless
  ``forward`` / ``heads`` is the zero-state function; ``step`` threads the
  state.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml

from catanrl.envs.encoding import (
    CRITIC_PLAYERS,
    check_dims,
    critic_level_of,
    hidden_feature_indices,
    level_name,
    make_encoder,
    row_layout,
)
from catanrl.models.base import CatanModel
from catanrl.models.cnn import CNN_DEFAULTS, CnnModel
from catanrl.models.graph import GRAPH_DEFAULTS, GraphNetwork
from catanrl.models.lstm import LSTM_DEFAULTS, LstmNetwork
from catanrl.models.mlp import MLP_DEFAULTS, FlatNetwork
from catanrl.models.rows import (
    RowReader,
    canonical_feature_names,
    feature_names,
    mask_logits,
    value_prior,
)

MODEL_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "models"
CLASSES: dict[str, dict[str, Any]] = {
    "graph": GRAPH_DEFAULTS, "mlp": MLP_DEFAULTS, "cnn": CNN_DEFAULTS, "lstm": LSTM_DEFAULTS,
}

__all__ = ["MODEL_CLASSES", "CatanModel", "build_model", "load_preset", "normalize_levels", "resolve_spec"]


def load_preset(name: str) -> dict[str, Any]:
    path = MODEL_CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        known = sorted(p.stem for p in MODEL_CONFIG_DIR.glob("*.yaml"))
        raise ValueError(f"unknown model preset {name!r}; known: {known}")
    with open(path) as f:
        spec = yaml.safe_load(f)
    if spec.get("preset") != name:
        raise ValueError(f"{path} names preset {spec.get('preset')!r}")
    return spec


def resolve_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """The complete spec: class defaults filled in, unknown keys refused."""
    spec = dict(spec)
    unknown = set(spec) - {"preset", "class", "args"}
    if unknown:
        raise ValueError(f"unknown model spec keys {sorted(unknown)}")
    if "class" not in spec:
        if "preset" not in spec:
            raise ValueError("a model spec needs a class or a preset")
        preset = load_preset(spec["preset"])
        args = {**preset.get("args", {}), **(spec.get("args") or {})}
        spec = {"preset": spec["preset"], "class": preset["class"], "args": args}
    cls = spec["class"]
    if cls not in CLASSES:
        raise ValueError(f"unknown model class {cls!r}; expected one of {sorted(CLASSES)}")
    args = dict(spec.get("args") or {})
    unknown = set(args) - set(CLASSES[cls])
    if unknown:
        raise ValueError(f"unknown {cls} args {sorted(unknown)}; expected {sorted(CLASSES[cls])}")
    resolved = {**copy.deepcopy(CLASSES[cls]), **args}
    return {"preset": spec.get("preset"), "class": cls, "args": resolved}


def normalize_levels(levels: Mapping[str, Any] | str) -> dict[str, str]:
    """``{"level": ..., "critic_level": ...}`` with names checked.

    ``critic_level`` None (or equal to ``level``) means the critic reads the
    actor's row.
    """
    if isinstance(levels, str):
        levels = {"level": levels}
    unknown = set(levels) - {"level", "critic_level"}
    if unknown or "level" not in levels:
        raise ValueError(f"levels need 'level' (and optionally 'critic_level'), got {sorted(levels)}")
    level = level_name(levels["level"])
    critic = levels.get("critic_level")
    return {"level": level, "critic_level": level_name(critic) if critic is not None else level}


class GraphModel(CatanModel):
    dims_agnostic = True

    def __init__(self, spec, dims, levels, layout) -> None:
        super().__init__(spec, dims, levels, layout)
        args = spec["args"]
        canonical = canonical_feature_names()
        names = feature_names(dims, levels["level"])
        action_layout = make_encoder(dims, levels["level"]).action_space.layout()
        actor_reader = RowReader(layout, names, canonical=canonical)
        self.present_index = actor_reader.feature("players", "present")
        self.aux = bool(args["aux"])
        if self.aux and not args["shared"]:
            raise ValueError("the aux head (model.args.aux) sits on a shared trunk: set model.args.shared")
        if args["shared"]:
            if self.has_critic_block and not self.aux:
                raise ValueError("a shared trunk cannot read a separate critic level")
            hidden = hidden_feature_indices(canonical["players"]) if self.aux else []
            net = GraphNetwork(actor_reader, action_layout, args=args, num_players=self.num_players,
                               policy=True, value=True, aux_features=len(hidden))
            self.backbone, self.policy_head, self.value_head = net.trunk, net.policy, net.value
            self.parts = {"actor": ["backbone.", "policy_head."], "critic": ["value_head."]}
            if self.aux:
                self.aux_head = net.aux
                self.parts["aux"] = ["aux_head."]
                self.register_buffer("hidden_index", torch.tensor(hidden, dtype=torch.long), persistent=False)
            # The trunk reads the actor's row only; a critic block, when the row has one, is the
            # aux head's target and no head reads it (the policy stays fair at the actor's level).
            self.reads_critic_block = False
        else:
            critic_names = feature_names(dims, levels["critic_level"])
            critic_reader = RowReader(layout, critic_names, players=self.critic_players, canonical=canonical)
            self.actor = GraphNetwork(actor_reader, action_layout, args=args, num_players=self.num_players,
                                      policy=True, value=False)
            self.critic = GraphNetwork(critic_reader, action_layout, args=args, num_players=self.num_players,
                                       policy=False, value=True)
            self.parts = {"actor": ["actor."], "critic": ["critic."]}
        self.shared = bool(args["shared"])

    def _present(self, rows) -> torch.Tensor:
        return rows.players[:, : self.num_players, self.present_index]

    def heads(self, obs):
        if self.shared:
            rows = self.backbone.reader(obs)
            emb = self.backbone(obs, rows)
            present = self._present(rows)
            return self.policy_head(emb, rows.mask), self.value_head(emb, present), present
        return super().heads(obs)

    def policy_logits(self, obs):
        if self.shared:
            return self.heads(obs)[0]
        rows = self.actor.trunk.reader(obs)
        return self.actor.policy(self.actor.trunk(obs, rows), rows.mask)

    def value_logits(self, obs):
        if self.shared:
            _, values, present = self.heads(obs)
            return values, present
        rows = self.critic.trunk.reader(obs)
        present = self._present(self.actor.trunk.reader(obs)) if self.has_critic_block else self._present(rows)
        return self.critic.value(self.critic.trunk(obs, rows), present), present

    # ---------------------------------------------------------- the aux head

    def heads_with_aux(self, obs):
        """(masked logits, value logits, present, aux predictions) in one trunk pass."""
        if not self.aux:
            raise ValueError("this network has no aux head (model.args.aux)")
        rows = self.backbone.reader(obs)
        emb = self.backbone(obs, rows)
        present = self._present(rows)
        return self.policy_head(emb, rows.mask), self.value_head(emb, present), present, self.aux_head(emb)

    def aux_predictions(self, obs):
        """The aux head's predictions [B, P-1, F]: row k holds slot k+1's hidden features
        (:data:`~catanrl.envs.encoding.HIDDEN_FEATURES`, in the encoder's scale), from the
        actor's row alone (no critic block needed)."""
        return self.heads_with_aux(obs)[3]

    def aux_targets(self, obs):
        """(targets [B, P-1, F], present [B, P-1]) for the aux head: slots 1..P-1's hidden
        features read from the critic block (the players block at ``critic_level``), and
        which of those seats the game has. Training only: a row without the block raises."""
        if not self.aux:
            raise ValueError("this network has no aux head (model.args.aux)")
        if not self.has_critic_block:
            raise ValueError("aux targets come from the critic block: set obs.critic_level above obs.level")
        reader = self.backbone.reader
        hidden = reader.players_block(obs, CRITIC_PLAYERS)[:, 1:]              # [B, P-1, Fp]
        targets = hidden.index_select(2, self.hidden_index)                    # [B, P-1, F]
        present = reader.players_block(obs)[:, 1:, self.present_index]         # [B, P-1]
        return targets, present


class MlpModel(CatanModel):
    dims_agnostic = False

    def __init__(self, spec, dims, levels, layout) -> None:
        super().__init__(spec, dims, levels, layout)
        args = spec["args"]
        names = feature_names(dims, levels["level"])
        self.present_index = names["players"].index("present")
        self.players_offset, (_, self.player_width) = layout["players"][0], layout["players"][1]
        if args["shared"]:
            if self.has_critic_block:
                raise ValueError("a shared trunk cannot read a separate critic level")
            self.backbone = FlatNetwork(layout, "players", self.num_actions + self.num_players, args, out_gain=0.01)
            self.parts = {"actor": ["backbone."], "critic": []}
        else:
            self.actor = FlatNetwork(layout, "players", self.num_actions, args, out_gain=0.01)
            self.critic = FlatNetwork(layout, self.critic_players, self.num_players, args, out_gain=0.01)
            self.parts = {"actor": ["actor."], "critic": ["critic."]}
        self.shared = bool(args["shared"])

    def _present(self, obs):
        players = obs[:, self.players_offset : self.players_offset + self.num_players * self.player_width]
        return players.reshape(-1, self.num_players, self.player_width)[..., self.present_index].float()

    def _mask(self, obs):
        return obs[:, self.mask_offset : self.mask_offset + self.num_actions]

    def heads(self, obs):
        if self.shared:
            out = self.backbone(obs)
            present = self._present(obs)
            logits = mask_logits(out[:, : self.num_actions], self._mask(obs))
            values = out[:, self.num_actions :].float() + value_prior(present)
            return logits, values, present
        return super().heads(obs)

    def policy_logits(self, obs):
        if self.shared:
            return self.heads(obs)[0]
        return mask_logits(self.actor(obs), self._mask(obs))

    def value_logits(self, obs):
        if self.shared:
            _, values, present = self.heads(obs)
            return values, present
        present = self._present(obs)
        return self.critic(obs).float() + value_prior(present), present


class LstmModel(CatanModel):
    """One graph trunk, an LSTM cell over its pooled embedding, both heads
    (:mod:`catanrl.models.lstm`)."""

    dims_agnostic = True
    recurrent = True

    def __init__(self, spec, dims, levels, layout) -> None:
        super().__init__(spec, dims, levels, layout)
        if self.has_critic_block:
            raise ValueError("an lstm network has one trunk for both heads and cannot read a separate critic level")
        canonical = canonical_feature_names()
        names = feature_names(dims, levels["level"])
        action_layout = make_encoder(dims, levels["level"]).action_space.layout()
        reader = RowReader(layout, names, canonical=canonical)
        self.present_index = reader.feature("players", "present")
        self.net = LstmNetwork(reader, action_layout, args=spec["args"], num_players=self.num_players)
        self.hidden_size = self.net.hidden_size
        self.parts = {"actor": ["net.trunk.", "net.cell.", "net.inject.", "net.policy."], "critic": ["net.value."]}
        self.shared = True

    def _present(self, rows) -> torch.Tensor:
        return rows.players[:, : self.num_players, self.present_index]

    # ---------------------------------------------------------- recurrent API

    def initial_state(self, rows: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net.initial_state(rows, device)

    def embed(self, obs: torch.Tensor):
        """The trunk's embeddings and the rows (shared by ``remember`` and ``decode``)."""
        return self.net.embed(obs)

    def remember(self, pooled: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        return self.net.remember(pooled, h, c)

    def decode(self, emb, rows, h: torch.Tensor):
        """(masked logits, value logits, present) from embeddings and a memory."""
        present = self._present(rows)
        logits, values = self.net.decode(emb, rows, h, present)
        return logits, values, present

    def step(self, obs: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        """One decision per row from state ``(h, c)``: (masked logits, value logits,
        present, h', c')."""
        emb, rows = self.embed(obs)
        h, c = self.remember(emb.pooled, h, c)
        logits, values, present = self.decode(emb, rows, h)
        return logits, values, present, h, c

    # ---------------------------------------------------------- stateless (zero state)

    def heads(self, obs):
        h, c = self.initial_state(obs.shape[0], obs.device)
        logits, values, present, _, _ = self.step(obs, h, c)
        return logits, values, present

    def policy_logits(self, obs):
        return self.heads(obs)[0]

    def value_logits(self, obs):
        _, values, present = self.heads(obs)
        return values, present


#: Model class name -> the :class:`CatanModel` subclass (the ``class`` of a spec).
MODEL_CLASSES: dict[str, type[CatanModel]] = {
    "graph": GraphModel, "mlp": MlpModel, "cnn": CnnModel, "lstm": LstmModel,
}


def build_model(
    spec: Mapping[str, Any],
    *,
    dims: Mapping[str, int],
    levels: Mapping[str, Any] | str,
    num_actions: int,
    layout: Mapping[str, Any] | None = None,
) -> CatanModel:
    """Build the network a spec describes for rows of ``dims`` at ``levels``.

    ``layout`` is the row layout (:func:`catanrl.envs.encoding.row_layout`);
    it must equal this engine's, as must ``num_actions``.
    """
    spec = resolve_spec(spec)
    dims = check_dims(dims)
    levels = normalize_levels(levels)
    expected = row_layout(dims, levels["level"], critic_level_of(levels["level"], levels["critic_level"]))
    if layout is not None:
        given = {name: (int(off), tuple(int(s) for s in shape)) for name, (off, shape) in dict(layout).items()}
        if given != expected:
            raise ValueError("the given row layout differs from this engine's encoder layout")
    expected_actions = expected["action_mask"][1][0]
    if int(num_actions) != expected_actions:
        raise ValueError(f"num_actions {num_actions} != the action space's {expected_actions}")
    return MODEL_CLASSES[spec["class"]](spec, dims, levels, expected)
