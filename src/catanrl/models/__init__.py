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

* ``graph``: :mod:`catanrl.models.graph` (dims-agnostic);
* ``mlp``: :mod:`catanrl.models.mlp` (padded dims only).
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from catanrl.envs.encoding import (
    check_dims,
    critic_level_of,
    level_name,
    make_encoder,
    row_layout,
    row_size,
)
from catanrl.models.graph import GRAPH_DEFAULTS, GraphNetwork
from catanrl.models.mlp import MLP_DEFAULTS, FlatNetwork
from catanrl.models.rows import (
    RowReader,
    canonical_feature_names,
    feature_names,
    mask_logits,
    value_prior,
)

MODEL_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "models"
CLASSES: dict[str, dict[str, Any]] = {"graph": GRAPH_DEFAULTS, "mlp": MLP_DEFAULTS}

__all__ = ["CatanModel", "build_model", "load_preset", "normalize_levels", "resolve_spec"]


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


class CatanModel(nn.Module):
    """Base of the networks: spec, parts and the row contract."""

    spec: dict[str, Any]
    dims_agnostic: bool
    parts: dict[str, list[str]]

    def __init__(self, spec, dims, levels, layout) -> None:
        super().__init__()
        self.spec = spec
        self.dims = dict(dims)
        self.levels = dict(levels)
        self.layout = dict(layout)
        self.num_players = self.dims["max_players"]
        self.obs_size = row_size(layout)
        self.mask_offset, (self.num_actions,) = layout["action_mask"]
        self.noop = self.num_actions - 1
        self.has_critic_block = any(name.startswith("critic/") for name in layout)

    # Subclasses implement these three; the rest derives from them.
    def policy_logits(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def value_logits(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-seat win logits [B, P] and the seats present [B, P]."""
        raise NotImplementedError

    def heads(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(masked logits, value logits, present) in one pass."""
        logits = self.policy_logits(obs)
        values, present = self.value_logits(obs)
        return logits, values, present

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, value_logits, present = self.heads(obs)
        return logits, torch.sigmoid(value_logits) * present

    def acting(self, obs: torch.Tensor) -> torch.Tensor:
        """[B] bool: the row must act (its mask has more than the no-op)."""
        return obs[:, self.mask_offset + self.noop] == 0

    def waiting_logits(self, rows: int, device, dtype=torch.float32) -> torch.Tensor:
        """Logits of rows that only have the no-op."""
        logits = torch.full((rows, self.num_actions), -1e9, device=device, dtype=dtype)
        logits[:, self.noop] = 0.0
        return logits


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
        if args["shared"]:
            if self.has_critic_block:
                raise ValueError("a shared trunk cannot read a separate critic level")
            net = GraphNetwork(actor_reader, action_layout, args=args, num_players=self.num_players,
                               policy=True, value=True)
            self.backbone, self.policy_head, self.value_head = net.trunk, net.policy, net.value
            self.parts = {"actor": ["backbone.", "policy_head."], "critic": ["value_head."]}
        else:
            critic_names = feature_names(dims, levels["critic_level"])
            prefix = "critic/" if self.has_critic_block else ""
            critic_reader = RowReader(layout, critic_names, prefix=prefix, canonical=canonical)
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


class MlpModel(CatanModel):
    dims_agnostic = False

    def __init__(self, spec, dims, levels, layout) -> None:
        super().__init__(spec, dims, levels, layout)
        args = spec["args"]
        names = feature_names(dims, levels["level"])
        self.present_index = names["players"].index("present")
        self.players_offset, (_, self.player_width) = layout["players"][0], layout["players"][1]
        critic_prefix = "critic/" if self.has_critic_block else ""
        if args["shared"]:
            if self.has_critic_block:
                raise ValueError("a shared trunk cannot read a separate critic level")
            self.backbone = FlatNetwork(layout, "", self.num_actions + self.num_players, args, out_gain=0.01)
            self.parts = {"actor": ["backbone."], "critic": []}
        else:
            self.actor = FlatNetwork(layout, "", self.num_actions, args, out_gain=0.01)
            self.critic = FlatNetwork(layout, critic_prefix, self.num_players, args, out_gain=0.01)
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
    cls = {"graph": GraphModel, "mlp": MlpModel}[spec["class"]]
    return cls(spec, dims, levels, expected)
