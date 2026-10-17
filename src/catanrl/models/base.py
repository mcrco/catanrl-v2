"""The base of the networks: spec, parts and the row contract (see :mod:`catanrl.models`)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from catanrl.envs.encoding import CRITIC_PLAYERS, row_size

__all__ = ["CatanModel"]


class CatanModel(nn.Module):
    """Base of the networks: spec, parts and the row contract.

    A recurrent network (``recurrent = True``) carries a state of
    ``hidden_size`` floats per row between a seat's decisions, through
    ``initial_state`` and ``step``; its stateless ``forward`` / ``heads`` is
    the zero-state function (what the checkpoint self-test pins).
    """

    spec: dict[str, Any]
    dims_agnostic: bool
    parts: dict[str, list[str]]
    recurrent: bool = False
    hidden_size: int = 0

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
        self.has_critic_block = CRITIC_PLAYERS in layout
        self.critic_players = CRITIC_PLAYERS if self.has_critic_block else "players"

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

    def check_maps(self, maps: Iterable[Any]) -> None:
        """Refuse ``cpp.CatanMap`` objects whose rows this network would misread: none, by
        default. Dims are the callers' check (``check_bindable``, ``Checkpoint.model``,
        the agents' ``bind``); a class that ties slots to one board (cnn) overrides this,
        because equal dims do not mean equal layouts."""

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
