"""Reading environment rows by their layout.

A row is ``[global | players | tiles | nodes | edges | edge_nodes | tile_nodes
| action_mask]`` (plus ``critic_players``, the players block at the critic's
level, when the critic reads another information level); :class:`RowReader`
slices it by the encoder's ``layout()``, never by hard-coded offsets.

Per-seat features (``settlement_<k>``, ``road_<k>``, ``players_<n>``, ...)
are remapped onto the canonical six-seat feature order of an encoder with
``max_players = 6``, zero-filling seats the env does not have, and the
players block is padded to six slots. Graph networks therefore have the same
parameters for any ``max_players`` and any map size.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from catanrl.envs.encoding import CRITIC_PLAYERS, FEATURE_BLOCKS, MAX_PLAYERS, make_encoder

MASKED_LOGIT = -1e9


def canonical_feature_names(level: str = "private") -> dict[str, list[str]]:
    """Per-slot feature names of an encoder for six seats (any map dims)."""
    encoder = make_encoder(
        {"max_players": MAX_PLAYERS, "max_tiles": 1, "max_nodes": 6, "max_edges": 1}, level
    )
    return {block: list(encoder.feature_names(block)) for block in FEATURE_BLOCKS}


def feature_names(dims: Mapping[str, int], level: str) -> dict[str, list[str]]:
    encoder = make_encoder(dims, level)
    return {block: list(encoder.feature_names(block)) for block in FEATURE_BLOCKS}


@dataclass
class Rows:
    """One batch of rows split into blocks (canonical feature order).

    ``players`` always has six slots; the others are padded to the env's dims.
    """

    glob: torch.Tensor          # [B, G]
    players: torch.Tensor       # [B, 6, Fp]
    tiles: torch.Tensor         # [B, T, Ft]
    nodes: torch.Tensor         # [B, N, Fn]
    edges: torch.Tensor         # [B, E, Fe]
    edge_nodes: torch.Tensor    # [B, E, 2] node slots, -1 = padding
    tile_nodes: torch.Tensor    # [B, T, 6] node slots, -1 = padding
    mask: torch.Tensor          # [B, A]


class RowReader(nn.Module):
    """Slices rows into :class:`Rows` and remaps features to canonical order.

    ``players`` names the block the players are read from: ``"players"``,
    or ``"critic_players"`` for a critic at another level; every other block
    is shared.
    """

    def __init__(
        self,
        layout: Mapping[str, Any],
        names: Mapping[str, Sequence[str]],
        *,
        players: str = "players",
        canonical: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        super().__init__()
        canonical = canonical or canonical_feature_names()
        self.spans: dict[str, tuple[int, tuple[int, ...]]] = {}
        for block in FEATURE_BLOCKS:
            offset, shape = layout[players if block == "players" else block]
            self.spans[block] = (int(offset), tuple(int(s) for s in shape))
        for block in ("edge_nodes", "tile_nodes", "action_mask"):
            offset, shape = layout[block]
            self.spans[block] = (int(offset), tuple(int(s) for s in shape))
        if CRITIC_PLAYERS in layout:  # readable by name through players_block, whichever block players is
            offset, shape = layout[CRITIC_PLAYERS]
            self.spans[CRITIC_PLAYERS] = (int(offset), tuple(int(s) for s in shape))
        self.num_players = self.spans["players"][1][0]
        if self.num_players > MAX_PLAYERS:
            raise ValueError(f"at most {MAX_PLAYERS} seats are supported")
        self.widths: dict[str, int] = {}
        for block in FEATURE_BLOCKS:
            actual = list(names[block])
            width = self.spans[block][1][-1]
            if len(actual) != width:
                raise ValueError(f"{block}: {len(actual)} feature names for width {width}")
            index = []
            for name in canonical[block]:
                index.append(actual.index(name) if name in actual else width)  # width = zero column
            unknown = sorted(set(actual) - set(canonical[block]))
            if unknown:
                raise ValueError(f"{block}: features outside the six-seat encoder: {unknown}")
            self.register_buffer(f"index_{block}", torch.tensor(index, dtype=torch.long), persistent=False)
            self.widths[block] = len(index)
        self.canonical = {block: list(canonical[block]) for block in FEATURE_BLOCKS}

    def feature(self, block: str, name: str) -> int:
        """Position of a feature in the canonical order of ``block``."""
        return self.canonical[block].index(name)

    def _slice(self, obs: torch.Tensor, block: str) -> torch.Tensor:
        offset, shape = self.spans[block]
        size = 1
        for s in shape:
            size *= s
        return obs[:, offset : offset + size].reshape(obs.shape[0], *shape)

    def _canonical(self, x: torch.Tensor, block: str) -> torch.Tensor:
        padded = torch.cat((x, x.new_zeros(*x.shape[:-1], 1)), dim=-1)
        return padded.index_select(-1, getattr(self, f"index_{block}"))

    def players_block(self, obs: torch.Tensor, block: str = "players") -> torch.Tensor:
        """One players block of the row, ``"players"`` (the one this reader's rows hold) or
        ``"critic_players"``, in canonical feature order: [B, P, Fp] over the env's seats.
        Both blocks have the players block's feature names."""
        if block not in ("players", CRITIC_PLAYERS) or block not in self.spans:
            raise ValueError(f"the row has no {block!r} block")
        return self._canonical(self._slice(obs.float(), block), "players")

    def forward(self, obs: torch.Tensor) -> Rows:
        obs = obs.float()
        players = self._canonical(self._slice(obs, "players"), "players")
        if players.shape[1] < MAX_PLAYERS:
            players = torch.cat(
                (players, players.new_zeros(players.shape[0], MAX_PLAYERS - players.shape[1], players.shape[2])),
                dim=1,
            )
        return Rows(
            glob=self._canonical(self._slice(obs, "global"), "global"),
            players=players,
            tiles=self._canonical(self._slice(obs, "tiles"), "tiles"),
            nodes=self._canonical(self._slice(obs, "nodes"), "nodes"),
            edges=self._canonical(self._slice(obs, "edges"), "edges"),
            edge_nodes=self._slice(obs, "edge_nodes"),
            tile_nodes=self._slice(obs, "tile_nodes"),
            mask=self._slice(obs, "action_mask"),
        )


def flat_features(obs: torch.Tensor, layout: Mapping[str, Any], players: str = "players") -> torch.Tensor:
    """The feature blocks of a row, concatenated in their actual (padded) order."""
    parts = []
    for block in FEATURE_BLOCKS:
        offset, shape = layout[players if block == "players" else block]
        size = 1
        for s in shape:
            size *= int(s)
        parts.append(obs[:, int(offset) : int(offset) + size])
    return torch.cat(parts, dim=1).float()


def flat_feature_size(layout: Mapping[str, Any], players: str = "players") -> int:
    total = 0
    for block in FEATURE_BLOCKS:
        _, shape = layout[players if block == "players" else block]
        size = 1
        for s in shape:
            size *= int(s)
        total += size
    return total


def mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    return torch.where(mask > 0, logits, torch.full_like(logits, MASKED_LOGIT))


def present_players(players: torch.Tensor, present_index: int, num_players: int) -> torch.Tensor:
    """[B, num_players] 1.0 where the seat slot holds a player."""
    return players[:, :num_players, present_index]


def value_prior(present: torch.Tensor) -> torch.Tensor:
    """logit(1/n) for n seated players: a fresh value head predicts 1/n per seat."""
    n = present.sum(dim=1, keepdim=True).clamp_min(2.0)
    return -torch.log(n - 1.0)


def orthogonal_(module: nn.Module, gain: float) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
