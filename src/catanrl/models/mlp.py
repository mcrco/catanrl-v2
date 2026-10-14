"""Flat MLP baseline over the same rows (ablations; padded maps only).

It reads the feature blocks at the env's padded dims, so its parameters
depend on the dims: unlike the graph network it is not dims-agnostic.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from catanrl.models.rows import flat_feature_size, flat_features, orthogonal_

MLP_DEFAULTS: dict[str, Any] = {"hidden": 512, "blocks": 2, "shared": False}


class ResidualMLP(nn.Module):
    """v1's flat encoder: an input layer and pre-norm residual blocks."""

    def __init__(self, inputs: int, hidden: int, blocks: int) -> None:
        super().__init__()
        self.input = nn.Linear(inputs, hidden)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            for _ in range(blocks)
        )
        self.out_norm = nn.LayerNorm(hidden)
        orthogonal_(self.input, math.sqrt(2.0))
        for block in self.blocks:
            orthogonal_(block[1], math.sqrt(2.0))
            orthogonal_(block[3], 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.input(x))
        for block in self.blocks:
            x = x + block(x)
        return self.out_norm(x)


class FlatNetwork(nn.Module):
    def __init__(self, layout: Mapping[str, Any], prefix: str, outputs: int, args: Mapping[str, Any],
                 *, out_gain: float) -> None:
        super().__init__()
        self.layout = dict(layout)
        self.prefix = prefix
        self.encoder = ResidualMLP(flat_feature_size(layout, prefix), args["hidden"], args["blocks"])
        self.output = nn.Linear(args["hidden"], outputs)
        orthogonal_(self.output, out_gain)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.output(self.encoder(flat_features(obs, self.layout, self.prefix)))
