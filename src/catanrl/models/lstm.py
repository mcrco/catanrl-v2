"""Recurrent graph network: the graph trunk with an LSTM over a seat's decisions.

The network of the "how markov is a decision" experiment (report, sections 1
and 3): the same :class:`~catanrl.models.graph.GraphTrunk` as ``graph-s``,
then one LSTM cell over the trunk's pooled embedding, then the structured
policy head and the per-seat value head of :mod:`catanrl.models.graph`. The
memory is injected into every head's input (added to the node, tile and
edge embeddings and concatenated to the pooled state), so a remembered fact
can move a placement as well as a global action or a value.

What the memory is a sequence of: a seat's **own decisions**. The cell steps
once per row that acts (its mask has more than the no-op), with the row's
state; waiting rows keep theirs, and a state is zero at the seat's first
decision of a game. That keeps the trainer's and the evaluator's invariant
that only acting rows run the network, and it is the same sequence in
training (:class:`~catanrl.train.puffer_policy.PufferPolicy`), in native
evaluation (:class:`~catanrl.agents.native.PolicyAgent`) and in the
checkpoint self-test (single positions: the zero-state function).

One trunk serves both heads (``graph-s`` with ``shared: true``, the best
recipe's shape): a separate critic would need a second trunk and a second
recurrent state threaded everywhere the first is, for a critic that is
discarded after training.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from catanrl.models.graph import (
    HIDDEN_GAIN,
    Embeddings,
    GraphPolicyHead,
    GraphTrunk,
    GraphValueHead,
)
from catanrl.models.rows import RowReader, Rows, orthogonal_

LSTM_DEFAULTS: dict[str, Any] = {
    "hidden": 96,
    "global_hidden": 64,
    "layers": 3,
    "head_hidden": 128,
    # The LSTM cell's state size: what a seat remembers between its decisions.
    "lstm_hidden": 128,
}


@dataclass(frozen=True)
class HeadShape:
    """What the graph heads read from a trunk: its entity width and pooled width."""

    hidden: int
    pooled_dim: int


class LstmNetwork(nn.Module):
    """Trunk, cell, memory injection and both heads.

    ``embed`` runs the trunk (per row, no memory), ``remember`` steps the
    cell, ``decode`` runs the heads with a memory; ``step`` does the three
    for one decision per row.
    """

    def __init__(self, reader: RowReader, action_layout: Mapping[str, Any], *, args: Mapping[str, Any],
                 num_players: int) -> None:
        super().__init__()
        self.trunk = GraphTrunk(reader, hidden=args["hidden"], global_hidden=args["global_hidden"],
                                layers=args["layers"])
        self.hidden_size = int(args["lstm_hidden"])
        if self.hidden_size < 1:
            raise ValueError("lstm_hidden must be positive")
        self.cell = nn.LSTMCell(self.trunk.pooled_dim, self.hidden_size)
        # PufferLib's LSTM initialization: orthogonal weights, zero biases.
        for name, param in self.cell.named_parameters():
            if name.startswith("weight"):
                nn.init.orthogonal_(param, 1.0)
            else:
                nn.init.zeros_(param)
        h = self.trunk.hidden
        self.inject = nn.ModuleDict({name: nn.Linear(self.hidden_size, h, bias=False)
                                     for name in ("nodes", "tiles", "edges")})
        for module in self.inject.values():
            orthogonal_(module, HIDDEN_GAIN)
        shape = HeadShape(hidden=h, pooled_dim=self.trunk.pooled_dim + self.hidden_size)
        self.policy = GraphPolicyHead(shape, action_layout, head_hidden=args["head_hidden"])
        self.value = GraphValueHead(shape, head_hidden=args["head_hidden"], num_players=num_players)

    def initial_state(self, rows: int, device=None, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
        """The zero memory of ``rows`` rows (before a seat's first decision)."""
        zeros = torch.zeros(rows, self.hidden_size, device=device, dtype=dtype)
        return zeros, zeros.clone()

    def embed(self, obs: torch.Tensor) -> tuple[Embeddings, Rows]:
        rows = self.trunk.reader(obs)
        return self.trunk(obs, rows), rows

    def remember(self, pooled: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One cell step for rows that act: (h', c') from the pooled embedding."""
        return self.cell(pooled, (h, c))

    def decode(self, emb: Embeddings, rows: Rows, h: torch.Tensor, present: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """(masked policy logits, per-seat value logits) with memory ``h`` injected."""
        topo = emb.topo
        remembered = Embeddings(
            nodes=(emb.nodes + self.inject["nodes"](h).unsqueeze(1)) * topo.node_valid,
            tiles=(emb.tiles + self.inject["tiles"](h).unsqueeze(1)) * topo.tile_valid,
            edges=(emb.edges + self.inject["edges"](h).unsqueeze(1)) * topo.edge_valid,
            players=emb.players,
            pooled=torch.cat((emb.pooled, h.to(emb.pooled.dtype)), dim=1),
            topo=topo,
        )
        return self.policy(remembered, rows.mask), self.value(remembered, present)
