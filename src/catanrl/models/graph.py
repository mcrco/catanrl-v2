"""Graph network: Canopy Nexus-v3 style heterogeneous message passing.

A port, to cppanatron's graph rows, of the Nexus-v3 network from Canopy by
cullback (https://github.com/cullback/canopy, tag ``catan-nexus-v3``, commit
6185983a88ba6802e7fa9893cef5a76a15de2595: ``examples/catan/model/nexus_v3.rs``),
which catanrl v1 ported to Catanatron's observation as graph-s / graph-m.
Canopy has no license file; see catanrl v1's README, "Credits".

Differences from v1's port, all because v2 rows carry the graph explicitly:

* topology is read per sample from the ``edge_nodes`` / ``tile_nodes``
  blocks (-1 padding), so one network runs on any map and player count;
* edges are a third entity type (node <-> edge messages besides node <-> node
  over edges and node <-> tile), and roads are scored from edge embeddings;
* the robber head scores tile x victim slot from the tile embedding and the
  victim's player embedding;
* the value head predicts each seat's win probability (independent sigmoids,
  seat-relative, slot 0 = the row's own seat) from the pooled state and that
  seat's embedding, starting from 1/n per seat;
* an optional auxiliary head (``aux``, on a shared trunk) predicts, from the
  same inputs, the hidden features of every other seat (slots 1..P-1): the
  players-block features a ``private`` row zeroes for them (DESIGN 2), in the
  encoder's scale. Training only; it never feeds the policy or the value.

Aggregations are dense per-sample incidence matrices (built by comparisons
and ``bmm``, row-normalized by degree): deterministic on CUDA, free of host
syncs and so capturable in CUDA graphs. Padding slots have no incidence, are
excluded from pooling and are zeroed after every layer, so padding a row to
larger dims leaves the outputs unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from catanrl.envs.encoding import MAX_PLAYERS
from catanrl.models.rows import RowReader, Rows, mask_logits, orthogonal_, value_prior

GRAPH_DEFAULTS: dict[str, Any] = {
    "hidden": 96,
    "global_hidden": 64,
    "layers": 3,
    "head_hidden": 128,
    # One trunk for both heads, or separate actor and critic networks (v1's "sep").
    "shared": False,
    # An auxiliary head on the shared trunk predicting the other seats' hidden features
    # (catanrl.envs.encoding.HIDDEN_FEATURES) from the row; trained by train.aux_coef.
    "aux": False,
}
POLICY_OUTPUT_GAIN = 0.01
VALUE_OUTPUT_GAIN = 0.01
AUX_OUTPUT_GAIN = 0.01
HIDDEN_GAIN = math.sqrt(2.0)


@dataclass
class Topology:
    node_valid: torch.Tensor     # [B, N, 1]
    tile_valid: torch.Tensor     # [B, T, 1]
    edge_valid: torch.Tensor     # [B, E, 1]
    node_from_node: torch.Tensor  # [B, N, N] row-normalized
    node_from_edge: torch.Tensor  # [B, N, E]
    node_from_tile: torch.Tensor  # [B, N, T]
    tile_from_node: torch.Tensor  # [B, T, N]
    edge_from_node: torch.Tensor  # [B, E, N] (mean of the two ends)
    edge_ends: torch.Tensor      # [B, E, N] 0/1 (sum of the two ends)


def topology(rows: Rows, valid: Mapping[str, int]) -> Topology:
    nodes = rows.nodes.shape[1]
    slots = torch.arange(nodes, device=rows.nodes.device, dtype=rows.edge_nodes.dtype)
    first = (rows.edge_nodes[..., 0:1] == slots).float()       # [B, E, N]
    second = (rows.edge_nodes[..., 1:2] == slots).float()
    ends = first + second                                         # [B, E, N]
    adjacency = torch.bmm(first.transpose(1, 2), second)
    adjacency = adjacency + adjacency.transpose(1, 2)            # [B, N, N]
    corners = (rows.tile_nodes.unsqueeze(-1) == slots).float().sum(dim=2)  # [B, T, N]
    node_edge = ends.transpose(1, 2)                              # [B, N, E]
    node_tile = corners.transpose(1, 2)                           # [B, N, T]
    return Topology(
        node_valid=rows.nodes[..., valid["nodes"]].unsqueeze(-1),
        tile_valid=rows.tiles[..., valid["tiles"]].unsqueeze(-1),
        edge_valid=rows.edges[..., valid["edges"]].unsqueeze(-1),
        node_from_node=adjacency / adjacency.sum(-1, keepdim=True).clamp_min(1.0),
        node_from_edge=node_edge / node_edge.sum(-1, keepdim=True).clamp_min(1.0),
        node_from_tile=node_tile / node_tile.sum(-1, keepdim=True).clamp_min(1.0),
        tile_from_node=corners / corners.sum(-1, keepdim=True).clamp_min(1.0),
        edge_from_node=ends / ends.sum(-1, keepdim=True).clamp_min(1.0),
        edge_ends=ends,
    )


class HeteroLayer(nn.Module):
    """Pre-norm residual message passing over nodes, tiles and edges.

    Nexus-v3's node/tile layer plus an edge stream. Message weights have no
    bias, so each is applied on whichever side of the (mean) aggregation has
    fewer rows.
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.node_norm = nn.LayerNorm(hidden)
        self.tile_norm = nn.LayerNorm(hidden)
        self.edge_norm = nn.LayerNorm(hidden)
        self.node_self = nn.Linear(hidden, hidden)
        self.node_from_node = nn.Linear(hidden, hidden, bias=False)
        self.node_from_tile = nn.Linear(hidden, hidden, bias=False)
        self.node_from_edge = nn.Linear(hidden, hidden, bias=False)
        self.tile_self = nn.Linear(hidden, hidden)
        self.tile_from_node = nn.Linear(hidden, hidden, bias=False)
        self.edge_self = nn.Linear(hidden, hidden)
        self.edge_from_node = nn.Linear(hidden, hidden, bias=False)

    def forward(self, nodes, tiles, edges, topo: Topology):
        n = self.node_norm(nodes)
        t = self.tile_norm(tiles)
        e = self.edge_norm(edges)
        node_update = (
            self.node_self(n)
            + self.node_from_node(torch.bmm(topo.node_from_node, n))
            + torch.bmm(topo.node_from_tile, self.node_from_tile(t))
            + self.node_from_edge(torch.bmm(topo.node_from_edge, e))
        )
        tile_update = self.tile_self(t) + self.tile_from_node(torch.bmm(topo.tile_from_node, n))
        edge_update = self.edge_self(e) + torch.bmm(topo.edge_from_node, self.edge_from_node(n))
        nodes = (nodes + torch.relu(node_update)) * topo.node_valid
        tiles = (tiles + torch.relu(tile_update)) * topo.tile_valid
        edges = (edges + torch.relu(edge_update)) * topo.edge_valid
        return nodes, tiles, edges


@dataclass
class Embeddings:
    nodes: torch.Tensor    # [B, N, H]
    tiles: torch.Tensor    # [B, T, H]
    edges: torch.Tensor    # [B, E, H]
    players: torch.Tensor  # [B, 6, H]
    pooled: torch.Tensor   # [B, 3H + Hg]
    topo: Topology


def _masked_mean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


class GraphTrunk(nn.Module):
    """Embeds a row's entities and runs the message-passing layers."""

    def __init__(self, reader: RowReader, *, hidden: int, global_hidden: int, layers: int) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("graph networks need at least one layer")
        self.reader = reader
        w = reader.widths
        self.valid = {
            "nodes": reader.feature("nodes", "valid"),
            "tiles": reader.feature("tiles", "valid"),
            "edges": reader.feature("edges", "valid"),
        }
        self.present = reader.feature("players", "present")
        self.hidden = hidden
        self.global_hidden = global_hidden
        self.pooled_dim = 3 * hidden + global_hidden
        self.global_projection = nn.Linear(w["global"] + MAX_PLAYERS * w["players"], global_hidden)
        self.player_projection = nn.Linear(w["players"] + MAX_PLAYERS, hidden)
        self.node_projection = nn.Linear(w["nodes"], hidden)
        self.tile_projection = nn.Linear(w["tiles"], hidden)
        self.edge_projection = nn.Linear(w["edges"], hidden)

        def injection() -> nn.Sequential:
            return nn.Sequential(nn.Linear(hidden + global_hidden, hidden), nn.LayerNorm(hidden), nn.ReLU())

        self.node_injection = injection()
        self.tile_injection = injection()
        self.edge_injection = injection()
        self.layers = nn.ModuleList(HeteroLayer(hidden) for _ in range(layers))
        self.register_buffer("slot_ids", torch.eye(MAX_PLAYERS), persistent=False)
        for module in self.modules():
            orthogonal_(module, HIDDEN_GAIN)

    def forward(self, obs: torch.Tensor, rows: Rows | None = None) -> Embeddings:
        rows = self.reader(obs) if rows is None else rows
        topo = topology(rows, self.valid)
        batch = obs.shape[0]
        glob = torch.relu(self.global_projection(torch.cat((rows.glob, rows.players.flatten(1)), dim=1)))
        slot_ids = self.slot_ids.unsqueeze(0).expand(batch, -1, -1)
        present = rows.players[..., self.present].unsqueeze(-1)
        players = torch.relu(self.player_projection(torch.cat((rows.players, slot_ids), dim=-1))) * present

        def inject(module: nn.Module, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            g = glob.unsqueeze(1).expand(-1, x.shape[1], -1)
            return module(torch.cat((x, g), dim=-1)) * valid

        nodes = inject(self.node_injection, torch.relu(self.node_projection(rows.nodes)), topo.node_valid)
        tiles = inject(self.tile_injection, torch.relu(self.tile_projection(rows.tiles)), topo.tile_valid)
        edges = inject(self.edge_injection, torch.relu(self.edge_projection(rows.edges)), topo.edge_valid)
        for layer in self.layers:
            nodes, tiles, edges = layer(nodes, tiles, edges, topo)
        pooled = torch.cat(
            (
                _masked_mean(nodes, topo.node_valid),
                _masked_mean(tiles, topo.tile_valid),
                _masked_mean(edges, topo.edge_valid),
                glob,
            ),
            dim=1,
        )
        return Embeddings(nodes, tiles, edges, players, pooled, topo)


def _mlp(inputs: int, hidden: int, outputs: int, depth: int, out_gain: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = inputs
    for _ in range(depth):
        layers += [nn.Linear(width, hidden), nn.ReLU()]
        width = hidden
    layers.append(nn.Linear(width, outputs))
    for module in layers[:-1]:
        orthogonal_(module, HIDDEN_GAIN)
    orthogonal_(layers[-1], out_gain)
    return nn.Sequential(*layers)


class GraphPolicyHead(nn.Module):
    """Structured logits in the action layout's block order.

    Globals from the pooled state (two hidden layers), settlement and city
    from node embeddings (one), roads from the edge embedding plus its two
    ends' node embeddings (two), robber moves from the tile embedding and the
    victim slot's player embedding (one); the no-op's logit is 0.

    Scoring each placement from its own position's embedding follows Canopy's
    Nexus-v3 heads, by way of v1's ``CatanGraphPolicyHead``; v1's
    ``HierarchicalPolicyHead`` (Nov 2025) had factored actions into a type and
    its parameters before that.
    """

    def __init__(self, trunk: GraphTrunk, action_layout: Mapping[str, Any], *, head_hidden: int) -> None:
        super().__init__()
        h = trunk.hidden
        blocks = sorted(action_layout.items(), key=lambda item: item[1][0])
        names = [name for name, _ in blocks]
        if names != ["globals", "settlement", "city", "road", "robber", "noop"]:
            raise ValueError(f"unexpected action layout order {names}")
        offset = 0
        for name, (start, shape) in blocks:
            if int(start) != offset:
                raise ValueError(f"action block {name} is not contiguous")
            offset += math.prod(int(s) for s in shape)
        self.num_actions = offset
        self.num_globals = int(action_layout["globals"][1][0])
        self.robber_slots = int(action_layout["robber"][1][1])
        self.other = _mlp(trunk.pooled_dim, head_hidden, self.num_globals, 2, POLICY_OUTPUT_GAIN)
        self.settlement = _mlp(h, head_hidden, 1, 1, POLICY_OUTPUT_GAIN)
        self.city = _mlp(h, head_hidden, 1, 1, POLICY_OUTPUT_GAIN)
        self.road = _mlp(2 * h, head_hidden, 1, 2, POLICY_OUTPUT_GAIN)
        self.robber_tile = nn.Linear(h, head_hidden)
        self.robber_victim = nn.Linear(h, head_hidden, bias=False)
        self.robber_output = nn.Linear(head_hidden, 1)
        self.nobody = nn.Parameter(torch.zeros(h))
        orthogonal_(self.robber_tile, HIDDEN_GAIN)
        orthogonal_(self.robber_victim, HIDDEN_GAIN)
        orthogonal_(self.robber_output, POLICY_OUTPUT_GAIN)

    def forward(self, emb: Embeddings, mask: torch.Tensor) -> torch.Tensor:
        batch = emb.nodes.shape[0]
        other = self.other(emb.pooled)
        settlement = self.settlement(emb.nodes).squeeze(-1)
        city = self.city(emb.nodes).squeeze(-1)
        ends = torch.bmm(emb.topo.edge_ends, emb.nodes)  # sum of both ends: symmetric
        road = self.road(torch.cat((emb.edges, ends), dim=-1)).squeeze(-1)
        victims = torch.cat(
            (self.nobody.expand(batch, 1, -1), emb.players[:, 1 : self.robber_slots]), dim=1
        )  # [B, P, H]: slot 0 steals from nobody
        hidden = torch.relu(
            self.robber_tile(emb.tiles).unsqueeze(2) + self.robber_victim(victims).unsqueeze(1)
        )  # [B, T, P, Hh]
        robber = self.robber_output(hidden).squeeze(-1).flatten(1)
        noop = other.new_zeros(batch, 1)
        logits = torch.cat((other, settlement, city, road, robber, noop), dim=1)
        return mask_logits(logits, mask)


class GraphValueHead(nn.Module):
    """Each seat's win probability (logits) from the pooled state and its embedding."""

    def __init__(self, trunk: GraphTrunk, *, head_hidden: int, num_players: int) -> None:
        super().__init__()
        self.num_players = num_players
        self.mlp = _mlp(trunk.pooled_dim + trunk.hidden, head_hidden, 1, 2, VALUE_OUTPUT_GAIN)

    def forward(self, emb: Embeddings, present: torch.Tensor) -> torch.Tensor:
        players = emb.players[:, : self.num_players]
        pooled = emb.pooled.unsqueeze(1).expand(-1, self.num_players, -1)
        logits = self.mlp(torch.cat((pooled, players), dim=-1)).squeeze(-1).float()
        return logits + value_prior(present)


class GraphAuxHead(nn.Module):
    """The other seats' hidden features (slots 1..P-1) from the pooled state and each
    seat's embedding: ``num_features`` regressed values per seat in the encoder's scale."""

    def __init__(self, trunk: GraphTrunk, *, head_hidden: int, num_players: int, num_features: int) -> None:
        super().__init__()
        if num_players < 2 or num_features < 1:
            raise ValueError("the aux head predicts at least one feature of at least one other seat")
        self.num_players = num_players
        self.num_features = num_features
        self.mlp = _mlp(trunk.pooled_dim + trunk.hidden, head_hidden, num_features, 2, AUX_OUTPUT_GAIN)

    def forward(self, emb: Embeddings) -> torch.Tensor:
        """[B, P-1, F]: slot k+1's features from row k of the output."""
        others = emb.players[:, 1 : self.num_players]
        pooled = emb.pooled.unsqueeze(1).expand(-1, self.num_players - 1, -1)
        return self.mlp(torch.cat((pooled, others), dim=-1)).float()


class GraphNetwork(nn.Module):
    """A trunk with a policy head, a value head or both, and ``aux_features`` > 0 adds the
    auxiliary head (built last, so the other parts' initial weights do not depend on it)."""

    def __init__(self, reader, action_layout, *, args, num_players, policy: bool, value: bool,
                 aux_features: int = 0) -> None:
        super().__init__()
        self.trunk = GraphTrunk(
            reader, hidden=args["hidden"], global_hidden=args["global_hidden"], layers=args["layers"]
        )
        self.policy = (
            GraphPolicyHead(self.trunk, action_layout, head_hidden=args["head_hidden"]) if policy else None
        )
        self.value = (
            GraphValueHead(self.trunk, head_hidden=args["head_hidden"], num_players=num_players) if value else None
        )
        self.aux = (
            GraphAuxHead(self.trunk, head_hidden=args["head_hidden"], num_players=num_players,
                         num_features=int(aux_features)) if aux_features else None
        )
