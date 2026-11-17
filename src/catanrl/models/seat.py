"""Seat-set graph network: the graph network with the seats as a set.

The graph network (:mod:`catanrl.models.graph`) ties every seat-relative slot to weights of
its own: a slot one-hot on the player embeddings, six ownership channels on every node, edge
and tile, and a global vector that reads the six seats' features side by side. What it learns
about the seat in slot 1 it learns again for slot 3, and a two-seat game never trains the
weights a four-seat game reads. Here the seats are a set processed with shared weights
(Marco, 2026-11-16), and the turn order, which a set forgets, is given back as features and as
an attention bias:

* **seat tokens**: one embedding per slot from one projection of the seat's features and its
  place in the turn order as seen from the row's own seat (slot 0): ``is_self``, a one-hot of
  the forward distance k (the seat moves k seats after the row's own) and a one-hot of the
  backward distance n - k (n = the seats present), from :func:`turn_order_features`. Absent
  slots are zero and stay zero;
* **seats in the graph**: a fourth entity type of the message passing (:class:`SeatLayer`).
  The rows' ownership channels are read as incidence matrices between seats and the board
  (:func:`relations`): seat <-> node by ``settlement_<k>``, by ``city_<k>`` and by
  ``distance_<k>`` in three bands (on the seat's network, one road away, two to five roads
  away weighted by closeness), seat <-> edge by ``road_<k>`` and seat <-> tile by
  ``buildings_<k>`` (weighted by the buildings). A board entity receives the sum of its seats'
  messages and a seat the weighted mean of its entities'; every relation has weights of its
  own, shared by all seats, through a narrow message (``relation_hidden``);
* **slot-free board inputs**: the per-slot channels of the nodes, edges and tiles shrink to
  "mine" (slot 0) and "another seat's" (any present slot k >= 1: the settlement, city, road and
  buildings channels summed, the distance of the nearest of them);
* **seat to seat**: in every layer, multi-head self-attention over the present seats' tokens
  (:class:`SeatAttention`) with a learned bias by relative turn order, from seat i to seat j
  the forward distance (j - i) mod n and the backward distance (i - j) mod n. ``attention:
  false`` puts the masked mean of the other seats' tokens in its place (:class:`SeatPool`);
* **the global vector** reads the plain global features, the row's own token and the mean and
  the maximum of the other present seats' tokens, not the six seats' features side by side;
  it is injected into the seat tokens as into the nodes, tiles and edges;
* **the heads** are the graph network's, on the final seat tokens: the value, aux and outcome
  heads read each seat's token, the robber head the victim's, and the pooled state ends with
  the row's own final token. Actions, masks and outputs are the graph network's.

So permuting the other seats (their features and their ownership channels) permutes the
network's per-seat outputs with them, but for what the turn order tells it, and whatever an
absent slot holds changes nothing.

The dice input (``obs.dice_state``) follows the graph network's rule (no-bias projections,
zero at initialization and built after every other weight): the seat-free global dice
features on the global vector (``dice_projection``), each seat's on its token
(``seat_dice_projection``: its odds, sevens, share and streak) and each tile's on the tile
(``tile_dice_projection``: the odds when the row's own seat rolls, the mean odds over the
other present seats, the deck and the window).

``kernels: fused`` and ``fused16`` run a shared-trunk network of this class (hidden 96, no
aux head) on :mod:`catanrl.models.seat_kernels`: the graph network's kernels for the board
and the heads, kernels of its own for the seats.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from catanrl.envs.encoding import DICE_PREFIX, MAX_PLAYERS
from catanrl.models.graph import (
    GRAPH_DEFAULTS,
    GRAPH_OPTIONAL,
    HIDDEN_GAIN,
    Embeddings,
    GraphAuxHead,
    GraphOutcomeHead,
    GraphPolicyHead,
    GraphValueHead,
    HeteroLayer,
    Topology,
    _masked_mean,
    topology,
)
from catanrl.models.rows import RowReader, Rows, orthogonal_

SEAT_DEFAULTS: dict[str, Any] = {
    **GRAPH_DEFAULTS,
    # The width of a seat's token.
    "seat_hidden": 48,
    # The width of one seat <-> board relation's message, in each direction.
    "relation_hidden": 4,
    # Seat-to-seat self-attention in every layer (with the turn-order bias); off, the masked
    # mean of the other seats' tokens takes its place.
    "attention": True,
    # The attention's width (queries, keys and values), split over its heads.
    "attention_hidden": 24,
    "attention_heads": 4,
}
SEAT_OPTIONAL: dict[str, Any] = dict(GRAPH_OPTIONAL)
#: The seat <-> node relations, in the order of their messages: the seat's settlement, its city,
#: and the node's road distance from the seat's network in three bands (0: on it, 1: one road
#: away, 2..5: further, weighted by 1 - distance / 6; 6 or more, the encoder's cap, is no relation).
NODE_RELATIONS = ("settlement", "city", "on_network", "one_road", "near")
#: ``distance_<k>`` is min(roads / 6, 1): the bands' edges, halfway between two road counts.
DISTANCE_ON = 0.5 / 6.0
DISTANCE_ONE = 1.5 / 6.0
DISTANCE_NEAR = 5.5 / 6.0
#: The global dice features of one seat, by name stem (``<stem>_<k>`` or ``<stem>_<k>_<total>``);
#: the other global dice features are seat-free.
SEAT_DICE = ("dice_odds", "dice_sevens", "dice_seven_share", "dice_streak")
TURN_FEATURES = 2 * MAX_PLAYERS - 1


def turn_order_features() -> torch.Tensor:
    """[MAX_PLAYERS + 1, MAX_PLAYERS, TURN_FEATURES] by the seats present n and the slot k: a
    seat's place in the turn order as seen from the row's own seat. Feature 0 is ``is_self``,
    features 1..5 the one-hot of the forward distance k (the seat moves k seats after the
    row's own), features 6..10 the one-hot of the backward distance n - k. Zero for absent
    slots."""
    table = torch.zeros(MAX_PLAYERS + 1, MAX_PLAYERS, TURN_FEATURES)
    for n in range(1, MAX_PLAYERS + 1):
        table[n, 0, 0] = 1.0
        for k in range(1, n):
            table[n, k, k] = 1.0
            table[n, k, MAX_PLAYERS - 1 + n - k] = 1.0
    return table


def relative_turn_order() -> torch.Tensor:
    """[MAX_PLAYERS + 1, MAX_PLAYERS, MAX_PLAYERS] long by the seats present n: from slot i to
    slot j, the forward distance (j - i) mod n (its transpose is the backward distance); 0
    where a slot is absent."""
    table = torch.zeros(MAX_PLAYERS + 1, MAX_PLAYERS, MAX_PLAYERS, dtype=torch.long)
    for n in range(1, MAX_PLAYERS + 1):
        for i in range(n):
            for j in range(n):
                table[n, i, j] = (j - i) % n
    return table


@dataclass
class Relations:
    """A batch's seats and their incidence with the board. ``*_from_seat`` are the weights of
    the sums a board entity receives; ``seat_from_*`` their transposes, normalized over the
    seat's entities (a weighted mean). Absent seats and padding entities have no incidence."""

    present: torch.Tensor         # [B, 6, 1]
    count: torch.Tensor           # [B] long: the seats present
    absent: torch.Tensor          # [B, 1, 1, 6] bool: the attention's key mask
    others: torch.Tensor          # [B, 6, 1] the other seats present, per seat (at least 1)
    order: torch.Tensor           # [B, 6, 6] long: forward distance from slot i to slot j
    node_from_seat: torch.Tensor  # [B, R, N, 6] by NODE_RELATIONS
    seat_from_node: torch.Tensor  # [B, R, 6, N]
    edge_from_seat: torch.Tensor  # [B, E, 6]
    seat_from_edge: torch.Tensor  # [B, 6, E]
    tile_from_seat: torch.Tensor  # [B, T, 6] (settlements + 2 x cities on the tile)
    seat_from_tile: torch.Tensor  # [B, 6, T]


def _seat_mean(weights: torch.Tensor) -> torch.Tensor:
    """[..., X, 6] weights of entities by seat -> [..., 6, X] normalized over each seat's entities."""
    return (weights / weights.sum(dim=-2, keepdim=True).clamp_min(1e-6)).transpose(-1, -2)


class SeatAttention(nn.Module):
    """Multi-head self-attention over the present seats' tokens. Its scores carry a learned bias
    per head by relative turn order: from seat i to seat j, one for the forward distance
    (j - i) mod n and one for the backward distance (i - j) mod n (zero at initialization)."""

    def __init__(self, seat_hidden: int, attention_hidden: int, heads: int) -> None:
        super().__init__()
        if heads < 1 or attention_hidden % heads:
            raise ValueError(f"attention_hidden {attention_hidden} must split over attention_heads {heads}")
        self.heads = heads
        self.scale = (attention_hidden // heads) ** -0.5
        self.query = nn.Linear(seat_hidden, attention_hidden, bias=False)
        self.key = nn.Linear(seat_hidden, attention_hidden, bias=False)
        self.value = nn.Linear(seat_hidden, attention_hidden, bias=False)
        self.output = nn.Linear(attention_hidden, seat_hidden, bias=False)
        self.forward_bias = nn.Parameter(torch.zeros(MAX_PLAYERS, heads))
        self.backward_bias = nn.Parameter(torch.zeros(MAX_PLAYERS, heads))

    def forward(self, seats: torch.Tensor, rel: Relations) -> torch.Tensor:
        batch = seats.shape[0]

        def split(x: torch.Tensor) -> torch.Tensor:
            return x.view(batch, MAX_PLAYERS, self.heads, -1).transpose(1, 2)  # [B, heads, 6, A / heads]

        scores = torch.matmul(split(self.query(seats)), split(self.key(seats)).transpose(-1, -2)) * self.scale
        bias = self.forward_bias[rel.order] + self.backward_bias[rel.order.transpose(1, 2)]  # [B, 6, 6, heads]
        scores = scores.float() + bias.permute(0, 3, 1, 2)
        weights = torch.softmax(scores.masked_fill(rel.absent, torch.finfo(scores.dtype).min), dim=-1)
        mixed = torch.matmul(weights, split(self.value(seats)))                              # [B, heads, 6, A / heads]
        return self.output(mixed.transpose(1, 2).reshape(batch, MAX_PLAYERS, -1))


class SeatPool(nn.Module):
    """The attention's stand-in (``attention: false``): each seat receives the mean of the
    other present seats' tokens."""

    def __init__(self, seat_hidden: int) -> None:
        super().__init__()
        self.others = nn.Linear(seat_hidden, seat_hidden, bias=False)

    def forward(self, seats: torch.Tensor, rel: Relations) -> torch.Tensor:
        seats = seats * rel.present
        return self.others((seats.sum(dim=1, keepdim=True) - seats) / rel.others)


class SeatLayer(HeteroLayer):
    """:class:`~catanrl.models.graph.HeteroLayer` with the seats as a fourth entity type.

    Besides the board's messages, a node, edge or tile receives its seats' (the sum over each
    relation, of a message ``relation_hidden`` wide per relation) and a seat its nodes', edges'
    and tiles' (the weighted mean over each relation) and the other seats' (attention, or their
    mean). The seats' weights are shared by every seat.
    """

    def __init__(self, hidden: int, seat_hidden: int, relation_hidden: int, *, attention: bool,
                 attention_hidden: int, attention_heads: int) -> None:
        super().__init__(hidden)
        self.relation_hidden = relation_hidden
        node_messages = len(NODE_RELATIONS) * relation_hidden
        self.seat_norm = nn.LayerNorm(seat_hidden)
        self.seat_self = nn.Linear(seat_hidden, seat_hidden)
        # A seat's messages to the board: one per node relation, then the edges', then the tiles'.
        self.seat_messages = nn.Linear(seat_hidden, node_messages + 2 * relation_hidden, bias=False)
        self.node_from_seat = nn.Linear(node_messages, hidden, bias=False)
        self.edge_from_seat = nn.Linear(relation_hidden, hidden, bias=False)
        self.tile_from_seat = nn.Linear(relation_hidden, hidden, bias=False)
        # The board's messages to a seat, in the same order.
        self.node_messages = nn.Linear(hidden, node_messages, bias=False)
        self.edge_messages = nn.Linear(hidden, relation_hidden, bias=False)
        self.tile_messages = nn.Linear(hidden, relation_hidden, bias=False)
        self.seat_from_board = nn.Linear(node_messages + 2 * relation_hidden, seat_hidden, bias=False)
        self.seat_from_seats = (SeatAttention(seat_hidden, attention_hidden, attention_heads) if attention
                                else SeatPool(seat_hidden))

    def forward(self, nodes, tiles, edges, seats, topo: Topology, rel: Relations):
        n = self.node_norm(nodes)
        t = self.tile_norm(tiles)
        e = self.edge_norm(edges)
        s = self.seat_norm(seats)
        batch, width, relations = s.shape[0], self.relation_hidden, len(NODE_RELATIONS)
        out = self.seat_messages(s)                                                       # [B, 6, (R + 2) D]
        to_nodes = out[..., : relations * width].view(batch, MAX_PLAYERS, relations, width).transpose(1, 2)
        to_nodes = torch.matmul(rel.node_from_seat, to_nodes)                             # [B, R, N, D]
        to_edges = torch.bmm(rel.edge_from_seat, out[..., relations * width : (relations + 1) * width])
        to_tiles = torch.bmm(rel.tile_from_seat, out[..., (relations + 1) * width :])
        node_update = (
            self.node_self(n)
            + self.node_from_node(torch.bmm(topo.node_from_node, n))
            + torch.bmm(topo.node_from_tile, self.node_from_tile(t))
            + self.node_from_edge(torch.bmm(topo.node_from_edge, e))
            + self.node_from_seat(to_nodes.transpose(1, 2).flatten(2))
        )
        tile_update = (self.tile_self(t) + self.tile_from_node(torch.bmm(topo.tile_from_node, n))
                       + self.tile_from_seat(to_tiles))
        edge_update = (self.edge_self(e) + torch.bmm(topo.edge_from_node, self.edge_from_node(n))
                       + self.edge_from_seat(to_edges))
        from_nodes = self.node_messages(n).view(batch, -1, relations, width).transpose(1, 2)  # [B, R, N, D]
        from_nodes = torch.matmul(rel.seat_from_node, from_nodes).transpose(1, 2).flatten(2)  # [B, 6, R D]
        from_board = torch.cat(
            (
                from_nodes,
                torch.bmm(rel.seat_from_edge, self.edge_messages(e)),
                torch.bmm(rel.seat_from_tile, self.tile_messages(t)),
            ),
            dim=-1,
        )
        seat_update = self.seat_self(s) + self.seat_from_board(from_board) + self.seat_from_seats(s, rel)
        nodes = (nodes + torch.relu(node_update)) * topo.node_valid
        tiles = (tiles + torch.relu(tile_update)) * topo.tile_valid
        edges = (edges + torch.relu(edge_update)) * topo.edge_valid
        seats = (seats + torch.relu(seat_update)) * rel.present
        return nodes, tiles, edges, seats


class SeatTrunk(nn.Module):
    """Embeds a row's entities, its seats as a set, and runs the message-passing layers."""

    def __init__(self, reader: RowReader, *, hidden: int, global_hidden: int, layers: int, seat_hidden: int,
                 relation_hidden: int, attention: bool, attention_hidden: int, attention_heads: int) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("graph networks need at least one layer")
        if seat_hidden < 1 or relation_hidden < 1:
            raise ValueError("seat_hidden and relation_hidden are positive widths")
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
        self.seat_hidden = seat_hidden
        self.pooled_dim = 3 * hidden + global_hidden + seat_hidden

        def channels(block: str, stem: str) -> list[int]:
            return [reader.feature(block, f"{stem}_{k}") for k in range(MAX_PLAYERS)]

        def without(block: str, *taken: list[int]) -> list[int]:
            names = reader.canonical[block]
            used = {j for group in taken for j in group}
            return [j for j, name in enumerate(names) if j not in used and not name.startswith(DICE_PREFIX)]

        def index(name: str, values: list) -> None:
            self.register_buffer(name, torch.tensor(values, dtype=torch.long), persistent=False)

        # The per-slot channels: the incidence between seats and the board, and, as "mine" and
        # "another seat's", inputs of the entity they sit on.
        settlement, city = channels("nodes", "settlement"), channels("nodes", "city")
        distance = channels("nodes", "distance")
        road, buildings = channels("edges", "road"), channels("tiles", "buildings")
        index("node_settlement", settlement)
        index("node_city", city)
        index("node_distance", distance)
        index("edge_road", road)
        index("tile_buildings", buildings)
        index("node_plain", without("nodes", settlement, city, distance))
        index("edge_plain", without("edges", road))
        index("tile_plain", without("tiles", buildings))
        # The dice input (obs.dice_state), when the rows have it: the last global features (some
        # of one seat each, the rest seat-free) and the last features of each tile.
        names = reader.canonical["global"]
        dice = [j for j, name in enumerate(names) if name.startswith(DICE_PREFIX)]
        tile_dice = [j for j, name in enumerate(reader.canonical["tiles"]) if name.startswith(DICE_PREFIX)]
        if (dice != list(range(w["global"] - len(dice), w["global"]))
                or tile_dice != list(range(w["tiles"] - len(tile_dice), w["tiles"])) or bool(dice) != bool(tile_dice)):
            raise ValueError("the dice inputs must end the global block and each tile's features")
        self.plain_globals = w["global"] - len(dice)

        def seat_of(name: str) -> int | None:
            for stem in SEAT_DICE:
                if name.startswith(stem + "_"):
                    return int(name[len(stem) + 1 :].split("_")[0])
            return None

        seat_dice = [[j for j in dice if seat_of(names[j]) == k] for k in range(MAX_PLAYERS)]
        free_dice = [j for j in dice if seat_of(names[j]) is None]
        if len({len(group) for group in seat_dice}) != 1:
            raise ValueError("every seat has the same global dice features")
        self.dice_inputs = len(free_dice)
        self.seat_dice_inputs = len(seat_dice[0])
        self.tile_dice_inputs = 0
        if dice:
            odds = channels("tiles", "dice_odds")
            rest = [j for j in tile_dice if j not in odds]
            index("global_dice", free_dice)
            index("global_seat_dice", seat_dice)
            index("tile_dice_odds", odds)
            index("tile_dice_plain", rest)
            self.tile_dice_inputs = 2 + len(rest)

        self.seat_projection = nn.Linear(w["players"] + TURN_FEATURES, seat_hidden)
        self.global_projection = nn.Linear(self.plain_globals + 3 * seat_hidden, global_hidden)
        self.node_projection = nn.Linear(len(self.node_plain) + 6, hidden)
        self.tile_projection = nn.Linear(len(self.tile_plain) + 2, hidden)
        self.edge_projection = nn.Linear(len(self.edge_plain) + 2, hidden)

        def injection(width: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(width + global_hidden, width), nn.LayerNorm(width), nn.ReLU())

        self.node_injection = injection(hidden)
        self.tile_injection = injection(hidden)
        self.edge_injection = injection(hidden)
        self.seat_injection = injection(seat_hidden)
        self.layers = nn.ModuleList(
            SeatLayer(hidden, seat_hidden, relation_hidden, attention=attention, attention_hidden=attention_hidden,
                      attention_heads=attention_heads)
            for _ in range(layers)
        )
        self.register_buffer("turn_features", turn_order_features(), persistent=False)
        self.register_buffer("turn_order", relative_turn_order(), persistent=False)
        for module in self.modules():
            orthogonal_(module, HIDDEN_GAIN)

        #: The dice input's weights (rows with obs.dice_state only), as the graph trunk's: zero at
        #: the start and built without drawing from the RNG (skip_init), so every other weight is
        #: drawn as without it.
        def dice_projection(inputs: int, outputs: int) -> nn.Linear | None:
            if not dice:
                return None
            projection = torch.nn.utils.skip_init(nn.Linear, inputs, outputs, bias=False)
            nn.init.zeros_(projection.weight)
            return projection

        self.dice_projection = dice_projection(self.dice_inputs, global_hidden)
        self.seat_dice_projection = dice_projection(self.seat_dice_inputs, seat_hidden)
        self.tile_dice_projection = dice_projection(self.tile_dice_inputs, hidden)

    def relations(self, rows: Rows, topo: Topology) -> Relations:
        """The seats of a batch and their incidence with its board, from the ownership channels."""
        seated = rows.players[..., self.present]                                       # [B, 6]
        count = seated.sum(dim=1).round().long().clamp(0, MAX_PLAYERS)
        seat = seated.unsqueeze(1)                                                     # [B, 1, 6]
        distance = rows.nodes.index_select(-1, self.node_distance)
        near = (distance >= DISTANCE_ONE) & (distance < DISTANCE_NEAR)
        node_seat = torch.stack(
            (
                rows.nodes.index_select(-1, self.node_settlement),
                rows.nodes.index_select(-1, self.node_city),
                (distance < DISTANCE_ON).float(),
                ((distance >= DISTANCE_ON) & (distance < DISTANCE_ONE)).float(),
                near.float() * (1.0 - distance),
            ),
            dim=1,
        ) * (topo.node_valid * seat).unsqueeze(1)                                      # [B, R, N, 6]
        edge_seat = rows.edges.index_select(-1, self.edge_road) * topo.edge_valid * seat
        tile_seat = rows.tiles.index_select(-1, self.tile_buildings) * 6.0 * topo.tile_valid * seat
        present = seated.unsqueeze(-1)
        return Relations(
            present=present,
            count=count,
            absent=(seated <= 0)[:, None, None, :],
            others=(seated.sum(dim=1, keepdim=True).unsqueeze(-1) - present).clamp_min(1.0),
            order=self.turn_order[count],
            node_from_seat=node_seat,
            seat_from_node=_seat_mean(node_seat),
            edge_from_seat=edge_seat,
            seat_from_edge=_seat_mean(edge_seat),
            tile_from_seat=tile_seat,
            seat_from_tile=_seat_mean(tile_seat),
        )

    def forward(self, obs: torch.Tensor, rows: Rows | None = None) -> Embeddings:
        rows = self.reader(obs) if rows is None else rows
        topo = topology(rows, self.valid)
        rel = self.relations(rows, topo)
        present = rel.present
        other = present[:, 1:].transpose(1, 2)                                         # [B, 1, 5]

        def mine_and_others(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return x[..., :1], (x[..., 1:] * other).sum(dim=-1, keepdim=True)

        seat_inputs = self.seat_projection(torch.cat((rows.players, self.turn_features[rel.count]), dim=-1))
        if self.seat_dice_projection is not None:
            seat_inputs = seat_inputs + self.seat_dice_projection(rows.glob[:, self.global_seat_dice])
        seats = torch.relu(seat_inputs) * present
        others = seats[:, 1:]  # zero where absent, and no token is below zero: the plain max is the masked one
        summary = torch.cat((seats[:, 0], others.sum(dim=1) / rel.others[:, 0], others.amax(dim=1)), dim=1)
        glob = self.global_projection(torch.cat((rows.glob[:, : self.plain_globals], summary), dim=1))
        if self.dice_projection is not None:
            glob = glob + self.dice_projection(rows.glob.index_select(-1, self.global_dice))
        glob = torch.relu(glob)

        def inject(module: nn.Module, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            g = glob.unsqueeze(1).expand(-1, x.shape[1], -1)
            return module(torch.cat((x, g), dim=-1)) * valid

        distance = rows.nodes.index_select(-1, self.node_distance)
        nearest = torch.where(other > 0, distance[..., 1:], torch.ones_like(distance[..., 1:])).amin(dim=-1, keepdim=True)
        node_inputs = torch.cat(
            (
                rows.nodes.index_select(-1, self.node_plain),
                *mine_and_others(rows.nodes.index_select(-1, self.node_settlement)),
                *mine_and_others(rows.nodes.index_select(-1, self.node_city)),
                distance[..., :1],
                nearest,
            ),
            dim=-1,
        )
        nodes = inject(self.node_injection, torch.relu(self.node_projection(node_inputs)), topo.node_valid)
        tile_inputs = self.tile_projection(torch.cat(
            (rows.tiles.index_select(-1, self.tile_plain),
             *mine_and_others(rows.tiles.index_select(-1, self.tile_buildings))), dim=-1))
        if self.tile_dice_projection is not None:
            odds = rows.tiles.index_select(-1, self.tile_dice_odds)
            tile_inputs = tile_inputs + self.tile_dice_projection(torch.cat(
                (odds[..., :1], (odds[..., 1:] * other).sum(dim=-1, keepdim=True) / rel.others[:, :1],
                 rows.tiles.index_select(-1, self.tile_dice_plain)), dim=-1))
        tiles = inject(self.tile_injection, torch.relu(tile_inputs), topo.tile_valid)
        edge_inputs = torch.cat(
            (rows.edges.index_select(-1, self.edge_plain),
             *mine_and_others(rows.edges.index_select(-1, self.edge_road))), dim=-1)
        edges = inject(self.edge_injection, torch.relu(self.edge_projection(edge_inputs)), topo.edge_valid)
        seats = inject(self.seat_injection, seats, present)
        for layer in self.layers:
            nodes, tiles, edges, seats = layer(nodes, tiles, edges, seats, topo, rel)
        pooled = torch.cat(
            (
                _masked_mean(nodes, topo.node_valid),
                _masked_mean(tiles, topo.tile_valid),
                _masked_mean(edges, topo.edge_valid),
                glob,
                seats[:, 0],
            ),
            dim=1,
        )
        return Embeddings(nodes, tiles, edges, seats, pooled, topo)


class SeatNetwork(nn.Module):
    """:class:`~catanrl.models.graph.GraphNetwork` on a :class:`SeatTrunk`: the same heads, built
    in the same order."""

    def __init__(self, reader, action_layout: Mapping[str, Any], *, args, num_players, policy: bool, value: bool,
                 aux_features: int = 0, outcome: bool = False) -> None:
        super().__init__()
        self.trunk = SeatTrunk(
            reader, hidden=args["hidden"], global_hidden=args["global_hidden"], layers=args["layers"],
            seat_hidden=args["seat_hidden"], relation_hidden=args["relation_hidden"],
            attention=bool(args["attention"]), attention_hidden=args["attention_hidden"],
            attention_heads=args["attention_heads"],
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
        self.outcome = (
            GraphOutcomeHead(self.trunk, head_hidden=args["head_hidden"], num_players=num_players)
            if outcome else None
        )
