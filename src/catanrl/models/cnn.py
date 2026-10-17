"""Cross-dimensional CNN: v1's xdim over one board laid on a plane.

A port of catanrl v1's ``CrossDimensionalBackbone`` (v1
``src/catanrl/models/backbones.py``), after Gendre and Kaneko's
cross-dimensional network for Catan (arXiv 2008.07079): the board tensor
goes through three convolutions, is flattened and projected; the numeric
features go through an MLP; a two-layer MLP fuses both branches; a linear
layer over the fused vector gives the logits (v1's ``FlatPolicyHead``) or the
values. The flat projection of the convolution output holds most of the
parameters (15.1M of the default's 16.8M), as it did in v1.

**The plane.** v1 read Catanatron's 21 x 11 board tensor: nodes on a grid
two cells apart, roads on the cells between them, and each tile's resource
probability spread onto its six corners. v2 rows carry per-tile, per-node
and per-edge feature blocks and the topology (DESIGN 1.2) but no
coordinates, so :func:`board_grid` derives the same grid once from the
engine's map of the ``board`` generator (BASE by default): a tile at cube
coordinate ``(x, y, z)`` sits at offset row ``z``, column ``x + (z - (z & 1))
// 2`` (Catanatron's ``cube_to_offset``) and takes v1's 3 x 5 footprint (rows
two apart, columns four apart, odd rows shifted by two), its corners NORTH
clockwise at the footprint's top middle, top right, bottom right, bottom
middle, bottom left and top left. Edges lie between their ends. Every map of
a fixed-shape generator lists the same tile coordinates in the same order
and, since node and edge ids follow the tile order (``CatanMap.from_tiles``),
has the same slot order, so the gather is fixed at build time and lives in
non-persistent buffers. The feature blocks are gathered onto the plane with
``index_select`` (no in-place writes: batch-independent, deterministic,
compile-friendly).

Deviations from v1, forced or chosen:

* **Tiles sit at the hex centre**, the cell v1 left empty. v1 spread a tile's
  resource probability onto its corner nodes; v2's node block already carries
  ``pips_<R>`` per node, so the tile block (terrain, number, pips, robber,
  buildings) goes to the centre, where a 3 x 5 kernel sees all six corners and
  edges.
* **Every feature is a channel**: the tile, node and edge blocks' features in
  full (``C = Ft + Fn + Fe``, 47 at two players) instead of v1's 20 hand-picked
  planes; the blocks' ``valid`` features double as cell-type planes.
* **GroupNorm instead of BatchNorm** by default (``norm: group``; v1's compact
  variant did the same). Batch statistics make a row's logits depend on the
  rest of the batch, so the network that collects rollouts (batch statistics
  over the acting rows) differs from the one evaluations run (running
  statistics): the only architecture in the ablation with that mismatch, and
  one that would also break the self-test's and strict evals' invariance
  across batch sizes. ``norm: batch`` restores v1's layers.
* **Heads** follow v2: a per-seat value head (``max_players`` logits plus
  :func:`~catanrl.models.rows.value_prior`) instead of v1's scalar or WDL
  head, and the policy logits land in the engine's action layout (globals,
  settlement and city per node, road per edge, robber per tile x victim slot,
  no-op), masked with :func:`~catanrl.models.rows.mask_logits`.
* **Numeric branch**: the global and players blocks (``critic_players`` for a
  critic at another level), 117 features at two players.
* **One board.** The gather ties every slot to a cell of the ``board``
  generator's map, so the network refuses other dims (any ``max_players`` is
  fine; it is not dims-agnostic, like the MLP). Dims do not tell boards apart,
  though: the shipped ``hexagon`` radius-2 generator has BASE's 19/54/72 slots
  in row-major rather than Catanatron's spiral order, and a BASE network given
  its rows would read a scrambled board with nothing but weaker play to show
  for it. So :meth:`CnnModel.check_maps` compares a map's topology (its tile
  corners and edge ends by slot, which is what the encoder's topology block
  holds) with the board's, and training (``check_bindable``), the eval harness
  and the bridge check their maps before playing. ``board: {type: hexagon,
  radius: 2}`` lays the plane out for those pools instead; any fixed-shape
  generator that fits the 11 x 21 grid (MINI, small rectangles) works the
  same way.

Sizes: ``channels: [64, 128, 128], hidden: 512`` is v1's arch-ablation
``xdim-default`` (16.8M parameters per network); v1's D-S is ``channels: [32,
64, 64], hidden: 1024`` and D-M ``hidden: 2048`` (78.5M).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any

import torch
from torch import nn

from catanrl.models.base import CatanModel
from catanrl.models.rows import feature_names, mask_logits, orthogonal_, value_prior

CNN_DEFAULTS: dict[str, Any] = {
    # The map generator (catanrl.maps.generate config) whose fixed tile order the plane is
    # laid out from; the network trains and plays on that generator's maps only.
    "board": {"type": "BASE"},
    # Convolution widths and kernel (rows x columns; the paper's 5 x 3 on a transposed board).
    "channels": [64, 128, 128],
    "kernel": [3, 5],
    # Width of the numeric MLP, the conv projection, the fusion MLP and the fused vector.
    "hidden": 512,
    "numeric_layers": 2,
    # group: batch-independent (default); batch: v1's BatchNorm2d.
    "norm": "group",
    # One trunk for both heads, or separate actor and critic networks (v1's "sep").
    "shared": False,
}
GRID_ROWS, GRID_COLS = 11, 21
HIDDEN_GAIN = math.sqrt(2.0)
POLICY_OUTPUT_GAIN = 0.01
VALUE_OUTPUT_GAIN = 0.01

Topology = tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, int], ...]]


@dataclass(frozen=True)
class BoardGrid:
    """Flat cell (``row * GRID_COLS + col``) of each tile, node and edge slot of a board,
    and the board's :func:`topology`, which a map must share to be laid out the same."""

    tiles: tuple[int, ...]
    nodes: tuple[int, ...]
    edges: tuple[int, ...]
    topology: Topology

    @property
    def dims(self) -> dict[str, int]:
        return {"max_tiles": len(self.tiles), "max_nodes": len(self.nodes), "max_edges": len(self.edges)}


def topology(catan_map: Any) -> Topology:
    """A map's tile corners (NORTH clockwise) and edge ends, by slot: the encoder's
    topology block. Two maps with equal topology put every slot on the same cell."""
    tiles = tuple(tuple(int(n) for n in catan_map.land_tiles[c].nodes) for c in catan_map.land_tile_coordinates)
    edges = tuple((int(u), int(v)) for u, v in catan_map.land_edges)
    return tiles, edges


def board_grid(board: Mapping[str, Any]) -> BoardGrid:
    """The ``board`` generator's map on v1's 11 x 21 grid (see the module docstring).

    Slots are the encoder's: tiles in map order, nodes ascending, edges
    ascending. The derivation checks itself: every node must land on one cell
    whichever tile places it, cells must not collide, and the map must fit the
    grid, so a change to the map's corner order or a board of another shape
    raises here.
    """
    if not isinstance(board, Mapping) or "type" not in board:
        raise ValueError(f"cnn board must be a map generator config such as {{type: BASE}}, not {board!r}")
    return _board_grid(json.dumps(dict(board), sort_keys=True))


@cache
def _board_grid(key: str) -> BoardGrid:
    import cppanatron as cpp

    from catanrl import maps

    board = json.loads(key)
    catan_map = cpp.CatanMap.from_spec(cpp.MapSpec.from_dict(maps.generate(board, 0)))
    node_cell: dict[int, tuple[int, int]] = {}
    tile_cells: list[tuple[int, int]] = []
    for x, y, z in catan_map.land_tile_coordinates:
        i, j = z, x + (z - (z & 1)) // 2
        r, c = 2 * i, 4 * j + 2 * (i % 2)
        corners = ((r, c + 2), (r, c + 4), (r + 2, c + 4), (r + 2, c + 2), (r + 2, c), (r, c))
        for node, cell in zip(catan_map.land_tiles[(x, y, z)].nodes, corners, strict=True):
            if node_cell.setdefault(node, cell) != cell:
                raise RuntimeError(f"{board}: node {node} falls on two cells: the map's corner order changed")
        tile_cells.append((r + 1, c + 2))
    top = min(r for r, _ in node_cell.values())
    left = min(c for _, c in node_cell.values())
    nodes = sorted(catan_map.land_nodes)
    if nodes != sorted(node_cell):
        raise RuntimeError(f"{board}: the map's land nodes are not its tiles' corners")

    def flat(cell: tuple[int, int]) -> int:
        r, c = cell[0] - top, cell[1] - left
        if not (0 <= r < GRID_ROWS and 0 <= c < GRID_COLS):
            raise RuntimeError(f"{board}: the map does not fit the cnn's {GRID_ROWS} x {GRID_COLS} grid")
        return r * GRID_COLS + c

    edge_cells = []
    for u, v in catan_map.land_edges:
        (r1, c1), (r2, c2) = node_cell[u], node_cell[v]
        if (r1 + r2) % 2 or (c1 + c2) % 2:
            raise RuntimeError(f"{board}: edge {(u, v)} has no cell between its ends")
        edge_cells.append(((r1 + r2) // 2, (c1 + c2) // 2))
    grid = BoardGrid(tiles=tuple(flat(cell) for cell in tile_cells),
                     nodes=tuple(flat(node_cell[node]) for node in nodes),
                     edges=tuple(flat(cell) for cell in edge_cells),
                     topology=topology(catan_map))
    cells = grid.tiles + grid.nodes + grid.edges
    if len(set(cells)) != len(cells):
        raise RuntimeError(f"{board}: two slots share a cell of the grid")
    return grid


def _span(layout: Mapping[str, Any], block: str) -> tuple[int, tuple[int, ...]]:
    offset, shape = layout[block]
    return int(offset), tuple(int(s) for s in shape)


def _slice(obs: torch.Tensor, span: tuple[int, tuple[int, ...]]) -> torch.Tensor:
    offset, shape = span
    return obs[:, offset : offset + math.prod(shape)].reshape(obs.shape[0], *shape)


class BoardPlane(nn.Module):
    """Gathers a row's tile, node and edge blocks onto ``[B, C, 11, 21]``.

    Each block's features form their own channels; a cell holding no slot of
    a block reads zero there (the gather's extra zero row).
    """

    def __init__(self, layout: Mapping[str, Any], grid: BoardGrid) -> None:
        super().__init__()
        self.spans = {block: _span(layout, block) for block in ("tiles", "nodes", "edges")}
        self.channels = sum(shape[1] for _, shape in self.spans.values())
        for block, cells in (("tiles", grid.tiles), ("nodes", grid.nodes), ("edges", grid.edges)):
            count = self.spans[block][1][0]
            if count != len(cells):
                raise ValueError(f"{block}: {count} slots in the row, {len(cells)} on the board's grid")
            source = torch.full((GRID_ROWS * GRID_COLS,), count, dtype=torch.long)  # count = the zero row
            source[torch.tensor(cells, dtype=torch.long)] = torch.arange(count)
            self.register_buffer(f"source_{block}", source, persistent=False)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        planes = []
        for block, span in self.spans.items():
            x = _slice(obs, span)
            x = torch.cat((x, x.new_zeros(x.shape[0], 1, x.shape[2])), dim=1)
            planes.append(x.index_select(1, getattr(self, f"source_{block}")))  # [B, cells, width]
        plane = torch.cat(planes, dim=-1)
        return plane.transpose(1, 2).reshape(obs.shape[0], self.channels, GRID_ROWS, GRID_COLS)


def _norm(kind: str, channels: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "group":
        groups = next(g for g in range(min(8, channels), 0, -1) if channels % g == 0)
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"cnn norm must be group or batch, not {kind!r}")


class CrossDimensionalTrunk(nn.Module):
    """v1's ``CrossDimensionalBackbone``: conv branch, numeric branch, fusion."""

    def __init__(self, layout: Mapping[str, Any], players: str, args: Mapping[str, Any], grid: BoardGrid) -> None:
        super().__init__()
        channels = [int(c) for c in args["channels"]]
        kernel = tuple(int(k) for k in args["kernel"])
        hidden = int(args["hidden"])
        if not channels or any(c < 1 for c in channels):
            raise ValueError("cnn channels must be a non-empty list of widths")
        if len(kernel) != 2 or any(k % 2 == 0 for k in kernel):
            raise ValueError("cnn kernel must be [rows, columns], both odd, to keep the plane's size")
        if int(args["numeric_layers"]) < 1:
            raise ValueError("cnn numeric_layers must be at least 1")
        self.plane = BoardPlane(layout, grid)
        self.numeric_spans = [_span(layout, "global"), _span(layout, players)]
        self.hidden = hidden

        layers: list[nn.Module] = []
        width = self.plane.channels
        for out in channels:
            layers += [nn.Conv2d(width, out, kernel, padding=((kernel[0] - 1) // 2, (kernel[1] - 1) // 2)),
                       _norm(args["norm"], out), nn.ReLU()]
            width = out
        self.cnn = nn.Sequential(*layers)
        self.projection = nn.Sequential(
            nn.Flatten(), nn.Linear(width * GRID_ROWS * GRID_COLS, hidden), nn.LayerNorm(hidden), nn.ReLU()
        )
        numeric: list[nn.Module] = []
        width = sum(math.prod(shape) for _, shape in self.numeric_spans)
        for _ in range(int(args["numeric_layers"])):
            numeric += [nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.ReLU()]
            width = hidden
        self.numeric = nn.Sequential(*numeric)
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
        )
        for module in self.modules():
            orthogonal_(module, HIDDEN_GAIN)
            if isinstance(module, nn.Conv2d):  # v1: orthogonal at gain 1, zero bias
                nn.init.orthogonal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.float()
        spatial = self.projection(self.cnn(self.plane(obs)))
        numeric = self.numeric(torch.cat([obs[:, o : o + math.prod(s)] for o, s in self.numeric_spans], dim=1))
        return self.fusion(torch.cat((spatial, numeric), dim=-1))


class CnnNetwork(nn.Module):
    """A trunk with a linear policy head, a linear value head or both."""

    def __init__(self, layout, players, args, grid, *, num_actions: int, num_players: int,
                 policy: bool, value: bool) -> None:
        super().__init__()
        self.trunk = CrossDimensionalTrunk(layout, players, args, grid)
        self.policy = nn.Linear(self.trunk.hidden, num_actions) if policy else None
        self.value = nn.Linear(self.trunk.hidden, num_players) if value else None
        if self.policy is not None:
            orthogonal_(self.policy, POLICY_OUTPUT_GAIN)
        if self.value is not None:
            orthogonal_(self.value, VALUE_OUTPUT_GAIN)


class CnnModel(CatanModel):
    dims_agnostic = False

    def __init__(self, spec, dims, levels, layout) -> None:
        super().__init__(spec, dims, levels, layout)
        args = spec["args"]
        self.grid = board_grid(args["board"])  # validates the config
        self.board = dict(args["board"])
        board_dims = {key: self.dims[key] for key in self.grid.dims}
        if board_dims != self.grid.dims:
            raise ValueError(f"the cnn network lays the {self.board} board on a plane: dims must be "
                             f"{self.grid.dims} (any max_players), not {board_dims}")
        names = feature_names(dims, levels["level"])
        self.present_index = names["players"].index("present")
        self.players_offset, (_, self.player_width) = layout["players"]
        if args["shared"]:
            if self.has_critic_block:
                raise ValueError("a shared trunk cannot read a separate critic level")
            net = CnnNetwork(layout, "players", args, self.grid, num_actions=self.num_actions,
                             num_players=self.num_players, policy=True, value=True)
            self.backbone, self.policy_head, self.value_head = net.trunk, net.policy, net.value
            self.parts = {"actor": ["backbone.", "policy_head."], "critic": ["value_head."]}
        else:
            self.actor = CnnNetwork(layout, "players", args, self.grid, num_actions=self.num_actions,
                                    num_players=self.num_players, policy=True, value=False)
            self.critic = CnnNetwork(layout, self.critic_players, args, self.grid, num_actions=self.num_actions,
                                     num_players=self.num_players, policy=False, value=True)
            self.parts = {"actor": ["actor."], "critic": ["critic."]}
        self.shared = bool(args["shared"])

    def check_maps(self, maps: Iterable[Any]) -> None:
        """Refuse a map whose slots are not in the board's order (see the module docstring)."""
        for i, catan_map in enumerate(maps):
            if topology(catan_map) != self.grid.topology:
                raise ValueError(
                    f"map {i} is not laid out like the cnn's board {self.board}: its tiles, nodes or edges "
                    f"come in another order and the plane would scramble it; a cnn trains and plays on its "
                    f"board's generator only (model.args.board)")

    def check_rows(self, obs: torch.Tensor) -> None:
        """Refuse a batch holding rows of a map laid out unlike the board, by their topology
        blocks. For tests and diagnostics: the forward pass trusts :meth:`check_maps`, and
        padded (all-zero) rows of a strict eval batch would fail here."""
        bad = torch.zeros(obs.shape[0], dtype=torch.bool, device=obs.device)
        for block, reference in zip(("tile_nodes", "edge_nodes"), self.grid.topology, strict=True):
            expected = torch.tensor(reference, dtype=torch.long, device=obs.device)
            bad |= (_slice(obs, _span(self.layout, block)).long() != expected).flatten(1).any(1)
        if bool(bad.any()):
            raise ValueError(f"{int(bad.sum())} of {obs.shape[0]} rows come from maps not laid out like "
                             f"the cnn's board {self.board}")

    def _present(self, obs):
        players = obs[:, self.players_offset : self.players_offset + self.num_players * self.player_width]
        return players.reshape(-1, self.num_players, self.player_width)[..., self.present_index].float()

    def _mask(self, obs):
        return obs[:, self.mask_offset : self.mask_offset + self.num_actions]

    def heads(self, obs):
        if self.shared:
            emb = self.backbone(obs)
            present = self._present(obs)
            logits = mask_logits(self.policy_head(emb), self._mask(obs))
            return logits, self.value_head(emb).float() + value_prior(present), present
        return super().heads(obs)

    def policy_logits(self, obs):
        if self.shared:
            return self.heads(obs)[0]
        return mask_logits(self.actor.policy(self.actor.trunk(obs)), self._mask(obs))

    def value_logits(self, obs):
        if self.shared:
            _, values, present = self.heads(obs)
            return values, present
        present = self._present(obs)
        return self.critic.value(self.critic.trunk(obs)).float() + value_prior(present), present
