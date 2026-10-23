"""Canopy's fixed board and its correspondence with a cppanatron map.

Canopy (cullback/canopy, ``examples/catan``) plays one board shape: the base game's 19 land
hexes in a fixed order (:data:`LAND_HEXES_AXIAL`), node and edge ids numbered first-seen over
the tiles' corners N, NE, SE, S, SW, NW (edge ``k`` of a tile joins corners ``k`` and ``k+1``),
and nine ports at one of two position sets. The constants here are copied from the bridge's
``{"topology": true}`` answer; :class:`catanrl.canopy.bridge.BridgeProcess` checks the binary
still answers them.

:meth:`CanopyBoard.from_map` pairs a cppanatron map with them: Catanatron's cube coordinate of
axial ``(q, r)`` is ``(q, -q - r, r)`` and a cppanatron tile lists its nodes in Canopy's corner
order. It raises :class:`MapError` for a map Canopy cannot represent (another shape, ports
off both position sets).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cppanatron as cpp

LAND_HEXES_AXIAL = ((0, -2), (1, -2), (2, -2), (2, -1), (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 2), (-2, 1),
                    (-2, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, 0))
TILE_NODES = (
    (0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 2, 1), (10, 11, 12, 13, 8, 7), (12, 14, 15, 16, 17, 13),
    (15, 18, 19, 20, 21, 16), (21, 20, 22, 23, 24, 25), (24, 23, 26, 27, 28, 29), (30, 29, 28, 31, 32, 33),
    (34, 33, 32, 35, 36, 37), (38, 39, 34, 37, 40, 41), (42, 43, 38, 41, 44, 45), (4, 3, 46, 43, 42, 47),
    (2, 9, 48, 49, 46, 3), (8, 13, 17, 50, 48, 9), (17, 16, 21, 25, 51, 50), (51, 25, 24, 29, 30, 52),
    (53, 52, 30, 33, 34, 39), (46, 49, 53, 39, 38, 43), (48, 50, 51, 52, 53, 49))
TILE_EDGES = (
    (0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 1, 10), (11, 12, 13, 14, 7, 15), (16, 17, 18, 19, 20, 13),
    (21, 22, 23, 24, 25, 18), (24, 26, 27, 28, 29, 30), (28, 31, 32, 33, 34, 35), (36, 34, 37, 38, 39, 40),
    (41, 39, 42, 43, 44, 45), (46, 47, 45, 48, 49, 50), (51, 52, 50, 53, 54, 55), (3, 56, 57, 51, 58, 59),
    (9, 60, 61, 62, 56, 2), (14, 20, 63, 64, 60, 8), (19, 25, 30, 65, 66, 63), (65, 29, 35, 36, 67, 68),
    (69, 67, 40, 41, 47, 70), (62, 71, 70, 46, 52, 57), (64, 66, 68, 69, 71, 61))
EDGE_NODES = (
    (0, 1), (2, 1), (3, 2), (4, 3), (4, 5), (5, 0), (6, 7), (8, 7), (9, 8), (2, 9), (1, 6), (10, 11), (11, 12),
    (13, 12), (8, 13), (7, 10), (12, 14), (14, 15), (16, 15), (17, 16), (13, 17), (15, 18), (18, 19), (19, 20),
    (21, 20), (16, 21), (20, 22), (22, 23), (24, 23), (25, 24), (21, 25), (23, 26), (26, 27), (27, 28), (29, 28),
    (24, 29), (29, 30), (28, 31), (31, 32), (33, 32), (30, 33), (33, 34), (32, 35), (35, 36), (36, 37), (34, 37),
    (39, 38), (34, 39), (37, 40), (40, 41), (38, 41), (43, 42), (38, 43), (41, 44), (44, 45), (45, 42), (46, 3),
    (43, 46), (42, 47), (47, 4), (48, 9), (49, 48), (46, 49), (50, 17), (48, 50), (51, 25), (50, 51), (52, 30),
    (51, 52), (52, 53), (53, 39), (53, 49))
PORT_NODES = {
    "primary": ((18, 19), (22, 23), (28, 31), (35, 36), (40, 41), (42, 47), (0, 5), (6, 7), (12, 14)),
    "alt": ((1, 6), (10, 11), (14, 15), (20, 22), (26, 27), (31, 32), (37, 40), (44, 45), (4, 47)),
}
NUM_TILES, NUM_NODES, NUM_EDGES, NUM_PORTS = 19, 54, 72, 9

#: Resources in cppanatron's (and Catanatron's) order, which is Canopy's (Lumber, Brick, Wool,
#: Grain, Ore); the names are the bridge's.
RESOURCES = (cpp.Resource.WOOD, cpp.Resource.BRICK, cpp.Resource.SHEEP, cpp.Resource.WHEAT, cpp.Resource.ORE)
RESOURCE_NAMES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")


def topology_answer() -> dict[str, Any]:
    """The bridge's ``{"topology": true}`` answer these constants were copied from (no ``id``)."""
    return {"land_hexes_axial": [list(h) for h in LAND_HEXES_AXIAL], "tile_nodes": [list(t) for t in TILE_NODES],
            "tile_edges": [list(t) for t in TILE_EDGES], "edge_nodes": [list(e) for e in EDGE_NODES],
            "port_nodes_primary": [list(p) for p in PORT_NODES["primary"]],
            "port_nodes_alt": [list(p) for p in PORT_NODES["alt"]]}


class MapError(ValueError):
    """A map Canopy's fixed board cannot represent."""


def _edge(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


@dataclass(frozen=True)
class CanopyBoard:
    """One cppanatron map in Canopy's ids."""

    tile_coords: tuple[tuple[int, int, int], ...]          # Canopy tile -> cube coordinate
    coord_to_tile: dict[tuple[int, int, int], int]
    canopy_to_node: tuple[int, ...]                         # Canopy node -> cppanatron node
    node_to_canopy: dict[int, int]
    canopy_to_edge: tuple[tuple[int, int], ...]             # Canopy edge -> cppanatron edge (sorted)
    edge_to_canopy: dict[tuple[int, int], int]
    terrains: tuple[str, ...]                                # "WOOD".."ORE" or "DESERT", Canopy tile order
    numbers: tuple[int | None, ...]
    ports: tuple[str | None, ...]                            # resource name or None (3:1), port-set order
    port_set: str

    @classmethod
    def from_map(cls, catan_map: cpp.CatanMap) -> CanopyBoard:
        land = catan_map.land_tiles
        coords = tuple((q, -q - r, r) for q, r in LAND_HEXES_AXIAL)
        if len(land) != NUM_TILES or set(land) != set(coords):
            raise MapError(f"Canopy plays the base game's 19 hexes; this map has {len(land)} land tiles"
                           + ("" if len(land) != NUM_TILES else " elsewhere"))
        node_to_canopy: dict[int, int] = {}
        edge_to_canopy: dict[tuple[int, int], int] = {}
        terrains, numbers = [], []
        for t, coord in enumerate(coords):
            tile = land[coord]
            nodes = tuple(int(n) for n in tile.nodes)
            for k in range(6):
                node, expected = nodes[k], TILE_NODES[t][k]
                if node_to_canopy.setdefault(node, expected) != expected:
                    raise MapError(f"node {node} of tile {coord} is Canopy node {node_to_canopy[node]} and {expected}")
                edge, expected = _edge(nodes[k], nodes[(k + 1) % 6]), TILE_EDGES[t][k]
                if edge_to_canopy.setdefault(edge, expected) != expected:
                    raise MapError(f"edge {edge} of tile {coord} is Canopy edge {edge_to_canopy[edge]} and {expected}")
            terrains.append("DESERT" if tile.resource is None else RESOURCE_NAMES[RESOURCES.index(tile.resource)])
            numbers.append(None if tile.number is None else int(tile.number))
        if len(node_to_canopy) != NUM_NODES or len(edge_to_canopy) != NUM_EDGES:
            raise MapError(f"the map's land has {len(node_to_canopy)} nodes and {len(edge_to_canopy)} edges, "
                           f"Canopy's {NUM_NODES} and {NUM_EDGES}")
        canopy_to_node = tuple(sorted(node_to_canopy, key=node_to_canopy.__getitem__))
        canopy_to_edge = tuple(sorted(edge_to_canopy, key=edge_to_canopy.__getitem__))
        for e, (a, b) in enumerate(EDGE_NODES):
            if _edge(canopy_to_node[a], canopy_to_node[b]) != canopy_to_edge[e]:
                raise MapError(f"Canopy edge {e} joins nodes {a} and {b}, the map's does not")
        ports, port_set = _ports(catan_map, canopy_to_node)
        return cls(tile_coords=coords, coord_to_tile={c: i for i, c in enumerate(coords)},
                   canopy_to_node=canopy_to_node, node_to_canopy=node_to_canopy, canopy_to_edge=canopy_to_edge,
                   edge_to_canopy=edge_to_canopy, terrains=tuple(terrains), numbers=tuple(numbers),
                   ports=ports, port_set=port_set)


def _ports(catan_map: cpp.CatanMap, canopy_to_node: tuple[int, ...]) -> tuple[tuple[str | None, ...], str]:
    """Each port's resource in the order of the position set the map's ports sit on."""
    by_node: dict[int, Any] = {}
    for resource, nodes in catan_map.port_nodes.items():
        for node in nodes:
            by_node[int(node)] = resource
    for name, pairs in PORT_NODES.items():
        cpp_pairs = [(canopy_to_node[a], canopy_to_node[b]) for a, b in pairs]
        if {n for pair in cpp_pairs for n in pair} != set(by_node):
            continue
        out = []
        for a, b in cpp_pairs:
            if by_node[a] != by_node[b]:
                break
            out.append(None if by_node[a] is None else RESOURCE_NAMES[RESOURCES.index(by_node[a])])
        else:
            return tuple(out), name
    raise MapError("the map's ports are on neither of Canopy's port position sets")
