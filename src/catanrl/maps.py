"""Map generators: a generator and a seed give an explicit map spec.

A map spec is the JSON shape both engines build maps from (cppanatron's
``MapSpec.from_dict`` and, through it, Catanatron's ``CatanMap.from_tiles``)::

    {"tiles": [{"coordinate": [x, y, z], "type": "land" | "water" | "port",
                "resource": "WOOD" | ... | None,     # land: None is the desert
                "number": int | None,
                "direction": "EAST" | ... | None,    # port: the side facing land
                "port_resource": "WOOD" | ... | None}]}  # port: None is 3:1

Tile order defines node and edge ids, so a spec is the whole map.
Coordinates are Catanatron's cube coordinates: x + y + z = 0, z is the row
(EAST is (1, -1, 0)) and hexes are pointy-topped.

Generators are addressed by a config dict, e.g. ``{"type": "BASE"}``,
``{"type": "hexagon", "radius": 3}``, ``{"type": "extension"}`` (the 5-6
player board), ``{"type": "blob", "land_tiles": 25}`` or ``{"type":
"explicit", "tiles": [...]}``; see :func:`generate`.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping
from typing import Any

Coordinate = tuple[int, int, int]
MapSpec = dict[str, Any]

RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")

# Catanatron's Direction -> unit vector.
DIRECTIONS: dict[str, Coordinate] = {
    "EAST": (1, -1, 0),
    "SOUTHEAST": (0, -1, 1),
    "SOUTHWEST": (-1, 0, 1),
    "WEST": (-1, 1, 0),
    "NORTHWEST": (0, 1, -1),
    "NORTHEAST": (1, 0, -1),
}

# The base game's 18 number tokens and 19 terrain tiles.
BASE_NUMBERS = (2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12)
BASE_TERRAIN = {"WOOD": 4, "BRICK": 3, "SHEEP": 4, "WHEAT": 4, "ORE": 3, None: 1}

# The 5-6 player extension: 30 terrain tiles, 28 numbers, 11 harbors.
EXTENSION_TERRAIN = {"WOOD": 6, "BRICK": 5, "SHEEP": 6, "WHEAT": 6, "ORE": 5, None: 2}
EXTENSION_NUMBERS = (
    (2, 2), (3, 3), (4, 3), (5, 3), (6, 3), (8, 3), (9, 3), (10, 3), (11, 3), (12, 2)
)
EXTENSION_PORTS = ("WOOD", "BRICK", "SHEEP", "SHEEP", "WHEAT", "ORE", None, None, None, None, None)

CATANATRON_TEMPLATES = ("BASE", "MINI", "TOURNAMENT")


def add(a: Coordinate, b: Coordinate) -> Coordinate:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def neighbors(coordinate: Coordinate) -> list[Coordinate]:
    return [add(coordinate, vector) for vector in DIRECTIONS.values()]


def cube(x: int, z: int) -> Coordinate:
    """The cube coordinate of column ``x`` (axial) in row ``z``."""
    return (x, -x - z, z)


# ---------------------------------------------------------------- shapes


def hexagon_shape(radius: int) -> list[Coordinate]:
    """A regular hexagon of land: radius 2 is the base game's 19 tiles."""
    return [
        cube(x, z)
        for z in range(-radius, radius + 1)
        for x in range(-radius, radius + 1)
        if abs(x + z) <= radius
    ]


def extension_shape() -> list[Coordinate]:
    """The 5-6 player board: rows of 3, 4, 5, 6, 5, 4, 3 tiles."""
    return [
        cube(x, z)
        for z in range(-3, 4)
        for x in range(-3, 3)
        if -3 <= x + z <= 2
    ]


def rectangle_shape(width: int, height: int) -> list[Coordinate]:
    """``height`` rows of ``width`` tiles, alternate rows offset by half a hex."""
    shape = []
    for row in range(height):
        z = row - height // 2
        start = -((row - (height // 2)) // 2) - width // 2
        shape.extend(cube(start + i, z) for i in range(width))
    return shape


def blob_shape(land_tiles: int, rng: random.Random, compactness: float = 2.0) -> list[Coordinate]:
    """A random connected island grown from the center.

    Frontier hexes are drawn with weight ``(land neighbors) ** compactness``,
    so larger ``compactness`` gives rounder islands and 0 gives stringy ones.
    """
    land = [(0, 0, 0)]
    taken = {land[0]}
    while len(land) < land_tiles:
        frontier = sorted({n for c in land for n in neighbors(c) if n not in taken})
        weights = [sum(n in taken for n in neighbors(c)) ** compactness for c in frontier]
        choice = rng.choices(frontier, weights=weights)[0]
        land.append(choice)
        taken.add(choice)
    return sorted(land, key=lambda c: (c[2], c[0]))


# ---------------------------------------------------------------- contents


def _balanced(counts: Mapping[Any, float], total: int) -> list[Any]:
    """``total`` items in proportion to ``counts`` (largest remainder)."""
    weight = sum(counts.values())
    exact = {key: total * value / weight for key, value in counts.items()}
    chosen = {key: math.floor(value) for key, value in exact.items()}
    by_remainder = sorted(exact, key=lambda key: exact[key] - chosen[key], reverse=True)
    for key in by_remainder[: total - sum(chosen.values())]:
        chosen[key] += 1
    return [key for key, count in chosen.items() for _ in range(count)]


def terrain_for(land_tiles: int) -> list[str | None]:
    if land_tiles == 19:
        return [key for key, count in BASE_TERRAIN.items() for _ in range(count)]
    if land_tiles == 30:
        return [key for key, count in EXTENSION_TERRAIN.items() for _ in range(count)]
    return _balanced(BASE_TERRAIN, land_tiles)


def numbers_for(numbered_tiles: int) -> list[int]:
    if numbered_tiles == 18:
        return list(BASE_NUMBERS)
    if numbered_tiles == 28:
        return [number for number, count in EXTENSION_NUMBERS for _ in range(count)]
    frequencies: dict[int, int] = {}
    for number in BASE_NUMBERS:
        frequencies[number] = frequencies.get(number, 0) + 1
    return _balanced(frequencies, numbered_tiles)


def ports_for(count: int) -> list[str | None]:
    """The base game has 5 resource harbors and 4 generic ones."""
    if count == 11:
        return list(EXTENSION_PORTS)
    generic = count * 4 // 9
    return [RESOURCES[i % 5] for i in range(count - generic)] + [None] * generic


def _place_numbers(
    land: list[Coordinate],
    resources: dict[Coordinate, str | None],
    rng: random.Random,
    attempts: int = 1000,
) -> dict[Coordinate, int]:
    """Shuffles numbers onto non-desert tiles, keeping 6s and 8s apart when possible."""
    tiles = [c for c in land if resources[c] is not None]
    numbers = numbers_for(len(tiles))
    best: dict[Coordinate, int] = {}
    best_conflicts = math.inf
    for _ in range(attempts):
        rng.shuffle(numbers)
        placed = dict(zip(tiles, numbers))
        conflicts = sum(
            1
            for c in tiles
            for n in neighbors(c)
            if placed[c] in (6, 8) and placed.get(n) in (6, 8)
        )
        if conflicts < best_conflicts:
            best, best_conflicts = placed, conflicts
        if conflicts == 0:
            break
    return best


def _coast(land: Iterable[Coordinate]) -> list[Coordinate]:
    """Water hexes touching land, in angular order around the island."""
    land = set(land)
    ring = {n for c in land for n in neighbors(c) if n not in land}

    def angle(c: Coordinate) -> float:
        x, _, z = c
        # Pixel position of a pointy-top hex in axial coordinates.
        return math.atan2(1.5 * z, math.sqrt(3) * (x + z / 2))

    return sorted(ring, key=angle)


def _port_direction(port: Coordinate, land: set[Coordinate]) -> str | None:
    """A side of ``port`` that faces land, preferring the middle of its land sides."""
    facing = [name for name, vector in DIRECTIONS.items() if add(port, vector) in land]
    if not facing:
        return None
    return facing[len(facing) // 2]


def _port_nodes(port: Coordinate, direction: str) -> frozenset[tuple[Coordinate, str]]:
    """The port's two corners, named by the hexes that meet there (for spacing)."""
    order = list(DIRECTIONS)
    i = order.index(direction)
    a = add(port, DIRECTIONS[direction])
    b = add(port, DIRECTIONS[order[(i + 1) % 6]])
    c = add(port, DIRECTIONS[order[(i - 1) % 6]])
    return frozenset({frozenset({port, a, b}), frozenset({port, a, c})})


def spec_from_shape(
    land: list[Coordinate],
    rng: random.Random,
    *,
    port_count: int | None = None,
) -> MapSpec:
    """Fills a land shape with shuffled terrain, numbers and harbors, and rings it with water."""
    land_set = set(land)
    terrain = terrain_for(len(land))
    rng.shuffle(terrain)
    resources = dict(zip(land, terrain))
    numbers = _place_numbers(land, resources, rng)

    coast = _coast(land)
    if port_count is None:
        port_count = len(coast) // 2
    harbors = ports_for(port_count)
    rng.shuffle(harbors)
    # Every other coastal hex, from a random start, skipping harbors whose
    # corners would touch a harbor already placed.
    start = rng.randrange(len(coast)) if coast else 0
    ports: dict[Coordinate, str] = {}
    used_corners: set = set()
    for step in [*range(0, len(coast), 2), *range(1, len(coast), 2)]:
        if len(ports) == len(harbors):
            break
        hexagon = coast[(start + step) % len(coast)]
        direction = _port_direction(hexagon, land_set)
        if direction is None:
            continue
        corners = _port_nodes(hexagon, direction)
        if corners & used_corners:
            continue
        ports[hexagon] = direction
        used_corners |= corners

    harbor_resources = iter(harbors)
    tiles: list[dict[str, Any]] = []
    for c in land:
        tiles.append(
            {"coordinate": list(c), "type": "land", "resource": resources[c],
             "number": numbers.get(c), "direction": None, "port_resource": None}
        )
    for c in coast:
        if c in ports:
            tiles.append(
                {"coordinate": list(c), "type": "port", "resource": None, "number": None,
                 "direction": ports[c], "port_resource": next(harbor_resources)}
            )
        else:
            tiles.append(
                {"coordinate": list(c), "type": "water", "resource": None, "number": None,
                 "direction": None, "port_resource": None}
            )
    return {"tiles": tiles}


# ---------------------------------------------------------------- entry points


def catanatron_template(name: str, seed: int, number_placement: str = "official_spiral") -> MapSpec:
    """Catanatron's own BASE / MINI / TOURNAMENT maps, shuffled by ``seed``."""
    from catanatron.models.map import (
        BASE_MAP_TEMPLATE,
        MINI_MAP_TEMPLATE,
        TOURNAMENT_MAP,
        CatanMap,
    )
    from cppanatron.catanatron import map_spec_from_catanatron

    if name == "TOURNAMENT":
        catan_map = TOURNAMENT_MAP
    else:
        template = BASE_MAP_TEMPLATE if name == "BASE" else MINI_MAP_TEMPLATE
        catan_map = CatanMap.from_template(
            template, number_placement=number_placement, rng=random.Random(seed)
        )
    spec = map_spec_from_catanatron(catan_map)
    return spec.to_dict() if hasattr(spec, "to_dict") else spec


def generate(config: Mapping[str, Any], seed: int) -> MapSpec:
    """The map ``config`` describes, shuffled by ``seed``."""
    kind = str(config.get("type", "BASE"))
    rng = random.Random(seed)
    if kind.upper() in CATANATRON_TEMPLATES:
        return catanatron_template(
            kind.upper(), seed, config.get("number_placement", "official_spiral")
        )
    if kind == "explicit":
        return {"tiles": [dict(tile) for tile in config["tiles"]]}
    if kind == "extension":
        return spec_from_shape(extension_shape(), rng, port_count=11)
    if kind == "hexagon":
        return spec_from_shape(hexagon_shape(int(config["radius"])), rng,
                               port_count=config.get("ports"))
    if kind == "rectangle":
        return spec_from_shape(
            rectangle_shape(int(config["width"]), int(config["height"])), rng,
            port_count=config.get("ports"),
        )
    if kind == "blob":
        land_tiles = config["land_tiles"]
        if isinstance(land_tiles, (list, tuple)):  # a range to draw from
            land_tiles = rng.randint(int(land_tiles[0]), int(land_tiles[1]))
        shape = blob_shape(int(land_tiles), rng, float(config.get("compactness", 2.0)))
        return spec_from_shape(shape, rng, port_count=config.get("ports"))
    raise ValueError(f"unknown map type {kind!r}")


def land_tile_count(spec: MapSpec) -> int:
    return sum(tile["type"] == "land" for tile in spec["tiles"])


def describe(spec: MapSpec) -> str:
    """A one-line summary, e.g. for logs."""
    land = land_tile_count(spec)
    ports = sum(tile["type"] == "port" for tile in spec["tiles"])
    deserts = sum(tile["type"] == "land" and tile["resource"] is None for tile in spec["tiles"])
    return f"{land} land tiles ({deserts} desert), {ports} harbors"
