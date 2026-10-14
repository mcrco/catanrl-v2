import random

import pytest

from catanrl import maps


@pytest.mark.parametrize(
    "config, land",
    [
        ({"type": "hexagon", "radius": 2}, 19),
        ({"type": "hexagon", "radius": 3}, 37),
        ({"type": "extension"}, 30),
        ({"type": "rectangle", "width": 5, "height": 4}, 20),
        ({"type": "blob", "land_tiles": 23}, 23),
    ],
)
def test_generated_maps_are_complete(config, land):
    for seed in range(5):
        spec = maps.generate(config, seed)
        tiles = spec["tiles"]
        land_tiles = [t for t in tiles if t["type"] == "land"]
        assert len(land_tiles) == land
        numbered = [t for t in land_tiles if t["resource"] is not None]
        assert all(t["number"] in (2, 3, 4, 5, 6, 8, 9, 10, 11, 12) for t in numbered)
        assert all(t["number"] is None for t in land_tiles if t["resource"] is None)
        coordinates = [tuple(t["coordinate"]) for t in tiles]
        assert len(set(coordinates)) == len(coordinates)
        assert all(sum(c) == 0 for c in coordinates)
        land_set = {tuple(t["coordinate"]) for t in land_tiles}
        for port in (t for t in tiles if t["type"] == "port"):
            facing = maps.add(tuple(port["coordinate"]), maps.DIRECTIONS[port["direction"]])
            assert facing in land_set
        assert maps.generate(config, seed) == spec


def test_extension_board_matches_the_official_counts():
    spec = maps.generate({"type": "extension"}, 0)
    land = [t for t in spec["tiles"] if t["type"] == "land"]
    assert sum(t["resource"] is None for t in land) == 2
    assert sorted(t["number"] for t in land if t["number"]) == sorted(
        n for n, c in maps.EXTENSION_NUMBERS for _ in range(c)
    )
    assert sum(t["type"] == "port" for t in spec["tiles"]) == 11


def test_six_and_eight_are_kept_apart_on_the_base_shape():
    for seed in range(20):
        spec = maps.generate({"type": "hexagon", "radius": 2}, seed)
        red = {tuple(t["coordinate"]) for t in spec["tiles"] if t["number"] in (6, 8)}
        assert not any(n in red for c in red for n in maps.neighbors(c))


def test_blob_is_connected():
    shape = maps.blob_shape(40, random.Random(3), compactness=0.0)
    seen, todo = {shape[0]}, [shape[0]]
    while todo:
        for n in maps.neighbors(todo.pop()):
            if n in shape and n not in seen:
                seen.add(n)
                todo.append(n)
    assert len(seen) == 40


def test_every_generated_map_has_a_desert_for_the_robber():
    for land in range(3, 41):
        assert None in maps.terrain_for(land)
    for seed in range(20):
        spec = maps.generate({"type": "blob", "land_tiles": [4, 12]}, seed)
        assert any(t["type"] == "land" and t["resource"] is None for t in spec["tiles"])
