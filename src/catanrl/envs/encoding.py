"""Graph encoders, their dims and the row layout networks read.

Every feature, action index and mask comes from cppanatron's ``GraphEncoder``
(DESIGN 1.2); this module only builds encoders and answers how large they
must be:

* :func:`make_encoder` builds the ``cpp.GraphEncoder`` for a dims dict and an
  information level;
* :func:`dims_for` gives the dims covering a list of scenarios, by default
  from the map generators' seed-independent bounds (``dims: auto``);
* :func:`row_layout` is the ``{block: (offset, shape)}`` dict of a full
  environment row, which :func:`catanrl.models.build_model` takes as
  ``layout``.

Dims are plain dicts with the keys of :data:`DIM_KEYS` so they serialize into
configs and checkpoint headers unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import cppanatron as cpp

from catanrl import games, maps

DIM_KEYS = ("max_players", "max_tiles", "max_nodes", "max_edges")
LEVELS = {
    "private": cpp.InfoLevel.PRIVATE,
    "public": cpp.InfoLevel.PUBLIC,
    "full": cpp.InfoLevel.FULL,
}
#: The most seats any game has; graph networks pad seat slots to it.
MAX_PLAYERS = 6
#: Blocks of a row that hold per-slot features.
FEATURE_BLOCKS = ("global", "players", "tiles", "nodes", "edges")
#: The critic block VecEnv appends when the critic reads another level.
CRITIC_PLAYERS = "critic_players"


def info_level(level: str | cpp.InfoLevel) -> cpp.InfoLevel:
    if isinstance(level, cpp.InfoLevel):
        return level
    try:
        return LEVELS[str(level)]
    except KeyError:
        raise ValueError(f"unknown information level {level!r}; expected one of {sorted(LEVELS)}")


def level_name(level: str | cpp.InfoLevel) -> str:
    level = info_level(level)
    return next(name for name, value in LEVELS.items() if value == level)


def check_dims(dims: Mapping[str, Any]) -> dict[str, int]:
    """``dims`` as a plain dict of ints with exactly :data:`DIM_KEYS`."""
    if isinstance(dims, cpp.EncoderDims):
        return {key: int(getattr(dims, key)) for key in DIM_KEYS}
    missing = [key for key in DIM_KEYS if key not in dims]
    extra = [key for key in dims if key not in DIM_KEYS]
    if missing or extra:
        raise ValueError(f"dims need exactly {DIM_KEYS}; missing {missing}, unknown {extra}")
    return {key: int(dims[key]) for key in DIM_KEYS}


def encoder_dims(dims: Mapping[str, Any] | cpp.EncoderDims) -> cpp.EncoderDims:
    if isinstance(dims, cpp.EncoderDims):
        return dims
    return cpp.EncoderDims(**check_dims(dims))


def make_encoder(dims: Mapping[str, Any] | cpp.EncoderDims, level: str | cpp.InfoLevel) -> cpp.GraphEncoder:
    """The graph encoder (and through it the action space) for ``dims`` at ``level``."""
    return cpp.GraphEncoder(encoder_dims(dims), info_level(level))


def action_size(dims: Mapping[str, Any]) -> int:
    return int(cpp.GraphActionSpace(encoder_dims(dims)).size)


# ---------------------------------------------------------------- row layout


def critic_level_of(level: str, critic_level: str | None) -> str | None:
    """The critic's level when it differs from the actor's, else None."""
    if critic_level is None:
        return None
    critic = level_name(critic_level)
    return None if critic == level_name(level) else critic


def row_layout(dims: Mapping[str, Any], level: str, critic_level: str | None = None) -> dict[str, Any]:
    """``{block: (offset, shape)}`` of one environment row.

    The blocks are the encoder's (``global``, ``players``, ``tiles``,
    ``nodes``, ``edges``, ``edge_nodes``, ``tile_nodes``, ``action_mask``).
    When the critic reads another information level, VecEnv appends the
    critic block (engine E5): ``critic_players``, the players block at the
    critic's level (the only block that differs between levels), right after
    the actor's row; its feature names are the players block's.
    """
    encoder = make_encoder(dims, level)
    layout: dict[str, Any] = {
        name: (int(offset), tuple(int(s) for s in shape))
        for name, (offset, shape) in encoder.layout().items()
    }
    if critic_level_of(level, critic_level) is not None:
        layout[CRITIC_PLAYERS] = (int(encoder.size), tuple(layout["players"][1]))
    return layout


def row_size(layout: Mapping[str, Any]) -> int:
    return max(offset + _numel(shape) for offset, shape in layout.values())


def _numel(shape: Sequence[int]) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


# ---------------------------------------------------------------- dims


def generator_bounds(map_config: Mapping[str, Any]) -> tuple[int, int, int]:
    """(land tiles, land nodes, land edges) no map of this generator exceeds.

    Fixed shapes (Catanatron templates, hexagons, rectangles, the extension
    board, explicit tiles) do not depend on the seed, so one map gives their
    counts. Random blobs of ``T`` land tiles are bounded by the worst case of
    a connected polyhex, a chain: ``4T + 2`` nodes and ``5T + 1`` edges.
    """
    kind = str(map_config.get("type", "BASE"))
    if kind == "blob":
        land = map_config["land_tiles"]
        tiles = max(int(x) for x in land) if isinstance(land, (list, tuple)) else int(land)
        return tiles, 4 * tiles + 2, 5 * tiles + 1
    catan_map = cpp.CatanMap.from_spec(cpp.MapSpec.from_dict(maps.generate(map_config, 0)))
    dims = cpp.EncoderDims.covering([catan_map], 2)
    return int(dims.max_tiles), int(dims.max_nodes), int(dims.max_edges)


def _game_of(scenario: Any) -> games.GameConfig:
    if isinstance(scenario, games.GameConfig):
        return scenario
    if isinstance(scenario, Mapping) and "game" in scenario:
        return _game_of(scenario["game"])
    if isinstance(scenario, (str, Mapping)):
        return games.load_game(scenario)
    raise TypeError(f"cannot read a game from {scenario!r}")


def dims_for(scenarios: Iterable[Any], generator_bounds: bool = True) -> dict[str, int]:
    """The smallest dims covering every scenario.

    ``scenarios`` holds game configs, game names, or scenario dicts with a
    ``game`` key. With ``generator_bounds`` (``dims: auto``) the dims come
    from :func:`generator_bounds`, so they do not depend on any seed.
    Otherwise every scenario dict must list its maps (``maps``: map specs, or
    a dict holding them under ``specs``) and the dims cover exactly those.
    """
    scenarios = list(scenarios)
    if not scenarios:
        raise ValueError("dims_for needs at least one scenario")
    result = {key: 0 for key in DIM_KEYS}
    for scenario in scenarios:
        game = _game_of(scenario)
        result["max_players"] = max(result["max_players"], game.num_players)
        if generator_bounds:
            bounds = [_bounds_for(game.map)]
        else:
            specs = _scenario_maps(scenario)
            bounds = [_spec_counts(spec) for spec in specs]
        for tiles, nodes, edges in bounds:
            result["max_tiles"] = max(result["max_tiles"], tiles)
            result["max_nodes"] = max(result["max_nodes"], nodes)
            result["max_edges"] = max(result["max_edges"], edges)
    return result


_BOUNDS_CACHE: dict[str, tuple[int, int, int]] = {}


def _bounds_for(map_config: Mapping[str, Any]) -> tuple[int, int, int]:
    key = repr(sorted(map_config.items()))
    if key not in _BOUNDS_CACHE:
        _BOUNDS_CACHE[key] = generator_bounds(map_config)
    return _BOUNDS_CACHE[key]


def _scenario_maps(scenario: Any) -> list[dict]:
    if not isinstance(scenario, Mapping) or "maps" not in scenario:
        raise ValueError("dims_for(generator_bounds=False) needs every scenario's maps")
    pool = scenario["maps"]
    if isinstance(pool, Mapping):
        pool = pool["specs"]
    return list(pool)


def _spec_counts(spec: Mapping[str, Any]) -> tuple[int, int, int]:
    catan_map = cpp.CatanMap.from_spec(cpp.MapSpec.from_dict(dict(spec)))
    dims = cpp.EncoderDims.covering([catan_map], 2)
    return int(dims.max_tiles), int(dims.max_nodes), int(dims.max_edges)


def covering_dims(catan_maps: Sequence[Any], max_players: int) -> dict[str, int]:
    """Exact dims for built ``cpp.CatanMap`` objects."""
    return check_dims(cpp.EncoderDims.covering(list(catan_maps), int(max_players)))
