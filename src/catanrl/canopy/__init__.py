"""Released Canopy (cullback/canopy's Catan AlphaZero) as an eval agent: ``CANOPY:sims=1600``.

Outside ``catanrl.agents`` on purpose: :func:`catanrl.provenance.code_id` hashes this package
only for records with a CANOPY agent in their lineup (its ``canopy`` component), so adding or
changing it leaves every other record's key as it was.

* :mod:`catanrl.canopy.release`: the pinned build (commit, patches, binary, checkpoint);
* :mod:`catanrl.canopy.topology`: Canopy's board and a cppanatron map in its ids;
* :mod:`catanrl.canopy.snapshot`: a decision as the bridge's snapshot, from the seat's view;
* :mod:`catanrl.canopy.bridge`: the bridge processes;
* :mod:`catanrl.canopy.agent`: the spec, its resolution and the native agent.

:func:`resolve` resolves any lineup spec (CANOPY here, every other key through
``catanrl.agents.resolve_agent``); eval records and the probe tool resolve through it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

CANOPY_KEY = "CANOPY"


def is_canopy_spec(spec: Any) -> bool:
    """Whether ``spec`` (a string, a ``{"key": ...}`` dict or a resolved agent) is a CANOPY spec."""
    if isinstance(spec, Mapping):
        return str(spec.get("key", "")).upper() == CANOPY_KEY
    key = getattr(spec, "key", None)
    if key is not None and not isinstance(spec, str):
        return key == CANOPY_KEY
    return isinstance(spec, str) and spec.split(":", 1)[0].strip().upper() == CANOPY_KEY


def resolve(spec: Any, *, harness: str = "native", games: Sequence[Any] = ()):
    """``catanrl.agents.resolve_agent``, plus the CANOPY key."""
    from catanrl.agents import ResolvedAgent, resolve_agent

    if isinstance(spec, ResolvedAgent):
        spec = spec.canonical
    if is_canopy_spec(spec):
        from catanrl.canopy.agent import resolve_canopy

        return resolve_canopy(spec, harness=harness, games=games)
    return resolve_agent(spec, harness=harness, games=games)


def lineup_has_canopy(entries: Sequence[Any]) -> bool:
    """Whether any of ``entries`` (specs, resolved agents or key dicts) is a CANOPY agent."""
    return any(entry is not None and is_canopy_spec(entry) for entry in entries)
