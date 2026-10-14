"""Play native agents in games where they see one seat's view only.

A catanatron-server room (protocol 4) or the Catanatron TUI hands a bot its
seat's ``client_view`` and the actions it may play. The bridge turns that into
what the native agents read, using nothing but that seat's information:

* :mod:`~catanrl.bridge.position`: the view's map, rules and seating as a
  cppanatron game, the public-state comparison, and the view's exact
  position declared through ``apply_setup`` (at any prompt);
* :mod:`~catanrl.bridge.replay`: the view's action log replayed on
  cppanatron, with placeholder hidden cards (re-dealt when a public move
  needs them);
* :mod:`~catanrl.bridge.tracking`: the seat's ``HandTracker``, fed the
  public events of every replayed transition, re-anchored on the declared
  position wherever the replay cannot explain a view;
* :mod:`~catanrl.bridge.agent`: :class:`ViewAgent`, a
  ``decide(view, playable_actions)`` callback for ``GameClient.play`` and
  TUI ``decide_fns``;
* :mod:`~catanrl.bridge.server_bot`: :func:`run_bot` (a server seat with a
  verified transcript) and :func:`tui_decide_fns`.
"""

from __future__ import annotations

_EXPORTS = {
    "ViewAgent": "catanrl.bridge.agent",
    "SeatTracker": "catanrl.bridge.tracking",
    "run_bot": "catanrl.bridge.server_bot",
    "play_bot": "catanrl.bridge.server_bot",
    "tui_decide_fns": "catanrl.bridge.server_bot",
    "play_tui": "catanrl.bridge.server_bot",
}

__all__ = ["SeatTracker", "ViewAgent", "play_bot", "play_tui", "run_bot", "tui_decide_fns"]


def __getattr__(name):
    if name in _EXPORTS:
        import importlib

        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'catanrl.bridge' has no attribute {name!r}")
