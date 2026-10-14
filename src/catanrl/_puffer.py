"""The only importer of ``pufferlib.pufferl``.

Importing ``pufferl`` turns ``RuntimeWarning`` into errors and installs an
``os._exit(0)`` SIGINT handler. :func:`import_pufferl` imports it inside
``warnings.catch_warnings()`` and puts the previous handlers back;
:func:`install_signal_handlers` installs catanrl's (the CLI does so for
every command): the first SIGINT/SIGTERM asks a trainer to stop at its next
epoch, a second one (or one with no trainer) raises ``KeyboardInterrupt``.
"""

from __future__ import annotations

import signal
import sys
import warnings
from collections.abc import Callable
from types import ModuleType

_on_stop: Callable[[str], None] | None = None
_stop_requested = False


def import_pufferl() -> ModuleType:
    if "pufferlib.pufferl" in sys.modules:
        return sys.modules["pufferlib.pufferl"]
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    hook = sys.excepthook
    with warnings.catch_warnings():
        from pufferlib import pufferl
    for sig, handler in handlers.items():
        signal.signal(sig, handler)
    sys.excepthook = hook
    return pufferl


def _handler(signum, frame) -> None:
    global _stop_requested
    if _on_stop is not None and not _stop_requested:
        _stop_requested = True
        _on_stop(signal.Signals(signum).name)
        return
    raise KeyboardInterrupt(signal.Signals(signum).name)


def install_signal_handlers(on_stop: Callable[[str], None] | None = None) -> None:
    """SIGINT and SIGTERM call ``on_stop`` once, then raise ``KeyboardInterrupt``."""
    global _on_stop, _stop_requested
    _on_stop, _stop_requested = on_stop, False
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
