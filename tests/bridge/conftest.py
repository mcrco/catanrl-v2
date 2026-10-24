"""The bridge tests run tiny networks one row at a time: one torch thread is
fastest, and keeps them quick on a loaded machine. Their Catanatron truth plays
the engine's (OpenHex's) friendly robber, for each bridge test only."""

from __future__ import annotations

import pytest
import torch
from bridge_games import openhex_friendly_robber


@pytest.fixture(autouse=True)
def _openhex_friendly_robber(monkeypatch):
    """Stock Catanatron re-ruled as OpenHex's friendly robber (bridge_games), undone after
    the test, so nothing else in the session plays the swapped rule. A server whose
    Catanatron plays the friendly robber (the house, if its 1v1 table turned it on)
    needs the same rule, or the replay refuses the moves its Catanatron allows."""
    openhex_friendly_robber(monkeypatch)


@pytest.fixture(autouse=True)
def _one_torch_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)
