"""Every test gets its own runs directory unless it is marked ``runs``
(those check the real one)."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _private_runs_dir(request, tmp_path_factory, monkeypatch):
    if request.node.get_closest_marker("runs") is None:
        monkeypatch.setenv("CATANRL_RUNS_DIR", str(tmp_path_factory.mktemp("runs")))
