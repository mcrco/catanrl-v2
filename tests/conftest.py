"""Every test gets its own runs directory unless it is marked ``runs``
(those check the real one)."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _private_runs_dir(request, tmp_path_factory, monkeypatch):
    if request.node.get_closest_marker("runs") is None:
        monkeypatch.setenv("CATANRL_RUNS_DIR", str(tmp_path_factory.mktemp("runs")))


@pytest.fixture(autouse=True)
def _allow_stale_engine_builds(monkeypatch):
    """Tests do not depend on the engine build matching its checkout (test_provenance checks the refusal)."""
    monkeypatch.setenv("CATANRL_ALLOW_STALE_BUILD", "1")
