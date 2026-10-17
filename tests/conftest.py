"""Every test gets its own runs directory unless it is marked ``runs``
(those check the real one), and no test reaches wandb."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _private_runs_dir(request, tmp_path_factory, monkeypatch):
    if request.node.get_closest_marker("runs") is None:
        monkeypatch.setenv("CATANRL_RUNS_DIR", str(tmp_path_factory.mktemp("runs")))


@pytest.fixture(autouse=True)
def _no_wandb(monkeypatch):
    """A pytest session never creates wandb runs, whatever the shell exports: the mirror
    (catanrl.wandb_mirror) is off under WANDB_MODE=disabled, and the switches that turn it
    on are cleared. tests/test_wandb_mirror.py turns it back on against a stub module."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    for name in ("CATANRL_WANDB_PROJECT", "CATANRL_WANDB_ENTITY", "CATANRL_WANDB_SMOKE"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _allow_stale_engine_builds(monkeypatch):
    """Tests do not depend on the engine build matching its checkout (test_provenance checks the refusal)."""
    monkeypatch.setenv("CATANRL_ALLOW_STALE_BUILD", "1")
