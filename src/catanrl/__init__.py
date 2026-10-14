"""Catan bots for any map and player count.

Importing catanrl sets ``CUBLAS_WORKSPACE_CONFIG`` (strict determinism needs
it before any CUDA use). The everyday entry points are re-exported lazily::

    run = catanrl.load_experiment("arch-lineage/graph-s-s42.g1")
    net = run.build_policy("best")
    net = catanrl.load_checkpoint("arch-lineage/graph-s-s42.g1@e600").model()
    bot = catanrl.player("POLICY:arch-lineage/graph-s-s42.g1@best", Color.RED)
"""

import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
#: When catanrl was first imported (the stale-code guard's fallback for the process start).
IMPORT_TIME = time.time()

_EXPORTS = {
    "load_experiment": "catanrl.runs",
    "load_checkpoint": "catanrl.runs",
    "list_runs": "catanrl.runs",
    "resolve_agent": "catanrl.agents",
    "player": "catanrl.agents",
    "evaluate": "catanrl.eval.results",
    "load_suite": "catanrl.eval.suites",
}

__all__ = ["evaluate", "list_runs", "load_checkpoint", "load_experiment", "load_suite", "player",
           "resolve_agent"]


def __getattr__(name):
    if name in _EXPORTS:
        import importlib

        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'catanrl' has no attribute {name!r}")
