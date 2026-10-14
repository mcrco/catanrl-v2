"""Test 27 (-m train): ``--smoke`` runs the first sweep cell of every training
experiment, every stage in order, on CPU (EXP+smoke runs in a temp runs dir)."""

from __future__ import annotations

from pathlib import Path

import pytest

from catanrl.config import load_plan
from catanrl.runner import run_variant

pytestmark = pytest.mark.train
EXPERIMENTS = sorted((Path(__file__).resolve().parents[1] / "configs" / "experiments").glob("*.yaml"))


@pytest.mark.parametrize("path", EXPERIMENTS, ids=[p.stem for p in EXPERIMENTS])
def test_smoke(path):
    plan = load_plan(path, smoke=True)
    if plan.variants[0].kind == "eval":
        pytest.skip("eval-only experiments need their referenced runs")
    cell = plan.variants[0].labels
    for v in [v for v in plan.variants if v.labels == cell]:
        assert run_variant(v, argv=["smoke"]) == "finished", v.ref
