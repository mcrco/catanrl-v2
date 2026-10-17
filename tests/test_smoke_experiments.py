"""Test 27 (-m train): ``--smoke`` runs the first sweep cell of every training
experiment, every stage in order, on CPU (EXP+smoke runs in a temp runs dir).
A final eval naming another experiment's run (a control trained elsewhere,
not in the smoke store) is deferred, and nothing is left for a later run."""

from __future__ import annotations

from pathlib import Path

import pytest

from catanrl.config import load_plan
from catanrl.runner import _missing_finals, run_variant
from catanrl.runs import load_experiment

pytestmark = pytest.mark.train
EXPERIMENTS = sorted((Path(__file__).resolve().parents[1] / "configs" / "experiments").glob("*.yaml"))


@pytest.mark.parametrize("path", EXPERIMENTS, ids=[p.stem for p in EXPERIMENTS])
def test_smoke(path):
    plan = load_plan(path, smoke=True)
    if plan.variants[0].kind == "eval":
        pytest.skip("eval-only experiments need their referenced runs")
    init = plan.variants[0].config.get("init_from") or {}
    refs = init.get("ckpt", {}).values() if isinstance(init.get("ckpt"), dict) else [init.get("ckpt")]
    if any(ref and not str(ref).startswith(f"{plan.experiment}/") for ref in refs):
        pytest.skip("starts from another experiment's run")
    cell = plan.variants[0].labels
    for v in [v for v in plan.variants if v.labels == cell]:
        assert run_variant(v, argv=["smoke"]) == "finished", v.ref
        assert not _missing_finals(load_experiment(v.ref), v), v.ref  # played, or deferred for a control elsewhere
