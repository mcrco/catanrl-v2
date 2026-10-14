"""Test 17: every experiment under configs/experiments resolves and validates."""

from __future__ import annotations

from pathlib import Path

import pytest

from catanrl.config import load_plan, validate_plan

EXPERIMENTS = sorted((Path(__file__).resolve().parents[1] / "configs" / "experiments").glob("*.yaml"))


@pytest.mark.parametrize("path", EXPERIMENTS, ids=[p.stem for p in EXPERIMENTS])
def test_experiment_validates(path):
    assert validate_plan(path) == []
    plan = load_plan(path, smoke=True)
    assert plan.variants and all(v.config["runtime"]["device"] == "cpu" for v in plan.variants)


def test_arch_lineage_shape():
    plan = load_plan("arch-lineage")
    assert len(plan.variants) == 18
    g1 = plan.variant("graph-m-s43.g1")
    assert g1.depends_on == ("graph-m-s43.warm",) and g1.config["train"]["gamma"] == 1.0
    assert g1.config["init_from"]["ckpt"] == "arch-lineage/graph-m-s43.warm@3000"
