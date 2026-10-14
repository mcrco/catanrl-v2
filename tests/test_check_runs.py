"""Test 29 (-m runs): every alias and latest checkpoint of every real run
loads, verifies its weights and passes its self-test."""

from __future__ import annotations

import pytest

from catanrl.runs import check_runs

pytestmark = pytest.mark.runs


def test_every_run_checks():
    bad = [row for row in check_runs() if not row["ok"]]
    assert not bad, "\n".join(f"{r['run']}@{r['which']}: {r['error']}" for r in bad)
