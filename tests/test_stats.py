import math

import pytest

from catanrl.eval.stats import _paired_comparison, binomial_two_sided, mcnemar_exact, wilson_interval


def test_wilson_matches_known_values():
    low, high = wilson_interval(896, 1000)
    assert math.isclose(low, 0.8758, abs_tol=5e-4)
    assert math.isclose(high, 0.9133, abs_tol=5e-4)
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_exact_tests():
    assert math.isclose(binomial_two_sided(5, 10), 1.0)
    assert math.isclose(binomial_two_sided(0, 10), 2 / 1024)
    assert math.isclose(mcnemar_exact(171, 136), binomial_two_sided(171, 307))


@pytest.mark.parametrize(("k", "n", "p"), [(540, 1030, 0.5), (2600, 5000, 0.5), (1210, 5000, 0.25), (0, 5000, 0.5),
                                           (5000, 5000, 0.5), (2500, 5000, 0.5)])
def test_binomial_matches_scipy_at_large_n(k, n, p):
    scipy_stats = pytest.importorskip("scipy.stats")
    expected = scipy_stats.binomtest(k, n, p).pvalue
    found = binomial_two_sided(k, n, p)
    assert math.isclose(found, expected, rel_tol=1e-6, abs_tol=1e-300)


def test_paired_comparison():
    result = _paired_comparison([True, True, False, False], [True, False, False, True])
    assert result["only_a"] == 1 and result["only_b"] == 1 and result["delta"] == 0
