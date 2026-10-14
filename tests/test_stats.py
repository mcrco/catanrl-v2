import math

from catanrl.eval.stats import binomial_two_sided, mcnemar_exact, paired_comparison, wilson_interval


def test_wilson_matches_known_values():
    low, high = wilson_interval(896, 1000)
    assert math.isclose(low, 0.8758, abs_tol=5e-4)
    assert math.isclose(high, 0.9133, abs_tol=5e-4)
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_exact_tests():
    assert math.isclose(binomial_two_sided(5, 10), 1.0)
    assert math.isclose(binomial_two_sided(0, 10), 2 / 1024)
    assert math.isclose(mcnemar_exact(171, 136), binomial_two_sided(171, 307))


def test_paired_comparison():
    result = paired_comparison([True, True, False, False], [True, False, False, True])
    assert result["only_a"] == 1 and result["only_b"] == 1 and result["delta"] == 0
