import pytest

from scieval.scoring.parity import parity_check


def test_within_3pp():
    v = parity_check(observed=0.62, n=10_000, published=0.60)
    assert v.ok and v.diff == pytest.approx(0.02)
    assert v.tolerance == pytest.approx(0.03)


def test_wide_ci_small_n():
    # n=198 (GPQA-D): CI가 3%p보다 넓다
    v = parity_check(observed=0.60, n=198, published=0.55)
    assert v.tolerance > 0.03
    assert v.ok


def test_fail():
    assert not parity_check(observed=0.75, n=10_000, published=0.60).ok
