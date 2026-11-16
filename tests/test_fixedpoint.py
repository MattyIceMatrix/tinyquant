"""The requantization arithmetic is the part most likely to be subtly wrong,
so it gets tested directly rather than only through end-to-end accuracy."""
import math

import pytest

from tinyquant.fixedpoint import (INT32_MAX, INT32_MIN, quantize_multiplier,
                                  requantize, rounding_divide_by_pot,
                                  saturating_rounding_doubling_high_mul)


@pytest.mark.parametrize("m", [1e-6, 3.1e-4, 0.0125, 0.5, 0.9999, 1.0, 1.75, 3.0, 100.0])
def test_multiplier_decomposition_reconstructs_value(m):
    q, ls, rs = quantize_multiplier(m)
    reconstructed = (q / 2**31) * 2 ** (ls - rs)
    assert math.isclose(reconstructed, m, rel_tol=1e-9)


@pytest.mark.parametrize("m", [3.1e-4, 0.0125, 0.5, 0.9999])
@pytest.mark.parametrize("acc", [-99999, -1024, -33, -1, 0, 1, 33, 1024, 99999])
def test_requantize_is_within_two_lsb_of_exact(m, acc):
    q, ls, rs = requantize, *quantize_multiplier(m)[1:]
    qm, ls, rs = quantize_multiplier(m)
    got = requantize(acc, qm, ls, rs)
    assert abs(got - acc * m) <= 2.0


def test_multiplier_of_zero_is_zero():
    assert quantize_multiplier(0.0) == (0, 0, 0)


def test_doubling_high_mul_saturates_at_int32_min():
    assert saturating_rounding_doubling_high_mul(INT32_MIN, INT32_MIN) == INT32_MAX


def test_rounding_divide_rounds_half_away_from_zero():
    assert rounding_divide_by_pot(3, 1) == 2     # 1.5 -> 2
    assert rounding_divide_by_pot(-3, 1) == -2   # -1.5 -> -2
    assert rounding_divide_by_pot(5, 0) == 5


def test_requantize_saturates_rather_than_overflowing():
    qm, ls, rs = quantize_multiplier(3.0)
    assert requantize(INT32_MAX, qm, ls, rs) <= INT32_MAX
    assert requantize(INT32_MIN + 1, qm, ls, rs) >= INT32_MIN
