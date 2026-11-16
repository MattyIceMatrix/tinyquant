"""Integer-only requantization arithmetic.

An INT8 layer accumulates into int32, then has to come back down to int8. The
scale factor for that is a real number (s_x * s_w / s_y), but an edge target
may have no FPU, and even where it does, floating-point rounding makes
bit-exact reproducibility across compilers awkward. So the multiplier is stored
as a Q0.31 fixed-point integer plus a shift, and applied with integer ops only.

These functions mirror gemmlowp/TFLite semantics so results are comparable with
the reference implementations people know -- and, more to the point, so the
Python here and the C in engine/ produce identical bytes. Every routine is
written to behave the same in both languages: no numpy dtype promotion, no
reliance on C undefined behaviour.
"""
from __future__ import annotations

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1


def quantize_multiplier(m: float) -> tuple[int, int, int]:
    """Decompose a positive real multiplier into (q31_multiplier, left_shift, right_shift).

    m is written as q * 2**exp with q in [0.5, 1), then q is stored as a Q0.31
    integer. The exponent becomes a shift: positive exponents shift the
    accumulator up before the multiply, negative ones shift the product down
    after it.
    """
    if m <= 0.0:
        return 0, 0, 0
    exp = 0
    while m < 0.5:
        m *= 2.0
        exp -= 1
    while m >= 1.0:
        m /= 2.0
        exp += 1
    q = int(round(m * (1 << 31)))
    if q == (1 << 31):  # rounded up to exactly 1.0
        q //= 2
        exp += 1
    assert 0 < q <= INT32_MAX, q
    return q, max(exp, 0), max(-exp, 0)


def saturating_rounding_doubling_high_mul(a: int, b: int) -> int:
    """Round((a * b) / 2**31), saturated to int32. Both inputs are int32."""
    if a == INT32_MIN and b == INT32_MIN:
        return INT32_MAX
    ab = a * b
    nudge = (1 << 30) if ab >= 0 else (1 - (1 << 30))
    out = (ab + nudge) >> 31
    return max(INT32_MIN, min(INT32_MAX, out))


def rounding_divide_by_pot(x: int, exponent: int) -> int:
    """Divide by 2**exponent, rounding half away from zero."""
    if exponent == 0:
        return x
    mask = (1 << exponent) - 1
    remainder = x & mask
    threshold = (mask >> 1) + (1 if x < 0 else 0)
    return (x >> exponent) + (1 if remainder > threshold else 0)


def requantize(acc: int, q_mult: int, left_shift: int, right_shift: int) -> int:
    """Apply the fixed-point multiplier to an int32 accumulator."""
    shifted = acc * (1 << left_shift)
    shifted = max(INT32_MIN, min(INT32_MAX, shifted))
    return rounding_divide_by_pot(
        saturating_rounding_doubling_high_mul(shifted, q_mult), right_shift
    )
