"""Evaluate polynomials and rational functions.

All of the coefficients polynomials are stored in reverse order,
i.e. if the polynomial is

    u_n x^n + u_{n - 1} x^{n - 1} + ... + u_0,

then coeffs[0] = u_n, coeffs[1] = u_{n - 1}, ..., coeffs[n] = u_0.

References
----------
[1] Knuth, "The Art of Computer Programming, Volume II"
[2] Holin et al., "Polynomial and Rational Function Evaluation",
    http://www.boost.org/doc/libs/1_61_0/libs/math/doc/html/math_toolkit/roots/rational.html

"""
import numpy as np
from numba import njit
from numba.types import complex128, float64, intc, Array

from .fma import _fma


@njit(float64(Array(float64, 1, "C", readonly=True), float64))
def _devalpoly(coeffs, x):
    """Evaluate a polynomial using Horner's method."""
    res = coeffs[0]

    for j in range(1, len(coeffs)):
        res = _fma(res, x, coeffs[j])

    return res


@njit(complex128(Array(float64, 1, "C", readonly=True), complex128))
def _cevalpoly(coeffs, z):
    """Evaluate a polynomial with real coefficients at a complex point.

    Uses equation (3) in section 4.6.4 of [1]. Note that it is more
    efficient than Horner's method.

    """
    a = coeffs[0]
    b = coeffs[1]
    r = 2*z.real
    s = z.real*z.real + z.imag*z.imag

    for j in range(2, len(coeffs)):
        tmp = b
        b = _fma(-s, a, coeffs[j])
        a = _fma(r, a, tmp)
    return z*a + b


@njit(float64(Array(float64, 1, "C", readonly=True),
              Array(float64, 1, "C", readonly=True), float64))
def _devalrational(coeffs_num, coeffs_denom, x):
    """Evaluate a polynomial with real coefficients at a real point.

    Uses the method outlined in [2].

    """
    if np.fabs(x) > 1:
        y = 1.0/x

        num = coeffs_num[-1]
        for j in range(len(coeffs_num) - 2, -1, -1):
            num = _fma(num, y, coeffs_num[j])

        denom = coeffs_denom[-1]
        for j in range(len(coeffs_denom) - 2, -1, -1):
            denom = _fma(denom, y, coeffs_denom[j])

        return x**(len(coeffs_denom) - len(coeffs_num))*num/denom
    else:
        return _devalpoly(coeffs_num, x)/_devalpoly(coeffs_denom, x)
