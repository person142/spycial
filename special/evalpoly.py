"""Evaluate polynomials.

All of the coefficients are stored in reverse order, i.e. if the
polynomial is

    u_n x^n + u_{n - 1} x^{n - 1} + ... + u_0,

then coeffs[0] = u_n, coeffs[1] = u_{n - 1}, ..., coeffs[n] = u_0.

References
----------
[1] Knuth, "The Art of Computer Programming, Volume II"

"""
from numba import jit

from .fma import _fma


@jit('complex128(float64[:], intc, complex128)', nopython=True)
def _cevalpoly(coeffs, degree, z):
    """Evaluate a polynomial with real coefficients at a complex point.

    Uses equation (3) in section 4.6.4 of [1]. Note that it is more
    efficient than Horner's method.

    """
    a = coeffs[0]
    b = coeffs[1]
    r = 2*z.real
    s = z.real*z.real + z.imag*z.imag

    for j in range(2, degree + 1):
        tmp = b
        b = _fma(-s, a, coeffs[j])
        a = _fma(r, a, tmp)
    return z*a + b
