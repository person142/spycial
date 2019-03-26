"""Implementation of the digamma function. The original code is from
Cephes:

Cephes Math Library Release 2.8:  June, 2000
Copyright 1984, 1987, 1992, 2000 by Stephen L. Moshier

Code for the rational approximation on [1, 2] is from Boost, which is:

(C) Copyright John Maddock 2006.
Use, modification and distribution are subject to the
Boost Software License, Version 1.0. (See accompanying file
LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

"""
from numba import njit, vectorize
import numpy as np

from . import settings
from .constants import _π
from .evalpoly import _devalpoly

# Harmonic numbers minus the Euler-Mascheroni constant
HARMONIC = np.array([
    -0.5772156649015328606065121,
    0.4227843350984671393934879,
    0.9227843350984671393934879,
    1.256117668431800472726821,
    1.506117668431800472726821,
    1.706117668431800472726821,
    1.872784335098467139393488,
    2.015641477955609996536345,
    2.140641477955609996536345,
    2.251752589066721107647456
])

ASYMP = np.array([
    8.33333333333333333333E-2,
    -2.10927960927960927961E-2,
    7.57575757575757575758E-3,
    -4.16666666666666666667E-3,
    3.96825396825396825397E-3,
    -8.33333333333333333333E-3,
    8.33333333333333333333E-2
])

RAT_NUM = np.array([
    -0.0020713321167745952,
    -0.045251321448739056,
    -0.28919126444774784,
    -0.65031853770896507,
    -0.32555031186804491,
    0.25479851061131551
])

RAT_DENOM = np.array([
    -0.55789841321675513e-6,
    0.0021284987017821144,
    0.054151797245674225,
    0.43593529692665969,
    1.4606242909763515,
    2.0767117023730469,
    1.0
])


@njit('float64(float64)', cache=settings.CACHE)
def _digamma_rational(x):
    """Rational approximation on [1, 2] taken from Boost.

    For the approximation, we use the form

    digamma(x) = (x - root) * (Y + R(x-1))

    where root is the location of the positive root of digamma, Y is a
    constant, and R is optimised for low absolute error compared to Y.

    Maximum deviation found: 1.466e-18
    At double precision, max error found: 2.452e-17

    """
    Y = np.float32(0.99558162689208984)
    root1 = 1569415565.0/1073741824.0
    root2 = (381566830.0/1073741824.0)/1073741824.0
    root3 = 0.9016312093258695918615325266959189453125e-19

    g = x - root1
    g -= root2
    g -= root3
    r = _devalpoly(RAT_NUM, x - 1.0)/_devalpoly(RAT_DENOM, x - 1.0)

    return g*Y + g*r


@njit('float64(float64)', cache=settings.CACHE)
def _digamma(x):
    res = 0.0

    if np.isnan(x) or x == np.inf:
        return x
    elif x == -np.inf:
        return np.nan
    elif x == 0:
        return np.copysign(np.inf, -x)
    elif x < 0.0:
	# Argument reduction before evaluating tan(πx).
        r = np.fmod(x, 1.0)
        if r == 0.0:
            return np.nan
        πr = _π*r
        # Reflection formula
        res = -_π*np.cos(πr)/np.sin(πr) - 1.0/x
        x = -x

    if x <= 10.0:
        if x == np.floor(x):
            # Exact values for for positive integers up to 10
            res += HARMONIC[np.intc(x)-1]
            return res
        # Use the recurrence relation to move x into [1, 2]
        if x < 1.0:
            res -= 1.0/x
            x += 1.0
        elif x < 10.0:
            while x > 2.0:
                x -= 1.0
                res += 1.0/x

        res += _digamma_rational(x)
        return res

    # We know x is large, use the asymptotic series.
    if x < 1.0e17:
        z = 1.0/(x*x)
        y = z*_devalpoly(ASYMP, z)
    else:
        y = 0.0
    res += np.log(x) - (0.5/x) - y
    return res


@vectorize(['float64(float64)'], nopython=True, cache=settings.CACHE)
def digamma(x):
    """Digamma function.

    Parameters
    ----------
    x : array-like
        Points on the real line
    out : ndarray, optional
        Output array for the values of `digamma` at `x`

    Returns
    -------
    ndarray
        Values of `digamma` at `x`

    """
    return _digamma(x)
