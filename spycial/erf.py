"""Code adapted from Boost, which is:

(C) Copyright John Maddock 2006.
Use, modification and distribution are subject to the
Boost Software License, Version 1.0. (See accompanying file
LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

"""
from numba import njit, vectorize
import numpy as np

from . import settings
from .evalpoly import _devalpoly

P1 = np.array([
    -0.000322780120964605683831,
    -0.00772758345802133288487,
    -0.0509990735146777432841,
    -0.338165134459360935041,
    0.0834305892146531832907
])

Q1 = np.array([
    0.000370900071787748000569,
    0.00858571925074406212772,
    0.0875222600142252549554,
    0.455004033050794024546,
    1.0
])

P2 = np.array([
    0.00180424538297014223957,
    0.0195049001251218801359,
    0.0888900368967884466578,
    0.191003695796775433986,
    0.178114665841120341155,
    -0.098090592216281240205
])

Q2 = np.array([
    0.337511472483094676155e-5,
    0.0113385233577001411017,
    0.12385097467900864233,
    0.578052804889902404909,
    1.42628004845511324508,
    1.84759070983002217845,
    1.0
])

P3 = np.array([
    0.000235839115596880717416,
    0.00323962406290842133584,
    0.0175679436311802092299,
    0.04394818964209516296,
    0.0386540375035707201728,
    -0.0243500476207698441272
])

Q3 = np.array([
    0.00410369723978904575884,
    0.0563921837420478160373,
    0.325732924782444448493,
    0.982403709157920235114,
    1.53991494948552447182,
    1.0
])

P4 = np.array([
    0.113212406648847561139e-4,
    0.000250269961544794627958,
    0.00212825620914618649141,
    0.00840807615555585383007,
    0.0137384425896355332126,
    0.00295276716530971662634
])

Q4 = np.array([
    0.000479411269521714493907,
    0.0105982906484876531489,
    0.0958492726301061423444,
    0.442597659481563127003,
    1.04217814166938418171,
    1.0
])

P5 = np.array([
    -2.8175401114513378771,
    -3.22729451764143718517,
    -2.5518551727311523996,
    -0.687717681153649930619,
    -0.212652252872804219852,
    0.0175389834052493308818,
    0.00628057170626964891937
])

Q5 = np.array([
    5.48409182238641741584,
    13.5064170191802889145,
    22.9367376522880577224,
    15.930646027911794143,
    11.0567237927800161565,
    2.79257750980575282228,
    1.0
])


@njit('float64(float64, bool_)', cache=settings.CACHE)
def _erf_erfc(x, invert):
    """Compute erf if invert is False and erfc if invert is True."""
    if x < 0:
        if not invert:
            return -_erf_erfc(-x, False)
        elif x < -0.5:
            return 2.0 - _erf_erfc(-x, True);
        else:
            return 1.0 + _erf_erfc(-x, False)

    if x < 0.5:
        # We're going to calculate erf
        if x < 1e-10:
            # Single term of the Taylor series
            res = 1.128379167095512573896159*x
        else:
            # - Maximum deviation found: 1.561e-17
            # - Expected error term: 1.561e-17
            # - Maximum relative change in control points: 1.155e-04
            # - Max error found at double precision: 2.961182e-17
            Y = np.float32(1.044948577880859375)
            xx = x*x
            res = x*(Y + _devalpoly(P1, xx)/_devalpoly(Q1, xx))
    elif (invert and x < 28) or (not invert and x < 5.8):
        # We'll be calculating erfc:
        invert = not invert

        if x < 1.5:
            # Maximum deviation found: 3.702e-17
            # Expected error term: 3.702e-17
            # Maximum relative change in control points: 2.845e-04
            # Max error found at double precision: 4.841816e-17
            Y = np.float32(0.405935764312744140625)
            res = Y + _devalpoly(P2, x - 0.5)/_devalpoly(Q2, x - 0.5)
            res *= np.exp(-x*x)/x
        elif x < 2.5:
            # Maximum deviation found: 3.909e-18
            # Expected error term: 3.909e-18
            # Maximum relative change in control points: 9.886e-05
            # Max error found at double precision: 6.599585e-18
            Y = np.float32(0.50672817230224609375)
            res = Y + _devalpoly(P3, x - 1.5)/_devalpoly(Q3, x - 1.5)
            res *= np.exp(-x*x)/x
        elif x < 4.5:
            # Maximum deviation found: 1.512e-17
            # Expected error term: 1.512e-17
            # Maximum relative change in control points: 2.222e-04
            # Max error found at double precision: 2.062515e-17
            Y = np.float32(0.5405750274658203125)
            res = Y + _devalpoly(P4, x - 3.5)/_devalpoly(Q4, x - 3.5)
            res *= np.exp(-x*x)/x
        else:
            # Maximum deviation found: 2.860e-17
            # Expected error term: 2.859e-17
            # Maximum relative change in control points: 1.357e-05
            # Max error found at double precision: 2.997958e-17
            Y = np.float32(0.5579090118408203125)
            res = Y + _devalpoly(P5, 1.0/x)/_devalpoly(Q5, 1.0/x)
            res *= np.exp(-x*x)/x
    else:
        # Any value of x larger than 28 will underflow to zero
        result = 0.0
        invert = not invert

    if invert:
        res = 1.0 - res
    return res


@njit('float64(float64)', cache=settings.CACHE)
def _erf(x):
    return _erf_erfc(x, False)


@njit('float64(float64)', cache=settings.CACHE)
def _erfc(x):
    return _erf_erfc(x, True)


@vectorize(['float64(float64)'], nopython=True, cache=settings.CACHE)
def erf(x):
    """Error function.

    Parameters
    ----------
    x : array-like
        Points on the real line
    out : ndarray, optional
        Output array for the values of `erf` at `x`

    Returns
    -------
    ndarray
        Values of `erf` at `x`

    """
    return _erf(x)


@vectorize(['float64(float64)'], nopython=True, cache=settings.CACHE)
def erfc(x):
    """Complementary error function.

    Parameters
    ----------
    x : array-like
        Points on the real line
    out : ndarray, optional
        Output array for the values of `erfc` at `x`

    Returns
    -------
    ndarray
        Values of `erf` at `x`

    """
    return _erfc(x)
