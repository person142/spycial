"""Code for real gamma adapted from Boost, which is:

Copyright John Maddock 2006-7, 2013-14.
Copyright Paul A. Bristow 2007, 2013-14.
Copyright Nikhar Agrawal 2013-14
Copyright Christopher Kormanyos 2013-14

Use, modification and distribution are subject to the
Boost Software License, Version 1.0. (See accompanying file
LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

"""
from numba import njit, vectorize
import numpy as np

from . import settings
from .constants import _π, _root_ε, _γ, _MAXEXP
from .trig import _dsinpi
from .lanczos import _lanczos_g, _lanczos_sum

FACTORIALS = np.array([
    1.0,
    1.0,
    2.0,
    6.0,
    24.0,
    120.0,
    720.0,
    5040.0,
    40320.0,
    362880.0
])


@njit('float64(float64)', cache=settings.CACHE)
def _dgamma(x):
    res = 1.0

    if x <= 0.0:
        if x == np.floor(x):
            return np.nan
        if x <= -20.0:
            return _π/(_dgamma(-x)*_dsinpi(x)*x);

        while x < 0:
            res /= x;
            x += 1;

    if x <= 10 and x == np.floor(x):
        res *= FACTORIALS[np.intc(x)-1]
    elif x < _root_ε:
        res *= 1.0/x - _γ
    else:
        res *= _lanczos_sum(x)
        xgh = x + _lanczos_g - 0.5
        log_xgh = np.log(xgh)
        xlog_xgh = x*log_xgh
        if xlog_xgh > _MAXEXP:
            if 0.5*xlog_xgh > _MAXEXP:
                return np.copysign(np.inf, res)
            hp = xgh**(0.5*x - 0.25)
            res *= hp/np.exp(xgh)
            res *= hp
        else:
            res *= xgh**(x - 0.5)/np.exp(xgh)

    return res


@vectorize(['float64(float64)'], nopython=True, cache=settings.CACHE)
def gamma(x):
    """The Gamma function

    Parameters
    ----------
    x : array-like
        Points on the real line.
    out : ndarray, optional
        Output array for the values of `gamma` at `x`

    Returns
    -------
    ndarray
        Values of `gamma` at `x`

    """
    return _dgamma(x)
