"""Implementation ported from Boost, which is:

Copyright John Maddock 2007.
Use, modification and distribution are subject to the
Boost Software License, Version 1.0. (See accompanying file
LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

"""
from numba import njit, vectorize
import numpy as np

from . import settings
from .constants import MINEXP
from .evalpoly import _devalpoly

P_LT1 = np.array([
    -0.000111507792921197858394,
    -0.00399167106081113256961,
    -0.0368031736257943745142,
    -0.245088216639761496153,
    0.0320913665303559189999,
    0.0865197248079397976498,
])

Q_LT1 = np.array([
    -0.528611029520217142048e-6,
    0.000131049900798434683324,
    0.00427347600017103698101,
    0.056770677104207528384,
    0.37091387659397013215,
    1.0,
])

P_GT1 = np.array([
    -1185.45720315201027667,
    -14751.4895786128450662,
    -54844.4587226402067411,
    -86273.1567711649528784,
    -66598.2652345418633509,
    -27182.6254466733970467,
    -6046.8250112711035463,
    -724.581482791462469795,
    -43.3058660811817946037,
    -0.999999999999998811143,
    -0.121013190657725568138e-18,
])

Q_GT1 = np.array([
    -0.776491285282330997549,
    1229.20784182403048905,
    18455.4124737722049515,
    86722.3403467334749201,
    180329.498380501819718,
    192104.047790227984431,
    113057.05869159631492,
    38129.5594484818471461,
    7417.37624454689546708,
    809.193214954550328455,
    45.3058660811801465927,
    1.0,
])


@njit('float64(float64)', cache=settings.CACHE)
def _e1(x):
    if x < 0:
        return np.nan
    elif x == 0:
        return np.inf
    elif x <= 1:
        Y = np.float32(0.66373538970947265625)
        result = _devalpoly(P_LT1, x) / _devalpoly(Q_LT1, x)
        result += x - np.log(x) - Y
        return result
    elif x < -MINEXP:
        recip = 1 / x
        result = 1 + _devalpoly(P_GT1, recip) / _devalpoly(Q_GT1, recip)
        result *= np.exp(-x) * recip
        return result
    else:
        return 0


@vectorize(['float64(float64)'], nopython=True, cache=settings.CACHE)
def e1(x):
    r"""Exponential integral :math:`E_1(x)`.

    For real :math:`x > 0` the exponential integral can be defined as
    [1]_

    .. math::

        E_1(x) = \int_x^\infty \frac{e^{-t}}{t} dt.

    Parameters
    ----------
    x: array-like
        Points on the real line
    out: ndarray, optional
        Output array for the values of `e1` at `x`

    Returns
    -------
    ndarray
        Values of `e1` at `x`

    See Also
    --------
    ei: Exponential integral :math:`Ei`
    en: Generalization of :math:`E_1`

    References
    ----------
    .. [1] Digital Library of Mathematical Functions, 6.2.1
           https://dlmf.nist.gov/6.2#E1

    """
    return _e1(x)
