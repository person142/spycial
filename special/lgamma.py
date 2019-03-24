"""Implementations of log-Gamma and related functions.

Author: Josh Wilson

Distributed under the same license as Scipy.

References
----------
[1] Hare, "Computing the Principal Branch of log-Gamma",
    Journal of Algorithms, 1997.

[2] Julia,
    https://github.com/JuliaLang/julia/blob/master/base/special/gamma.jl

[3] Boost, "Log Gamma",
    http://www.boost.org/doc/libs/1_53_0/libs/math/doc/sf_and_dist/html/math_toolkit/special/sf_gamma/lgamma.html

"""
import numba
from numba import njit, generated_jit, vectorize
import numpy as np

from . import settings
from .constants import _2π, _2πj, _logπ, _log2π_2, _e
from .trig import _csinpi, _dsinpi
from .evalpoly import _cevalpoly, _devalrational
from .lanczos import _lanczos_g, _lanczos_sum_expg_scaled


SMALLX = 7
SMALLY = 7
TAYLOR_RADIUS = 0.2

STIRLING_COEFFS = np.array([
    -2.955065359477124183e-2, 6.4102564102564102564e-3,
    -1.9175269175269175269e-3, 8.4175084175084175084e-4,
    -5.952380952380952381e-4, 7.9365079365079365079e-4,
    -2.7777777777777777778e-3, 8.3333333333333333333e-2
])

TAYLOR_COEFFS = np.array([
    -4.3478266053040259361e-2, 4.5454556293204669442e-2,
    -4.7619070330142227991e-2, 5.000004769810169364e-2,
    -5.2631679379616660734e-2, 5.5555767627403611102e-2,
    -5.8823978658684582339e-2, 6.2500955141213040742e-2,
    -6.6668705882420468033e-2, 7.1432946295361336059e-2,
    -7.6932516411352191473e-2, 8.3353840546109004025e-2,
    -9.0954017145829042233e-2, 1.0009945751278180853e-1,
    -1.1133426586956469049e-1, 1.2550966952474304242e-1,
    -1.4404989676884611812e-1, 1.6955717699740818995e-1,
    -2.0738555102867398527e-1, 2.7058080842778454788e-1,
    -4.0068563438653142847e-1, 8.2246703342411321824e-1,
    -5.7721566490153286061e-1
])

# The following coefficients for lgamma come from Boost and are:
#
# (C) Copyright John Maddock 2006.
# Use, modification and distribution are subject to the
# Boost Software License, Version 1.0. (See accompanying file
# LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

LGAMMA_2TO3_NUM = np.array([
    -0.324588649825948492091e-4,
    -0.541009869215204396339e-3,
    -0.259453563205438108893e-3,
    0.172491608709613993966e-1,
    0.494103151567532234274e-1,
    0.25126649619989678683e-1,
    -0.180355685678449379109e-1
])

LGAMMA_2TO3_DENOM = np.array([
    -0.223352763208617092964e-6,
    0.224936291922115757597e-3,
    0.82130967464889339326e-2,
    0.988504251128010129477e-1,
    0.541391432071720958364e0,
    0.148019669424231326694e1,
    0.196202987197795200688e1,
    0.1e1
])

LGAMMA_1TO1_5_NUM = np.array([
    -0.100346687696279557415e-2,
    -0.240149820648571559892e-1,
    -0.158413586390692192217e0,
    -0.406567124211938417342e0,
    -0.414983358359495381969e0,
    -0.969117530159521214579e-1,
    0.490622454069039543534e-1
])

LGAMMA_1TO1_5_DENOM = np.array([
    0.195768102601107189171e-2,
    0.577039722690451849648e-1,
    0.507137738614363510846e0,
    0.191415588274426679201e1,
    0.348739585360723852576e1,
    0.302349829846463038743e1,
    0.1e1
])

LGAMMA_1_5TO2_NUM = np.array([
    0.431171342679297331241e-3,
    -0.850535976868336437746e-2,
    0.542809694055053558157e-1,
    -0.142440390738631274135e0,
    0.144216267757192309184e0,
    -0.292329721830270012337e-1
])

LGAMMA_1_5TO2_DENOM = np.array([
    -0.827193521891290553639e-6,
    -0.100666795539143372762e-2,
    0.25582797155975869989e-1,
    -0.220095151814995745555e0,
    0.846973248876495016101e0,
    -0.150169356054485044494e1,
    0.1e1
])


@njit('float64(float64)', cache=settings.CACHE)
def _lgamma_positive(x):
    """Evaluate lgamma for positive arguments.

    See [3] for what's going on here.

    """
    if x < 2.0:
        res = 0.0
        if x < 1.0:
            # One step of the recurrence relation
            res = -np.log(x)
            x += 1.0
        if x < 1.5:
            dx = x - 1.0
            Y = np.float32(0.52815341949462890625)
            R = _devalrational(LGAMMA_1TO1_5_NUM, LGAMMA_1TO1_5_DENOM,
                               dx)
            res += dx*(x - 2.0)*(Y + R)
            return res
        else:
            dx = 2.0 - x
            Y = np.float32(0.452017307281494140625)
            R = _devalrational(LGAMMA_1_5TO2_NUM, LGAMMA_1_5TO2_DENOM,
                               dx)
            res += dx*(1.0 - x)*(Y + R)
        return res
    elif x < 4.0:
        res = 0.0
        if x > 3.0:
            # One step of the recurrence relation
            x -= 1.0
            res = np.log(x)
        dx = x - 2.0
        Y = np.float32(0.158963680267333984375)
        R = _devalrational(LGAMMA_2TO3_NUM, LGAMMA_2TO3_DENOM, dx)
        res += dx*(x + 1.0)*(Y + R)
        return res
    else:
        return ((x - 0.5)*np.log((x + _lanczos_g - 0.5)/_e)
                + np.log(_lanczos_sum_expg_scaled(x)))


@njit('float64(float64)', cache=settings.CACHE)
def _lgamma(x):
    if x == np.inf or np.isnan(x):
        return x
    elif x == -np.inf:
        return np.nan
    elif x <= 0.0 and x == np.floor(x):
        return np.inf
    elif x > 0:
        return _lgamma_positive(x)
    else:
        return (_logπ - np.log(np.abs(x*_dsinpi(x)))
                - _lgamma_positive(-x))


@njit('complex128(complex128)', cache=settings.CACHE)
def _cloggamma_stirling(z):
    """Stirling series for log-Gamma.

    The coefficients are B[2*n]/(2*n*(2*n - 1)) where B[2*n] is the
    (2*n)th Bernoulli number. See (1.1) in [1].

    """
    rz = 1.0/z
    rzz = rz/z

    return ((z - 0.5)*np.log(z) - z + _log2π_2
            + rz*_cevalpoly(STIRLING_COEFFS, rzz))


@njit('complex128(complex128)', cache=settings.CACHE)
def _cloggamma_recurrence(z):
    """Backward recurrence relation.

    See Proposition 2.2 in [1] and the Julia implementation [2].

    """
    signflips = 0
    sb = 0
    shiftprod = z

    z += 1
    while z.real <= SMALLX:
        shiftprod *= z
        nsb = np.signbit(shiftprod.imag)
        signflips += 1 if nsb != 0 and sb == 0 else 0
        sb = nsb
        z += 1
    return _cloggamma_stirling(z) - np.log(shiftprod) - signflips*_2πj


@njit('complex128(complex128)', cache=settings.CACHE)
def _cloggamma_taylor(z):
    """Taylor series for log-Gamma around z = 1.

    It is

    loggamma(z + 1) = -gamma*z + zeta(2)*z**2/2 - zeta(3)*z**3/3 ...

    where gamma is the Euler-Mascheroni constant.

    """
    z = z - 1
    return z*_cevalpoly(TAYLOR_COEFFS, z)


@njit('complex128(complex128)', cache=settings.CACHE)
def _cloggamma(z):
    """Compute the principal branch of log-Gamma."""

    if np.isnan(z):
        return np.complex(np.nan, np.nan)
    elif z.real <= 0 and z == np.floor(z.real):
        return np.complex(np.nan, np.nan)
    elif z.real > SMALLX or abs(z.imag) > SMALLY:
        return _cloggamma_stirling(z)
    elif abs(z - 1) <= TAYLOR_RADIUS:
        return _cloggamma_taylor(z)
    elif abs(z - 2) <= TAYLOR_RADIUS:
        # Recurrence relation and the Taylor series around 1
        return np.log(z - 1) + _cloggamma_taylor(z - 1)
    elif z.real < 0.1:
        # Reflection formula; see Proposition 3.1 in [1]
        tmp = np.copysign(_2π, z.imag)*np.floor(0.5*z.real + 0.25)
        return (np.complex(_logπ, tmp) - np.log(_csinpi(z))
                - _cloggamma(1 - z))
    elif np.signbit(z.imag) == 0:
        # z.imag >= 0 but is not -0.0
        return _cloggamma_recurrence(z)
    else:
        return _cloggamma_recurrence(z.conjugate()).conjugate()


@njit('float64(float64)', cache=settings.CACHE)
def _dloggamma(x):
    if x <= 0.0:
        return np.nan
    return _lgamma(x)


@generated_jit(nopython=True, cache=settings.CACHE)
def _loggamma(a):
    if a == numba.types.float64:
        return lambda a: _dloggamma(a)
    elif a == numba.types.complex128:
        return lambda a: _cloggamma(a)


@vectorize(['float64(float64)'], nopython=True, cache=settings.CACHE)
def lgamma(x):
    r"""Logarithm of the absolute value of the Gamma function.

    Parameters
    ----------
    x : array-like
        Points on the real line
    out : ndarray, optional
        Output array for the values of `lgamma` at `x`

    Returns
    -------
    ndarray
        Values of `lgamma` at `x`

    """
    return _lgamma(x)


@vectorize(
    ['float64(float64)', 'complex128(complex128)'],
    nopython=True,
    cache=settings.CACHE,
)
def loggamma(z):
    r"""Principal branch of the logarithm of the Gamma function.

    Defined to be :math:`\log(\Gamma(x))` for :math:`x > 0` and
    extended to the complex plane by analytic continuation. The
    function has a single branch cut on the negative real axis.

    Parameters
    ----------
    z : array-like
        Points on the real line or complex plain
    out : ndarray, optional
        Output array for the values of `loggamma` at `z`

    Returns
    -------
    ndarray
        Values of `loggamma` at `z`

    """
    return _loggamma(z)
