"""Implementation for positive arguments is taken from Boost, which is:

Copyright John Maddock 2007, 2014.
Use, modification and distribution are subject to the
Boost Software License, Version 1.0. (See accompanying file
LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

"""
from numba import njit, vectorize
import numpy as np

from . import settings
from .evalpoly import _devalpoly
from .lanczos import _lanczos_g, _lanczos_sum_expg_scaled
from .gamma import _dgamma
from .trig import _dsinpi
from .constants import sqrt_2_π, _2π, _2πe, _log2π_2, _root_ε


# The nth entry is the value of zeta(2n)
ZETA_EVEN_INTEGERS = np.array([
    -0.5,
    1.6449340668482264365,
    1.0823232337111381915,
    1.0173430619844491397,
    1.0040773561979443394,
    1.0009945751278180853,
    1.0002460865533080483,
    1.0000612481350587048,
    1.0000152822594086519,
    1.0000038172932649998,
    1.0000009539620338728,
    1.0000002384505027277,
    1.0000000596081890513,
    1.0000000149015548284,
    1.0000000037253340248,
    1.0000000009313274324,
    1.0000000002328311834,
    1.0000000000582077209,
    1.0000000000145519219,
    1.0000000000036379795,
    1.0000000000009094948,
    1.0000000000002273737,
    1.0000000000000568434,
    1.0000000000000142109,
    1.0000000000000035527,
    1.0000000000000008882,
    1.000000000000000222,
    1.0000000000000000555,
])

# Rational approximation for s < 1
P1 = np.array([
    -0.933241270357061460782e-5,
    0.000451534528645796438704,
    -0.00320912498879085894856,
    0.0557616214776046784287,
    -0.49092470516353571651,
    0.24339294433593750202,
])

Q1 = np.array([
    -0.101855788418564031874e-4,
    0.00024978985622317935355,
    -0.00413421406552171059003,
    0.0419676223309986037706,
    -0.279960334310344432495,
    1,
])

# Rational approximation for 1 ≤ s < 2
P2 = np.array([
    0.110108440976732897969e-4,
    0.000249606367151877175456,
    0.00390252087072843288378,
    0.0417364673988216497593,
    0.243210646940107164097,
    0.577215664901532860516,
])

Q2 = np.array([
    0.10991819782396112081e-4,
    0.000255784226140488490982,
    0.00434930582085826330659,
    0.043460910607305495864,
    0.295201277126631761737,
    1.0,
])

# Rational approximation for 2 ≤ s < 4
P4 = np.array([
    0.328032510000383084155e-5,
    0.769875101573654070925e-4,
    0.00097541770457391752726,
    0.0128677673534519952905,
    0.0445163473292365591906,
    -0.0537258300023595030676,
])

Q4 = np.array([
    0.236276623974978646399e-7,
    0.106951867532057341359e-4,
    0.000270776703956336357707,
    0.00479039708573558490716,
    0.0487798431291407621462,
    0.33383194553034051422,
    1.0,
])

# Rational approximation for 4 ≤ s < 7
P7 = np.array([
    -0.229257310594893932383e-4,
    -0.00701721240549802377623,
    -0.138448617995741530935,
    -0.939260435377109939261,
    -2.60013301809475665334,
    -2.49710190602259410021,
])

Q7 = np.array([
    -0.1129200113474947419e-9,
    0.718833729365459760664e-8,
    -0.234055487025287216506e-6,
    0.493409563927590008943e-5,
    -0.36910273311764618902e-4,
    0.0106117950976845084417,
    0.15739599649558626358,
    0.706039025937745133628,
    1.0,
])

# Rational approximation for 7 ≤ s < 15
P15 = np.array([
    0.139348932445324888343e-5,
    0.639949204213164496988e-4,
    0.00115140923889178742086,
    -0.000189204758260076688518,
    -0.211407134874412820099,
    -1.89197364881972536382,
    -4.78558028495135619286,
])

Q15 = np.array([
    0.699841545204845636531e-12,
    -0.833378440625385520576e-10,
    0.471001264003076486547e-8,
    -0.21750464515767984778e-5,
    -0.743743682899933180415e-4,
    -0.00117592765334434471562,
    0.00873370754492288653669,
    0.244345337378188557777,
    1.0,
])

# Rational approximation for 15 ≤ s < 36
P36 = np.array([
     -0.821465709095465524192e-8,
     -0.785523633796723466968e-6,
     -0.382529323507967522614e-4,
     -0.00119459173416968685689,
     -0.0251156064655346341766,
     -0.347728266539245787271,
     -2.85827219671106697179,
     -10.3948950573308896825,
])

Q36 = np.array([
    0.222609483627352615142e-14,
    0.118507153474022900583e-7,
    0.955561123065693483991e-6,
    0.408507746266039256231e-4,
    0.00111079638102485921877,
    0.0195687657317205033485,
    0.208196333572671890965,
    1.0,
])


@njit('float64(float64)', cache=settings.CACHE)
def _zeta_between_1_and_2(sc):
    # sc = 1 - s.
    res = _devalpoly(P2, -sc) / _devalpoly(Q2, -sc)
    res += 1.0 / -sc;
    return res


@njit('float64(float64)', cache=settings.CACHE)
def _zeta_positive_arguments(s):
    if s < 1:
        sc = 1.0 - s
        res = _devalpoly(P1, sc) / _devalpoly(Q1, sc)
        res -= np.float32(1.2433929443359375)
        res += sc
        res /= sc
        return res
    elif s <= 2:
        sc = 1.0 - s
        return _zeta_between_1_and_2(sc)
    elif s <= 4:
        sc = 1.0 - s
        sm2 = s - 2
        Y = np.float32(0.6986598968505859375)
        res = _devalpoly(P4, sm2) / _devalpoly(Q4, sm2)
        res += Y + 1 / -sc
        return res
    elif s <= 7:
        sm4 = s - 4
        res = _devalpoly(P7, sm4) / _devalpoly(Q7, sm4)
        res = 1 + np.exp(res)
        return res
    elif s < 15:
        sm7 = s - 7
        res = _devalpoly(P15, sm7) / _devalpoly(Q15, sm7)
        res = 1 + np.exp(res)
        return res
    elif s < 36:
        sm15 = s - 15
        res = _devalpoly(P36, sm15) / _devalpoly(Q36, sm15)
        res = 1 + np.exp(res)
        return res
    elif s < 56:
        return 1.0 + 2**(-s)
    else:
        return 1.0


@njit('float64(float64)', cache=settings.CACHE)
def _zeta_between_negative_1_and_0(s):
    # We have to compute `sc` carefully to avoid loss of
    # precision. Here we are using the identity 1 - (1 + (-s)) = s.
    sc = s
    s = -s
    return (
        -2 * _2π**(-s - 1)
        * _dsinpi(0.5 * s)
        * s * _dgamma(s)
        * _zeta_between_1_and_2(sc)
    )


@njit('float64(float64)', cache=settings.CACHE)
def _zeta_negative_arguments(s):
    s = -s
    base = (s + _lanczos_g + 0.5) / _2πe
    fac = base**(s + 0.5)
    if fac == np.inf:
        overflowed = True
        # Compute the square root of the large factor and then
        # multiply by the small factors to see if we can stave off
        # overflow.
        fac = base**(0.5 * (s + 0.5))
    else:
        overflowed = False

    res = (
        -sqrt_2_π
        * _dsinpi(0.5 * s)
        * _lanczos_sum_expg_scaled(1 + s)
        * _zeta_positive_arguments(1 + s)
    ) * fac

    if overflowed:
        return res * fac
    else:
        return res


@njit('float64(float64)', cache=settings.CACHE)
def _zeta(s):
    if np.isnan(s):
        return s
    elif s == 1:
        return np.nan
    elif abs(s) < _root_ε:
        # Taylor series
        return -0.5 - _log2π_2 * s
    elif s > 0:
        if s < 56:
            half_s = s / 2
            int_half_s = np.int(half_s)
            if half_s == int_half_s:
                return ZETA_EVEN_INTEGERS[int_half_s]

        return _zeta_positive_arguments(s)
    elif s == -np.inf:
        return np.nan
    elif s < 0 and s > -1:
        return _zeta_between_negative_1_and_0(s)

    half_s = s / 2
    if half_s == np.floor(half_s):
        return 0

    return _zeta_negative_arguments(s)


@vectorize(['float64(float64)'], nopython=True, cache=settings.CACHE)
def zeta(s):
    """Riemann zeta function.

    Parameters
    ----------

    s: array-like
        Points on the real line
    out: ndarray, optional
        Output array for the values of `zeta` at `s`

    Returns
    -------
    ndarray
        Values of `zeta` at `s`

    """
    return _zeta(s)
