"""Exponential integral `E_n`.

The implementation was original ported from SciPy and was then
improved for the the region with `n <= 50` and `0.5 < x < 1.5`.

"""
from numba import njit, vectorize
import numpy as np

from . import settings
from .constants import _ε, MINEXP
from .evalpoly import _devalpoly
from .fma import _fma
from .e1 import _e1
from .gamma import _dgamma

A = (
    np.array([
        1.0,
    ]),
    np.array([
        1.0,
    ]),
    np.array([
        -2.0,
        1.0,
    ]),
    np.array([
        6.0,
        -8.0,
        1.0,
    ]),
    np.array([
        -24.0,
        58.0,
        -22.0,
        1.0,
    ]),
    np.array([
        120.0,
        -444.0,
        328.0,
        -52.0,
        1.0,
    ]),
    np.array([
        -720.0,
        3708.0,
        -4400.0,
        1452.0,
        -114.0,
        1.0,
    ]),
    np.array([
        5040.0,
        -33984.0,
        58140.0,
        -32120.0,
        5610.0,
        -240.0,
        1.0,
    ]),
    np.array([
        -40320.0,
        341136.0,
        -785304.0,
        644020.0,
        -195800.0,
        19950.0,
        -494.0,
        1.0,
    ]),
    np.array([
        362880.0,
        -3733920.0,
        11026296.0,
        -12440064.0,
        5765500.0,
        -1062500.0,
        67260.0,
        -1004.0,
        1.0,
    ]),
    np.array([
        -3628800.0,
        44339040.0,
        -162186912.0,
        238904904.0,
        -155357384.0,
        44765000.0,
        -5326160.0,
        218848.0,
        -2026.0,
        1.0,
    ]),
    np.array([
        39916800.0,
        -568356480.0,
        2507481216.0,
        -4642163952.0,
        4002695088.0,
        -1648384304.0,
        314369720.0,
        -25243904.0,
        695038.0,
        -4072.0,
        1.0,
    ]),
    np.array([
        -479001600.0,
        7827719040.0,
        -40788301824.0,
        92199790224.0,
        -101180433024.0,
        56041398784.0,
        -15548960784.0,
        2051482776.0,
        -114876376.0,
        2170626.0,
        -8166.0,
        1.0,
    ]),
)


# The nth entry is ψ(n).
PSI = np.array([
    np.nan,  # This value will never get accessed
    -0.57721566490153286061,
    0.42278433509846713939,
    0.92278433509846713939,
    1.2561176684318004727,
    1.5061176684318004727,
    1.7061176684318004727,
    1.8727843350984671394,
    2.0156414779556099965,
    2.1406414779556099965,
    2.2517525890667211076,
    2.3517525890667211076,
    2.4426616799758120167,
    2.5259950133091453501,
    2.6029180902322222731,
    2.6743466616607937017,
    2.7410133283274603684,
    2.8035133283274603684,
    2.8623368577392250743,
    2.9178924132947806298,
    2.9705239922421490509,
    3.0205239922421490509,
    3.0681430398611966699,
    3.1135975853157421245,
    3.1570758461853073419,
    3.1987425128519740085,
    3.2387425128519740085,
    3.2772040513135124701,
    3.3142410883505495071,
    3.3499553740648352214,
    3.3844381326855248766,
    3.4177714660188582099,
    3.4500295305349872422,
    3.4812795305349872422,
    3.5115825608380175452,
    3.5409943255438998981,
    3.5695657541153284696,
    3.5973435318931062473,
    3.6243705589201332744,
    3.6506863483938174849,
    3.6763273740348431259,
    3.7013273740348431259,
    3.7257176179372821503,
    3.7495271417468059598,
    3.7727829557002943319,
    3.7955102284275670592,
    3.8177324506497892814,
    3.8394715810845718901,
    3.8607481768292527412,
    3.8815815101625860745,
    3.901989673427892197,
])


# The nth value is E_n(1)
EN_AT_1 = np.array([
    0.3678794411714423216,
    0.21938393439552027368,
    0.14849550677592204792,
    0.10969196719776013684,
    0.086062491324560728252,
    0.070454237461720398336,
    0.059485040741944384652,
    0.051399066738249656157,
    0.045211482061884666491,
    0.040333494888694706888,
    0.036393994031416401634,
    0.033148544714002591996,
    0.030430081496130884509,
    0.028120779972942619757,
    0.026135281630653823218,
    0.024410297110056321313,
    0.022897942937425733352,
    0.021561343639626036765,
    0.020371652795989193225,
    0.019305988243080729354,
    0.018345971206755873276,
    0.017476673498234322416,
    0.01668584607967657139,
    0.015963345231443897737,
    0.015300699823478192342,
    0.014690780889498505386,
    0.014127546411277752648,
    0.013605842106160175729,
    0.013121244409825264662,
    0.012669935598629180605,
    0.012248603640441832448,
    0.011854361251033349638,
    0.011484679997432547482,
    0.011137336286687805441,
    0.010810366814689530793,
    0.010502031598728023259,
    0.010210783130648979952,
    0.009935240501133148379,
    0.0096741675856840317086,
    0.0094264545680462707865,
    0.0091911022205998987387,
    0.0089672084737710605714,
    0.0087539568950651527079,
    0.0085506067684851706878,
    0.0083564845209990035095,
    0.008170976287510075411,
    0.0079935214418651610263,
    0.0078236069506429817515,
    0.0076607624302297731882,
    0.0075045558071085947585,
    0.0073545894972313005477,
])


@njit('float64(uint64, float64)', cache=settings.CACHE)
def _en_continued_fraction(n, x):
    """Continued fraction from DLMF 8.19.17."""
    # Start with k = 1.
    Akm2 = 1.0
    Akm1 = 1.0
    Bkm2 = x
    Bkm1 = x + n
    xkm1 = Akm1 / Bkm1
    for k in range(2, 200):
        if k % 2 == 0:
            ak = 0.5 * k
            bk = x
        else:
            ak = n + 0.5 * (k - 1)
            bk = 1
        Ak = bk * Akm1 + ak * Akm2
        Bk = bk * Bkm1 + ak * Bkm2
        xk = Ak / Bk
        if abs(xk - xkm1) < abs(xk) * _ε:
            break
        else:
            Akm2 = Akm1
            Akm1 = Ak
            Bkm2 = Bkm1
            Bkm1 = Bk
            xkm1 = xk

    return np.exp(-x) * xk


@njit('float64(uint64, float64)', cache=settings.CACHE)
def _en_finite_series(n, x):
    """DLMF 8.19.7."""
    negx = -x
    s = 1 + negx
    for k in range(2, n - 1):
        s = _fma(negx / k, s, 1)

    return (
        negx**(n - 1) * _e1(x) / _dgamma(n)
        + np.exp(negx) * s / (n - 1)
    )


@njit('float64(uint64, float64)', cache=settings.CACHE)
def _en_taylor_series_at_1(n, x):
    """Taylor series for `E_n` at `x = 1`.

    The coefficients are computed using DLMF 8.19.13.

    """
    xm1 = x - 1.0
    fac = 1.0
    result = EN_AT_1[n]
    for k in range(1, 16):
        fac *= -xm1 / k
        term = EN_AT_1[n - k] * fac
        result += term
        if abs(term) < _ε * abs(result):
            break
    return result


@njit('float64(uint64, float64)', cache=settings.CACHE)
def _en_power_series(n, x):
    """Power series from DLMF 8.19.8."""
    neg_x = -x
    xk = 0.0
    yk = 1.0
    pk = 1.0 - n
    sk = 1.0 / pk
    for _ in range(100):
        pk += 1.0
        xk += 1.0
        yk *= neg_x / xk
        if pk == 0:
            continue
        term = yk / pk
        sk += term
        if abs(term) < _ε * abs(sk):
            break

    return neg_x**(n - 1) * (PSI[n] - np.log(x)) / _dgamma(n) - sk


@njit('float64(uint64, float64)', cache=settings.CACHE)
def _en_asymptotic_series_large_n(n, x):
    """Asymptotic expansion for large n from DLMF 8.20(ii)."""
    lmbda = x / n
    multiplier = 1 / n / (lmbda + 1) / (lmbda + 1)
    fac = 1
    res = 1  # A[0] = [1]

    expfac = np.exp(-lmbda * n) / (lmbda + 1) / n
    if expfac == 0:
        return 0

    # Do the k = 1 term outside the loop since A[1] = [1]
    fac *= multiplier;
    res += fac;

    for k in range(2, len(A)):
        fac *= multiplier
        term = fac * _devalpoly(A[k], lmbda)
        res += term
        if abs(term) < _ε * abs(res):
            break

    return expfac * res;


@njit('float64(uint64, float64)', cache=settings.CACHE)
def _en(n, x):
    if n == 1:
        return _e1(x)
    elif np.isnan(x):
        return np.nan
    elif x < 0:
        return np.nan
    elif x == 0:
        if n == 0:
            return np.inf
        else:
            return 1 / (n - 1)
    elif n == 0:
        return np.exp(-x) / x
    elif x > -MINEXP:
        # By e.g. DLMF 8.19.21, E_n(x) < exp(-x).
        return 0
    elif n > 50:
        return _en_asymptotic_series_large_n(n, x)
    elif x < 0.5:
        return _en_power_series(n, x)
    elif x > 1.5:
        return _en_continued_fraction(n, x)
    else:
        if n == 2:
            return np.exp(-x) - x * _e1(x)
        elif n < 15:
            return _en_finite_series(n, x)
        else:
            return _en_taylor_series_at_1(n, x)


@vectorize(
    ['float64(uint64, float64)'],
    nopython=True,
    cache=settings.CACHE,
)
def en(n, x):
    r"""Generalized exponential integral :math:`E_n(x)`.

    For integer :math:`n \geq 0` and real :math:`x \geq 0` the
    generalized exponential integral can be defined as [1]_

    .. math::

        E_n(x) = x^{n - 1} \int_x^\infty \frac{e^{-t}}{t^n} dt.

    Parameters
    ----------
    n: array-like
        Non-negative integers
    x: array-like
        Points on the real line

    Returns
    -------
    ndarray
        Values of `en` at `n` and `x`.

    See Also
    --------
    e1: Special case of :math:`E_n` for :math:`n = 1`

    References
    ----------
    .. [1] Digital Library of Mathematical Functions, 8.19.2
           https://dlmf.nist.gov/8.19#E2

    """
    return _en(n, x)
