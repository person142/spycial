"""The implementation here is ported from Boost, which is:

(C) Copyright John Maddock 2006.
Use, modification and distribution are subject to the
Boost Software License, Version 1.0. (See accompanying file
LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

"""
from numba import njit, vectorize
import numpy as np

from . import settings
from .evalpoly import _devalpoly

P_ONE_HALF = np.array([
    -0.00538772965071242932965,
    0.00822687874676915743155,
    0.0219878681111168899165,
    -0.0365637971411762664006,
    -0.0126926147662974029034,
    0.0334806625409744615033,
    -0.00836874819741736770379,
    -0.000508781949658280665617,
])

Q_ONE_HALF = np.array([
    0.000886216390456424707504,
    -0.00233393759374190016776,
    0.0795283687341571680018,
    -0.0527396382340099713954,
    -0.71228902341542847553,
    0.662328840472002992063,
    1.56221558398423026363,
    -1.56574558234175846809,
    -0.970005043303290640362,
    1.0,
])

P_ONE_QUARTER = np.array([
    -3.67192254707729348546,
    21.1294655448340526258,
    17.445385985570866523,
    -44.6382324441786960818,
    -18.8510648058714251895,
    17.6447298408374015486,
    8.37050328343119927838,
    0.105264680699391713268,
    -0.202433508355938759655,
])

Q_ONE_QUARTER = np.array([
    1.72114765761200282724,
    -22.6436933413139721736,
    10.8268667355460159008,
    48.5609213108739935468,
    -20.1432634680485188801,
    -28.6608180499800029974,
    3.9713437953343869095,
    6.24264124854247537712,
    1.0,
])

P3 = np.array([
    -0.681149956853776992068e-9,
    0.285225331782217055858e-7,
    -0.679465575181126350155e-6,
    0.00214558995388805277169,
    0.0290157910005329060432,
    0.142869534408157156766,
    0.337785538912035898924,
    0.387079738972604337464,
    0.117030156341995252019,
    -0.163794047193317060787,
    -0.131102781679951906451,
])

Q3 = np.array([
    0.01105924229346489121,
    0.152264338295331783612,
    0.848854343457902036425,
    2.59301921623620271374,
    4.77846592945843778382,
    5.38168345707006855425,
    3.46625407242567245975,
    1.0,
])

P6 = np.array([
    0.266339227425782031962e-11,
    -0.230404776911882601748e-9,
    0.460469890584317994083e-5,
    0.000157544617424960554631,
    0.00187123492819559223345,
    0.00950804701325919603619,
    0.0185573306514231072324,
    -0.00222426529213447927281,
    -0.0350353787183177984712,
])

Q6 = np.array([
    0.764675292302794483503e-4,
    0.00263861676657015992959,
    0.0341589143670947727934,
    0.220091105764131249824,
    0.762059164553623404043,
    1.3653349817554063097,
    1.0,
])

P18 = np.array([
    0.99055709973310326855e-16,
    -0.281128735628831791805e-13,
    0.462596163522878599135e-8,
    0.449696789927706453732e-6,
    0.149624783758342370182e-4,
    0.000209386317487588078668,
    0.00105628862152492910091,
    -0.00112951438745580278863,
    -0.0167431005076633737133,
])

Q18 = np.array([
    0.282243172016108031869e-6,
    0.275335474764726041141e-4,
    0.000964011807005165528527,
    0.0160746087093676504695,
    0.138151865749083321638,
    0.591429344886417493481,
    1.0,
])

# In the original Boost code there are larger coefficients for
# long-doubles, so these coefficients are only for x < 44. Double
# precision underflows before we hit 44, however.
P44 = np.array([
    -0.116765012397184275695e-17,
    0.145596286718675035587e-11,
    0.411632831190944208473e-9,
    0.396341011304801168516e-7,
    0.162397777342510920873e-5,
    0.254723037413027451751e-4,
    -0.779190719229053954292e-5,
    -0.0024978212791898131227,
])

Q44 = np.array([
    0.509761276599778486139e-9,
    0.144437756628144157666e-6,
    0.145007359818232637924e-4,
    0.000690538265622684595676,
    0.0169410838120975906478,
    0.207123112214422517181,
    1.0,
])


@njit('float64(float64, float64)', cache=settings.CACHE)
def _erf_erfc_inv(p, q):
    if p <= 0.5:
        Y = np.float32(0.0891314744949340820313)
        g = p * (p + 10)
        r = _devalpoly(P_ONE_HALF, p) / _devalpoly(Q_ONE_HALF, p)
        result = g * Y + g * r
        return result
    elif q >= 0.25:
        Y = np.float32(2.249481201171875)
        g = np.sqrt(-2 * np.log(q))
        xs = q - 0.25
        r = _devalpoly(P_ONE_QUARTER, xs) / _devalpoly(Q_ONE_QUARTER, xs)
        result = g / (Y + r)
        return result

    x = np.sqrt(-np.log(q))
    if x < 3:
        Y = np.float32(0.807220458984375)
        xs = x - 1.125
        R = _devalpoly(P3, xs) / _devalpoly(Q3, xs)
        result = Y * x + R * x
        return result
    elif x < 6:
        Y = np.float32(0.93995571136474609375)
        xs = x - 3
        R = _devalpoly(P6, xs) / _devalpoly(Q6, xs)
        result = Y * x + R * x
        return result
    elif x < 18:
        Y = np.float32(0.98362827301025390625)
        xs = x - 6
        R = _devalpoly(P18, xs) / _devalpoly(Q18, xs)
        result = Y * x + R * x
        return result
    else:
        Y = np.float32(0.99714565277099609375)
        xs = x - 18
        R = _devalpoly(P44, xs) / _devalpoly(Q44, xs)
        result = Y * x + R * x
        return result


@njit('float64(float64)', cache=settings.CACHE)
def _erfinv(x):
    if np.isnan(x):
        return x
    elif x < -1 or x > 1:
        return np.nan
    elif x == 1:
        return np.inf
    elif x == -1:
        return -np.inf
    elif x == 0:
        return 0

    if x < 0:
        p = -x
        q = 1 - p
        s = -1
    else:
        p = x
        q = 1 - x
        s = 1
    return s * _erf_erfc_inv(p, q)


@njit('float64(float64)', cache=settings.CACHE)
def _erfcinv(x):
    if np.isnan(x):
        return x
    elif x < 0 or x > 2:
        return np.nan
    elif x == 0:
        return np.inf
    elif x == 2:
        return -np.inf

    if x > 1:
        q = 2 - x
        p = 1 - q
        s = -1
    else:
        p = 1 - x
        q = x
        s = 1

    return s * _erf_erfc_inv(p, q)


@vectorize(['float64(float64)'], nopython=True, cache=settings.CACHE)
def erfinv(x):
    """Inverse error function.

    Parameters
    ----------
    x : array-like
        Points on the real line
    out : ndarray, optional
        Output array for the values of `erfinv` at `x`

    Returns
    -------
    ndarray
        Values of `erfinv` at `x`

    """
    return _erfinv(x)


@vectorize(['float64(float64)'], nopython=True, cache=settings.CACHE)
def erfcinv(x):
    """Inverse complementary error function.

    Parameters
    ----------
    x : array-like
        Points on the real line
    out : ndarray, optional
        Output array for the values of `erfcinv` at `x`

    Returns
    -------
    ndarray
        Values of `erfcinv` at `x`

    """
    return _erfcinv(x)
