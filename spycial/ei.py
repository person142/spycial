"""Implementation ported from Boost, which is:

Copyright John Maddock 2007.
Use, modification and distribution are subject to the
Boost Software License, Version 1.0. (See accompanying file
LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

"""
from numba import njit, vectorize
import numpy as np

from . import settings
from .constants import _MAXEXP
from .evalpoly import _devalpoly
from .e1 import _e1

P6 = np.array([
    0.2777056254402008721e-6,
    0.798296365679269702435e-5,
    0.000116419523609765200999,
    0.00115478237227804306827,
    0.00726224593341228159561,
    0.0499434773576515260534,
    0.114670926327032002811,
    0.780836076283730801839,
    0.356343618769377415068,
    2.98677224343598593013,
])

Q6 = np.array([
    -0.138972589601781706598e-4,
    0.000389034007436065401822,
    -0.00504800158663705747345,
    0.0391523431392967238166,
    -0.195114782069495403315,
    0.62215109846016746276,
    -1.17090412365413911947,
    1.0,
])

P10 = np.array([
    -0.396487648924804510056e-5,
    -0.554086272024881826253e-4,
    -0.000374885917942100256775,
    -0.00247496209592143627977,
    -0.00761224003005476438412,
    -0.0264095520754134848538,
    -0.0349921221823888744966,
    0.00139324086199402804173,
])

Q10 = np.array([
    0.263649630720255691787e-4,
    0.000402453408512476836472,
    0.00365334190742316650106,
    0.0223851099128506347278,
    0.100128624977313872323,
    0.329061095011767059236,
    0.744625566823272107711,
    1.0,
])

P20 = np.array([
    -0.138652200349182596186e-4,
    -0.000209750022660200888349,
    -0.00155941947035972031334,
    -0.00720603636917482065907,
    -0.0226059218923777094596,
    -0.0478447572647309671455,
    -0.0652810444222236895772,
    -0.0484607730127134045806,
    -0.00893891094356945667451,
])

Q20 = np.array([
    0.000159150281166108755531,
    0.00278170769163303669021,
    0.0233458478275769288159,
    0.122537731979686102756,
    0.438873285773088870812,
    1.09601437090337519977,
    1.86232465043073157508,
    1.97017214039061194971,
    1.0,
])

P40 = np.array([
    -0.113161784705911400295e-9,
    -0.000192178045857733706044,
    -0.00207592267812291726961,
    -0.00994403059883350813295,
    -0.0272050837209380717069,
    -0.0453759383048193402336,
    -0.0449814350482277917716,
    -0.0229930320357982333406,
    -0.00356165148914447597995,
])

Q40 = np.array([
    0.00488071077519227853585,
    0.0651165455496281337831,
    0.383213198510794507409,
    1.2985244073998398643,
    2.75088464344293083595,
    3.6599610090072393012,
    2.84354408840148561131,
    1.0,
])

P_GT40 = np.array([
    -38703.1431362056714134,
    18932.0850014925993025,
    -2516.35323679844256203,
    94.7365094537197236011,
    0.19029710559486576682,
    -0.0130653381347656243849,
])

Q_GT40 = np.array([
    8297.16296356518409347,
    54738.2833147775537106,
    -70126.245140396567133,
    22329.1459489893079041,
    -2354.56211323420194283,
    61.9733592849439884145,
    1.0,
])

EXP40 = 2.3538526683701998541e17


@njit('float64(float64)', cache=settings.CACHE)
def _ei(x):
    if x < 0:
        return -_e1(-x)
    elif x == 0:
        return -np.inf

    if x <= 6:
        r1 = 0.37250741078136662132
        r2 = 0.13140183414386028201e-16
        r = 0.37250741078136663446
        t = (x / 3) - 1
        result = _devalpoly(P6, t) / _devalpoly(Q6, t)
        t = (x - r1) - r2
        result *= t
        if abs(t) < 0.1:
            result += np.log1p(t / r)
        else:
            result += np.log(x / r)
        return result
    elif x <= 10:
        Y = np.float32(1.158985137939453125)
        t = x / 2 - 4
        result = Y + _devalpoly(P10, t) / _devalpoly(Q10, t)
        result *= np.exp(x) / x
        result += x
        return result
    elif x <= 20:
        Y = np.float32(1.0869731903076171875)
        t = x / 5 - 3
        result = Y + _devalpoly(P20, t) / _devalpoly(Q20, t)
        result *= np.exp(x) / x
        result += x
        return result
    elif x <= 40:
        Y = np.float32(1.03937530517578125)
        t = x / 10 - 3
        result = Y + _devalpoly(P40, t) / _devalpoly(Q40, t)
        result *= np.exp(x) / x
        result += x
        return result
    else:
        Y = np.float32(1.013065338134765625)
        t = 1 / x
        result = Y + _devalpoly(P_GT40, t) / _devalpoly(Q_GT40, t)
        if x < 41:
            result *= np.exp(x) / x
        else:
            # Avoid premature overflow if we can
            t = x - 40
            if t > _MAXEXP:
                return np.inf
            else:
                result *= (np.exp(x - 40) / x) * EXP40
        result += x
        return result


@vectorize(['float64(float64)'], nopython=True, cache=settings.CACHE)
def ei(x):
    r"""Exponential integral :math:`Ei(x)`.

    The exponential integral is defined as [1]_

    .. math::

        Ei(x) = \int_{-\infty}^x \frac{e^t}{t} dt.

    For :math:`x > 0` the integral is understood as a Cauchy principle
    value.

    Parameters
    ----------
    x: array-like
        Points on the real line
    out: ndarray, optional
        Output array for the values of `ei` at `x`

    Returns
    -------
    ndarray
        Values of `ei` at `x`

    Notes
    -----
    The exponential integrals :math:`E_1` and :math:`Ei` satisfy the
    relation

    .. math::

        E_1(x) = -Ei(-x)

    for :math:`x > 0`.

    See Also
    --------
    e1: Exponential integral :math:`E_1`

    References
    ----------

    .. [1] Digital Library of Mathematical Functions, 6.2.5
           https://dlmf.nist.gov/6.2#E5

    """
    return _ei(x)
