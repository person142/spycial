import numpy as np
import mpmath

import spycial as sc
from spycial.test_utilities import mpmath_allclose, Arg
from spycial.constants import _ε, MINEXP


def test_special_cases():
    assert np.isnan(sc.e1(-1))
    assert sc.e1(0) == np.inf
    assert np.isnan(sc.e1(-np.inf))
    assert sc.e1(np.inf) == 0


def test_between_0_and_1():
    mpmath_allclose(
        sc.e1,
        mpmath.e1,
        [Arg(0, 1, inclusive_a=False)],
        100,
        rtol=2*_ε,
    )


def test_between_1_and_underflow():
    mpmath_allclose(
        sc.e1,
        mpmath.e1,
        [Arg(1, -MINEXP)],
        200,
        rtol=2*_ε,
    )


def test_immediately_after_underflow():
    x = np.nextafter(-MINEXP, np.inf)
    assert sc.e1(x) == float(mpmath.e1(x))
