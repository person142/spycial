import numpy as np
from numpy.testing import assert_equal
import mpmath

import spycial as sc
from spycial.test_utilities import Arg, mpmath_allclose


def test_digamma():
    def mpmath_digamma(x):
        try:
            return mpmath.digamma(x)
        except ValueError:
            # Hit a pole.
            if x == 0.0:
                return -np.copysign(np.inf, x)
            else:
                return np.nan

    mpmath_allclose(sc.digamma, mpmath_digamma, [Arg()],
                    1000, 5e-14, dps=40)


def test_digamma_int():
    x = np.arange(1, 11, dtype=np.float64)
    with mpmath.workdps(30):
        y = [float(mpmath.digamma(x0)) for x0 in x]
    assert_equal(sc.digamma(x), y)
