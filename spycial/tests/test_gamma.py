import numpy as np
from numpy.testing import assert_equal
import mpmath

import spycial as sc
from spycial.test_utilities import Arg, mpmath_allclose


def test_gamma():
    def mpmath_gamma(x):
        try:
            return mpmath.gamma(x)
        except ValueError:
            # Gamma pole
            return np.nan

    # Gamma overflows around 170 so there's no point in testing beyond
    # that.
    mpmath_allclose(sc.gamma, mpmath_gamma,
                    [Arg(-np.inf, 180)], 1000, 1e-14)


def test_gamma_int():
    # These values are hard-coded, so they should be exactly correct
    x = np.arange(1, 11, dtype=np.float64)
    y = [float(mpmath.factorial(x0 - 1)) for x0 in x]
    assert_equal(sc.gamma(x), y)
