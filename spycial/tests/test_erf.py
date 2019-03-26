import numpy as np
from numpy.testing import assert_equal
import mpmath

import spycial as sc
from spycial.test_utilities import mpmath_allclose, Arg


def test_erf():
    rtol = 2*np.finfo(float).eps
    mpmath_allclose(sc.erf, mpmath.erf, [Arg()], 1000, rtol)


def test_erfc():
    rtol = 64*np.finfo(float).eps  # ~1.4e-14
    mpmath_allclose(sc.erfc, mpmath.erfc, [Arg(-1e100, 1e100)],
                    1000, rtol)


def test_erfc_large():
    # Mpmath raises for very large arguments, but we know erfc should
    # just be -1 or 1 to double precision.
    x = np.linspace(100, 300)
    assert_equal(sc.erfc(x), 0.0)
    assert_equal(sc.erfc(-x), 2.0)
