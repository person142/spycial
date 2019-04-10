import numpy as np
from numpy.testing import assert_allclose
import mpmath

import spycial as sc
from spycial.test_utilities import mpmath_allclose, Arg
from spycial.constants import _ε


def test_special_points():
    assert np.isnan(sc.erfinv(np.nan))
    assert np.all(np.isnan([sc.erfinv(-2), sc.erfinv(2)]))
    assert sc.erfinv(-1) == -np.inf
    assert sc.erfinv(1) == np.inf
    assert sc.erfinv(0) == 0


def test_erf():
    mpmath_allclose(sc.erfinv, mpmath.erfinv, [Arg(-1, 1)], 1000, 2*_ε)


def test_extreme_values():
    x_near_n1 = np.nextafter(-1, np.inf)
    assert_allclose(
        sc.erfinv(x_near_n1),
        float(mpmath.erfinv(x_near_n1)),
        atol=0,
        rtol=1e-14,
    )
    x_near_1 = np.nextafter(1, -np.inf)
    assert_allclose(
        sc.erfinv(x_near_1),
        float(mpmath.erfinv(x_near_1)),
        atol=0,
        rtol=1e-14,
    )
