import numpy as np
from numpy.testing import assert_allclose
import mpmath

import spycial as sc
from spycial.test_utilities import mpmath_allclose, Arg
from spycial.constants import _ε


def test_erfinv_special_points():
    assert np.isnan(sc.erfinv(np.nan))
    assert np.all(np.isnan([sc.erfinv(-2), sc.erfinv(2)]))
    assert sc.erfinv(-1) == -np.inf
    assert sc.erfinv(1) == np.inf
    assert sc.erfinv(0) == 0


def test_erfinv():
    mpmath_allclose(sc.erfinv, mpmath.erfinv, [Arg(-1, 1)], 1000, 2*_ε)


def test_erfinv_extreme_values():
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


def test_erfinv_inverts_erf():
    # Restrict the range because if erf returns ±1, then erfinv will
    # (correctly) invert that to ±∞. The relative tolerance is also
    # quite high because
    #
    # κ(erf(x)) = erf(x) (√(π) / 2) (exp(x²) / x)
    #
    # where κ is the condition number of erfinv.
    x = np.linspace(-5, 5, 200)
    assert_allclose(sc.erfinv(sc.erf(x)), x, atol=0, rtol=5e-7)
    x = np.linspace(-1, 1, 200)
    assert_allclose(sc.erf(sc.erfinv(x)), x, atol=0, rtol=2*_ε)


def test_erfcinv_special_points():
    assert np.isnan(sc.erfcinv(np.nan))
    assert np.all(np.isnan([sc.erfcinv(-1), sc.erfcinv(3)]))
    assert sc.erfcinv(0) == np.inf
    assert sc.erfcinv(2) == -np.inf
    assert sc.erfcinv(1) == 0


def mpmath_erfcinv(x):
    return mpmath.erfinv(mpmath.mpf(1) - x)


def test_erfcinv():
    mpmath_allclose(
        sc.erfcinv,
        mpmath_erfcinv,
        [Arg(0, 2)],
        200,
        2*_ε,
        dps=42,
    )


def test_erfinv_extreme_values():
    assert_allclose(
        sc.erfcinv(5e-324),
        27.213293210812948815,  # Computed using Mpmath
        atol=0,
        rtol=_ε,
    )
    x_near_2 = np.nextafter(2, -np.inf)
    assert_allclose(
        sc.erfcinv(x_near_2),
        float(mpmath_erfcinv(x_near_2)),
        atol=0,
        rtol=5e-14,
    )


def test_erfcinv_inverts_erfc():
    # To double precision erfc becomes 2 shortly before -2 and 0
    # shortly after 27.
    x = np.linspace(-2, 27)
    assert_allclose(sc.erfcinv(sc.erfc(x)), x, atol=0, rtol=5e-10)
    x = np.linspace(0, 2)
    assert_allclose(sc.erfc(sc.erfcinv(x)), x, atol=0, rtol=4*_ε)
