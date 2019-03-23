import numpy as np
from numpy.testing import assert_allclose
import mpmath

import special as sc
from special.test_utilities import mpmath_allclose, Arg
from special.constants import _ε


def test_special_cases():
    assert np.isnan(sc.zeta(np.nan))
    assert np.isnan(sc.zeta(1))
    assert sc.zeta(np.inf) == 1
    assert sc.zeta(0) == -0.5
    assert sc.zeta(-263) == np.inf
    assert sc.zeta(-265) == -np.inf
    assert np.isnan(sc.zeta(-np.inf))


def test_zeta_small_arguments():
    mpmath_allclose(
        sc.zeta,
        mpmath.zeta,
        [Arg(-1, 1, inclusive_b=False)],
        50,
        rtol=4*_ε,
    )


def test_zeta_less_than_negative_1():
    mpmath_allclose(
        sc.zeta,
        mpmath.zeta,
        [Arg(-261, -1)],
        300,
        rtol=5e-14,
    )


def test_avoid_overflow_for_large_negative_arguments():
    # If we computed ((s + g + 0.5) / 2πe)**(s + 0.5) naively in the
    # implementation of `zeta`, then we would overflow here. Check
    # that we instead avoid overflow.
    s = -260.00000000001
    assert_allclose(sc.zeta(s), float(mpmath.zeta(s)), atol=0, rtol=5e-14)


def test_zeta_between_1_and_56():
    mpmath_allclose(
        sc.zeta,
        mpmath.zeta,
        [Arg(1, 56, inclusive_a=False)],
        100,
        rtol=_ε,
    )


def test_zeta_greater_than_56():
    # For s >= 56, `zeta` just returns 1. Since `zeta` monotonically
    # decreases to 1, if we are exactly correct to double precision at
    # 56, then we should be exactly correct for all larger numbers.
    assert sc.zeta(56) == np.float64(mpmath.zeta(56))
