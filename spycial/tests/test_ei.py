import numpy as np
from numpy.testing import assert_allclose
import mpmath

import spycial as sc
from spycial.constants import _ε
from spycial.test_utilities import Arg, mpmath_allclose


def test_special_cases():
    res = sc.ei(-np.inf)
    assert res == 0
    assert np.signbit(res)
    assert sc.ei(0) == -np.inf
    assert sc.ei(np.inf) == np.inf


def test_negative_values():
    # For negative values `ei` is simply implemented using `e1`, which
    # is tested elsewhere. Just check a single value to make sure we
    # got the connection formula correct.
    assert_allclose(sc.ei(-1), float(mpmath.ei(-1)), atol=0, rtol=_ε)


def test_between_0_and_6():
    mpmath_allclose(
        sc.ei,
        mpmath.ei,
        [Arg(0, 6, inclusive_a=False)],
        50,
        rtol=4*_ε,
    )


def test_between_6_and_10():
    mpmath_allclose(
        sc.ei,
        mpmath.ei,
        [Arg(6, 10)],
        50,
        rtol=2*_ε,
    )


def test_between_10_and_20():
    mpmath_allclose(
        sc.ei,
        mpmath.ei,
        [Arg(10, 20)],
        50,
        rtol=2*_ε,
    )


def test_between_20_and_40():
    mpmath_allclose(
        sc.ei,
        mpmath.ei,
        [Arg(20, 40)],
        50,
        rtol=2*_ε,
    )


def test_greater_than_40():
    mpmath_allclose(
        sc.ei,
        mpmath.ei,
        [Arg(40, 720)],  # By 720 Ei has overflowed.
        200,
        rtol=2*_ε,
    )
