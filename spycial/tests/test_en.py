import numpy as np
from numpy.testing import assert_allclose
import mpmath

import spycial as sc
from spycial.test_utilities import (
    mpmath_allclose,
    Arg,
    UIntArg,
)
from spycial.constants import MINEXP, _ε


def test_special_cases():
    for n in range(2, 70):
        # Assuming IEEE arithmetic we should get the exact same
        # results here.
        assert sc.en(np.uint64(n), 0) == 1 / (n - 1)

    assert np.isnan(sc.en(np.uint64(5), np.nan))
    assert np.isnan(sc.en(np.uint64(5), -1))
    assert sc.en(np.uint64(5), np.inf) == 0


def test_n_equals_0():
    # n = 0 is special cased; make sure we got it right by
    # spot-checking a couple of values.
    assert sc.en(np.uint64(0), 0) == np.inf
    assert np.isnan(sc.en(np.uint64(0), -1))
    assert_allclose(
        sc.en(np.uint64(0), 3),
        float(mpmath.expint(0, 3)),
        atol=0,
        rtol=0,
    )


def test_n_equals_1():
    # n = 1 is special cased; make sure we got it right by
    # spot-checking a couple of values.
    assert sc.en(np.uint64(1), 0) == np.inf
    assert np.isnan(sc.en(np.uint64(1), -1))
    assert_allclose(
        sc.en(np.uint64(1), 3),
        float(mpmath.expint(1, 3)),
        atol=0,
        rtol=0,
    )


def test_x_less_than_one_half_and_n_less_than_50():
    mpmath_allclose(
        sc.en,
        mpmath.expint,
        [UIntArg(2, 50), Arg(0, 0.5, inclusive_a=False)],
        400,
        rtol=4*_ε,
    )


def test_large_n():
    mpmath_allclose(
        sc.en,
        mpmath.expint,
        # We don't cover the full range of x because mpmath takes to
        # long to return.
        [UIntArg(51, 200), Arg(0, -0.5 * MINEXP, inclusive_a=False)],
        400,
        rtol=1e-13,
        dps=160,
    )


def test_intermediate_x_and_n_less_than_16():
    mpmath_allclose(
        sc.en,
        mpmath.expint,
        [UIntArg(0, 15), Arg(0.5, 1.5)],
        100,
        rtol=4*_ε,
    )


def test_intermediate_x_and_n_between_16_and_50():
    mpmath_allclose(
        sc.en,
        mpmath.expint,
        [UIntArg(16, 50), Arg(0.5, 1.5)],
        100,
        rtol=4*_ε,
    )


def test_large_x():
    mpmath_allclose(
        sc.en,
        mpmath.expint,
        [UIntArg(0, 50), Arg(1.5, -MINEXP)],
        400,
        rtol=2e-15,
        # Mpmath hangs if you use lower precision.
        dps=160,
    )
