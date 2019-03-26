import numpy as np
import mpmath

import spycial as sc
from spycial.test_utilities import mpmath_allclose, Arg, ComplexArg


def test_sinpi_real():
    rtol = np.finfo(float).eps
    mpmath_allclose(sc.sinpi, mpmath.sinpi, [Arg()], 1000, rtol)


def test_sinpi_complex():
    rtol = 16*np.finfo(float).eps
    mpmath_allclose(sc.sinpi, mpmath.sinpi, [ComplexArg()], 1000, rtol)


def test_cospi_real():
    rtol = np.finfo(float).eps
    mpmath_allclose(sc.cospi, mpmath.cospi, [Arg()], 1000, rtol)


def test_cospi_complex():
    rtol = 16*np.finfo(float).eps
    mpmath_allclose(sc.cospi, mpmath.cospi, [ComplexArg()], 1000, rtol)


def test_zeros():
    y = sc.sinpi(-0.0)
    assert y == 0.0
    assert np.signbit(y)

    y = sc.sinpi(0.0)
    assert y == 0.0
    assert not np.signbit(y)

    y = sc.cospi(0.5)
    assert y == 0.0
    assert not np.signbit(y)
