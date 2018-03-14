import numpy as np
import mpmath

import special as sc
from special.test_utilities import mpmath_allclose, Arg, ComplexArg


def test_sinpi_real():
    rtol = np.finfo(float).eps
    mpmath_allclose(sc.sinpi, mpmath.sinpi, [Arg()], 1000, rtol)


def test_sinpi_complex():
    a, b = np.inf, 100
    mpmath_allclose(sc.sinpi, mpmath.sinpi,
                    [ComplexArg(complex(-a, -b), complex(a, b))],
                    1000, 1e-13)


def test_cospi_real():
    rtol = 4*np.finfo(float).eps
    mpmath_allclose(sc.cospi, mpmath.cospi, [Arg()], 1000, rtol)


def test_cospi_complex():
    a, b = np.inf, 100
    mpmath_allclose(sc.cospi, mpmath.cospi,
                    [ComplexArg(complex(-a, -b), complex(a, b))],
                    1000, 1e-13)
