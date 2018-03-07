import numpy as np
from numpy.testing import assert_allclose
import scipy.special as scipy_sc
import scipy.special._ufuncs as scipy_sc_ufuncs

import special as sc


def test_sinpi_real():
    x = np.linspace(-100, 100)
    assert_allclose(sc.sinpi(x), scipy_sc_ufuncs._sinpi(x))


def test_sinpi_complex():
    x = np.linspace(-100, 100)
    x, y = np.meshgrid(x, x)
    z = x + 1j*y
    assert_allclose(sc.sinpi(z), scipy_sc_ufuncs._sinpi(z))


def test_cospi_real():
    x = np.linspace(-100, 100)
    assert_allclose(sc.cospi(x), scipy_sc_ufuncs._cospi(x))


def test_cospi_complex():
    x = np.linspace(-100, 100)
    x, y = np.meshgrid(x, x)
    z = x + 1j*y
    assert_allclose(sc.cospi(z), scipy_sc_ufuncs._cospi(z))
