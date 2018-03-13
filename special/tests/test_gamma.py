import numpy as np
from numpy.testing import assert_allclose

import special as sc
import scipy.special as scipy_sc


def test_loggamma():
    x = np.linspace(-1000, 1000)
    x, y = np.meshgrid(x, x)
    z = x + 1j*y
    assert_allclose(sc.loggamma(z), scipy_sc.loggamma(z))


def test_lgamma():
    x = np.linspace(-1000, 1000, 1500)
    assert_allclose(sc.lgamma(x), scipy_sc.gammaln(x))
