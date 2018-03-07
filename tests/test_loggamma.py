import numpy as np
from numpy.testing import assert_allclose

import special as sc
import scipy.special as scipy_sc


def test_loggamma():
    x = np.linspace(-1000, 1000)
    x, y = np.meshgrid(x, x)
    z = x + 1j*y
    assert_allclose(sc.loggamma(z), scipy_sc.loggamma(z))
