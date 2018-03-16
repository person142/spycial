import numpy as np
import mpmath

import special as sc
from special.test_utilities import mpmath_allclose, Arg, ComplexArg


def test_loggamma_complex():
    def mpmath_loggamma(x):
        try:
            return mpmath.loggamma(x)
        except ValueError:
            # Hit a pole
            return complex(np.nan, np.nan)

    b = 1e200
    mpmath_allclose(sc.loggamma, mpmath_loggamma,
                    [ComplexArg(complex(-b, -b), complex(b, b))],
                     1000, 5e-12)


def test_loggamma_real():
    # For positive real values loggamma dispatches to lgamma, so we
    # don't need to check many points
    mpmath_allclose(sc.loggamma, mpmath.loggamma,
                    [Arg(0, np.inf, inclusive_a=False)], 50, 1e-13)

    assert all(np.isnan(sc.loggamma([0, -0.5])))


def test_lgamma():
    def mpmath_lgamma(x):
        try:
            return mpmath.loggamma(x).real
        except ValueError:
            # Hit a pole
            return np.inf

    mpmath_allclose(sc.lgamma, mpmath_lgamma,
                    [Arg()], 1000, 5e-14)
