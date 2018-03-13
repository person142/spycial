import mpmath

import special as sc
from special.test_utilities import mpmath_allclose, Arg, ComplexArg


def test_loggamma():
    def mpmath_loggamma(x):
        try:
            return mpmath.loggamma(x)
        except ValueError:
            # Hit a pole
            return np.nan

    b = 1e300
    mpmath_allclose(sc.loggamma, mpmath_loggamma,
                    [ComplexArg(complex(-b, -b), complex(b, b))],
                     1000, 1e-13)


def test_lgamma():
    x = np.linspace(-1000, 1000, 1500)
    assert_allclose(sc.lgamma(x), scipy_sc.gammaln(x))
