import numpy as np
import mpmath

import special as sc
from special.test_utilities import Arg, mpmath_allclose


def test_gamma():
    def mpmath_gamma(x):
        try:
            return mpmath.gamma(x)
        except ValueError:
            # Gamma pole
            return np.nan

    # Gamma overflows around 170 so there's no point in testing beyond
    # that.
    mpmath_allclose(sc.gamma, mpmath_gamma,
                    [Arg(-np.inf, 180)], 1000, 1e-14)
