import numpy as np
from numba import jit, generated_jit, vectorize, types


eps = np.finfo('float64').eps


@vectorize('float64(float64)', nopython=True)
def _dsinpi(x):
    """Compute sin(pi*x) for real arguments."""
    p = np.ceil(x)
    hp = p/2

    # Make p the even integer closest to x
    if hp != np.ceil(hp):
        p -= 1
    # x is in (-1, 1]
    x -= p
    # Reflect x in (0.5, 1] to [0, 0.5).
    if x > 0.5:
        x = 1 - x
    # Reflect x in (-1, -0.5) to (-0.5, 0)
    if x < -0.5:
        x = -1 - x
    return np.sin(np.pi*x)


@jit('float64(float64)', nopython=True)
def _dcospi_taylor(x):
    """Taylor series for cos(pi*x) around x = 0.5. Since the root is
    exactly representable in double precision we get gains over just
    using cos(z) here.

    """
    x = np.pi*(x - 0.5)
    xx = x*x
    term = -x
    res = term
    for n in range(1, 20):
        term *= -xx/((2*n + 1)*(2*n))
        res += term
        if abs(term) <= eps*abs(res):
            break
    return res


@vectorize(['float64(float64)'], nopython=True)
def _dcospi(x):
    """Compute cos(pi*x) for real arguments."""
    p = np.ceil(x)
    hp = p/2

    # Make p the even integer closest to x
    if hp != np.ceil(hp):
        p -= 1
    # x is in (-1, 1].
    x -= p
    if abs(x - 0.5) < 0.2:
        return _dcospi_taylor(x)
    elif abs(x + 0.5) < 0.2:
        return _dcospi_taylor(-x)
    else:
        return np.cos(np.pi*x)


@vectorize('complex128(complex128)', nopython=True)
def _csinpi(z):
    """Compute sin(pi*z) for complex arguments."""
    x = z.real
    piy = np.pi*z.imag
    abspiy = abs(piy)
    sinpix = _dsinpi(x)
    cospix = _dcospi(x)

    if abspiy < 700:
        return np.complex(sinpix*np.cosh(piy), cospix*np.sinh(piy))

    # Have to be careful--sinh/cosh could overflow while cos/sin are
    # small. At this large of values
    #
    # cosh(y) ~ exp(y)/2
    # sinh(y) ~ sgn(y)*exp(y)/2
    #
    # so we can compute exp(y/2), scale by the right factor of sin/cos
    # and then multiply by exp(y/2) to avoid overflow.
    exphpiy = np.exp(abspiy/2)
    if exphpiy == np.inf:
        if sinpix == 0:
            # Preserve the sign of zero
            coshfac = np.copysign(0.0, sinpix)
        else:
            coshfac = np.copysign(np.inf, sinpix)
        if cospix == 0:
            sinhfac = np.copysign(0.0, cospix)
        else:
            sinhfac = np.copysign(np.inf, cospix)
        return np.complex(coshfac, sinhfac)

    coshfac = 0.5*sinpix*exphpiy
    sinhfac = 0.5*cospix*exphpiy
    return np.complex(coshfac*exphpiy, sinhfac*exphpiy)


@vectorize(['complex128(complex128)'], nopython=True)
def _ccospi(z):
    """Compute cos(pi*z) for complex arguments."""
    x = z.real
    piy = np.pi*z.imag
    abspiy = abs(piy)
    sinpix = _dsinpi(x)
    cospix = _dcospi(x)

    if abspiy < 700:
        return np.complex(cospix*np.cosh(piy), -sinpix*np.sinh(piy))

    # See csinpi(z) for an idea of what's going on here
    exphpiy = np.exp(abspiy/2)
    if exphpiy == np.inf:
        if sinpix == 0:
            coshfac = np.copysign(0.0, cospix)
        else:
            coshfac = np.copysign(np.inf, cospix)
        if cospix == 0:
            sinhfac = np.copysign(0.0, sinpix)
        else:
            sinhfac = np.copysign(np.inf, sinpix)
        return np.complex(coshfac, sinhfac)

    coshfac = 0.5*cospix*exphpiy
    sinhfac = 0.5*sinpix*exphpiy
    return np.complex(coshfac*exphpiy, sinhfac*exphpiy)


@generated_jit(nopython=True)
def sinpi(arr):
    if arr.dtype == types.float64:
        return lambda arr: _dsinpi(arr)
    elif arr.dtype == types.complex128:
        return lambda arr: _csinpi(arr)
    else:
        raise NotImplementedError


@generated_jit(nopython=True)
def cospi(arr):
    if arr.dtype == types.float64:
        return lambda arr: _dcospi(arr)
    elif arr.dtype == types.complex128:
        return lambda arr: _ccospi(arr)
    else:
        raise NotImplementedError
