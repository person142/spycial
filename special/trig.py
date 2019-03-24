from numba import njit, generated_jit, vectorize, types
import numpy as np

from . import settings
from .constants import _MAXCOSH


@njit('float64(float64)', cache=settings.CACHE)
def _dsinpi(x):
    """Compute sin(pi*x) for real arguments."""
    s = 1.0
    if x < 0.0:
        x = -x
        s = -1.0

    r = np.fmod(x, 2.0)
    if r < 0.5:
        return s*np.sin(np.pi*r)
    elif r > 1.5:
        return s*np.sin(np.pi*(r - 2.0))
    else:
        return -s*np.sin(np.pi*(r - 1.0))


@njit('float64(float64)', cache=settings.CACHE)
def _dcospi(x):
    """Compute cos(pi*x) for real arguments."""
    if x < 0.0:
        x = -x

    r = np.fmod(x, 2.0)
    if r == 0.5:
        # Don't want to return -0.0
        return 0.0
    elif r < 1.0:
        return -np.sin(np.pi*(r - 0.5))
    else:
        return np.sin(np.pi*(r - 1.5))


@njit('complex128(complex128)', cache=settings.CACHE)
def _csinpi(z):
    """Compute sin(pi*z) for complex arguments."""
    x = z.real
    piy = np.pi*z.imag
    abspiy = abs(piy)
    sinpix = _dsinpi(x)
    cospix = _dcospi(x)

    if abspiy <= _MAXCOSH:
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
            coshfac = sinpix
        else:
            coshfac = np.copysign(np.inf, sinpix)
        if cospix == 0:
            sinhfac = cospix
        else:
            sinhfac = np.sign(piy)*np.copysign(np.inf, cospix)
        return np.complex(coshfac, sinhfac)

    coshfac = 0.5*sinpix*exphpiy
    sinhfac = 0.5*cospix*exphpiy
    return np.complex(coshfac*exphpiy, sinhfac*exphpiy)


@njit('complex128(complex128)', cache=settings.CACHE)
def _ccospi(z):
    """Compute cos(pi*z) for complex arguments."""
    x = z.real
    piy = np.pi*z.imag
    abspiy = abs(piy)
    sinpix = _dsinpi(x)
    cospix = _dcospi(x)

    if abspiy <= _MAXCOSH:
        return np.complex(cospix*np.cosh(piy), -sinpix*np.sinh(piy))

    # See csinpi(z) for an idea of what's going on here
    exphpiy = np.exp(abspiy/2)
    if exphpiy == np.inf:
        if cospix == 0:
            coshfac = cospix
        else:
            coshfac = np.copysign(np.inf, cospix)
        if sinpix == 0:
            sinhfac = sinpix
        else:
            sinhfac = -np.sign(piy)*np.copysign(np.inf, sinpix)
        return np.complex(coshfac, sinhfac)

    coshfac = 0.5*cospix*exphpiy
    sinhfac = -0.5*sinpix*exphpiy
    return np.complex(coshfac*exphpiy, sinhfac*exphpiy)


@generated_jit(nopython=True, cache=settings.CACHE)
def _sinpi(a):
    if a == types.float64:
        return lambda a: _dsinpi(a)
    elif a == types.complex128:
        return lambda a: _csinpi(a)
    else:
        raise NotImplementedError


@vectorize(
    ['float64(float64)', 'complex128(complex128)'],
    nopython=True,
    cache=settings.CACHE,
)
def sinpi(x):
    r"""Compute :math:`\sin(\pi x)`.

    Parameters
    ----------
    x : array-like
        Points on the real line or complex plane
    out : ndarray, optional
        Output array for the values of `sinpi` at `x`

    Returns
    -------
    ndarray
        Values of `sinpi` at `x`

    """
    return _sinpi(x)


@generated_jit(nopython=True, cache=settings.CACHE)
def _cospi(a):
    if a == types.float64:
        return lambda a: _dcospi(a)
    elif a == types.complex128:
        return lambda a: _ccospi(a)
    else:
        raise NotImplementedError


@vectorize(
    ['float64(float64)', 'complex128(complex128)'],
    nopython=True,
    cache=settings.CACHE,
)
def cospi(x):
    r"""Compute :math:`\cos(\pi x)`.

    Parameters
    ----------
    x : array-like
        Points on the real line or complex plane
    out : ndarray, optional
        Output array for the values of `cospi` at `x`

    Returns
    -------
    ndarray
        Values of `cospi` at `x`

    """
    return _cospi(x)
