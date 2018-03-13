"""Utilties to simplify testing from Mpmath. They were adapted from
SciPy.

"""
import numpy as np
import mpmath


def only_some(func, x, y):
    isx = func(x)
    isy = func(y)
    if isx and isy:
        return False
    if not isx and not isy:
        return False
    else:
        return True


def func_allclose(x, f, g, rtol, atol):
    """Check whether the results of two functions are equal within a
    desired tolerance. NaNs are considered equal.

    """
    badpts = []
    for x0, f0, g0 in np.nditer([x, f, g]):
        if only_some(np.isnan, f0, g0):
            badpts.append((x0, f0, g0, np.nan, np.nan))
        elif (only_some(np.isposinf, np.real(f0), np.real(g0)) or
              only_some(np.isposinf, np.imag(f0), np.imag(g0)) or
              only_some(np.isneginf, np.real(f0), np.real(g0)) or
              only_some(np.isneginf, np.imag(f0), np.imag(g0))):
            badpts.append((x0, f0, g0, np.inf, np.inf))

        abserr = np.abs(f0 - g0)
        if abserr > rtol*np.abs(g0) or abserr > atol:
            if g0 == 0.0:
                relerr = np.inf
            else:
                relerr = abserr/abs(g0)
            badpts.append((x0, f0, g0, relerr, abserr))

    if len(badpts) > 0:
        msg = ["\n"]
        template = "At {}: {} != {}, relerr = {}, abserr = {}"
        for x0, f0, g0, relerr, abserr in badpts:
            msg.append(template.format(x0, f0, g0, relerr, abserr))
        raise ValueError("\n".join(msg))


class Arg():
    """Generate a set of test points on the real axis."""

    def __init__(self, a=-np.inf, b=np.inf, inclusive_a=True, inclusive_b=True):
        if a > b:
            raise ValueError("a should be less than or equal to b")
        if a == -np.inf:
            a = -0.5*np.finfo(float).max
        if b == np.inf:
            b = 0.5*np.finfo(float).max
        self.a, self.b = a, b

        self.inclusive_a, self.inclusive_b = inclusive_a, inclusive_b

    def _positive_values(self, a, b, n):
        if a < 0:
            raise ValueError("a should be positive")

        # Try to put half of the points into a linspace between a and
        # 10 the other half in a logspace.
        if n % 2 == 0:
            nlogpts = n//2
            nlinpts = nlogpts
        else:
            nlogpts = n//2
            nlinpts = nlogpts + 1

        if a >= 10:
            # Outside of linspace range; just return a logspace.
            pts = np.logspace(np.log10(a), np.log10(b), n)
        elif a > 0 and b < 10:
            # Outside of logspace range; just return a linspace
            pts = np.linspace(a, b, n)
        elif a > 0:
            # Linspace between a and 10 and a logspace between 10 and
            # b.
            linpts = np.linspace(a, 10, nlinpts, endpoint=False)
            logpts = np.logspace(1, np.log10(b), nlogpts)
            pts = np.hstack((linpts, logpts))
        elif a == 0 and b <= 10:
            # Linspace between 0 and b and a logspace between 0 and
            # the smallest positive point of the linspace
            linpts = np.linspace(0, b, nlinpts)
            if linpts.size > 1:
                right = np.log10(linpts[1])
            else:
                right = -30
            logpts = np.logspace(-30, right, nlogpts, endpoint=False)
            pts = np.hstack((logpts, linpts))
        else:
            # Linspace between 0 and 10, logspace between 0 and the
            # smallest positive point of the linspace, and a logspace
            # between 10 and b.
            if nlogpts % 2 == 0:
                nlogpts1 = nlogpts//2
                nlogpts2 = nlogpts1
            else:
                nlogpts1 = nlogpts//2
                nlogpts2 = nlogpts1 + 1
            linpts = np.linspace(0, 10, nlinpts, endpoint=False)
            if linpts.size > 1:
                right = np.log10(linpts[1])
            else:
                right = -30
            logpts1 = np.logspace(-30, right, nlogpts1, endpoint=False)
            logpts2 = np.logspace(1, np.log10(b), nlogpts2)
            pts = np.hstack((logpts1, linpts, logpts2))

        return np.sort(pts)

    def values(self, n):
        """Return an array containing n numbers."""
        a, b = self.a, self.b
        if a == b:
            return np.zeros(n)

        if not self.inclusive_a:
            n += 1
        if not self.inclusive_b:
            n += 1

        if n % 2 == 0:
            n1 = n//2
            n2 = n1
        else:
            n1 = n//2
            n2 = n1 + 1

        if a >= 0:
            pospts = self._positive_values(a, b, n)
            negpts = []
        elif b <= 0:
            pospts = []
            negpts = -self._positive_values(-b, -a, n)
        else:
            pospts = self._positive_values(0, b, n1)
            negpts = -self._positive_values(0, -a, n2 + 1)
            # Don't want to get zero twice
            negpts = negpts[1:]
        pts = np.hstack((negpts[::-1], pospts))

        if not self.inclusive_a:
            pts = pts[1:]
        if not self.inclusive_b:
            pts = pts[:-1]
        return pts


class ComplexArg():
    """Generate a set of test points in the complex plane."""

    def __init__(self, a=complex(-np.inf, -np.inf), b=complex(np.inf, np.inf)):
        self.real = Arg(a.real, b.real)
        self.imag = Arg(a.imag, b.imag)

    def values(self, n):
        m = int(np.floor(np.sqrt(n)))
        x = self.real.values(m)
        y = self.imag.values(m + 1)
        x, y = np.meshgrid(x, y)
        z = x + 1j*y
        return z.flatten()


def getargs(argspec, n):
    nargs = len(argspec)
    ms = np.asarray([1.5 if isinstance(spec, ComplexArg) else 1.0 for spec in argspec])
    ms = (n**(ms/sum(ms))).astype(int) + 1

    args = []
    for spec, m in zip(argspec, ms):
        args.append(spec.values(m))
    args = np.meshgrid(args)
    return args


def mpmath_allclose(func, mpmath_func, argspec, n, rtol, dps=None):
    if dps is None:
        dps = 20
    if any([isinstance(arg, ComplexArg) for arg in argspec]):
        cast = lambda x: complex(x)
    else:
        cast = lambda x: float(x)

    def vec_mpmath_func(x):
        with mpmath.workdps(dps):
            return cast(mpmath_func(x))

    vec_mpmath_func = np.vectorize(vec_mpmath_func)

    argarr = getargs(argspec, n)
    f, g = func(argarr), vec_mpmath_func(argarr)
    func_allclose(argarr, f, g, rtol, np.inf)
