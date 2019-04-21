import numpy as np
import scipy.special as scipy_sc

import spycial as sc
import scipy.special._ufuncs as scipy_sc_ufuncs


class Trig:
    params = [
        ('cospi', 'sinpi'),
        ('real', 'complex'),
        ('SciPy', 'Spycial'),
    ]
    param_names = ['Function', 'Type', 'Library']

    def setup(self, f, typ, library):
        x = np.linspace(-1000, 1000, 100)
        if typ == 'real':
            self.x = x
        else:
            x, y = np.meshgrid(x, x)
            self.x = x + 1j*y

        if f == 'cospi' and library == 'SciPy':
            self.f = scipy_sc_ufuncs._cospi
        elif f == 'cospi' and library == 'Spycial':
            self.f = sc.cospi
        elif f == 'sinpi' and library == 'SciPy':
            self.f = scipy_sc_ufuncs._sinpi
        else:
            self.f = sc.sinpi

    def time_trig(self, f, typ, library):
        self.f(self.x)


class Loggamma:
    params = [('gamma', 'lgamma', 'loggamma'), ('SciPy', 'Spycial')]
    param_names = ['Function', 'Library']

    def setup(self, name, library):
        x = np.linspace(-1000, 1000, 100)
        if name == 'gamma':
            self.x = x
            if library == 'SciPy':
                self.f = scipy_sc.gamma
            else:
                self.f = sc.gamma
        elif name == 'lgamma':
            self.x = x
            if library == 'SciPy':
                self.f = scipy_sc.gammaln
            else:
                self.f = sc.lgamma
        else:
            x, y = np.meshgrid(x, x)
            self.x = x + 1j*y
            if library == 'SciPy':
                self.f = scipy_sc.loggamma
            else:
                self.f = sc.loggamma

    def time_loggamma(self, name, library):
        self.f(self.x)


class Erf:
    params = [('erf', 'erfc'), ('SciPy', 'Spycial')]
    param_names = ['Function', 'Library']

    def setup(self, name, library):
        self.x = np.linspace(-20, 20, 100)
        if name == 'erf':
            if library == 'SciPy':
                self.f = scipy_sc.erf
            else:
                self.f = sc.erf
        else:
            if library == 'SciPy':
                self.f = scipy_sc.erfc
            else:
                self.f = sc.erfc

    def time_erf(self, name, library):
        self.f(self.x)


class Erfinv:
    params = [('erfinv',), ('SciPy', 'Spycial')]
    param_names = ['Function', 'Library']

    def setup(self, name, library):
        self.x = np.linspace(-1, 1, 500)
        if library == 'SciPy':
            self.f = scipy_sc.erfinv
        else:
            self.f = sc.erfinv

    def time_erfinv(self, name, library):
        self.f(self.x)


class Zeta:
    params = [('zeta',), ('SciPy', 'Spycial')]
    param_names = ['Function', 'Library']

    def setup(self, name, library):
        self.x = np.linspace(0, 56, 200)
        if library == 'SciPy':
            self.f = scipy_sc.zeta
        else:
            self.f = sc.zeta

    def time_zeta(self, name, library):
        self.f(self.x)


class ExponentialIntegrals:
    params = [('e1', 'ei'), ('SciPy', 'Spycial')]
    param_names = ['Function', 'Library']

    def setup(self, name, library):
        if name == 'e1':
            self.x = np.linspace(1e-8, 750, 100)
            if library == 'SciPy':
                self.f = scipy_sc.exp1
            else:
                self.f = sc.e1
        else:
            self.x = np.linspace(-700, 700, 200)
            if library == 'SciPy':
                self.f = scipy_sc.expi
            else:
                self.f = sc.ei

    def time_exponential_integrals(self, name, library):
        self.f(self.x)


class GeneralizedExponentialIntegral:
    params = [
        (
            'small_x_and_n_less_than_51',
            'intermediate_x_and_n_between_2_and_15',
            'intermediate_x_and_n_between_16_and_50',
            'large_n',
            'large_x',
        ),
        ('SciPy', 'Spycial'),
    ]
    param_names = ['Parameter Region', 'Library']

    def setup(self, parameter_range, library):
        if library == 'SciPy':
            self.f = scipy_sc.expn
            # Use int64 to make sure we aren't spending time casting
            # the input type.
            dtype = np.int64
        else:
            self.f = sc.en
            dtype = np.uint64

        if parameter_range == 'small_x_and_n_less_than_51':
            # Start at n = 2 to exclude the special-cased n = 0 and
            # n = 1 cases.
            n = np.arange(2, 51, dtype=dtype)
            x = np.linspace(0, 1, 200)
        elif parameter_range == 'intermediate_x_and_n_between_2_and_15':
            n = np.arange(2, 16, dtype=dtype)
            x = np.linspace(0.5, 1.5, 200)
        elif parameter_range == 'intermediate_x_and_n_between_16_and_50':
            n = np.arange(16, 51, dtype=dtype)
            x = np.linspace(0.5, 1.5, 200)
        elif parameter_range == 'large_n':
            n = np.arange(51, 100, dtype=dtype)
            x = np.linspace(0, 500, 200)
        else:
            n = np.arange(2, 50, dtype=dtype)
            x = np.linspace(1, 500, 200)
        self.args = np.meshgrid(n, x)

    def time_generalized_exponential_integral(self, name, library):
        self.f(*self.args)
