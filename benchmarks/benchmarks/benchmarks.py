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


class E1:
    params = [('e1',), ('SciPy', 'Spycial')]
    param_names = ['Function', 'Library']

    def setup(self, name, library):
        self.x = np.linspace(1e-8, 750, 100)
        if library == 'SciPy':
            self.f = scipy_sc.exp1
        else:
            self.f = sc.e1

    def time_e1(self, name, library):
        self.f(self.x)
