import numpy as np
import scipy.special as scipy_sc

import special as sc
import scipy.special._ufuncs as scipy_sc_ufuncs


class Trig():
    params = [('cospi', 'sinpi'), ('real', 'complex'),
              ('SciPy', 'Numba')]
    param_names = ['Function', 'Type', 'API']

    def setup(self, f, typ, api):
        x = np.linspace(-1000, 1000, 100)
        if typ == 'real':
            self.x = x
        else:
            x, y = np.meshgrid(x, x)
            self.x = x + 1j*y

        if f == 'cospi' and api == 'SciPy':
            self.f = scipy_sc_ufuncs._cospi
        elif f == 'cospi' and api == 'Numba':
            self.f = sc.cospi
        elif f == 'sinpi' and api == 'SciPy':
            self.f = scipy_sc_ufuncs._sinpi
        else:
            self.f = sc.sinpi

    def time_trig(self, f, typ, api):
        self.f(self.x)


class Loggamma():
    params = [('lgamma', 'loggamma'), ('SciPy', 'Numba')]
    param_names = ['Function', 'API']

    def setup(self, name, api):
        x = np.linspace(-1000, 1000, 100)
        if name == 'lgamma':
            self.x = x
            if api == 'SciPy':
                self.f = scipy_sc.gammaln
            else:
                self.f = sc.lgamma
        elif name == 'loggamma':
            x, y = np.meshgrid(x, x)
            self.x = x + 1j*y
            if api == 'SciPy':
                self.f = scipy_sc.loggamma
            else:
                self.f = sc.loggamma

    def time_loggamma(self, name, api):
        self.f(self.x)
