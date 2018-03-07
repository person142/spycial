import numpy as np
import scipy.special as scipy_sc

import special as sc
import scipy.special._ufuncs as scipy_sc_ufuncs


class Trig():

    def setup(self):
        x = np.linspace(-1000, 1000, 100)
        self.x = x
        x, y = np.meshgrid(x, x)
        self.z = x + 1j*y

    def time_sinpi_real(self):
        sc.sinpi(self.x)

    def time_scipy_sinpi_real(self):
        scipy_sc_ufuncs._sinpi(self.x)

    def time_cospi_real(self):
        sc.cospi(self.x)

    def time_scipy_cospi_real(self):
        scipy_sc_ufuncs._cospi(self.x)

    def time_sinpi_complex(self):
        sc.sinpi(self.z)

    def time_scipy_sinpi_complex(self):
        scipy_sc_ufuncs._sinpi(self.z)

    def time_cospi_complex(self):
        sc.cospi(self.z)

    def time_scipy_cospi_complex(self):
        scipy_sc_ufuncs._cospi(self.z)


class Loggamma():

    def setup(self):
        x = np.linspace(-1000, 1000, 100)
        x, y = np.meshgrid(x, x)
        self.z = x + 1j*y

    def time_loggamma(self):
        sc.loggamma(self.z)

    def time_scipy_loggamma(self):
        scipy_sc.loggamma(self.z)
