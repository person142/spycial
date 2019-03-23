"""
Available special functions
===========================

Gamma and related functions
---------------------------

.. autosummary::
   :toctree: generated

   gamma
   lgamma
   loggamma
   digamma

Trigonometric functions
-----------------------

.. autosummary::
   :toctree: generated

   sinpi
   cospi

Error function and related functions
------------------------------------

.. autosummary::
   :toctree: generated

   erf
   erfc

"""
# Hack to avoid trapping floating point errors in ufuncs
from numpy import seterr
seterr(all='ignore')
del(seterr)

from .trig import sinpi, cospi
from .gamma import gamma
from .lgamma import lgamma, loggamma
from .digamma import digamma
from .erf import erf, erfc
from .zeta import zeta
