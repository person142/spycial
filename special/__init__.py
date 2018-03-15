"""
Available special functions
===========================

Gamma and related functions
---------------------------

.. autosummary::
   :toctree: generated

   lgamma
   loggamma

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
from .gamma import lgamma, loggamma
from .erf import erf, erfc
