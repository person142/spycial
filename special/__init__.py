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

"""
# Hack to avoid trapping floating point errors in ufuncs
from numpy import seterr
seterr(all='ignore')
del(seterr)

from .trig import sinpi, cospi
from .gamma import lgamma, loggamma
