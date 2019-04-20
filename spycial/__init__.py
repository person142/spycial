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
   erfinv
   erfcinv

Exponential integral and related functions
------------------------------------------

.. autosummary::
   :toctree: generated

   e1
   ei
   en

Riemann zeta function
---------------------

.. autosummary::
   :toctree: generated

   zeta

"""
# Hack to avoid trapping floating point errors in ufuncs
from numpy import seterr
seterr(all='ignore')
del seterr

from .trig import sinpi, cospi
from .gamma import gamma
from .lgamma import lgamma, loggamma
from .digamma import digamma
from .erf import erf, erfc
from .erfinv import erfinv, erfcinv
from .zeta import zeta
from .ei import ei
from .e1 import e1
from .en import en
