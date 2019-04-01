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

Exponential integral and related functions
------------------------------------------

.. autosummary::
   :toctree: generated

   e1
   ei

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
from .zeta import zeta
from .e1 import e1
from .ei import ei