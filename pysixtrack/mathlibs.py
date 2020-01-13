import numpy as np
from importlib import util
import types
from scipy.special import wofz

class MathlibDefault(object):

    from numpy import sqrt, exp, sin, cos, abs, pi, tan

    @classmethod
    def wfun(cls, z_re, z_im):
        w = wofz(z_re + 1j * z_im)
        return w.real, w.imag


class MathlibConstants(object):
    @staticmethod
    def REAL_TYPE(mathlib=MathlibDefault):
        if not( mpmath_spec is None ) and \
            isinstance( mathlib, types.ModuleType ) and \
            mathlib.__name__ == 'mpmath':
            from mpmath import mp
            return mp.mpf
        else:
            return np.float64

    @staticmethod
    def INT_TYPE(mathlib=MathlibDefault):
        return np.int64


    def __init__(self, mathlib=MathlibDefault,
            real_type=np.float64, int_type=np.int64 ):
        self.CLIGHT_STR = "299792458.0"
        self.ECHARGE_STR = "1.602176565e-19"
        self.EMASS_STR = "0.510998928e6"
        self.NMASS_STR = "931.49410242e6"
        self.PMASS_STR = "938.272081e6"
        self.EPSILON0_STR = "8.854187817e-12"
        self.ANUMBER_STR = "6.02214129e23"
        self.KBOLZ_STR = "1.3806488e-23"

        mpmath_spec = util.find_spec("mpmath")

        if mpmath_spec is not None and \
            isinstance( mathlib, types.ModuleType ) and \
            mathlib.__name__ == 'mpmath':
            from mpmath import mp, pi as mp_pi
            real_type = mp.mpf
            self.PI = mp_pi
        else:
            self.PI = np.pi

        self.ZERO     = real_type("0.0")
        self.ONE      = real_type("1.0")
        self.TWO      = real_type("2.0")
        self.ONE_HALF = real_type("0.5")
        self.CLIGHT   = real_type(self.CLIGHT_STR)
        self.ECHARGE  = real_type(self.ECHARGE_STR)
        self.EMASS    = real_type(self.EMASS_STR)
        self.NMASS    = real_type(self.NMASS_STR)
        self.PMASS    = real_type(self.PMASS_STR)
        self.EPSILON0 = real_type(self.EPSILON0_STR)
        self.ANUMBER  = real_type(self.ANUMBER_STR)
        self.KBOLZ    = real_type(self.KBOLZ_STR)

        self._m = mathlib
        self._real_type = real_type
        self._int_type = int_type

    @property
    def real_type(self):
        return self._real_type

    @property
    def int_type(self):
        return self._int_type

    @property
    def m(self):
        return self._m

    @property
    def MU0(self):
        return self._real_type("4e-7") * self.PI

    @property
    def ERADIUS(self):
        return self.ECHARGE * self.ECHARGE / ( self._real_type( 4 ) *
            self.PI * self.EPSILON0 * self.EMASS * self.CLIGHT * self.CLIGHT )

    @property
    def PRADIUS(self):
        return self.ECHARGE * self.ECHARGE / ( self._real_type( 4 ) *
            self.PI * self.EPSILON0 * self.PMASS * self.CLIGHT * self.CLIGHT )

    def sign(self, val):
         return self.TWO * self._real_type( self._real_type(val) >= self.ZERO ) - self.ONE
