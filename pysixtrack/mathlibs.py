from importlib import util
import numpy as np
from scipy.special import wofz


class MathlibDefault(object):
    from numpy import sqrt, exp, sin, cos, abs, pi, tan, sign
    real_type = property(lambda: np.float_)
    int_type = property(lambda: np.int64)

    clight = property(lambda: np.float_(299792458.0))
    echarge = property(lambda: np.float_(1.602176565e-19))
    emass = property(lambda: np.float_(0.510998928e6))
    nmass = property(lambda: np.float_(931.49410242e6))
    pmass = property(lambda: np.float_(938.272081e6))
    anumber = property(lambda: np.float_(6.02214129e23))
    kbolz = property(lambda: np.float_(1.3806488e-23))
    epsilon0 = property(lambda: np.float_(8.854187817e-12))
    mu0 = property(lambda: np.float(4e-7) * np.pi)

    @property
    def eradius(self):
        return self.echarge * self.echarge / \
            (4 * self.pi * self.epsilon0 * self.emass * self.clight * self.clight)

    @property
    def pradius(self):
        return self.echarge * self.echarge / \
            (4 * self.pi * self.epsilon0 * self.pmass * self.clight * self.clight)

    @staticmethod
    def wfun(z_re, z_im):
        w = wofz(z_re + 1j * z_im)
        return w.real, w.imag


if not util.find_spec("mpmath") is None:
    import mpmath

    class MpMathLib(object):
        from mpmath import sqrt, exp, sin, cos, tan, fabs as abs, pi, sign, erfc
        real_type = property(lambda: mpmath.mpf)
        int_type = property(lambda: np.int64)

        clight = property(lambda: mpmath.mpf("299792458.0"))
        echarge = property(lambda: mpmath.mpf("1.602176565e-19"))
        emass = property(lambda: mpmath.mpf("0.510998928e6"))
        nmass = property(lambda: mpmath.mpf("931.49410242e6"))
        pmass = property(lambda: mpmath.mpf("938.272081e6"))
        anumber = property(lambda: mpmath.mpf("6.02214129e23"))
        kbolz = property(lambda: mpmath.mpf("1.3806488e-23"))
        epsilon0 = property(lambda: mpmath.mpf("8.854187817e-12"))
        mu0 = property(lambda: mpmath.mpf("4e-7") * mpmath.pi)

        @property
        def eradius(self):
            return self.echarge * self.echarge / \
                (4 * self.pi * self.epsilon0 * self.emass * self.clight * self.clight)

        @property
        def pradius(self):
            return self.echarge * self.echarge / \
                (4 * self.pi * self.epsilon0 * self.pmass * self.clight * self.clight)

        @staticmethod
        def wfun(z_re, z_im):
            z = mpmath.mpc(z_re, z_im)
            w = mpmath.exp(-z * z) * mpmath.erfc(mpmath.mpc(z.imag, -z.real))
            return w.real, w.imag


else:
    class MpMathLib(object):
        pass
