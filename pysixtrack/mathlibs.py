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


def convert_attr(attr, dtype=np.float64):
    """Casts the attribute attr to type dtype

    Keyword arguments:
    attr -- attribute to be cast. scalar, string or iterable object assumed
    dtype -- type alias to be used for casting
       If attr is iterable, this is done for each element in the sequence

    Returns:
    attr after the conversion

    Throws:
    nothing

    """
    try:
        it_range = iter(attr)
    except TypeError:
        attr = dtype(attr)
    else:
        for it in it_range:
            it = dtype(it)
    return attr
