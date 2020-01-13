import numpy as np
from .mathlibs import MathlibDefault, MathlibConstants
from importlib import util
import types

mpmath_spec = util.find_spec("mpmath")
if mpmath_spec is not None:
    from mpmath import mp
    from mpmath.ctx_mp import MPContext as mpmath_ctx

def count_not_none(*lst):
    return len(lst) - sum(p is None for p in lst)

def _find_attr_length(sep=',', **kwargs):
    length = None
    has_processed_any=False

    for key, value in kwargs.items():
        ll = None
        if value is None:
            continue
        elif isinstance(value, str):
            ll = sep is not None and sep in value and len( value.split(sep) ) or 1
        else:
            try:
                it = iter(value)
            except TypeError:
                ll = 1
            else:
                ll = sum( 1 for _ in iter )

        if not( ll is None ):
            if length is None:
                length = ll
            elif length == ll:
                continue
            elif length == 1 and ll > 1:
                length = ll
            elif length > 1 and ll == 1:
                continue
            else:
                assert length != ll
                length = None
                raise ValueError("inconsistent argument lengths provided")

    if length is None and len(kwargs):
        length = 1

    return length

def _make_attr(value, length, dtype=np.float64, sep=',', default=0, as_vector=True):
    assert not( dtype is None )
    ret_val = None
    if value is None:
        ret_val = as_vector and np.array( [ dtype(default) ] ) or dtype(default)
    elif isinstance(value, str):
        if sep is not None and sep in value:
            ret_val = as_vector and np.fromstring(value, sep=sep, count=length) \
                or dtype(value.split(sep)[ 0 ])
        else:
            value = dtype(value)
            if as_vector:
                ret_val = np.zeros(length, dtype=dtype)
                ret_val[:] = value
            else:
                ret_val = value
    else:
        try:
            it = iter(value)
        except TypeError:
            if as_vector:
                ret_val = np.zeros(length)
                ret_val[:] = dtype(value)
            else:
                ret_val = dtype(value)
        else:
            ret_val = as_vector and np.fromiter(it, count=length, dtype=dtype) \
                or np.fromiter(it, count=length, dtype=dtype)[ 0 ]
    return ret_val


class Particles(object):
    """
    Coordinates:

    **fields**

    **properties

    s       [m]  Reference accumulated pathlength
    x       [m]  Horizontal offset
    px      [1]  Px / (m/m0 * p0c) = beta_x gamma /(beta0 gamma0)
    y       [m   Vertical offset]
    py      [1]  Py / (m/m0 * p0c)
    delta   [1]  Pc / (m/m0 * p0c) - 1
    ptau    [1]  Energy / (m/m0 * p0c) - 1
    psigma  [1]  ptau/beta0
    rvv     [1]  beta/beta0
    rpp     [1]  1/(1+delta) = (m/m0 * p0c) / Pc
    zeta    [m]  beta (s/beta0 - ct )
    tau     [m]
    sigma   [m]  s - beta0 ct = rvv * zeta
    mass0   [eV]
    q0      [e]  reference carge
    p0c     [eV] reference momentum
    energy0 [eV] refernece energy
    gamma0  [1]  reference relativistic gamma
    beta0   [1]  reference relativistix beta
    chi     [1]  q/ q0 * m0/m = qratio / mratio
    mratio  [1]  mass/mass0
    qratio  [1]  q / q0
    partid  int
    turn    int
    state   int
    elemid  int
    """

    def __init_mathlib(self, mathlib=MathlibDefault,
            real_t=np.float64, int_t=np.int64 ):
        self.CONSTANTS = MathlibConstants(mathlib=mathlib, real_type=real_t, int_type=int_t)
        self._m = self.CONSTANTS.m


    def _g1(self, mass0, p0c, energy0):
        real_t = self.CONSTANTS.real_type
        length = self._arg_length
        is_vec = self._is_vector

        p0c = _make_attr(p0c, length, dtype=real_t, as_vector=is_vec)
        energy0 = _make_attr(energy0, length, dtype=real_t, as_vector=is_vec)
        mass0 = _make_attr(mass0, length, dtype=real_t, default=self.CONSTANTS.PMASS, as_vector=is_vec)
        beta0 = p0c / energy0
        gamma0 = energy0 / mass0
        return mass0, beta0, gamma0, p0c, energy0

    def _g2(self, mass0, beta0, gamma0):
        real_t = self.CONSTANTS.real_type
        length = self._arg_length
        is_vec = self._is_vector

        mass0 = _make_attr(mass0, length, dtype=real_t, default=self.CONSTANTS.PMASS, as_vector=is_vec)
        beta0 = _make_attr(beta0, length, dtype=real_t, as_vector=is_vec)
        gamma0 = _make_attr(gamma0, length, dtype=real_t, as_vector=is_vec)
        energy0 = mass0 * gamma0
        p0c = energy0 * beta0
        return mass0, beta0, gamma0, p0c, energy0

    def _f1(self, mass0, p0c):
        sqrt = self._m.sqrt
        real_t = self.CONSTANTS.real_type
        length = self._arg_length
        is_vec = self._is_vector

        mass0 = _make_attr(mass0, length, dtype=real_t, default=self.CONSTANTS.PMASS, as_vector=is_vec)
        p0c = _make_attr(p0c, length, dtype=real_t, as_vector=is_vec)
        energy0 = sqrt(p0c * p0c + mass0 * mass0)
        return self._g1(mass0, p0c, energy0)

    def _f2(self, mass0, energy0):
        sqrt = self._m.sqrt
        real_t = self.CONSTANTS.real_type
        length = self._arg_length
        is_vec = self._is_vector

        mass0 = _make_attr(mass0, length, dtype=real_t, default=self.CONSTANTS.PMASS, as_vector=is_vec)
        energy0 = _make_attr(energy0, length, dtype=real_t, as_vector=is_vec)
        p0c = sqrt(energy0 * energy0 - mass0 * mass0)
        return self._g1(mass0, p0c, energy0)

    def _f3(self, mass0, beta0):
        sqrt = self._m.sqrt
        real_t = self.CONSTANTS.real_type
        length = self._arg_length
        is_vec = self._is_vector

        gamma0 = self._one / sqrt( self._one - beta0 * beta0)
        return self._g2(mass0, beta0, gamma0)

    def _f4(self, mass0, gamma0):
        sqrt = self._m.sqrt
        real_t = self.CONSTANTS.real_type
        length = self._arg_length
        is_vec = self._is_vector

        mass0 = _make_attr(mass0, length, dtype=real_t, default=self.CONSTANTS.PMASS, as_vector=is_vec)
        gamma0 = _make_attr(gamma0, length, dtype=real_t, as_vector=is_vec)
        beta0 = sqrt(self._one - self._one / gamma0 * gamma0)
        return self._g2(mass0, beta0, gamma0)

    def copy(self, index=None):
        p = Particles()
        for k, v in list(self.__dict__.items()):
            if type(v) in [np.ndarray, dict]:
                if index is None:
                    v = v.copy()
                else:
                    v = v[index]
            p.__dict__[k] = v
        return p

    def __init__ref(self, p0c, energy0, gamma0, beta0):
        not_none = count_not_none(beta0, gamma0, p0c, energy0)
        _new = None
        if not_none == 1:
            if p0c is not None:
                _new = self._f1(self.mass0, p0c)
            elif energy0 is not None:
                _new = self._f2(self.mass0, energy0)
            elif gamma0 is not None:
                _new = self._f4(self.mass0, gamma0)
            elif beta0 is not None:
                _new = self._f3(self.mass0, beta0)
        elif not_none == 0:
            _new = self._f1(self.mass0, "1e9")
            # raise ValueError("Particles defined without energy reference")
        else:
            raise ValueError(
                f"""\
            Particles defined with multiple energy references:
            p0c    = {p0c},
            energy0     = {energy0},
            gamma0 = {gamma0},
            beta0  = {beta0}"""
            )
        if not( _new is None ):
            self._update_ref(*_new)

    def __init__delta(self, delta, ptau, psigma):
        not_none = count_not_none(delta, ptau, psigma)
        if not_none == 1:
            if not( delta is None ):
                self.delta = delta
            elif not( ptau is None ):
                self.ptau = ptau
            else:
                assert not( psigma is None )
                self.psigma = psigma
        elif not_none == 0:
            self.delta = _make_attr( "0.0", length=self._arg_length,
                dtype=self.CONSTANTS.real_type, as_vector=self._is_vector)
        else:
            raise ValueError(
                f"""
            Particles defined with multiple energy deviations:
            delta  = {delta},
            ptau     = {ptau},
            psigma = {psigma}"""
            )

    def __init__zeta(self, zeta, tau, sigma):
        not_none = count_not_none(zeta, tau, sigma)
        if not_none == 1:
            if not( zeta is None ):
                self.zeta = zeta
            elif not( tau is None ):
                self.tau = tau
            elif not( sigma is None ):
                self.sigma = sigma
        elif not_none == 0:
            self.zeta = _make_attr("0.0",
                length=self._arg_length, dtype=self.CONSTANTS.real_type, as_vector=self._is_vector)
        else:
            raise ValueError(
                f"""\
            Particles defined with multiple time deviations:
            zeta  = {zeta},
            tau   = {tau},
            sigma = {sigma}"""
            )

    def __init__chi(self, mratio, qratio, chi):
        not_none = count_not_none(mratio, qratio, chi)
        real_t = self.CONSTANTS.real_type
        length = self._arg_length
        is_vec = self._is_vector

        if not_none == 0:
            self._chi = _make_attr("1.0", length, dtype=real_t, as_vector=is_vec)
            self._mratio = _make_attr("1.0", length, dtype=real_t, as_vector=is_vec)
            self._qratio = _make_attr("1.0", length, dtype=real_t, as_vector=is_vec)
        elif not_none == 1:
            raise ValueError(
                f"""\
            Particles defined with insufficient mass/charge information:
            chi    = {chi},
            mratio = {mratio},
            qratio = {qratio}"""
            )
        elif not_none == 2:
            if chi is None:
                self._mratio = _make_attr(mratio, length, dtype=real_t, default=1, as_vector=is_vec)
                self._qratio = _make_attr(qratio, length, dtype=real_t, default=1, as_vector=is_vec)
                self._chi = self._qratio / self._mratio
            elif mratio is None:
                self._chi = _make_attr(chi, length, dtype=real_t, default=1, as_vector=is_vec)
                self._qratio = _make_attr(qratio, length, dtype=real_t, default=1, as_vector=is_vec)
                self._mratio = self._qratio / self._chi
            elif qratio is None:
                self._chi = _make_attr(chi, length, dtype=real_t, default=1, as_vector=is_vec)
                self._mratio = _make_attr(qratio, length, dtype=real_t, default=1, as_vector=is_vec)
                self._qratio = self._chi * self._mratio
        else:
            _mratio = _make_attr(mratio, length, dtype=real_t, default=1, as_vector=is_vec)
            _qratio = _make_attr(qratio, length, dtype=real_t, default=1, as_vector=is_vec)
            _chi = _make_attr(chi, length, dtype=real_t, default=1, as_vector=is_vec)

            abs_tol = real_t(1e-12)
            rel_tol = real_t(1e-10)

            is_consistent = False

            if as_vec:
                is_consistent = np.allclose(
                    _chi, _qratio/_mratio, rel_tol=rel_tol, abs_tol=abs_tol )
            else:
                diff = _chi - _qratio / _mratio
                if diff < real_t("0.0"):
                    diff = -diff
                is_consistent = diff <= abs_tol and diff * _chi <= rel_tol

            if is_consistent:
                self._chi = _chi
                self._mratio = _mratio
                self._qratio = _qratio
            else:
                raise ValueError( f""" Particles defined with multiple mass/charge information:
                    chi    = {chi},
                    mratio = {mratio},
                    qratio = {qratio}"""
                    )

    def __init__(
        self,
        s="0.0",
        x="0.0",
        px="0.0",
        y="0.0",
        py="0.0",
        delta=None,
        ptau=None,
        psigma=None,
        rvv=None,
        zeta=None,
        tau=None,
        sigma=None,
        mass0=None,
        q0="1.0",
        p0c=None,
        energy0=None,
        gamma0=None,
        beta0=None,
        chi="1.0",
        mratio="1.0",
        qratio="1.0",
        partid=None,
        turn=0,
        state=1,  # ==1 particle lost
        elemid=0,
        mathlib=MathlibDefault,
        **args,
    ):

        self.__init_mathlib(mathlib=mathlib)
        real_t = self.CONSTANTS.real_type
        int_t  = self.CONSTANTS.int_type
        length = _find_attr_length(
            x=x,y=y,px=px,py=py,zeta=zeta,mass0=mass0,q0=q0,p0c=p0c,delta=delta)

        is_vec = not( length is None ) and length > 1
        assert not( length is None ) and ( not is_vec or length > 1 )
        self._is_vector = as_vec
        self._arg_length = length
        self._one = _make_attr( 1, length, dtype=real_t, as_vector=is_vec)

        self._update_coordinates = False
        self.s = _make_attr(s, length, dtype=real_t, as_vector=is_vec)
        self.x = _make_attr(x, length, dtype=real_t, as_vector=is_vec)
        self.y = _make_attr(y, length, dtype=real_t, as_vector=is_vec)
        self.px = _make_attr(px, length, dtype=real_t, as_vector=is_vec)
        self.py = _make_attr(py, length, dtype=real_t, as_vector=is_vec)
        self._mass0 = _make_attr(mass0, length, dtype=real_t, default=self.CONSTANTS.PMASS, as_vector=is_vec)
        self.q0 = _make_attr(q0, length, dtype=real_t, default=self.CONSTANTS.ECHARGE, as_vector=is_vec)

        if is_vec and partid is None:
            self.partid = np.arange(length,dtype=int_t)
        else:
            self.partid = _make_attr(partid, length, dtype=int_t, as_vector=is_vec)

        self.turn = _make_attr(turn, length, dtype=int_t, as_vector=is_vec)
        self.elemid = _make_attr(elemid, length, dtype=int_t, as_vector=is_vec )
        self.state = _make_attr(state, length, dtype=int_t, as_vector=is_vec)

        self.__init__ref(p0c, energy0, gamma0, beta0)
        self.__init__delta(delta, ptau, psigma)
        self.__init__zeta(zeta, tau, sigma)
        self.__init__chi(chi, mratio, qratio)
        self.lost_particles = []
        self._update_coordinates = True


    Px = property(lambda p: p.px * p.p0c * p.mratio)
    Py = property(lambda p: p.py * p.p0c * p.mratio)
    energy = property(lambda p: (p.ptau * p.p0c + p.energy0) * p.mratio)
    pc = property(lambda p: (p.delta * p.p0c + p.p0c) * p.mratio)
    mass = property(lambda p: p.mass0 * p.mratio)

    @property
    def beta( self ):
        return ( self._one + self.delta ) / ( self._one / self.beta0 + self.tau )
    # rvv = property(lambda self: self.beta/self.beta0)
    # rpp = property(lambda self: 1/(1+self.delta))

    rvv = property(lambda self: self._rvv)
    rpp = property(lambda self: self._rpp)

    def add_to_energy(self, energy):
        sqrt = self._m.sqrt
        real_t = self.CONSTANTS.real_type
        length = self._arg_length
        is_vec = self._is_vector
        oldrvv = self._rvv
        deltabeta0 = self.delta * self.beta0
        ptaubeta0 = sqrt(deltabeta0 ** 2 + self.CONSTANTS.TWO * deltabeta0 *
                         self.beta0 + self._one) - self._one
        ptaubeta0 += _make_attr( energy, length, dtype=real_t, as_vector=is_vec ) / self.energy0
        ptau = ptaubeta0 / self.beta0
        self._delta = sqrt(ptau ** 2 + self.CONSTANTS.TWO * ptau / self.beta0 + self._one ) - self._one
        one_plus_delta = self._one + self._delta
        self._rvv = one_plus_delta / (self._one + ptaubeta0)
        self._rpp = self._one / one_plus_delta
        self.zeta *= self._rvv / oldrvv

    delta = property(lambda self: self._delta)

    @delta.setter
    def delta(self, delta):
        sqrt = self._m.sqrt
        real_t = self.CONSTANTS.real_type
        length = self._arg_length
        is_vec = self._is_vector
        self._delta = _make_attr(delta, length, dtype=real_t, as_vector=is_vec)
        deltabeta0 = self.delta * self.beta0
        ptaubeta0 = sqrt(deltabeta0 ** 2 + self.CONSTANTS.TWO * deltabeta0 * self.beta0 + self._one) - self._one
        self._rvv = (self._one + self.delta) / (self._one + ptaubeta0)
        self._rpp = self._one / (self._one + self.delta)

    psigma = property(lambda self: self.ptau / self.beta0)

    @psigma.setter
    def psigma(self, psigma):
        self.ptau = _make_attr(psigma, self._arg_length, dtype=self.CONSTANTS.real_type,
                as_vector=self._is_vector) * self.beta0

    tau = property(lambda self: self.zeta / self.beta)

    @tau.setter
    def tau(self, tau):
        self.zeta = self.beta * _make_attr(
            tau, self._arg_length, dtype=self.CONSTANTS.real_type, as_vector=self._is_vector)

    sigma = property(lambda self: self.zeta / self._rvv)

    @sigma.setter
    def sigma(self, sigma):
        self.zeta = self._rvv * _make_attr( sigma, self._arg_length,
            dtype=self.CONSTANTS.real_type, as_vector=self._is_vector)

    @property
    def ptau(self):
        sqrt = self._m.sqrt
        real_t = self.CONSTANTS.real_type
        is_vec = self._is_vector
        length = self._arg_length
        return sqrt(self._delta ** 2 + self.CONSTANTS.TWO * self._delta + self._one / self.beta0 ** 2) - self._one / self.beta0

    @ptau.setter
    def ptau(self, ptau):
        sqrt = self._m.sqrt
        real_t = self.CONSTANTS.real_type
        is_vec = self._is_vector
        length = self._arg_length
        ptau = _make_attr(ptau, length, dtype=real_t, as_vector=is_vec)
        self._delta = sqrt(ptau ** 2 + self.CONSTANTS.TWO * ptau / self.beta0 + self._one) - self._one
        deltabeta0 = self._delta * self.beta0
        ptaubeta0 = sqrt( deltabeta0 ** 2 + self.CONSTANTS.TWO * deltabeta0 * self.beta0 + self._one ) - self._one
        one_plus_delta = self._one + self._delta
        self._rvv = one_plus_delta / (self._one + ptaubeta0)
        self._rpp = self._one / one_plus_delta

    mass0 = property(lambda self: self._mass0)

    @mass0.setter
    def mass0(self, mass0):
        new = self._f1(mass0, self.p0c)
        _abs = self._get_absolute()
        self._update_ref(*new)
        self._update_particles_from_absolute(*_abs)

    beta0 = property(lambda self: self._beta0)

    @beta0.setter
    def beta0(self, beta0):
        new = self._f3(self.mass0, beta0)
        _abs = self._get_absolute()
        self._update_ref(*new)
        self._update_particles_from_absolute(*_abs)

    gamma0 = property(lambda self: self._gamma0)

    @gamma0.setter
    def gamma0(self, gamma0):
        new = self._f4(self.mass0, gamma0)
        _abs = self._get_absolute()
        self._update_ref(*new)
        self._update_particles_from_absolute(*_abs)

    p0c = property(lambda self: self._p0c)

    @p0c.setter
    def p0c(self, p0c):
        new = self._f1(self.mass0, p0c)
        _abs = self._get_absolute()
        self._update_ref(*new)
        self._update_particles_from_absolute(*_abs)

    energy0 = property(lambda self: self._energy0)

    @energy0.setter
    def energy0(self, energy0):
        new = self._f2(self.mass0, energy0)
        _abs = self._get_absolute()
        self._update_ref(*new)
        self._update_particles_from_absolute(*_abs)

    mratio = property(lambda self: self._mratio)

    @mratio.setter
    def mratio(self, mratio):
        self._mratio = _make_attr(mratio, self._arg_length,
                dtype=self.CONSTANTS.real_type, as_vector=self._is_vector)
        self._chi = self._qratio / self._mratio

    qratio = property(lambda self: self._qratio)

    @qratio.setter
    def qratio(self, qratio):
        real_t = self.CONSTANTS.real_type
        is_vec = self._is_vector
        length = self._arg_length
        self._qratio = _make_attr(qratio, self._arg_length,
            dtype=self.CONSTANTS.real_type, as_vector=self._is_vector)
        self._chi = self._qratio / self._mratio

    chi = property(lambda self: self._chi)

    @chi.setter
    def chi(self, chi):
        self._qratio = self._chi * self._mratio
        self._chi = _make_attr(chi, self._arg_length, dtype=self.CONSTANTS.real_type, as_vector=self._is_vector)
        self._mratio = self._qratio / self._chi

    def _get_absolute(self):
        return self.Px, self.Py, self.pc, self.energy

    def _update_ref(self, mass0, beta0, gamma0, p0c, energy0):
        real_t = self.CONSTANTS.real_type
        is_vec = self._is_vector
        length = self._arg_length

        self._mass0 = _make_attr(mass0, length, dtype=real_t, as_vector=is_vec)
        self._beta0 = _make_attr(beta0, length, dtype=real_t, as_vector=is_vec)
        self._gamma0 = _make_attr(gamma0, length, dtype=real_t, as_vector=is_vec)
        self._p0c = _make_attr(p0c, length, dtype=real_t, as_vector=is_vec)
        self._energy0 = _make_attr(energy0, length, dtype=real_t, as_vector=is_vec)

    def _update_particles_from_absolute(self, Px, Py, pc, energy):
        if self._update_coordinates:
            real_t = self.CONSTANTS.real_type
            is_vec = self._is_vector
            length = self._arg_length

            mratio = self.mass / self.mass0
            norm = mratio * self.p0c
            self._mratio = mratio
            self._chi = self._qratio / mratio
            self._ptau = energy / norm - self._one
            self._delta = pc / norm - self._one
            self.px = _make_attr(Px, length, dtype=real_t, as_vector=is_vec) / norm
            self.py = _make_attr(Py, length, dtype=real_t, as_vector=is_vec) / norm

    def __repr__(self):
        out = f"""\
        mass0   = {self.mass0}
        p0c     = {self.p0c}
        energy0 = {self.energy0}
        beta0   = {self.beta0}
        gamma0  = {self.gamma0}
        s       = {self.s}
        x       = {self.x}
        px      = {self.px}
        y       = {self.y}
        py      = {self.py}
        zeta    = {self.zeta}
        delta   = {self.delta}
        ptau    = {self.ptau}
        mratio  = {self.mratio}
        qratio  = {self.qratio}
        chi     = {self.chi}"""
        return out

    _dict_vars = (
        "s",
        "x",
        "px",
        "y",
        "py",
        "delta",
        "zeta",
        "mass0",
        "q0",
        "p0c",
        "chi",
        "mratio",
        "partid",
        "turn",
        "state",
    )

    def remove_lost_particles(self, keep_memory=True):

        if hasattr(self.state, "__iter__"):
            mask_valid = self.state == self.CONSTANTS.int_type(1)

            if np.any(~mask_valid):
                if keep_memory:
                    to_trash = (
                        self.copy()
                    )  # Not exactly efficient (but robust)
                    for ff in self._dict_vars:
                        if hasattr(getattr(self, ff), "__iter__"):
                            setattr(
                                to_trash, ff, getattr(self, ff)[~mask_valid]
                            )
                    self.lost_particles.append(to_trash)

            for ff in self._dict_vars:
                if hasattr(getattr(self, ff), "__iter__"):
                    setattr(self, ff, getattr(self, ff)[mask_valid])

    def to_dict(self):
        return {kk: getattr(self, kk) for kk in self._dict_vars}

    @classmethod
    def from_dict(cls, dct):
        return cls(**dct)

    def compare(self, particle, rel_tol=1e-6, abs_tol=1e-15):
        res = True
        for kk in self._dict_vars:
            v1 = getattr(self, kk)
            v2 = getattr(particle, kk)
            if v1 is not None and v2 is not None:
                diff = v1 - v2
                if hasattr(diff, "__iter__"):
                    for nn in range(len(diff)):
                        vv1 = v1[nn] if hasattr(v1, "__iter__") else v1
                        vv2 = v2[nn] if hasattr(v2, "__iter__") else v2
                        if abs(diff[nn]) > abs_tol:
                            print(f"{kk}[{nn}] {vv1} {vv2}  diff:{diff[nn]}")
                            res = False
                        if abs(vv1) > 0 and abs(diff[nn]) / vv1 > rel_tol:
                            print(
                                f"{kk}[{nn}] {vv1} {vv2} rdiff:{diff[nn]/vv1}"
                            )
                            res = False
                else:
                    if abs(diff) > abs_tol:
                        print(f"{kk} {v1} {v2}  diff:{diff}")
                        res = False
                    if abs(v1) > 0 and abs(diff) / v1 > rel_tol:
                        print(f"{kk} {v1} {v2} rdiff:{diff/v1}")
                        res = False
        return res

    @classmethod
    def from_madx_twiss(cls, twiss):
        out = cls(
            p0c=twiss.summary.pc * 1e6,
            mass0=twiss.summary.mass * 1e6,
            q0=twiss.summary.charge,
            s=twiss.s[:],
            x=twiss.x[:],
            px=twiss.px[:],
            y=twiss.y[:],
            py=twiss.py[:],
            tau=twiss.t[:],
            ptau=twiss.pt[:],
        )
        return out

    @classmethod
    def from_madx_track(cls, mad):
        tracksumm = mad.table.tracksumm
        mad_beam = mad.sequence().beam
        out = cls(
            p0c=mad_beam.pc * 1e6,
            mass0=mad_beam.mass * 1e6,
            q0=mad_beam.charge,
            s=tracksumm.s[:],
            x=tracksumm.x[:],
            px=tracksumm.px[:],
            y=tracksumm.y[:],
            py=tracksumm.py[:],
            tau=tracksumm.t[:],
            ptau=tracksumm.pt[:],
        )
        return out

    @classmethod
    def from_list(cls, lst):
        ll = len(lst)
        dct = {nn: np.zeros(ll) for nn in cls._dict_vars}
        for ii, pp in enumerate(lst):
            for nn in cls._dict_vars:
                dct[nn][ii] = getattr(pp, nn, 0)
        return cls(**dct)
