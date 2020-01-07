import numpy as np
from .mathlibs import MathlibDefault
from importlib import util
import types
import pdb

mpmath_spec = util.find_spec("mpmath")
if mpmath_spec is not None:
    from mpmath.ctx_mp import MPContext as mpmath_ctx

def count_not_none(*lst):
    return len(lst) - sum(p is None for p in lst)


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

    CLIGHT_STR = "299792458.0"
    ECHARGE_STR = "1.602176565e-19"
    EMASS_STR = "0.510998928e6"
    NMASS_STR = "931.49410242e6"
    PMASS_STR = "938.272081e6"
    EPSILON0_STR = "8.854187817e-12"
    ANUMBER_STR = "6.02214129e23"
    KBOLZ_STR = "1.3806488e-23"

    def __init_mathlib(self, mathlib=MathlibDefault,
            real_type=np.float64, int_type=np.int64 ):
        if mpmath_spec is not None and \
            isinstance( mathlib, types.ModuleType ) and \
            mathlib.__name__ == 'mpmath':
            from mpmath import mp, pi as mp_pi
            real_type = mp.mpf
            self.PI = mp_pi
        else:
            self.PI = np.pi

        self.CLIGHT   = real_type(self.CLIGHT_STR)
        self.ECHARGE  = real_type(self.ECHARGE_STR)
        self.EMASS    = real_type(self.EMASS_STR)
        self.NMASS    = real_type(self.NMASS_STR)
        self.PMASS    = real_type(self.PMASS_STR)
        self.EPSILON0 = real_type(self.EPSILON0_STR)
        self.ANUMBER  = real_type(self.ANUMBER_STR)
        self.KBOLZ    = real_type(self.KBOLZ_STR)
        self.MU0      = real_type("4.0e-7") * self.PI
        self.ERADIUS  = self.ECHARGE * self.ECHARGE / ( real_type( 4 ) *
            self.PI * self.EPSILON0 * self.EMASS * self.CLIGHT * self.CLIGHT )
        self.PRADIUS  = self.ECHARGE * self.ECHARGE / ( real_type( 4 ) *
            self.PI * self.EPSILON0 * self.PMASS * self.CLIGHT * self.CLIGHT )

        self._m = mathlib
        self._real_type = real_type
        self._int_type = int_type


    def _g1(self, mass0, p0c, energy0):
        p0c = self._real_type(p0c)
        energy0 = self._real_type(energy0)
        mass0 = self._real_type(mass0)
        beta0 = p0c / energy0
        gamma0 = energy0 / mass0
        return mass0, beta0, gamma0, p0c, energy0

    def _g2(self, mass0, beta0, gamma0):
        mass0 = self._real_type(mass0)
        beta0 = self._real_type(beta0)
        gamma0 = self._real_type(gamma0)
        energy0 = mass0 * gamma0
        p0c = energy0 * beta0
        return mass0, beta0, gamma0, p0c, energy0

    def _f1(self, mass0, p0c):
        sqrt = self._m.sqrt
        mass0 = self._real_type(mass0)
        p0c=self._real_type(p0c)
        energy0 = sqrt(p0c * p0c + mass0 * mass0)
        return self._g1(mass0, p0c, energy0)

    def _f2(self, mass0, energy0):
        sqrt = self._m.sqrt
        mass0 = self._real_type(mass0)
        energy0 = self._real_type(energy0)
        p0c = sqrt(energy0 * energy0 - mass0 * mass0)
        return self._g1(mass0, p0c, energy0)

    def _f3(self, mass0, beta0):
        sqrt = self._m.sqrt
        one = self._real_type("1")
        gamma0 = 1 / sqrt(one - beta0 * beta0)
        return self._g2(mass0, beta0, gamma0)

    def _f4(self, mass0, gamma0):
        sqrt = self._m.sqrt
        mass0 = self._real_type(mass0)
        gamma0 = self._real_type(gamma0)
        one = self._real_type("1")
        beta0 = sqrt(one - one / gamma0 * gamma0)
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
        if not_none == 0:
            p0c = "1e9"
            not_none = 1
            # raise ValueError("Particles defined without energy reference")
        if not_none == 1:
            if p0c is not None:
                new = self._f1(self.mass0, p0c)
                self._update_ref(*new)
            elif energy0 is not None:
                new = self._f2(self.mass0, energy0)
                self._update_ref(*new)
            elif gamma0 is not None:
                new = self._f4(self.mass0, gamma0)
                self._update_ref(*new)
            elif beta0 is not None:
                new = self._f3(self.mass0, beta0)
                self._update_ref(*new)
        else:
            raise ValueError(
                f"""\
            Particles defined with multiple energy references:
            p0c    = {p0c},
            energy0     = {energy0},
            gamma0 = {gamma0},
            beta0  = {beta0}"""
            )

    def __init__delta(self, delta, ptau, psigma):
        not_none = count_not_none(delta, ptau, psigma)
        if not_none == 0:
            self.delta = "0.0"
        elif not_none == 1:
            if delta is not None:
                self.delta = delta
            elif ptau is not None:
                self.ptau = ptau
            elif psigma is not None:
                self.psigma = psigma
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
        if not_none == 0:
            self.zeta = "0.0"
        elif not_none == 1:
            if zeta is not None:
                self.zeta = zeta
            elif tau is not None:
                self.tau = tau
            elif sigma is not None:
                self.sigma = sigma
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
        if not_none == 0:
            self._chi = self._real_type("1.0")
            self._mratio = self._real_type("1.0")
            self._qratio = self._real_type("1.0")
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
                self._mratio = mratio
                self.qratio = qratio
            elif mratio is None:
                self._chi = chi
                self.qratio = qratio
            elif qratio is None:
                self._chi = chi
                self.mratio = mratio
        else:
            raise ValueError(
                f"""
            Particles defined with multiple mass/charge information:
            chi    = {chi},
            mratio = {mratio},
            qratio = {qratio}"""
            )

    @staticmethod
    def __init_attr(val, length, default=0, dtype=np.float64, sep="," ):
        assert not( dtype is None )
        is_vector_like = bool( not( length is None ) and length > 0 )

        if val is None:
            if is_vector_like:
                ret_val = np.array( [ dtype(default), ] )
            else:
                ret_val = dtype(default)
        elif isinstance(val, str):
            if sep is not None and sep in val:
                if is_vector_like:
                    ret_val = np.fromstring(val, sep=sep, count=length)
                else:
                    ret_val = dtype(val.split(sep)[ 0 ])
            else:
                val = dtype(val)
                if is_vector_like:
                    ret_val = np.zeros(length, dtype=dtype)
                    ret_val[:] = val
                else:
                    ret_val = val
        else:
            try:
                it = iter(val)
            except TypeError:
                if is_vector_like:
                    ret_val = np.zeros(length)
                    ret_val[:] = dtype(val)
                else:
                    ret_val = dtype(val)
            else:
                if is_vector_like:
                    ret_val = np.fromiter(it, count=length, dtype=dtype)
                else:
                    ret_val = np.fromiter(it, count=length, dtype=dtype)[ 0 ]
        return ret_val


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
        turn=None,
        state=None,  # ==1 particle lost
        elemid=None,
        mathlib=MathlibDefault,
        **args,
    ):

        self.__init_mathlib(mathlib=mathlib)
        length = Particles._check_array_length(
            x=x,y=y,px=px,py=py,zeta=zeta,mass0=mass0,q0=q0,p0c=p0c,delta=delta)
        real_t = self._real_type
        int_t  = self._int_type

        if not( mass0 is None ):
            mass0 = self.PMASS

        self._update_coordinates = False
        pdb.set_trace()
        self.s  = Particles.__init_attr(s, length, dtype=real_t )
        self.x  = Particles.__init_attr(x, length, dtype=real_t )
        self.px = Particles.__init_attr(px, length, dtype=real_t )
        self.y  = Particles.__init_attr(y, length, dtype=real_t )
        self.py = Particles.__init_attr(py, length, dtype=real_t )
        self.zeta = Particles.__init_attr(zeta, length, dtype=real_t )
        self._mass0 = Particles.__init_attr(mass0, length, dtype=real_t, default=self.PMASS )
        self.q0 = Particles.__init_attr(q0, length, dtype=real_t, default=1 )

        if not( length is None ) and length > 1 and partid is None:
            self.partid = np.arange(length,dtype=int_t)
        else:
            self.partid = Particles.__init_attr(partid, length, dtype=int_t )
        self.turn = Particles.__init_attr(turn, length, dtype=int_t )
        self.elemid = Particles.__init_attr(elemid, length, dtype=int_t )
        self.state = Particles.__init_attr(state, length, default=1, dtype=int_t )

        self.__init__ref(p0c, energy0, gamma0, beta0)
        self.__init__delta(delta, ptau, psigma)
        self.__init__zeta(zeta, tau, sigma)
        self.__init__chi(chi, mratio, qratio)
        self.lost_particles = []
        self._update_coordinates = True


    @staticmethod
    def _check_array_length(dtype=np.float64, sep=',', **kwargs):
        names = ["x", "px", "y", "py", "zeta", "mass0", "q0", "p0c"]
        length = None
        for nn in names:
            ll = None
            val = kwargs.get(nn, None)
            if val is None:
                if not( length is None ):
                    length = None
                    break
                else:
                    continue

            if isinstance(val,str):
                if sep in val:
                    val = np.fromstring(val,dtype=dtype,sep=sep)
                    ll = len(val)
                else:
                    val = dtype(val)
            else:
                try:
                    it = iter(val)
                except TypeError:
                    ll = None
                else:
                    ll = len(val)

            if length is None:
                length = ll
            elif length != ll:
                raise ValueError(f"invalid length len({val})={len(val)}")
            assert ll == length
        return length

    Px = property(lambda p: p.px * p.p0c * p.mratio)
    Py = property(lambda p: p.py * p.p0c * p.mratio)
    energy = property(lambda p: (p.ptau * p.p0c + p.energy0) * p.mratio)
    pc = property(lambda p: (p.delta * p.p0c + p.p0c) * p.mratio)
    mass = property(lambda p: p.mass0 * p.mratio)
    beta = property(lambda p: (p._real_type(1) + p.delta) / (p._real_type(1) / p.beta0 + p.ptau))
    # rvv = property(lambda self: self.beta/self.beta0)
    # rpp = property(lambda self: 1/(1+self.delta))

    rvv = property(lambda self: self._rvv)
    rpp = property(lambda self: self._rpp)

    def add_to_energy(self, energy):
        sqrt = self._m.sqrt
        one = self._real_type(1)
        two = self._real_type(2)
        oldrvv = self._rvv
        deltabeta0 = self.delta * self.beta0
        ptaubeta0 = sqrt(deltabeta0 ** 2 + two * deltabeta0 * self.beta0 + one) - one
        ptaubeta0 += self._real_type( energy ) / self.energy0
        ptau = ptaubeta0 / self.beta0
        self._delta = sqrt(ptau ** 2 + two * ptau / self.beta0 + one) - one
        self._rvv = (one + self.delta) / (one + ptaubeta0)
        self._rpp =one / (one + self.delta)
        self.zeta *= self._rvv / oldrvv

    delta = property(lambda self: self._delta)

    @delta.setter
    def delta(self, delta):
        sqrt = self._m.sqrt
        one = self._real_type(1)
        two = self._real_type(2)
        self._delta = self._real_type(delta)
        deltabeta0 = self._real_type(delta) * self.beta0
        ptaubeta0 = sqrt(deltabeta0 ** 2 + two * deltabeta0 * self.beta0 + one) - one
        self._rvv = (one + self.delta) / (one + ptaubeta0)
        self._rpp = one / (one + self.delta)

    psigma = property(lambda self: self.ptau / self.beta0)

    @psigma.setter
    def psigma(self, psigma):
        self.ptau = self._real_type(psigma) * self.beta0

    tau = property(lambda self: self.zeta / self.beta)

    @tau.setter
    def tau(self, tau):
        self.zeta = self.beta * self._real_type(tau)

    sigma = property(lambda self: (self.beta0 / self.beta) * self.zeta)

    @sigma.setter
    def sigma(self, sigma):
        self.zeta = self.beta / self.beta0 * self._real_type(sigma)

    @property
    def ptau(self):
        sqrt = self._m.sqrt
        one = self._real_type(1)
        return (
            sqrt(self.delta ** 2 + self._real_type(2) * self.delta +
                 one / self.beta0 ** 2) - one / self.beta0 )

    @ptau.setter
    def ptau(self, ptau):
        sqrt = self._m.sqrt
        one = self._real_type(1)
        self.delta = sqrt(ptau ** 2 + self._real_type(2) * self._real_type(ptau) /
                            self.beta0 + one) - one

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
        self._mratio = mratio
        self._chi = self._qratio / self._mratio

    qratio = property(lambda self: self._qratio)

    @qratio.setter
    def qratio(self, qratio):
        self._qratio = self._real_type(qratio)
        self._chi = self._qratio / self._mratio

    chi = property(lambda self: self._chi)

    @chi.setter
    def chi(self, chi):
        self._qratio = self._chi * self._mratio
        self._chi = self._real_type(chi)

    def _get_absolute(self):
        return self.Px, self.Py, self.pc, self.energy

    def _update_ref(self, mass0, beta0, gamma0, p0c, energy0):
        self._mass0 = self._real_type(mass0)
        self._beta0 = self._real_type(beta0)
        self._gamma0 = self._real_type(gamma0)
        self._p0c = self._real_type(p0c)
        self._energy0 = self._real_type(energy0)

    def _update_particles_from_absolute(self, Px, Py, pc, energy):
        if self._update_coordinates:
            one = self._real_type(1)
            mratio = self.mass / self.mass0
            norm = self._real_type(mratio) * self.p0c
            self._mratio = mratio
            self._chi = self._qratio / mratio
            self._ptau = energy / norm - one
            self._delta = pc / norm - one
            self.px = self._real_type(Px) / norm
            self.py = self._real_type(Py) / norm

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
            mask_valid = self.state == self._int_type(1)

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
