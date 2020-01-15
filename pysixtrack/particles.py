import numpy as np
from .mathlibs import MathlibDefault

def count_not_none(*lst):
    return len(lst) - sum(p is None for p in lst)


def _find_attr_length(sep=",", **kwargs):
    """Estimates the attribute size from an iterable list of key-value pairs

    Keyword arguments:
    sep -- separator for string-valued arguments (default ',')
    kwargs -- all the key - value pairs that should be considered

    The following cases for values are handled:
    * a string valued value containing the separator is interpreted as a
      vector of length >= 1
    * a string valued value without separator is interpreted as a scalar
    * iterable values contribute the number of elements in the range
    * everything else is considered a scalar

    Returns:
    The number of elements required in each attribute, None in case of
    inconsistencies / errors

    Throws:
    If non-scalar (i.e. length > 1) values differ in their size, an instance
    of ValueError is thrown

    """
    length = None
    has_processed_any = False

    for key, value in kwargs.items():
        ll = None
        if value is None:
            continue
        elif isinstance(value, str):
            ll = sep is not None and sep in value and len(value.split(sep)) or 1
        else:
            try:
                it = iter(value)
            except TypeError:
                ll = 1
            else:
                ll = sum(1 for _ in iter)

        if not(ll is None):
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


def _make_attr(value, length, dtype, as_vec, default=0, sep=","):
    """Creates an attribute (either a scalar or a numpy array)

    Keyword arguments:
    value   -- the input value; can be a scalar, an iterable object,
               a string or a separated list encoded in a string
    length  -- the requested length of the attribute, has to an integer >= 1
    dtype   -- numpy.dtype instance describing the storage type for the
               vector attribute or any type that can be used for casting scalar
               attributes
    as_vec  -- boolean flag indicating whether the attribute should be a
               vector or a scalar; note that length > 1 requires as_vec to be
               True but length == 1 and as_vec == True is legal as well
    default -- default value, used in case the value is None or if
               the number of provided elements is smaller than the requeisted
               length of the attribute to fill it
    sep     -- separator string used to parse string-valued concatenated lists
               of elements

    Returns:
    Either a scalar attribute (requiring length == 1, as_vec = False) or a
    vector valued attribute with length entries of type dtype, filled with
    values taken from value (or default, in case a not sufficiant number of
    elements can be taken from value)

    Throws:
    Nothing

    """
    assert not (dtype is None)
    assert not as_vec or (not (length is None) and length > 0)
    ret_val = (
        as_vec
        and np.full(length, dtype(default), dtype=dtype)
        or dtype(default)
    )

    if isinstance(value, str):
        if sep is not None and sep in value:
            if as_vec:
                _temp = np.fromstring(value, sep=sep, dtype=dtype)
                _temp_len = len(_temp)
                if _temp_len <= length:
                    ret_val[0:_temp_len] = _temp[:]
                else:
                    ret_val[:] = _temp[0:length]
            else:
                ret_val = dtype(value.split(sep)[0])
        else:
            if as_vec:
                ret_val[:] = dtype(value)
            else:
                ret_val = dtype(value)
    elif not(value is None):
        try:
            it = iter(value)
        except TypeError:
            if as_vec:
                ret_val[:] = dtype(value)
            else:
                ret_val = dtype(value)
        else:
            _temp = np.fromiter(it, dtype=dtype)
            _temp_len = len(_temp)
            if as_vec:
                if _temp_len <= length:
                    ret_val[0:_temp_len] = _temp[:]
                else:
                    ret_val[:] = _temp[0:length]
            elif _temp_len > 0:
                ret_val = _temp[0]
    return ret_val


class Particles(object):
    """
    Coordinates:

    **fields**

    **properties

    s       [m]  Reference accumulated pathlength
    x       [m]  Horizontal offset
    px      [1]  Px / (m/m0 * pc0) = beta_x gamma /(beta0 gamma0)
    y       [m   Vertical offset]
    py      [1]  Py / (m/m0 * pc0)
    delta   [1]  Pc / (m/m0 * pc0) - 1
    ptau    [1]  Energy / (m/m0 * pc0) - 1
    psigma  [1]  ptau/beta0
    rvv     [1]  beta/beta0
    rpp     [1]  1/(1+delta) = (m/m0 * pc0) / Pc
    zeta    [m]  beta (s/beta0 - ct )
    tau     [m]
    sigma   [m]  s - beta0 ct = rvv * zeta
    mass0   [eV]
    charge0      [e]  reference carge
    pc0     [eV] reference momentum
    energy0 [eV] refernece energy
    gamma0  [1]  reference relativistic gamma
    beta0   [1]  reference relativistix beta
    chi     [1]  q/ charge0 * m0/m = qratio / mratio
    mratio  [1]  mass/mass0
    qratio  [1]  q / charge0
    partid  int
    turn    int
    state   int
    elemid  int
    """

    def _make_attr(self, value, dtype=None, default=0, sep=','):
        if dtype is None:
            dtype = self._m.real_type
        return _make_attr(value, self.num_particles, dtype, self.is_vector,
                          default=default, sep=sep)

    def _g1(self, mass0, pc0, energy0):
        pc0 = self._make_attr(pc0)
        energy0 = self._make_attr(energy0)
        mass0 = self._make_attr(mass0, default=self._m.pmass)
        beta0 = pc0 / energy0
        gamma0 = energy0 / mass0
        return mass0, beta0, gamma0, pc0, energy0

    def _g2(self, mass0, beta0, gamma0):
        beta0 = self._make_attr(beta0)
        gamma0 = self._make_attr(gamma0)
        mass0 = self._make_attr(mass0, default=self._m.pmass)

        energy0 = mass0 * gamma0
        pc0 = energy0 * beta0
        return mass0, beta0, gamma0, pc0, energy0

    def _f1(self, mass0, pc0):
        pc0 = self._make_attr(pc0)
        mass0 = self._make_attr(mass0, default=self._m.pmass)
        energy0 = self._m.sqrt(pc0 * pc0 + mass0 * mass0)
        return self._g1(mass0, pc0, energy0)

    def _f2(self, mass0, energy0):
        energy0 = self._make_attr(energy0)
        mass0 = self._make_attr(mass0, default=self._m.pmass)
        pc0 = self._m.sqrt(energy0 * energy0 - mass0 * mass0)
        return self._g1(mass0, pc0, energy0)

    def _f3(self, mass0, beta0):
        gamma0 = self._one / self._m.sqrt(self._one - beta0 * beta0)
        return self._g2(mass0, beta0, gamma0)

    def _f4(self, mass0, gamma0):
        gamma0 = self._make_attr(gamma0)
        mass0 = self._make_attr(mass0, default=self._m.pmass)
        beta0 = self._m.sqrt(self._one - self._one / gamma0 * gamma0)
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

    def __init__ref(self, pc0, energy0, gamma0, beta0):
        not_none = count_not_none(beta0, gamma0, pc0, energy0)
        _new = None
        if not_none == 1:
            if pc0 is not None:
                _new = self._f1(self.mass0, pc0)
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
            pc0     = {pc0},
            energy0 = {energy0},
            gamma0  = {gamma0},
            beta0   = {beta0}"""
            )
        if not(_new is None):
            self._update_ref(*_new)

    def __init__delta(self, delta, ptau, psigma):
        not_none = count_not_none(delta, ptau, psigma)
        if not_none == 1:
            if not(delta is None):
                self.delta = delta
            elif not(ptau is None):
                self.ptau = ptau
            else:
                assert not(psigma is None)
                self.psigma = psigma
        elif not_none == 0:
            self.delta = self._make_attr(0)
        else:
            raise ValueError(
                f"""
            Particles defined with multiple energy deviations:
            delta  = {delta},
            ptau   = {ptau},
            psigma = {psigma}"""
            )

    def __init__zeta(self, zeta, tau, sigma):
        not_none = count_not_none(zeta, tau, sigma)
        if not_none == 1:
            if not(zeta is None):
                self.zeta = zeta
            elif not(tau is None):
                self.tau = tau
            elif not(sigma is None):
                self.sigma = sigma
        elif not_none == 0:
            self.zeta = self._make_attr(0)
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
            self._chi = self._make_attr(1)
            self._mratio = self._make_attr(1)
            self._qratio = self._make_attr(1)
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
                self._mratio = self._make_attr(mratio)
                self._qratio = self._make_attr(qratio)
                self._chi = self._qratio / self._mratio
            elif mratio is None:
                self._chi = self._make_attr(chi)
                self._qratio = self._make_attr(qratio)
                self._mratio = self._qratio / self._chi
            elif qratio is None:
                self._chi = self._make_attr(chi)
                self._mratio = self._make_attr(qratio)
                self._qratio = self._chi * self._mratio
        else:
            _mratio = self._make_attr(mratio)
            _qratio = self._make_attr(qratio)
            _chi = self._make_attr(chi)

            abs_tol = self._m.real_type("1e-12")
            rel_tol = self._m.real_type("1e-10")

            is_consistent = False

            if self.is_vector:
                is_consistent = np.allclose(
                    _chi, _qratio / _mratio, rel_tol=rel_tol, abs_tol=abs_tol)
            else:
                diff = _chi - _qratio / _mratio
                if diff < self._m.real_type(0):
                    diff = -diff
                is_consistent = diff <= abs_tol and diff * _chi <= rel_tol

            if is_consistent:
                self._chi = _chi
                self._mratio = _mratio
                self._qratio = _qratio
            else:
                raise ValueError(
                    f""" Particles defined with multiple mass/charge information:
                    chi    = {chi},
                    mratio = {mratio},
                    qratio = {qratio}""")

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
        charge0=None,
        pc0=None,
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

        self._m = mathlib
        length = _find_attr_length(
            x=x,
            y=y,
            px=px,
            py=py,
            zeta=zeta,
            mass0=mass0,
            charge0=charge0,
            pc0=pc0,
            delta=delta)

        self._is_vector = not(length is None) and length > 1
        self._nparticles = not(length is None) and length or 0
        self._one = self._make_attr(1)

        self._update_coordinates = False
        self.s = self._make_attr(s)
        self.x = self._make_attr(x)
        self.y = self._make_attr(y)
        self.px = self._make_attr(px)
        self.py = self._make_attr(py)
        self._mass0 = self._make_attr(mass0, default=mathlib.pmass)
        self.charge0 = self._make_attr(charge0, default=mathlib.echarge)

        if self._is_vector and partid is None:
            self.partid = np.arange(length, dtype=mathlib.int_type)
        else:
            self.partid = self._make_attr(partid, dtype=mathlib.int_type)

        self.turn = self._make_attr(turn, dtype=mathlib.int_type)
        self.elemid = self._make_attr(elemid, dtype=mathlib.int_type)
        self.state = self._make_attr(state, dtype=mathlib.int_type)

        self.__init__ref(pc0, energy0, gamma0, beta0)
        self.__init__delta(delta, ptau, psigma)
        self.__init__zeta(zeta, tau, sigma)
        self.__init__chi(chi, mratio, qratio)
        self.lost_particles = []
        self._update_coordinates = True

    Px = property(lambda p: p.px * p.pc0 * p.mratio)
    Py = property(lambda p: p.py * p.pc0 * p.mratio)
    energy = property(lambda p: (p.ptau * p.pc0 + p.energy0) * p.mratio)
    pc = property(lambda p: (p.delta * p.pc0 + p.pc0) * p.mratio)
    mass = property(lambda p: p.mass0 * p.mratio)
    charge = property(lambda p: p.charge0 * p.qratio)

    @property
    def beta(self):
        return (self._one + self.delta) / (self._one / self.beta0 + self.tau)

    rvv = property(lambda self: self._rvv)
    rpp = property(lambda self: self._rpp)
    num_particles = property(lambda self: self._nparticles)
    is_vector = property(lambda self: self._is_vector and self._nparticles > 0)

    def add_to_energy(self, energy):
        oldrvv = self._rvv
        deltabeta0 = self.delta * self.beta0
        ptaubeta0 = self._m.sqrt(deltabeta0 ** 2 + 2 * deltabeta0 *
                                 self.beta0 + self._one) - self._one
        ptaubeta0 += self._make_attr(energy) / self.energy0
        ptau = ptaubeta0 / self.beta0
        self._delta = self._m.sqrt(
            ptau ** 2 + 2 * ptau / self.beta0 + self._one) - self._one
        one_plus_delta = self._one + self._delta
        self._rvv = one_plus_delta / (self._one + ptaubeta0)
        self._rpp = self._one / one_plus_delta
        self.zeta *= self._rvv / oldrvv

    delta = property(lambda self: self._delta)

    @delta.setter
    def delta(self, delta):
        self._delta = self._make_attr(delta)
        deltabeta0 = self._delta * self.beta0
        ptaubeta0 = self._m.sqrt(
            deltabeta0 ** 2 + 2 * deltabeta0 * self.beta0 + self._one) - self._one
        one_plus_delta = self._one + self._delta
        self._rvv = one_plus_delta / (self._one + ptaubeta0)
        self._rpp = self._one / one_plus_delta

    psigma = property(lambda self: self.ptau / self.beta0)

    @psigma.setter
    def psigma(self, psigma):
        self.ptau = self._make_attr(psigma) * self.beta0

    tau = property(lambda self: self.zeta / self.beta)

    @tau.setter
    def tau(self, tau):
        self.zeta = self.beta * self._make_attr(tau)

    sigma = property(lambda self: self.zeta / self._rvv)

    @sigma.setter
    def sigma(self, sigma):
        self.zeta = self._rvv * self._make_attr(sigma)

    @property
    def ptau(self):
        return self._m.sqrt(self._delta ** 2 + 2 * self._delta +
                            self._one / (self.beta0 ** 2)) - self._one / self.beta0

    @ptau.setter
    def ptau(self, ptau):
        ptau = self._make_attr(ptau)
        self._delta = self._m.sqrt(
            ptau ** 2 + 2 * ptau / self.beta0 + self._one) - self._one
        deltabeta0 = self._delta * self.beta0
        ptaubeta0 = self._m.sqrt(
            deltabeta0 ** 2 + 2 * deltabeta0 * self.beta0 + self._one) - self._one
        one_plus_delta = self._one + self._delta
        self._rvv = one_plus_delta / (self._one + ptaubeta0)
        self._rpp = self._one / one_plus_delta

    mass0 = property(lambda self: self._mass0)

    @mass0.setter
    def mass0(self, mass0):
        _new = self._f1(mass0, self.pc0)
        _abs = self._get_absolute()
        self._update_ref(*_new)
        self._update_particles_from_absolute(*_abs)

    beta0 = property(lambda self: self._beta0)

    @beta0.setter
    def beta0(self, beta0):
        _new = self._f3(self.mass0, beta0)
        _abs = self._get_absolute()
        self._update_ref(*_new)
        self._update_particles_from_absolute(*_abs)

    gamma0 = property(lambda self: self._gamma0)

    @gamma0.setter
    def gamma0(self, gamma0):
        _new = self._f4(self.mass0, gamma0)
        _abs = self._get_absolute()
        self._update_ref(*_new)
        self._update_particles_from_absolute(*_abs)

    pc0 = property(lambda self: self._pc0)

    @pc0.setter
    def pc0(self, pc0):
        _new = self._f1(self.mass0, pc0)
        _abs = self._get_absolute()
        self._update_ref(*_new)
        self._update_particles_from_absolute(*_abs)

    energy0 = property(lambda self: self._energy0)

    @energy0.setter
    def energy0(self, energy0):
        _new = self._f2(self.mass0, energy0)
        _abs = self._get_absolute()
        self._update_ref(*_new)
        self._update_particles_from_absolute(*_abs)

    mratio = property(lambda self: self._mratio)

    @mratio.setter
    def mratio(self, mratio):
        self._mratio = self._make_attr(mratio, default=1)
        self._chi = self._qratio / self._mratio

    qratio = property(lambda self: self._qratio)

    @qratio.setter
    def qratio(self, qratio):
        self._qratio = self._make_attr(qratio, default=1)
        self._chi = self._qratio / self._mratio

    chi = property(lambda self: self._chi)

    @chi.setter
    def chi(self, chi):
        self._qratio = self._chi * self._mratio
        self._chi = self._make_attr(chi)
        self._mratio = self._qratio / self._chi

    def _get_absolute(self):
        return self.Px, self.Py, self.pc, self.energy

    def _update_ref(self, mass0, beta0, gamma0, pc0, energy0):
        self._mass0 = self._make_attr(mass0)
        self._beta0 = self._make_attr(beta0)
        self._gamma0 = self._make_attr(gamma0)
        self._pc0 = self._make_attr(pc0)
        self._energy0 = self._make_attr(energy0)

    def _update_particles_from_absolute(self, Px, Py, pc, energy):
        if self._update_coordinates:
            mratio = self.mass / self.mass0
            norm = mratio * self.pc0
            self._mratio = mratio
            self._chi = self._qratio / mratio
            self._ptau = energy / norm - self._one
            self._delta = pc / norm - self._one
            self.px = self._make_attr(Px) / norm
            self.py = self._make_attr(Py) / norm

    def __len__(self):
        return self._nparticles

    def __repr__(self):
        out = f"""\
        mass0   = {self.mass0}
        charge0 = {self.charge0}
        pc0     = {self.pc0}
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
        "charge0",
        "pc0",
        "chi",
        "mratio",
        "partid",
        "turn",
        "elemid",
        "state",
    )

    def remove_lost_particles(self, keep_memory=True):
        if self.is_vector:
            mask_valid = self.state == self._m.int_type(1)
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
    def from_madx_twiss(cls, twiss, mathlib=MathlibDefault):
        mev_to_ev = mathlib.real_type("1e6")
        out = cls(
            pc0=twiss.summary.pc * mev_to_ev,
            mass0=twiss.summary.mass * mev_to_ev,
            charge0=twiss.summary.charge,
            s=twiss.s[:],
            x=twiss.x[:],
            px=twiss.px[:],
            y=twiss.y[:],
            py=twiss.py[:],
            tau=twiss.t[:],
            ptau=twiss.pt[:],
            mathlib=mathlib
        )
        return out

    @classmethod
    def from_madx_track(cls, mad, mathlib=MathlibDefault):
        tracksumm = mad.table.tracksumm
        mad_beam = mad.sequence().beam
        mev_to_ev = mathlib.real_type("1e6")
        out = cls(
            pc0=mad_beam.pc * mev_to_ev,
            mass0=mad_beam.mass * mev_to_ev,
            charge0=mad_beam.charge,
            s=tracksumm.s[:],
            x=tracksumm.x[:],
            px=tracksumm.px[:],
            y=tracksumm.y[:],
            py=tracksumm.py[:],
            tau=tracksumm.t[:],
            ptau=tracksumm.pt[:],
            mathlib=mathlib
        )
        return out

    @classmethod
    def from_list(cls, lst, mathlib=MathlibDefault):
        ll = len(lst)
        dct = {nn: np.zeros(ll, dtype=mathlib.real_type)
               for nn in cls._dict_vars}
        for ii, pp in enumerate(lst):
            for nn in cls._dict_vars:
                dct[nn][ii] = getattr(pp, nn, 0)
        return cls(**dct)
