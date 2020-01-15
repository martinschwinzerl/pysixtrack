import numpy as np

from .base_classes import Element
from .be_beamfields.beambeam import BeamBeam4D
from .be_beamfields.beambeam import BeamBeam6D
from .be_beamfields.spacecharge import SpaceChargeCoasting
from .be_beamfields.spacecharge import SpaceChargeBunched

_factorial = np.array(
    [
        1,
        1,
        2,
        6,
        24,
        120,
        720,
        5040,
        40320,
        362880,
        3628800,
        39916800,
        479001600,
        6227020800,
        87178291200,
        1307674368000,
        20922789888000,
        355687428096000,
        6402373705728000,
        121645100408832000,
        2432902008176640000,
    ]
)


class Drift(Element):
    """Drift in expanded form"""

    _description = [("length", "m", "Length of the drift", 0)]

    def track(self, p):
        length = self.length
        rpp = p.rpp
        xp = p.px * rpp
        yp = p.py * rpp
        p.x += xp * length
        p.y += yp * length
        p.zeta += length * (p.rvv - (1 + (xp * xp + yp * yp) / 2))
        p.s += length


class DriftExact(Drift):
    """Drift in exact form"""

    _description = [("length", "m", "Length of the drift", 0)]

    def track(self, p):
        sqrt = p._m.sqrt
        length = self.length
        opd = 1 + p.delta
        lpzi = length / sqrt(opd * opd - p.px ** 2 - p.py ** 2)
        p.x += p.px * lpzi
        p.y += p.py * lpzi
        p.zeta += p.rvv * length - opd * lpzi
        p.s += length


def _arrayofsize(ar, size, dtype=None):
    if dtype is None:
        dtype = ar.dtype
    ar = np.array(ar, dtype=dtype)
    if len(ar) == 0:
        return np.zeros(size, dtype=dtype)
    elif len(ar) < size:
        ar = np.hstack([ar, np.zeros(size - len(ar), dtype=dtype)])
    return ar


class Multipole(Element):
    """ Multipole """

    _description = [
        (
            "knl",
            "m^-n",
            "Normalized integrated strength of normal components",
            lambda: [0],
        ),
        (
            "ksl",
            "m^-n",
            "Normalized integrated strength of skew components",
            lambda: [0],
        ),
        (
            "hxl",
            "rad",
            "Rotation angle of the reference trajectory"
            "in the horizzontal plane",
            0,
        ),
        (
            "hyl",
            "rad",
            "Rotation angle of the reference trajectory in the vertical plane",
            0,
        ),
        ("length", "m", "Length of the originating thick multipole", 0),
    ]

    @property
    def order(self):
        return max(len(self.knl), len(self.ksl)) - 1

    def track(self, p):
        order = self.order
        length = self.length
        hxl = self.hxl
        hyl = self.hyl
        knl = _arrayofsize(self.knl, order + 1)
        ksl = _arrayofsize(self.ksl, order + 1)

        x = p.x
        y = p.y
        chi = p.chi
        dpx = knl[order]
        dpy = ksl[order]
        for ii in range(order, 0, -1):
            zre = (dpx * x - dpy * y) / ii
            zim = (dpx * y + dpy * x) / ii
            dpx = knl[ii - 1] + zre
            dpy = ksl[ii - 1] + zim
        dpx = -chi * dpx
        dpy = chi * dpy

        # curvature effect kick
        if hxl > 0 or hyl > 0:
            b1l = chi * knl[0]
            a1l = chi * ksl[0]
            hxlx = hxl * x
            hyly = hyl * y
            if length > 0:
                hxx = hxlx / length
                hyy = hyly / length
            else:  # non physical weak focusing disabled (SixTrack mode)
                hxx = 0
                hyy = 0
            dpx += hxl + hxl * p.delta - b1l * hxx
            dpy -= hyl + hyl * p.delta - a1l * hyy
            p.zeta -= chi * (hxlx - hyly)
        p.px += dpx
        p.py += dpy


class RFMultipole(Element):
    """
    H= -l sum   Re[ (kn[n](zeta) + i ks[n](zeta) ) (x+iy)**(n+1)/ n ]

    kn[n](z) = k_n cos(2pi w tau + pn/180*pi)
    ks[n](z) = k_n cos(2pi w tau + pn/180*pi)

    """

    _description = [
        ("voltage", "volt", "Voltage", 0),
        ("frequency", "hertz", "Frequency", 0),
        ("lag", "degree", "Delay in the cavity sin(lag - w tau)", 0),
        ("knl", "", "...", lambda: [0]),
        ("ksl", "", "...", lambda: [0]),
        ("pn", "", "...", lambda: [0]),
        ("ps", "", "...", lambda: [0]),
    ]

    @property
    def order(self):
        return max(len(self.knl), len(self.ksl)) - 1

    def track(self, p):
        pi = p._m.pi
        c_light = p._m.clight
        cos = p._m.cos
        sin = p._m.sin
        order = self.order
        k = 2 * pi * self.frequency / c_light
        tau = p.zeta / p.rvv / p.beta0
        ktau = k * tau

        knl = _arrayofsize(self.knl, order + 1)
        ksl = _arrayofsize(self.ksl, order + 1)
        pn = _arrayofsize(self.pn, order + 1)
        ps = _arrayofsize(self.ps, order + 1)

        pn *= pi / 180
        ps *= pi / 180

        x = p.x
        y = p.y
        dpx = 0.0
        dpy = 0.0
        dptr = 0.0
        zre = 1.0
        zim = 0.0
        for ii in range(order + 1):
            pn_ii = pn[ii] - ktau
            ps_ii = ps[ii] - ktau
            cn = cos(pn_ii)
            sn = sin(pn_ii)
            cs = cos(ps_ii)
            ss = sin(ps_ii)
            # transverse kick order i!
            dpx += cn * knl[ii] * zre - cs * ksl[ii] * zim
            dpy += cs * ksl[ii] * zre + cn * knl[ii] * zim
            # compute z**(i+1)/(i+1)!
            zret = (zre * x - zim * y) / (ii + 1)
            zim = (zim * x + zre * y) / (ii + 1)
            zre = zret
            fnr = knl[ii] * zre
            # fni = knl[ii] * zim
            # fsr = ksl[ii] * zre
            fsi = ksl[ii] * zim
            # energy kick order i+1
            dptr += sn * fnr - ss * fsi

        p.px += -p.chi * dpx
        p.py += p.chi * dpy
        dv0 = self.voltage * sin(self.lag * pi / 180 - ktau)
        p.add_to_energy(p.charge * (dv0 - p.pc0 * k * dptr))


class Cavity(Element):
    """Radio-frequency cavity"""

    _description = [
        ("voltage", "V", "Integrated energy change", 0),
        ("frequency", "Hz", "Frequency of the cavity", 0),
        ("lag", "degree", "Delay in the cavity sin(lag - w tau)", 0),
    ]

    def track(self, p):
        pi = p._m.pi
        c_light = p._m.clight
        sin = p._m.sin

        k = 2 * pi * self.frequency / c_light
        tau = p.zeta / p.rvv / p.beta0
        phase = self.lag * pi / 180 - k * tau
        p.add_to_energy(p.charge * self.voltage * sin(phase))


class SawtoothCavity(Element):
    """Radio-frequency cavity"""

    _description = [
        ("voltage", "V", "Integrated energy change", 0),
        ("frequency", "Hz", "Equivalent Frequency of the cavity", 0),
        ("lag", "degree", "Delay in the cavity `lag - w tau`", 0),
    ]

    def track(self, p):
        pi = p._m.pi
        c_light = p._m.clight

        k = 2 * pi * self.frequency / c_light
        tau = p.zeta / p.rvv / p.beta0
        phase = self.lag * pi / 180 - k * tau
        phase = (phase + pi) % (2 * pi) - pi
        p.add_to_energy(p.charge * self.voltage * phase)


class XYShift(Element):
    """shift of the reference"""

    _description = [
        ("dx", "m", "Horizontal shift", 0),
        ("dy", "m", "Vertical shift", 0),
    ]

    def track(self, p):
        p.x -= self.dx
        p.y -= self.dy


class SRotation(Element):
    """anti-clockwise rotation of the reference frame"""

    _description = [("angle", "", "Rotation angle", 0)]

    def track(self, p):
        pi = p._m.pi
        cos = p._m.cos
        sin = p._m.sin
        angle_rad = self.angle * pi / 180
        cz = cos(angle_rad)
        sz = sin(angle_rad)
        xn = cz * p.x + sz * p.y
        yn = -sz * p.x + cz * p.y
        p.x = xn
        p.y = yn
        xn = cz * p.px + sz * p.py
        yn = -sz * p.px + cz * p.py
        p.px = xn
        p.py = yn


class LimitRect(Element):
    _description = [
        ("min_x", "m", "Minimum horizontal aperture", -1.0),
        ("max_x", "m", "Maximum horizontal aperture", 1.0),
        ("min_y", "m", "Minimum vertical aperture", -1.0),
        ("max_y", "m", "Minimum vertical aperture", 1.0),
    ]

    def track(self, particle):
        x = particle.x
        y = particle.y

        new_state = (
            particle.state
            & (x >= self.min_x)
            & (x <= self.max_x)
            & (y >= self.min_y)
            & (y <= self.max_y)
        )

        if new_state != particle.state:
            if particle.is_vector:
                particle.state[:] = new_state[:]
            else:
                particle.state = new_state
            particle.remove_lost_particles()

            if particle.num_particles == 0:
                return "All particles lost"


class LimitEllipse(Element):
    _description = [
        ("a", "m", "Horizontal semiaxis", 1.0),
        ("b", "m", "Vertical semiaxis", 1.0),
    ]

    def track(self, particle):
        x = particle.x
        y = particle.y
        a = self.a
        b = self.b

        new_state = particle.state & (
            (x * x) / (a * a) + (y * y) / (b * b) <= 1.0
        )

        if new_state != particle.state:
            if particle.is_vector:
                particle.state[:] = new_state[:]
            else:
                particle.state = new_state
            particle.remove_lost_particles()
            if particle.num_particles == 0:
                return "All particles lost"


class LimitRectEllipse(Element):
    _description = [
        ("max_x", "m", "Maximum horizontal aperture", 1.0),
        ("max_y", "m", "Maximum vertical aperture", 1.0),
        ("a", "m", "Horizontal semiaxis", 1.0),
        ("b", "m", "Vertical semiaxis", 1.0),
    ]

    def track(self, particle):
        x = particle.x
        y = particle.y
        x_limit = self.max_x
        y_limit = self.max_y
        a = self.a
        b = self.b

        new_state = (
            particle.state
            & (x >= -x_limit)
            & (x <= x_limit)
            & (y >= -y_limit)
            & (y <= y_limit)
            & ((x * x) / (a * a) + (y * y) / (b * b))
            <= 1
        )

        if new_state != particle.state:
            if particle.is_vector:
                particle.state[:] = new_state[:]
            else:
                particle.state = new_state
            particle.remove_lost_particles()
            if particle.num_particles == 0:
                return "All particles lost"


class BeamMonitor(Element):
    _description = [
        ("num_stores", "", "...", 0),
        ("start", "", "...", 0),
        ("skip", "", "...", 1),
        ("max_particle_id", "", "", 0),
        ("min_particle_id", "", "", 0),
        ("is_rolling", "", "", False),
        ("is_turn_ordered", "", "", True),
        ("data", "", "...", lambda: []),
    ]

    def offset(self, particle):
        _offset = -1
        nn = (
            self.max_particle_id >= self.min_particle_id
            and (self.max_particle_id - self.min_particle_id + 1)
            or -1
        )
        assert self.is_turn_ordered

        if (
            particle.turn >= self.start
            and nn > 0
            and particle.partid >= self.min_particle_id
            and particle.partid <= self.max_particle_id
        ):
            turns_since_start = particle.turns - self.start
            store_index = turns_since_start // self.skip
            if store_index < self.num_stores:
                pass
            elif self.is_rolling:
                store_index = store_index % self.num_stores
            else:
                store_index = -1

            if store_index >= 0:
                _offset = store_index * nn + particle.partid

        return _offset

    def track(self, p):
        self.data.append(p.copy)


class DipoleEdge(Element):
    _description = [
        ("h", "1/m", "Curvature", 0),
        ("e1", "rad", "Face angle", 0),
        ("hgap", "m", "Equivalent gap", 0),
        ("fint", "", "Fringe integral", 0),
    ]

    def track(self, p):
        cos = p._m.cos
        sin = p._m.sin
        tan = p._m.tan
        cor = 2 * self.h * self.hgap * self.fint
        r21 = +self.h * tan(self.e1)
        r43 = -self.h * tan(
            self.e1 - cor / cos(self.e1) * (1 + sin(self.e1) ** 2)
        )
        p.px += r21 * p.x
        p.py += r43 * p.y


__all__ = [
    "BeamBeam4D",
    "BeamBeam6D",
    "BeamMonitor",
    "Cavity",
    "DipoleEdge",
    "Drift",
    "DriftExact",
    "Element",
    "LimitEllipse",
    "LimitRect",
    "Multipole",
    "RFMultipole",
    "SRotation",
    "SpaceChargeBunched",
    "SpaceChargeCoasting",
    "XYShift",
]

# =============================================================================


def element_change_type(
    elem, real_type=np.float64, int_type=np.int64, keepextra=False
):
    if isinstance(
        elem,
        (
            Drift,
            DriftExact,
            Multipole,
            RFMultipole,
            Cavity,
            SawtoothCavity,
            XYShift,
            SRotation,
            DipoleEdge,
            LimitEllipse,
            LimitRect,
            LimitRectEllipse,
        ),
    ):
        elem.change_all_fields_type(real_type, keepextra=keepextra)
    elif isinstance(elem, BeamMonitor):
        _no_cast_for = ("is_rolling", "is_turn_ordered", "data")
        elem.change_all_fields_type(
            int_type, keepextra=keepextra, exclude_fields=_no_cast_for
        )
