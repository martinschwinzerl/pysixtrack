import numpy as np
from .particles import _make_attr

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
        length = p._m.real_type(self.length)
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
        length = p._m.real_type(self.length)
        opd = 1 + p.delta
        lpzi = length / p._m.sqrt(opd * opd - p.px ** 2 - p.py ** 2)
        p.x += p.px * lpzi
        p.y += p.py * lpzi
        p.zeta += p.rvv * length - opd * lpzi
        p.s += length


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
        real_t = p._m.real_type
        ZERO = real_t(0)
        order = real_t(self.order)
        length = real_t(self.length)
        hxl = real_t(self.hxl)
        hyl = real_t(self.hyl)

        always_vec = True
        knl = _make_attr(self.knl, order + 1, real_t, always_vec)
        ksl = _make_attr(self.ksl, order + 1, real_t, always_vec)
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
        if hxl > ZERO or hyl > ZERO:
            b1l = chi * knl[0]
            a1l = chi * ksl[0]
            hxlx = hxl * x
            hyly = hyl * y
            if length > 0:
                hxx = hxlx / length
                hyy = hyly / length
            else:  # non physical weak focusing disabled (SixTrack mode)
                hxx = ZERO
                hyy = ZERO
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
        real_t = p._m.real_type
        ZERO = real_t(0)
        PI = pi = p._m.pi
        DEG2RAD = PI / 180
        sin = p._m.sin
        cos = p._m.cos

        order = real_t(self.order)
        voltage = real_t(self.voltage)
        lag = real_t(self.lag)

        k = 2 * PI * self.frequency / p._m.clight
        tau = p.zeta / p.rvv / p.beta0
        ktau = k * tau

        always_vec = True
        knl = _make_attr(self.knl, order + 1, real_t, always_vec)
        ksl = _make_attr(self.ksl, order + 1, real_t, always_vec)
        pn = _make_attr(self.pn, order + 1, real_t, always_vec) * DEG2RAD
        ps = _make_attr(self.ps, order + 1, real_t, always_vec) * DEG2RAD

        x = p.x
        y = p.y
        dpx = ZERO
        dpy = ZERO
        dptr = ZERO
        zre = p._m.real_type(1)
        zim = ZERO
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
        dv0 = voltage * sin(lag * DEG2RAD - ktau)
        p.add_to_energy(p.charge * (dv0 - p.pc0 * k * dptr))


class Cavity(Element):
    """Radio-frequency cavity"""

    _description = [
        ("voltage", "V", "Integrated energy change", 0),
        ("frequency", "Hz", "Frequency of the cavity", 0),
        ("lag", "degree", "Delay in the cavity sin(lag - w tau)", 0),
    ]

    def track(self, p):
        real_t = p._m.real_type
        k = 2 * p._m.pi * real_t(self.frequency) / p._m.clight
        tau = p.zeta / p.rvv / p.beta0
        phase = real_t(self.lag) * p._m.pi / 180 - k * tau
        p.add_to_energy(p.charge * real_t(self.voltage) * p._m.sin(phase))


class SawtoothCavity(Element):
    """Radio-frequency cavity"""

    _description = [
        ("voltage", "V", "Integrated energy change", 0),
        ("frequency", "Hz", "Equivalent Frequency of the cavity", 0),
        ("lag", "degree", "Delay in the cavity `lag - w tau`", 0),
    ]

    def track(self, p):
        real_t = p._m.real_type
        PI = p._m.pi

        k = 2 * PI * real_t(self.frequency) / p._m.clight
        tau = p.zeta / p.rvv / p.beta0
        phase = real_t(self.lag) * PI / 180 - k * tau
        phase = (phase + PI) % (2 * PI) - PI
        p.add_to_energy(p.charge * real_t(self.voltage) * phase)


class XYShift(Element):
    """shift of the reference"""

    _description = [
        ("dx", "m", "Horizontal shift", 0),
        ("dy", "m", "Vertical shift", 0),
    ]

    def track(self, p):
        p.x -= p._m.real_type(self.dx)
        p.y -= p._m.real_type(self.dy)


class SRotation(Element):
    """anti-clockwise rotation of the reference frame"""

    _description = [("angle", "", "Rotation angle", 0)]

    def track(self, p):
        DEG2RAD = p._m.pi / 180
        angle_rad = p._m.real_type(self.angle) * DEG2RAD
        cz = p._m.cos(angle_rad)
        sz = p._m.sin(angle_rad)
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

        real_t = particle._m.real_type
        x = particle.x
        y = particle.y

        new_state = (particle.state) & \
            (x >= real_t(self.min_x)) & (x <= real_t(self.max_x)) & \
            (y >= real_t(self.min_y)) & (y <= real_t(self.max_y))

        has_particles_left = particle.num_particles > 0

        if particle.is_vector:
            if particle.state != new_state:
                particle.state[:] = new_state[:]
                particle.remove_lost_particles()
                has_particles_left = particle.num_particles > 0
        elif particle.state == particle._m.int_type(0):
            has_particles_left = False

        if not has_particles_left:
            return "All particles lost"


class LimitEllipse(Element):
    _description = [
        ("a", "m", "Horizontal semiaxis", 1.0),
        ("b", "m", "Vertical semiaxis", 1.0),
    ]

    def track(self, particle):
        real_t = particle._m.real_type
        x_squ = particle.x * particle.x
        y_squ = particle.y * particle.y
        a_squ = real_t(self.a) * real_t(self.a)
        b_squ = real_t(self.b) * real_t(self.b)

        new_state = (particle.state) & \
            ((x_squ / a_squ + y_squ / b_squ) <= real_t(1))

        has_particles_left = particle.num_particles > 0

        if particle.is_vector:
            if particle.state != new_state:
                particle.state[:] = new_state[:]
                particle.remove_lost_particles()
                has_particles_left = particle.num_particles > 0
        elif particle.state == particle._m.int_type(0):
            has_particles_left = False

        if not has_particles_left:
            return "All particles lost"


class LimitRectEllipse(Element):
    _description = [
        ("max_x", "m", "Maximum horizontal aperture", 1.0),
        ("max_y", "m", "Maximum vertical aperture", 1.0),
        ("a", "m", "Horizontal semiaxis", 1.0),
        ("b", "m", "Vertical semiaxis", 1.0),
    ]

    def track(self, particle):
        real_t = particle._m.real_type
        x = particle.x
        y = particle.y
        a_squ = real_t(self.a) * real_t(self.a)
        b_squ = real_t(self.b) * real_t(self.b)
        x_limit = real_t(self.max_x)
        y_limit = real_t(self.max_y)

        new_state = (particle.state) & \
            (x >= -x_limit) & (x <= x_limit) & \
            (y >= -y_limit) & (y <= y_limit) & \
            (((x * x) / a_squ + (y * y) / b_squ) <= real_t(1))

        has_particles_left = particle.num_particles > 0

        if particle.is_vector:
            if particle.state != new_state:
                particle.state[:] = new_state[:]
                particle.remove_lost_particles()
                has_particles_left = particle.num_particles > 0
        elif particle.state == particle._m.int_type(0):
            has_particles_left = False

        if not has_particles_left:
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
        real_t = p._m.real_type
        h = real_t(self.h)
        hgap = real_t(self.hgap)
        fint = real_t(self.fint)
        e1 = real_t(self.e1)

        corr = 2 * h * hgap * fint
        r21 = h * p._m.tan(e1)
        r43 = -h * p._m.tan(e1 - corr / p._m.cos(e1) * (1 + p._m.sin(e1) ** 2))
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
