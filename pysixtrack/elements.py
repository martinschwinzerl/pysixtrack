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

    def convert_types(self, real_type=np.float64, int_type=np.int64):
        self.length = real_type(self.length)

    def track(self, p):
        real_t = p._real_type
        length = real_t(self.length)
        rpp = p.rpp
        xp = p.px * rpp
        yp = p.py * rpp
        p.x += xp * length
        p.y += yp * length
        p.zeta += length * (p.rvv - (real_t(1) + (xp * xp + yp * yp ) / real_t(2)))
        p.s += length


class DriftExact(Drift):
    """Drift in exact form"""

    _description = [("length", "m", "Length of the drift", 0)]

    def convert_types(self, real_type=np.float64, int_type=np.int64):
        self.length = real_type(self.length)

    def track(self, p):
        sqrt = p._m.sqrt
        real_t = p._real_type
        length = real_t(self.length)
        opd = real_t(1) + p.delta
        lpzi = length / sqrt(opd * opd - p.px ** 2 - p.py ** 2)
        p.x += p.px * lpzi
        p.y += p.py * lpzi
        p.zeta += p.rvv * length - opd * lpzi
        p.s += length


def _arrayofsize(ar, size):
    ar = np.array(ar)
    if len(ar) == 0:
        return np.zeros(size, dtype=ar.dtype)
    elif len(ar) < size:
        ar = np.hstack([ar, np.zeros(size - len(ar), dtype=ar.dtype)])
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

    def convert_types(self, real_type=np.float64, int_type=np.int64):
        self.order = int_type(self.order)
        self.hxl = real_type(self.hxl)
        self.hyl = real_type(self.hyl)
        self.length = real_type(self.length)

        _knl = [ real_type( k ) for k in self.knl ]
        _ksl = [ real_type( k ) for k in self.ksl ]

        self.knl = np.array( _knl )
        self.ksl = np.array( _ksl )

    def track(self, p):
        real_t = p._real_type
        order = self.order
        length = self.length
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
        hxl = self.hxl
        hyl = self.hyl
        delta = p.delta
        if hxl > real_t(0) or hyl > real_t(0):
            b1l = chi * knl[0]
            a1l = chi * ksl[0]
            hxlx = hxl * x
            hyly = hyl * y
            if length > real_t(0):
                hxx = hxlx / length
                hyy = hyly / length
            else:  # non physical weak focusing disabled (SixTrack mode)
                hxx = 0
                hyy = 0
            dpx += hxl + hxl * delta - b1l * hxx
            dpy -= hyl + hyl * delta - a1l * hyy
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

    def convert_types(self, real_type=np.float64, int_type=np.int64):
        self.voltage = real_type(self.voltage)
        self.frequency = real_type(self.frequency)
        self.lag = real_type(self.lag)

        _knl = [ real_type( k ) for k in self.knl ]
        _ksl = [ real_type( k ) for k in self.ksl ]
        _pn  = [ real_type( p ) for p in self.pn ]
        _ps  = [ real_type( p ) for p in self.ps ]

        self.knl = np.array( _knl )
        self.ksl = np.array( _ksl )
        self.pn = np.array( _pn )
        self.ps = np.array( _ps )


    def track(self, p):
        real_t = p._real_type
        sin = p._m.sin
        cos = p._m.cos
        pi = p.PI
        order = self.order
        k = real_t(2) * pi * self.frequency / p.CLIGHT
        tau = p.zeta / p.rvv / p.beta0
        ktau = k * tau
        deg2rad = pi / real_t(180)
        knl = _arrayofsize(self.knl, order + 1)
        ksl = _arrayofsize(self.ksl, order + 1)
        pn = _arrayofsize(self.pn, order + 1) * deg2rad
        ps = _arrayofsize(self.ps, order + 1) * deg2rad
        x = p.x
        y = p.y
        dpx = real_t(0)
        dpy = real_t(0)
        dptr = real_t(0)
        zre = real_t(1)
        zim = real_t(0)
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
            zret = (zre * x - zim * y) / real_t(ii + 1)
            zim = (zim * x + zre * y) / real_t(ii + 1)
            zre = zret
            fnr = knl[ii] * zre
            # fni = knl[ii] * zim
            # fsr = ksl[ii] * zre
            fsi = ksl[ii] * zim
            # energy kick order i+1
            dptr += sn * fnr - ss * fsi

        chi = p.chi
        p.px += -chi * dpx
        p.py += chi * dpy
        dv0 = self.voltage * sin(self.lag * deg2rad - ktau)
        p.add_to_energy(p.qratio * p.q0 * (dv0 - p.p0c * k * dptr))


class Cavity(Element):
    """Radio-frequency cavity"""

    _description = [
        ("voltage", "V", "Integrated energy change", 0),
        ("frequency", "Hz", "Frequency of the cavity", 0),
        ("lag", "degree", "Delay in the cavity sin(lag - w tau)", 0),
    ]

    def convert_types(self, real_type=np.float64, int_type=np.int64):
        self.voltage = real_type(self.voltage)
        self.frequency = real_type(self.frequency)
        self.lag = real_type(self.lag)

    def track(self, p):
        sin = p._m.sin
        pi = p.PI
        real_t = p._real_type
        k = real_t( 2 ) * pi * self.frequency / p.CLIGHT
        tau = p.zeta / p.rvv / p.beta0
        phase = self.lag * pi / real_t(180) - k * tau
        p.add_to_energy(p.qratio * p.q0 * self.voltage * sin(phase))


class SawtoothCavity(Element):
    """Radio-frequency cavity"""

    _description = [
        ("voltage", "V", "Integrated energy change", 0),
        ("frequency", "Hz", "Equivalent Frequency of the cavity", 0),
        ("lag", "degree", "Delay in the cavity `lag - w tau`", 0),
    ]

    def convert_types(self, real_type=np.float64, int_type=np.int64):
        self.voltage = real_type(self.voltage)
        self.frequency = real_type(self.frequency)
        self.lag = real_type(self.lag)

    def track(self, p):
        pi = p.PI
        real_t = p._real_type
        k = real_t(2) * pi * self.frequency / p.CLIGHT
        tau = p.zeta / p.rvv / p.beta0
        phase = self.lag * pi / real_t(180) - k * tau
        phase = (phase + pi) % (real_t(2) * pi) - pi
        p.add_to_energy(p.qratio * p.q0 * self.voltage * phase)


class XYShift(Element):
    """shift of the reference"""

    _description = [
        ("dx", "m", "Horizontal shift", 0),
        ("dy", "m", "Vertical shift", 0),
    ]

    def convert_types(self, real_type=np.float64, int_type=np.int64):
        self.dx = real_type(self.dx)
        self.dy = real_type(self.dy)

    def track(self, p):
        p.x -= self.dx
        p.y -= self.dy


class SRotation(Element):
    """anti-clockwise rotation of the reference frame"""

    _description = [("angle", "", "Rotation angle", 0)]

    def convert_types(self, real_type=np.float64, int_type=np.int64):
        self.angle = real_type(self.angle)

    def track(self, p):
        deg2rag = p.PI / p._real_type(180)
        cz = p._m.cos(self.angle * deg2rag)
        sz = p._m.sin(self.angle * deg2rag)
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

    def convert_types(self, real_type=np.float64, int_type=np.int64):
        self.min_x = real_type(self.min_x)
        self.max_x = real_type(self.max_x)
        self.min_y = real_type(self.min_y)
        self.max_y = real_type(self.max_y)

    def track(self, particle):

        x = particle.x
        y = particle.y

        if not hasattr(particle.state, "__iter__"):
            particle.state = particle._int_type(
                x >= self.min_x
                and x <= self.max_x
                and y >= self.min_y
                and y <= self.max_y
            )
            if particle.state != particle._int_type(1):
                return "Particle lost"
        else:
            particle.state = particle._int_type(
                (x >= self.min_x)
                & (x <= self.max_x)
                & (y >= self.min_y)
                & (y <= self.max_y)
            )
            particle.remove_lost_particles()
            if len(particle.state) == particle._int_type(0):
                return "All particles lost"


class LimitEllipse(Element):
    _description = [
        ("a", "m", "Horizontal semiaxis", 1.0),
        ("b", "m", "Vertical semiaxis", 1.0),
    ]

    def convert_types(self, real_type=np.float64, int_type=np.int64):
        self.a = real_type(self.a)
        self.b = real_type(self.b)

    def track(self, particle):

        x = particle.x
        y = particle.y

        if not hasattr(particle.state, "__iter__"):
            particle.state = particle._int_type(
                x * x / (self.a * self.a) + y * y / (self.b * self.b) <= 1.0
            )
            if particle.state != particle._int_type(1):
                return "Particle lost"
        else:
            particle.state = particle._int_type(
                x * x / (self.a * self.a) + y * y / (self.b * self.b) <= 1.0
            )
            particle.remove_lost_particles()
            if len(particle.state) == particle._int_type(0):
                return "All particles lost"


class LimitRectEllipse(Element):
    _description = [
        ("max_x", "m", "Maximum horizontal aperture", 1.0),
        ("max_y", "m", "Maximum vertical aperture", 1.0),
        ("a", "m", "Horizontal semiaxis", 1.0),
        ("b", "m", "Vertical semiaxis", 1.0),
    ]

    def convert_types(self, real_type=np.float64, int_type=np.int64):
        self.max_x = real_type(self.max_x)
        self.max_y = real_type(self.max_y)
        self.a = real_type(self.a)
        self.b = real_type(self.b)


    def track(self, particle):

        x = particle.x
        y = particle.y

        if not hasattr(particle.state, "__iter__"):
            particle.state = particle._int_type(
                x >= -self.max_x
                and x <= self.max_x
                and y >= -self.max_y
                and y <= self.max_y
                and x * x / (self.a * self.a) + y * y / (self.b * self.b)
                <= 1.0
            )
            if particle.state != particle._int_type(1):
                return "Particle lost"
        else:
            particle.state = particle._int_type(
                (x >= -self.max_x)
                & (x <= self.max_x)
                & (y >= -self.max_y)
                & (y <= self.max_y)
                & (
                    x * x / (self.a * self.a) + y * y / (self.b * self.b)
                    <= 1.0
                )
            )
            particle.remove_lost_particles()
            if len(particle.state) == particle._int_type(0):
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

    def convert_types(self, real_type=np.float64, int_type=np.int64):
        self.num_stores = int_type(self.num_stores)
        self.start = int_type(self.start)
        self.skip = int_type(self.skip)
        self.max_particle_id = int_type(self.max_particle_id)
        self.min_particle_id = int_type(self.min_particle_id)
        self.is_rolling = int_type(self.is_rolling)
        self.is_turn_ordered = int_type(self.is_turn_ordered)

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

    def convert_types(self, real_type=np.float64, int_type=np.int64):
        self.h = real_type(self.h)
        self.e1 = real_type(self.e1)
        self.hgap = real_type(self.hgap)
        self.fint= real_type(self.fint)

    def track(self, p):
        tan = p._m.tan
        sin = p._m.sin
        cos = p._m.cos
        corr = 2 * self.h * self.hgap * self.fint
        r21 = self.h * tan(self.e1)
        r43 = -self.h * tan(
            self.e1 - corr / cos(self.e1) * (p._real_type(1) + sin(self.e1) ** 2)
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
