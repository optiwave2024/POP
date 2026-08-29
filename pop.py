"""
pop.py
============================================================================
A from-scratch implementation of the Zemax "Physical Optics
Propagation" (POP) algorithm to simulate coherent (wave) light transmission
through a lens, and to compute fiber-coupling efficiency.

WHY TWO PROPAGATORS (this is the heart of the Zemax POP method)
----------------------------------------------------------------------------
Zemax POP represents the beam as a 2-D array of complex amplitudes and moves
it surface-by-surface. To go from one plane to the next it uses ONE of two
scalar-diffraction propagators and *automatically picks the more accurate one*:

  (1) Angular-Spectrum "Plane-to-Plane" (PTP)  -> near field / collimated
      E_out = IFFT{ FFT{E_in} * H(fx,fy) },   grid size is PRESERVED.
      Best when the beam stays about the same size (|z| << Rayleigh range).

  (2) Fresnel single-FFT "Waist-to-Plane"      -> far field / through focus
      One FFT with quadratic phase pre/post factors; the output GRID IS
      RESCALED (magnified). Best when the beam changes size a lot, e.g.
      propagating through a focus (|z| >~ Rayleigh range).

Zemax tracks an ideal Gaussian "pilot beam" (via the complex q-parameter /
ABCD matrices) to (a) know the reference sphere / waist location and (b) decide
which propagator gives the highest numerical accuracy. This file reproduces
that logic.

References (grounding):
  * Zemax KB, "Exploring / About Physical Optics Propagation (POP)" (Ansys).
  * Zemax Community, "Explanation of Angular Propagation" (PTP operator).
  * Aumeyr et al., "Advanced simulations of optical transition and diffraction
    radiation", Phys. Rev. ST Accel. Beams 18, 042801 (2015) - benchmarks POP
    against analytic theory.
  * Konijnenberg/Adam/Urbach, "Optics", Ch.6 (Angular Spectrum, Rayleigh-
    Sommerfeld, Fresnel).

Author: generated for optiwave2024 engineering use.
All lengths are in METERS internally. Convenience wrappers accept mm/um.
============================================================================
"""

from __future__ import annotations
import numpy as np


# ----------------------------------------------------------------------------
# 1.  COMPLEX WAVEFRONT CONTAINER
# ----------------------------------------------------------------------------
class Wavefront:
    """A sampled complex scalar field E(x,y) on a uniform square grid.

    Attributes
    ----------
    E    : (N,N) complex ndarray   -- complex amplitude
    L    : float                   -- physical side length of the grid [m]
    wl   : float                   -- wavelength [m]
    z    : float                   -- current axial position [m] (bookkeeping)
    """

    def __init__(self, E, L, wl, z=0.0):
        self.E = np.asarray(E, dtype=np.complex128)
        self.N = self.E.shape[0]
        self.L = float(L)
        self.wl = float(wl)
        self.z = float(z)

    # ---- grid helpers -------------------------------------------------------
    @property
    def dx(self):
        """Sample spacing [m]."""
        return self.L / self.N

    def coords(self):
        """Return (x, y) 1-D coordinate axes centred on 0 [m]."""
        n = np.arange(self.N) - self.N // 2
        x = n * self.dx
        return x, x.copy()

    def meshgrid(self):
        x, y = self.coords()
        return np.meshgrid(x, y)

    # ---- physical quantities ------------------------------------------------
    @property
    def k(self):
        return 2.0 * np.pi / self.wl

    def intensity(self):
        return np.abs(self.E) ** 2

    def total_power(self):
        return np.sum(self.intensity()) * self.dx ** 2

    def normalize_power(self, P=1.0):
        cur = self.total_power()
        if cur > 0:
            self.E *= np.sqrt(P / cur)
        return self

    def copy(self):
        return Wavefront(self.E.copy(), self.L, self.wl, self.z)


# ----------------------------------------------------------------------------
# 2.  PILOT GAUSSIAN BEAM  (drives the algorithm-selection logic)
# ----------------------------------------------------------------------------
class PilotBeam:
    """Ideal Gaussian beam tracked with the complex q-parameter.

    q(z) = z + i * zR   (measured from the waist);  1/q = 1/R - i*lambda/(pi w^2)

    ABCD update:  q' = (A q + B) / (C q + D)
        free space L:  [[1, L],[0,1]]
        thin lens  f:  [[1, 0],[-1/f, 1]]
    """

    def __init__(self, w0, wl, z_from_waist=0.0):
        self.wl = float(wl)
        self.zR = np.pi * w0 ** 2 / self.wl          # Rayleigh range
        self.q = complex(z_from_waist, self.zR)      # q at current plane

    @classmethod
    def from_waist(cls, w0, wl):
        return cls(w0, wl, 0.0)

    # ---- current beam parameters -------------------------------------------
    @property
    def w(self):
        """1/e^2 amplitude radius at the current plane [m]."""
        inv_q = 1.0 / self.q
        return np.sqrt(-self.wl / (np.pi * inv_q.imag))

    @property
    def R(self):
        """Radius of curvature at current plane [m] (inf at waist)."""
        inv_q = 1.0 / self.q
        return np.inf if abs(inv_q.real) < 1e-30 else 1.0 / inv_q.real

    @property
    def dist_to_waist(self):
        """Signed distance from current plane to the waist [m].
        Positive means the waist is *ahead* (downstream)."""
        return -self.q.real

    @property
    def rayleigh(self):
        return self.q.imag

    # ---- ABCD operations ----------------------------------------------------
    def _apply(self, A, B, C, D):
        self.q = (A * self.q + B) / (C * self.q + D)

    def propagate(self, L):
        self._apply(1.0, L, 0.0, 1.0)

    def lens(self, f):
        self._apply(1.0, 0.0, -1.0 / f, 1.0)


# ----------------------------------------------------------------------------
# 3.  FREE-SPACE PROPAGATORS
# ----------------------------------------------------------------------------
def propagate_angular_spectrum(wf: Wavefront, dz: float,
                               fresnel_approx: bool = False) -> Wavefront:
    """Angular-Spectrum "Plane-to-Plane" propagation (grid preserved).

    E_out = IFFT{ FFT{E_in} * H(fx,fy) }

    Exact Rayleigh-Sommerfeld transfer function:
        H = exp( i * 2*pi * dz * sqrt(1/wl^2 - fx^2 - fy^2) )   (evanescent killed)
    Fresnel (paraxial) transfer function:
        H = exp( i*k*dz ) * exp( -i*pi*wl*dz*(fx^2+fy^2) )
    """
    N, dx, wl, k = wf.N, wf.dx, wf.wl, wf.k
    fx = np.fft.fftfreq(N, d=dx)
    FX, FY = np.meshgrid(fx, fx)

    if fresnel_approx:
        H = np.exp(1j * k * dz) * np.exp(-1j * np.pi * wl * dz * (FX**2 + FY**2))
    else:
        arg = 1.0 / wl**2 - FX**2 - FY**2
        evan = arg < 0                          # evanescent components
        arg = np.clip(arg, 0, None)
        H = np.exp(1j * 2 * np.pi * dz * np.sqrt(arg))
        H[evan] = 0.0

    A = np.fft.fft2(np.fft.ifftshift(wf.E))
    E_out = np.fft.fftshift(np.fft.ifft2(A * H))
    return Wavefront(E_out, wf.L, wl, wf.z + dz)


def propagate_fresnel_1fft(wf: Wavefront, dz: float) -> Wavefront:
    """Fresnel single-FFT "Waist-to-Plane" propagation (grid RESCALED).

    Implements the Fresnel diffraction integral evaluated with ONE FFT:

        E(x2,y2) = (e^{ikz}/(i wl z)) * e^{i k/(2z)(x2^2+y2^2)}
                   * FT{ E(x1,y1) e^{i k/(2z)(x1^2+y1^2)} }

    The output sample spacing becomes  dx2 = wl*|z| / (N*dx1), so the physical
    window magnifies/demagnifies. This is the propagator Zemax favours through
    a focus, where the beam size changes strongly.
    """
    N, dx1, wl, k = wf.N, wf.dx, wf.wl, wf.k
    x1, y1 = wf.coords()
    X1, Y1 = np.meshgrid(x1, y1)

    dx2 = wl * abs(dz) / (N * dx1)              # new (rescaled) spacing
    x2 = (np.arange(N) - N // 2) * dx2
    X2, Y2 = np.meshgrid(x2, x2)

    pre = np.exp(1j * k / (2 * dz) * (X1**2 + Y1**2))
    U = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(wf.E * pre))) * (dx1**2)

    post = (np.exp(1j * k * dz) / (1j * wl * dz)
            * np.exp(1j * k / (2 * dz) * (X2**2 + Y2**2)))
    E_out = post * U
    return Wavefront(E_out, N * dx2, wl, wf.z + dz)


# ----------------------------------------------------------------------------
# 4.  ZEMAX-STYLE AUTO-SELECTING PROPAGATOR
# ----------------------------------------------------------------------------
def propagate(wf: Wavefront, dz: float, pilot: PilotBeam | None = None,
              force: str | None = None, verbose: bool = False) -> Wavefront:
    """Propagate a distance dz choosing the more accurate algorithm, à la POP.

    Selection rule (mirrors Zemax behaviour):
      * If the pilot beam stays in its near field over this hop
        (|dz| <= zR  AND the beam does not cross its waist) -> ANGULAR SPECTRUM,
        which preserves the grid and is exact for near-field/collimated beams.
      * Otherwise (long hop, or the segment passes through the waist/focus)
        -> FRESNEL single-FFT, which rescales the grid to follow the beam.

    Parameters
    ----------
    force : None | 'as' | 'fresnel'    -- override the automatic choice.
    """
    if force == 'as':
        method = 'Angular-Spectrum (forced)'
        out = propagate_angular_spectrum(wf, dz)
    elif force == 'fresnel':
        method = 'Fresnel-1FFT (forced)'
        out = propagate_fresnel_1fft(wf, dz)
    else:
        use_as = True
        if pilot is not None:
            zR = pilot.rayleigh
            d_waist = pilot.dist_to_waist
            crosses_waist = (0.0 < d_waist < dz) or (dz < d_waist < 0.0)
            far_hop = abs(dz) > zR
            use_as = (not crosses_waist) and (not far_hop)
        else:
            use_as = abs(dz) < 1e-3     # fallback heuristic w/o a pilot beam
        if use_as:
            method = 'Angular-Spectrum (auto)'
            out = propagate_angular_spectrum(wf, dz)
        else:
            method = 'Fresnel-1FFT (auto)'
            out = propagate_fresnel_1fft(wf, dz)

    if pilot is not None:
        pilot.propagate(dz)
    if verbose:
        print(f"    propagate {dz*1e3:8.3f} mm  ->  {method}"
              + (f"   [pilot w={pilot.w*1e6:7.2f} um]" if pilot else ""))
    return out


# ----------------------------------------------------------------------------
# 5.  OPTICAL ELEMENTS (thin lens, aperture)
# ----------------------------------------------------------------------------
def apply_thin_lens(wf: Wavefront, f: float, pilot: PilotBeam | None = None,
                    D: float | None = None) -> Wavefront:
    """Multiply by an ideal thin-lens phase  t = exp(-i k r^2 / (2 f)).

    Optionally clip to a circular clear aperture of diameter D (vignetting).
    """
    X, Y = wf.meshgrid()
    r2 = X**2 + Y**2
    t = np.exp(-1j * wf.k * r2 / (2.0 * f))
    if D is not None:
        t *= (r2 <= (D / 2.0) ** 2)
    out = Wavefront(wf.E * t, wf.L, wf.wl, wf.z)
    if pilot is not None:
        pilot.lens(f)
    return out


def apply_aperture(wf: Wavefront, D: float) -> Wavefront:
    """Hard circular aperture (diameter D)."""
    X, Y = wf.meshgrid()
    mask = (X**2 + Y**2) <= (D / 2.0) ** 2
    return Wavefront(wf.E * mask, wf.L, wf.wl, wf.z)


# ----------------------------------------------------------------------------
# 6.  SOURCES AND FIBER MODES
# ----------------------------------------------------------------------------
def gaussian_source(N, L, wl, w0, z_from_waist=0.0, power=1.0):
    """Create a Gaussian beam wavefront (optionally offset from its waist)."""
    wf = Wavefront(np.zeros((N, N)), L, wl)
    X, Y = wf.meshgrid()
    r2 = X**2 + Y**2
    pilot = PilotBeam(w0, wl, z_from_waist)
    w = pilot.w
    R = pilot.R
    amp = (w0 / w) * np.exp(-r2 / w**2)
    phase = np.zeros_like(r2) if np.isinf(R) else np.exp(1j * wf.k * r2 / (2 * R))
    gouy = np.exp(-1j * np.arctan2(z_from_waist, pilot.rayleigh))
    wf.E = amp * (phase if np.iscomplexobj(phase) else 1.0) * gouy
    wf.normalize_power(power)
    return wf, pilot


def fiber_mode_gaussian(wf_like: Wavefront, mfd):
    """Return a normalised Gaussian fiber mode field on the same grid.

    mfd = mode-field diameter (1/e^2 intensity). The mode-field radius is
    w_f = mfd/2, so the amplitude 1/e radius equals w_f.
    """
    X, Y = wf_like.meshgrid()
    wf_radius = mfd / 2.0
    mode = np.exp(-(X**2 + Y**2) / wf_radius**2)
    norm = np.sqrt(np.sum(np.abs(mode)**2) * wf_like.dx**2)
    return mode / norm


# ----------------------------------------------------------------------------
# 7.  FIBER-COUPLING EFFICIENCY  (overlap integral)
# ----------------------------------------------------------------------------
def coupling_efficiency(wf: Wavefront, fiber_mode):
    """Power-coupling efficiency into a fiber mode via the overlap integral:

              | integral( E_beam * conj(E_mode) ) |^2
        eta = ---------------------------------------------
              integral(|E_beam|^2) * integral(|E_mode|^2)

    Returns eta in [0,1]  (this is the "receiver/mode-matching" efficiency;
    multiply by the optical-system transmission to get total CE, exactly as in
    your TOSA POP workflow).
    """
    dA = wf.dx ** 2
    num = np.abs(np.sum(wf.E * np.conj(fiber_mode)) * dA) ** 2
    den = (np.sum(np.abs(wf.E)**2) * dA) * (np.sum(np.abs(fiber_mode)**2) * dA)
    return float(num / den)


# ----------------------------------------------------------------------------
# 8.  BEAM DIAGNOSTICS
# ----------------------------------------------------------------------------
def second_moment_radius(wf: Wavefront):
    """1/e^2 intensity radius from the second moment (D4sigma / 2)."""
    X, Y = wf.meshgrid()
    I = wf.intensity()
    P = I.sum()
    if P <= 0:
        return 0.0
    xc = (X * I).sum() / P
    yc = (Y * I).sum() / P
    sx2 = ((X - xc)**2 * I).sum() / P
    sy2 = ((Y - yc)**2 * I).sum() / P
    return 2.0 * np.sqrt(0.5 * (sx2 + sy2))     # = 2*sigma (== w for a Gaussian)
