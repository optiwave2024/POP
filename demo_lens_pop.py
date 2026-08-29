"""
demo_lens_pop.py
============================================================================
Demonstration + validation of the Zemax-style POP engine in `zemax_pop.py`.

Scenario (a simplified TOSA-style coupling problem):
    Gaussian source (DFB-like)  ->  free space  ->  thin lens  ->
    free space to focus  ->  single-mode fiber (Gaussian mode)

We do three things:
  (A) VALIDATE the propagators against the exact analytic Gaussian solution.
  (B) Propagate a collimated beam THROUGH A LENS to its focus and watch the
      algorithm auto-switch from Angular-Spectrum to Fresnel near focus.
  (C) Compute the fiber-coupling efficiency (mode-overlap) at best focus and
      scan it vs. axial fiber position (a POPD-style CE curve).

Outputs: console report + a PNG figure "pop_lens_simulation.png".
============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from zemax_pop import (Wavefront, PilotBeam, gaussian_source, propagate,
                       propagate_angular_spectrum, propagate_fresnel_1fft,
                       apply_thin_lens, fiber_mode_gaussian,
                       coupling_efficiency, second_moment_radius)

um = 1e-6
mm = 1e-3

# ---------------------------------------------------------------------------
# (A) VALIDATION: analytic Gaussian vs. numerical propagators
# ---------------------------------------------------------------------------
print("=" * 74)
print("(A)  VALIDATION  -  numerical propagator vs. analytic Gaussian beam")
print("=" * 74)

wl = 1.31 * um                 # 1310 nm (your DFB wavelength)
w0 = 5.0 * um                  # 5 um waist  -> strongly diverging (like a DFB)
N = 512
L = 400 * um                   # grid window
zR = np.pi * w0**2 / wl
print(f"  wavelength      = {wl*1e9:.1f} nm")
print(f"  waist  w0       = {w0*1e6:.2f} um")
print(f"  Rayleigh range  = {zR*1e6:.2f} um  ({zR*1e3:.4f} mm)")

def analytic_w(z):
    return w0 * np.sqrt(1 + (z / zR) ** 2)

for z in [0.5 * zR, 5 * zR, 50 * zR]:
    wf0, _ = gaussian_source(N, L, wl, w0, z_from_waist=0.0)
    # near-field angular spectrum
    wf_as = propagate_angular_spectrum(wf0, z)
    w_as = second_moment_radius(wf_as)
    # far-field Fresnel
    wf_fr = propagate_fresnel_1fft(wf0, z)
    w_fr = second_moment_radius(wf_fr)
    w_th = analytic_w(z)
    print(f"\n  z = {z*1e6:8.1f} um  ({z/zR:5.1f} zR):  analytic w = {w_th*1e6:8.2f} um")
    print(f"     Angular-Spectrum w = {w_as*1e6:8.2f} um "
          f"(err {100*(w_as-w_th)/w_th:+5.2f}%")
    print(f"     Fresnel-1FFT    w = {w_fr*1e6:8.2f} um "
          f"(err {100*(w_fr-w_th)/w_th:+5.2f}%")

# ---------------------------------------------------------------------------
# (B) COLLIMATED BEAM THROUGH A LENS -> FOCUS  (auto algorithm switching)
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("(B)  WAVE PROPAGATION THROUGH A LENS  (POP auto algorithm selection)")
print("=" * 74)

wl = 1.31 * um
w_in = 0.5 * mm                # collimated input beam, 0.5 mm waist radius
f = 8.0 * mm                   # lens focal length
N = 1024
L = 8.0 * mm                   # input window big enough for the collimated beam

print(f"  Input: collimated Gaussian, w = {w_in*1e3:.2f} mm, f = {f*1e3:.1f} mm")
print(f"  Expected focal spot (Gaussian): w0' = f*wl/(pi*w_in) = "
      f"{f*wl/(np.pi*w_in)*1e6:.2f} um")


def build_post_lens():
    """Fresh collimated Gaussian -> thin lens; return (field, pilot)."""
    wf, pilot = gaussian_source(N, L, wl, w_in, z_from_waist=0.0, power=1.0)
    wf = apply_thin_lens(wf, f, pilot=pilot)
    return wf, pilot


def beam_at(z):
    """Propagate the post-lens field a SINGLE hop of length z (POP style:
    propagate from the reference plane, letting propagate() auto-select the
    algorithm and rescale the grid to follow the converging beam)."""
    wf, pilot = build_post_lens()
    if z <= 0:
        return wf
    return propagate(wf, z, pilot=pilot)


_, post_pilot = build_post_lens()
print(f"  After lens: pilot predicts waist {post_pilot.dist_to_waist*1e3:.3f} mm"
      f" downstream, focal w0 = {f*wl/(np.pi*w_in)*1e6:.2f} um\n")

# Sample the beam at many planes (each a single hop from the reference plane)
z_positions = np.linspace(0.2 * f, 1.6 * f, 25)
widths = []
print("  Sampling beam size vs. z (single-hop from post-lens reference plane):")
for i, z in enumerate(z_positions):
    wf0, p0 = build_post_lens()
    wfz = propagate(wf0, z, pilot=p0, verbose=(i % 5 == 0))
    widths.append(second_moment_radius(wfz))
widths = np.array(widths)
i_focus = int(np.argmin(widths))
print(f"\n  Minimum simulated spot: w = {widths[i_focus]*1e6:.2f} um "
      f"at z = {z_positions[i_focus]*1e3:.3f} mm  "
      f"(analytic focus at {post_pilot.dist_to_waist*1e3:.3f} mm)")

# ---------------------------------------------------------------------------
# (C) FIBER-COUPLING EFFICIENCY  (mode overlap, POPD-style)
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("(C)  FIBER COUPLING EFFICIENCY  (overlap integral vs. axial position)")
print("=" * 74)

MFD = 9.2 * um                 # SMF-28 / PM1300-XP-like mode-field diameter
# Fine axial scan around the predicted focus; each point is a single hop
z_focus = post_pilot.dist_to_waist
scan = np.linspace(z_focus - 0.5*mm, z_focus + 0.5*mm, 25)
eta = []
for zf in scan:
    cur = beam_at(zf)
    mode = fiber_mode_gaussian(cur, MFD)          # mode sampled on THIS grid
    eta.append(coupling_efficiency(cur, mode))
eta = np.array(eta)
k_best = int(np.argmax(eta))
print(f"  Fiber MFD           = {MFD*1e6:.2f} um")
print(f"  Peak mode-overlap CE= {eta[k_best]*100:.2f} %  "
      f"at z = {scan[k_best]*1e3:.3f} mm")
print(f"  (multiply by optical-system transmission T for total CE, per your")
print(f"   TOSA workflow: CE_total = T_system * eta_overlap)")

# ---------------------------------------------------------------------------
# FIGURE
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(12, 9))

# Panel 1: analytic vs numeric spot size in a lens focus sweep
ax[0, 0].plot(z_positions*1e3, widths*1e6, 'o-', ms=4, label='POP simulation')
# analytic prediction of the focusing Gaussian
w0f = f*wl/(np.pi*w_in)
zRf = np.pi*w0f**2/wl
z_an = z_positions - z_focus
ax[0, 0].plot(z_positions*1e3, w0f*np.sqrt(1+(z_an/zRf)**2)*1e6, 'r--',
              label='analytic Gaussian')
ax[0, 0].axvline(z_focus*1e3, color='gray', ls=':')
ax[0, 0].set_xlabel('z after lens [mm]'); ax[0, 0].set_ylabel('spot radius w [um]')
ax[0, 0].set_title('(B) Beam focusing through lens')
ax[0, 0].legend(); ax[0, 0].grid(alpha=.3)

# Panel 2: coupling efficiency vs axial fiber position
ax[0, 1].plot(scan*1e3, eta*100, 'g.-')
ax[0, 1].axvline(scan[k_best]*1e3, color='gray', ls=':')
ax[0, 1].set_xlabel('fiber axial position z [mm]')
ax[0, 1].set_ylabel('mode-overlap CE [%]')
ax[0, 1].set_title('(C) Fiber coupling efficiency (POPD-style)')
ax[0, 1].grid(alpha=.3)

# Panel 3: intensity at focus
cur_focus = beam_at(z_focus)
half = 40*um
xx, _ = cur_focus.coords()
sel = np.abs(xx) <= half
I = cur_focus.intensity()
I = I[np.ix_(sel, sel)]
ext = [xx[sel][0]*1e6, xx[sel][-1]*1e6, xx[sel][0]*1e6, xx[sel][-1]*1e6]
im = ax[1, 0].imshow(I/I.max(), extent=ext, cmap='inferno')
ax[1, 0].set_xlabel('x [um]'); ax[1, 0].set_ylabel('y [um]')
ax[1, 0].set_title('Focal-plane intensity (normalised)')
fig.colorbar(im, ax=ax[1, 0], fraction=.046)

# Panel 4: line profile at focus vs Gaussian
xxf, _ = cur_focus.coords()
line = cur_focus.intensity()[cur_focus.N//2, :]
line = line/line.max()
ax[1, 1].plot(xxf*1e6, line, label='POP |E|^2')
ax[1, 1].plot(xxf*1e6, np.exp(-2*(xxf/w0f)**2), 'r--',
              label=f'Gaussian w0={w0f*1e6:.2f}um')
ax[1, 1].set_xlim(-30, 30)
ax[1, 1].set_xlabel('x [um]'); ax[1, 1].set_ylabel('norm. intensity')
ax[1, 1].set_title('Focal profile vs analytic Gaussian')
ax[1, 1].legend(); ax[1, 1].grid(alpha=.3)

plt.tight_layout()
plt.savefig('pop_lens_simulation.png', dpi=130)
print("\nSaved figure -> pop_lens_simulation.png")
