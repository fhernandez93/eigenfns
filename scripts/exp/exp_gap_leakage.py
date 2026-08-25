#!/usr/bin/env python
"""Is the KPM in-gap count distinguishable from Jackson edge leakage?

KPM is the only completeness-independent handle on in-gap DOS -- it never
asks the eigensolver what it found. But the Jackson kernel smooths a hard
band edge into the gap, so a TRULY EMPTY gap still reports a nonzero count.
This asks: convolve a hard-zero gap of the measured width with the measured
kernel -- how many states land in the S_gap interval [1.925, 1.985]?

If the answer is near the measured 4.87 +- 0.41, KPM is SILENT on in-gap
states and the eigensolver is the sole evidence.  CPU only.
"""
import numpy as np

Z = np.load('results/exp/n10k_G256_dos_kpm.npz')
mom, LM, P = Z['moments'], float(Z['lam_max']), 12001
mu = mom.mean(0)[:P]                       # probe-averaged moments
D = 2 * 256 ** 3                           # 2 transverse comps per plane wave

k = np.arange(P)
jack = ((P - k + 1) * np.cos(np.pi * k / (P + 1))
        + np.sin(np.pi * k / (P + 1)) / np.tan(np.pi / (P + 1))) / (P + 1)


def rho_lam(lam):
    """Jackson-damped KPM density of states, per unit lambda."""
    lam = np.atleast_1d(lam)
    x = 2 * lam / LM - 1
    th = np.arccos(np.clip(x, -1, 1))
    # sum_k g_k mu_k cos(k theta), vectorised over lam
    c = np.cos(np.outer(th, k))
    s = (c * (jack * mu)[None, :]).sum(1) * 2 - jack[0] * mu[0]
    # Rademacher probes estimate Tr T_k directly (NOT (1/D) Tr T_k), so the
    # moments are already un-normalised -- no factor of D here. Verified by
    # the reproduction check below against the independently measured rho.
    return s / (np.pi * np.sqrt(np.maximum(1 - x ** 2, 1e-300))) * (2 / LM)


def sigma(lam):
    """Measured Jackson smearing width in lambda at this lambda."""
    x = 2 * lam / LM - 1
    return 0.998 * (np.pi / P) * np.sqrt(1 - x ** 2) * (LM / 2)


# --- validate the reconstruction against numbers already in the report ---
print("reconstruction check (vs report/round-3 measurements)")
for L_, want in [(1.757, 1286), (1.93, 70), (1.98, 110), (2.117, 995)]:
    print(f"  rho({L_:.3f}) = {rho_lam(L_)[0]:8.1f}   (measured {want})")
print(f"  sigma(1.757) = {sigma(1.757):.4f} (measured 0.0219), "
      f"sigma(2.117) = {sigma(2.117):.4f} (measured 0.0240)")

# --- band edges: where does the DOS actually collapse? ---
g = np.linspace(1.70, 2.20, 4001)
r = rho_lam(g)
print(f"\n  min rho in [1.86,2.00]: {r[(g>1.86)&(g<2.00)].min():.1f} "
      f"at lam={g[(g>1.86)&(g<2.00)][np.argmin(r[(g>1.86)&(g<2.00)])]:.4f}")

A, B = 1.925, 1.985          # the S_gap interval, as registered
dl = g[1] - g[0]

# --- the model: hard-zero gap [a,b], true band shape outside it ---
# The band edge shape is taken from KPM itself at >=4 sigma outside the gap
# (where smoothing is harmless) and continued inward as a power law
# rho ~ A|lam-edge|^p, which is then set to EXACTLY ZERO inside [a,b].
def leak(a, b, p=0.5):
    lo_ref, hi_ref = a - 4 * sigma(a), b + 4 * sigma(b)
    Alo = rho_lam(lo_ref)[0] / (a - lo_ref) ** p
    Ahi = rho_lam(hi_ref)[0] / (hi_ref - b) ** p
    rt = np.where(g < a, Alo * np.abs(a - g) ** p,
                  np.where(g > b, Ahi * np.abs(g - b) ** p, 0.0))
    rt = np.where(g < lo_ref, rho_lam(g), rt)      # far field: use measured
    rt = np.where(g > hi_ref, rho_lam(g), rt)
    # convolve with the Jackson kernel (Gaussian of the measured width)
    sg = sigma(0.5 * (a + b))
    kern = np.exp(-0.5 * ((g - g.mean()) / sg) ** 2)
    kern /= kern.sum()
    sm = np.convolve(rt, kern, mode='same')
    m = (g >= A) & (g <= B)
    return sm[m].sum() * dl, rt[m].sum() * dl


print(f"\nJackson leakage into S_gap [{A}, {B}] from a HARD-ZERO gap:")
print(f"{'gap [a,b]':>22} {'width':>7} {'p':>5} {'leaked count':>13}")
for a, b in [(1.864, 1.996), (1.88, 1.99), (1.90, 1.99), (1.87, 2.00),
             (1.91, 1.995), (1.925, 1.985)]:
    for p in (0.5, 1.0):
        c, t = leak(a, b, p)
        print(f"  [{a:.3f}, {b:.3f}] {b-a:7.3f} {p:5.1f} {c:13.2f}"
              + ("   <- gap == interval, so all of it is leakage" if a == A else ""))
print(f"\nmeasured KPM count in S_gap: 4.87 +- 0.41 (stochastic)")
l = np.load("results/n10k_G192_window/window_eigenvalues.npy")
m = (l >= A) & (l <= B)
print(f"eigensolver found in [{A},{B}]: {m.sum()} certified states "
      f"{np.round(l[m],5)}")
print("NOTE: KPM consumes the SAME rasterised eps(r), seam included, so an\n"
      "excess over leakage proves the states are in the STRUCTURE, not that\n"
      "they are physics. Only the periodic re-solve separates those.")
