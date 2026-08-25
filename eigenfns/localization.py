"""Per-mode localization metrics: participation ratio and envelope-decay ξ.

Inputs are per-mode energy densities u(r) = ε|E|² on a periodic cube. Both
metrics carry the finite-size ceiling explicitly: a periodic box of side L
resolves decay lengths only up to ξ_ceil = L/2 — any fitted ξ ≥ that (or a
fit without enough decay dynamic range) is flagged `unresolved` and must be
reported as a lower bound, never as "extended, ξ = X".

Conventions:
- PR (participation ratio) in voxels: PR = (Σu)² / Σu²; participation volume
  V_p = PR · dx³ (µm³); participation fraction p = PR / N_vox ∈ (0, 1].
- ξ from the azimuthally averaged radial profile of u around its peak
  (periodic minimum-image distances): robust linear fit of ln u vs r over
  [r_fit_lo, r_fit_hi]; u ~ e^{−2r/ξ} ⇒ slope = −2/ξ.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


def participation(u: np.ndarray) -> dict:
    """PR metrics for one non-negative density field (any normalization)."""
    u = np.asarray(u, np.float64)
    s1 = float(u.sum())
    s2 = float((u * u).sum())
    pr_vox = s1 * s1 / max(s2, 1e-300)
    return {"pr_vox": pr_vox, "pr_fraction": pr_vox / u.size}


def radial_profile(u: np.ndarray, box_size: float, n_bins: int = 48):
    """Azimuthally averaged u(r) around the peak voxel, periodic distances.

    Returns (r_centers, mean_u) with r up to L/2 (the periodic ceiling)."""
    u = np.asarray(u)
    G = u.shape[0]
    dx = box_size / G
    pk = np.unravel_index(int(np.argmax(u)), u.shape)
    idx = np.arange(G)
    # minimum-image offsets in voxels along each axis
    d = ((idx[None, :] - np.array(pk)[:, None] + G // 2) % G) - G // 2
    R = np.sqrt(d[0][:, None, None] ** 2 + d[1][None, :, None] ** 2
                + d[2][None, None, :] ** 2) * dx
    rmax = box_size / 2.0
    bins = np.linspace(0.0, rmax, n_bins + 1)
    which = np.digitize(R.ravel(), bins) - 1
    ok = (which >= 0) & (which < n_bins)
    sums = np.bincount(which[ok], weights=u.ravel()[ok].astype(np.float64),
                       minlength=n_bins)
    cnts = np.bincount(which[ok], minlength=n_bins)
    prof = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    r = 0.5 * (bins[:-1] + bins[1:])
    return r, prof


@dataclass
class XiFit:
    xi_um: float           # fitted decay length (envelope e^{-r/xi} of the field)
    xi_ceiling_um: float   # L/2 — the resolvable maximum
    unresolved: bool       # True -> report as lower bound only, NEVER extended
    r_lo: float
    r_hi: float
    n_pts: int
    r2: float              # fit quality
    dyn_range_dec: float   # decades of profile decay across the fit range


def fit_xi(u: np.ndarray, box_size: float, n_bins: int = 48,
           r_lo_frac: float = 0.10, r_hi_frac: float = 0.95,
           min_dynamic_decades: float = 1.0, r2_min: float = 0.7) -> XiFit:
    """Envelope-decay fit with built-in ceiling logic.

    `unresolved` is raised when ANY of: fitted ξ ≥ L/2; the profile decays
    by < `min_dynamic_decades` decades across the fit range (no real decay to
    fit); or the linear fit explains < r2_min of the variance (non-exponential
    profile — typical for extended modes). The numeric thresholds are
    pre-registered in the Phase 2 plan; change them there, not here.
    """
    L = float(box_size)
    ceil = L / 2.0
    r, prof = radial_profile(u, L, n_bins)
    r_lo, r_hi = r_lo_frac * ceil, r_hi_frac * ceil
    m = (r >= r_lo) & (r <= r_hi) & np.isfinite(prof) & (prof > 0)
    if m.sum() < 6:
        return XiFit(np.inf, ceil, True, r_lo, r_hi, int(m.sum()), 0.0, 0.0)
    x, y = r[m], np.log(prof[m])
    A = np.stack([x, np.ones_like(x)], 1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope = float(coef[0])
    pred = A @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-300)
    dyn = float((y.max() - y.min()) / np.log(10.0))
    xi = np.inf if slope >= 0 else -2.0 / slope
    unresolved = (xi >= ceil) or (dyn < min_dynamic_decades) or (r2 < r2_min)
    return XiFit(float(xi), ceil, bool(unresolved), float(x.min()),
                 float(x.max()), int(m.sum()), float(r2), dyn)


def mode_report(u: np.ndarray, box_size: float, **kw) -> dict:
    d = participation(u)
    d.update(asdict(fit_xi(u, box_size, **kw)))
    return d
