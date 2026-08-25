"""Structure loading and rasterization to permittivity grids.

The rasterization reproduces `create_permittivity_grid_penlike` from the parent
project's `20250903_create_h5_from_ends.ipynb` algorithmically exactly (binary
voxels, flat-capped cylinders, elliptical cross-section via a global z-warp) —
the convention the reference montage was computed with. Two documented caveats
(adversarial review 2026-08-12): (i) we compute membership in float64 where the
notebook used float32, so boundary voxels within ~1e-6 relative of the rod
surface may differ (O(tens) of voxels at 500³, 0–2 at 64³); (ii) like the
notebook, rods are NOT periodically wrapped — a rod whose *radius* pokes
through a face (endpoints inside) misses its wrap-image voxels, since the PBC
duplicate rows in the *_ends.txt files cover only face-crossing segments. Both
codes share this convention, so the grid is faithful to the montage; fixing it
would be a new convention and belongs behind a flag with a re-validation.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# Settled constants of the LSU family — never change (see parent repos' READMEs).
D0_UM = 0.8
L_REF_UM = 11.44
N_REF = 1000


def box_size_for_n(n_vertices: int) -> float:
    """Density-matched cubic box side (µm) for an N-vertex network."""
    return (n_vertices / N_REF) ** (1.0 / 3.0) * L_REF_UM


def load_rods(path: str | Path) -> tuple[np.ndarray, int, float]:
    """Load a 6-column rod endpoint file; infer N from the filename.

    Returns (rods (R,6) float64, N, box_size_um). Rod files store face-crossing
    rods twice (PBC-duplicated) — rasterizing every row is correct and idempotent.
    """
    path = Path(path)
    rods = np.loadtxt(path)
    if rods.ndim != 2 or rods.shape[1] != 6:
        raise ValueError(f"{path}: expected (R,6) rod endpoints, got {rods.shape}")
    m = re.search(r"N(\d+)", path.name) or re.search(r"ak(\d+)", path.name)
    if not m:
        raise ValueError(f"{path.name}: cannot infer N from filename (need 'N<digits>')")
    n = int(m.group(1))
    return rods, n, box_size_for_n(n)


def rasterize_penlike(
    rods: np.ndarray,
    grid_size: int,
    box_size: float,
    minor_radius: float = 0.2252,
    aspect_ratio: float = 2.5,
    eps_rod: float = 2.9275**2,
    eps_bg: float = 1.0,
    periodic: bool = False,
) -> np.ndarray:
    """Binary 'pen-like' rasterization — the montage convention.

    Right circular cylinders of radius `minor_radius` are built in an unwarped
    space (x, y, z' = z/aspect_ratio) and the global warp z = s·z' stretches
    them along z. The cross-section ⊥ the final axis is the shadow of the
    ellipsoid (r, r, s·r): semi-axes r and r·√(cos²θ + s²·sin²θ) where θ is the
    rod's final angle from ẑ — i.e. rods ⊥ ẑ get the full r × s·r ellipse,
    near-vertical rods stay circular (exactly the DLW 'laser-pen' Minkowski
    sweep, up to flat vs ellipsoidal caps). Endpoints are given in final
    (warped) world coordinates. Membership: axial clamp 0 ≤ t ≤ L (flat caps)
    and circular radial test, both in unwarped space. Voxels are assigned
    eps_rod if their *center* is inside any rod (no averaging).

    `periodic=False` (default) reproduces the montage/notebook convention
    exactly, including its non-wrapping edge handling: a rod whose *radius*
    pokes through a box face loses the voxels of its wrap image. Measured
    consequence at 192³/N=10k: the outermost voxel shell carries ff = 0.1975
    against 0.2211 in the interior — an 11% material deficit forming a thin
    seam on the box faces, which in a gapped medium hosts spurious localized
    defect states (found 2026-08-24 during the in-gap audit).

    `periodic=True` wraps the voxel indices (minimum-image assignment) so a
    rod contributes its full volume wherever it sits. This is a CONVENTION
    CHANGE relative to the reference montage — hence the flag; comparisons
    against montage-convention results must be re-validated.
    """
    rods = np.asarray(rods, dtype=np.float64)
    G = int(grid_size)
    s = float(aspect_ratio)
    b = float(minor_radius)
    grid = np.full((G, G, G), eps_bg, dtype=np.float32)
    dx = box_size / G
    coords = (np.arange(G, dtype=np.float64) + 0.5) * dx - box_size / 2.0

    def _rng(lo: float, hi: float) -> tuple[int, int]:
        i0 = int(np.searchsorted(coords, lo, side="left"))
        i1 = int(np.searchsorted(coords, hi, side="right") - 1)
        i0, i1 = max(i0, 0), min(i1, G - 1)
        if i1 < i0:
            mid = min(max(int(np.searchsorted(coords, 0.5 * (lo + hi))), 0), G - 1)
            i0 = i1 = mid
        return i0, i1

    pad = b * max(1.0, s) + dx
    for rod in rods:
        p1w, p2w = rod[:3], rod[3:]
        p1u = p1w.copy(); p1u[2] /= s
        p2u = p2w.copy(); p2u[2] /= s
        vu = p2u - p1u
        Lu = float(np.linalg.norm(vu))
        if Lu <= 0.0:
            continue
        nu = vu / Lu
        if periodic:
            # unclipped voxel index ranges: positions come from the UNWRAPPED
            # indices (so the geometry test is right), assignment wraps
            def _rng_p(lo, hi):
                return (int(np.floor((lo + box_size / 2.0) / dx - 0.5)),
                        int(np.ceil((hi + box_size / 2.0) / dx - 0.5)))
            (ix0, ix1) = _rng_p(min(p1w[0], p2w[0]) - pad, max(p1w[0], p2w[0]) + pad)
            (iy0, iy1) = _rng_p(min(p1w[1], p2w[1]) - pad, max(p1w[1], p2w[1]) + pad)
            (iz0, iz1) = _rng_p(min(p1w[2], p2w[2]) - pad, max(p1w[2], p2w[2]) + pad)
            gx = np.arange(ix0, ix1 + 1)
            gy = np.arange(iy0, iy1 + 1)
            gz = np.arange(iz0, iz1 + 1)
            cx = (gx + 0.5) * dx - box_size / 2.0
            cy = (gy + 0.5) * dx - box_size / 2.0
            cz = (gz + 0.5) * dx - box_size / 2.0
            X, Y, Z = np.meshgrid(cx, cy, cz, indexing="ij")
        else:
            (ix0, ix1) = _rng(min(p1w[0], p2w[0]) - pad, max(p1w[0], p2w[0]) + pad)
            (iy0, iy1) = _rng(min(p1w[1], p2w[1]) - pad, max(p1w[1], p2w[1]) + pad)
            (iz0, iz1) = _rng(min(p1w[2], p2w[2]) - pad, max(p1w[2], p2w[2]) + pad)
            X, Y, Z = np.meshgrid(
                coords[ix0 : ix1 + 1], coords[iy0 : iy1 + 1], coords[iz0 : iz1 + 1],
                indexing="ij",
            )
        RX, RY, RZ = X - p1u[0], Y - p1u[1], Z / s - p1u[2]
        t = RX * nu[0] + RY * nu[1] + RZ * nu[2]
        rX, rY, rZ = RX - t * nu[0], RY - t * nu[1], RZ - t * nu[2]
        mask = (t >= 0.0) & (t <= Lu) & (rX * rX + rY * rY + rZ * rZ <= b * b)
        if mask.any():
            if periodic:
                wx, wy, wz = gx % G, gy % G, gz % G
                ii, jj, kk = np.nonzero(mask)
                grid[wx[ii], wy[jj], wz[kk]] = eps_rod
            else:
                sub = grid[ix0 : ix1 + 1, iy0 : iy1 + 1, iz0 : iz1 + 1]
                sub[mask] = eps_rod

    return grid


def filling_fraction(eps: np.ndarray, eps_bg: float = 1.0) -> float:
    return float((eps != eps_bg).mean())
