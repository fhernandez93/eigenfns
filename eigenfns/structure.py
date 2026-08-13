"""Structure loading and rasterization to permittivity grids.

The rasterization reproduces `create_permittivity_grid_penlike` from the parent
project's `20250903_create_h5_from_ends.ipynb` exactly (binary voxels, flat-capped
cylinders, elliptical cross-section via a global z-warp), which is the convention
the reference montage was computed with. A subpixel-smoothed variant is provided
separately; its effect on eigenfrequencies is quantified in the validation report.
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
) -> np.ndarray:
    """Binary 'pen-like' rasterization — the montage convention.

    Right circular cylinders of radius `minor_radius` are built in an unwarped
    space (x, y, z' = z/aspect_ratio) and the global warp z = s·z' makes every
    cross-section an ellipse with major/minor = s along z. Endpoints are given in
    final (warped) world coordinates. Membership: axial clamp 0 ≤ t ≤ L (flat
    caps) and circular radial test, both in unwarped space. Voxels are assigned
    eps_rod if their *center* is inside any rod (no averaging).
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
            sub = grid[ix0 : ix1 + 1, iy0 : iy1 + 1, iz0 : iz1 + 1]
            sub[mask] = eps_rod

    return grid


def filling_fraction(eps: np.ndarray, eps_bg: float = 1.0) -> float:
    return float((eps != eps_bg).mean())
