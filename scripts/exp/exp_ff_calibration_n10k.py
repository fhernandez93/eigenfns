#!/usr/bin/env python
"""Phase 1.3 decoration calibration (CPU only).

New decoration for the N=10k run: circular rods (aspect_ratio=1.0),
eps_rod = 2.9^2, target MEASURED filling fraction 22.0% on the production
grid. Bisect minor_radius on each candidate production grid (256^3, 288^3),
record (radius, achieved ff, grid); sanity-check the pinned radii on the
gold N=1000 structure (density-matched box -> ff should land close).

Usage: conda run -n lsu_ml python scripts/exp/exp_ff_calibration_n10k.py
Writes results/exp/ff_calibration_n10k.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eigenfns.structure import load_rods, rasterize_penlike, filling_fraction

ROOT = Path(__file__).resolve().parents[2]
N10K = ROOT / "Structures" / "20260701_N10000_lsu_generated.txt"
N1K = Path("/home/francisco/Documents/Create LSU Structures  - Claude/Example/"
           "N1000_lsu_example_ends.txt")

TARGET_FF = 0.220
ASPECT = 1.0
EPS_ROD = 2.9**2


def measure_ff(rods, grid, L, radius):
    t0 = time.perf_counter()
    eps = rasterize_penlike(rods, grid, L, minor_radius=radius,
                            aspect_ratio=ASPECT, eps_rod=EPS_ROD)
    ff = filling_fraction(eps)
    dt = time.perf_counter() - t0
    print(f"    grid {grid:4d}  r={radius:.6f}  ff={ff:.5f}  ({dt:.1f}s)", flush=True)
    return ff


def bisect_radius(rods, grid, L, lo=0.20, hi=0.40, iters=18, tol_ff=1e-4):
    """Bisection on the measured ff (monotone in radius)."""
    ff_lo = measure_ff(rods, grid, L, lo)
    ff_hi = measure_ff(rods, grid, L, hi)
    assert ff_lo < TARGET_FF < ff_hi, (ff_lo, ff_hi)
    trace = [(lo, ff_lo), (hi, ff_hi)]
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ff = measure_ff(rods, grid, L, mid)
        trace.append((mid, ff))
        if abs(ff - TARGET_FF) < tol_ff:
            lo = hi = mid
            break
        if ff < TARGET_FF:
            lo = mid
        else:
            hi = mid
    r = 0.5 * (lo + hi)
    ff = measure_ff(rods, grid, L, r) if lo != hi else ff
    return r, ff, trace


def main():
    out = {}
    rods10k, n10k, L10k = load_rods(N10K)
    print(f"N=10k: {rods10k.shape[0]} rod rows, N={n10k}, L={L10k:.4f} um", flush=True)
    out["n10k"] = {"rows": int(rods10k.shape[0]), "N": n10k, "L_um": L10k,
                   "target_ff": TARGET_FF, "aspect": ASPECT, "eps_rod": EPS_ROD}

    for grid in (256, 288):
        print(f"  bisecting minor_radius at {grid}^3 ...", flush=True)
        r, ff, trace = bisect_radius(rods10k, grid, L10k)
        out[f"G{grid}"] = {"radius_um": r, "ff": ff,
                           "trace": [[float(a), float(b)] for a, b in trace]}

    # sanity check on N=1000 (density-matched): same radius, production 128^3
    rods1k, n1k, L1k = load_rods(N1K)
    out["n1000_check"] = {}
    for grid_tag, key in (("G256", 128), ("G288", 144)):
        r = out[grid_tag]["radius_um"]
        # match physical resolution: N10k grid / L10k vs N1k grid / L1k
        ff1 = measure_ff(rods1k, key, L1k, r)
        out["n1000_check"][f"r_from_{grid_tag}_at_{key}"] = {
            "radius_um": r, "grid": key, "ff": ff1,
            "vox_per_um": key / L1k}
    # also the coarse quick look at analytic expectation
    # ff ~ rho_edge * pi r^2 * mean_len corrected for overlaps -- recorded only.
    outp = ROOT / "results" / "exp" / "ff_calibration_n10k.json"
    outp.write_text(json.dumps(out, indent=1))
    print(f"wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
