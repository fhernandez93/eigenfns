#!/usr/bin/env python
"""Do the in-gap modes actually LIVE in the rare regions?

exp_rare_regions.py shows the two networks share local statistics and that
the 10x box reaches further into the tail. That is necessary for the
rare-region explanation but not sufficient: it is a statement about geometry,
not about where the modes are. This closes the loop by asking whether each
in-gap mode sits at an anomalous local filling fraction.

Two-sided by construction: a state pushed UP out of the dielectric band lives
in an air-like (low-ff) region, one pulled DOWN out of the air band lives in a
dielectric-like (high-ff) region. Either is a rare region; what would refute
the picture is in-gap modes sitting at TYPICAL local ff.

Metric: energy-weighted local filling fraction, <ff>_u = sum_r u(r) ff_xi(r),
u the normalized eps|E|^2, ff_xi the xi-coarse-grained filling fraction.
Compared against bulk control modes from the window edges.  CPU only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eigenfns.structure import load_rods, rasterize_penlike  # noqa: E402

WIN = ROOT / "results" / "n10k_G192_window"
N10K = ROOT / "Structures" / "20260701_N10000_lsu_generated.txt"
RADIUS, ASPECT, EPS_ROD, XI, G = 0.331836, 1.0, 8.41, 2.0, 192
# provenance: the ten states inside the KPM 10% bracket [1.864, 1.996], with
# the round-3 classification. Seam/extended labels are NOT used to select --
# they are printed so the table can be read against the pending periodic test.
# seam list is REPORT section 3: 1.8709/1.8732/1.9297/1.9473 carry 18/24/44/42%
# of their energy in the outer 2-voxel shell. Candidates are the five survivors.
LABEL = {1.8709: "SEAM", 1.8732: "SEAM", 1.9297: "SEAM", 1.9473: "SEAM",
         1.9441: "EXTENDED",
         1.8690: "candidate", 1.8861: "candidate", 1.9264: "candidate",
         1.9738: "candidate", 1.9902: "candidate"}


def main() -> int:
    rods, _n, L = load_rods(N10K)
    eps = rasterize_penlike(rods, G, L, RADIUS, aspect_ratio=ASPECT,
                            eps_rod=EPS_ROD, periodic=True)
    soft = (np.asarray(eps) > 1.5).astype(np.float32)
    del eps
    w = int(round(XI / (L / G)))
    ffx = uniform_filter(soft, size=w, mode="wrap")
    del soft
    mu, sd = float(ffx.mean()), float(ffx.std())
    print(f"xi-coarse-grained ff at {G}^3: mean {mu:.4f} sd {sd:.4f} "
          f"(box {w} vox = {w*L/G:.2f} um)\n")

    lam = np.load(WIN / "window_eigenvalues.npy")
    ed = np.load(WIN / "window_energy_density.npy", mmap_mode="r")
    ingap = np.where((lam >= 1.864) & (lam <= 1.996))[0]
    # bulk controls: the 6 lowest and 6 highest of the window, far from the gap
    ctrl = np.r_[np.arange(6), np.arange(len(lam) - 6, len(lam))]

    def eff(i):
        u = np.asarray(ed[i], dtype=np.float64)
        u /= u.sum()
        return float((u * ffx).sum())

    print(f"{'lambda':>9} {'class':>10} {'<ff>_u':>8} {'z':>7}")
    rows = []
    for i in ingap:
        v = eff(i)
        z = (v - mu) / sd
        k = min(LABEL, key=lambda t: abs(t - lam[i]))
        cls = LABEL[k] if abs(k - lam[i]) < 3e-3 else "?"
        rows.append((float(lam[i]), cls, v, z))
        print(f"{lam[i]:9.4f} {cls:>10} {v:8.4f} {z:+7.2f}")
    print()
    cz = []
    for i in ctrl:
        v = eff(i)
        cz.append((v - mu) / sd)
        print(f"{lam[i]:9.4f} {'control':>10} {v:8.4f} {(v-mu)/sd:+7.2f}")

    cz = np.array(cz)
    gz = np.array([r[3] for r in rows])
    kz = np.array([r[3] for r in rows if r[1] == "candidate"])
    sz = np.array([r[3] for r in rows if r[1] == "SEAM"])
    print(f"\ncontrols      : z = {cz.mean():+.2f} +- {cz.std():.2f} "
          f"(|z| max {np.abs(cz).max():.2f})")
    print(f"all in-gap    : z = {gz.mean():+.2f} +- {gz.std():.2f} "
          f"(|z| max {np.abs(gz).max():.2f})")
    print(f"candidates    : |z| = {np.abs(kz).mean():.2f} "
          f"(max {np.abs(kz).max():.2f})   <- rare-region claim rests on these")
    print(f"seam artifacts: |z| = {np.abs(sz).mean():.2f} "
          f"(max {np.abs(sz).max():.2f})   <- expect ~control if boundary-made")
    print(f"controls      : |z| = {np.abs(cz).mean():.2f} "
          f"(max {np.abs(cz).max():.2f})")
    print(f"  candidate/control |z| ratio = {np.abs(kz).mean()/np.abs(cz).mean():.2f}")
    print("\nREAD: rare-region picture predicts in-gap |z| >> control |z|.")
    print("Typical-ff in-gap modes would refute it. Energy-weighting over a")
    print("mode of finite extent regresses z toward 0, so this UNDERSTATES")
    print("the anomaly of the region the mode sits in.")
    (WIN / "rare_region_modes.json").write_text(json.dumps(
        {"mu": mu, "sd": sd, "xi_um": XI, "grid": G,
         "in_gap": [{"lam": a, "class": b, "eff": c, "z": d} for a, b, c, d in rows],
         "control_z": cz.tolist()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
