#!/usr/bin/env python
"""Why does the N=1000 network have a clean gap and the N=10,000 one not?

Hypothesis (Lifshitz-tail / rare-region): the two networks are drawn from the
SAME local statistics, so the per-unit-volume rate of gap-filling rare
configurations is the same. The 10x larger box simply gets 10x the draws and
so samples further into the tail. If true:
  (a) the coarse-grained density distributions should COINCIDE, and
  (b) the N=10k extremes should sit further out purely from having more
      independent cells -- not from a different distribution.
The competing explanations (a real difference in local structure; the coarser
grid at N=10k; the boundary seam) make different predictions and are checked
or noted here too.

Coarse-graining scale is xi ~ 2 um, the measured in-gap localization length:
that is the volume a rare region has to fill to host one of these states.
CPU only, no GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eigenfns.structure import load_rods, rasterize_penlike  # noqa: E402

N1K = Path("/home/francisco/Documents/Create LSU Structures  - Claude/Example/"
           "N1000_lsu_example_ends.txt")
N10K = ROOT / "Structures" / "20260701_N10000_lsu_generated.txt"
RADIUS, ASPECT, EPS_ROD = 0.331836, 1.0, 8.41
VOX = 0.18          # um per voxel -- MATCHED between the two, so the
XI = 2.0            # coarse-graining is a like-for-like comparison
CASES = [("N1000", N1K, 11.4405), ("N10000", N10K, 24.64673285396475)]


def coarse_ff(path: Path, L: float):
    """Filling fraction coarse-grained over xi-sized boxes, periodic."""
    rods, _n, L_file = load_rods(path)
    assert abs(L_file - L) < 1e-3, f"{path.name}: L {L_file} != {L}"
    G = int(round(L / VOX))
    # periodic=True: the seam is a KNOWN artifact (round-3 F3) and would
    # otherwise contaminate the very tail we are measuring.
    eps = rasterize_penlike(rods, G, L, RADIUS, aspect_ratio=ASPECT,
                            eps_rod=EPS_ROD, periodic=True)
    soft = (np.asarray(eps) > 1.5).astype(np.float32)
    w = int(round(XI / VOX))
    sm = uniform_filter(soft, size=w, mode="wrap")
    n_indep = (L / XI) ** 3
    return G, len(rods), float(soft.mean()), sm.ravel(), n_indep


print(f"coarse-graining at xi = {XI} um, matched voxel {VOX} um\n")
out = {}
for tag, path, L in CASES:
    G, nrod, ff, sm, n_ind = coarse_ff(path, L)
    out[tag] = (sm, n_ind, L)
    print(f"{tag}: L={L:.4f} um  grid={G}^3  rods={nrod}  global ff={ff:.4f}"
          f"  independent xi-cells={n_ind:.0f}")
    q = np.percentile(sm, [0.01, 0.1, 1, 50, 99, 99.9, 99.99])
    print(f"  local ff  mean {sm.mean():.4f}  sd {sm.std():.4f}"
          f"  min {sm.min():.4f}  max {sm.max():.4f}")
    print(f"  pct .01/.1/1/50/99/99.9/99.99: "
          + " ".join(f"{v:.4f}" for v in q))

a, b = out["N1000"][0], out["N10000"][0]
print(f"\n(a) SAME DISTRIBUTION?  mean {a.mean():.4f} vs {b.mean():.4f} "
      f"(diff {b.mean()-a.mean():+.4f});  sd {a.std():.4f} vs {b.std():.4f} "
      f"(ratio {b.std()/a.std():.3f})")
print("    -> if these coincide, the local statistics are the same and the")
print("       only difference between the two boxes is how many draws.")

print("\n(b) TAIL REACH -- how far out does each box get to sample?")
for tag in ("N1000", "N10000"):
    sm, n_ind, L = out[tag]
    z_lo = (sm.min() - a.mean()) / a.std()
    z_hi = (sm.max() - a.mean()) / a.std()
    print(f"  {tag}: extreme local ff in units of the N=1000 sd: "
          f"{z_lo:+.2f} sd (sparsest) ... {z_hi:+.2f} sd (densest)")

# Poisson consistency: is "0 in the small box" even surprising?
print("\n(c) POISSON CHECK -- is finding ZERO in-gap states at N=1000 "
      "surprising,\n    if the N=10k rate is what it looks like?")
for k10 in (5, 6, 10):
    rate1k = k10 / 10.0
    print(f"    if N=10k truly has {k10:2d} in-gap states -> rate "
          f"{rate1k:.1f} per N=1000-volume -> P(0 at N=1000) = "
          f"{np.exp(-rate1k):.2f}")
print("    (10x is the exact volume ratio: 24.6467/11.4405 = "
      f"{24.64673285396475/11.4405:.4f} = 10^(1/3))")
