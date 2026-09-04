#!/usr/bin/env python
"""Re-verify the structure constants and the rasterizer facts from the rod
files (CPU, numpy): box sizes, rod counts, filling fractions per grid and
decoration, the boundary-seam material deficit at 192^3 and its periodic fix,
and mid-plane eps slices for Fig. 1. Writes report/numbers/s03_structure.json
and report/figures/src/eps_slice_*.npy.
"""
from __future__ import annotations

import time

import numpy as np

from common import (ASPECT_CIRC, ASPECT_ELL, D0_UM, EPS_CIRC, EPS_ELL, FIG, L_N1K, L_N10K,
                    R_CIRC, R_ELL, STRUCT_N1K, STRUCT_N10K, Ledger, rel)
from eigenfns.structure import (box_size_for_n, filling_fraction, load_rods,
                                rasterize_penlike)

led = Ledger(__file__)
SRC = FIG / "src"
SRC.mkdir(parents=True, exist_ok=True)

rods1, n1, L1 = load_rods(STRUCT_N1K)
rods10, n10, L10 = load_rods(STRUCT_N10K)
led.add("n1k_rod_rows", len(rods1), "rows", rel(STRUCT_N1K), "PBC-duplicated face-crossing rods included")
led.add("n10k_rod_rows", len(rods10), "rows", rel(STRUCT_N10K))
led.add("n1k_L", L1, "um", rel(STRUCT_N1K), "box_size_for_n: (N/1000)^(1/3) x 11.44")
led.add("n10k_L", L10, "um", rel(STRUCT_N10K))
led.add("L_ratio_n10k_n1k", L10 / L1, "ratio", "geometry", "10^(1/3) = 2.15443")
assert abs(L1 - L_N1K) < 1e-9 and abs(L10 - L_N10K) < 1e-9
# rod lengths: mean and the d0 convention
def seglen(r):
    return np.linalg.norm(r[:, 3:] - r[:, :3], axis=1)
l1, l10 = seglen(rods1), seglen(rods10)
led.add("n1k_rod_length_mean_median", [float(l1.mean()), float(np.median(l1))], "um", rel(STRUCT_N1K),
        "includes clipped face-crossing segments")
led.add("n10k_rod_length_mean_median", [float(l10.mean()), float(np.median(l10))], "um", rel(STRUCT_N10K))
led.add("d0_um", D0_UM, "um", "eigenfns/structure.py", "settled constant of the LSU family")
led.add("n1k_rods_per_vertex", len(rods1) / 1000, "ratio", rel(STRUCT_N1K), "trivalent network: 1.5 edges/vertex + PBC duplicates")
led.add("n10k_rods_per_vertex", len(rods10) / 10000, "ratio", rel(STRUCT_N10K))
led.add("n10k_vertex_density", 10000 / L10 ** 3, "um^-3", rel(STRUCT_N10K))
led.add("n1k_vertex_density", 1000 / L1 ** 3, "um^-3", rel(STRUCT_N1K))
led.add("eps_ell", EPS_ELL, "dimensionless", "n = 2.9275")
led.add("eps_circ", EPS_CIRC, "dimensionless", "n = 2.9")


def shell_ff(eps, depths=(0, 1, 2)):
    G = eps.shape[0]
    rod = eps > 1.5
    out = {}
    for d in depths:
        m = np.zeros((G, G, G), bool)
        for ax in range(3):
            idx = [slice(None)] * 3
            idx[ax] = d
            m[tuple(idx)] = True
            idx[ax] = G - 1 - d
            m[tuple(idx)] = True
        # exactly depth d (exclude shallower shells)
        for dd in range(d):
            for ax in range(3):
                idx = [slice(None)] * 3
                idx[ax] = dd
                m[tuple(idx)] = False
                idx[ax] = G - 1 - dd
                m[tuple(idx)] = False
        out[str(d)] = float(rod[m].mean())
    inner = rod[10:G - 10, 10:G - 10, 10:G - 10].mean()
    out["interior_ge10"] = float(inner)
    return out


# --------------------------------------------------------------- N=10k @192^3
t0 = time.time()
eps_m = rasterize_penlike(rods10, 192, L10, R_CIRC, ASPECT_CIRC, EPS_CIRC, periodic=False)
print(f"192^3 montage-convention rasterized in {time.time()-t0:.0f}s")
eps_p = rasterize_penlike(rods10, 192, L10, R_CIRC, ASPECT_CIRC, EPS_CIRC, periodic=True)
print(f"192^3 periodic rasterized in {time.time()-t0:.0f}s")
ffm, ffp = filling_fraction(eps_m), filling_fraction(eps_p)
led.add("n10k_ff_192_montage", ffm, "fraction", rel(STRUCT_N10K), "REPORT_N10K: 22.011%; interior_report.json 0.2201073")
led.add("n10k_ff_192_periodic", ffp, "fraction", rel(STRUCT_N10K), "REPORT_N10K: 0.22089")
sm, sp = shell_ff(eps_m), shell_ff(eps_p)
led.add("n10k_shell_ff_montage", sm, "fraction", rel(STRUCT_N10K), "ff by depth from a box face; ADV17 R2: d=0 0.1975, d=1 0.2144, d=2 0.2201, interior 0.2211")
led.add("n10k_shell_ff_periodic", sp, "fraction", rel(STRUCT_N10K), "ADV17 R2: outer shell 0.2177 after the fix")
led.add("n10k_shell_deficit_pct", 100 * (1 - sm["0"] / sm["interior_ge10"]), "%", rel(STRUCT_N10K), "REPORT_N10K: 11%")
chg = int((eps_m != eps_p).sum())
led.add("n10k_periodic_fix_changed_voxels", chg, "voxels", rel(STRUCT_N10K), "ADV17 R2: 5,528 (0.078%)")
led.add("n10k_periodic_fix_changed_pct", 100 * chg / eps_m.size, "%", rel(STRUCT_N10K))
# where do the changed voxels live?
G = 192
ii = np.argwhere(eps_m != eps_p)
dist = np.min(np.minimum(ii, G - 1 - ii), axis=1)
led.add("n10k_periodic_fix_max_depth_of_changed_voxel", int(dist.max()), "voxels", rel(STRUCT_N10K), "ADV17 R2: every one within 3 voxels of a face")
np.save(SRC / "eps_slice_n10k_192.npy", eps_m[:, :, G // 2])
np.save(SRC / "eps_slice_n10k_192_periodic.npy", eps_p[:, :, G // 2])
led.add("n10k_vox_per_um_192", 192 / L10, "vox/um", "geometry")
led.add("n10k_dx_192", L10 / 192, "um", "geometry")
led.add("n10k_rod_diameter_in_voxels_192", 2 * R_CIRC / (L10 / 192), "voxels", "geometry")
del eps_p

# 256^3 ff (I7 calibration target) -- N=10k
t0 = time.time()
eps256 = rasterize_penlike(rods10, 256, L10, R_CIRC, ASPECT_CIRC, EPS_CIRC, periodic=False)
led.add("n10k_ff_256_montage", filling_fraction(eps256), "fraction", rel(STRUCT_N10K), "ff_calibration_n10k.json: 0.2200044 at r=0.331836")
s256 = shell_ff(eps256, depths=(0,))
led.add("n10k_shell0_ff_256", s256["0"], "fraction", rel(STRUCT_N10K))
led.add("n10k_shell_deficit_pct_256", 100 * (1 - s256["0"] / s256["interior_ge10"]), "%", rel(STRUCT_N10K))
print(f"256^3 done {time.time()-t0:.0f}s")
del eps256
eps160 = rasterize_penlike(rods10, 160, L10, R_CIRC, ASPECT_CIRC, EPS_CIRC, periodic=False)
led.add("n10k_ff_160_montage", filling_fraction(eps160), "fraction", rel(STRUCT_N10K), "interior_report (G160): 0.22006")
del eps160

# --------------------------------------------------------------- N=1000
eps1e = rasterize_penlike(rods1, 128, L1, R_ELL, ASPECT_ELL, EPS_ELL, periodic=False)
led.add("n1k_ff_128_ell", filling_fraction(eps1e), "fraction", rel(STRUCT_N1K), "i1 interior_report: 0.21724")
se = shell_ff(eps1e)
led.add("n1k_shell_ff_128_ell", se, "fraction", rel(STRUCT_N1K))
led.add("n1k_shell_deficit_pct_128_ell", 100 * (1 - se["0"] / se["interior_ge10"]), "%", rel(STRUCT_N1K))
np.save(SRC / "eps_slice_n1k_128_ell.npy", eps1e[:, :, 64])
eps1c = rasterize_penlike(rods1, 128, L1, R_CIRC, ASPECT_CIRC, EPS_CIRC, periodic=False)
led.add("n1k_ff_128_circ", filling_fraction(eps1c), "fraction", rel(STRUCT_N1K), "i4int interior_report: 0.21908")
sc = shell_ff(eps1c)
led.add("n1k_shell_ff_128_circ", sc, "fraction", rel(STRUCT_N1K))
led.add("n1k_shell_deficit_pct_128_circ", 100 * (1 - sc["0"] / sc["interior_ge10"]), "%", rel(STRUCT_N1K))
np.save(SRC / "eps_slice_n1k_128_circ.npy", eps1c[:, :, 64])
eps1cp = rasterize_penlike(rods1, 128, L1, R_CIRC, ASPECT_CIRC, EPS_CIRC, periodic=True)
led.add("n1k_ff_128_circ_periodic", filling_fraction(eps1cp), "fraction", rel(STRUCT_N1K))
led.add("n1k_periodic_fix_changed_pct_128_circ", 100 * float((eps1c != eps1cp).mean()), "%", rel(STRUCT_N1K))
del eps1cp
eps1e256 = rasterize_penlike(rods1, 256, L1, R_ELL, ASPECT_ELL, EPS_ELL, periodic=False)
led.add("n1k_ff_256_ell", filling_fraction(eps1e256), "fraction", rel(STRUCT_N1K), "docs/REPORT_N1000.md: 21.7% at 256^3 (montage convention)")
del eps1e256
eps1e64 = rasterize_penlike(rods1, 64, L1, R_ELL, ASPECT_ELL, EPS_ELL, periodic=False)
led.add("n1k_ff_64_ell", filling_fraction(eps1e64), "fraction", rel(STRUCT_N1K), "golden 0.21733856 (64^3) in tests")
led.add("n1k_vox_per_um_128", 128 / L1, "vox/um", "geometry")
led.add("n1k_dx_128", L1 / 128, "um", "geometry")
led.add("n1k_rod_minor_diameter_in_voxels_128_ell", 2 * R_ELL / (L1 / 128), "voxels", "geometry")
led.add("n1k_rod_diameter_in_voxels_128_circ", 2 * R_CIRC / (L1 / 128), "voxels", "geometry")
led.add("resolution_ratio_n10k192_over_n1k128", (192 / L10) / (128 / L1), "ratio", "geometry", "REPORT_N10K: 70%")
# N=1000 at the N=10k voxel size (matched-dx slice for Fig. 1): G = round(L1/dx10)
Gm = int(round(L1 / (L10 / 192)))
eps1cm = rasterize_penlike(rods1, Gm, L1, R_CIRC, ASPECT_CIRC, EPS_CIRC, periodic=False)
np.save(SRC / f"eps_slice_n1k_{Gm}_circ_matched.npy", eps1cm[:, :, Gm // 2])
led.add("n1k_matched_dx_grid", Gm, "voxels", "geometry", "N=1000 grid with the N=10k 192^3 voxel size")
led.add("n1k_ff_matched_dx_circ", filling_fraction(eps1cm), "fraction", rel(STRUCT_N1K))
led.save()
print("done")
