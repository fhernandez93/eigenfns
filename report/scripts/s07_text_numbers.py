#!/usr/bin/env python
"""Numbers quoted in the text that are derived from the primary files but were
not produced by s01-s06 (added after the adversarial reviews): the count of
N=10^4 bulk states inside the N=1000 empty-gap interval and the resulting
Poisson probability; pairs closer than the Weyl radius; the I3 cross-slice
Gram bound; certified-state counts per KPM criterion; seam enhancement of the
near-edge level-statistics band; xi of the 12- and 18-level bands; mean/s.d.
of the coarse-grained filling fraction (exp_rare_regions logic).
Writes report/numbers/s07_text_numbers.json.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from common import (ASPECT_CIRC, EPS_CIRC, GAP_HI_10K, GAP_LO_10K, L_N1K, L_N10K, R_CIRC, RES, STRUCT_N1K,
                    STRUCT_N10K, TAB, Ledger, load_json, rel)
from eigenfns.structure import load_rods, rasterize_penlike

led = Ledger(__file__)
W = RES / "n10k_G192_window"
lam = np.load(W / "window_eigenvalues.npy")
res = np.load(W / "window_residuals.npy")
seam = [1.8707585792024861, 1.8730821374768227, 1.929596209673228, 1.947209447202747]
ext = 1.944051593936228
s1 = load_json(TAB.parent / "numbers" / "s01_spectra.json")
g_lo, g_hi = s1["n1k_circ_gap_lo_128"]["value"], s1["n1k_circ_gap_hi_128"]["value"]
inside = (lam > g_lo) & (lam < g_hi)
bulk = inside & ~np.isin(np.round(lam, 9), np.round(seam + [ext], 9))
led.add("n10k_states_inside_n1k_circ_gap_interval", int(inside.sum()), "states", rel(W / "window_eigenvalues.npy"),
        f"certified N=10k states in the N=1000 circular gap interval [{g_lo:.4f}, {g_hi:.4f}]")
led.add("n10k_bulk_states_inside_n1k_circ_gap_interval", int(bulk.sum()), "states", rel(W / "window_eigenvalues.npy"),
        "excluding the four seam-flagged states and the non-exponential 1.9441")
led.add("p_empty_n1k_gap_from_bulk_rate", float(np.exp(-bulk.sum() / 10.0)), "probability", rel(W / "window_eigenvalues.npy"),
        "exp(-n_bulk/10): Poisson probability of an empty N=1000 gap at one tenth the N=10k rate")
led.add("p_empty_n1k_gap_from_five_candidates", float(np.exp(-0.5)), "probability", "arithmetic", "exp(-5/10)")
# criterion-dependent in-gap counts
s4 = load_json(TAB.parent / "numbers" / "s04_kpm.json")
for thr in (80, 160, 320):
    a, b = s4[f"kpm10k_gap_bracket_rho_lt_{thr}"]["value"]
    led.add(f"n10k_certified_inside_rho_lt_{thr}", int(((lam > a) & (lam < b)).sum()), "states", rel(W / "window_eigenvalues.npy"),
            f"bracket [{a:.4f}, {b:.4f}]")
# Weyl radius pairs
d = np.diff(lam)
close = d < 1e-4 * lam[:-1]
led.add("n10k_pairs_closer_than_weyl_radius", int(close.sum()), "pairs", rel(W / "window_eigenvalues.npy"),
        "consecutive pairs with dlambda < 1e-4 lambda: " + ", ".join(f"{lam[i]:.5f}|{lam[i+1]:.5f}" for i in np.where(close)[0]))
led.add("n10k_weyl_radius_abs_at_1p9", 1e-4 * 1.9, "um^-2", "arithmetic")
led.add("n10k_weyl_radius_over_median_spacing", float(1e-4 * 1.9 / np.median(d)), "ratio", rel(W / "window_eigenvalues.npy"))
# I3 bound for the worst pair and top list
g = load_json(RES / "gates" / "gate_results.json")["I3 residuals+orthonormality (N=10k 192^3)"]
wp = g["diagnosis"]["worst_pair"]
i = int(np.argmin(np.abs(lam - wp["lam_i"]))); j = int(np.argmin(np.abs(lam - wp["lam_j"])))
bound = (res[i] * lam[i] + res[j] * lam[j]) / abs(lam[j] - lam[i])
led.add("i3_worst_pair_overlap_bound", float(bound), "dimensionless", rel(RES / "gates" / "gate_results.json"),
        f"(r_i lam_i + r_j lam_j)/|dlam| for the worst pair {wp['lam_i']}|{wp['lam_j']}; measured {wp['overlap']}")
diag = load_json(W / "i3_gram_diagnosis.json")
ok = True
for w in diag["worst"]:
    ii = int(np.argmin(np.abs(lam - w["lam_i"]))); jj = int(np.argmin(np.abs(lam - w["lam_j"])))
    ok &= w["ov"] <= (res[ii] * lam[ii] + res[jj] * lam[jj]) / abs(lam[jj] - lam[ii])
led.add("i3_all_listed_pairs_within_bound", bool(ok), "bool", rel(W / "i3_gram_diagnosis.json"))
# near-edge level-statistics bands: seam enhancement and xi
loc = load_json(TAB / "loc_n10k.json")["rows"]
enh = np.array([r["shell2_enhancement"] for r in loc]); xi = np.array([r["xi_um"] or np.nan for r in loc]); pf = np.array([r["pr_fraction"] for r in loc])
near_lo = (lam >= 1.8208) & (lam <= 1.8502)
near_hi = (lam >= 2.005) & (lam <= 2.050)
led.add("n10k_near_edge_below_n", int(near_lo.sum()), "levels", rel(W / "window_eigenvalues.npy"), "1.8208 <= lambda <= 1.8502")
led.add("n10k_near_edge_below_n_shell_gt2", int((enh[near_lo] > 2).sum()), "levels", rel(W / "window_energy_density.npy"))
led.add("n10k_near_edge_below_max_shell_frac", float(max(r["shell2_energy_frac"] for r, m in zip(loc, near_lo) if m)), "fraction", rel(W / "window_energy_density.npy"))
led.add("n10k_near_edge_below_xi_range", [float(np.nanmin(xi[near_lo])), float(np.nanmax(xi[near_lo]))], "um", rel(W / "window_energy_density.npy"))
led.add("n10k_near_edge_above_n", int(near_hi.sum()), "levels", rel(W / "window_eigenvalues.npy"), "2.005 <= lambda <= 2.050")
led.add("n10k_near_edge_above_xi_median", float(np.nanmedian(xi[near_hi])), "um", rel(W / "window_energy_density.npy"))
led.add("n10k_near_edge_above_p_median", float(np.median(pf[near_hi])), "fraction", rel(W / "window_energy_density.npy"))
led.add("n10k_n_modes_shell_gt2", int((enh > 2).sum()), "modes", rel(W / "window_energy_density.npy"))
# matched-decoration: participation-volume ratio of the upper-edge pair
locc = load_json(TAB / "loc_n1k_circ.json")["rows"]
vp_1k_501 = locc[501 - 398]["pr_volume_um3"]
vp_10k_hi = [r["pr_volume_um3"] for r in loc if abs(r["lam"] - 1.926413256982914) < 1e-9][0]
led.add("matched_upper_edge_vp_ratio", float(vp_1k_501 / vp_10k_hi), "ratio", rel(TAB / "loc_n1k_circ.json"), "V_p(N=1000 band 501)/V_p(N=10k 1.9264)")
# coarse-grained ff statistics (exp_rare_regions logic; periodic rasterization at 0.18 um voxels)
out = {}
for tag, path, L in (("n1k", STRUCT_N1K, L_N1K), ("n10k", STRUCT_N10K, L_N10K)):
    rods, _, _ = load_rods(path)
    G = int(round(L / 0.18))
    eps = rasterize_penlike(rods, G, L, R_CIRC, ASPECT_CIRC, EPS_CIRC, periodic=True)
    soft = (eps > 1.5).astype(np.float32)
    sm = uniform_filter(soft, size=int(round(2.0 / 0.18)), mode="wrap")
    out[tag] = (float(sm.mean()), float(sm.std()), (L / 2.0) ** 3)
    led.add(f"coarse_ff_mean_{tag}", out[tag][0], "fraction", rel(path), "ff coarse-grained over 2 um cubes, periodic rasterization, 0.18 um voxels")
    led.add(f"coarse_ff_sd_{tag}", out[tag][1], "fraction", rel(path))
    led.add(f"n_xi_cells_{tag}", out[tag][2], "cells", "geometry", "(L/2 um)^3")
# fp64 verification of two scored fp32 quantities (round-2 fact-check)
led.add("n10k_duplicate_overlaps_fp64", [0.999999943, 0.999999903], "overlap", "results/n10k_G192_Sgap|Sbelow/window_vecs_spectral.npy",
        "normalised complex128 overlaps of the two cross-slice duplicates; the merge recorded 0.9988 (fp32 vdot)")
led.add("i4_min_proj2_fp64", 0.99989, "dimensionless", "results/i4int_n1000_*/window_vecs_spectral.npy vs results/i4_n1000_circ_G128/window_vecs_spectral.npy",
        "min over 210 matched pairs of normalised |<x,y>|^2 in complex128; scorer recorded 0.9961 (fp32)")
led.add("kpm10k_bracket_count_bias_model", 2.6, "states",
        "results/exp/n10k_G256_dos_kpm.npz", "(sigma^2/2) rho' at the bracket edges; value from the round-2 checker's recomputation (project record ~+1)")
led.save()
print({k: v["value"] for k, v in led.d.items() if not k.startswith("coarse") and "cells" not in k})
print({k: v["value"] for k, v in led.d.items() if k.startswith("coarse")})
