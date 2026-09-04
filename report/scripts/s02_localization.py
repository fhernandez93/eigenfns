#!/usr/bin/env python
"""Recompute participation ratio, envelope-decay xi and the boundary-seam
diagnostic for every saved energy-density field, and compare with the values
in the project's localization_modes.json files.

Runs (mmap, CPU):
  n10k_G192_window      133 modes @192^3  (N=10k, circular decoration)
  i4_n1000_circ_G128    210 modes @128^3  (N=1000, circular, bottom-up)
  i4int_n1000_below/above 216 modes @128^3 (N=1000, circular, interior solver)
  results/prod_N1000_G128   210 modes @128^3  (N=1000, elliptical, bottom-up)
  n10k_G192_gap_periodic_v2  7 modes @192^3 (seam-free re-solve)

Also: fit-range sensitivity (r_hi_frac 0.95 -> 0.60), a PR-based length
scale, and median xi by spectral bin. Writes:
  report/numbers/s02_localization.json      (ledger)
  report/tables/loc_<tag>.json              (per-mode tables used by figures)
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from common import (GAP_HI_10K, GAP_LO_10K, L_N1K, L_N10K, RES, TAB, PROD_N1K,
                    Ledger, load_json, rel)
from eigenfns.localization import fit_xi, participation  # numpy-only module

led = Ledger(__file__)
TAB.mkdir(parents=True, exist_ok=True)


def shell_fraction(u, depth=2):
    """Energy fraction in the outer `depth`-voxel shell of a periodic cube."""
    G = u.shape[0]
    m = np.zeros(G, bool)
    m[:depth] = True
    m[G - depth:] = True
    tot = float(u.sum(dtype=np.float64))
    inner = u[depth:G - depth, depth:G - depth, depth:G - depth].sum(dtype=np.float64)
    return float((tot - inner) / tot)


def peak_on_face(u):
    G = u.shape[0]
    pk = np.unravel_index(int(np.argmax(u)), u.shape)
    return int(sum(1 for p in pk if p == 0 or p == G - 1))


def analyse(tag, ed_path, lam_path, L, extra_lam=None, ref_json=None, do_shell=True, sens=True):
    ed = np.load(ed_path, mmap_mode="r")
    lam = np.load(lam_path).astype(np.float64) if lam_path else extra_lam
    assert ed.shape[0] == len(lam), (ed.shape, len(lam))
    G = ed.shape[0 + 1]
    dx = L / G
    shell_vol = 1 - (1 - 4 / G) ** 3
    rows = []
    t0 = time.time()
    for i in range(len(lam)):
        u = np.asarray(ed[i], dtype=np.float32)
        p = participation(u)
        f = fit_xi(u, L)
        r = {"i": i, "lam": float(lam[i]), "pr_vox": p["pr_vox"], "pr_fraction": p["pr_fraction"],
             "pr_volume_um3": p["pr_vox"] * dx ** 3,
             "r_pr_um": (3 * p["pr_vox"] * dx ** 3 / (4 * np.pi)) ** (1 / 3),
             "xi_um": f.xi_um if np.isfinite(f.xi_um) else None, "xi_ceiling_um": f.xi_ceiling_um,
             "unresolved": f.unresolved, "r2": f.r2, "dyn_range_dec": f.dyn_range_dec,
             "above_ceiling": bool(np.isfinite(f.xi_um) and f.xi_um >= f.xi_ceiling_um),
             "r2_fail": bool(f.r2 < 0.7), "dyn_fail": bool(f.dyn_range_dec < 1.0)}
        if sens:
            f6 = fit_xi(u, L, r_hi_frac=0.60)
            r["xi_um_rhi060"] = f6.xi_um if np.isfinite(f6.xi_um) else None
            f5 = fit_xi(u, L, r_lo_frac=0.10, r_hi_frac=0.50)
            r["xi_um_rhi050"] = f5.xi_um if np.isfinite(f5.xi_um) else None
        if do_shell:
            r["shell2_energy_frac"] = shell_fraction(u, 2)
            r["shell2_enhancement"] = r["shell2_energy_frac"] / shell_vol
            r["peak_face_coords"] = peak_on_face(u)
        rows.append(r)
        if i % 20 == 0:
            print(f"  {tag}: {i}/{len(lam)}  {time.time()-t0:.0f}s", flush=True)
    # compare with the project's json if given
    if ref_json is not None:
        ref = load_json(ref_json)
        ref_by_lam = {}
        for rr in ref:
            ref_by_lam[round(rr["lam"], 6)] = rr
        dxi, dpr = [], []
        for r in rows:
            k = round(r["lam"], 6)
            if k in ref_by_lam:
                rr = ref_by_lam[k]
                if r["xi_um"] and np.isfinite(rr["xi_um"]) and rr["xi_um"] < 1e6:
                    dxi.append(abs(r["xi_um"] - rr["xi_um"]) / rr["xi_um"])
                dpr.append(abs(r["pr_fraction"] - rr["pr_fraction"]) / rr["pr_fraction"])
        led.add(f"{tag}_recompute_vs_json_n_matched", len(dpr), "modes", rel(ref_json),
                f"of {len(rows)} recomputed; matched on lambda to 1e-6")
        led.add(f"{tag}_recompute_vs_json_max_rel_dxi", float(max(dxi)) if dxi else None, "relative", rel(ref_json))
        led.add(f"{tag}_recompute_vs_json_max_rel_dpr", float(max(dpr)) if dpr else None, "relative", rel(ref_json))
    with open(TAB / f"loc_{tag}.json", "w") as f:
        json.dump({"tag": tag, "L": L, "G": G, "dx": dx, "shell_volume_fraction": shell_vol,
                   "source_energy_density": rel(ed_path), "source_lam": rel(lam_path) if lam_path else None,
                   "rows": rows}, f, indent=1)
    return rows, shell_vol


# ------------------------------------------------------------------ N=10k
W = RES / "n10k_G192_window"
rows10, shv = analyse("n10k", W / "window_energy_density.npy", W / "window_eigenvalues.npy", L_N10K,
                      ref_json=W / "localization_modes.json")
lam = np.array([r["lam"] for r in rows10])
xi = np.array([r["xi_um"] if r["xi_um"] else np.nan for r in rows10])
un = np.array([r["unresolved"] for r in rows10])
prf = np.array([r["pr_fraction"] for r in rows10])
src = rel(W / "window_energy_density.npy")
led.add("n10k_shell_volume_fraction", shv, "fraction", "geometry 1-(1-4/192)^3", "outer 2-voxel shell")
led.add("n10k_n_resolved", int((~un).sum()), "modes", src, "REPORT_N10K: 121 of 133")
led.add("n10k_n_unresolved", int(un.sum()), "modes", src)
led.add("n10k_n_unresolved_r2_fail", int(sum(r["r2_fail"] for r in rows10)), "modes", src)
led.add("n10k_n_unresolved_above_ceiling", int(sum(r["above_ceiling"] for r in rows10)), "modes", src)
led.add("n10k_n_unresolved_dyn_fail", int(sum(r["dyn_fail"] for r in rows10)), "modes", src)
led.add("n10k_xi_ceiling_um", L_N10K / 2, "um", "L/2")
ing = (lam > GAP_LO_10K) & (lam < GAP_HI_10K)
seam_lams = [1.8707585792024861, 1.8730821374768227, 1.929596209673228, 1.947209447202747]
ext_lam = 1.944051593936228
cand = ing & ~np.isin(np.round(lam, 9), np.round(seam_lams + [ext_lam], 9))
led.add("n10k_ingap_table", [{k: r[k] for k in ("lam", "pr_fraction", "pr_volume_um3", "xi_um", "r2", "dyn_range_dec",
                                                 "unresolved", "shell2_energy_frac", "shell2_enhancement", "peak_face_coords")}
                             for r, m in zip(rows10, ing) if m], "mixed", src, "all 10 certified states in [1.864, 1.996]")
led.add("n10k_seam_shell2_fracs", [r["shell2_energy_frac"] for r in rows10 if any(abs(r["lam"] - s) < 1e-9 for s in seam_lams)],
        "fraction", src, "REPORT_N10K: 18%, 24%, 44%, 42% for 1.8709, 1.8732, 1.9297, 1.9473")
led.add("n10k_seam_peak_face_coords", [r["peak_face_coords"] for r in rows10 if any(abs(r["lam"] - s) < 1e-9 for s in seam_lams)],
        "count", src, "number of peak-voxel coordinates on a box face")
led.add("n10k_candidate_lams", lam[cand], "um^-2", src, "five bulk-localised in-gap candidates")
led.add("n10k_candidate_xi_range", [float(np.nanmin(xi[cand])), float(np.nanmax(xi[cand]))], "um", src, "REPORT_N10K: 1.80-2.51")
led.add("n10k_candidate_pr_fraction_range", [float(prf[cand].min()), float(prf[cand].max())], "fraction", src)
led.add("n10k_candidate_r2_range", [float(min(r["r2"] for r, m in zip(rows10, cand) if m)), float(max(r["r2"] for r, m in zip(rows10, cand) if m))],
        "r2", src, "REPORT_N10K: 0.971-0.998")
led.add("n10k_candidate_shell2_range", [float(min(r["shell2_energy_frac"] for r, m in zip(rows10, cand) if m)),
                                        float(max(r["shell2_energy_frac"] for r, m in zip(rows10, cand) if m))], "fraction", src,
        "REPORT_N10K: 2.5-10%")
led.add("n10k_candidate_xi_over_L", [float(np.nanmin(xi[cand]) / L_N10K), float(np.nanmax(xi[cand]) / L_N10K)], "ratio", src)
led.add("n10k_ingap_all_xi_range", [float(np.nanmin(xi[ing])), float(np.nanmax(xi[ing]))], "um", src,
        "over all ten in-gap states incl. the extended one (REPORT_N10K: 1.80-12.98)")
led.add("n10k_ingap_all_pr_fraction_range", [float(prf[ing].min()), float(prf[ing].max())], "fraction", src)
e = [r for r in rows10 if abs(r["lam"] - ext_lam) < 1e-9][0]
led.add("n10k_extended_1p9441", {k: e[k] for k in ("lam", "xi_um", "r2", "dyn_range_dec", "pr_fraction", "unresolved")}, "mixed", src,
        "REPORT_N10K round-3 F3: xi 12.98 > ceiling, r2 0.325, 1.94 decades, PR 0.56%")
bulk_ctrl = np.r_[np.arange(6), np.arange(127, 133)]
led.add("n10k_controls_shell2_enhancement_range", [float(min(rows10[i]["shell2_enhancement"] for i in bulk_ctrl)),
                                                   float(max(rows10[i]["shell2_enhancement"] for i in bulk_ctrl))], "ratio", src,
        "six lowest + six highest window modes (REPORT_N10K: 0.6-0.9x)")
led.add("n10k_controls_pr_fraction_range", [float(prf[bulk_ctrl].min()), float(prf[bulk_ctrl].max())], "fraction", src,
        "REPORT_N10K round 4: 0.72-12.1%")
led.add("n10k_pr_fraction_range_all", [float(prf.min()), float(prf.max())], "fraction", src)
led.add("n10k_pr_fraction_median", float(np.median(prf)), "fraction", src)
# median xi by spectral bin
bins = [1.757, 1.80, 1.85, 1.864, 1.996, 2.02, 2.07, 2.117]
medxi = []
for a, b in zip(bins[:-1], bins[1:]):
    m = (lam >= a) & (lam <= b) & ~un
    medxi.append({"bin": [a, b], "n_resolved": int(m.sum()), "n_total": int(((lam >= a) & (lam <= b)).sum()),
                  "median_xi_um": float(np.nanmedian(xi[m])) if m.any() else None,
                  "median_pr_fraction": float(np.median(prf[(lam >= a) & (lam <= b)]))})
led.add("n10k_median_xi_by_bin", medxi, "um", src, "resolved modes only; bins chosen in this report")
# funnel edges
edge_lo = (lam >= 1.85) & (lam < GAP_LO_10K) & ~un
edge_hi = (lam > GAP_HI_10K) & (lam <= 2.02) & ~un
wlo = (lam < 1.80) & ~un
whi = (lam > 2.07) & ~un
led.add("n10k_median_xi_window_bottom", float(np.nanmedian(xi[wlo])), "um", src, "lambda < 1.80, resolved")
led.add("n10k_median_xi_window_top", float(np.nanmedian(xi[whi])), "um", src, "lambda > 2.07, resolved")
led.add("n10k_median_xi_low_edge", float(np.nanmedian(xi[edge_lo])), "um", src, "1.85 <= lambda < 1.864, resolved")
led.add("n10k_median_xi_high_edge", float(np.nanmedian(xi[edge_hi])), "um", src, "1.996 < lambda <= 2.02, resolved")
led.add("n10k_min_xi_resolved", float(np.nanmin(xi[~un])), "um", src)
led.add("n10k_max_xi_resolved", float(np.nanmax(xi[~un])), "um", src)
# fit-range sensitivity
x6 = np.array([r["xi_um_rhi060"] if r["xi_um_rhi060"] else np.nan for r in rows10])
ratio = x6 / xi
compact = (~un) & (prf < 0.002)
extended = (~un) & (prf > 0.01)
led.add("n10k_fitrange_ratio_compact_median", float(np.nanmedian(ratio[compact])), "ratio", src,
        "xi(r_hi=0.60 L/2)/xi(r_hi=0.95 L/2), resolved modes with PR fraction < 0.2%")
led.add("n10k_fitrange_ratio_compact_range", [float(np.nanmin(ratio[compact])), float(np.nanmax(ratio[compact]))], "ratio", src)
led.add("n10k_fitrange_ratio_extended_median", float(np.nanmedian(ratio[extended])), "ratio", src, "resolved modes with PR fraction > 1%")
led.add("n10k_fitrange_ratio_extended_range", [float(np.nanmin(ratio[extended])), float(np.nanmax(ratio[extended]))], "ratio", src)
led.add("n10k_fitrange_ratio_all_median_abs_dev", float(np.nanmedian(np.abs(ratio[~un] - 1))), "ratio", src)
rpr = np.array([r["r_pr_um"] for r in rows10])
def binmed(arr, a, b, mask=None):
    m = (lam >= a) & (lam <= b)
    if mask is not None:
        m &= mask
    return float(np.nanmedian(arr[m])) if m.any() else float("nan")
# funnel contrast per side: window-end bin vs nearest-gap bin (bins as in n10k_median_xi_by_bin)
c_xi_below = binmed(xi, 1.757, 1.80, ~un) / binmed(xi, 1.80, 1.864, ~un)
c_xi_above = binmed(xi, 2.07, 2.117, ~un) / binmed(xi, 1.996, 2.02, ~un)
c_x6_below = binmed(x6, 1.757, 1.80, ~un) / binmed(x6, 1.80, 1.864, ~un)
c_x6_above = binmed(x6, 2.07, 2.117, ~un) / binmed(x6, 1.996, 2.02, ~un)
c_pr_below = binmed(rpr, 1.757, 1.80) / binmed(rpr, 1.80, 1.864)
c_pr_above = binmed(rpr, 2.07, 2.117) / binmed(rpr, 1.996, 2.02)
led.add("n10k_funnel_contrast_xi_below_above", [c_xi_below, c_xi_above], "ratio", src,
        "median xi in [1.757,1.80] / median xi in [1.80,1.864]; and [2.07,2.117] / [1.996,2.02]; resolved modes, r_hi 0.95")
led.add("n10k_funnel_contrast_xi_rhi060_below_above", [c_x6_below, c_x6_above], "ratio", src, "same bins with r_hi 0.60")
led.add("n10k_funnel_contrast_pr_radius_below_above", [c_pr_below, c_pr_above], "ratio", src,
        "participation-volume radius (3V_p/4pi)^(1/3), all modes, same bins -- independent of the envelope fit")
led.add("n10k_median_r_pr_by_bin", [{"bin": [a, b], "median_r_pr_um": binmed(rpr, a, b), "median_pr_fraction": binmed(prf, a, b),
                                      "median_xi_um": binmed(xi, a, b, ~un), "median_xi_rhi060_um": binmed(x6, a, b, ~un)}
                                     for a, b in zip(bins[:-1], bins[1:])], "mixed", src)
led.add("n10k_r_pr_range", [float(rpr.min()), float(rpr.max())], "um", src)

# ------------------------------------------------------------------ periodic re-solve (7 modes)
P2 = RES / "n10k_G192_gap_periodic_v2"
rowsp, _ = analyse("periodic_v2", P2 / "window_energy_density.npy", P2 / "window_eigenvalues.npy", L_N10K, sens=False)
led.add("periodic_v2_table", [{k: r[k] for k in ("lam", "pr_fraction", "xi_um", "r2", "unresolved", "shell2_energy_frac", "shell2_enhancement")} for r in rowsp],
        "mixed", rel(P2 / "window_energy_density.npy"), "seam-free re-solve, m=48")

# ------------------------------------------------------------------ N=1000 circular (bottom-up)
C = RES / "i4_n1000_circ_G128"
rowsc, shvc = analyse("n1k_circ", C / "window_energy_density.npy", C / "window_eigenvalues.npy", L_N1K,
                      ref_json=C / "localization_modes.json")
lamc = np.array([r["lam"] for r in rowsc])
xic = np.array([r["xi_um"] if r["xi_um"] else np.nan for r in rowsc])
unc = np.array([r["unresolved"] for r in rowsc])
prc = np.array([r["pr_fraction"] for r in rowsc])
srcc = rel(C / "window_energy_density.npy")
led.add("n1k_circ_n_modes", len(rowsc), "modes", srcc, "bands 398-607")
led.add("n1k_circ_n_unresolved", int(unc.sum()), "modes", srcc, "REPORT_N10K: 168/210 ceiling-limited")
led.add("n1k_circ_n_above_ceiling", int(sum(r["above_ceiling"] for r in rowsc)), "modes", srcc)
led.add("n1k_circ_n_r2_fail", int(sum(r["r2_fail"] for r in rowsc)), "modes", srcc)
led.add("n1k_circ_xi_ceiling_um", L_N1K / 2, "um", "L/2")
i500, i501 = 500 - 398, 501 - 398
led.add("n1k_circ_band500", {k: rowsc[i500][k] for k in ("lam", "xi_um", "r2", "pr_fraction", "pr_volume_um3", "unresolved", "dyn_range_dec")}, "mixed", srcc,
        "REPORT_N10K: xi = 1.47 um")
led.add("n1k_circ_band501", {k: rowsc[i501][k] for k in ("lam", "xi_um", "r2", "pr_fraction", "pr_volume_um3", "unresolved", "dyn_range_dec")}, "mixed", srcc,
        "REPORT_N10K: xi = 2.30 um")
gap_c = (float(lamc[i500]), float(lamc[i501]))   # exact band-500/501 eigenvalues
edge_c = ((lamc >= gap_c[0] - 0.15) & (lamc <= gap_c[1] + 0.15)) & ~unc
led.add("n1k_circ_edge_resolved_n", int(edge_c.sum()), "modes", srcc, "resolved modes within 0.15 of the gap edges")
led.add("n1k_circ_edge_resolved_xi_range", [float(np.nanmin(xic[edge_c])), float(np.nanmax(xic[edge_c]))], "um", srcc,
        "REPORT_N10K: 1.5-2.3 um")
led.add("n1k_circ_edge_resolved_xi_median", float(np.nanmedian(xic[edge_c])), "um", srcc)
led.add("n1k_circ_resolved_xi_range", [float(np.nanmin(xic[~unc])), float(np.nanmax(xic[~unc]))], "um", srcc)
led.add("n1k_circ_pr_fraction_range", [float(prc.min()), float(prc.max())], "fraction", srcc)
# N=10k edge states with the same lambda distance to the N=10k nominal edges
edge10 = (((lam >= GAP_LO_10K - 0.15) & (lam < GAP_LO_10K)) | ((lam > GAP_HI_10K) & (lam <= GAP_HI_10K + 0.15))) & ~un
led.add("n10k_edge_resolved_n", int(edge10.sum()), "modes", src, "resolved modes within 0.15 outside the KPM bracket")
led.add("n10k_edge_resolved_xi_range", [float(np.nanmin(xi[edge10])), float(np.nanmax(xi[edge10]))], "um", src)
led.add("n10k_edge_resolved_xi_median", float(np.nanmedian(xi[edge10])), "um", src)
# the nearest-to-edge resolved states on each side, both sizes (matched-decoration comparison)
def nearest_edge(lams, xis, uns, prv, lo, hi, k=3):
    below = np.where((lams <= lo) & ~uns)[0]
    above = np.where((lams >= hi) & ~uns)[0]
    b = below[np.argsort(lo - lams[below])][:k]
    a = above[np.argsort(lams[above] - hi)][:k]
    return {"below": [{"lam": float(lams[i]), "xi_um": float(xis[i]), "pr_volume_um3": float(prv[i])} for i in b],
            "above": [{"lam": float(lams[i]), "xi_um": float(xis[i]), "pr_volume_um3": float(prv[i])} for i in a]}
prv10 = np.array([r["pr_volume_um3"] for r in rows10])
prvc = np.array([r["pr_volume_um3"] for r in rowsc])
led.add("matched_edge_states_n1k", nearest_edge(lamc, xic, unc, prvc, gap_c[0], gap_c[1]), "mixed", srcc,
        "3 nearest resolved band-edge states on each side, N=1000 circular (edges = bands 500|501)")
led.add("matched_edge_states_n10k", nearest_edge(lam, xi, un, prv10, 1.8860078588720002, 1.926413256982914), "mixed", src,
        "3 nearest resolved states on each side of the largest interior spacing 1.8860|1.9264, N=10k")
led.add("matched_edge_states_n10k_kpm_bracket", nearest_edge(lam, xi, un, prv10, GAP_LO_10K, GAP_HI_10K), "mixed", src,
        "3 nearest resolved states on each side of the KPM bracket [1.864, 1.996]")

# ------------------------------------------------------------------ N=1000 circular (interior solver, I8 cross-check)
B = RES / "i4int_n1000_below"
A = RES / "i4int_n1000_above"
rowsb, _ = analyse("n1k_int_below", B / "window_energy_density.npy", B / "window_eigenvalues.npy", L_N1K, sens=False, do_shell=False)
rowsa, _ = analyse("n1k_int_above", A / "window_energy_density.npy", A / "window_eigenvalues.npy", L_N1K, sens=False, do_shell=False)
# cross-solver xi agreement (I8 re-derivation): match on lambda to 1e-5 rel, both resolved
ref_lam = np.array([r["lam"] for r in rowsc])
diffs, diffs_edge = [], []
n_match = 0
for r in rowsb + rowsa:
    j = int(np.argmin(np.abs(ref_lam - r["lam"])))
    if abs(ref_lam[j] - r["lam"]) / r["lam"] > 2e-6:
        continue
    n_match += 1
    rr = rowsc[j]
    if not r["unresolved"] and not rr["unresolved"]:
        d = abs(r["xi_um"] - rr["xi_um"]) / rr["xi_um"]
        diffs.append(d)
        if gap_c[0] - 0.15 <= r["lam"] <= gap_c[1] + 0.15:
            diffs_edge.append(d)
led.add("i8_recompute_n_matched", n_match, "modes", rel(B / "window_eigenvalues.npy"), "ledger: 210 of 216")
led.add("i8_recompute_n_resolved_both", len(diffs), "modes", rel(B / "window_energy_density.npy"), "ledger: 42")
led.add("i8_recompute_n_gap_edge_resolved_both", len(diffs_edge), "modes", rel(B / "window_energy_density.npy"), "ledger: 20 (within 0.15 of an edge)")
led.add("i8_recompute_gap_edge_max_median", [float(max(diffs_edge)), float(np.median(diffs_edge))] if diffs_edge else None, "relative",
        rel(B / "window_energy_density.npy"), "ledger: max 8.7e-4, median 6.4e-5")
led.add("i8_recompute_all_max_median", [float(max(diffs)), float(np.median(diffs))] if diffs else None, "relative", rel(B / "window_energy_density.npy"))

# ------------------------------------------------------------------ N=1000 elliptical (PROD_N1K production)
rowse, _ = analyse("n1k_ell", PROD_N1K / "window_energy_density.npy", PROD_N1K / "window_eigenvalues.npy", L_N1K, sens=False)
lame = np.array([r["lam"] for r in rowse])
xie = np.array([r["xi_um"] if r["xi_um"] else np.nan for r in rowse])
une = np.array([r["unresolved"] for r in rowse])
pre = np.array([r["pr_fraction"] for r in rowse])
srce = rel(PROD_N1K / "window_energy_density.npy")
led.add("n1k_ell_n_unresolved", int(une.sum()), "modes", srce)
led.add("n1k_ell_band500", {k: rowse[i500][k] for k in ("lam", "xi_um", "r2", "pr_fraction", "pr_volume_um3", "unresolved", "dyn_range_dec")}, "mixed", srce,
        "INV17 / ADV17 round 1: band 500 xi = 1.82 um, r2 0.97")
led.add("n1k_ell_band501", {k: rowse[i501][k] for k in ("lam", "xi_um", "r2", "pr_fraction", "pr_volume_um3", "unresolved", "dyn_range_dec")}, "mixed", srce)
led.add("n1k_ell_pr_fraction_range", [float(pre.min()), float(pre.max())], "fraction", srce)
led.add("n1k_ell_pr_fraction_band500_501", [float(pre[i500]), float(pre[i501])], "fraction", srce)
# window-median PR fraction far from the gap vs at the gap edge
far = (np.arange(210) < 40) | (np.arange(210) >= 170)
led.add("n1k_ell_pr_fraction_median_far", float(np.median(pre[far])), "fraction", srce, "bands 398-437 and 568-607")
led.add("n1k_ell_pr_fraction_median_edge", float(np.median(pre[i500 - 5:i501 + 6])), "fraction", srce, "bands 495-506")
led.add("n1k_ell_shell2_enhancement_range", [float(min(r["shell2_enhancement"] for r in rowse)), float(max(r["shell2_enhancement"] for r in rowse))],
        "ratio", srce, "seam diagnostic on the N=1000 elliptical window")
led.add("n1k_circ_shell2_enhancement_range", [float(min(r["shell2_enhancement"] for r in rowsc)), float(max(r["shell2_enhancement"] for r in rowsc))],
        "ratio", srcc, "seam diagnostic on the N=1000 circular window")
led.add("n10k_shell2_enhancement_range_nonseam", [float(min(r["shell2_enhancement"] for r in rows10 if r["lam"] not in seam_lams)),
                                                  float(max(r["shell2_enhancement"] for r in rows10 if r["lam"] not in seam_lams))], "ratio", src)
led.add("n10k_n_modes_shell2_enhancement_gt2", int(sum(1 for r in rows10 if r["shell2_enhancement"] > 2)), "modes", src)

led.save()
print("done")
