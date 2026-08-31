#!/usr/bin/env python
"""Recompute every eigenvalue-level number quoted in the paper from the saved
spectra (CPU). Writes report/numbers/s01_spectra.json.

Covers: N=1000 elliptical (USB) and circular (i4) spectra and gaps; the 64^3 /
96^3 / 128^3 gap-width series; N=10k 133-mode window statistics, in-gap list,
largest interior spacing, residuals, Rayleigh correction statistics; the
periodic re-solve; cross-grid (I6) per-band shifts; I1 / I4 eigenvalue parity
against the bottom-up references; G9 precision; MPB parity at 32^3 / 64^3.
"""
from __future__ import annotations

import re

import numpy as np

from common import (GAP_HI_10K, GAP_LO_10K, L_N1K, L_N10K, RES, USB, Ledger,
                    load_json, nu_from_lam, rel, rel_gap)

led = Ledger(__file__)

# ---------------------------------------------------------------- N=1000 ell
ev_ell = np.load(USB / "eigenvalues_all.npy").astype(np.float64)
assert ev_ell.shape == (611,) and np.all(np.diff(ev_ell) > 0)
# solver index i (0-based) is MPB band i+3; MPB band b -> index b-3
b500, b501 = ev_ell[500 - 3], ev_ell[501 - 3]
led.add("n1k_ell_n_bands", 611, "bands", rel(USB / "eigenvalues_all.npy"),
        "611 eigenvalues = MPB bands 3..613 (bands 1-2 are omega=0 at Gamma)")
led.add("n1k_ell_lam_min", ev_ell[0], "um^-2", rel(USB / "eigenvalues_all.npy"))
led.add("n1k_ell_lam_max", ev_ell[-1], "um^-2", rel(USB / "eigenvalues_all.npy"))
led.add("n1k_ell_gap_lo_128", b500, "um^-2", rel(USB / "eigenvalues_all.npy"),
        "MPB band 500 (top of lower band), 128^3, elliptical decoration")
led.add("n1k_ell_gap_hi_128", b501, "um^-2", rel(USB / "eigenvalues_all.npy"),
        "MPB band 501 (bottom of upper band), 128^3")
gap128 = rel_gap(b500, b501)
led.add("n1k_ell_gap_pct_128", 100 * gap128, "%", rel(USB / "eigenvalues_all.npy"),
        "Delta nu / nu_mid = 2 (w501 - w500)/(w501 + w500); REPORT.md quotes 2.08%")
led.add("n1k_ell_gap_center_nu", float(nu_from_lam(0.5 * (b500 + b501))), "dimensionless",
        rel(USB / "eigenvalues_all.npy"), "nu = omega a / 2 pi c with a = L/5 = 2.288 um")
# largest interior spacing check: is 500|501 the largest gap in the window?
d = np.diff(ev_ell)
imax = int(np.argmax(d))
led.add("n1k_ell_largest_spacing_between_mpb_bands", [imax + 3, imax + 4], "MPB band",
        rel(USB / "eigenvalues_all.npy"), "largest lambda spacing over all 611 bands")
led.add("n1k_ell_largest_spacing_rel_gap_pct", 100 * rel_gap(ev_ell[imax], ev_ell[imax + 1]), "%",
        rel(USB / "eigenvalues_all.npy"))
# the 114|115 shell jump disclosed in G8
j = 114 - 3
led.add("n1k_ell_jump_114_115_rel", 100 * rel_gap(ev_ell[j], ev_ell[j + 1]), "%",
        rel(USB / "eigenvalues_all.npy"), "low-band shell-structure jump disclosed in REPORT.md G8")
# window bands 398-607
wv = np.load(USB / "window_eigenvalues.npy").astype(np.float64)
assert wv.shape == (210,)
assert np.allclose(wv, ev_ell[398 - 3:608 - 3], rtol=1e-6)
led.add("n1k_ell_window_bands", [398, 607], "MPB band", rel(USB / "window_eigenvalues.npy"))
led.add("n1k_ell_window_lam", [wv[0], wv[-1]], "um^-2", rel(USB / "window_eigenvalues.npy"))

# resolution series 64/96/128 from the G5 ledger (sqrt(lambda) stored as w)
g = load_json(RES / "gates" / "gate_results.json")
g5 = g["G5 convergence (64/96/128)"]
w = {b["mpb_band"]: b for b in g5["bands"]}
series = {}
for G in (64, 96, 128):
    w500, w501 = w[500][f"w{G}"], w[501][f"w{G}"]
    series[G] = 100 * 2 * (w501 - w500) / (w501 + w500)
led.add("n1k_ell_gap_pct_series_64_96_128", [series[64], series[96], series[128]], "%",
        rel(RES / "gates" / "gate_results.json"),
        "G5 ledger sqrt(lambda) of bands 500/501; REPORT.md quotes 2.35 -> 1.93 -> 2.08%")
assert abs(series[128] - 100 * gap128) < 0.01, (series[128], gap128)
# 64^3 gap from the full 680-band 64^3 spectrum (independent file)
e64 = np.load(RES / "exp" / "e3_vals_G64.npy")
led.add("n1k_ell_gap_pct_64_from_e3", 100 * rel_gap(e64[500 - 3], e64[501 - 3]), "%",
        rel(RES / "exp" / "e3_vals_G64.npy"), "680-band 64^3 solve (G3w reference)")
led.add("n1k_ell_gap_edges_64", [e64[500 - 3], e64[501 - 3]], "um^-2", rel(RES / "exp" / "e3_vals_G64.npy"))
# G5 non-monotone spread on band 500 and Richardson-type residuals
spread500 = 100 * (max(w[500][f"w{G}"] for G in (64, 96, 128)) - min(w[500][f"w{G}"] for G in (64, 96, 128))) / w[500]["w128"]
led.add("n1k_ell_g5_band500_spread_pct", spread500, "%", rel(RES / "gates" / "gate_results.json"),
        "max-min of sqrt(lambda) over 64/96/128 relative to 128^3; REPORT.md quotes 0.27%")
led.add("n1k_ell_g5_worst_est_resid_128_pct", 100 * g5["worst_est_resid_128_rel"], "%",
        rel(RES / "gates" / "gate_results.json"), "worst Richardson residual among the 5 monotone bands")
led.add("n1k_ell_g5_monotone_bands", [b["mpb_band"] for b in g5["bands"] if b["monotone"]], "MPB band",
        rel(RES / "gates" / "gate_results.json"))

# ---------------------------------------------------------------- N=1000 circ
ev_c = np.load(RES / "i4_n1000_circ_G128" / "eigenvalues_all.npy")
assert ev_c.shape == (611,) and np.all(np.diff(ev_c) > 0)
c500, c501 = ev_c[500 - 3], ev_c[501 - 3]
dc = np.diff(ev_c)
led.add("n1k_circ_gap_lo_128", c500, "um^-2", rel(RES / "i4_n1000_circ_G128" / "eigenvalues_all.npy"))
led.add("n1k_circ_gap_hi_128", c501, "um^-2", rel(RES / "i4_n1000_circ_G128" / "eigenvalues_all.npy"))
led.add("n1k_circ_gap_pct_128", 100 * rel_gap(c500, c501), "%",
        rel(RES / "i4_n1000_circ_G128" / "eigenvalues_all.npy"), "REPORT_N10K quotes 5.07%")
led.add("n1k_circ_largest_spacing_between_mpb_bands", [int(np.argmax(dc)) + 3, int(np.argmax(dc)) + 4],
        "MPB band", rel(RES / "i4_n1000_circ_G128" / "eigenvalues_all.npy"))
led.add("n1k_circ_gap_center_nu", float(nu_from_lam(0.5 * (c500 + c501))), "dimensionless",
        rel(RES / "i4_n1000_circ_G128" / "eigenvalues_all.npy"), "a = 2.288 um")
led.add("n1k_circ_lam_range", [ev_c[0], ev_c[-1]], "um^-2", rel(RES / "i4_n1000_circ_G128" / "eigenvalues_all.npy"))
# bulk median spacing near the gap (bands 450-550 excluding the gap itself)
sp = np.delete(dc, 500 - 3)
led.add("n1k_circ_median_spacing_450_550", float(np.median(sp[450 - 3:550 - 3])), "um^-2",
        rel(RES / "i4_n1000_circ_G128" / "eigenvalues_all.npy"))

# ---------------------------------------------------------------- N=10k window
W = RES / "n10k_G192_window"
lam = np.load(W / "window_eigenvalues.npy")
lam_raw = np.load(W / "window_eigenvalues_raw.npy")
nrm = np.load(W / "window_norms.npy")
res = np.load(W / "window_residuals.npy")
assert lam.shape == (133,) and np.all(np.diff(lam) > 0)
assert np.allclose(lam, lam_raw / nrm, rtol=0, atol=1e-12), "corrected != raw/||x||^2"
led.add("n10k_n_certified", 133, "states", rel(W / "window_eigenvalues.npy"),
        "after cross-slice dedup of 135 (69 S_below + 5 S_gap + 61 S_above)")
led.add("n10k_lam_range", [lam[0], lam[-1]], "um^-2", rel(W / "window_eigenvalues.npy"), "corrected endpoints")
led.add("n10k_residual_max_reported", res.max(), "relative", rel(W / "window_residuals.npy"),
        "reported residuals were computed with the raw lambda and carry a floor ~ |lambda_raw-lambda|/lambda")
led.add("n10k_residual_median_reported", float(np.median(res)), "relative", rel(W / "window_residuals.npy"))
# true residual estimate: r_true^2 = r_rep^2 - delta^2 with delta = relative shift
delta = np.abs(lam_raw - lam) / lam
r_true = np.sqrt(np.maximum(res ** 2 - delta ** 2, 0))
led.add("n10k_residual_max_corrected_est", r_true.max(), "relative", rel(W / "window_residuals.npy"),
        "sqrt(r_rep^2 - delta^2), delta = |lam_raw - lam|/lam per vector; ledger I3 measured 5.88e-5 directly")
led.add("n10k_norm_sq_minus_1_range", [float((nrm - 1).min()), float((nrm - 1).max())], "dimensionless",
        rel(W / "window_norms.npy"))
led.add("n10k_norm_sq_minus_1_mean", float((nrm - 1).mean()), "dimensionless", rel(W / "window_norms.npy"))
led.add("n10k_rayleigh_rel_shift_mean", float(((lam - lam_raw) / lam_raw).mean()), "relative",
        rel(W / "window_eigenvalues_raw.npy"), "REPORT_N10K quotes -4.7e-5")
led.add("n10k_rayleigh_rel_shift_range", [float(((lam - lam_raw) / lam_raw).min()), float(((lam - lam_raw) / lam_raw).max())],
        "relative", rel(W / "window_eigenvalues_raw.npy"))
led.add("n10k_rayleigh_abs_shift_at_1p94", float(4.73e-5 * 1.94), "um^-2", rel(W / "window_norms.npy"),
        "mean relative shift x lambda~1.94")
# spacing statistics
d10 = np.diff(lam)
imax = int(np.argmax(d10))
led.add("n10k_largest_interior_spacing", [lam[imax], lam[imax + 1]], "um^-2", rel(W / "window_eigenvalues.npy"),
        "REPORT_N10K: 1.8860 - 1.9264")
led.add("n10k_largest_interior_spacing_width", d10[imax], "um^-2", rel(W / "window_eigenvalues.npy"))
led.add("n10k_largest_interior_spacing_rel_gap_pct", 100 * rel_gap(lam[imax], lam[imax + 1]), "%",
        rel(W / "window_eigenvalues.npy"))
ingap_mask = (lam > GAP_LO_10K) & (lam < GAP_HI_10K)
outside = ~ingap_mask
led.add("n10k_bulk_median_spacing", float(np.median(np.diff(lam[lam < GAP_LO_10K]).tolist()
                                                     + np.diff(lam[lam > GAP_HI_10K]).tolist())),
        "um^-2", rel(W / "window_eigenvalues.npy"), "median spacing outside the KPM bracket; REPORT_N10K 1.35e-3")
led.add("n10k_median_spacing_all", float(np.median(d10)), "um^-2", rel(W / "window_eigenvalues.npy"))
led.add("n10k_ingap_kpm_bracket", [GAP_LO_10K, GAP_HI_10K], "um^-2", "REPORT_N10K.md",
        "KPM 10%-criterion bracket; s04_kpm.py recomputes")
led.add("n10k_ingap_states_lam", lam[ingap_mask], "um^-2", rel(W / "window_eigenvalues.npy"),
        "all certified states inside [1.864, 1.996]")
led.add("n10k_ingap_count_montage", int(ingap_mask.sum()), "states", rel(W / "window_eigenvalues.npy"))
led.add("n10k_ingap_max_spacing_over_bulk", float(d10[imax] / np.median(d10)), "ratio", rel(W / "window_eigenvalues.npy"))
led.add("n10k_n_below_bracket", int((lam < GAP_LO_10K).sum()), "states", rel(W / "window_eigenvalues.npy"))
led.add("n10k_n_above_bracket", int((lam > GAP_HI_10K).sum()), "states", rel(W / "window_eigenvalues.npy"))
# per slice counts
for tag in ("Sbelow", "Sgap", "Sabove"):
    r = load_json(RES / f"n10k_G192_{tag}" / "interior_report.json")
    led.add(f"n10k_{tag}_n_converged", r["n_converged"], "states", rel(RES / f"n10k_G192_{tag}" / "interior_report.json"))
    led.add(f"n10k_{tag}_window", r["window"], "um^-2", rel(RES / f"n10k_G192_{tag}" / "interior_report.json"))
    led.add(f"n10k_{tag}_m", r["m"], "vectors", rel(RES / f"n10k_G192_{tag}" / "interior_report.json"))
    led.add(f"n10k_{tag}_degrees", [r["build_degree"], r["polish_degree"]], "degree", rel(RES / f"n10k_G192_{tag}" / "interior_report.json"))
    led.add(f"n10k_{tag}_theta_apps", r["theta_applications"], "applications", rel(RES / f"n10k_G192_{tag}" / "interior_report.json"))
    led.add(f"n10k_{tag}_wall_h", r["wall_seconds"] / 3600, "h", rel(RES / f"n10k_G192_{tag}" / "interior_report.json"))
    led.add(f"n10k_{tag}_worst_res_reported", r["worst_res_reported"], "relative", rel(RES / f"n10k_G192_{tag}" / "interior_report.json"))
    led.add(f"n10k_{tag}_n_inwindow_unconverged", r["n_inwindow_unconverged"], "states", rel(RES / f"n10k_G192_{tag}" / "interior_report.json"))
rw = load_json(W / "interior_report.json")
led.add("n10k_duplicates_removed", [[d_[0], d_[3]] for d_ in rw["duplicates_removed"]], "[lambda, overlap]",
        rel(W / "interior_report.json"), "two states found by both S_gap and S_below")
led.add("n10k_lam_max_192", rw["lam_max"], "um^-2", rel(W / "interior_report.json"))
led.add("n10k_ff_192_montage", rw["ff"], "fraction", rel(W / "interior_report.json"))
led.add("n10k_L", rw["L"], "um", rel(W / "interior_report.json"))
assert abs(rw["L"] - L_N10K) < 1e-9
# cross-slice reproducibility of the two duplicates (from the S_gap and S_below raw files)
lb = np.load(RES / "n10k_G192_Sbelow" / "window_eigenvalues.npy")
lg = np.load(RES / "n10k_G192_Sgap" / "window_eigenvalues.npy")
dups = []
for l0 in lg:
    j = int(np.argmin(np.abs(lb - l0)))
    if abs(lb[j] - l0) / l0 < 1e-5:
        dups.append(abs(lb[j] - l0) / l0)
led.add("n10k_cross_slice_dup_rel_diff", dups, "relative", rel(RES / "n10k_G192_Sgap" / "window_eigenvalues.npy"),
        "REPORT_N10K: 1.4e-6 and 1.7e-6 (same normalisation bias in both, cancels)")

# ---------------------------------------------------------------- seam / periodic
pm = load_json(RES / "gates" / "periodic_overlap_match.json")
pm2 = load_json(RES / "gates" / "periodic_overlap_match_n10k_G192_gap_periodic_v2.json")
lp = np.load(RES / "n10k_G192_gap_periodic" / "window_eigenvalues.npy")
lp2 = np.load(RES / "n10k_G192_gap_periodic_v2" / "window_eigenvalues.npy")
led.add("periodic_v1_lams", lp, "um^-2", rel(RES / "n10k_G192_gap_periodic" / "window_eigenvalues.npy"))
led.add("periodic_v2_lams", lp2, "um^-2", rel(RES / "n10k_G192_gap_periodic_v2" / "window_eigenvalues.npy"))
led.add("periodic_v1_v2_max_rel_diff", float(np.max(np.abs(lp2 - lp) / lp)), "relative",
        rel(RES / "n10k_G192_gap_periodic_v2" / "window_eigenvalues.npy"),
        "v2 was solved with the fixed (normalised) rr_extract, v1 with the raw one; ~4.7e-5 expected")
led.add("periodic_v1_unconverged", load_json(RES / "n10k_G192_gap_periodic" / "interior_report.json")["unconverged_lams"], "um^-2",
        rel(RES / "n10k_G192_gap_periodic" / "interior_report.json"))
led.add("periodic_v2_unconverged", load_json(RES / "n10k_G192_gap_periodic_v2" / "interior_report.json")["unconverged_lams"], "um^-2",
        rel(RES / "n10k_G192_gap_periodic_v2" / "interior_report.json"))
led.add("periodic_v2_unconverged_res", np.load(RES / "n10k_G192_gap_periodic_v2" / "window_unconverged_residuals.npy"), "relative",
        rel(RES / "n10k_G192_gap_periodic_v2" / "window_unconverged_residuals.npy"))
led.add("periodic_ff_192", load_json(RES / "n10k_G192_gap_periodic" / "interior_report.json")["ff"], "fraction",
        rel(RES / "n10k_G192_gap_periodic" / "interior_report.json"))
rows = []
for p in pm["pairs"]:
    budget = float(np.sum(np.asarray(p["all_overlaps"]) ** 2))
    rows.append({"lam_mont": p["lam_mont"], "best_overlap": p["overlap"], "lam_peri": p["lam_peri"],
                 "dlam": p["dlam"], "budget": budget})
led.add("periodic_match_table", rows, "mixed", rel(RES / "gates" / "periodic_overlap_match.json"),
        "montage in-gap state -> best periodic partner (|overlap|), dlam, total overlap budget sum_j |<m,p_j>|^2")
seam_lams = [1.8707585792024861, 1.8730821374768227, 1.929596209673228, 1.947209447202747]
bulk = [r for r in rows if r["lam_mont"] not in seam_lams]
seam = [r for r in rows if r["lam_mont"] in seam_lams]
led.add("periodic_bulk_overlap_range", [min(r["best_overlap"] for r in bulk), max(r["best_overlap"] for r in bulk)],
        "overlap", rel(RES / "gates" / "periodic_overlap_match.json"))
led.add("periodic_bulk_dlam_range", [min(r["dlam"] for r in bulk), max(r["dlam"] for r in bulk)], "um^-2",
        rel(RES / "gates" / "periodic_overlap_match.json"), "all negative")
led.add("periodic_bulk_all_negative", bool(all(r["dlam"] < 0 for r in bulk)), "bool", rel(RES / "gates" / "periodic_overlap_match.json"))
led.add("periodic_seam_best_overlaps", [r["best_overlap"] for r in seam], "overlap", rel(RES / "gates" / "periodic_overlap_match.json"))
led.add("periodic_seam_budgets", [r["budget"] for r in seam], "sum |overlap|^2", rel(RES / "gates" / "periodic_overlap_match.json"),
        "REPORT_N10K round 4: 0.0001, 0.0201, 0.0087, 0.0971 (order 1.87076, 1.87308, 1.92960, 1.94721)")
led.add("periodic_bulk_overlap_range_v2", [min(p["overlap"] for p in pm2["pairs"] if p["lam_mont"] not in seam_lams),
                                           max(p["overlap"] for p in pm2["pairs"] if p["lam_mont"] not in seam_lams)],
        "overlap", rel(RES / "gates" / "periodic_overlap_match_n10k_G192_gap_periodic_v2.json"))
# full scan: the periodic 1.98401 partner above the gap
pf = load_json(RES / "gates" / "periodic_overlap_match_full.json")
best_for_peri = {}
for p in pf["pairs"]:
    for j, ov in enumerate(p["all_overlaps"]):
        if ov > best_for_peri.get(j, (0, None))[0]:
            best_for_peri[j] = (ov, p["lam_mont"])
led.add("periodic_full_scan_best_partner_per_periodic_state",
        [{"lam_peri": float(lp[j]), "best_overlap": best_for_peri[j][0], "lam_mont": best_for_peri[j][1],
          "dlam": float(lp[j] - best_for_peri[j][1])} for j in sorted(best_for_peri)],
        "mixed", rel(RES / "gates" / "periodic_overlap_match_full.json"),
        "every periodic state's best montage partner over all 133 modes")
led.add("periodic_ingap_count_after_fix", int(((lp > GAP_LO_10K) & (lp < GAP_HI_10K)).sum()), "states",
        rel(RES / "n10k_G192_gap_periodic" / "window_eigenvalues.npy"))

# ---------------------------------------------------------------- I6 cross-grid
for tag, key, f in (("edgelow", "low", "crossgrid_match.json"),
                    ("edgehigh_narrow", "high", "crossgrid_match_n10k_G256_edgehigh_narrow.json")):
    cg = load_json(RES / "gates" / f)
    pr = [p for p in cg["pairs"] if p["overlap"] > 0.5]
    dww = np.array([(np.sqrt(p["lam_fine"]) - np.sqrt(p["lam_coarse"])) / np.sqrt(p["lam_coarse"]) for p in pr])
    led.add(f"i6_{key}_edge_window", cg["window"], "um^-2", rel(RES / "gates" / f))
    led.add(f"i6_{key}_edge_n_matched", len(pr), "states", rel(RES / "gates" / f), f"of {len(cg['pairs'])} coarse states")
    led.add(f"i6_{key}_edge_abs_max_dw_w_pct", 100 * float(np.abs(dww).max()), "%", rel(RES / "gates" / f))
    led.add(f"i6_{key}_edge_median_dw_w_pct", 100 * float(np.median(dww)), "%", rel(RES / "gates" / f))
    led.add(f"i6_{key}_edge_overlap_min_median", [min(p["overlap"] for p in pr), float(np.median([p["overlap"] for p in pr]))],
            "overlap", rel(RES / "gates" / f))
    led.add(f"i6_{key}_edge_power_retention_range", [min(cg["fine_power_in_coarse_kset"]), max(cg["fine_power_in_coarse_kset"])],
            "fraction", rel(RES / "gates" / f), "saturated statistic (withdrawn as evidence, round 4)")
    if key == "high":
        led.add("i6_high_edge_n_fine_converged", int(sum(1 for p in cg["pairs"] if p.get("fine_converged", True))), "states",
                rel(RES / "gates" / f), "only 3 of 11 fine pairs residual-certified; the rest are unconverged vectors used for overlap only")
        unm = [p for p in cg["pairs"] if p["overlap"] <= 0.5]
        led.add("i6_high_edge_unmatched", [{"lam_coarse": p["lam_coarse"], "overlap": p["overlap"]} for p in unm], "mixed", rel(RES / "gates" / f))
ret = load_json(RES / "gates" / "retention.json")
r96 = [d_["96"] for d_ in ret["edgelow"] + ret["edgehigh_narrow"]]
led.add("i6_retention_at_96_cube_range", [min(r96), max(r96)], "fraction", rel(RES / "gates" / "retention.json"),
        "power of 256^3 modes inside a 96^3 k-cube (half the production resolution) -- shows the statistic is saturated")
r192 = [d_["96"] for d_ in ret["edgelow"]]
led.add("i6_G160_converged", load_json(RES / "n10k_G160_gapedge" / "interior_report.json")["n_converged"], "states",
        rel(RES / "n10k_G160_gapedge" / "interior_report.json"))
led.add("i6_G160_unconverged", load_json(RES / "n10k_G160_gapedge" / "interior_report.json")["n_inwindow_unconverged"], "states",
        rel(RES / "n10k_G160_gapedge" / "interior_report.json"), "all within 0.025 of a window edge (edge-truncated)")
l160 = np.load(RES / "n10k_G160_gapedge" / "window_eigenvalues.npy")
led.add("i6_G160_converged_range", [l160.min(), l160.max()], "um^-2", rel(RES / "n10k_G160_gapedge" / "window_eigenvalues.npy"))
l256 = np.load(RES / "n10k_G256_edgelow" / "window_eigenvalues.npy")
led.add("i6_G256_edgelow_n", len(l256), "states", rel(RES / "n10k_G256_edgelow" / "window_eigenvalues.npy"))
led.add("i6_G192_count_in_edgelow_window", int(((lam >= 1.84) & (lam <= 1.95)).sum()), "states", rel(W / "window_eigenvalues.npy"))
led.add("i6_G256_edgelow_ingap_count", int(((l256 > GAP_LO_10K) & (l256 < GAP_HI_10K)).sum()), "states",
        rel(RES / "n10k_G256_edgelow" / "window_eigenvalues.npy"))
led.add("i6_G256_ff", load_json(RES / "n10k_G256_edgelow" / "interior_report.json")["ff"], "fraction",
        rel(RES / "n10k_G256_edgelow" / "interior_report.json"))
led.add("i6_vox_per_um", {"160": 160 / L_N10K, "192": 192 / L_N10K, "256": 256 / L_N10K, "n1k_128": 128 / L_N1K}, "vox/um",
        "geometry", "")

# ---------------------------------------------------------------- I1 parity (eigenvalues)
li1 = np.load(RES / "i1_n1000_slice" / "window_eigenvalues.npy")
li1_raw = np.load(RES / "i1_n1000_slice" / "window_eigenvalues_raw.npy")
ri1 = load_json(RES / "i1_n1000_slice" / "interior_report.json")
# match each interior eigenvalue to the nearest reference eigenvalue (reference is float32 on disk)
idx = np.array([int(np.argmin(np.abs(ev_ell - l0))) for l0 in li1])
dl = np.abs(ev_ell[idx] - li1) / li1
dl_raw = np.abs(ev_ell[idx] - li1_raw) / li1_raw
inwin = (li1 >= ri1["window"][0]) & (li1 <= ri1["window"][1])
led.add("i1_n_reported", len(li1), "states", rel(RES / "i1_n1000_slice" / "window_eigenvalues.npy"))
led.add("i1_window", ri1["window"], "um^-2", rel(RES / "i1_n1000_slice" / "interior_report.json"))
led.add("i1_matched_ref_index_range_mpb", [int(idx.min()) + 3, int(idx.max()) + 3], "MPB band", rel(USB / "eigenvalues_all.npy"))
led.add("i1_targets_in_slice_473_522_found", int(np.sum((idx + 3 >= 473) & (idx + 3 <= 522))), "states", rel(USB / "eigenvalues_all.npy"),
        "registered slice: MPB bands 473..522 (50 bands)")
led.add("i1_max_dlam_lam_corrected", float(dl.max()), "relative", rel(RES / "i1_n1000_slice" / "window_eigenvalues.npy"),
        "eigenvalue parity vs the bottom-up reference (float32 on disk, ~6e-8 quantisation); ledger recomputation 2.35e-7 over 55")
led.add("i1_median_dlam_lam_corrected", float(np.median(dl)), "relative", rel(RES / "i1_n1000_slice" / "window_eigenvalues.npy"))
led.add("i1_max_dw_w_corrected", float(dl.max() / 2), "relative", rel(RES / "i1_n1000_slice" / "window_eigenvalues.npy"),
        "Delta omega/omega = Delta lambda / (2 lambda)")
led.add("i1_max_dlam_lam_raw", float(dl_raw.max()), "relative", rel(RES / "i1_n1000_slice" / "window_eigenvalues_raw.npy"),
        "pre-correction: ledger 2.83e-5")
led.add("i1_improvement_factor", float(dl_raw.max() / dl.max()), "ratio", rel(RES / "i1_n1000_slice" / "window_eigenvalues.npy"))
led.add("i1_unconverged_lams", ri1["unconverged_lams"], "um^-2", rel(RES / "i1_n1000_slice" / "interior_report.json"),
        "1.7314 lies outside the registered 473..522 slice (below the window's interior)")
led.add("i1_worst_res", ri1["worst_res_reported"], "relative", rel(RES / "i1_n1000_slice" / "interior_report.json"))
led.add("i1_theta_apps", ri1["theta_applications"], "applications", rel(RES / "i1_n1000_slice" / "interior_report.json"))
led.add("i1_ref_dtype", str(np.load(USB / "eigenvalues_all.npy").dtype), "dtype", rel(USB / "eigenvalues_all.npy"))

# ---------------------------------------------------------------- I4 parity
for tag in ("below", "above"):
    l4 = np.load(RES / f"i4int_n1000_{tag}" / "window_eigenvalues.npy")
    idx4 = np.array([int(np.argmin(np.abs(ev_c - l0))) for l0 in l4])
    dl4 = np.abs(ev_c[idx4] - l4) / l4
    led.add(f"i4int_{tag}_n", len(l4), "states", rel(RES / f"i4int_n1000_{tag}" / "window_eigenvalues.npy"))
    led.add(f"i4int_{tag}_max_dlam_lam", float(dl4.max()), "relative", rel(RES / f"i4int_n1000_{tag}" / "window_eigenvalues.npy"),
            "eigenvalue-only parity vs i4_n1000_circ_G128 (fp64 reference); ledger vector-matched 2.85e-7 / 4.11e-7")
    led.add(f"i4int_{tag}_median_dlam_lam", float(np.median(dl4)), "relative", rel(RES / f"i4int_n1000_{tag}" / "window_eigenvalues.npy"))
    led.add(f"i4int_{tag}_ref_mpb_band_range", [int(idx4.min()) + 3, int(idx4.max()) + 3], "MPB band", rel(RES / "i4_n1000_circ_G128" / "eigenvalues_all.npy"))
    led.add(f"i4int_{tag}_distinct_matches", int(len(set(idx4.tolist()))), "states", rel(RES / f"i4int_n1000_{tag}" / "window_eigenvalues.npy"),
            "== n means one-to-one (no ghosts, no duplicates)")
    r4 = load_json(RES / f"i4int_n1000_{tag}" / "interior_report.json")
    led.add(f"i4int_{tag}_worst_res", r4["worst_res_reported"], "relative", rel(RES / f"i4int_n1000_{tag}" / "interior_report.json"))
    led.add(f"i4int_{tag}_window", r4["window"], "um^-2", rel(RES / f"i4int_n1000_{tag}" / "interior_report.json"))
    led.add(f"i4int_{tag}_wall_h", r4["wall_seconds"] / 3600, "h", rel(RES / f"i4int_n1000_{tag}" / "interior_report.json"))
    led.add(f"i4int_{tag}_m", r4["m"], "vectors", rel(RES / f"i4int_n1000_{tag}" / "interior_report.json"))
    led.add(f"i4int_{tag}_ff_128", r4["ff"], "fraction", rel(RES / f"i4int_n1000_{tag}" / "interior_report.json"))

# ---------------------------------------------------------------- G9 precision
c64 = np.load(RES / "exp" / "prec48_c64.npy")
c128 = np.load(RES / "exp" / "prec48_c128.npy")
dww9 = np.abs(np.sqrt(c64) - np.sqrt(c128)) / np.sqrt(c128)
led.add("g9_n_bands", len(c64), "bands", rel(RES / "exp" / "prec48_c64.npy"))
led.add("g9_max_dw_w", float(dww9.max()), "relative", rel(RES / "exp" / "prec48_c128.npy"), "REPORT.md: 2.4e-7 over 96 bands")

# ---------------------------------------------------------------- MPB parity
def mpb_freqs(path):
    txt = open(path).read()
    lines = [ln for ln in txt.splitlines() if ln.startswith("freqs:") and not ln.startswith("freqs:, k index")]
    vals = [float(x) for x in lines[0].split(",")[6:]]
    return np.array(vals)


def parity(ours_lam, mpb_path, L):
    nu_ours = np.sqrt(ours_lam) * L / (2 * np.pi)           # MPB units c/a with a = L
    nu_mpb = mpb_freqs(mpb_path)
    nu_mpb = nu_mpb[nu_mpb > 1e-6]                           # drop the two omega=0 modes
    n = min(len(nu_ours), len(nu_mpb))
    d_ = np.abs(nu_ours[:n] - nu_mpb[:n]) / nu_mpb[:n]
    return n, float(d_.max()), float(np.median(d_)), int(np.argmax(d_)) + 3


n32, mx32, md32, b32 = parity(np.load(RES / "exp" / "parity32_ours.npy"), RES / "exp" / "mpb32.out", L_N1K)
led.add("mpb_parity_32_n_bands", n32, "bands", rel(RES / "exp" / "mpb32.out"))
led.add("mpb_parity_32_max_dw_w", mx32, "relative", rel(RES / "exp" / "parity32_ours.npy"), f"worst at MPB band {b32}")
led.add("mpb_parity_32_median_dw_w", md32, "relative", rel(RES / "exp" / "parity32_ours.npy"))
n64, mx64, md64, b64 = parity(np.load(RES / "gates" / "parity64_ours.npy"), RES / "gates" / "mpb64.out", L_N1K)
led.add("mpb_parity_64_n_bands", n64, "bands", rel(RES / "gates" / "mpb64.out"), "G3: REPORT.md 298 bands, 8.95e-6")
led.add("mpb_parity_64_max_dw_w", mx64, "relative", rel(RES / "gates" / "parity64_ours.npy"), f"worst at MPB band {b64}")
led.add("mpb_parity_64_median_dw_w", md64, "relative", rel(RES / "gates" / "parity64_ours.npy"))
n64w, mx64w, md64w, b64w = parity(np.load(RES / "gates" / "parity64w_ours.npy"), RES / "gates" / "mpb64w.out", L_N1K)
led.add("mpb_parity_64w_n_bands", n64w, "bands", rel(RES / "gates" / "mpb64w.out"), "G3w: REPORT.md 658 bands, 3.5e-5")
led.add("mpb_parity_64w_max_dw_w", mx64w, "relative", rel(RES / "gates" / "parity64w_ours.npy"), f"worst at MPB band {b64w}")
led.add("mpb_parity_64w_median_dw_w", md64w, "relative", rel(RES / "gates" / "parity64w_ours.npy"))
# MPB reported fill fraction at 32^3 (epsilon line)
m = re.search(r'([0-9.]+)% "fill"', open(RES / "exp" / "mpb32.out").read())
led.add("mpb32_fill_pct", float(m.group(1)), "%", rel(RES / "exp" / "mpb32.out"))
led.add("mpb_tolerance", 1e-9, "relative", rel(RES / "exp" / "mpb32.ctl"))
# gap position in the 64^3 MPB spectrum too (independent code!)
nu64w = mpb_freqs(RES / "gates" / "mpb64w.out")
nu64w = nu64w[nu64w > 1e-6]
if len(nu64w) >= 501:
    led.add("mpb64w_gap_500_501_pct", 100 * 2 * (nu64w[501 - 3] - nu64w[500 - 3]) / (nu64w[501 - 3] + nu64w[500 - 3]), "%",
            rel(RES / "gates" / "mpb64w.out"), "MPB's own 64^3 gap between bands 500|501 (index = band-3 after dropping omega=0 pair)")
    jj = int(np.argmax(np.diff(nu64w)[300:])) + 300
    led.add("mpb64w_largest_spacing_bands", [jj + 3, jj + 4], "MPB band", rel(RES / "gates" / "mpb64w.out"))

# ---------------------------------------------------------------- gate ledger scalars
led.add("g3_max_dw_w", g["G3 disordered parity (300 bands, 64^3)"]["max_dw_w"], "relative", rel(RES / "gates" / "gate_results.json"))
led.add("g3w_max_dw_w", g["G3w full-window parity (660 bands, 64^3)"]["max_dw_w"], "relative", rel(RES / "gates" / "gate_results.json"))
led.add("g3w_median_dw_w", g["G3w full-window parity (660 bands, 64^3)"]["median"], "relative", rel(RES / "gates" / "gate_results.json"))
led.add("g2_srs_gap_pct", g["G2 literature (srs 28.06%)"]["measured_pct"], "%", rel(RES / "gates" / "gate_results.json"), "published 28.06%")
led.add("g6_kpm_missed_below_midgap", [g["G6 completeness"]["kpm_missed_below_midgap"], g["G6 completeness"]["kpm_se"]], "states",
        rel(RES / "gates" / "gate_results.json"))
led.add("g7_worst_rel_res", g["G7 residuals+orthonormality (window)"]["worst_rel_res"], "relative", rel(RES / "gates" / "gate_results.json"))
led.add("g4_min_cos", min(c["min_cos_principal_angle"] for c in g["G4 degeneracy subspaces (64^3, lowest 28)"]["clusters"]), "cos",
        rel(RES / "gates" / "gate_results.json"))
i3 = g["I3 residuals+orthonormality (N=10k 192^3)"]
led.add("i3_worst_res", i3["worst_res"], "relative", rel(RES / "gates" / "gate_results.json"), "recomputed with fp64 normalised quotient")
led.add("i3_median_res", i3["median_res"], "relative", rel(RES / "gates" / "gate_results.json"))
led.add("i3_worst_gram_dev", i3["worst_gram_dev"], "dimensionless", rel(RES / "gates" / "gate_results.json"), "gate 5e-5: FAIL")
led.add("i3_cross_slice_max", i3["diagnosis"]["cross_slice"]["max_offdiag"], "dimensionless", rel(RES / "gates" / "gate_results.json"))
led.add("i3_same_slice_max", i3["diagnosis"]["same_slice"]["max_offdiag"], "dimensionless", rel(RES / "gates" / "gate_results.json"))
i2 = g["I2 completeness (N=10k window, v2 estimator)"]
led.add("i2_subgap_missed", [i2["missed_subgap"], i2["missed_subgap_se"]], "states", rel(RES / "gates" / "gate_results.json"),
        "certifying sub-interval [1.9063, 1.9606]")
led.add("i2_window_missed", [i2["missed_window"], i2["missed_window_se"]], "states", rel(RES / "gates" / "gate_results.json"),
        "consistency check only (leakage model biased by >= 1.23 +- 0.23, N=1000 calibration)")
led.add("i2_subgap_interval", [1.9063, 1.9606], "um^-2", "REPORT_N10K.md", "")
i2c = g["I2 calibration (N=1000 I1 slice, v2 estimator)"]
led.add("i2_calibration_missed", [i2c["missed_window"], i2c["missed_window_se"]], "states", rel(RES / "gates" / "gate_results.json"))
i2p = g["I2 completeness (N=10k periodic gap window, v2 estimator)"]
led.add("i2_periodic_subgap_missed", [i2p["missed_subgap"], i2p["missed_subgap_se"]], "states", rel(RES / "gates" / "gate_results.json"))
i8 = g["I8 localization (cross-solver xi agreement, N=1000)"]
led.add("i8_gap_edge_rel_diff_max_median", [i8["gap_edge_rel_diff"]["max"], i8["gap_edge_rel_diff"]["median"]], "relative",
        rel(RES / "gates" / "gate_results.json"))
led.add("i8_n_gap_edge_resolved", i8["n_gap_edge_resolved"], "modes", rel(RES / "gates" / "gate_results.json"))
led.add("i8_n_resolved_in_both", i8["n_resolved_in_both"], "modes", rel(RES / "gates" / "gate_results.json"))
i4b = g["I4-interior cross-solve (N=1000 new decoration, below-gap)"]
i4a = g["I4-interior cross-solve (N=1000 new decoration, above-gap)"]
led.add("i4_ledger_max_dlam_rel", max(i4b["max_dlam_rel"], i4a["max_dlam_rel"]), "relative", rel(RES / "gates" / "gate_results.json"))
led.add("i4_ledger_min_proj2", min(i4b["min_proj2"], i4a["min_proj2"]), "dimensionless", rel(RES / "gates" / "gate_results.json"))
led.add("i4_ledger_targets", i4b["targets_found"] + i4a["targets_found"], "states", rel(RES / "gates" / "gate_results.json"),
        "210 vector-matched targets, 0 missed, 0 ghosts; 6 more checked on eigenvalues only")
i1g = g["I1 ground-truth parity (50-band slice, production config)"]
led.add("i1_ledger_pre_correction_max", i1g["max_dlam_rel"], "relative", rel(RES / "gates" / "gate_results.json"))

print("N=1000 ell gap 128^3: %.4f -> %.4f  %.3f%%" % (b500, b501, 100 * gap128))
print("series 64/96/128:", series)
print("N=1000 circ gap: %.4f -> %.4f  %.3f%%" % (c500, c501, 100 * rel_gap(c500, c501)))
print("N=10k largest spacing: %.4f -> %.4f" % (lam[imax], lam[imax + 1]))
print("N=10k in-gap:", lam[ingap_mask])
print("I1 max dlam/lam corrected %.3g raw %.3g" % (dl.max(), dl_raw.max()))
print("G9 %.3g ; MPB32 %.3g/%.3g (%d) ; MPB64 %.3g/%.3g (%d) ; MPB64w %.3g/%.3g (%d)" % (dww9.max(), mx32, md32, n32, mx64, md64, n64, mx64w, md64w, n64w))
led.save()

# ---------------------------------------------------------------- addenda (round-0 self-audit findings)
# (i) corrected cross-slice duplicates agree ~180x better than the raw ones
lbr = np.load(RES / "n10k_G192_Sbelow" / "window_eigenvalues_raw.npy")
lgr = np.load(RES / "n10k_G192_Sgap" / "window_eigenvalues_raw.npy")
raw_d = []
for l0, l0r in zip(lg, lgr):
    j = int(np.argmin(np.abs(lb - l0)))
    if abs(lb[j] - l0) / l0 < 1e-5:
        raw_d.append(abs(lbr[j] - l0r) / l0r)
led.add("n10k_cross_slice_dup_rel_diff_raw", raw_d, "relative", rel(RES / "n10k_G192_Sgap" / "window_eigenvalues_raw.npy"),
        "REPORT_N10K quotes these raw figures (1.4e-6, 1.7e-6); the corrected values agree to ~1e-8 because the two solves had different ||x||^2")
# (ii) MPB judge grid (trilinear read-back of our 64^3 grid) vs our binary 64^3 raster at the gap edge
nuo = np.sqrt(np.load(RES / "gates" / "parity64w_ours.npy")) * L_N1K / (2 * np.pi)
nue = np.sqrt(e64) * L_N1K / (2 * np.pi)
led.add("judge64_vs_binary64_rel_dnu_bands_498_503", [float((nuo[b - 3] - nue[b - 3]) / nue[b - 3]) for b in range(498, 504)], "relative",
        rel(RES / "gates" / "parity64w_ours.npy"), "same solver, MPB's interpolated eps vs our binary eps at 64^3: -1.4% to -3.3% at the gap edge")
led.add("judge64_gap_500_501_pct", 100 * 2 * (nuo[498] - nuo[497]) / (nuo[498] + nuo[497]), "%", rel(RES / "gates" / "parity64w_ours.npy"),
        "on the judge grid the 500|501 spacing is 0.60%; largest spacing 501|502 (0.83%); binary raster: 500|501 at 2.35%")
led.add("judge64_gap_501_502_pct", 100 * 2 * (nuo[499] - nuo[498]) / (nuo[499] + nuo[498]), "%", rel(RES / "gates" / "parity64w_ours.npy"))
led.add("binary64_gap_501_502_pct", 100 * 2 * (nue[499] - nue[498]) / (nue[499] + nue[498]), "%", rel(RES / "exp" / "e3_vals_G64.npy"))
# (iii) gap spacing relative to neighbouring level spacings at 128^3 (robustness of the '500|501' statement)
led.add("n1k_ell_gap_over_neighbour_spacing_128", float(d[497] / np.median(np.r_[d[487:497], d[498:508]])), "ratio", rel(USB / "eigenvalues_all.npy"),
        "gap spacing / median of the 20 neighbouring spacings")
led.add("n1k_circ_gap_over_neighbour_spacing_128", float(dc[497] / np.median(np.r_[dc[487:497], dc[498:508]])), "ratio", rel(RES / "i4_n1000_circ_G128" / "eigenvalues_all.npy"))
led.save()
