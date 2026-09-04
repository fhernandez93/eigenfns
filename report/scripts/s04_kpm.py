#!/usr/bin/env python
"""Recompute the KPM density-of-states numbers from the saved Chebyshev
moments (CPU): mid-gap count, gap brackets at the 5/10/20% criteria, window
populations, Jackson smearing widths, the smoothing bias, and the N=1000
validation against the exact 611-band spectrum. Saves the DOS curves used by
Fig. 1 / SM. Writes report/numbers/s04_kpm.json and report/figures/src/dos_*.npz.
"""
from __future__ import annotations

import json

import numpy as np

from common import FIG, RES, PROD_N1K, Ledger, rel, rel_gap

led = Ledger(__file__)
SRC = FIG / "src"
SRC.mkdir(parents=True, exist_ok=True)


def jackson(p):
    k = np.arange(p + 1)
    g = ((p - k + 1) * np.cos(np.pi * k / (p + 1)) + np.sin(np.pi * k / (p + 1)) / np.tan(np.pi / (p + 1)))
    return g / (p + 1)


def step_coeffs(xb, p):
    k = np.arange(1, p + 1)
    tb = np.arccos(np.clip(xb, -1.0, 1.0))
    c = np.empty(p + 1)
    c[0] = 1 - tb / np.pi
    c[1:] = -2 * np.sin(k * tb) / (k * np.pi)
    return c


def counting(mom, lam_max, lams):
    p = mom.shape[1] - 1
    g = jackson(p)
    xs = (2.0 * np.asarray(lams) - lam_max) / lam_max
    est = np.empty((mom.shape[0], len(xs)))
    for j, xb in enumerate(xs):
        est[:, j] = mom @ (step_coeffs(xb, p) * g)
    return est


def dos_on_grid(mom, lam_max, lam):
    p = mom.shape[1] - 1
    mu = mom.mean(0) * jackson(p)
    x = np.clip(2 * lam / lam_max - 1, -1 + 1e-12, 1 - 1e-12)
    th = np.arccos(x)
    T = np.cos(np.outer(np.arange(p + 1), th))
    rho = (mu[0] + 2 * (mu[1:] @ T[1:])) / (np.pi * np.sin(th)) * 2 / lam_max
    return rho


def dos_per_probe(mom, lam_max, lam):
    p = mom.shape[1] - 1
    g = jackson(p)
    x = np.clip(2 * lam / lam_max - 1, -1 + 1e-12, 1 - 1e-12)
    th = np.arccos(x)
    T = np.cos(np.outer(np.arange(p + 1), th))
    out = np.empty((mom.shape[0], len(lam)))
    for i in range(mom.shape[0]):
        mu = mom[i] * g
        out[i] = (mu[0] + 2 * (mu[1:] @ T[1:])) / (np.pi * np.sin(th)) * 2 / lam_max
    return out


def smearing_width(lam, lam_max, p):
    x = np.clip(2 * lam / lam_max - 1, -1 + 1e-12, 1 - 1e-12)
    return np.pi / p * np.sqrt(1 - x ** 2) * lam_max / 2


def gap_bracket(lam, rho, frac, lo=1.0, hi=3.0):
    m = (lam > lo) & (lam < hi)
    lam_m, rho_m = lam[m], rho[m]
    med = np.median(rho_m)
    low = rho_m < frac * med
    best, cur, s0 = None, None, None
    for i, fl in enumerate(low):
        if fl and cur is None:
            cur, s0 = 0, i
        elif fl:
            cur += 1
        elif cur is not None:
            if best is None or cur > best[0]:
                best = (cur, s0, i)
            cur = None
    if cur is not None and (best is None or cur > best[0]):
        best = (cur, s0, len(low))
    return float(lam_m[best[1]]), float(lam_m[best[2] - 1]), float(med)


# ------------------------------------------------------------------ N=10k 256^3
z = np.load(RES / "exp" / "n10k_G256_dos_kpm.npz", allow_pickle=True)
mom, lam_max = z["moments"], float(z["lam_max"])
meta = json.loads(str(z["meta"]))
p = mom.shape[1] - 1
src = rel(RES / "exp" / "n10k_G256_dos_kpm.npz")
led.add("kpm10k_degree", p, "degree", src)
led.add("kpm10k_probes", mom.shape[0], "probes", src)
led.add("kpm10k_lam_max", lam_max, "um^-2", src, "Lanczos x 1.05 at 256^3")
led.add("kpm10k_grid", meta["grid"], "voxels", src)
led.add("kpm10k_ms_per_vec", meta.get("ms_per_vec"), "ms", src)
led.add("kpm10k_wall_s", meta.get("wall_seconds"), "s", src, "REPORT_N10K: 8,477 s")
led.add("kpm10k_meta", meta, "mixed", src)
# fine grid DOS around the gap
lam_f = np.arange(1.0, 3.0 + 1e-9, 5e-4)
rho_f = dos_on_grid(mom, lam_max, lam_f)
brackets = {}
for frac, key in ((0.05, "5%"), (0.10, "10%"), (0.20, "20%")):
    a, b, med = gap_bracket(lam_f, rho_f, frac, 1.0, 3.0)
    brackets[key] = [a, b]
    led.add(f"kpm10k_gap_bracket_{key.replace('%', 'pct')}", [a, b], "um^-2", src,
            f"rho < {frac:.2f} x local median on a 5e-4 grid in [1,3]; REPORT_N10K 10%: [1.864, 1.996]")
    led.add(f"kpm10k_gap_width_{key.replace('%', 'pct')}", b - a, "um^-2", src)
    led.add(f"kpm10k_gap_rel_{key.replace('%', 'pct')}", 100 * rel_gap(a, b), "%", src)
# absolute-threshold brackets (the registered project brackets are reproduced by rho < 80 / 160 / 320)
def abs_bracket(lam, rho, thr):
    idx = np.where(rho < thr)[0]
    runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    r = max(runs, key=len)
    return float(lam[r[0]]), float(lam[r[-1]])
lam_w = np.arange(1.70, 2.20, 5e-4)
rho_w = dos_on_grid(mom, lam_max, lam_w)
for thr in (80, 160, 320):
    a, b = abs_bracket(lam_w, rho_w, thr)
    led.add(f"kpm10k_gap_bracket_rho_lt_{thr}", [a, b], "um^-2", src,
            f"interval where the Jackson-smoothed DOS < {thr} states per unit lambda; project brackets 5%/10%/20% = [1.885,1.968]/[1.864,1.996]/[1.837,2.022]")
    led.add(f"kpm10k_gap_rel_rho_lt_{thr}", 100 * rel_gap(a, b), "%", src)
led.add("kpm10k_registered_bracket_dos_values", [float(dos_on_grid(mom, lam_max, np.array([1.864]))[0]), float(dos_on_grid(mom, lam_max, np.array([1.996]))[0])],
        "states/unit lambda", src, "DOS at the registered 10% bracket edges [1.864, 1.996]")
led.add("kpm10k_dos_median_1p757_2p117", float(np.median(dos_on_grid(mom, lam_max, np.arange(1.757, 2.117, 5e-4)))), "states/unit lambda", src,
        "median DOS over the production window (the normalisation of the original criterion could not be reproduced; see PROGRESS.md)")
a10, b10 = brackets["10%"]
centre = 0.5 * (a10 + b10)
led.add("kpm10k_gap_centre_10pct", centre, "um^-2", src)
led.add("kpm10k_smearing_width_at_gap", float(smearing_width(centre, lam_max, p)), "um^-2", src, "REPORT_N10K: ~0.023")
led.add("kpm10k_local_median_dos_1_3", float(np.median(rho_f)), "states/unit lambda", src)
# mid-gap count
est = counting(mom, lam_max, [centre, 1.93, 1.957])
for j, lamc in enumerate([centre, 1.93, 1.957]):
    led.add(f"kpm10k_count_below_{lamc:.4f}", [float(est[:, j].mean()), float(est[:, j].std(ddof=1) / np.sqrt(mom.shape[0]))],
            "states", src, "transverse states below lambda (excludes the two omega=0 modes); REPORT_N10K mid-gap 5010.3 +- 12.5")
mid = float(est[:, 0].mean())
led.add("kpm10k_midgap_count", [mid, float(est[:, 0].std(ddof=1) / np.sqrt(mom.shape[0]))], "states", src,
        "count at the centre of the 10% bracket")
led.add("kpm10k_midgap_mpb_band_pair", [round(mid) + 2, round(mid) + 3], "MPB band", src, "+2 for the omega=0 pair; ±13")
led.add("kpm10k_bands_per_vertex", mid / 10000, "bands/vertex", src, "expected 0.5")
# window populations
for a, b, note in ((1.757, 2.117, "production window; PREREG18 139 +- 3"), (1.71, 2.19, "INV17 derived window 317.1 +- 5.1"),
                   (1.925, 1.985, "S_gap slice; REPORT_N10K 4.87 +- 0.41"), (1.864, 1.996, "10% bracket; REPORT_N10K 11.18 +- 0.58"),
                   (1.757, 1.93, "S_below; 69 +- 1.5"), (1.98, 2.117, "S_above; 66 +- 2.4"), (1.9063, 1.9606, "I2 certifying sub-interval")):
    e2 = counting(mom, lam_max, [a, b])
    d = e2[:, 1] - e2[:, 0]
    led.add(f"kpm10k_window_count_{a}_{b}", [float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))], "states", src, note)
# DOS values and slopes at the production window edges + the smoothing bias (sigma^2/2) rho'
rho_pp = dos_per_probe(mom, lam_max, np.array([1.757 - 2e-3, 1.757, 1.757 + 2e-3, 2.117 - 2e-3, 2.117, 2.117 + 2e-3, centre, 1.93]))
rho_mean = rho_pp.mean(0)
d_lo = (rho_mean[2] - rho_mean[0]) / 4e-3
d_hi = (rho_mean[5] - rho_mean[3]) / 4e-3
sig_lo = float(smearing_width(1.757, lam_max, p))
sig_hi = float(smearing_width(2.117, lam_max, p))
led.add("kpm10k_dos_at_window_edges", [float(rho_mean[1]), float(rho_mean[4])], "states/unit lambda", src, "REPORT_N10K F2: 1286 below, 995 above")
led.add("kpm10k_dos_slope_at_window_edges", [float(d_lo), float(d_hi)], "states/lambda^2", src, "REPORT_N10K F2: -1.8e4, +1.1e4")
led.add("kpm10k_dos_in_gap", [float(rho_mean[6]), float(rho_mean[7])], "states/unit lambda", src, "REPORT_N10K: in-gap floor ~60")
led.add("kpm10k_smearing_width_at_window_edges", [sig_lo, sig_hi], "um^-2", src)
# bias with the Jackson kernel treated as Gaussian of width sigma (the report's model)
bias = 0.5 * sig_lo ** 2 * (-d_lo) + 0.5 * sig_hi ** 2 * d_hi
led.add("kpm10k_window_count_bias_model", float(bias), "states", src,
        "(sigma^2/2) rho' at each edge, both positive for a count between two DOS shoulders; REPORT_N10K F2: +7.5 +- 0.3 (their kernel-width convention)")
led.add("kpm10k_index_bias_at_1p757", float(0.5 * sig_lo ** 2 * d_lo), "states", src, "REPORT_N10K: -4.4 (labels read ~4 low)")
# save curve for figures
lam_c = np.arange(1.60, 2.30 + 1e-9, 5e-4)
rho_c = dos_on_grid(mom, lam_max, lam_c)
lam_full = np.linspace(0.02, lam_max * 0.999, 6000)
rho_full = dos_on_grid(mom, lam_max, lam_full)
np.savez(SRC / "dos_n10k_256.npz", lam=lam_c, rho=rho_c, lam_full=lam_full, rho_full=rho_full, lam_max=lam_max,
         bracket10=np.array(brackets["10%"]), bracket5=np.array(brackets["5%"]), bracket20=np.array(brackets["20%"]))
led.add("kpm10k_total_states_check", float(counting(mom, lam_max, [lam_max * 0.9999])[:, 0].mean()), "states", src,
        "should be ~2*256^3 - 2 = 33,554,430 (trace normalisation check)")
led.add("kpm10k_dim", 2 * 256 ** 3 - 2, "states", "geometry")

# ------------------------------------------------------------------ N=1000 128^3 validation
z1 = np.load(RES / "exp" / "kpm_n1000_G128_prod_kpm.npz", allow_pickle=True)
mom1, lmax1 = z1["moments"], float(z1["lam_max"])
p1 = mom1.shape[1] - 1
src1 = rel(RES / "exp" / "kpm_n1000_G128_prod_kpm.npz")
ev = np.load(PROD_N1K / "eigenvalues_all.npy").astype(np.float64)
led.add("kpm1k_degree_probes", [p1, mom1.shape[0]], "degree, probes", src1)
led.add("kpm1k_lam_max", lmax1, "um^-2", src1)
e = counting(mom1, lmax1, [ev[398 - 3] - 1e-6, ev[607 - 3] + 1e-6, 0.5 * (ev[500 - 3] + ev[501 - 3])])
w = e[:, 1] - e[:, 0]
led.add("kpm1k_window_count_398_607", [float(w.mean()), float(w.std(ddof=1) / np.sqrt(len(w)))], "states", src1,
        "exact 210; INV17: 213.4 +- 2.6")
led.add("kpm1k_midgap_count", [float(e[:, 2].mean()), float(e[:, 2].std(ddof=1) / np.sqrt(len(w)))], "states", src1,
        "exact 498 (MPB band 500 = index 498 in the 611 list counting from band 3); INV17: 501.9 +- 2.7")
lam_f1 = np.arange(1.0, 3.0 + 1e-9, 5e-4)
rho_f1 = dos_on_grid(mom1, lmax1, lam_f1)
for frac, key in ((0.10, "10pct"), (0.20, "20pct")):
    a, b, _ = gap_bracket(lam_f1, rho_f1, frac, 1.0, 3.0)
    led.add(f"kpm1k_gap_bracket_{key}", [a, b], "um^-2", src1, "INV17: 10% [1.886, 1.948], 20% [1.855, 1.982]; exact [1.883, 1.963]")
    led.add(f"kpm1k_gap_rel_{key}", 100 * rel_gap(a, b), "%", src1)
led.add("kpm1k_smearing_width_at_gap", float(smearing_width(1.92, lmax1, p1)), "um^-2", src1, "INV17: 0.037")
# nested-window bias calibration (round 3 F2): six windows straddling the gap
errs = []
for half in (0.05, 0.08, 0.10, 0.12, 0.15, 0.20):
    a, b = 1.923 - half, 1.923 + half
    e2 = counting(mom1, lmax1, [a, b])
    d = e2[:, 1] - e2[:, 0]
    exact = int(((ev > a) & (ev < b)).sum())
    errs.append({"window": [a, b], "kpm": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))), "exact": exact,
                 "err": float(d.mean() - exact)})
led.add("kpm1k_nested_window_errors", errs, "states", src1, "REPORT_N10K F2: all positive, +2.2..+4.0")
led.add("kpm1k_nested_window_errors_all_positive", bool(all(x["err"] > 0 for x in errs)), "bool", src1)
lam_c1 = np.arange(1.40, 2.60 + 1e-9, 5e-4)
np.savez(SRC / "dos_n1k_128.npz", lam=lam_c1, rho=dos_on_grid(mom1, lmax1, lam_c1), lam_max=lmax1)
# in-gap weight comparison: 10% bracket of each structure
a1, b1, _ = gap_bracket(lam_f1, rho_f1, 0.10, 1.0, 3.0)
w1 = counting(mom1, lmax1, [a1, b1]); w1 = w1[:, 1] - w1[:, 0]
w10 = counting(mom, lam_max, [a10, b10]); w10 = w10[:, 1] - w10[:, 0]
led.add("kpm_ingap_weight_n1k_vs_n10k", [[float(w1.mean()), float(w1.std(ddof=1) / np.sqrt(len(w1)))],
                                         [float(w10.mean()), float(w10.std(ddof=1) / np.sqrt(len(w10)))]], "states", src,
        "KPM weight inside each structure's own 10% bracket; N=1000 gap is exactly empty (leakage only); INV17: 1.6 +- 0.2 vs 16.1 +- 0.9")
led.save()
print("done")
