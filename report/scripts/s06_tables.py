#!/usr/bin/env python
"""Build the LaTeX tables and CSVs of the Supplemental Material from the
saved run records (CPU): bake-off, timings/memory, the full N=10k certified
state table, the N=1000 window tables, the in-gap table, the seam/periodic
match table and the I6 cross-grid table. Writes report/tables/*.tex|csv and
report/numbers/s06_tables.json.
"""
from __future__ import annotations

import csv
import json
import re

import numpy as np

from common import (A_NORM_N1K, GAP_HI_10K, GAP_LO_10K, L_N1K, L_N10K, RES, TAB, USB,
                    Ledger, load_json, nu_from_lam, rel)

led = Ledger(__file__)
TAB.mkdir(parents=True, exist_ok=True)
led.add("a_norm_um", A_NORM_N1K, "um", "REPORT.md", "a = L_1000/5 = 2.288 um; identical for N=10k (density-matched: L_10k/1250^(1/3))")
led.add("a_norm_n10k_check", L_N10K / 1250 ** (1 / 3), "um", "geometry")


def tex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%")


def fmt_e(x, d=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:.{d}e}".replace("e-0", r"\times10^{-").replace("e+0", r"\times10^{").replace("e-", r"\times10^{-").replace("e+", r"\times10^{") + "}"


def sci(x, d=1):
    if x is None:
        return "--"
    m, e = f"{x:.{d}e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


# ------------------------------------------------------------------ bake-off
bo = {}
for tag in ("bandpass_m80_d3300", "hybrid_m80", "hybrid_m80c", "shiftinv_m64b"):
    bo[tag] = load_json(RES / "exp" / f"bakeoff_{tag}.json")
fold_log = (RES / "exp" / "bakeoff_folded_m32.log").read_text()
m = re.search(r"locked\s+(\d+)/50.*?iters\s+(\d+).*?lam \[([0-9.]+)\.\.([0-9.]+)\].*?worst res ([0-9.e+-]+).*?elapsed\s+([0-9.]+)s", fold_log)
fold = {"locked": int(m.group(1)), "iters": int(m.group(2)), "mu_lo": float(m.group(3)), "mu_hi": float(m.group(4)),
        "worst_res": float(m.group(5)), "elapsed": float(m.group(6))}
led.add("bakeoff_folded_block", fold, "mixed", rel(RES / "exp" / "bakeoff_folded_m32.log"),
        "first block of 20 locked at maxit=200 with folded Rayleigh quotients mu in [0.096, 0.279]; all 50 targets have mu <= 0.0497 (INV17)")
rows = []
b = bo["bandpass_m80_d3300"]
rows.append(("Bandpass ChebSI, one stage", "$m=80$, $d=3300$, 3 outers", b["n_converged"], len(b["ghosts"]), b["theta_applications"], b["wall_seconds"], None, None,
             "subspace captured 50/50 to $2\\times10^{-3}$; med. res.\\ $6.6\\to4.2\\to2.5\\times10^{-2}$"))
h = bo["hybrid_m80c"]
rows.append(("Two-stage bandpass (build\\,+\\,polish)", "build $d=3300$ $m=80$ $\\times2$, trim 56, polish $d=8000$ $\\times6$", h["n_converged"], len(h["ghosts"]),
             h["theta_applications"], h["wall_seconds"] + bo["hybrid_m80"]["wall_seconds"] + b["wall_seconds"] * 0, h["max_dlam_rel"], h["min_proj2"],
             "26 unconverged = slice-edge/gap-edge pairs still descending; trim-limited ($1.12\\times$ oversampling)"))
s = bo["shiftinv_m64b"]
rows.append(("Shift-invert PMINRES SI", "$m=64$, inner tol $10^{-2}$, maxit 150, 2 outers", s["n_converged"], len(s["ghosts"]), s["theta_applications"], s["wall_seconds"], None, None,
             "inner solves hit maxit (150/150); med. res.\\ $1.8\\times10^{-1}$"))
rows.append(("Folded-spectrum LOBPCG", "$m=32$, WZ preconditioner $\\alpha^2\\in\\{0.05,1,20\\}\\sigma^2$", 0, 0, None, fold["elapsed"], None, None,
             f"first block locked 20 at maxit 200 with folded $\\mu\\in[{fold['mu_lo']:.3f},{fold['mu_hi']:.3f}]$ vs targets $\\mu\\le0.0497$"))
rows.append(("Expansion-RR polish (5 variants)", "on the two-stage build subspace", 0, 0, None, None, None, None,
             "med. res.\\ $4.2\\times10^{-2}\\to0.9$ in one sweep; continuity/strip variants oscillate $7$--$23\\times10^{-3}$"))
with open(TAB / "bakeoff.tex", "w") as f:
    f.write("\\begin{tabular}{@{}p{2.5cm}p{3.4cm}rrrrp{4.2cm}@{}}\n\\hline\\hline\n")
    f.write("Method & Configuration & verified & ghosts & $\\Theta$-applies & wall (s) & Failure mode / note \\\\\n\\hline\n")
    for r in rows:
        ta = f"{r[4]:,}" if r[4] else "--"
        ws = f"{r[5]:,.0f}" if r[5] else "--"
        f.write(f"{r[0]} & {r[1]} & {r[2]}/50 & {r[3]} & {ta} & {ws} & {r[8]} \\\\\n")
    f.write("\\hline\\hline\n\\end{tabular}\n")
led.add("bakeoff_twostage_verified", h["n_converged"], "pairs", rel(RES / "exp" / "bakeoff_hybrid_m80c.json"))
led.add("bakeoff_twostage_max_dlam_rel", h["max_dlam_rel"], "relative", rel(RES / "exp" / "bakeoff_hybrid_m80c.json"), "pre-Rayleigh-correction figure (2.8e-5)")
led.add("bakeoff_twostage_theta_apps", h["theta_applications"], "applications", rel(RES / "exp" / "bakeoff_hybrid_m80c.json"))
led.add("bakeoff_twostage_wall_s", h["wall_seconds"] + bo["hybrid_m80"]["wall_seconds"], "s", rel(RES / "exp" / "bakeoff_hybrid_m80c.json"),
        "polish legs only (hybrid_m80 + hybrid_m80c); the build is bandpass_m80_d3300 (7,109 s)")
led.add("bakeoff_onestage_theta_apps", b["theta_applications"], "applications", rel(RES / "exp" / "bakeoff_bandpass_m80_d3300.json"))
led.add("bakeoff_onestage_wall_s", b["wall_seconds"], "s", rel(RES / "exp" / "bakeoff_bandpass_m80_d3300.json"))
led.add("bakeoff_onestage_medres_trajectory", [o["median_res_inwin"] for o in b["outer_stats"]], "relative", rel(RES / "exp" / "bakeoff_bandpass_m80_d3300.json"))
led.add("bakeoff_shiftinv_theta_apps", s["theta_applications"], "applications", rel(RES / "exp" / "bakeoff_shiftinv_m64b.json"))
led.add("bakeoff_shiftinv_wall_s", s["wall_seconds"], "s", rel(RES / "exp" / "bakeoff_shiftinv_m64b.json"))
led.add("bakeoff_shiftinv_medres", s["outer_stats"][1]["median_res_inwin"], "relative", rel(RES / "exp" / "bakeoff_shiftinv_m64b.json"))
led.add("bakeoff_sigma", b["sigma"], "um^-2", rel(RES / "exp" / "bakeoff_bandpass_m80_d3300.json"))
led.add("bakeoff_slice", b["slice"], "0-based solver index", rel(RES / "exp" / "bakeoff_bandpass_m80_d3300.json"), "= MPB bands 476..525")
led.add("bakeoff_polish_trajectory_hybrid_m80c", [o["median_res_inwin"] for o in h["outer_stats"]["polish"]], "relative", rel(RES / "exp" / "bakeoff_hybrid_m80c.json"))
led.add("bakeoff_polish_trajectory_hybrid_m80", [o["median_res_inwin"] for o in bo["hybrid_m80"]["outer_stats"]["polish"]], "relative", rel(RES / "exp" / "bakeoff_hybrid_m80.json"))

# ------------------------------------------------------------------ timings / memory
tim = {}
for G in (192, 224, 256, 288):
    t = load_json(RES / "exp" / f"n10k_G{G}_timing.json")
    tim[G] = t
    led.add(f"matvec_ms_G{G}", t["ms_per_vec"], "ms/vector", rel(RES / "exp" / f"n10k_G{G}_timing.json"), f"chunk {t['chunk']}, degree {t['degree']} x {t['probes']} probes")
    led.add(f"lam_max_G{G}", t["lam_max"], "um^-2", rel(RES / "exp" / f"n10k_G{G}_timing.json"))
    led.add(f"vector_GB_G{G}", 2 * G ** 3 * 8 / 1e9, "GB", "geometry", "two complex64 transverse components")
led.add("matvec_ms_G128_n1k", 2.7, "ms/vector", "REPORT.md / plans log", "batched, 128^3 c64 (log-quoted; no JSON)")
led.add("matvec_ms_G256_n1k_log", 28.6, "ms/vector", "REPORT.md", "N=1000 256^3, chunks <= 4 (log-quoted)")
with open(TAB / "timing.tex", "w") as f:
    f.write("\\begin{tabular}{@{}lrrrr@{}}\n\\hline\\hline\n grid & vox/$\\mu$m & $\\Theta$ apply (ms/vector) & $\\lambda_{\\max}$ ($\\mu$m$^{-2}$) & vector (GB) \\\\\n\\hline\n")
    for G in (192, 224, 256, 288):
        f.write(f" ${G}^3$ & {G / L_N10K:.2f} & {tim[G]['ms_per_vec']:.1f} & {tim[G]['lam_max']:.0f} & {2 * G ** 3 * 8 / 1e9:.3f} \\\\\n")
    f.write("\\hline\\hline\n\\end{tabular}\n")
# production run table
prod = []
for tag, note in (("n10k_G192_Sbelow", "$S_\\mathrm{below}$"), ("n10k_G192_Sgap", "$S_\\mathrm{gap}$"), ("n10k_G192_Sabove", "$S_\\mathrm{above}$"),
                  ("n10k_G192_gap_periodic", "periodic re-solve v1"), ("n10k_G192_gap_periodic_v2", "periodic re-solve v2"),
                  ("n10k_G160_gapedge", "I6 $160^3$ leg"), ("n10k_G256_edgelow", "I6 $256^3$ low-edge anchor"),
                  ("n10k_G256_edgehigh_narrow", "I6 $256^3$ high-edge anchor"), ("i1_n1000_slice", "I1 slice ($N=1000$, $128^3$)"),
                  ("i4int_n1000_below", "I4 below ($N=1000$ circ.)"), ("i4int_n1000_above", "I4 above ($N=1000$ circ.)")):
    r = load_json(RES / tag / "interior_report.json")
    prod.append((note, r["grid"], r["window"], r["m"], r["build_degree"], r["polish_degree"], r["n_converged"], r["n_inwindow_unconverged"],
                 r["theta_applications"], r["wall_seconds"], r["worst_res_reported"]))
with open(TAB / "runs.tex", "w") as f:
    f.write("\\begin{tabular}{@{}lrlrrrrrrrr@{}}\n\\hline\\hline\n run & grid & window & $m$ & $d_\\mathrm{b}$ & $d_\\mathrm{p}$ & conv. & unc. & $\\Theta$-applies & wall (h) & worst res.\\ \\\\\n\\hline\n")
    for p in prod:
        ta = f"{p[8]/1e6:.2f}M" if p[8] else "n/a"
        f.write(f" {p[0]} & ${p[1]}^3$ & [{p[2][0]:.3f}, {p[2][1]:.3f}] & {p[3]} & {p[4]} & {p[5]} & {p[6]} & {p[7]} & {ta} & {p[9]/3600:.1f} & {sci(p[10])} \\\\\n")
    f.write("\\hline\\hline\n\\end{tabular}\n")
led.add("runs_table_rows", [{"run": p[0], "grid": p[1], "window": p[2], "m": p[3], "conv": p[6], "unconv": p[7], "theta": p[8], "wall_s": p[9], "worst_res": p[10]} for p in prod],
        "mixed", "results/*/interior_report.json")
tot_h = sum(p[9] for p in prod[:3]) / 3600
led.add("n10k_production_total_wall_h", tot_h, "h", "results/n10k_G192_S*/interior_report.json", "three slices")
led.add("n10k_production_total_theta_apps", sum(p[8] for p in prod[:3]), "applications", "results/n10k_G192_S*/interior_report.json")
# N=1000 bottom-up production figures (log-quoted in REPORT.md)
led.add("n1k_prod_wall_s", 19908, "s", "REPORT.md / results/prod_N1000_G128.log", "5 h 32 m, 611 bands, 128^3")
log = (RES / "prod_N1000_G128.log").read_text()
mm = re.findall(r"solve wall (\d+)s, theta applications (\d+)", log)
if mm:
    led.add("n1k_prod_wall_s_log", int(mm[-1][0]), "s", rel(RES / "prod_N1000_G128.log"))
    led.add("n1k_prod_theta_apps_log", int(mm[-1][1]), "applications", rel(RES / "prod_N1000_G128.log"))
mm2 = re.findall(r"locked\s+(\d+)/611.*?iters\s+(\d+)", log)
if mm2:
    its = [int(x[1]) for x in mm2]
    led.add("n1k_prod_iters_per_block_min_max", [min(its), max(its)], "iterations", rel(RES / "prod_N1000_G128.log"), f"{len(its)} blocks logged")
i4log = (RES / "i4_n1000_circ_G128.log").read_text()
mm = re.findall(r"solve wall (\d+)s, theta applications (\d+)", i4log)
if mm:
    led.add("n1k_circ_prod_wall_s", int(mm[-1][0]), "s", rel(RES / "i4_n1000_circ_G128.log"), "REPORT_N10K: 8,876 s")
    led.add("n1k_circ_prod_theta_apps", int(mm[-1][1]), "applications", rel(RES / "i4_n1000_circ_G128.log"))
g96 = (RES / "conv_N1000_G96.log").read_text()
mm = re.findall(r"solve wall (\d+)s, theta applications (\d+)", g96)
if mm:
    led.add("n1k_G96_wall_s_theta", [int(mm[0][0]), int(mm[0][1])], "s, applications", rel(RES / "conv_N1000_G96.log"))

# ------------------------------------------------------------------ N=10k full state table
W = RES / "n10k_G192_window"
lam = np.load(W / "window_eigenvalues.npy")
lam_raw = np.load(W / "window_eigenvalues_raw.npy")
res = np.load(W / "window_residuals.npy")
nrm = np.load(W / "window_norms.npy")
man = load_json(W / "vec_manifest.json")
loc = load_json(TAB / "loc_n10k.json")["rows"]
seam = {1.8707585792024861: "seam", 1.8730821374768227: "seam", 1.929596209673228: "seam", 1.947209447202747: "seam*"}
ext = {1.944051593936228: "ext."}
pm = load_json(RES / "gates" / "periodic_overlap_match.json")
peri = {round(p["lam_mont"], 9): p for p in pm["pairs"]}
with open(TAB / "states_n10k.tex", "w") as f, open(TAB / "states_n10k.csv", "w", newline="") as fc:
    wr = csv.writer(fc)
    wr.writerow(["i", "tile_label", "lambda_um-2", "lambda_raw", "norm_sq", "nu_a2.288", "residual_reported", "pr_fraction", "pr_volume_um3", "xi_um", "r2", "dyn_range_dec",
                 "resolved", "shell2_energy_frac", "slice", "in_gap_bracket", "class", "periodic_partner_lam", "periodic_overlap"])
    f.write("\\begin{longtable}{@{}rrllllllllll@{}}\n")
    f.write("\\caption{All 133 residual-certified $N=10^4$ eigenstates (corrected $\\lambda=\\lambda_\\mathrm{raw}/\\|x\\|^2$). $\\nu=\\sqrt{\\lambda}\\,a/2\\pi$ with $a=2.288\\,\\mu$m. Residuals are the reported (raw-$\\lambda$) values. $p$ is the participation fraction; $\\xi$ the envelope decay length (`u' = unresolved: lower bound only; ceiling $L/2=12.32\\,\\mu$m); $f_2$ the energy fraction in the outer 2-voxel shell (volume fraction 6.1\\%). Class: g = inside the nominal bracket [1.864, 1.996]; seam = seam-flagged (18--44\\% outer-shell energy; no partner in the incomplete periodic re-solve; seam* = undetermined, see text); ext.\\ = compact but non-exponential in-gap mode (unresolved $\\xi$). Periodic: best-overlap partner in the seam-free re-solve (in-gap states only).}\\label{tab:states}\\\\\n")
    f.write("\\hline\\hline\n \\# & tile & $\\lambda$ ($\\mu$m$^{-2}$) & $\\nu$ & res. & $p$ (\\%) & $\\xi$ ($\\mu$m) & $r^2$ & $f_2$ (\\%) & slice & class & periodic \\\\\n\\hline\n\\endfirsthead\n")
    f.write("\\hline\n \\# & tile & $\\lambda$ & $\\nu$ & res. & $p$ (\\%) & $\\xi$ & $r^2$ & $f_2$ (\\%) & slice & class & periodic \\\\\n\\hline\n\\endhead\n\\hline\n\\endfoot\n\\hline\\hline\n\\endlastfoot\n")
    for i in range(133):
        r = loc[i]
        assert abs(r["lam"] - lam[i]) < 1e-9
        ing = GAP_LO_10K < lam[i] < GAP_HI_10K
        cls = seam.get(lam[i], ext.get(lam[i], "g" if ing else ""))
        if cls == "g":
            cls = "g (cand.)"
        xi_s = "--" if r["xi_um"] is None else (f"{r['xi_um']:.2f}" + ("u" if r["unresolved"] else ""))
        sl = man[i]["dir"].split("_")[-1]
        pp = peri.get(round(lam[i], 9))
        per_s = f"{pp['lam_peri']:.4f} ({pp['overlap']:.3f})" if pp and pp["overlap"] > 0.5 else (f"-- ({pp['overlap']:.3f})" if pp else "")
        f.write(f" {i} & {4942 + i} & {lam[i]:.5f} & {nu_from_lam(lam[i]):.4f} & {sci(res[i])} & {100 * r['pr_fraction']:.3f} & {xi_s} & {r['r2']:.2f} & {100 * r['shell2_energy_frac']:.1f} & {sl} & {cls} & {per_s} \\\\\n")
        wr.writerow([i, 4942 + i, f"{lam[i]:.9f}", f"{lam_raw[i]:.9f}", f"{nrm[i]:.9f}", f"{nu_from_lam(lam[i]):.6f}", f"{res[i]:.4e}", f"{r['pr_fraction']:.6e}",
                     f"{r['pr_volume_um3']:.4f}", "" if r["xi_um"] is None else f"{r['xi_um']:.4f}", f"{r['r2']:.4f}", f"{r['dyn_range_dec']:.3f}",
                     not r["unresolved"], f"{r['shell2_energy_frac']:.5f}", sl, ing, cls, pp["lam_peri"] if pp and pp["overlap"] > 0.5 else "", pp["overlap"] if pp else ""])
    f.write("\\end{longtable}\n")
led.add("n10k_state_table_rows", 133, "rows", rel(TAB / "states_n10k.csv"))
# in-gap compact table
with open(TAB / "ingap.tex", "w") as f:
    f.write("\\begin{tabular}{@{}lrrrrrrl@{}}\n\\hline\\hline\n $\\lambda$ ($\\mu$m$^{-2}$) & $\\nu$ & $p$ (\\%) & $V_p$ ($\\mu$m$^3$) & $\\xi$ ($\\mu$m) & $r^2$ & $f_2$ (\\%) & verdict \\\\\n\\hline\n")
    for i in range(133):
        if not (GAP_LO_10K < lam[i] < GAP_HI_10K):
            continue
        r = loc[i]
        cls = seam.get(lam[i], ext.get(lam[i], "cand."))
        pp = peri.get(round(lam[i], 9))
        if cls == "cand.":
            verdict = f"localized; persists (ov.\\ {pp['overlap']:.3f}, $\\Delta\\lambda={pp['dlam']:+.4f}$)"
        elif cls == "seam":
            verdict = f"seam-flagged; no partner in the incomplete re-solve (best ov.\\ {pp['overlap']:.3f})"
        elif cls == "seam*":
            verdict = f"seam-flagged; undetermined (best ov.\\ {pp['overlap']:.2f}, budget 0.097)"
        else:
            verdict = f"compact, non-exponential envelope ($r^2=0.33$, unresolved); partner 1.9407 fits $\\xi=2.2$"
        xi_s = f"{r['xi_um']:.2f}" + ("$^\\mathrm{u}$" if r["unresolved"] else "")
        f.write(f" {lam[i]:.4f} & {nu_from_lam(lam[i]):.4f} & {100 * r['pr_fraction']:.3f} & {r['pr_volume_um3']:.1f} & {xi_s} & {r['r2']:.3f} & {100 * r['shell2_energy_frac']:.1f} & {verdict} \\\\\n")
    f.write("\\hline\\hline\n\\end{tabular}\n")

# ------------------------------------------------------------------ N=1000 windows (CSV) + band 495-506 table
for tag, d, evp, lamw in (("n1k_ell", USB, USB / "eigenvalues_all.npy", USB / "window_eigenvalues.npy"),
                          ("n1k_circ", RES / "i4_n1000_circ_G128", RES / "i4_n1000_circ_G128" / "eigenvalues_all.npy", RES / "i4_n1000_circ_G128" / "window_eigenvalues.npy")):
    locw = load_json(TAB / f"loc_{tag}.json")["rows"]
    ev = np.load(evp).astype(np.float64)
    with open(TAB / f"states_{tag}.csv", "w", newline="") as fc:
        wr = csv.writer(fc)
        wr.writerow(["mpb_band", "lambda_um-2", "nu_a2.288", "pr_fraction", "pr_volume_um3", "xi_um", "r2", "resolved", "shell2_energy_frac"])
        for i, r in enumerate(locw):
            wr.writerow([398 + i, f"{r['lam']:.9f}", f"{nu_from_lam(r['lam']):.6f}", f"{r['pr_fraction']:.6e}", f"{r['pr_volume_um3']:.4f}",
                         "" if r["xi_um"] is None else f"{r['xi_um']:.4f}", f"{r['r2']:.4f}", not r["unresolved"], f"{r['shell2_energy_frac']:.5f}"])
    with open(TAB / f"spectrum_{tag}.csv", "w", newline="") as fc:
        wr = csv.writer(fc)
        wr.writerow(["mpb_band", "lambda_um-2", "nu_a2.288"])
        for i, l in enumerate(ev):
            wr.writerow([i + 3, f"{l:.9f}", f"{nu_from_lam(l):.6f}"])
    with open(TAB / f"edge_{tag}.tex", "w") as f:
        f.write("\\begin{tabular}{@{}rrrrrrl@{}}\n\\hline\\hline\n band & $\\lambda$ ($\\mu$m$^{-2}$) & $\\nu$ & $p$ (\\%) & $V_p$ ($\\mu$m$^3$) & $\\xi$ ($\\mu$m) & $r^2$ \\\\\n\\hline\n")
        for bnd in range(494, 508):
            r = locw[bnd - 398]
            xi_s = "--" if r["xi_um"] is None else (f"{r['xi_um']:.2f}" + ("$^\\mathrm{u}$" if r["unresolved"] else ""))
            f.write(f" {bnd} & {r['lam']:.5f} & {nu_from_lam(r['lam']):.4f} & {100 * r['pr_fraction']:.2f} & {r['pr_volume_um3']:.1f} & {xi_s} & {r['r2']:.2f} \\\\\n")
        f.write("\\hline\\hline\n\\end{tabular}\n")

# ------------------------------------------------------------------ I6 cross-grid table
with open(TAB / "crossgrid.tex", "w") as f:
    f.write("\\begin{tabular}{@{}lrrrrl@{}}\n\\hline\\hline\n edge & $\\lambda_{192}$ & $\\lambda_{256}$ & overlap & $\\Delta\\omega/\\omega$ (\\%) & note \\\\\n\\hline\n")
    for key, fn, note in (("low", "crossgrid_match.json", ""), ("high", "crossgrid_match_n10k_G256_edgehigh_narrow.json", "")):
        cg = load_json(RES / "gates" / fn)
        for p in cg["pairs"]:
            dww = 100 * (np.sqrt(p["lam_fine"]) - np.sqrt(p["lam_coarse"])) / np.sqrt(p["lam_coarse"])
            n_ = "unconverged fine vector" if p.get("fine_converged") is False else ""
            if p["overlap"] <= 0.5:
                n_ = "no partner (filter transition zone)"
            f.write(f" {key} & {p['lam_coarse']:.5f} & {p['lam_fine']:.5f} & {p['overlap']:.3f} & {dww:+.3f} & {n_} \\\\\n")
    f.write("\\hline\\hline\n\\end{tabular}\n")

# ------------------------------------------------------------------ MPB parity table (from ledger of s01)
s01 = load_json(TAB.parent / "numbers" / "s01_spectra.json")
with open(TAB / "parity.tex", "w") as f:
    f.write("\\begin{tabular}{@{}llrrr@{}}\n\\hline\\hline\n test & grid / bands & $n$ & max $\\Delta\\omega/\\omega$ & median \\\\\n\\hline\n")
    for lab, k in (("MPB parity (32$^3$, 150 bands, tol $10^{-9}$)", "32"), ("G3: MPB parity (64$^3$, 300 bands, tol $10^{-7}$)", "64"), ("G3w: MPB parity (64$^3$, 660 bands)", "64w")):
        f.write(f" {lab} & & {s01[f'mpb_parity_{k}_n_bands']['value']} & {sci(s01[f'mpb_parity_{k}_max_dw_w']['value'])} & {sci(s01[f'mpb_parity_{k}_median_dw_w']['value'])} \\\\\n")
    f.write(f" G9: c64 vs c128 (48$^3$, 96 bands) & & {s01['g9_n_bands']['value']} & {sci(s01['g9_max_dw_w']['value'])} & -- \\\\\n")
    f.write(f" I1: interior vs bottom-up ($N=1000$, 128$^3$; $\\Delta\\lambda/\\lambda$) & & {s01['i1_n_reported']['value']} & {sci(s01['i1_max_dlam_lam_corrected']['value'])} & {sci(s01['i1_median_dlam_lam_corrected']['value'])} \\\\\n")
    f.write(f" I4: interior vs bottom-up ($N=1000$ circ.; $\\Delta\\lambda/\\lambda$) & & {s01['i4int_below_n']['value'] + s01['i4int_above_n']['value']} & {sci(s01['i4int_pooled_max_dlam_lam']['value'])} & {sci(s01['i4int_pooled_median_dlam_lam']['value'])} \\\\\n")
    f.write("\\hline\\hline\n\\end{tabular}\n")
led.save()
print("tables written:", sorted(p.name for p in TAB.iterdir()))
