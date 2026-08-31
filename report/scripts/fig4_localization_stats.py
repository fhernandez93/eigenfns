#!/usr/bin/env python
"""Fig. 4: (a) adjacent-gap ratio <r> per spectral band (N=10^4 and N=10^3)
with Poisson / GOE references and a sliding-window <r>(lambda) for N=10^4;
(b) matched-decoration comparison: xi of resolved states vs distance to the
nearest gap edge for N=10^3 (L=11.44) and N=10^4 (L=24.65). Data:
report/tables/levelstats.json, loc_*.json."""
from __future__ import annotations

import numpy as np

from common import FIG, GAP_HI_10K, GAP_LO_10K, L_N1K, L_N10K, TAB, Ledger, load_json
from figstyle import BLUE, GREEN, GREY, LGREY, ORANGE, SINGLE, VERM, panel_label, plt, save

led = Ledger(__file__)
ls = load_json(TAB / "levelstats.json")
s5 = load_json(FIG.parent / "numbers" / "s05_levelstats.json")
RP, RG = s5["r_poisson"]["value"], s5["r_goe"]["value"]
s1 = load_json(FIG.parent / "numbers" / "s01_spectra.json")
gap_c = (s1["n1k_circ_gap_lo_128"]["value"], s1["n1k_circ_gap_hi_128"]["value"])

fig, (ax, bx) = plt.subplots(2, 1, figsize=(SINGLE, 4.0), gridspec_kw=dict(hspace=0.42, left=0.15, right=0.98, top=0.97, bottom=0.11))
# ---- (a) sliding window r(lambda) N=10k + band points
sl = ls["sliding_n10k_w15"]["rows"]
lc = np.array([r["lam_centre"] for r in sl]); rm = np.array([r["r_mean"] for r in sl]); rs = np.array([r["r_se"] for r in sl])
ax.axvspan(GAP_LO_10K, GAP_HI_10K, color=ORANGE, alpha=0.22, lw=0)
ax.axhline(RP, color=GREY, ls=":", lw=0.9); ax.text(1.56, RP - 0.035, "Poisson 0.386", fontsize=6, ha="left", color=GREY)
ax.axhline(RG, color=GREY, ls="--", lw=0.9); ax.text(1.56, RG + 0.012, "GOE 0.531", fontsize=6, ha="left", color=GREY)
ax.fill_between(lc, rm - rs, rm + rs, color=BLUE, alpha=0.18, lw=0)
ax.plot(lc, rm, color=BLUE, lw=1.0, label=r"$N=10^4$, sliding 15-level window")
bands = [("n10k_below_far", BLUE, "o"), ("n10k_below_near", BLUE, "o"), ("n10k_in_gap_all10", VERM, "D"), ("n10k_above_near", BLUE, "o"), ("n10k_above_far", BLUE, "o")]
for k, c, mk in bands:
    o = ls[k]
    x0, x1 = o["lam_range"]
    ax.errorbar(0.5 * (x0 + x1), o["r_mean"], yerr=o["r_se_boot"], xerr=0.5 * (x1 - x0), fmt=mk, ms=3.5, color=c, mec="none", elinewidth=0.8, capsize=0, zorder=4)
    ax.text(0.5 * (x0 + x1), o["r_mean"] + o["r_se_boot"] + 0.02, f"n={o['n_levels']}", fontsize=5.5, ha="center", color=c)
# N=1000 bands drawn as short horizontal bars at their lambda ranges
for k in ("n1k_circ_below_edge_401_500", "n1k_circ_above_edge_501_600"):
    o = ls[k]
    x0, x1 = o["lam_range"]
    ax.errorbar(0.5 * (x0 + x1), o["r_mean"], yerr=o["r_se_boot"], xerr=0.5 * (x1 - x0), fmt="s", ms=3.5, color=GREEN, mec="none", elinewidth=0.8, capsize=0, zorder=4)
    ax.text(0.5 * (x0 + x1), o["r_mean"] - o["r_se_boot"] - 0.045, f"$N=10^3$, n={o['n_levels']}", fontsize=5.5, ha="center", color=GREEN)
ax.set_xlim(1.55, 2.37); ax.set_ylim(0.15, 0.72)
ax.set_xlabel(r"$\lambda$ ($\mu$m$^{-2}$)", labelpad=1)
ax.set_ylabel(r"$\langle r\rangle$")
ax.legend(loc="lower right", fontsize=6, handlelength=1.2)
panel_label(ax, "(a)", x=-0.18, y=0.93)
# ---- (b) matched decoration: xi vs distance to nearest gap edge
r10 = load_json(TAB / "loc_n10k.json")["rows"]
r1 = load_json(TAB / "loc_n1k_circ.json")["rows"]
seam = {1.8707585792024861, 1.8730821374768227, 1.929596209673228, 1.947209447202747}
edges10 = (1.8860078588720002, 1.926413256982914)   # largest interior spacing (exact edges)


def dist(lam, e):
    # signed distance: negative below the lower edge, positive above the upper edge, 0 inside
    if lam < e[0]:
        return lam - e[0]
    if lam > e[1]:
        return lam - e[1]
    return 0.0


d10 = np.array([dist(r["lam"], edges10) for r in r10]); x10 = np.array([r["xi_um"] if r["xi_um"] else np.nan for r in r10]); u10 = np.array([r["unresolved"] for r in r10])
s10 = np.array([r["lam"] in seam for r in r10])
d1 = np.array([dist(r["lam"], gap_c) for r in r1]); x1 = np.array([r["xi_um"] if r["xi_um"] else np.nan for r in r1]); u1 = np.array([r["unresolved"] for r in r1])
bx.axhline(L_N1K / 2, ls="--", color=VERM, lw=0.8); bx.text(0.3, L_N1K / 2 + 0.25, r"$L/2$, $N=10^3$", fontsize=6, color=VERM, ha="right")
bx.axhline(L_N10K / 2, ls="--", color=BLUE, lw=0.8); bx.text(0.3, L_N10K / 2 + 0.25, r"$L/2$, $N=10^4$", fontsize=6, color=BLUE, ha="right")
bx.axvline(0, color=GREY, lw=0.5)
bx.plot(d10[~u10 & ~s10], x10[~u10 & ~s10], "o", ms=3, color=BLUE, mec="none", label=r"$N=10^4$ resolved")
bx.plot(d10[~u10 & s10], x10[~u10 & s10], "o", ms=3.5, mfc="none", mec=BLUE, mew=0.8, label="seam artefacts")
bx.plot(d10[u10], np.full(u10.sum(), L_N10K / 2), "^", ms=3.5, color=BLUE, alpha=0.6, mec="none")
bx.plot(d1[~u1], x1[~u1], "s", ms=3, color=VERM, mec="none", label=r"$N=10^3$ resolved")
bx.plot(d1[u1], np.full(u1.sum(), L_N1K / 2), "^", ms=3.5, color=VERM, alpha=0.6, mec="none", label="unresolved (lower bound)")
bx.set_xlim(-0.4, 0.3); bx.set_ylim(0, 14)
bx.set_xlabel(r"$\lambda-\lambda_\mathrm{edge}$ ($\mu$m$^{-2}$; own gap edges, 0 = inside)", labelpad=1)
bx.set_ylabel(r"$\xi$ ($\mu$m)")
bx.legend(loc="upper left", fontsize=6, ncol=2, handlelength=1.0, columnspacing=0.8)
panel_label(bx, "(b)", x=-0.18, y=0.93)
save(fig, str(FIG / "fig4_localization_stats"))
# numbers used in caption: resolved xi within 0.05 of an edge, both sizes
near10 = (~u10) & (~s10) & (np.abs(d10) <= 0.05)
near1 = (~u1) & (np.abs(d1) <= 0.05)
led.add("fig4_near_edge_xi_n10k", [float(np.nanmin(x10[near10])), float(np.nanmax(x10[near10])), int(near10.sum())], "um, um, n", "report/tables/loc_n10k.json",
        "resolved non-seam states within 0.05 of the exact edges 1.8860|1.9264 (in-gap states count as distance 0)")
led.add("fig4_near_edge_xi_n1k", [float(np.nanmin(x1[near1])), float(np.nanmax(x1[near1])), int(near1.sum())], "um, um, n", "report/tables/loc_n1k_circ.json",
        "resolved states within 0.05 of the exact edges 1.8276|2.0225")
led.save()
