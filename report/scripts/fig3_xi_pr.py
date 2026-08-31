#!/usr/bin/env python
"""Fig. 3: (a) envelope decay length xi vs lambda for N=10^4 (192^3) and
N=1000 circular (128^3), same decoration; ceilings L/2 drawn; unresolved
fits as upward lower-bound markers at the ceiling; nominal gaps shaded.
(b) participation fraction vs lambda for both. Data: report/tables/loc_*.json."""
from __future__ import annotations

import numpy as np

from common import FIG, GAP_HI_10K, GAP_LO_10K, L_N1K, L_N10K, TAB, Ledger, load_json
from figstyle import BLUE, GREY, ORANGE, SINGLE, VERM, panel_label, plt, save

led = Ledger(__file__)
s1 = load_json(FIG.parent / "numbers" / "s01_spectra.json")
gap_c = (s1["n1k_circ_gap_lo_128"]["value"], s1["n1k_circ_gap_hi_128"]["value"])
r10 = load_json(TAB / "loc_n10k.json")["rows"]
r1 = load_json(TAB / "loc_n1k_circ.json")["rows"]
seam = {1.8707585792024861, 1.8730821374768227, 1.929596209673228, 1.947209447202747}


def arrays(rows):
    lam = np.array([r["lam"] for r in rows])
    xi = np.array([r["xi_um"] if r["xi_um"] else np.nan for r in rows])
    un = np.array([r["unresolved"] for r in rows])
    pr = np.array([r["pr_fraction"] for r in rows])
    return lam, xi, un, pr


l10, x10, u10, p10 = arrays(r10)
l1, x1, u1, p1 = arrays(r1)
fig, (ax, bx) = plt.subplots(2, 1, figsize=(SINGLE, 3.3), sharex=True, gridspec_kw=dict(hspace=0.08, left=0.14, right=0.98, top=0.97, bottom=0.12))
# gaps
for a in (ax, bx):
    a.axvspan(GAP_LO_10K, GAP_HI_10K, color=ORANGE, alpha=0.22, lw=0)
    a.axvspan(gap_c[0], gap_c[1], color="none", edgecolor=GREY, hatch="////", lw=0, alpha=0.5)
# xi
ax.axhline(L_N10K / 2, ls="--", color=BLUE, lw=0.8)
ax.axhline(L_N1K / 2, ls="--", color=VERM, lw=0.8)
ax.text(2.54, L_N10K / 2 - 0.9, r"$L/2$ ($N=10^4$)", fontsize=6, color=BLUE, ha="right")
ax.text(2.54, L_N1K / 2 + 0.35, r"$L/2$ ($N=10^3$)", fontsize=6, color=VERM, ha="right")
s10 = np.array([l in seam for l in l10])
ax.plot(l10[~u10 & ~s10], x10[~u10 & ~s10], "o", ms=3, mfc=BLUE, mec="none", label=r"$N=10^4$ resolved (%d)" % (~u10).sum(), zorder=3)
ax.plot(l10[~u10 & s10], x10[~u10 & s10], "o", ms=3.5, mfc="none", mec=BLUE, mew=0.8, label="seam-flagged (4)", zorder=4)
ax.plot(l10[u10], np.full(u10.sum(), L_N10K / 2), "^", ms=4, color=BLUE, mec="none", alpha=0.7, label=r"$N=10^4$ unresolved: $\xi\geq$ bound (%d)" % u10.sum(), zorder=3)
ax.plot(l1[~u1], x1[~u1], "s", ms=3, color=VERM, mec="none", label=r"$N=10^3$ resolved (%d)" % (~u1).sum(), zorder=3)
ax.plot(l1[u1], np.full(u1.sum(), L_N1K / 2), "^", ms=4, color=VERM, mec="none", alpha=0.6, label=r"$N=10^3$ unresolved (%d)" % u1.sum(), zorder=2)
ax.set_ylim(0, 17.5)
ax.set_ylabel(r"$\xi$ ($\mu$m)")
ax.legend(loc="upper left", fontsize=5.5, ncol=2, handlelength=1.0, columnspacing=0.8, borderaxespad=0.2, bbox_to_anchor=(0.0, 1.0))
panel_label(ax, "(a)", x=-0.17, y=0.94)
# PR
bx.semilogy(l10[~s10], 100 * p10[~s10], "o", ms=3, color=BLUE, mec="none", label=r"$N=10^4$ (133)")
bx.semilogy(l10[s10], 100 * p10[s10], "o", ms=3.5, mfc="none", mec=BLUE, mew=0.8)
bx.semilogy(l1, 100 * p1, "s", ms=3, color=VERM, mec="none", label=r"$N=10^3$ (210)")
bx.set_ylim(0.02, 60)
bx.set_ylabel(r"participation fraction $p$ (%)")
bx.set_xlabel(r"$\lambda=(\omega/c)^2$ ($\mu$m$^{-2}$)")
bx.set_xlim(1.45, 2.56)
bx.legend(loc="lower right", fontsize=6, handlelength=1.0)
panel_label(bx, "(b)", x=-0.17, y=0.94)
save(fig, str(FIG / "fig3_xi_pr"))
led.add("fig3_n10k_resolved_shown", int((~u10).sum()), "modes", "report/tables/loc_n10k.json")
led.add("fig3_n1k_resolved_shown", int((~u1).sum()), "modes", "report/tables/loc_n1k_circ.json")
led.save()
