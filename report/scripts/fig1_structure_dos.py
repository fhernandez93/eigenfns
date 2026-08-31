#!/usr/bin/env python
"""Fig. 1: (a,b) mid-plane eps(r) slices of the N=1000 and N=10^4 networks
with the same circular decoration at the production grids; (c) the N=1000
exact spectrum (611 bands, circular decoration) on its KPM DOS and (d) the
N=10^4 KPM DOS (256^3) with the 133 certified 192^3 eigenvalues; nominal gap
brackets shaded. Sources: report/figures/src/eps_slice_*.npy (s03),
dos_*.npz (s04), the eigenvalue files."""
from __future__ import annotations

import numpy as np
from matplotlib import gridspec
from matplotlib.colors import ListedColormap

from common import FIG, GAP_HI_10K, GAP_LO_10K, L_N1K, L_N10K, RES, Ledger, load_json
from figstyle import BLACK, BLUE, DOUBLE, GREY, LGREY, ORANGE, VERM, panel_label, plt, save

SRC = FIG / "src"
led = Ledger(__file__)
s1 = load_json(FIG.parent / "numbers" / "s01_spectra.json")
gap_c = (s1["n1k_circ_gap_lo_128"]["value"], s1["n1k_circ_gap_hi_128"]["value"])

fig = plt.figure(figsize=(DOUBLE, 2.4))
gs = gridspec.GridSpec(2, 3, width_ratios=[1.0, 1.0, 2.6], height_ratios=[1, 1], wspace=0.28, hspace=0.12,
                       left=0.03, right=0.99, top=0.90, bottom=0.16)
cmap = ListedColormap(["white", "#404040"])
# (a) N=1000 circular 128^3
ax = fig.add_subplot(gs[:, 0])
e1 = np.load(SRC / "eps_slice_n1k_128_circ.npy")
ax.imshow((e1 > 1.5).T, cmap=cmap, origin="lower", extent=[0, L_N1K, 0, L_N1K], interpolation="nearest")
ax.set_xticks([]); ax.set_yticks([])
ax.plot([0.6, 5.6], [0.7, 0.7], color=BLACK, lw=1.5)
ax.text(3.1, 1.1, "5 µm", ha="center", va="bottom", fontsize=7)
ax.set_title(r"$N=10^3$, $L=11.44\,\mu$m, $128^3$", fontsize=7, pad=2)
panel_label(ax, "(a)", x=-0.02, y=1.10)
# (b) N=10k 192^3 -- same physical scale would make (a) tiny; equal panel size, scale bars
ax = fig.add_subplot(gs[:, 1])
e10 = np.load(SRC / "eps_slice_n10k_192.npy")
ax.imshow((e10 > 1.5).T, cmap=cmap, origin="lower", extent=[0, L_N10K, 0, L_N10K], interpolation="nearest")
ax.set_xticks([]); ax.set_yticks([])
ax.plot([1.2, 6.2], [1.4, 1.4], color=BLACK, lw=1.5)
ax.text(3.7, 2.2, "5 µm", ha="center", va="bottom", fontsize=7)
ax.set_title(r"$N=10^4$, $L=24.65\,\mu$m, $192^3$", fontsize=7, pad=2)
panel_label(ax, "(b)", x=-0.02, y=1.10)
# (c) N=1000 circular spectrum + DOS
axc = fig.add_subplot(gs[0, 2])
axd = fig.add_subplot(gs[1, 2], sharex=axc)
ev_c = np.load(RES / "i4_n1000_circ_G128" / "eigenvalues_all.npy")
lo, hi = 1.60, 2.30
# exact density of states of the SAME spectrum: 0.01-wide bins -> states per unit lambda
edges = np.arange(lo, hi + 1e-9, 0.01)
h, _ = np.histogram(ev_c, bins=edges)
rho_h = np.maximum(h / 0.01, 1)
axc.fill_between(edges[:-1], 1, rho_h, step="post", color=LGREY, lw=0)
axc.step(edges[:-1], rho_h, where="post", color=GREY, lw=0.8, label="exact DOS (0.01 bins)")
axc.axvspan(gap_c[0], gap_c[1], color=ORANGE, alpha=0.25, lw=0, label="gap 500|501 (exact)")
mm = (ev_c >= lo) & (ev_c <= hi)
axc.vlines(ev_c[mm], 1, 3.5, color=BLUE, lw=0.6, label=f"exact eigenvalues ({mm.sum()})")
axc.set_yscale("log"); axc.set_ylim(1, 1e4)
axc.set_ylabel(r"$\rho(\lambda)$", labelpad=1)
axc.text(0.01, 0.04, r"$N=10^3$ circ., $L/2=5.72\,\mu$m, gap $\Delta\nu/\nu=5.07\%$", transform=axc.transAxes, va="bottom", fontsize=6.5)
axc.legend(loc="upper right", fontsize=6, ncol=1, handlelength=1.2, frameon=True, framealpha=0.9, edgecolor="none", borderaxespad=0.1)
plt.setp(axc.get_xticklabels(), visible=False)
panel_label(axc, "(c)", x=-0.13, y=0.85)
# (d) N=10k DOS + 133 eigenvalues
z = np.load(SRC / "dos_n10k_256.npz")
lam10 = np.load(RES / "n10k_G192_window" / "window_eigenvalues.npy")
m = (z["lam"] >= lo) & (z["lam"] <= hi)
axd.fill_between(z["lam"][m], 1, np.maximum(z["rho"][m], 1), color=LGREY, lw=0)
axd.plot(z["lam"][m], np.maximum(z["rho"][m], 1), color=GREY, lw=0.8, label=r"KPM DOS ($256^3$, $d=12000$)")
axd.axvspan(GAP_LO_10K, GAP_HI_10K, color=ORANGE, alpha=0.25, lw=0, label=r"nominal gap, $\rho<160$")
ing = (lam10 > GAP_LO_10K) & (lam10 < GAP_HI_10K)
axd.vlines(lam10[~ing], 1, 3.5, color=BLUE, lw=0.6, label=f"other certified eigenvalues ({(~ing).sum()})")
axd.vlines(lam10[ing], 1, 12, color=VERM, lw=1.0, label=f"in-gap states ({ing.sum()})")
axd.axvspan(1.757, 2.117, ymin=0.0, ymax=0.04, color=BLUE, alpha=0.3, lw=0)
axd.set_yscale("log"); axd.set_ylim(1, 1e5)
axd.set_xlim(lo, hi)
axd.set_xlabel(r"$\lambda=(\omega/c)^2$ ($\mu$m$^{-2}$)", labelpad=1)
axd.set_ylabel(r"$\rho(\lambda)$", labelpad=1)
axd.text(0.01, 0.04, r"$N=10^4$, $L/2=12.32\,\mu$m, window [1.757, 2.117] (bar)", transform=axd.transAxes, va="bottom", fontsize=6.5)
axd.legend(loc="upper right", fontsize=6, ncol=2, handlelength=1.2, columnspacing=0.8, frameon=True, framealpha=0.9, edgecolor="none", borderaxespad=0.1)
panel_label(axd, "(d)", x=-0.13, y=0.85)
# secondary axis: nu = sqrt(lam) a / 2pi, a = 2.288
sec = axc.secondary_xaxis("top", functions=(lambda l: np.sqrt(np.maximum(l, 0)) * 2.288 / (2 * np.pi), lambda n: (2 * np.pi * n / 2.288) ** 2))
sec.set_xlabel(r"$\nu=\omega a/2\pi c$ ($a=2.288\,\mu$m)", labelpad=2, fontsize=7)
sec.tick_params(labelsize=6)
save(fig, str(FIG / "fig1_structure_dos"))
led.add("fig1_lambda_range", [lo, hi], "um^-2", "figure axis")
led.add("fig1_n1k_eigenvalues_shown", int(mm.sum()), "states", "results/i4_n1000_circ_G128/eigenvalues_all.npy")
led.save()
