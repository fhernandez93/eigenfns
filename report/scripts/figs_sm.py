#!/usr/bin/env python
"""Supplemental Material figures (all from saved data, CPU):
  figS_montage_n1k_ell       full 210-tile N=1000 elliptical montage (regenerated, downscaled)
  figS_montage_sbs           original cluster montage vs our regeneration (USB side-by-side)
  figS_montage_n10k          133-tile N=10^4 gap-edge montage
  figS_montage_n1k_circ      210-tile N=1000 circular montage
  figS_seam                  outer-shell energy fraction per mode (N=10^4 and both N=1000 windows)
  figS_bakeoff               median in-window residual vs outer iteration for the bake-off arms
  figS_residuals_rayleigh    residual histograms per slice + Delta lambda/lambda vs ||x||^2-1
  figS_dos_full              full-bandwidth KPM DOS, both structures, with the gap regions
  figS_levelstats            P(r) and P(s) for N=1000 (611) and N=10^4 outside / in-gap, with Poisson/GOE
  figS_crossgrid             I6: Delta omega/omega and overlap per state, 192^3 vs 256^3
  figS_periodic              seam test: overlap matrix montage in-gap states x periodic states
  figS_fitrange              xi(r_hi=0.60) vs xi(r_hi=0.95)
  figS_g5                    N=1000 omega(G) for the six G5 bands
  figS_kpm_validation        N=1000 KPM counting function vs the exact spectrum
  figS_sliding_r_n1k         sliding <r>(lambda) for the two N=1000 spectra
"""
from __future__ import annotations

import json

import numpy as np
from PIL import Image

from common import FIG, GAP_HI_10K, GAP_LO_10K, L_N1K, L_N10K, RES, TAB, USB, Ledger, load_json
from figstyle import BLACK, BLUE, DOUBLE, GREEN, GREY, LGREY, ORANGE, PURPLE, SINGLE, SKY, VERM, panel_label, plt, save

Image.MAX_IMAGE_PIXELS = None
led = Ledger(__file__)
SRC = FIG / "src"


def downscale(path, out, width=2400):
    im = Image.open(path).convert("RGB")
    im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
    im.save(out, optimize=True)
    print("wrote", out, im.size)
    return im.size


# ---- montages (PNG only; embedded raster) -------------------------------------
sz = downscale(USB / "band_montage_398_607_15_non_ideal_regen.png", FIG / "figS_montage_n1k_ell.png")
led.add("figS_montage_n1k_ell_src", str(USB / "band_montage_398_607_15_non_ideal_regen.png"), "file", "USB", f"downscaled to {sz}")
sz = downscale(USB / "montage_side_by_side.png", FIG / "figS_montage_sbs.png", width=1800)
led.add("figS_montage_sbs_src", str(USB / "montage_side_by_side.png"), "file", "USB", f"downscaled to {sz}")
sz = downscale(RES / "n10k_G192_window" / "band_montage_n10k_gapedge_15.png", FIG / "figS_montage_n10k.png")
sz = downscale(RES / "i4_n1000_circ_G128" / "band_montage_398_607_15_non_ideal_regen.png", FIG / "figS_montage_n1k_circ.png")
# PDF wrappers for the montages so \includegraphics can use either
for name in ("figS_montage_n1k_ell", "figS_montage_sbs", "figS_montage_n10k", "figS_montage_n1k_circ"):
    im = Image.open(FIG / f"{name}.png")
    fig = plt.figure(figsize=(DOUBLE, DOUBLE * im.height / im.width))
    ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(im); ax.set_axis_off()
    fig.savefig(FIG / f"{name}.pdf", dpi=200)
    plt.close(fig)

# ---- seam diagnostic ------------------------------------------------------------
r10 = load_json(TAB / "loc_n10k.json")["rows"]
re_ = load_json(TAB / "loc_n1k_ell.json")["rows"]
rc = load_json(TAB / "loc_n1k_circ.json")["rows"]
seam = {1.8707585792024861, 1.8730821374768227, 1.929596209673228, 1.947209447202747}
fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.4), gridspec_kw=dict(wspace=0.25, left=0.07, right=0.99, top=0.92, bottom=0.18))
ax = axs[0]
lam = np.array([r["lam"] for r in r10]); enh = np.array([r["shell2_enhancement"] for r in r10]); pf = np.array([r["peak_face_coords"] for r in r10])
ax.axvspan(GAP_LO_10K, GAP_HI_10K, color=ORANGE, alpha=0.22, lw=0)
ax.axhline(1, color=GREY, lw=0.6)
s = np.array([l in seam for l in lam])
ax.plot(lam[~s], enh[~s], "o", ms=3, color=BLUE, mec="none", label="N=10$^4$ modes")
ax.plot(lam[s], enh[s], "o", ms=5, mfc="none", mec=VERM, mew=1.0, label="seam-flagged (4)")
ax.plot(lam[pf >= 2], enh[pf >= 2], "x", ms=4, color=BLACK, label="peak voxel on $\\geq$2 faces")
ax.set_xlabel(r"$\lambda$ ($\mu$m$^{-2}$)"); ax.set_ylabel("outer-shell energy enhancement $f_2/0.061$")
ax.set_ylim(0, 8); ax.legend(fontsize=6, loc="upper right")
ax.set_title(r"$N=10^4$, $192^3$: shell = outer 2 voxels (6.1\% of volume)", fontsize=7)
panel_label(ax, "(a)", x=-0.12)
ax = axs[1]
for rows, c, mk, lab, sv in ((re_, VERM, "s", "N=10$^3$ elliptical (210)", 1 - (1 - 4 / 128) ** 3), (rc, GREEN, "^", "N=10$^3$ circular (210)", 1 - (1 - 4 / 128) ** 3)):
    l_ = np.array([r["lam"] for r in rows]); e_ = np.array([r["shell2_energy_frac"] for r in rows]) / sv
    ax.plot(l_, e_, mk, ms=3, color=c, mec="none", label=lab)
ax.axhline(1, color=GREY, lw=0.6)
ax.set_xlabel(r"$\lambda$ ($\mu$m$^{-2}$)"); ax.set_ylim(0, 8)
ax.set_title(r"$N=10^3$, $128^3$: shell = outer 2 voxels (12.1\% of volume)", fontsize=7)
ax.legend(fontsize=6, loc="upper right")
panel_label(ax, "(b)", x=-0.12)
save(fig, str(FIG / "figS_seam"))
led.add("figS_seam_n1k_shell_volume_fraction", 1 - (1 - 4 / 128) ** 3, "fraction", "geometry")

# ---- bake-off convergence ----------------------------------------------------------
bo = {t: load_json(RES / "exp" / f"bakeoff_{t}.json") for t in ("bandpass_m80_d3300", "hybrid_m80", "hybrid_m80c", "shiftinv_m64b")}
fig, ax = plt.subplots(figsize=(SINGLE, 2.4), gridspec_kw=dict(left=0.15, right=0.98, top=0.95, bottom=0.18))
y = [o["median_res_inwin"] for o in bo["bandpass_m80_d3300"]["outer_stats"]]
ax.semilogy(range(len(y)), y, "o-", color=BLUE, label="bandpass ChebSI, $d=3300$, $m=80$ (build)")
yp = [o["median_res_inwin"] for o in bo["hybrid_m80"]["outer_stats"]["polish"]] + [o["median_res_inwin"] for o in bo["hybrid_m80c"]["outer_stats"]["polish"]]
ax.semilogy(range(len(y) - 1, len(y) - 1 + len(yp)), yp, "s-", color=VERM, label="two-stage: polish $d=8000$ on trimmed basis (56)")
ys = [o["median_res_inwin"] for o in bo["shiftinv_m64b"]["outer_stats"] if o["median_res_inwin"] == o["median_res_inwin"]]
ax.semilogy([1], ys, "D", color=GREEN, label="shift-invert PMINRES SI, $m=64$ (2 outers)")
ax.axhline(1e-3, color=GREY, ls=":", lw=0.8); ax.text(7.9, 1.25e-3, "bake-off tol $10^{-3}$", fontsize=6, ha="right", color=GREY)
ax.axhline(1e-4, color=GREY, ls="--", lw=0.8); ax.text(7.9, 1.25e-4, "production gate $10^{-4}$", fontsize=6, ha="right", color=GREY)
ax.set_xlabel("outer iteration"); ax.set_ylabel("median in-window residual")
ax.set_ylim(3e-5, 1); ax.legend(fontsize=6, loc="upper right")
ax.text(0.02, 0.04, "folded-spectrum LOBPCG: 0/50 after 200 its\n(locked $\\mu\\in[0.096,0.279]$ vs targets $\\leq0.050$)", transform=ax.transAxes, fontsize=6, va="bottom")
save(fig, str(FIG / "figS_bakeoff"))

# ---- residuals + Rayleigh scatter -------------------------------------------------------
W = RES / "n10k_G192_window"
lam = np.load(W / "window_eigenvalues.npy"); lraw = np.load(W / "window_eigenvalues_raw.npy"); nrm = np.load(W / "window_norms.npy"); res = np.load(W / "window_residuals.npy")
man = load_json(W / "vec_manifest.json")
fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 2.3), gridspec_kw=dict(wspace=0.35, left=0.06, right=0.99, top=0.92, bottom=0.2))
ax = axs[0]
for tag, c in (("Sbelow", BLUE), ("Sgap", VERM), ("Sabove", GREEN)):
    r_ = np.load(RES / f"n10k_G192_{tag}" / "window_residuals.npy")
    ax.hist(r_ * 1e5, bins=np.linspace(2, 10, 17), histtype="step", color=c, lw=1.0, label=f"$S_\\mathrm{{{tag[1:]}}}$ ({len(r_)})")
ax.axvline(10, color=GREY, ls="--", lw=0.8); ax.set_xlabel(r"reported relative residual ($\times10^{-5}$)"); ax.set_ylabel("modes")
ax.legend(fontsize=6); panel_label(ax, "(a)", x=-0.2)
ax = axs[1]
ax.plot(1e5 * (nrm - 1), 1e5 * (lraw - lam) / lam, "o", ms=3, color=BLUE, mec="none")
xx = np.linspace(4.2, 6.6, 2); ax.plot(xx, xx, "-", color=GREY, lw=0.8)
ax.set_xlabel(r"$\|x\|^2-1$ ($\times10^{-5}$)"); ax.set_ylabel(r"$(\lambda_\mathrm{raw}-\lambda)/\lambda$ ($\times10^{-5}$)")
ax.set_title("Rayleigh-normalisation correction, 133 modes", fontsize=7); panel_label(ax, "(b)", x=-0.2)
ax = axs[2]
# I1 parity per state, corrected vs raw
ev = np.load(USB / "eigenvalues_all.npy").astype(np.float64)
li = np.load(RES / "i1_n1000_slice" / "window_eigenvalues.npy"); lir = np.load(RES / "i1_n1000_slice" / "window_eigenvalues_raw.npy")
idx = np.array([int(np.argmin(np.abs(ev - l0))) for l0 in li])
ax.semilogy(li, np.abs(ev[idx] - lir) / lir, "s", ms=3, color=GREY, mec="none", label="raw (unnormalised quotient)")
ax.semilogy(li, np.abs(ev[idx] - li) / li, "o", ms=3, color=VERM, mec="none", label=r"corrected $\lambda_\mathrm{raw}/\|x\|^2$")
ax.axhline(1e-4, color=GREY, ls="--", lw=0.8); ax.text(2.15, 1.3e-4, "gate", fontsize=6, ha="right", color=GREY)
ax.set_xlabel(r"$\lambda$ ($\mu$m$^{-2}$)"); ax.set_ylabel(r"$|\Delta\lambda|/\lambda$ vs bottom-up reference")
ax.set_ylim(1e-9, 3e-4); ax.legend(fontsize=6, loc="center right"); ax.set_title("I1: $N=10^3$ interior slice vs exact (55 states)", fontsize=7)
panel_label(ax, "(c)", x=-0.2)
save(fig, str(FIG / "figS_residuals_rayleigh"))

# ---- full DOS ----------------------------------------------------------------------------
z = np.load(SRC / "dos_n10k_256.npz"); z1 = np.load(SRC / "dos_n1k_128.npz")
fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.3), gridspec_kw=dict(wspace=0.25, left=0.07, right=0.99, top=0.92, bottom=0.2))
ax = axs[0]
ax.loglog(z["lam_full"], np.maximum(z["rho_full"], 1e-2), color=BLUE, lw=0.8, label=r"$N=10^4$, $256^3$, $d=12000$, 12 probes")
ax.axvspan(GAP_LO_10K, GAP_HI_10K, color=ORANGE, alpha=0.3, lw=0)
ax.set_xlabel(r"$\lambda$ ($\mu$m$^{-2}$)"); ax.set_ylabel(r"$\rho(\lambda)$ (states per unit $\lambda$)"); ax.legend(fontsize=6, loc="lower right")
ax.set_xlim(0.05, 4e3); panel_label(ax, "(a)", x=-0.14)
ax = axs[1]
ev_c = np.load(RES / "i4_n1000_circ_G128" / "eigenvalues_all.npy")
m = (z1["lam"] > 1.4) & (z1["lam"] < 2.6)
ax.plot(z1["lam"][m], z1["rho"][m], color=VERM, lw=0.8, label=r"$N=10^3$ circ., KPM $128^3$, $d=8000$")
h, e = np.histogram(ev_c, bins=np.arange(1.4, 2.6, 0.02))
ax.step(e[:-1], h / 0.02, where="post", color=BLACK, lw=0.7, label="exact 611 eigenvalues (0.02 bins)")
m = (z["lam"] > 1.4) & (z["lam"] < 2.6)
ax.plot(z["lam"][m], z["rho"][m] / 10, color=BLUE, lw=0.8, label=r"$N=10^4$ KPM $/10$")
h, e = np.histogram(lam, bins=np.arange(1.4, 2.6, 0.02))
ax.step(e[:-1], h / 0.02 / 10, where="post", color=SKY, lw=0.7, label=r"133 certified $/10$")
ax.set_xlabel(r"$\lambda$ ($\mu$m$^{-2}$)"); ax.set_ylabel(r"$\rho$ (states per unit $\lambda$)"); ax.legend(fontsize=6, loc="upper left")
ax.set_xlim(1.4, 2.6); panel_label(ax, "(b)", x=-0.14)
save(fig, str(FIG / "figS_dos_full"))

# ---- level statistics distributions ---------------------------------------------------------
ls = load_json(TAB / "levelstats.json")
grid = np.linspace(0, 1, 400)
pr_p = 2 / (1 + grid) ** 2
pr_g = 27 / 4 * (grid + grid ** 2) / (1 + grid + grid ** 2) ** 2.5
sg = np.linspace(0, 4, 400)
ps_p = np.exp(-sg); ps_g = np.pi / 2 * sg * np.exp(-np.pi * sg ** 2 / 4)
fig, axs = plt.subplots(2, 3, figsize=(DOUBLE, 4.2), gridspec_kw=dict(wspace=0.3, hspace=0.45, left=0.07, right=0.99, top=0.95, bottom=0.1))
panels = [("n1k_ell_all_611", r"$N=10^3$ elliptical, 611 levels"), ("n1k_circ_all_611", r"$N=10^3$ circular, 611 levels"), ("n10k_outside_gap_pooled_r", r"$N=10^4$ outside the bracket, 119 $r$")]
for j, (k, title) in enumerate(panels):
    o = ls[k]
    rv = np.array(o["r_values"]) if "r_values" in o else np.concatenate([ls["n10k_below_edge"]["r_values"], ls["n10k_above_edge"]["r_values"]])
    ax = axs[0, j]
    ax.hist(rv, bins=np.linspace(0, 1, 13), density=True, color=LGREY, edgecolor=GREY, lw=0.5)
    ax.plot(grid, pr_p, ":", color=BLACK, lw=1, label="Poisson"); ax.plot(grid, pr_g, "--", color=VERM, lw=1, label="GOE surmise")
    ax.set_title(f"{title}\n" + rf"$\langle r\rangle={o['r_mean']:.3f}\pm{o['r_se_boot']:.3f}$", fontsize=7)
    ax.set_xlabel("$r$"); ax.set_ylabel("$P(r)$"); ax.set_ylim(0, 2.2)
    if j == 0:
        ax.legend(fontsize=6)
    ax = axs[1, j]
    if "unfolded_spacings" in o:
        su = np.array(o["unfolded_spacings"])
    else:
        su = np.concatenate([ls["n10k_below_edge"]["unfolded_spacings"], ls["n10k_above_edge"]["unfolded_spacings"]])
    ax.hist(su, bins=np.linspace(0, 3.5, 15), density=True, color=LGREY, edgecolor=GREY, lw=0.5)
    ax.plot(sg, ps_p, ":", color=BLACK, lw=1, label="Poisson"); ax.plot(sg, ps_g, "--", color=VERM, lw=1, label="Wigner (GOE)")
    ax.set_xlabel("$s$ (locally unfolded, 9-level window)"); ax.set_ylabel("$P(s)$"); ax.set_xlim(0, 3.5)
    ax.set_title(rf"$\langle s^2\rangle={o.get('unfolded_s2', np.mean(su**2)):.2f}$ (Poisson 2, GOE 1.27)", fontsize=7)
save(fig, str(FIG / "figS_levelstats"))
led.add("figS_levelstats_panels", [p[0] for p in panels], "keys", "report/tables/levelstats.json")

# ---- cross-grid I6 ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.3), gridspec_kw=dict(wspace=0.3, left=0.07, right=0.99, top=0.9, bottom=0.2))
for key, fn, c in (("low edge [1.84, 1.95]", "crossgrid_match.json", BLUE), ("high edge [1.99, 2.035]", "crossgrid_match_n10k_G256_edgehigh_narrow.json", GREEN)):
    cg = load_json(RES / "gates" / fn)
    lc = np.array([p["lam_coarse"] for p in cg["pairs"]]); lf = np.array([p["lam_fine"] for p in cg["pairs"]]); ov = np.array([p["overlap"] for p in cg["pairs"]])
    conv = np.array([p.get("fine_converged", True) for p in cg["pairs"]])
    dww = 100 * (np.sqrt(lf) - np.sqrt(lc)) / np.sqrt(lc)
    ok = ov > 0.5
    axs[0].plot(lc[ok & conv], dww[ok & conv], "o", color=c, ms=4, mec="none", label=f"{key}, certified fine pair")
    axs[0].plot(lc[ok & ~conv], dww[ok & ~conv], "o", mfc="none", mec=c, ms=4.5, mew=0.9, label=f"{key}, uncertified fine vector")
    axs[1].plot(lc[ok], ov[ok], "o", color=c, ms=4, mec="none")
    axs[1].plot(lc[~ok], ov[~ok], "x", color=c, ms=5, label="no partner (filter transition zone)" if (~ok).any() else None)
axs[0].axhline(0, color=GREY, lw=0.5); axs[0].axhspan(-0.6, 0.6, color=LGREY, alpha=0.4, lw=0)
axs[0].set_ylabel(r"$\Delta\omega/\omega$ ($256^3$ vs $192^3$, %)"); axs[0].set_xlabel(r"$\lambda_{192}$ ($\mu$m$^{-2}$)"); axs[0].legend(fontsize=5.5, loc="lower left")
axs[0].set_ylim(-0.7, 0.7); axs[0].text(0.98, 0.95, "registered bound $\\pm0.6\\%$", transform=axs[0].transAxes, ha="right", va="top", fontsize=6, color=GREY)
panel_label(axs[0], "(a)", x=-0.15)
axs[1].set_ylabel("eigenvector overlap (shared $k$-set)"); axs[1].set_xlabel(r"$\lambda_{192}$ ($\mu$m$^{-2}$)"); axs[1].set_ylim(0, 1.05)
axs[1].axvspan(GAP_LO_10K, GAP_HI_10K, color=ORANGE, alpha=0.2, lw=0); axs[1].legend(fontsize=6, loc="lower left"); panel_label(axs[1], "(b)", x=-0.15)
save(fig, str(FIG / "figS_crossgrid"))

# ---- periodic overlap matrix -------------------------------------------------------------------
pm = load_json(RES / "gates" / "periodic_overlap_match.json")
lp = np.load(RES / "n10k_G192_gap_periodic" / "window_eigenvalues.npy")
M = np.array([p["all_overlaps"] for p in pm["pairs"]])
fig, ax = plt.subplots(figsize=(SINGLE, 3.0), gridspec_kw=dict(left=0.22, right=0.97, top=0.97, bottom=0.2))
im = ax.imshow(M, cmap="Greys", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(lp))); ax.set_xticklabels([f"{l:.4f}" for l in lp], rotation=60, fontsize=6)
ax.set_yticks(range(len(pm["pairs"])))
ax.set_yticklabels([f"{p['lam_mont']:.4f}" + (" (seam)" if p["lam_mont"] in seam else "") for p in pm["pairs"]], fontsize=6)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if M[i, j] > 0.05:
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color="white" if M[i, j] > 0.6 else BLACK)
ax.set_xlabel(r"periodic re-solve states $\lambda$ ($\mu$m$^{-2}$)"); ax.set_ylabel(r"montage-convention in-gap states $\lambda$")
plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=r"$|\langle m|p\rangle|$")
save(fig, str(FIG / "figS_periodic"))

# ---- fit-range sensitivity ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(SINGLE, 2.6), gridspec_kw=dict(left=0.15, right=0.98, top=0.95, bottom=0.17))
x = np.array([r["xi_um"] if r["xi_um"] else np.nan for r in r10]); x6 = np.array([r["xi_um_rhi060"] if r["xi_um_rhi060"] else np.nan for r in r10])
un = np.array([r["unresolved"] for r in r10]); pf_ = np.array([r["pr_fraction"] for r in r10])
sc = ax.scatter(x[~un], x6[~un], c=np.log10(100 * pf_[~un]), cmap="viridis", s=12, edgecolors="none")
ax.plot([0, 13], [0, 13], color=GREY, lw=0.6); ax.plot([0, 13], [0, 13 * 0.5], color=GREY, lw=0.5, ls=":"); ax.plot([0, 13], [0, 13 * 1.5], color=GREY, lw=0.5, ls=":")
ax.set_xlabel(r"$\xi$, fit range $[0.10, 0.95]\,L/2$ ($\mu$m)"); ax.set_ylabel(r"$\xi$, fit range $[0.10, 0.60]\,L/2$ ($\mu$m)")
ax.set_xlim(0, 13); ax.set_ylim(0, 13)
plt.colorbar(sc, ax=ax, label=r"$\log_{10}$ participation fraction (%)")
save(fig, str(FIG / "figS_fitrange"))

# ---- G5 omega(G) ---------------------------------------------------------------------------------
g = load_json(RES / "gates" / "gate_results.json")["G5 convergence (64/96/128)"]
fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.2), gridspec_kw=dict(wspace=0.3, left=0.07, right=0.99, top=0.9, bottom=0.2))
for b in g["bands"]:
    w = np.array([b["w64"], b["w96"], b["w128"]])
    axs[0].plot([64, 96, 128], 100 * (w / w[-1] - 1), "o-", ms=3, label=f"band {b['mpb_band']}" + (" (gap edge, non-monotone)" if b["mpb_band"] == 500 else ""))
axs[0].set_xlabel("grid $G$ ($G^3$ voxels)"); axs[0].set_ylabel(r"$\omega(G)/\omega(128)-1$ (%)"); axs[0].legend(fontsize=5.5, ncol=2)
axs[0].set_xticks([64, 96, 128]); panel_label(axs[0], "(a)")
gaps = [2.3545, 1.9286, 2.0755]
axs[1].plot([64, 96, 128], gaps, "s-", color=VERM, ms=4)
axs[1].set_xlabel("grid $G$"); axs[1].set_ylabel(r"gap 500|501, $\Delta\nu/\nu$ (%)"); axs[1].set_xticks([64, 96, 128]); axs[1].set_ylim(1.8, 2.5)
panel_label(axs[1], "(b)")
save(fig, str(FIG / "figS_g5"))

# ---- KPM validation N=1000 -------------------------------------------------------------------------
z1m = np.load(RES / "exp" / "kpm_n1000_G128_prod_kpm.npz", allow_pickle=True)
mom1, lmax1 = z1m["moments"], float(z1m["lam_max"]); p1 = mom1.shape[1] - 1
def jackson(p):
    k = np.arange(p + 1); return (((p - k + 1) * np.cos(np.pi * k / (p + 1)) + np.sin(np.pi * k / (p + 1)) / np.tan(np.pi / (p + 1))) / (p + 1))
def counting(mom, lam_max, lams):
    p = mom.shape[1] - 1; gj = jackson(p); xs = (2.0 * np.asarray(lams) - lam_max) / lam_max
    est = np.empty((mom.shape[0], len(xs)))
    for j, xb in enumerate(xs):
        k = np.arange(1, p + 1); tb = np.arccos(np.clip(xb, -1, 1)); c = np.empty(p + 1); c[0] = 1 - tb / np.pi; c[1:] = -2 * np.sin(k * tb) / (k * np.pi)
        est[:, j] = mom @ (c * gj)
    return est
ts = np.linspace(0.1, 2.45, 120)
e = counting(mom1, lmax1, ts); exact = np.searchsorted(ev, ts)
fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.2), gridspec_kw=dict(wspace=0.3, left=0.07, right=0.99, top=0.9, bottom=0.2))
axs[0].plot(ts, exact, color=BLACK, lw=1, label="exact count (611 bands, 128$^3$)"); axs[0].plot(ts, e.mean(0), "--", color=VERM, lw=1, label="KPM $N(\\lambda)$, $d=8000$, 16 probes")
axs[0].set_xlabel(r"$\lambda$ ($\mu$m$^{-2}$)"); axs[0].set_ylabel(r"$N(\lambda)$"); axs[0].legend(fontsize=6); panel_label(axs[0], "(a)")
axs[1].errorbar(ts, e.mean(0) - exact, yerr=e.std(0, ddof=1) / np.sqrt(mom1.shape[0]), fmt="o", ms=2, color=VERM, elinewidth=0.6)
axs[1].axhline(0, color=GREY, lw=0.5); axs[1].axvspan(ev[497], ev[498], color=ORANGE, alpha=0.3, lw=0)
axs[1].set_xlabel(r"$\lambda$ ($\mu$m$^{-2}$)"); axs[1].set_ylabel("KPM $-$ exact (states)"); panel_label(axs[1], "(b)")
save(fig, str(FIG / "figS_kpm_validation"))

# ---- sliding r N=1000 ---------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(SINGLE, 2.4), gridspec_kw=dict(left=0.15, right=0.98, top=0.95, bottom=0.18))
for k, c, lab in (("sliding_n1k_ell_w41", VERM, r"$N=10^3$ elliptical"), ("sliding_n1k_circ_w41", GREEN, r"$N=10^3$ circular")):
    rows = ls[k]["rows"]; lc = np.array([r["lam_centre"] for r in rows]); rm = np.array([r["r_mean"] for r in rows]); rs = np.array([r["r_se"] for r in rows])
    ax.fill_between(lc, rm - rs, rm + rs, color=c, alpha=0.15, lw=0); ax.plot(lc, rm, color=c, lw=0.9, label=lab + " (41-level window)")
ax.axhline(2 * np.log(2) - 1, color=GREY, ls=":", lw=0.8); ax.axhline(0.5307, color=GREY, ls="--", lw=0.8)
ax.axvspan(1.8276, 2.0225, color=GREEN, alpha=0.12, lw=0); ax.axvspan(1.8830, 1.9628, color=VERM, alpha=0.12, lw=0)
ax.set_xlabel(r"$\lambda$ ($\mu$m$^{-2}$)"); ax.set_ylabel(r"$\langle r\rangle$"); ax.set_ylim(0.3, 0.7); ax.legend(fontsize=6, loc="lower left")
save(fig, str(FIG / "figS_sliding_r_n1k"))
led.save()
