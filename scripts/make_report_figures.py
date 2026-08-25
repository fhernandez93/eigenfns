#!/usr/bin/env python
"""Report figures for the N=10k interior study (CPU only).

    conda run -n lsu_ml python scripts/make_report_figures.py

Produces in results/figures/:
  fig_dos_spectrum.png  — KPM DOS of the N=10k structure with every converged
                          eigenvalue drawn as a tick; gap region and in-gap
                          states highlighted.
  fig_xi_omega.png      — ξ(ω) for N=10k (L=24.65 µm) and N=1000 (L=11.44 µm),
                          same decoration, each with its own L/2 ceiling;
                          unresolved fits drawn as lower-bound arrows.
  fig_montage_sbs.png   — the two montages side by side (finite-size compare).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "exp"))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
N10K = ROOT / "results" / "n10k_G192_window"
N1K = ROOT / "results" / "i4_n1000_circ_G128"
KPM = ROOT / "results" / "exp" / "n10k_G256_dos_kpm.npz"

GAP_LO, GAP_HI = 1.864, 1.996   # KPM 10%-criterion bracket (fine grid)


def dos_curve(npz, lo, hi, n=4000):
    from exp_kpm_analyze import jackson
    z = np.load(npz, allow_pickle=True)
    mom, lam_max = z["moments"], float(z["lam_max"])
    p = mom.shape[1] - 1
    mu = mom.mean(0)[:p + 1] * jackson(p)
    lam = np.linspace(lo, hi, n)
    x = np.clip(2 * lam / lam_max - 1, -1 + 1e-12, 1 - 1e-12)
    th = np.arccos(x)
    T = np.cos(np.outer(np.arange(p + 1), th))
    rho = (mu[0] + 2 * (mu[1:] @ T[1:])) / (np.pi * np.sin(th)) * 2 / lam_max
    return lam, rho


def fig_dos():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ev = np.load(N10K / "window_eigenvalues.npy")
    lam, rho = dos_curve(KPM, 1.70, 2.18)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.fill_between(lam, 1e-1, np.maximum(rho, 1e-1), color="#cfe3ff",
                    lw=0, label="KPM density of states (256³, degree 12k)")
    ax.plot(lam, np.maximum(rho, 1e-1), color="#1f6feb", lw=1.2)
    ax.axvspan(GAP_LO, GAP_HI, color="#ffd8a8", alpha=0.45, zorder=0,
               label="KPM gap bracket (10% criterion)")
    ymin = 1e-1
    ingap = (ev > GAP_LO) & (ev < GAP_HI)
    ax.vlines(ev[~ingap], ymin, ymin * 6, color="#24292f", lw=0.8,
              label=f"converged eigenvalues ({(~ingap).sum()})")
    ax.vlines(ev[ingap], ymin, ymin * 20, color="#cf222e", lw=1.6,
              label=f"in-gap localized states ({ingap.sum()})")
    ax.set(yscale="log", xlabel=r"$\lambda = (\omega/c)^2\ \ (\mu m^{-2})$",
           ylabel="DOS (states per unit λ)", xlim=(1.70, 2.18),
           title="N=10,000 network: measured spectrum across the gap "
                 "(192³ eigenvalues on the 256³ KPM DOS)")
    ax.legend(loc="upper center", fontsize=8, ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_dos_spectrum.png", dpi=160)
    print("wrote fig_dos_spectrum.png")


def fig_xi():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (path, L, label, gap) in zip(axs, [
            (N10K / "localization_modes.json", 24.6467,
             "N = 10,000   (L = 24.65 µm, 133 modes)", (GAP_LO, GAP_HI)),
            (N1K / "localization_modes.json", 11.44,
             "N = 1,000   (L = 11.44 µm, 210 modes)", (1.82759, 2.02249))]):
        rows = json.load(open(path))
        lam = np.array([r["lam"] for r in rows])
        xi = np.array([r["xi_um"] for r in rows])
        un = np.array([r["unresolved"] for r in rows])
        ceil = L / 2
        ax.axvspan(gap[0], gap[1], color="#ffd8a8", alpha=0.5, zorder=0,
                   label="photonic gap")
        ax.axhline(ceil, ls="--", color="#cf222e", lw=1.4,
                   label=f"finite-size ceiling L/2 = {ceil:.2f} µm")
        ax.scatter(lam[~un], xi[~un], s=16, color="#1f6feb", zorder=3,
                   label=f"resolved ξ ({(~un).sum()})")
        if un.any():
            ax.scatter(lam[un], np.full(un.sum(), ceil), s=26, marker="^",
                       color="#bf8700", zorder=4,
                       label=f"unresolved: lower bound only ({un.sum()})")
        ax.set(xlabel=r"$\lambda\ (\mu m^{-2})$", title=label,
               ylim=(0, 14.5))
        ax.legend(fontsize=8, loc="upper right")
    axs[0].set_ylabel(r"envelope decay length $\xi$ (µm)")
    fig.suptitle("Localization vs finite box size — identical decoration "
                 "(circular rods, n = 2.9, ff = 22%)", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig_xi_omega.png", dpi=160)
    print("wrote fig_xi_omega.png")


def fig_sbs():
    from PIL import Image, ImageDraw
    a = Image.open(N1K / "band_montage_398_607_15_non_ideal_regen.png")
    b = Image.open(N10K / "band_montage_n10k_gapedge_15.png")
    w = 1600
    a = a.resize((w, int(a.height * w / a.width)), Image.LANCZOS)
    b = b.resize((w, int(b.height * w / b.width)), Image.LANCZOS)
    pad, top = 30, 70
    canvas = Image.new("RGB", (w * 2 + pad * 3, max(a.height, b.height) + top + pad),
                       "white")
    canvas.paste(a, (pad, top))
    canvas.paste(b, (w + pad * 2, top))
    d = ImageDraw.Draw(canvas)
    d.text((pad, 20), "N = 1,000  (L = 11.44 µm) — bands 398–607, "
                      "bottom-up solver, same decoration", fill="black")
    d.text((w + pad * 2, 20), "N = 10,000  (L = 24.65 µm) — 133 gap-edge modes, "
                              "interior solver", fill="black")
    d.text((pad, 46), "ξ ceiling L/2 = 5.72 µm: 168/210 modes unresolved",
           fill="#cf222e")
    d.text((w + pad * 2, 46), "ξ ceiling L/2 = 12.32 µm: only 12/133 unresolved",
           fill="#cf222e")
    canvas.save(OUT / "fig_montage_sbs.png")
    print("wrote fig_montage_sbs.png", canvas.size)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_dos()
    fig_xi()
    fig_sbs()
