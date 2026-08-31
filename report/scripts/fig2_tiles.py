#!/usr/bin/env python
"""Fig. 2: eigenfunction tiles (eps|E|^2, montage convention) -- top row the
N=1000 elliptical production window around the gap (USB tiles, bands
398 / 498 / 499 / 500 | 501 / 502 / 607); bottom row seven N=10^4 states from
extended (window bottom) through edge and in-gap localized states to extended
(window top). Tiles are the project's renders (pyvista volume render of
eps|E|^2, 'hot' colormap clipped at the 99.9th percentile of each tile,
network in translucent grey). Also writes the SM tile figure with the seam
artefacts and the extended in-gap mode."""
from __future__ import annotations

import json

import numpy as np
from PIL import Image

from common import FIG, RES, TAB, USB, Ledger, load_json
from figstyle import DOUBLE, plt, save

led = Ledger(__file__)
loc10 = load_json(TAB / "loc_n10k.json")["rows"]
loce = load_json(TAB / "loc_n1k_ell.json")["rows"]
lam10 = np.load(RES / "n10k_G192_window" / "window_eigenvalues.npy")


def n10k_idx(l0):
    return int(np.argmin(np.abs(lam10 - l0)))


def xi_label(r):
    if r["xi_um"] is None:
        return r"$\xi$: n/a"
    if r["unresolved"]:
        return r"$\xi \geq L/2$" if r["above_ceiling"] else r"$\xi$ unres."
    return rf"$\xi={r['xi_um']:.1f}\,\mu$m"


def crop(im, frac=0.06):
    w, h = im.size
    return im.crop((int(w * frac), int(h * frac), int(w * (1 - frac)), int(h * (1 - frac))))


top_bands = [398, 498, 499, 500, 501, 502, 607]
bot_lams = [1.7570469501878812, 1.850147869329301, 1.868981147045425, 1.8860078588720002, 1.926413256982914, 2.005207555911053, 2.1165695086620007]
bot_idx = [n10k_idx(l) for l in bot_lams]
fig, axs = plt.subplots(2, 7, figsize=(DOUBLE, 2.45))
fig.subplots_adjust(left=0.035, right=0.995, top=0.91, bottom=0.005, wspace=0.03, hspace=0.30)
for j, b in enumerate(top_bands):
    ax = axs[0, j]
    im = crop(Image.open(USB / "tiles" / f"band_{b:04d}.png").convert("RGB"))
    ax.imshow(im, interpolation="lanczos")
    ax.set_axis_off()
    r = loce[b - 398]
    tag = "gap edge" if b in (500, 501) else ""
    ax.set_title(f"band {b}  $\\lambda$={r['lam']:.3f}\n{xi_label(r)}, $p$={100 * r['pr_fraction']:.1f}%", fontsize=6, pad=1.5)
for j, i in enumerate(bot_idx):
    ax = axs[1, j]
    im = crop(Image.open(RES / "n10k_G192_window" / "tiles" / f"band_{4942 + i:04d}.png").convert("RGB"))
    ax.imshow(im, interpolation="lanczos")
    ax.set_axis_off()
    r = loc10[i]
    ax.set_title(f"$\\lambda$={r['lam']:.4f}\n{xi_label(r)}, $p$={100 * r['pr_fraction']:.2f}%", fontsize=6, pad=1.5)
axs[0, 0].text(-0.03, 0.5, "(a) " r"$N=10^3$" "\n" r"$L=11.4\,\mu$m", transform=axs[0, 0].transAxes, rotation=90, va="center", ha="right", fontsize=7)
axs[1, 0].text(-0.03, 0.5, "(b) " r"$N=10^4$" "\n" r"$L=24.6\,\mu$m", transform=axs[1, 0].transAxes, rotation=90, va="center", ha="right", fontsize=7)
# mark gap position
save(fig, str(FIG / "fig2_tiles"))
led.add("fig2_top_bands", top_bands, "MPB band", str(USB / "tiles"))
led.add("fig2_bottom_lams", [float(lam10[i]) for i in bot_idx], "um^-2", "results/n10k_G192_window/tiles")
led.add("fig2_bottom_tiles", [f"band_{4942 + i:04d}.png" for i in bot_idx], "file", "results/n10k_G192_window/tiles")

# ---- SM: all ten in-gap states + the seam-free re-solve counterparts are not rendered (no tiles); show the ten montage tiles
ing = [i for i in range(133) if 1.864 < lam10[i] < 1.996]
fig, axs = plt.subplots(2, 5, figsize=(DOUBLE, 3.0))
fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.01, wspace=0.03, hspace=0.32)
seam = {1.8707585792024861, 1.8730821374768227, 1.929596209673228, 1.947209447202747}
for ax, i in zip(axs.ravel(), ing):
    im = crop(Image.open(RES / "n10k_G192_window" / "tiles" / f"band_{4942 + i:04d}.png").convert("RGB"))
    ax.imshow(im, interpolation="lanczos"); ax.set_axis_off()
    r = loc10[i]
    cls = "seam" if lam10[i] in seam else ("extended" if abs(lam10[i] - 1.944051593936228) < 1e-9 else "candidate")
    ax.set_title(f"$\\lambda$={r['lam']:.4f}  [{cls}]\n{xi_label(r)}, $p$={100 * r['pr_fraction']:.2f}%, $f_2$={100 * r['shell2_energy_frac']:.0f}%", fontsize=6, pad=1.5)
save(fig, str(FIG / "figS_ingap_tiles"))
led.save()
