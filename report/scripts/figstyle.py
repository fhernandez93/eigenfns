"""Shared matplotlib style for the PRL figures: 8.6 cm single / 17.8 cm double
column, >= 8 pt fonts, Okabe-Ito colour-blind-safe palette, greyscale-safe
marker/linestyle redundancy."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CM = 1 / 2.54
SINGLE = 8.6 * CM
DOUBLE = 17.8 * CM

# Okabe-Ito
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERM = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
YELLOW = "#F0E442"
BLACK = "#000000"
GREY = "#7f7f7f"
LGREY = "#d9d9d9"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"], "mathtext.fontset": "dejavusans",
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "lines.linewidth": 1.0, "lines.markersize": 3.5, "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.unicode_minus": False,
})


def save(fig, path_noext):
    """Save PDF (vector where possible) and the PNG it was made from."""
    fig.savefig(f"{path_noext}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{path_noext}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    print("wrote", path_noext + ".pdf/.png")


def panel_label(ax, s, x=-0.18, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")
