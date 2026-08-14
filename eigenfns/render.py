"""Montage tile rendering: grey semi-transparent network + orange/red field
energy, perspective volume render on white — the reference montage's style.

Rendered quantity: time-averaged electric energy density ε|E|² (the
Joannopoulos / MPB `output-dpwr` convention for dielectric-band modes).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def render_tile(
    eps: np.ndarray,
    field: np.ndarray | None,
    out_png: str | Path,
    img_px: int = 350,
    structure_opacity: float = 0.08,
    field_clim_q: float = 0.999,
    camera=("iso", 15.0, 8.0, 1.28),
) -> None:
    """Render one montage tile (off-screen; safe to call in a worker process)."""
    import pyvista as pv

    pv.OFF_SCREEN = True
    G = eps.shape[0]
    base = pv.ImageData(dimensions=(G, G, G), spacing=(1, 1, 1))
    pl = pv.Plotter(off_screen=True, window_size=(img_px * 2, img_px * 2))
    pl.set_background("white")

    gs = base.copy()
    gs["s"] = (eps > eps.min()).astype(np.float32).ravel(order="F")
    # low uniform opacity so individual rods read as a translucent wireframe
    pl.add_volume(gs, scalars="s", cmap="Greys",
                  opacity=[0.0, structure_opacity * 255, structure_opacity * 255],
                  clim=[0, 6.0], show_scalar_bar=False, shade=False)

    if field is not None:
        f = np.asarray(field, np.float32)
        hi = np.quantile(f, field_clim_q)
        f = np.clip(f / max(hi, 1e-30), 0, 1)
        gf = base.copy()
        gf["f"] = f.ravel(order="F")
        # reference look: mid = red-orange, high = bright yellow core ("hot"
        # unreversed); opacity emphasizes upper-mid intensities
        pl.add_volume(gf, scalars="f", cmap="hot",
                      opacity=[0.0, 0.05 * 255, 0.35 * 255, 0.7 * 255, 0.9 * 255],
                      clim=[0, 1.15], show_scalar_bar=False, shade=False)

    pos, azi, ele, zoom = camera
    pl.camera_position = pos
    pl.camera.azimuth = azi
    pl.camera.elevation = ele
    pl.camera.zoom(zoom)
    pl.screenshot(str(out_png), scale=1)
    pl.close()


def assemble_montage(
    tile_pngs: list[str | Path],
    out_png: str | Path,
    per_row: int = 15,
    tile_px: tuple[int, int] = (350, 364),
) -> None:
    """Stack tiles into the reference layout (15 per row, row-major)."""
    from PIL import Image

    tw, th = tile_px
    n = len(tile_pngs)
    rows = (n + per_row - 1) // per_row
    canvas = Image.new("RGB", (per_row * tw, rows * th), "white")
    for i, p in enumerate(tile_pngs):
        im = Image.open(p).convert("RGB").resize((tw, th))
        canvas.paste(im, ((i % per_row) * tw, (i // per_row) * th))
    canvas.save(out_png)
