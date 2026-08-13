"""Prototype the montage tile renderer: grey semi-transparent network volume +
orange/red field energy volume, perspective camera, white background — to match
band_montage_398_607_15_non_ideal.png's style.

Renders the N=1000 structure at 128^3 with a quickly-computed low mode's
eps|E|^2 (CPU, small solve at 48^3 upsampled) just to derisk the pipeline.

    JAX_PLATFORMS=cpu conda run --no-capture-output -n lsu_ml python scripts/exp/exp_render_style.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

RES = Path(__file__).resolve().parents[2] / "results" / "exp"
RES.mkdir(parents=True, exist_ok=True)


def render_tile(eps, field, out_png, img_px=350):
    import pyvista as pv
    pv.OFF_SCREEN = True
    G = eps.shape[0]
    grid = pv.ImageData(dimensions=(G, G, G), spacing=(1, 1, 1))
    pl = pv.Plotter(off_screen=True, window_size=(img_px * 2, img_px * 2))
    pl.set_background("white")

    struct = (eps > 1.0).astype(np.float32)
    grid_s = grid.copy()
    grid_s["s"] = struct.ravel(order="F")
    pl.add_volume(grid_s, scalars="s", cmap="Greys", opacity=[0, 0.05],
                  clim=[0, 4.0], show_scalar_bar=False, shade=False)

    if field is not None:
        f = field / field.max()
        grid_f = grid.copy()
        grid_f["f"] = f.ravel(order="F")
        pl.add_volume(grid_f, scalars="f", cmap="hot_r",
                      opacity=[0.0, 0.0, 0.35, 0.8, 1.0],
                      clim=[0, 0.5], show_scalar_bar=False, shade=False)

    pl.camera_position = "iso"
    pl.camera.azimuth = 15
    pl.camera.elevation = 8
    pl.camera.zoom(1.25)
    pl.screenshot(str(out_png), scale=1)
    pl.close()
    print("wrote", out_png)


def main():
    import jax.numpy as jnp
    from eigenfns.operator import MaxwellOperator
    from eigenfns.solver import lobpcg_blocks
    from eigenfns.structure import load_rods, rasterize_penlike

    rods, N, L = load_rods(
        "/home/francisco/Documents/Create LSU Structures  - Claude/"
        "Example/N1000_lsu_example_ends.txt")
    G = 48
    eps = rasterize_penlike(rods, G, L)
    op = MaxwellOperator(eps, L)
    vals, vecs, _ = lobpcg_blocks(op, 8, m=24, guard=16, tol=1e-4, maxit=200,
                                  verbose=False)
    E = np.asarray(op.e_realspace(vecs[:8], jnp.asarray(vals[:8])))
    ee = (eps[None] * (np.abs(E) ** 2).sum(1)).astype(np.float32)  # eps|E|^2
    eps_hi = rasterize_penlike(rods, 128, L)
    # render band 1 (extended low mode) with the 48^3 field upsampled to 128^3
    f = np.kron(ee[0], np.ones((1, 1, 1)))  # keep 48^3; render on its own grid
    render_tile(eps, ee[0], RES / "style_tile_band1.png")
    render_tile(eps_hi, None, RES / "style_structure_128.png")


if __name__ == "__main__":
    main()
