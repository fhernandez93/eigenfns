#!/usr/bin/env python
"""Assemble the band montage from a run_modes.py output directory.

    conda run -n lsu_ml python scripts/make_montage.py results/<tag> \
        [--band-lo 398] [--band-hi 607] [--per-row 15] [--out montage.png] \
        [--labels]

Renders each band's ε|E|² over the structure (montage style: grey network +
orange/red field, white background) and stacks tiles 15 per row, band-major.
Band indices are MPB-numbered (see run_modes.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir")
    ap.add_argument("--band-lo", type=int, default=398)
    ap.add_argument("--band-hi", type=int, default=607)
    ap.add_argument("--per-row", type=int, default=15)
    ap.add_argument("--out", default=None)
    ap.add_argument("--labels", action="store_true",
                    help="draw the band index on each tile")
    ap.add_argument("--render-grid", type=int, default=None,
                    help="re-rasterize the structure at this resolution for the "
                         "wireframe (default: the solve grid)")
    ap.add_argument("--band-offset", type=int, default=None,
                    help="interior runs: MPB band number of stored mode 0 "
                         "(from gate I2)")
    args = ap.parse_args()

    rundir = Path(args.rundir)
    if (rundir / "solve_meta.json").exists():
        meta = json.loads((rundir / "solve_meta.json").read_text())
        win_lo = meta.get("band_lo", 398)
    else:  # run_interior.py layout: bands = offset + index (offset from I2)
        meta = json.loads((rundir / "interior_report.json").read_text())
        if args.band_offset is None:
            raise SystemExit("interior run: pass --band-offset (MPB band of "
                             "the first stored mode, certified by gate I2)")
        win_lo = args.band_offset
    ed = np.load(rundir / "window_energy_density.npy", mmap_mode="r")
    from eigenfns.render import assemble_montage, render_tile
    from eigenfns.structure import load_rods, rasterize_penlike

    rods, N, L = load_rods(meta["structure"])
    Gr = args.render_grid or meta["grid"]
    # wireframe must use the run's OWN decoration (recorded in meta), not the
    # rasterizer defaults (old production decoration)
    eps_r = rasterize_penlike(rods, Gr, L,
                              minor_radius=meta.get("radius", 0.2252),
                              aspect_ratio=meta.get("aspect", 2.5),
                              eps_rod=meta.get("eps_rod", 2.9275**2))

    lo, hi = args.band_lo, args.band_hi
    tiles = []
    tiledir = rundir / "tiles"
    tiledir.mkdir(exist_ok=True)
    for band in range(lo, hi + 1):
        idx = band - win_lo
        if idx < 0 or idx >= ed.shape[0]:
            raise SystemExit(f"band {band} not in stored window")
        f = np.asarray(ed[idx])
        if f.shape[0] != Gr:
            reps = Gr // f.shape[0]
            f = np.kron(f, np.ones((reps, reps, reps), np.float32))
        png = tiledir / f"band_{band:04d}.png"
        if not png.exists():
            render_tile(eps_r, f, png)
        if args.labels:
            from PIL import Image, ImageDraw
            im = Image.open(png)
            ImageDraw.Draw(im).text((10, 10), str(band), fill="black")
            im.save(png)
        tiles.append(png)
        if band % 25 == 0:
            print(f"  rendered through band {band}", flush=True)

    out = args.out or str(rundir / f"band_montage_{lo}_{hi}_{args.per_row}_non_ideal_regen.png")
    assemble_montage(tiles, out, per_row=args.per_row)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
