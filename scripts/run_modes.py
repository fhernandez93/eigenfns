#!/usr/bin/env python
"""Compute Maxwell eigenmodes of an LSU rod-network structure.

    conda run --no-capture-output -n lsu_ml python scripts/run_modes.py \
        <ends.txt> --grid 128 --band-lo 398 --band-hi 607 [--tag NAME] \
        [--m 96] [--guard 32] [--tol 1e-4] [--resume]

Bands are MPB-numbered (bands 1-2 at Γ are the ω=0 modes; the solver's nth
mode is band n+2). Computes bands bottom-up through band-hi, checkpointing
each locked block to results/<tag>/; auto-resumes with --resume. Saves
eigenvalues, window eigenvectors (H spectral), and per-band ε|E|² energy
densities for the requested window.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def gpu_is_busy() -> bool:
    """Single-GPU discipline: refuse to start over a foreign heavy job."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return False
    for line in out.splitlines():
        pid_s, mem_s = [x.strip() for x in line.split(",")]
        if int(mem_s) > 2000:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("structure", help="6-column rod endpoints file (N inferred from name)")
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--band-lo", type=int, default=398)
    ap.add_argument("--band-hi", type=int, default=607)
    ap.add_argument("--m", type=int, default=96)
    ap.add_argument("--guard", type=int, default=32)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true", help="skip the busy-GPU check")
    args = ap.parse_args()

    if not args.force and gpu_is_busy():
        print("Another process holds >2 GB GPU memory — one heavy job at a time "
              "(--force to override).", file=sys.stderr)
        return 2

    from eigenfns.io import BlockCheckpointer
    from eigenfns.operator import MaxwellOperator
    from eigenfns.solver import lobpcg_blocks_resumable
    from eigenfns.structure import load_rods, rasterize_penlike

    rods, N, L = load_rods(args.structure)
    tag = args.tag or f"{Path(args.structure).stem}_G{args.grid}"
    outdir = Path(__file__).resolve().parents[1] / "results" / tag
    print(f"structure N={N} L={L:.3f}  grid {args.grid}^3  bands "
          f"{args.band_lo}-{args.band_hi} (MPB numbering)  -> {outdir}", flush=True)

    eps = rasterize_penlike(rods, args.grid, L)
    print(f"ff = {(eps != 1).mean():.4f}", flush=True)
    op = MaxwellOperator(eps, L)

    # MPB numbering: band n (MPB) = solver mode n-2 at Γ; need modes through band_hi-2
    nev = args.band_hi - 2 + args.guard // 2
    ck = BlockCheckpointer(outdir, "solve", meta={
        "structure": str(args.structure), "grid": args.grid, "N": N, "L": L,
        "tol": args.tol, "m": args.m, "guard": args.guard, "nev": nev,
        "band_numbering": "MPB (bands 1-2 are the omega=0 Gamma modes)",
    })
    t0 = time.perf_counter()
    vals, vecs, stats = lobpcg_blocks_resumable(
        op, nev, m=args.m, guard=args.guard, tol=args.tol,
        checkpointer=ck if True else None, resume=args.resume)
    print(f"solve wall {time.perf_counter()-t0:.0f}s, theta applications "
          f"{stats.theta_applications}", flush=True)

    lo_i, hi_i = args.band_lo - 2 - 1, args.band_hi - 2  # 0-based window [lo_i, hi_i)
    np.save(outdir / "eigenvalues_all.npy", vals)
    win_vals = vals[lo_i:hi_i]
    np.save(outdir / "window_eigenvalues.npy", win_vals)

    import jax.numpy as jnp
    Hwin = vecs[lo_i:hi_i]
    np.save(outdir / "window_vecs_spectral.npy", np.asarray(Hwin))
    # per-band eps|E|^2, streamed in chunks
    ed = np.empty((hi_i - lo_i,) + eps.shape, np.float32)
    for s in range(0, hi_i - lo_i, 8):
        E = op.e_realspace(jnp.asarray(Hwin[s:s + 8]), jnp.asarray(win_vals[s:s + 8]))
        ed[s:s + 8] = np.asarray(eps[None] * (np.abs(np.asarray(E)) ** 2).sum(1))
    np.save(outdir / "window_energy_density.npy", ed)
    print(f"saved bands {args.band_lo}-{args.band_hi}: eigenvalues + spectral "
          f"vectors + energy densities in {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
