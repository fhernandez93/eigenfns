"""E4 — precision policy: c64 vs c128 eigenvalues on a 48^3 disordered case.

Runs the SAME solver twice on the same grid (CPU): complex128 reference at tight
tolerance, then complex64 at production tolerance. Reports per-band dω/ω.

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu conda run --no-capture-output -n lsu_ml \
        python scripts/exp/exp_precision48.py [nev]
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

RES = Path(__file__).resolve().parents[2] / "results" / "exp"
RES.mkdir(parents=True, exist_ok=True)


def main():
    nev = int(sys.argv[1]) if len(sys.argv) > 1 else 96
    import jax.numpy as jnp
    from eigenfns.operator import MaxwellOperator
    from eigenfns.solver import lobpcg_blocks
    from eigenfns.structure import load_rods, rasterize_penlike

    rods, N, L = load_rods(
        "/home/francisco/Documents/Create LSU Structures  - Claude/"
        "Example/N1000_lsu_example_ends.txt")
    eps = rasterize_penlike(rods, 48, L)
    print(f"48^3, ff={(eps != 1).mean():.4f}", flush=True)

    op64 = MaxwellOperator(eps, L, dtype=jnp.complex128)
    print("== c128 reference (tol 1e-6) ==", flush=True)
    vals64, _, st64 = lobpcg_blocks(op64, nev, m=96, guard=48, tol=1e-6, maxit=300)
    np.save(RES / "prec48_c128.npy", vals64)

    op32 = MaxwellOperator(eps, L, dtype=jnp.complex64)
    print("== c64 (tol 1e-4) ==", flush=True)
    vals32, _, st32 = lobpcg_blocks(op32, nev, m=96, guard=48, tol=1e-4, maxit=300)
    np.save(RES / "prec48_c64.npy", vals32)

    w64, w32 = np.sqrt(vals64[:nev]), np.sqrt(vals32[:nev])
    rel = np.abs(w32 - w64) / w64
    print(f"\nE4 result over {nev} bands: max dω/ω = {rel.max():.3e}, "
          f"median = {np.median(rel):.3e}, q90 = {np.quantile(rel, 0.9):.3e}")
    print("PASS Δω/ω ≤ 1e-3 everywhere:" , bool((rel <= 1e-3).all()))


if __name__ == "__main__":
    main()
