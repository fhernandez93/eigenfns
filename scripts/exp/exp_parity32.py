"""E-parity-32: our solver vs MPB on MPB's own effective 32^3 grid (disordered N=1000).

Protocol (see docs/plans/2026-08-12 log): rasterize binary eps -> h5 -> MPB CLI run
(file input => scalar eps, no tensor smoothing) -> read back <tag>-epsilon.h5:data
(the exact grid MPB used) -> run our solver on that grid -> compare eigenvalues.

Usage: conda run --no-capture-output -n lsu_ml python scripts/exp/exp_parity32.py [nev]
(MPB must have been run first via exp_parity32_mpb.sh; both operate in results/exp/)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

RES = Path(__file__).resolve().parents[2] / "results" / "exp"


def main():
    nev = int(sys.argv[1]) if len(sys.argv) > 1 else 160
    import h5py
    from eigenfns.operator import MaxwellOperator
    from eigenfns.solver import lobpcg_blocks

    eps = np.asarray(h5py.File(RES / "mpb32-epsilon.h5")["data"]).astype(np.float32)
    print(f"MPB effective grid: {eps.shape}, eps in [{eps.min():.3f},{eps.max():.3f}]")
    op = MaxwellOperator(eps, 11.44)
    vals, vecs, stats = lobpcg_blocks(op, nev, m=96, guard=48, tol=1e-4,
                                      maxit=400, log_every=50)
    np.save(RES / "parity32_ours.npy", vals)

    mpb_lines = [l for l in open(RES / "mpb32.out") if l.startswith("freqs:, 1,")]
    nu_mpb = np.array([float(x) for x in mpb_lines[0].split(",")[6:]])
    lam_mpb = (2 * np.pi * nu_mpb / 11.44) ** 2
    nb = min(nev, len(lam_mpb) - 2)
    rel = np.abs(np.sqrt(vals[:nb]) - np.sqrt(lam_mpb[2:2 + nb])) / np.sqrt(lam_mpb[2:2 + nb])
    print(f"\nparity vs MPB over {nb} bands: max dw/w {rel.max():.2e}  "
          f"median {np.median(rel):.2e}")
    worst = np.argsort(rel)[-5:]
    print("worst bands (1-based):", worst + 1, np.round(rel[worst], 5))


if __name__ == "__main__":
    main()
