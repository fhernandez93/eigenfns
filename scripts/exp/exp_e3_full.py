"""E3 — full bottom-up solve of the N=1000 gold structure (binary montage-convention
eps) at a given grid size: spectrum layout around bands 398-607 + measured cost.

    conda run --no-capture-output -n lsu_ml python scripts/exp/exp_e3_full.py [G] [nev] [m] [guard]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

RES = Path(__file__).resolve().parents[2] / "results" / "exp"
RES.mkdir(parents=True, exist_ok=True)


def main():
    G = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    nev = int(sys.argv[2]) if len(sys.argv) > 2 else 680
    m = int(sys.argv[3]) if len(sys.argv) > 3 else 96
    guard = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    from eigenfns.operator import MaxwellOperator
    from eigenfns.solver import lobpcg_blocks
    from eigenfns.structure import load_rods, rasterize_penlike

    rods, N, L = load_rods(
        "/home/francisco/Documents/Create LSU Structures  - Claude/"
        "Example/N1000_lsu_example_ends.txt")
    eps = rasterize_penlike(rods, G, L)
    print(f"G={G} ff={(eps != 1).mean():.4f} nev={nev} m={m} guard={guard}", flush=True)
    op = MaxwellOperator(eps, L)
    vals, vecs, stats = lobpcg_blocks(op, nev, m=m, guard=guard, tol=1e-4, maxit=300)
    np.save(RES / f"e3_vals_G{G}.npy", vals)
    print(f"\nwall {stats.wall_seconds:.1f}s  theta applications {stats.theta_applications}")
    nu = np.sqrt(np.maximum(vals, 0)) * 2.288 / (2 * np.pi)  # a = srs cubic cell
    for i in [0, 199, 397, 449, 499, 500, 549, 606, nev - 1]:
        if i < len(nu):
            print(f"  band {i+1:4d}: lambda {vals[i]:.5f}  nu(a=2.288) {nu[i]:.4f}")
    if len(vals) > 620:
        d = np.diff(vals)
        i = int(np.argmax(d[380:620]) + 380)
        rel = (np.sqrt(vals[i+1]) - np.sqrt(vals[i])) / ((np.sqrt(vals[i+1]) + np.sqrt(vals[i])) / 2)
        print(f"largest spectral gap in bands [381,621]: {i+1}|{i+2}: "
              f"lam {vals[i]:.4f} -> {vals[i+1]:.4f}  (dnu/nu = {rel:.4f})")


if __name__ == "__main__":
    main()
