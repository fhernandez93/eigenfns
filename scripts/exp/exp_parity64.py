"""G3 disordered parity: our solver vs MPB, 64^3 N=1000 gold structure, 300 bands.

Protocol: our solver consumes MPB's exported effective grid (results/gates/
mpb64-epsilon.h5:data) so both solve the identical discrete problem.

    conda run --no-capture-output -n lsu_ml python scripts/exp/exp_parity64.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    import h5py
    from eigenfns.operator import MaxwellOperator
    from eigenfns.solver import lobpcg_blocks

    eps = np.asarray(h5py.File(ROOT / "results/gates/mpb64-epsilon.h5")["data"],
                     np.float32)
    print(f"grid {eps.shape}, eps [{eps.min():.3f},{eps.max():.3f}]", flush=True)
    op = MaxwellOperator(eps, 11.44)
    nev = 300
    vals, vecs, st = lobpcg_blocks(op, nev, m=40, guard=16, tol=1e-4, maxit=300,
                                   locked_storage="host", verbose=True)
    np.save(ROOT / "results/gates/parity64_ours.npy", vals)

    line = [l for l in open(ROOT / "results/gates/mpb64.out")
            if l.startswith("freqs:, 1,")][0]
    nu_mpb = np.array([float(x) for x in line.split(",")[6:]])
    lam_mpb = (2 * np.pi * nu_mpb / 11.44) ** 2
    nb = min(nev, len(lam_mpb) - 2)
    ours, mpb = vals[:nb], lam_mpb[2:2 + nb]
    rel = np.abs(np.sqrt(ours) - np.sqrt(mpb)) / np.sqrt(mpb)
    print(f"\nG3 parity over {nb} bands: max dw/w {rel.max():.3e}  "
          f"median {np.median(rel):.3e}  q99 {np.quantile(rel, 0.99):.3e}")
    print("PASS dw/w <= 1e-4 per band:", bool((rel <= 1e-4).all()))
    worst = np.argsort(rel)[-5:]
    for i in worst[::-1]:
        print(f"  band {i+1} (MPB {i+3}): ours {np.sqrt(ours[i]):.6f} "
              f"mpb {np.sqrt(mpb[i]):.6f} rel {rel[i]:.2e}")


if __name__ == "__main__":
    main()
