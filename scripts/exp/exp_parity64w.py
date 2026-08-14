"""G3w full-window parity, our side: 660 bands on MPB's 64^3 effective grid.

    JAX_PLATFORMS=cpu conda run -n lsu_ml python scripts/exp/exp_parity64w.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    import h5py
    from eigenfns.operator import MaxwellOperator
    from eigenfns.solver import lobpcg_blocks

    eps = np.asarray(h5py.File(ROOT / "results/gates/mpb64-epsilon.h5")["data"],
                     np.float32)
    op = MaxwellOperator(eps, 11.44)
    vals, vecs, st = lobpcg_blocks(op, 660, m=48, guard=16, tol=1e-4, maxit=300,
                                   locked_storage="host", verbose=True)
    np.save(ROOT / "results/gates/parity64w_ours.npy", vals)
    print(f"done: 660 bands, wall {st.wall_seconds:.0f}s", flush=True)


if __name__ == "__main__":
    main()
