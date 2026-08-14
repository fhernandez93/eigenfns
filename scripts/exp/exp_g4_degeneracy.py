"""G4 — degeneracy handling: subspace principal angles vs MPB H-fields (64^3).

MPB exported h-fields for bands 1..30 (results/gates/mpb64_fields-h.k01.b*.h5).
We re-solve the lowest 28 modes on MPB's effective grid, build real-space H,
and compare *clusters* (groups with relative spacing < 1e-3) as subspaces.

    JAX_PLATFORMS=cpu conda run -n lsu_ml python scripts/exp/exp_g4_degeneracy.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GATES = ROOT / "results" / "gates"


def load_mpb_h(band: int) -> np.ndarray:
    """Load MPB h-field for a band -> (3, G, G, G) complex."""
    import h5py
    f = h5py.File(GATES / f"mpb64_fields-h.k01.b{band:02d}.h5")
    out = []
    for c in ("x", "y", "z"):
        out.append(np.asarray(f[f"{c}.r"]) + 1j * np.asarray(f[f"{c}.i"]))
    return np.stack(out)


def main():
    import h5py
    import jax.numpy as jnp
    from eigenfns.operator import MaxwellOperator
    from eigenfns.solver import lobpcg_blocks

    eps = np.asarray(h5py.File(GATES / "mpb64-epsilon.h5")["data"], np.float32)
    op = MaxwellOperator(eps, 11.44)
    nev = 28
    vals, vecs, _ = lobpcg_blocks(op, nev, m=20, guard=8, tol=1e-6, maxit=500,
                                  verbose=False)
    H_ours = np.asarray(op.h_realspace(jnp.asarray(vecs)))  # (nev,3,G,G,G)

    # clusters by relative spacing (< 1e-3)
    w = np.sqrt(vals)
    clusters, cur = [], [0]
    for i in range(1, nev):
        if (w[i] - w[i - 1]) / w[i] < 1e-3:
            cur.append(i)
        else:
            clusters.append(cur); cur = [i]
    clusters.append(cur)

    results = []
    for cl in clusters:
        if len(cl) < 2:
            continue
        A = np.stack([H_ours[i].ravel() for i in cl])              # ours
        B = np.stack([load_mpb_h(i + 3).ravel() for i in cl])      # MPB (+2 zero modes, 1-based)
        # orthonormalize each span, principal angles via SVD of overlap
        Qa, _ = np.linalg.qr(A.T)
        Qb, _ = np.linalg.qr(B.T)
        s = np.linalg.svd(Qa.conj().T @ Qb, compute_uv=False)
        results.append({"bands_solver": [int(i) + 1 for i in cl],
                        "bands_mpb": [int(i) + 3 for i in cl],
                        "min_cos_principal_angle": float(s.min()),
                        "subspace_overlap_ok": bool(s.min() >= 0.99)})
        print(results[-1], flush=True)

    ok = all(r["subspace_overlap_ok"] for r in results)
    rec = {"gate": "G4 degeneracy subspaces (64^3, lowest 28)",
           "n_clusters": len(results), "clusters": results, "pass": ok}
    out = GATES / "gate_results.json"
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing[rec["gate"]] = rec
    out.write_text(json.dumps(existing, indent=1))
    print(f"[{'PASS' if ok else 'FAIL'}] G4 over {len(results)} clusters")


if __name__ == "__main__":
    main()
