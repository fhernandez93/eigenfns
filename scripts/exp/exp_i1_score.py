#!/usr/bin/env python
"""Score gates I1/I4: a run_interior.py output vs a bottom-up reference run.

    conda run -n lsu_ml python scripts/exp/exp_i1_score.py \
        --interior results/i1_n1000_slice --reference results/prod_N1000_G128 \
        --ref-lo 395 --slice-lo 473 --slice-hi 523 --gate-name I1

reference layout: eigenvalues_all.npy + window_vecs_spectral.npy (window
starts at 0-based index --ref-lo). Match: |Δλ|/λ < 5e-4 to nearest reference
AND projection² onto the reference cluster subspace ≥ 0.99 (cluster = refs
within rel 1e-3). Ghost: converged interior pair in the slice λ-range
matching no reference. Appends to results/gates/gate_results.json.
CPU is enough (projections stream) but uses GPU if free.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("XLA_FLAGS",
                      "--xla_gpu_enable_cublaslt=false --xla_gpu_autotune_level=0")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

MATCH_RTOL = 5e-4
CLUSTER_TOL = 1e-3
PROJ_TOL = 0.99
DLAM_GATE = 1e-4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interior", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--ref-lo", type=int, required=True,
                    help="0-based solver index of reference window vector 0")
    ap.add_argument("--slice-lo", type=int, required=True)
    ap.add_argument("--slice-hi", type=int, required=True, help="exclusive")
    ap.add_argument("--gate-name", required=True)
    args = ap.parse_args()

    import jax.numpy as jnp
    from eigenfns.solver import gram

    idir, rdir = Path(args.interior), Path(args.reference)
    lam = np.load(idir / "window_eigenvalues.npy")
    res = np.load(idir / "window_residuals.npy")
    X = np.load(idir / "window_vecs_spectral.npy", mmap_mode="r")
    ref_vals = np.load(rdir / "eigenvalues_all.npy")
    ref_vecs = np.load(rdir / "window_vecs_spectral.npy", mmap_mode="r")
    tgt_idx = np.arange(args.slice_lo, args.slice_hi)
    tgt_set = set(int(t) for t in tgt_idx)

    matched, ghosts, proj_vals = {}, [], {}
    for i in range(len(lam)):
        j = int(np.argmin(np.abs(ref_vals - lam[i])))
        if abs(ref_vals[j] - lam[i]) / ref_vals[j] < MATCH_RTOL:
            matched.setdefault(j, []).append(i)
        else:
            ghosts.append((int(i), float(lam[i])))
    for j, iis in sorted(matched.items()):
        cl = np.where(np.abs(ref_vals - ref_vals[j]) / ref_vals[j] < CLUSTER_TOL)[0]
        cl_win = cl[(cl >= args.ref_lo) & (cl < args.ref_lo + ref_vecs.shape[0])]
        if len(cl_win) == 0:
            continue
        Vref = jnp.asarray(np.asarray(ref_vecs[cl_win - args.ref_lo]))
        for i in iis:
            C = np.asarray(gram(Vref, jnp.asarray(np.asarray(X[i:i + 1]))))
            proj_vals[(j, i)] = float((np.abs(C) ** 2).sum())
        del Vref

    found = sorted(j for j in (set(matched) & tgt_set)
                   if all(proj_vals.get((j, i), 1.0) >= PROJ_TOL
                          for i in matched[j]))
    missed = sorted(tgt_set - set(found))
    dl = [abs(lam[matched[j][0]] - ref_vals[j]) / ref_vals[j] for j in found]
    out = {
        "gate": args.gate_name, "when": time.strftime("%Y-%m-%d %H:%M"),
        "interior": str(idir), "reference": str(rdir),
        "slice": [args.slice_lo, args.slice_hi],
        "n_reported": int(len(lam)), "worst_res": float(res.max()),
        "targets_found": len(found), "targets_missed": len(missed),
        "missed_idx": missed, "ghosts": ghosts,
        "max_dlam_rel": float(max(dl)) if dl else None,
        "median_dlam_rel": float(np.median(dl)) if dl else None,
        "min_proj2": float(min(proj_vals.values())) if proj_vals else None,
        "pass": (len(missed) == 0 and not ghosts and dl
                 and max(dl) <= DLAM_GATE
                 and (min(proj_vals.values()) >= PROJ_TOL)),
    }
    gates = Path(__file__).resolve().parents[2] / "results" / "gates" / "gate_results.json"
    data = json.loads(gates.read_text()) if gates.exists() else {}
    data[args.gate_name] = out
    gates.write_text(json.dumps(data, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
