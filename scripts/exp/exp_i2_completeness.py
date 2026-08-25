#!/usr/bin/env python
"""Gate I2: deflated-probe KPM completeness audit of an interior window.

Counts eigenvalues in [lam_lo, lam_hi] AFTER deflating the probes against the
converged window vectors: expected ~0 (plus the known small Jackson-leakage
bias, G6 precedent 0.2). Any count >= 1 means missed state(s).

    conda run -n lsu_ml python scripts/exp/exp_i2_completeness.py \
        --rundir results/n10k_G192_Sbelow --degree 12000 --probes 8 \
        --gate-name "I2 completeness (S_below)"

Uses the run's own structure/decoration/grid from interior_report.json.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", required=True)
    ap.add_argument("--degree", type=int, default=12000)
    ap.add_argument("--probes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--gate-name", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from run_modes import gpu_is_busy
    if not args.force and gpu_is_busy():
        print("GPU busy.", file=sys.stderr)
        return 2

    import jax.numpy as jnp
    from eigenfns.operator import MaxwellOperator
    from eigenfns.structure import load_rods, rasterize_penlike
    from eigenfns.chebyshev import kpm_count_below

    rundir = Path(args.rundir)
    meta = json.loads((rundir / "interior_report.json").read_text())
    lam_lo, lam_hi = meta["window"]
    rods, N, L = load_rods(meta["structure"])
    eps = rasterize_penlike(rods, meta["grid"], L, minor_radius=meta["radius"],
                            aspect_ratio=meta["aspect"], eps_rod=meta["eps_rod"])
    op = MaxwellOperator(eps, L)
    lam_max = meta["lam_max"]
    locked = jnp.asarray(np.load(rundir / "window_vecs_spectral.npy"))
    print(f"deflating {locked.shape[0]} converged vectors; window "
          f"[{lam_lo}, {lam_hi}] degree {args.degree} probes {args.probes}",
          flush=True)
    t0 = time.perf_counter()
    hi_m, hi_se = kpm_count_below(op, lam_hi, lam_max, degree=args.degree,
                                  n_probe=args.probes, seed=args.seed,
                                  locked=locked)
    lo_m, lo_se = kpm_count_below(op, lam_lo, lam_max, degree=args.degree,
                                  n_probe=args.probes, seed=args.seed,
                                  locked=locked)
    cnt = hi_m - lo_m
    se = float(np.hypot(hi_se, lo_se))  # conservative (probes shared -> correlated)
    out = {
        "gate": args.gate_name, "when": time.strftime("%Y-%m-%d %H:%M"),
        "rundir": str(rundir), "window": [lam_lo, lam_hi],
        "n_deflated": int(locked.shape[0]),
        "degree": args.degree, "probes": args.probes,
        "missed_count": float(cnt), "se": se,
        "wall_seconds": time.perf_counter() - t0,
        "pass": bool(cnt < 0.5),
        "note": "expected ~0.2 Jackson-leakage bias (G6 precedent); "
                ">= 1 means missed state(s)",
    }
    gates = Path(__file__).resolve().parents[2] / "results" / "gates" / "gate_results.json"
    data = json.loads(gates.read_text()) if gates.exists() else {}
    data[args.gate_name] = out
    gates.write_text(json.dumps(data, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
