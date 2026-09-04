#!/usr/bin/env python
"""Gate I2 v2 — completeness audit with an edge-leakage-corrected estimator.

Why v2 (v1 recorded as a failed design, see docs/plans/):
  v1 counted N(λ_hi) − N(λ_lo) with two independent `kpm_count_below` calls.
  Each counts ~5,000 states, so its stochastic error is ~√(2·5000/n_probe) ≈ 26
  bands, and the two calls were combined as if independent — an error bar 50×
  too wide to detect one missing state. It also OOM'd: the locked set (69
  vectors at 192³ = 7.8 GB) was pushed to the GPU instead of streamed.

v2 estimator. With Rademacher probes z (deflated against the converged window
vectors) and ρ(λ) the Jackson-damped bandpass of [lo, hi]:

    E[zᵀ ρ(Θ) z] = Σ_{all states} ρ(λ_i) − Σ_{found} ρ(λ_i)
                 = Σ_{missed in window} ρ(λ_i)  +  L,

where L = Σ_{states outside the window} ρ(λ_i) is the transition-zone leakage
— unavoidable when the window edges sit in the bulk (this is exactly what
produced G6's "86.6 phantom missing bands" before that gate was amended).
v2 handles L in two independent ways:

  (A) predict L from the measured KPM density of states:
      L ≈ ∫_outside ρ(λ)·DOS(λ) dλ, computed from the saved DOS moments;
  (B) evaluate on a sub-interval whose edges sit in the sparse gap region,
      where L is small by construction (the G6 amendment's logic).

Both are reported. The estimator is a SINGLE Chebyshev recurrence per probe
chunk (half of v1's cost) and returns per-probe values, so the quoted error
bar is the paired standard error, not a sum of two large independent ones.

    conda run -n lsu_ml python scripts/exp/exp_i2_v2.py \
        --rundir results/n10k_G192_window --degree 24000 --probes 12 \
        --kpm results/exp/n10k_G256_dos_kpm.npz \
        --sub-lo 1.9063 --sub-hi 1.9820 --gate-name "I2 completeness (N=10k)"
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


def _bandpass_rho(lam, lam_lo, lam_hi, lam_max, degree):
    """Evaluate the Jackson-damped bandpass ρ(λ) used by the estimator."""
    from eigenfns.interior import bandpass_coeffs
    coef = bandpass_coeffs(lam_lo, lam_hi, lam_max, degree)
    x = np.clip(2 * np.asarray(lam) / lam_max - 1, -1, 1)
    th = np.arccos(x)
    k = np.arange(len(coef))
    return np.cos(np.outer(th, k)) @ coef


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", required=True)
    ap.add_argument("--degree", type=int, default=24000)
    ap.add_argument("--probes", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--kpm", default=None, help="DOS npz for the leakage prediction")
    ap.add_argument("--sub-lo", type=float, default=None)
    ap.add_argument("--sub-hi", type=float, default=None)
    ap.add_argument("--gate-name", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from run_modes import gpu_is_busy
    if not args.force and gpu_is_busy():
        print("GPU busy.", file=sys.stderr)
        return 2

    import jax
    import jax.numpy as jnp
    from eigenfns.operator import MaxwellOperator
    from eigenfns.structure import load_rods, rasterize_penlike
    from eigenfns.interior import bandpass_coeffs
    from eigenfns.solver import _flat, deflate, deflate_chunk_rows

    rundir = Path(args.rundir)
    meta = json.loads((rundir / "interior_report.json").read_text())
    lam_found = np.load(rundir / "window_eigenvalues.npy")
    lam_lo, lam_hi = meta["window"]
    G = meta["grid"]
    lam_max = meta["lam_max"]
    rods, N, L = load_rods(meta["structure"])
    eps = rasterize_penlike(rods, G, L, minor_radius=meta["radius"],
                            aspect_ratio=meta["aspect"], eps_rod=meta["eps_rod"])
    op = MaxwellOperator(eps, L)

    # locked set stays on the HOST; `deflate` streams it in fixed chunks
    if (rundir / "window_vecs_spectral.npy").exists():
        locked = np.load(rundir / "window_vecs_spectral.npy", mmap_mode="r")
        locked = np.asarray(locked)
    else:
        man = json.loads((rundir / "vec_manifest.json").read_text())
        locked = np.empty((len(man), 2, G, G, G), np.complex64)
        for i, e in enumerate(man):
            locked[i] = np.load(e["dir"] + "/window_vecs_spectral.npy",
                                mmap_mode="r")[e["index"]]
        print(f"assembled {len(man)} locked vectors from manifest "
              f"({locked.nbytes/2**30:.1f} GiB host)", flush=True)

    intervals = [("window", lam_lo, lam_hi)]
    if args.sub_lo is not None and args.sub_hi is not None:
        intervals.append(("sub_gap", args.sub_lo, args.sub_hi))

    results = {}
    t_all = time.perf_counter()
    for name, a, b in intervals:
        coef = bandpass_coeffs(a, b, lam_max, args.degree)
        cj = jnp.asarray(coef, jnp.float32)
        mask = (op.basis.kn > 0).astype(jnp.float32)[None, None]
        inv_l = 2.0 / lam_max
        cr = deflate_chunk_rows(G)
        est = np.zeros(args.probes)
        t0 = time.perf_counter()

        def Bx(V):
            return inv_l * op.theta(V) - V

        for s in range(0, args.probes, args.chunk):
            c = min(args.chunk, args.probes - s)
            kz = jax.random.fold_in(jax.random.PRNGKey(args.seed), s)
            Z = jnp.where(jax.random.bernoulli(kz, 0.5, (c, 2, G, G, G)),
                          1.0, -1.0).astype(jnp.complex64) * mask
            Z = deflate(Z, locked, cr)
            Z = deflate(Z, locked, cr)
            zf = _flat(Z)
            T0, T1 = Z, Bx(Z)
            e = (cj[0] * jnp.real(jnp.sum(zf.conj() * _flat(T0), axis=1))
                 + cj[1] * jnp.real(jnp.sum(zf.conj() * _flat(T1), axis=1)))
            for k in range(2, args.degree + 1):
                T2 = 2.0 * Bx(T1) - T0
                e = e + cj[k] * jnp.real(jnp.sum(zf.conj() * _flat(T2), axis=1))
                T0, T1 = T1, T2
                if k % 2000 == 0:
                    e.block_until_ready()
                    print(f"    [{name}] probes {s}..{s+c-1} k={k}/{args.degree} "
                          f"{time.perf_counter()-t0:.0f}s", flush=True)
            est[s:s + c] = np.asarray(e)
            del Z, T0, T1, T2, e
        mean = float(est.mean())
        se = float(est.std(ddof=1) / np.sqrt(args.probes))

        # --- leakage prediction from the measured DOS ---------------------
        leak = leak_err = None
        if args.kpm:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp"))
            from exp_kpm_analyze import jackson
            z = np.load(args.kpm, allow_pickle=True)
            mom, lmx = z["moments"], float(z["lam_max"])
            p = mom.shape[1] - 1
            gj = jackson(p)
            mu_p = mom.mean(0)[:p + 1] * gj
            se_p = mom.std(0, ddof=1)[:p + 1] / np.sqrt(mom.shape[0]) * gj
            grid = np.linspace(max(a - 0.6, 1e-3), b + 0.6, 6000)
            xg = np.clip(2 * grid / lmx - 1, -1 + 1e-12, 1 - 1e-12)
            thg = np.arccos(xg)
            Tg = np.cos(np.outer(np.arange(p + 1), thg))
            dos = (mu_p[0] + 2 * (mu_p[1:] @ Tg[1:])) / (np.pi * np.sin(thg)) * 2 / lmx
            dos_se = np.sqrt((se_p[0]) ** 2 + 4 * ((se_p[1:] ** 2) @ (Tg[1:] ** 2))) \
                / (np.pi * np.sin(thg)) * 2 / lmx
            rho = _bandpass_rho(grid, a, b, lam_max, args.degree)
            # The "outside" region is TWO DISJOINT intervals. Integrating it
            # as one array makes np.trapezoid bridge the gap between them with
            # a single trapezoid spanning the whole window -- width (b-a),
            # height ~ half the DOS at each edge. On the production window
            # that phantom segment was 199.98 of a reported 207.68, driving
            # missed_estimate to -198. Integrate each side on its own.
            lo_m, hi_m = grid < a, grid > b
            def _piece(y):
                return (float(np.trapezoid(y[lo_m], grid[lo_m]))
                        + float(np.trapezoid(y[hi_m], grid[hi_m])))
            leak = _piece(rho * dos)
            leak_err = _piece(rho * dos_se)

        # states we already have inside this interval (for context)
        n_in = int(((lam_found >= a) & (lam_found <= b)).sum())
        rho_found = _bandpass_rho(lam_found, a, b, lam_max, args.degree)
        results[name] = {
            "interval": [a, b], "n_found_inside": n_in,
            "found_rho_weight": float(rho_found.sum()),
            "deflated_estimate": mean, "se": se,
            "predicted_edge_leakage": leak, "leakage_se": leak_err,
            "missed_estimate": (mean - leak) if leak is not None else None,
            "wall_seconds": time.perf_counter() - t0,
        }
        print(json.dumps({name: results[name]}, indent=1), flush=True)

    win = results["window"]
    sub = results.get("sub_gap")
    tot_se = float(np.hypot(win["se"], win["leakage_se"] or 0.0))
    out = {
        "gate": args.gate_name, "when": time.strftime("%Y-%m-%d %H:%M"),
        "rundir": str(rundir), "degree": args.degree, "probes": args.probes,
        "n_deflated": int(locked.shape[0]),
        "estimator": "v2 paired bandpass, deflated probes, DOS-predicted edge leakage",
        "results": results,
        "missed_window": win.get("missed_estimate"),
        "missed_window_se": tot_se,
        "missed_subgap": sub.get("missed_estimate") if sub else None,
        "missed_subgap_se": float(np.hypot(sub["se"], sub["leakage_se"] or 0.0))
                            if sub else None,
        "pass": bool(win.get("missed_estimate") is not None
                     and abs(win["missed_estimate"]) < max(0.5, 2 * tot_se)
                     and (sub is None or abs(sub["missed_estimate"]) < 0.5)),
        "wall_seconds": time.perf_counter() - t_all,
    }
    gates = Path(__file__).resolve().parents[2] / "results" / "gates" / "gate_results.json"
    data = json.loads(gates.read_text()) if gates.exists() else {}
    data[args.gate_name] = out
    gates.write_text(json.dumps(data, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
