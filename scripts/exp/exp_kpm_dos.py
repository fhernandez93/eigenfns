#!/usr/bin/env python
"""Phase 1.1 KPM DOS / eigenvalue-counting via stochastic Chebyshev moments.

One recurrence pass per probe chunk collects ALL moments mu_k = z^T T_k(B) z
(B = (2 Theta - lam_max)/lam_max mapped to [-1,1]); the counting function
N(lam) and the DOS follow for every threshold at once — unlike
`kpm_count_below`, which pays a full recurrence per threshold. Uses the
standard even/odd doubling identities (mu_2k = 2<T_k,T_k> - mu_0,
mu_2k+1 = 2<T_k+1,T_k> - mu_1) to get degree-p moments from p/2 matvecs.

Per-probe moments are saved so the analysis can quote stochastic error bars.

    conda run -n lsu_ml python scripts/exp/exp_kpm_dos.py <ends.txt> \
        --grid 128 --degree 8000 --probes 16 [--radius R --aspect A --eps-rod E] \
        --tag NAME [--lam-max X] [--time-only]

Writes results/exp/<tag>_kpm.npz (per-probe moments, lam_max, timings, meta).
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
    ap.add_argument("structure")
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--degree", type=int, default=8000)
    ap.add_argument("--probes", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=0, help="probe vectors per GPU chunk")
    ap.add_argument("--radius", type=float, default=0.2252)
    ap.add_argument("--aspect", type=float, default=2.5)
    ap.add_argument("--eps-rod", type=float, default=2.9275**2)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lam-max", type=float, default=0.0, help="reuse a known lam_max")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--time-only", action="store_true",
                    help="short run: measure ms/matvec + lam_max, no full moments")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from run_modes import gpu_is_busy
    if not args.force and gpu_is_busy():
        print("GPU busy — one heavy job at a time (--force to override).",
              file=sys.stderr)
        return 2

    import jax
    import jax.numpy as jnp
    from eigenfns.operator import MaxwellOperator
    from eigenfns.structure import load_rods, rasterize_penlike
    from eigenfns.chebyshev import lanczos_lambda_max

    rods, N, L = load_rods(args.structure)
    print(f"N={N} L={L:.4f} grid {args.grid}^3 decoration: r={args.radius} "
          f"aspect={args.aspect} eps={args.eps_rod:.4f}", flush=True)
    t0 = time.perf_counter()
    eps = rasterize_penlike(rods, args.grid, L, minor_radius=args.radius,
                            aspect_ratio=args.aspect, eps_rod=args.eps_rod)
    ff = float((eps != 1).mean())
    print(f"rasterized in {time.perf_counter()-t0:.1f}s  ff={ff:.5f}", flush=True)
    op = MaxwellOperator(eps, L)
    G = args.grid

    if args.lam_max > 0:
        lam_max = args.lam_max
        print(f"lam_max (given) = {lam_max:.4f}", flush=True)
    else:
        t0 = time.perf_counter()
        lam_max = 1.05 * lanczos_lambda_max(op)
        print(f"lam_max (Lanczos x1.05) = {lam_max:.4f}  "
              f"[{time.perf_counter()-t0:.1f}s]", flush=True)

    chunk = args.chunk or max(2, int(1.0e9 // (2 * G**3 * 8)))
    chunk = min(chunk, args.probes)
    print(f"probe chunk = {chunk}", flush=True)

    mask = (op.basis.kn > 0).astype(jnp.float32)[None, None]
    inv_l = 2.0 / lam_max

    def Bx(V):
        return inv_l * op.theta(V) - V

    # measure ms/matvec on the chunk shape (after warmup/compile)
    key = jax.random.PRNGKey(args.seed)
    kz, _ = jax.random.split(key)
    Zw = jnp.where(jax.random.bernoulli(kz, 0.5, (chunk, 2, G, G, G)),
                   1.0, -1.0).astype(jnp.complex64) * mask
    Bx(Zw).block_until_ready()  # compile
    t0 = time.perf_counter()
    n_time = 10
    V = Zw
    for _ in range(n_time):
        V = Bx(V)
    V.block_until_ready()
    ms_per_vec = (time.perf_counter() - t0) / (n_time * chunk) * 1e3
    print(f"measured {ms_per_vec:.2f} ms/vector-matvec at {G}^3 (chunk {chunk})",
          flush=True)
    del V, Zw

    outdir = Path(__file__).resolve().parents[2] / "results" / "exp"
    meta = dict(structure=str(args.structure), N=N, L=L, grid=G, ff=ff,
                radius=args.radius, aspect=args.aspect, eps_rod=args.eps_rod,
                degree=args.degree, probes=args.probes, chunk=chunk,
                lam_max=lam_max, seed=args.seed, ms_per_vec=ms_per_vec)
    if args.time_only:
        (outdir / f"{args.tag}_timing.json").write_text(json.dumps(meta, indent=1))
        print("time-only: wrote timing json", flush=True)
        return 0

    half = args.degree // 2  # matvecs per probe; moments up to 2*half+1
    n_mom = 2 * half + 2
    mom = np.zeros((args.probes, n_mom), np.float64)

    def dots(A, Bv):
        fA = A.reshape(A.shape[0], -1)
        fB = Bv.reshape(Bv.shape[0], -1)
        return np.asarray(jnp.real(jnp.sum(fA.conj() * fB, axis=1)), np.float64)

    t_start = time.perf_counter()
    done = 0
    for s in range(0, args.probes, chunk):
        c = min(chunk, args.probes - s)
        kz = jax.random.fold_in(jax.random.PRNGKey(args.seed), s)
        Z = jnp.where(jax.random.bernoulli(kz, 0.5, (c, 2, G, G, G)),
                      1.0, -1.0).astype(jnp.complex64) * mask
        t0v, t1v = Z, Bx(Z)
        mu0 = dots(t0v, t0v)          # = n_dim exactly for Rademacher probes
        mu1 = dots(t0v, t1v)
        mom[s:s+c, 0], mom[s:s+c, 1] = mu0, mu1
        for k in range(1, half + 1):
            # have T_k = t1v, T_{k-1} = t0v at loop entry after first advance;
            # bookkeeping: at iteration k we advance to T_{k+1} then record
            t2v = 2.0 * Bx(t1v) - t0v
            # mu_{2k} = 2<T_k,T_k> - mu_0 ; mu_{2k+1} = 2<T_{k+1},T_k> - mu_1
            mom[s:s+c, 2*k] = 2.0 * dots(t1v, t1v) - mu0
            mom[s:s+c, 2*k+1] = 2.0 * dots(t2v, t1v) - mu1
            t0v, t1v = t1v, t2v
            if k % 200 == 0:
                t1v.block_until_ready()
                el = time.perf_counter() - t_start
                total_mv = (done + c * k / half) if half else done
                print(f"  probes {s}..{s+c-1}: k={k}/{half}  elapsed {el:7.1f}s",
                      flush=True)
        t1v.block_until_ready()
        done += c
        print(f"  chunk done ({done}/{args.probes} probes)  "
              f"elapsed {time.perf_counter()-t_start:.1f}s", flush=True)
        del Z, t0v, t1v, t2v

    wall = time.perf_counter() - t_start
    meta["wall_seconds"] = wall
    outp = outdir / f"{args.tag}_kpm.npz"
    np.savez_compressed(outp, moments=mom, lam_max=lam_max,
                        meta=json.dumps(meta))
    print(f"wrote {outp}  ({wall:.0f}s for {args.probes}x{half} block-matvecs)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
