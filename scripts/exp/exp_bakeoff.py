#!/usr/bin/env python
"""Phase 1.2 method bake-off on N=1000 @ 128³ against production ground truth.

Target: the 50-band interior slice straddling the gap — solver-mode indices
473..522 (0-based; MPB bands 476..525, band = index+3), i.e. 25 bands each
side of the 500|501 gap, λ ∈ [1.71129, 2.14583]. Ground truth: results/prod_N1000_G128 (eigenvalues_all.npy +
window_vecs_spectral.npy, window = MPB bands 398..607 -> indices 395..604).

    conda run -n lsu_ml python scripts/exp/exp_bakeoff.py --method folded \
        [--m 64] [--degree 2000] [--inner-tol 1e-2] [--tag t] [--force]

Scores: matched (|Δλ|/λ < 5e-4 AND projection onto the reference cluster
subspace > cos_tol), missed targets, ghosts (converged in-slice pairs
matching no reference band), theta applications, wall-clock.
Writes results/exp/bakeoff_<tag>.json (+ .npz with pairs).
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

ROOT = Path(__file__).resolve().parents[2]
PROD = ROOT / "results" / "prod_N1000_G128"
STRUCT = Path("/home/francisco/Documents/Create LSU Structures  - Claude/"
              "Example/N1000_lsu_example_ends.txt")
SLICE_LO, SLICE_HI = 473, 523  # 0-based solver-mode indices, python slice
WIN_LO = 395                   # index of first window vector (MPB band 398)
RES_TOL = 1e-4                 # production residual gate
MATCH_RTOL = 5e-4              # eigenvalue match tolerance (< half band spacing)
CLUSTER_TOL = 1e-3             # relative λ width for reference cluster subspace
PROJ_TOL = 0.99                # projection^2 onto reference cluster to count as matched


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["folded", "bandpass", "shift_invert", "hybrid"])
    ap.add_argument("--build-outers", type=int, default=2,
                    help="hybrid: bandpass build outers before polish")
    ap.add_argument("--polish-sweeps", type=int, default=8)
    ap.add_argument("--strip-degree", type=int, default=300)
    ap.add_argument("--resume-build", default=None,
                    help="hybrid: skip build, load subspace from this .npy")
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--guard", type=int, default=12)
    ap.add_argument("--tol", type=float, default=1e-3, help="folded: LOBPCG tol")
    ap.add_argument("--precond-reg", type=float, default=None,
                    help="folded: WZ alpha^2 (default sigma^2)")
    ap.add_argument("--maxit", type=int, default=300)
    ap.add_argument("--degree", type=int, default=2000, help="bandpass filter degree")
    ap.add_argument("--outers", type=int, default=8)
    ap.add_argument("--inner-tol", type=float, default=1e-2)
    ap.add_argument("--inner-maxit", type=int, default=250)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "scripts"))
    from run_modes import gpu_is_busy
    if not args.force and gpu_is_busy():
        print("GPU busy — one heavy job at a time.", file=sys.stderr)
        return 2

    import jax.numpy as jnp
    from eigenfns.operator import MaxwellOperator
    from eigenfns.structure import load_rods, rasterize_penlike
    from eigenfns.chebyshev import lanczos_lambda_max
    from eigenfns.interior import (folded_subspace, bandpass_subspace,
                                   shift_invert_subspace, rr_extract)
    from eigenfns.solver import _flat, gram

    tag = args.tag or f"{args.method}_m{args.m}"
    ref_vals = np.load(PROD / "eigenvalues_all.npy")
    sigma = 0.5 * (ref_vals[497] + ref_vals[498])
    tgt_idx = np.arange(SLICE_LO, SLICE_HI)
    tgt_vals = ref_vals[tgt_idx]
    lam_lo, lam_hi = tgt_vals[0] - 0.010, tgt_vals[-1] + 0.010
    print(f"bake-off [{tag}]: sigma={sigma:.5f} slice λ [{tgt_vals[0]:.5f}, "
          f"{tgt_vals[-1]:.5f}] filter window [{lam_lo:.5f}, {lam_hi:.5f}]",
          flush=True)

    rods, N, L = load_rods(STRUCT)
    eps = rasterize_penlike(rods, 128, L)  # production decoration defaults
    op = MaxwellOperator(eps, L)

    t0 = time.perf_counter()
    if args.method == "folded":
        X, stats = folded_subspace(op, sigma, nev=len(tgt_idx), m=args.m,
                                   guard=args.guard, tol=args.tol,
                                   maxit=args.maxit,
                                   precond_reg=args.precond_reg)
        lam, Xr, rn = rr_extract(op, X)
        stats["theta_applications"] += 2 * int(X.shape[0])
    elif args.method == "bandpass":
        lam_max = 1.05 * lanczos_lambda_max(op)
        print(f"lam_max = {lam_max:.1f}", flush=True)
        Xr, stats = bandpass_subspace(op, lam_lo, lam_hi, lam_max, m=args.m,
                                      degree=args.degree, max_outer=args.outers,
                                      res_tol=RES_TOL,
                                      target_count=len(tgt_idx))
        stats["theta_applications"] += 48  # lanczos
        lam, rn = stats["lam"], stats["res"]
    elif args.method == "hybrid":
        from eigenfns.interior import polish_subspace
        lam_max = 1.05 * lanczos_lambda_max(op)
        print(f"lam_max = {lam_max:.1f}", flush=True)
        build_path = ROOT / "results" / "exp" / f"bakeoff_{tag}_build.npz"
        if args.resume_build:
            from eigenfns.interior import rr_extract
            zb = np.load(args.resume_build)
            Xk = jnp.asarray(zb["X"])
            bstats = {"theta_applications": int(zb["n_theta"]),
                      "outer_stats": "resumed"}
            print(f"hybrid: resumed build subspace {Xk.shape} from "
                  f"{args.resume_build}", flush=True)
            if Xk.shape[0] > 56:
                hold = [Xk]
                del Xk
                rlam, Xr56, _ = rr_extract(op, hold)
                cen = 0.5 * (lam_lo + lam_hi)
                half = 0.5 * (lam_hi - lam_lo)
                sc = np.where(np.isfinite(rlam),
                              np.maximum(np.abs(rlam - cen) - half, 0.0), np.inf)
                keep = np.sort(np.argsort(sc, kind="stable")[:56])
                Xk = Xr56[keep]
                Xk.block_until_ready()
                del Xr56
                print(f"hybrid: trimmed resumed subspace to {Xk.shape[0]}",
                      flush=True)
        else:
            Xb, bstats = bandpass_subspace(op, lam_lo, lam_hi, lam_max,
                                           m=args.m, degree=args.degree,
                                           max_outer=args.build_outers,
                                           res_tol=RES_TOL, target_count=None,
                                           chunk=16)
            # shrink to window+margin, capped at 64 (polish peak-memory
            # budget at 128^3), keeping the pairs nearest the window
            blam = bstats["lam"]
            fin = np.isfinite(blam)
            cen, half = 0.5 * (lam_lo + lam_hi), 0.5 * (lam_hi - lam_lo)
            score = np.where(fin, np.maximum(np.abs(blam - cen) - half, 0.0),
                             np.inf)
            keep = np.sort(np.argsort(score, kind="stable")[:56])
            keep = keep[score[keep] <= 0.02]
            print(f"hybrid: keeping {len(keep)}/{args.m} pairs for polish",
                  flush=True)
            Xk = Xb[keep]
            Xk.block_until_ready()
            del Xb
            np.savez_compressed(build_path, X=np.asarray(Xk),
                                n_theta=bstats["theta_applications"])
            print(f"hybrid: build subspace checkpointed -> {build_path}",
                  flush=True)
        # polish = the SAME filtered-SI machinery at high degree on the
        # trimmed basis (the expansion-RR polish oscillated at this scale —
        # 5 instrumented variants, recorded finding; no new numerics wins)
        holder = [Xk]
        del Xk
        Xr, pstats = bandpass_subspace(
            op, lam_lo, lam_hi, lam_max, degree=args.strip_degree,
            max_outer=args.polish_sweeps, res_tol=RES_TOL,
            target_count=len(tgt_idx), chunk=8, X0=holder)
        lam, rn = pstats["lam"], pstats["res"]
        pstats["sweeps"] = pstats.pop("outer_stats")
        stats = {"method": "hybrid", "build": bstats.get("outer_stats"),
                 "polish": pstats["sweeps"],
                 "theta_applications": (bstats["theta_applications"] + 48
                                        + pstats["theta_applications"]),
                 "outer_stats": {"build": bstats.get("outer_stats"),
                                 "polish": pstats["sweeps"]}}
    else:
        Xr, stats = shift_invert_subspace(op, sigma, m=args.m,
                                          max_outer=args.outers,
                                          inner_tol=args.inner_tol,
                                          inner_maxit=args.inner_maxit,
                                          res_tol=RES_TOL,
                                          lam_window=(lam_lo, lam_hi),
                                          target_count=len(tgt_idx))
        lam, rn = stats["lam"], stats["res"]
    wall = time.perf_counter() - t0

    # ---- score against ground truth ------------------------------------
    conv = np.isfinite(lam) & (rn < RES_TOL)
    in_slice = conv & (lam >= tgt_vals[0] - MATCH_RTOL * 2) \
                    & (lam <= tgt_vals[-1] + MATCH_RTOL * 2)
    print(f"converged pairs: {conv.sum()}  in target slice: {in_slice.sum()}",
          flush=True)

    # eigenvalue matching (nearest reference band)
    matched_bands = {}
    ghosts = []
    for i in np.where(in_slice)[0]:
        j = int(np.argmin(np.abs(ref_vals - lam[i])))
        if abs(ref_vals[j] - lam[i]) / ref_vals[j] < MATCH_RTOL:
            matched_bands.setdefault(j, []).append(i)
        else:
            ghosts.append((int(i), float(lam[i])))

    # subspace projection check against reference window vectors (clusters)
    ref_vecs = np.load(PROD / "window_vecs_spectral.npy", mmap_mode="r")
    proj_ok, proj_vals = {}, {}
    import jax
    for j, iis in sorted(matched_bands.items()):
        # reference cluster: all reference bands within CLUSTER_TOL of band j
        cl = np.where(np.abs(ref_vals - ref_vals[j]) / ref_vals[j] < CLUSTER_TOL)[0]
        cl_win = cl[(cl >= WIN_LO) & (cl < WIN_LO + ref_vecs.shape[0])]
        if len(cl_win) == 0:
            proj_ok[j] = None
            continue
        Vref = jnp.asarray(np.asarray(ref_vecs[cl_win - WIN_LO]))
        for i in iis:
            C = np.asarray(gram(Vref, Xr[i:i + 1]))  # (ncl, 1)
            p2 = float((np.abs(C) ** 2).sum())
            proj_vals[(j, int(i))] = p2
            proj_ok[j] = (proj_ok.get(j, True) and p2 > PROJ_TOL)
        del Vref

    tgt_set = set(int(t) for t in tgt_idx)
    # found = eigenvalue match AND projection onto the reference cluster
    # subspace (review F5: a Δλ match with proj² < tol is NOT found; note
    # proj_ok None = cluster outside the reference window, counted found
    # on the eigenvalue criterion alone and disclosed via bad_proj_bands)
    found = sorted(j for j in (set(matched_bands) & tgt_set)
                   if proj_ok.get(j) is not False)
    missed = sorted(tgt_set - set(found))
    dl = [abs(lam[iis[0]] - ref_vals[j]) / ref_vals[j]
          for j, iis in matched_bands.items() if j in tgt_set]
    bad_proj = [j for j in found if proj_ok.get(j) is False]

    out = {
        "tag": tag, "method": args.method, "params": vars(args),
        "sigma": float(sigma), "slice": [int(SLICE_LO), int(SLICE_HI)],
        "wall_seconds": wall,
        "theta_applications": int(stats["theta_applications"]),
        "n_converged": int(conv.sum()),
        "targets_found": len(found), "targets_missed": len(missed),
        "missed_idx": [int(x) for x in missed],
        "ghosts": ghosts,
        "max_dlam_rel": float(max(dl)) if dl else None,
        "median_dlam_rel": float(np.median(dl)) if dl else None,
        "worst_res_matched": float(max(rn[iis[0]] for j, iis in
                                       matched_bands.items() if j in tgt_set)) if found else None,
        "min_proj2": (float(min(v for v in proj_vals.values()))
                      if proj_vals else None),
        "bad_proj_bands": [int(b) for b in bad_proj],
        "outer_stats": stats.get("outer_stats"),
    }
    outp = ROOT / "results" / "exp" / f"bakeoff_{tag}.json"
    outp.write_text(json.dumps(out, indent=1, default=str))
    np.savez_compressed(ROOT / "results" / "exp" / f"bakeoff_{tag}.npz",
                        lam=lam, res=rn,
                        proj=np.array([[j, i, v] for (j, i), v in proj_vals.items()]))
    np.save(ROOT / "results" / "exp" / f"bakeoff_{tag}_subspace.npy",
            np.asarray(Xr))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("outer_stats", "params")}, indent=1,
                     default=str), flush=True)
    print(f"wrote {outp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
