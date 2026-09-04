#!/usr/bin/env python
"""Interior gap-window eigenmodes via two-stage bandpass ChebSI (pre-registered
method, docs/plans/2026-08-18_interior_preregistration.md).

    conda run --no-capture-output -n lsu_ml python scripts/run_interior.py \
        <ends.txt> --grid 192 --lam-lo 1.757 --lam-hi 1.930 --m 104 \
        --build-degree 3000 --build-outers 2 \
        --polish-degree 12000 --polish-outers 4 --tol 1e-4 \
        --radius 0.331836 --aspect 1.0 --eps-rod 8.41 \
        --tag n10k_G192_Sbelow [--resume]

Per-outer rolling checkpoint (subspace + stage) with auto-resume; final
extraction is Rayleigh-Ritz on Θ + the residual gate — unconverged in-window
pairs are listed, never silently reported. Saves eigenvalues, spectral H
vectors, per-mode ε|E|², and a run-report JSON.
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("structure")
    ap.add_argument("--grid", type=int, required=True)
    ap.add_argument("--lam-lo", type=float, required=True)
    ap.add_argument("--lam-hi", type=float, required=True)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--build-degree", type=int, default=3000)
    ap.add_argument("--build-outers", type=int, default=2)
    ap.add_argument("--polish-degree", type=int, default=12000)
    ap.add_argument("--polish-outers", type=int, default=4)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--radius", type=float, default=0.331836)
    ap.add_argument("--aspect", type=float, default=1.0)
    ap.add_argument("--eps-rod", type=float, default=8.41)
    ap.add_argument("--periodic", action="store_true",
                    help="wrap rod radii across box faces (CONVENTION CHANGE "
                         "vs the montage: removes the outer-shell material "
                         "deficit — see eigenfns/structure.py)")
    ap.add_argument("--lam-max", type=float, default=0.0)
    ap.add_argument("--chunk", type=int, default=8, help="filter chunk (vectors)")
    ap.add_argument("--theta-chunk", type=int, default=8)
    ap.add_argument("--keep-checkpoint", action="store_true",
                    help="retain interior_state.npz after a successful save "
                         "(default: delete it; it only serves --resume)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from run_modes import gpu_is_busy
    if not args.force and gpu_is_busy():
        print("GPU busy — one heavy job at a time (--force to override).",
              file=sys.stderr)
        return 2

    import jax.numpy as jnp
    from eigenfns.operator import MaxwellOperator
    from eigenfns.structure import load_rods, rasterize_penlike
    from eigenfns.chebyshev import lanczos_lambda_max
    from eigenfns.interior import (bandpass_subspace, bandpass_subspace_hosted,
                                   bandpass_coeffs, _apply_cheb_poly,
                                   rr_extract)

    outdir = Path(__file__).resolve().parents[1] / "results" / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    state_path = outdir / "interior_state.npz"
    t_start = time.perf_counter()

    rods, N, L = load_rods(args.structure)
    eps = rasterize_penlike(rods, args.grid, L, minor_radius=args.radius,
                            aspect_ratio=args.aspect, eps_rod=args.eps_rod,
                            periodic=args.periodic)
    ff = float((eps != 1).mean())
    print(f"N={N} L={L:.4f} grid {args.grid}^3 ff={ff:.5f} window "
          f"[{args.lam_lo}, {args.lam_hi}] m={args.m} "
          f"rasterization={'PERIODIC' if args.periodic else 'montage-convention'}",
          flush=True)
    op = MaxwellOperator(eps, L)

    # --- resume state -----------------------------------------------------
    X0 = None
    stage, outer0, n_theta, lam_max = "build", 0, 0, args.lam_max
    history = []
    if args.resume and state_path.exists():
        st = np.load(state_path, allow_pickle=True)
        X0 = np.asarray(st["X"])  # host; device transfer happens chunk-wise
        if X0.nbytes < 3.5e9:
            X0 = jnp.asarray(X0)
        stage = str(st["stage"])
        outer0 = int(st["outer"]) + 1
        n_theta = int(st["n_theta"])
        lam_max = float(st["lam_max"])
        history = json.loads(str(st["history"]))
        print(f"resumed: stage={stage} next-outer={outer0} n_theta={n_theta}",
              flush=True)
    if lam_max <= 0:
        lam_max = 1.05 * lanczos_lambda_max(op)
        n_theta += 48
    print(f"lam_max = {lam_max:.2f}", flush=True)

    def checkpoint(X, stage_now, outer_now):
        tmp = state_path.with_suffix(".tmp.npz")
        np.savez(tmp, X=np.asarray(X), stage=stage_now, outer=outer_now,
                 n_theta=n_theta, lam_max=lam_max,
                 history=json.dumps(history))
        tmp.replace(state_path)

    # Device-resident path needs headroom for the basis AND the filter
    # recurrence (4 blocks of `chunk` vectors) AND Θ's 3-component FFT
    # transient. Measured OOMs: m=104/192³ (11.8 GB basis) and m=16/192³
    # (1.8 GB basis, chunk 8 → recurrence 3.6 GB + FFT temps). Threshold at
    # 1 GB of basis keeps the device path for small/coarse cases only.
    basis_bytes = args.m * 2 * args.grid ** 3 * 8
    hosted = basis_bytes > 1.0e9
    vec_gb = 2 * args.grid ** 3 * 8 / 2 ** 30
    # cap the filter recurrence transient (~4 chunks) at ~2 GiB
    chunk = min(args.chunk, max(2, int(2.0 / (4 * vec_gb))))
    # Cap the RR/Gram chunk too. Only the filter chunk was capped, so
    # --theta-chunk kept its default of 8 at every grid; at 256^3 that is
    # 8 x 0.25 GiB per block and rr_extract_hosted's gram(Bc, Hc) asked for
    # 6.00 GiB on a 12 GB card and died 1.5 h in (2026-08-26). The peak is
    # ~3 x theta_chunk x vec_gb (Bc + Hc + the conj copy inside gram), so
    # budget 3 GiB. At 192^3 this evaluates to 9 and the cap is inert, which
    # is deliberate: every production result was produced at theta_chunk 8
    # and this must not perturb them.
    tchunk = min(args.theta_chunk, max(2, int(1.0 / vec_gb)))
    if tchunk != args.theta_chunk:
        print(f"theta chunk {args.theta_chunk} -> {tchunk} "
              f"(RR/Gram cap, {vec_gb:.3f} GiB/vector)", flush=True)
    args.theta_chunk = tchunk
    if chunk != args.chunk:
        print(f"filter chunk {args.chunk} -> {chunk} (recurrence transient cap)",
              flush=True)
    args.chunk = chunk
    print(f"basis {basis_bytes / 2**30:.1f} GiB -> "
          f"{'HOST-resident streamed' if hosted else 'device-resident'} path",
          flush=True)

    def one_stage(X0_, stage_name, degree, first_outer, max_outers):
        nonlocal n_theta
        coef = bandpass_coeffs(args.lam_lo, args.lam_hi, lam_max, degree)
        X = X0_
        lam = rn = None
        for outer in range(first_outer, max_outers):
            t0 = time.perf_counter()
            if hosted:
                X0h = np.asarray(X) if X is not None else None
                X, st = bandpass_subspace_hosted(
                    op, args.lam_lo, args.lam_hi, lam_max, m=args.m,
                    degree=degree, max_outer=1, res_tol=args.tol,
                    chunk=args.chunk, theta_chunk=args.theta_chunk,
                    seed=args.seed, X0h=X0h, verbose=False)
                n_theta += st["theta_applications"]
                lam, rn = st["lam"], st["res"]
            elif X is None:  # fresh random start via bandpass_subspace 1 outer
                X, st = bandpass_subspace(op, args.lam_lo, args.lam_hi, lam_max,
                                          m=args.m, degree=degree, max_outer=1,
                                          res_tol=args.tol, chunk=args.chunk,
                                          theta_chunk=args.theta_chunk,
                                          seed=args.seed, verbose=False)
                n_theta += st["theta_applications"]
                lam, rn = st["lam"], st["res"]
            else:
                outs = []
                for s in range(0, X.shape[0], args.chunk):
                    piece = _apply_cheb_poly(op, X[s:s + args.chunk], coef, lam_max)
                    piece.block_until_ready()
                    outs.append(piece)
                    n_theta += degree * int(X[s:s + args.chunk].shape[0])
                del X
                X = jnp.concatenate(outs, axis=0)
                X.block_until_ready()
                del outs, piece
                holder = [X]
                del X
                lam, X, rn = rr_extract(op, holder, args.theta_chunk)
                n_theta += 2 * args.m
            inwin = np.isfinite(lam) & (lam >= args.lam_lo) & (lam <= args.lam_hi)
            conv = inwin & (rn < args.tol)
            rec = {"stage": stage_name, "outer": outer,
                   "in_window": int(inwin.sum()), "converged": int(conv.sum()),
                   "med_res_inwin": float(np.median(rn[inwin])) if inwin.any() else None,
                   "outer_seconds": time.perf_counter() - t0,
                   "elapsed": time.perf_counter() - t_start}
            history.append(rec)
            print(f"  {stage_name} outer {outer}: in-window {rec['in_window']:3d} "
                  f"converged {rec['converged']:3d} med-res "
                  f"{rec['med_res_inwin'] if rec['med_res_inwin'] else float('nan'):.1e} "
                  f"[{rec['outer_seconds']:.0f}s]", flush=True)
            checkpoint(X, stage_name, outer)
            if stage_name == "polish" and inwin.any() and conv.sum() == inwin.sum():
                break
        return X, lam, rn

    X = X0
    if stage == "build":
        X, lam, rn = one_stage(X, "build", args.build_degree, outer0,
                               args.build_outers)
        stage, outer0 = "polish", 0
    X, lam, rn = one_stage(X, "polish", args.polish_degree, outer0,
                           args.polish_outers)

    # --- final extraction + save -----------------------------------------
    inwin = np.isfinite(lam) & (lam >= args.lam_lo) & (lam <= args.lam_hi)
    conv = inwin & (rn < args.tol)
    unconv = inwin & ~conv
    order = np.argsort(lam[conv], kind="stable")
    idx = np.where(conv)[0][order]
    vals = lam[idx]
    print(f"FINAL: {conv.sum()} converged in-window pairs "
          f"({unconv.sum()} in-window NOT converged: "
          f"{[f'{v:.4f}' for v in np.sort(lam[unconv])]} )", flush=True)

    np.save(outdir / "window_eigenvalues.npy", vals)
    np.save(outdir / "window_residuals.npy", rn[idx])
    Xw = X[np.asarray(idx)]
    np.save(outdir / "window_vecs_spectral.npy", np.asarray(Xw))

    # Save the IN-WINDOW UNCONVERGED pairs too. They were previously listed in
    # the report and then discarded, which cost us a real measurement: the
    # periodic gap re-solve left one pair unconverged at lambda = 1.9095, and
    # when I2 later showed that solve was incomplete, the question "is 1.9095
    # a heavily-shifted counterpart of the seam state at 1.92960?" could not
    # be answered, because no vector for it existed anywhere on disk. An
    # unconverged pair is not worthless -- its residual is reported, so it can
    # be used for overlap and locality tests as long as nobody mistakes it for
    # a certified eigenpair. Kept in separate files for exactly that reason.
    uidx = np.where(unconv)[0][np.argsort(lam[unconv], kind="stable")]
    if len(uidx):
        np.save(outdir / "window_unconverged_lams.npy", lam[uidx])
        np.save(outdir / "window_unconverged_residuals.npy", rn[uidx])
        np.save(outdir / "window_unconverged_vecs_spectral.npy",
                np.asarray(X[np.asarray(uidx)]))
        print(f"saved {len(uidx)} UNCONVERGED in-window pairs separately "
              f"(residuals {rn[uidx].min():.2e}-{rn[uidx].max():.2e}; NOT "
              f"certified eigenpairs)", flush=True)
    # Chunk from args.theta_chunk, NOT a hard-coded 8. e_realspace returns
    # THREE real-space components per vector, so a block costs 3x what the
    # solver's own blocks do: at 256^3 that is 0.402 GB/vector, and 8 of them
    # asked for exactly the 3.00 GiB that killed the edgelow anchor at 23:31
    # on 2026-08-28 -- AFTER 20.5 h of solving and after the eigenvalues and
    # vectors had been written. Halved again (theta_chunk is already capped to
    # 4 at 256^3) to keep this well inside the budget, since it runs at peak
    # memory with the whole converged basis still resident.
    ed_chunk = max(1, args.theta_chunk // 2)
    ed = np.empty((len(idx), args.grid, args.grid, args.grid), np.float32)
    for s in range(0, len(idx), ed_chunk):
        E = op.e_realspace(jnp.asarray(np.asarray(Xw[s:s + ed_chunk])),
                           jnp.asarray(vals[s:s + ed_chunk]))
        ed[s:s + ed_chunk] = np.asarray(
            eps[None] * (np.abs(np.asarray(E)) ** 2).sum(1))
        del E
    np.save(outdir / "window_energy_density.npy", ed)
    report = {
        "structure": str(args.structure), "N": N, "L": L, "grid": args.grid,
        "ff": ff, "radius": args.radius, "aspect": args.aspect,
        "eps_rod": args.eps_rod, "window": [args.lam_lo, args.lam_hi],
        "m": args.m, "build_degree": args.build_degree,
        "polish_degree": args.polish_degree, "tol": args.tol,
        "lam_max": lam_max, "n_converged": int(conv.sum()),
        "n_inwindow_unconverged": int(unconv.sum()),
        "unconverged_lams": [float(v) for v in np.sort(lam[unconv])],
        "theta_applications": int(n_theta),
        "wall_seconds": time.perf_counter() - t_start,
        "worst_res_reported": float(rn[idx].max()) if len(idx) else None,
        "history": history,
        "band_numbering": "MPB (+2); absolute index via KPM count (I2)",
    }
    (outdir / "interior_report.json").write_text(json.dumps(report, indent=1))
    print(f"saved {len(idx)} pairs -> {outdir}  "
          f"(theta {n_theta}, wall {report['wall_seconds']:.0f}s)", flush=True)

    # The checkpoint exists only to serve --resume. The run is now complete
    # and every scientific product (eigenvalues, residuals, spectral vectors,
    # energy densities, report) is on disk, so it is dead weight -- and it is
    # large: m * 2 * G^3 * 8 bytes, i.e. 4.9 GB for the 128^3 I4 slice and
    # ~4.5 GB per 256^3 anchor. The queue as of 2026-08-26 needs ~43 GB of
    # writes against 16 GB free, so this is what keeps it alive. Dropped only
    # after the report write above succeeds; --keep-checkpoint opts out when
    # the run may need extending with more outers.
    ck = outdir / "interior_state.npz"
    if ck.exists() and not args.keep_checkpoint:
        mb = ck.stat().st_size / 2 ** 20
        ck.unlink()
        print(f"removed checkpoint ({mb:.0f} MB freed; run complete, "
              f"--keep-checkpoint to retain)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
