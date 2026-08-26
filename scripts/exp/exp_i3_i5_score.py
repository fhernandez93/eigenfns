#!/usr/bin/env python
"""Gates I3 (residuals + orthonormality) and I5 (spectrum consistency).

    conda run -n lsu_ml python scripts/exp/exp_i3_i5_score.py \
        --rundir results/n10k_G192_window \
        --kpm results/exp/n10k_G256_dos_kpm.npz --gate-suffix "(N=10k 192^3)"

I3: worst per-pair relative residual (recomputed on Θ, not trusted from the
run) + full window Gram ‖G−I‖_max, streamed in chunks.
I5: window eigenvalue extremes and the largest interior spacing (the empty
gap) vs the KPM DOS criterion bracket; the registered "gap empty of converged
pairs" clause is evaluated and reported as-registered.
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

RES_GATE = 1e-4
GRAM_GATE = 5e-5


def _save(entries: dict) -> None:
    """Merge entries into the gate ledger immediately."""
    g = Path(__file__).resolve().parents[2] / "results" / "gates" / "gate_results.json"
    d = json.loads(g.read_text()) if g.exists() else {}
    d.update(entries)
    g.write_text(json.dumps(d, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", required=True)
    ap.add_argument("--kpm", default=None)
    ap.add_argument("--gate-suffix", default="")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--gap-lo", type=float, default=None,
                    help="registered gap interval (default: KPM 10%% criterion)")
    ap.add_argument("--gap-hi", type=float, default=None)
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
    from eigenfns.solver import _flat, gram

    rundir = Path(args.rundir)
    meta = json.loads((rundir / "interior_report.json").read_text())
    lam = np.load(rundir / "window_eigenvalues.npy")
    m = len(lam)
    if (rundir / "window_vecs_spectral.npy").exists():
        X = np.load(rundir / "window_vecs_spectral.npy", mmap_mode="r")
    else:  # merged window: stream vectors from the slice dirs via manifest
        man = json.loads((rundir / "vec_manifest.json").read_text())
        assert len(man) == m
        maps = {}

        class _ManifestView:
            """Minimal read-only fancy/slice view over per-slice .npy files."""

            def __getitem__(self, key):
                idx = range(*key.indices(m)) if isinstance(key, slice) else [key]
                out = []
                for i in idx:
                    e = man[i]
                    p = e["dir"] + "/window_vecs_spectral.npy"
                    if p not in maps:
                        maps[p] = np.load(p, mmap_mode="r")
                    out.append(np.asarray(maps[p][e["index"]]))
                return np.stack(out) if isinstance(key, slice) else out[0]

        X = _ManifestView()
        print(f"streaming vectors via manifest ({len(set(e['dir'] for e in man))} slices)",
              flush=True)
    rods, N, L = load_rods(meta["structure"])
    eps = rasterize_penlike(rods, meta["grid"], L, minor_radius=meta["radius"],
                            aspect_ratio=meta["aspect"], eps_rod=meta["eps_rod"])
    op = MaxwellOperator(eps, L)
    t0 = time.perf_counter()

    # --- I3a: recomputed residuals ---------------------------------------
    res = np.empty(m)
    for s in range(0, m, args.chunk):
        Xc = jnp.asarray(np.asarray(X[s:s + args.chunk]))
        Hc = op.theta(Xc)
        lc = jnp.asarray(lam[s:s + Xc.shape[0]].astype(np.float32))
        Rc = Hc - lc[:, None, None, None, None] * Xc
        res[s:s + Xc.shape[0]] = np.asarray(
            jnp.linalg.norm(_flat(Rc), axis=1)
            / jnp.maximum(jnp.linalg.norm(_flat(Hc), axis=1), 1e-30))
        del Xc, Hc, Rc
    print(f"I3 residuals: worst {res.max():.2e} median {np.median(res):.2e}",
          flush=True)

    # --- I3b: full window Gram -------------------------------------------
    worst_off = 0.0
    worst_diag = 0.0
    for si in range(0, m, args.chunk):
        Ai = jnp.asarray(np.asarray(X[si:si + args.chunk]))
        for sj in range(si, m, args.chunk):
            Aj = jnp.asarray(np.asarray(X[sj:sj + args.chunk]))
            Gb = np.asarray(gram(Ai, Aj))
            for a in range(Gb.shape[0]):
                for b in range(Gb.shape[1]):
                    if si + a == sj + b:
                        worst_diag = max(worst_diag, abs(abs(Gb[a, b]) - 1.0))
                    else:
                        worst_off = max(worst_off, abs(Gb[a, b]))
            del Aj
        del Ai
        if si % (args.chunk * 5) == 0:
            print(f"  gram rows {si}/{m}", flush=True)
    gram_err = max(worst_off, worst_diag)
    print(f"I3 Gram: worst |G-I| = {gram_err:.2e} (offdiag {worst_off:.2e}, "
          f"diag {worst_diag:.2e})", flush=True)

    i3 = {"gate": f"I3 residuals+orthonormality {args.gate_suffix}".strip(),
          "when": time.strftime("%Y-%m-%d %H:%M"), "rundir": str(rundir),
          "n_pairs": int(m), "worst_res": float(res.max()),
          "median_res": float(np.median(res)),
          "worst_gram_dev": float(gram_err),
          "worst_offdiag": float(worst_off), "worst_diag_dev": float(worst_diag),
          "transversality": "exact by construction (2 transverse components)",
          "pass": bool(res.max() <= RES_GATE and gram_err <= GRAM_GATE),
          "wall_seconds": time.perf_counter() - t0}

    # Persist I3 NOW, before I5 can fail. On 2026-08-26 I5 raised
    # TypeError on crit["10%"][0] (no contiguous sub-10% DOS region was
    # found, so the criterion bracket was None) and the exception discarded
    # a completed I3 measurement -- 45 s of GPU work on 133 vectors at 192^3,
    # thrown away for a missing command-line argument.
    _save({i3["gate"]: i3})

    # --- I5: spectrum consistency ----------------------------------------
    i5 = None
    if args.kpm:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp"))
        from exp_kpm_analyze import jackson
        z = np.load(args.kpm, allow_pickle=True)
        mom, lam_max = z["moments"], float(z["lam_max"])
        p = mom.shape[1] - 1
        g = jackson(p)
        mu = mom.mean(0)[:p + 1] * g
        grid = np.linspace(lam[0] * 0.98, lam[-1] * 1.02, 2000)
        x = np.clip(2 * grid / lam_max - 1, -1 + 1e-12, 1 - 1e-12)
        th = np.arccos(x)
        T = np.cos(np.outer(np.arange(p + 1), th))
        rho = (mu[0] + 2 * (mu[1:] @ T[1:])) / (np.pi * np.sin(th)) * 2 / lam_max
        med = np.median(rho)
        crit = {}
        for c in (0.05, 0.10, 0.20):
            low = rho < c * med
            best, cur, s0 = None, None, 0
            for i, fl in enumerate(low):
                if fl and cur is None:
                    cur, s0 = 0, i
                elif fl:
                    cur += 1
                elif cur is not None:
                    if best is None or cur > best[0]:
                        best = (cur, s0, i)
                    cur = None
            if cur is not None and (best is None or cur > best[0]):
                best = (cur, s0, len(low))
            crit[f"{int(c*100)}%"] = ([float(grid[best[1]]), float(grid[best[2]-1])]
                                      if best else None)
        d = np.diff(lam)
        k = int(np.argmax(d))
        eig_gap = [float(lam[k]), float(lam[k + 1])]
        # crit["10%"] is None when no contiguous region falls below the
        # criterion on this grid -- data-dependent, so never assume it.
        auto = crit.get("10%")
        if args.gap_lo is None or args.gap_hi is None:
            if auto is None:
                raise SystemExit(
                    "I5: the 10% DOS criterion found no contiguous bracket on "
                    "this grid, so the gap interval cannot be inferred. Pass "
                    "--gap-lo/--gap-hi explicitly (registered KPM bracket: "
                    "1.864 1.996). I3 has already been saved to the ledger.")
        gl = args.gap_lo if args.gap_lo is not None else auto[0]
        gh = args.gap_hi if args.gap_hi is not None else auto[1]
        in_gap = [float(v) for v in lam if gl < v < gh]
        i5 = {"gate": f"I5 spectrum consistency {args.gate_suffix}".strip(),
              "when": time.strftime("%Y-%m-%d %H:%M"),
              "eigen_window": [float(lam[0]), float(lam[-1])],
              "largest_interior_spacing": eig_gap,
              "eigen_gap_width": float(d[k]),
              "kpm_criterion_bracket": crit,
              "registered_gap_interval": [float(gl), float(gh)],
              "states_inside_registered_gap": in_gap,
              "empty_gap_clause_pass": len(in_gap) == 0,
              "dos_bracket_clause_pass": bool(
                  crit["20%"] and crit["5%"]
                  and crit["20%"][0] <= eig_gap[0] <= crit["5%"][1] + 0.1),
              "note": ("empty-gap clause evaluated as registered; in-gap states "
                       "are a physics finding (see Amendment A2), not a solver "
                       "artifact — each is residual-certified in I3"),
              }
        print(json.dumps({k2: v for k2, v in i5.items() if k2 != "note"}, indent=1),
              flush=True)

    gates = Path(__file__).resolve().parents[2] / "results" / "gates" / "gate_results.json"
    data = json.loads(gates.read_text()) if gates.exists() else {}
    data[i3["gate"]] = i3
    if i5:
        data[i5["gate"]] = i5
    gates.write_text(json.dumps(data, indent=1))
    print(json.dumps(i3, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
