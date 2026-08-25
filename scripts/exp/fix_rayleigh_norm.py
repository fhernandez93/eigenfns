#!/usr/bin/env python
"""Retroactively correct eigenvalues computed with an unnormalized Rayleigh
quotient (adversarial round 3, F1).

`rr_extract`/`rr_extract_hosted` reported λ = ⟨x,Θx⟩ without dividing by
‖x‖², and SVQB leaves ‖x‖² off unity by ~3e-5 (128³) to ~5e-5 (192³). The
corrected value is exactly λ_raw / ‖x‖² for the SAME saved vector — a pure
post-processing step, no re-solve needed. Norms are accumulated in fp64
(an fp32 sum over millions of terms has its own ~1e-4 error and would hide
the effect entirely).

    conda run -n lsu_ml python scripts/exp/fix_rayleigh_norm.py results/<dir> [...]

Writes per directory: window_eigenvalues_raw.npy (the original, kept),
window_norms.npy, RAYLEIGH_CORRECTION.md; and replaces
window_eigenvalues.npy with the corrected values. CPU only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def correct(run: Path) -> dict | None:
    ev_p = run / "window_eigenvalues.npy"
    vec_p = run / "window_vecs_spectral.npy"
    if not ev_p.exists():
        print(f"  {run}: no eigenvalues, skip")
        return None
    lam = np.load(ev_p)
    if (run / "window_eigenvalues_raw.npy").exists():
        print(f"  {run}: already corrected, skip")
        return None

    if vec_p.exists():
        src = [(vec_p, i) for i in range(len(lam))]
    else:  # merged window: stream via manifest
        man = json.loads((run / "vec_manifest.json").read_text())
        src = [(Path(e["dir"]) / "window_vecs_spectral.npy", e["index"]) for e in man]
    assert len(src) == len(lam)

    n2 = np.empty(len(lam))
    maps: dict = {}
    for k, (p, i) in enumerate(src):
        if p not in maps:
            maps[p] = np.load(p, mmap_mode="r")
        v = np.asarray(maps[p][i]).astype(np.complex128).ravel()
        n2[k] = float((v.conj() * v).sum().real)
        del v
        if k % 25 == 0:
            print(f"    {k}/{len(lam)}", flush=True)
    lam_c = lam / n2
    shift = (lam_c - lam) / lam

    np.save(run / "window_eigenvalues_raw.npy", lam)
    np.save(run / "window_norms.npy", n2)
    np.save(ev_p, lam_c)
    info = {
        "correction": "lambda_corrected = lambda_raw / ||x||^2 (fp64 norms)",
        "reason": "unnormalized Rayleigh quotient in rr_extract (round-3 F1)",
        "norm_sq_minus_1": {"min": float((n2 - 1).min()), "max": float((n2 - 1).max()),
                            "mean": float((n2 - 1).mean())},
        "relative_shift": {"min": float(shift.min()), "max": float(shift.max()),
                           "mean": float(shift.mean())},
        "n_modes": int(len(lam)),
        "note": ("residuals in window_residuals.npy were computed with the raw "
                 "lambda and therefore carry a floor of the same size "
                 "(r_reported^2 = r_true^2 + delta^2); recompute with the "
                 "fixed code for exact values."),
    }
    (run / "RAYLEIGH_CORRECTION.md").write_text(
        "# Eigenvalue correction (adversarial round 3, F1)\n\n"
        + json.dumps(info, indent=1) + "\n\nOriginals kept in "
        "window_eigenvalues_raw.npy; norms in window_norms.npy.\n")
    print(f"  {run.name}: {len(lam)} modes, ||x||^2-1 mean {info['norm_sq_minus_1']['mean']:.3e}, "
          f"lambda shift mean {info['relative_shift']['mean']:.3e}")
    return info


if __name__ == "__main__":
    for d in sys.argv[1:]:
        correct(Path(d))
