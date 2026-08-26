#!/usr/bin/env python
"""Match montage-convention in-gap states to periodic ones by EIGENVECTOR
OVERLAP, not by a lambda window.

exp_periodic_verdict.py pairs states with an absolute tolerance tol=2e-3 on
lambda and reported four in-gap states as "vanished (unexpected)" alongside
five "new" periodic states. Their lambdas differ by 0.0022-0.0034 -- just
outside that window -- which is suspicious: periodic wrapping ADDS the
material missing from the outer shell, and adding dielectric pushes
frequencies DOWN, so a real state is expected to move, not to stay put.

Widening the tolerance to make the answer come out is exactly the move the
pre-registration forbids. So this decides identity the way the project's own
cross-slice dedup rule does -- by overlap of the H-field spectral vectors,
which live in the SAME plane-wave basis in both runs and do not depend on
eps. |<a,b>| near 1 means the same mode; near 0 means genuinely different.

Accumulates in complex128: an fp32 sum over 14M terms carries ~1e-4 error
(the round-3 F1 lesson).  CPU only, streams from disk.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MONT = ROOT / "results" / "n10k_G192_window"
PERI = ROOT / "results" / "n10k_G192_gap_periodic"
GAP_LO, GAP_HI = 1.864, 1.996
CHUNK = 1 << 21


def norm_and_dots(a_path, a_idx, B):
    """<a, B[j]> for all j, plus ||a||, accumulated in fp64."""
    A = np.load(a_path, mmap_mode="r")[a_idx].ravel()
    n = A.shape[0]
    dots = np.zeros(len(B), dtype=np.complex128)
    na = 0.0
    for s in range(0, n, CHUNK):
        ac = np.asarray(A[s:s + CHUNK]).astype(np.complex128)
        na += float(np.real(np.vdot(ac, ac)))
        for j, b in enumerate(B):
            dots[j] += np.vdot(ac, b[s:s + CHUNK].astype(np.complex128))
    return dots, np.sqrt(na)


def main() -> int:
    lam_m = np.load(MONT / "window_eigenvalues.npy")
    lam_p = np.load(PERI / "window_eigenvalues.npy")
    man = json.loads((MONT / "vec_manifest.json").read_text())
    full = "--full" in sys.argv
    # --full scans EVERY montage mode, not just the in-gap ten. Used to ask
    # whether a periodic state has any montage partner at all -- a completeness
    # datapoint independent of I2.
    ingap = (np.arange(len(lam_m)) if full
             else np.where((lam_m >= GAP_LO) & (lam_m <= GAP_HI))[0])

    Vp = np.load(PERI / "window_vecs_spectral.npy", mmap_mode="r")
    B = [np.asarray(Vp[j]).ravel() for j in range(len(lam_p))]
    nb = np.array([np.sqrt(float(np.real(np.vdot(b.astype(np.complex128),
                                                 b.astype(np.complex128)))))
                   for b in B])
    print(f"montage in-gap states: {len(ingap)}   periodic states: {len(lam_p)}")
    print(f"periodic lambdas: {np.round(lam_p, 5)}\n")

    print(f"{'mont lam':>9} {'best peri':>10} {'|overlap|':>10} {'dlam':>9} "
          f"{'rel':>9}  {'2nd best':>9}")
    out = []
    for i in ingap:
        e = man[i]
        dots, na = norm_and_dots(Path(e["dir"]) / "window_vecs_spectral.npy",
                                 e["index"], B)
        ov = np.abs(dots) / (na * nb)
        k = int(np.argmax(ov))
        second = float(np.sort(ov)[-2]) if len(ov) > 1 else 0.0
        dl = float(lam_p[k] - lam_m[i])
        print(f"{lam_m[i]:9.5f} {lam_p[k]:10.5f} {ov[k]:10.4f} {dl:+9.5f} "
              f"{dl/lam_m[i]:+9.2e}  {second:9.4f}")
        out.append({"lam_mont": float(lam_m[i]), "lam_peri": float(lam_p[k]),
                    "overlap": float(ov[k]), "dlam": dl,
                    "second_best": second,
                    "all_overlaps": [float(x) for x in ov]})

    print("\nRULE (same as cross-slice dedup): |overlap| > 0.5 == same state.")
    kept = [r for r in out if r["overlap"] > 0.5]
    print(f"matched by overlap: {len(kept)} of {len(out)} montage in-gap states")
    if kept:
        d = np.array([r["dlam"] for r in kept])
        print(f"lambda shift of matched states: {d.min():+.5f} .. {d.max():+.5f}"
              f"  (all negative == wrapping ADDS dielectric, as expected)")
    unmatched = [r for r in out if r["overlap"] <= 0.5]
    print(f"genuinely gone (no periodic partner above 0.5): "
          f"{[round(r['lam_mont'], 5) for r in unmatched]}")
    # per-periodic-state best partner over everything scanned
    print("\nPER-PERIODIC best montage partner"
          + (" (scanned ALL montage modes)" if full else " (in-gap only)"))
    M = np.array([r["all_overlaps"] for r in out])          # (n_scanned, n_peri)
    for j in range(M.shape[1]):
        b = int(np.argmax(M[:, j]))
        flag = "" if M[b, j] > 0.5 else "   <- NO PARTNER"
        print(f"  peri {lam_p[j]:.5f}  best montage {lam_m[ingap[b]]:.5f}"
              f"  |ov| {M[b, j]:.4f}{flag}")
    name = "periodic_overlap_match_full.json" if full else "periodic_overlap_match.json"
    (ROOT / "results" / "gates" / name).write_text(
        json.dumps({"rule": "|overlap|>0.5", "scanned": "all" if full else "in-gap",
                    "pairs": out}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
