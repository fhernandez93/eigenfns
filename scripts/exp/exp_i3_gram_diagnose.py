#!/usr/bin/env python
"""Why does the merged window fail the I3 orthonormality gate?

I3 measured |G-I|max = 4.86e-4 against a registered <= 5e-5, entirely in the
off-diagonal (the diagonal is 2.0e-5). Hypothesis: the merged window
concatenates three INDEPENDENTLY solved slices. Within a slice, SVQB
enforces orthonormality directly. Across slices nothing does -- the only
thing making two vectors orthogonal is that they are eigenvectors of the
same Hermitian operator, which degrades as the residuals grow and the
eigenvalue spacing shrinks:

    |<x_i, x_j>|  <~  (r_i + r_j) * lambda / |lambda_i - lambda_j|

If the hypothesis holds, the large off-diagonals are CROSS-SLICE and track
1/|dlambda|. If instead they are within-slice, SVQB itself is suspect and
this is a solver bug, not a merge artifact.

Sliding window over lambda-ordered modes (orthogonality degrades with
spacing, so neighbours carry the worst cases). Accumulates in complex128.
CPU only, streams from disk.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WIN = ROOT / "results" / "n10k_G192_window"
NEIGH = 8
CHUNK = 1 << 21


def load(entry):
    v = np.load(Path(entry["dir"]) / "window_vecs_spectral.npy",
                mmap_mode="r")[entry["index"]]
    return np.asarray(v).ravel()


def dot(a, b):
    s = 0j
    for k in range(0, a.shape[0], CHUNK):
        s += np.vdot(a[k:k + CHUNK].astype(np.complex128),
                     b[k:k + CHUNK].astype(np.complex128))
    return s


def main() -> int:
    lam = np.load(WIN / "window_eigenvalues.npy")
    man = json.loads((WIN / "vec_manifest.json").read_text())
    order = np.argsort(lam)
    lam, man = lam[order], [man[i] for i in order]
    slic = [Path(e["dir"]).name for e in man]
    res = np.load(WIN / "window_residuals.npy")[order]
    print(f"{len(lam)} modes from {sorted(set(slic))}\n")

    buf: dict[int, np.ndarray] = {}
    rows = []
    for i in range(len(lam)):
        buf[i] = load(man[i])
        nb = buf[i] / np.sqrt(float(np.real(dot(buf[i], buf[i]))))
        buf[i] = nb
        for j in range(max(0, i - NEIGH), i):
            ov = abs(complex(dot(buf[j], nb)))
            rows.append({"i": j, "j": i, "lam_i": float(lam[j]),
                         "lam_j": float(lam[i]), "ov": ov,
                         "cross": slic[i] != slic[j],
                         "dlam": float(lam[i] - lam[j]),
                         "res_sum": float(res[i] + res[j])})
        buf.pop(i - NEIGH, None)
        if i % 25 == 0:
            print(f"  {i}/{len(lam)}", flush=True)

    rows.sort(key=lambda r: -r["ov"])
    print(f"\nworst 12 off-diagonals (of {len(rows)} neighbour pairs):")
    print(f"{'lam_i':>9} {'lam_j':>9} {'dlam':>10} {'|<i,j>|':>10} "
          f"{'r_i+r_j':>9} {'bound':>9}  slices")
    for r in rows[:12]:
        bound = r["res_sum"] * r["lam_i"] / max(abs(r["dlam"]), 1e-12)
        tag = "CROSS" if r["cross"] else "same "
        print(f"{r['lam_i']:9.5f} {r['lam_j']:9.5f} {r['dlam']:10.2e} "
              f"{r['ov']:10.2e} {r['res_sum']:9.2e} {bound:9.2e}  {tag}")

    cross = np.array([r["ov"] for r in rows if r["cross"]])
    same = np.array([r["ov"] for r in rows if not r["cross"]])
    print(f"\ncross-slice pairs: n={len(cross):4d}  max |ov| {cross.max():.3e}"
          f"  median {np.median(cross):.3e}")
    print(f"same-slice  pairs: n={len(same):4d}  max |ov| {same.max():.3e}"
          f"  median {np.median(same):.3e}")
    print(f"\nI3 gate is 5e-5. Worst neighbour off-diagonal found: "
          f"{rows[0]['ov']:.3e} ({'CROSS' if rows[0]['cross'] else 'same'}-slice)")
    print("(I3 scanned ALL pairs and found 4.858e-04; if that does not appear "
          "here, the worst pair is not a lambda-neighbour and the spacing "
          "explanation fails.)")
    (WIN / "i3_gram_diagnosis.json").write_text(json.dumps(
        {"neighbours": NEIGH, "worst": rows[:40],
         "cross_max": float(cross.max()), "same_max": float(same.max())},
        indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
