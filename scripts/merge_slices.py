#!/usr/bin/env python
"""Merge interior-solve slices into one eigenvalue-ordered window directory.

    conda run -n lsu_ml python scripts/merge_slices.py \
        --out results/n10k_G192_window \
        results/n10k_G192_Sbelow results/n10k_G192_Sgap results/n10k_G192_Sabove

Concatenates eigenvalues + ε|E|² (+ residuals) sorted by λ, deduplicating
pairs that two slices both found (eigenvalue coincidence within rel 1e-6 AND
eigenvector overlap > 0.5 — the registered cross-slice dedup rule). Writes an
interior_report.json so make_montage.py / analyze_localization.py can consume
the merged window directly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    # 5e-6: the two real duplicates in the production window differ by
    # 1.42e-6 and 1.73e-6, so the old 1e-6 default missed both and the
    # documented command reproduced 135 instead of 133 (round-3 F4). The
    # eigenvector-overlap test (>0.5) is the actual discriminator; this
    # window only has to be wider than the accuracy of two independent
    # solves of the same state and narrower than the true level spacing
    # (smallest genuine spacing here: 6.8e-5 relative, 14x this).
    ap.add_argument("--dedup-rtol", type=float, default=5e-6)
    args = ap.parse_args()

    dirs = [Path(d) for d in args.dirs if (Path(d) / "window_eigenvalues.npy").exists()]
    print(f"merging {len(dirs)} slices: {[d.name for d in dirs]}", flush=True)
    vals, srcs, idxs = [], [], []
    for d in dirs:
        v = np.load(d / "window_eigenvalues.npy")
        vals.append(v)
        srcs += [d] * len(v)
        idxs += list(range(len(v)))
    lam = np.concatenate(vals)
    order = np.argsort(lam, kind="stable")
    lam = lam[order]
    srcs = [srcs[i] for i in order]
    idxs = [idxs[i] for i in order]

    # dedup: adjacent duplicates from overlapping slice edges
    keep = [0]
    dups = []
    for i in range(1, len(lam)):
        j = keep[-1]
        if abs(lam[i] - lam[j]) / lam[j] < args.dedup_rtol and srcs[i] != srcs[j]:
            # numpy overlap (CPU): keeps the merge off the GPU so it can run
            # alongside gate jobs
            A = np.asarray(np.load(srcs[j] / "window_vecs_spectral.npy",
                                   mmap_mode="r")[idxs[j]]).ravel()
            B = np.asarray(np.load(srcs[i] / "window_vecs_spectral.npy",
                                   mmap_mode="r")[idxs[i]]).ravel()
            ov = float(abs(np.vdot(A, B)) / (np.linalg.norm(A) * np.linalg.norm(B)))
            del A, B
            if ov > 0.5:
                dups.append((float(lam[i]), srcs[i].name, srcs[j].name, ov))
                continue
        keep.append(i)
    print(f"{len(lam)} pairs -> {len(keep)} after dedup ({len(dups)} duplicates)",
          flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lam_k = lam[keep]
    np.save(out / "window_eigenvalues.npy", lam_k)
    res = np.empty(len(keep))
    for n, i in enumerate(keep):
        res[n] = np.load(srcs[i] / "window_residuals.npy")[idxs[i]]
    np.save(out / "window_residuals.npy", res)

    meta0 = json.loads((dirs[0] / "interior_report.json").read_text())
    G = meta0["grid"]
    # Energy densities are copied (montage/localization need them contiguous;
    # ~28 MB/mode). Spectral H vectors are NOT copied — 0.113 GB/mode at 192³
    # would be ~15 GB of duplicate data on a 31 GB disk. Instead a manifest
    # records where each mode lives, and the gate scripts stream from there.
    ed = np.lib.format.open_memmap(out / "window_energy_density.npy", mode="w+",
                                   dtype=np.float32, shape=(len(keep), G, G, G))
    manifest = []
    for n, i in enumerate(keep):
        ed[n] = np.load(srcs[i] / "window_energy_density.npy", mmap_mode="r")[idxs[i]]
        manifest.append({"lam": float(lam[i]), "dir": str(srcs[i]), "index": idxs[i]})
        if n % 25 == 0:
            print(f"  copied {n}/{len(keep)}", flush=True)
    ed.flush()
    (out / "vec_manifest.json").write_text(json.dumps(manifest, indent=1))

    meta = dict(meta0)
    meta.update({
        "merged_from": [d.name for d in dirs],
        "window": [float(lam_k[0]), float(lam_k[-1])],
        "n_converged": len(keep), "duplicates_removed": dups,
        "worst_res_reported": float(res.max()),
        "slice_windows": {d.name: json.loads((d / "interior_report.json").read_text())["window"]
                          for d in dirs},
    })
    (out / "interior_report.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out}: {len(keep)} modes, λ [{lam_k[0]:.5f}, {lam_k[-1]:.5f}], "
          f"worst res {res.max():.2e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
