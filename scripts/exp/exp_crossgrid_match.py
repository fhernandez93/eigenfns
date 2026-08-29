#!/usr/bin/env python
"""Match 192^3 and 256^3 eigenmodes by EIGENVECTOR OVERLAP across grids.

The I6 anchor showed 192^3 and 256^3 both converge 11 states in [1.84, 1.95].
That count agreement is the headline, but it is weaker than it looks: the
typical inter-grid eigenvalue shift (0.0057) is 3.2x the smallest level
spacing in the window (0.0018), so pairing states between grids by sorted
eigenvalue order can mispair them. This does the pairing properly.

WHY IT IS WELL-DEFINED. Both solves use the same box L and the same
transverse frame construction, and make_basis() builds (t1, t2) purely from
the physical k+G vector via a deterministic reference axis. So for the SAME
physical k the frame is IDENTICAL regardless of grid size, and the two
transverse amplitudes mean the same thing in both runs. The reciprocal grid
is k = 2*pi*n/L with n in FFT order, so a 192^3 coefficient at integer triple
n maps to the 256^3 coefficient at the same n -- the 192^3 mode is the
256^3 mode band-limited to |n| < 96. Overlap is then a plain inner product
over the shared k-set.

INTERPRETING THE DEFICIT. The 256^3 mode carries coefficients outside the
192^3 k-set which contribute to its norm but not to the overlap, so
|<a,b>| <= 1 and the shortfall measures how much of the finer-grid mode lives
at wavenumbers the coarse grid cannot represent. A well-resolved mode should
overlap near 1; a mode whose identity depends on the grid should not.

    conda run -n lsu_ml python scripts/exp/exp_crossgrid_match.py

CPU only, complex128 accumulation, streams from disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COARSE = ROOT / "results" / "n10k_G192_window"      # merged 192^3 window
FINE = ROOT / "results" / "n10k_G256_edgelow"       # 256^3 anchor
LO, HI = 1.84, 1.95
G_C, G_F = 192, 256


def fine_to_coarse_index(gc: int, gf: int) -> np.ndarray:
    """FFT indices in the fine grid holding the coarse grid's wavenumbers."""
    n = np.fft.fftfreq(gc, d=1.0 / gc).astype(int)   # [0..gc/2-1, -gc/2..-1]
    assert np.abs(n).max() <= gf // 2
    return n % gf


def main() -> int:
    lam_c_all = np.load(COARSE / "window_eigenvalues.npy")
    man = json.loads((COARSE / "vec_manifest.json").read_text())
    sel = np.where((lam_c_all >= LO) & (lam_c_all <= HI))[0]
    lam_c = lam_c_all[sel]
    lam_f = np.load(FINE / "window_eigenvalues.npy")
    order = np.argsort(lam_f)
    lam_f = lam_f[order]
    Vf = np.load(FINE / "window_vecs_spectral.npy", mmap_mode="r")
    print(f"coarse {G_C}^3: {len(lam_c)} states in [{LO}, {HI}]")
    print(f"fine   {G_F}^3: {len(lam_f)} states\n")

    idx = fine_to_coarse_index(G_C, G_F)
    ix = idx[:, None, None]
    iy = idx[None, :, None]
    iz = idx[None, None, :]

    # restrict every fine vector to the coarse k-set once
    F = np.empty((len(lam_f), 2, G_C, G_C, G_C), np.complex128)
    keep = np.empty(len(lam_f))
    for a, j in enumerate(order):
        v = np.asarray(Vf[j]).astype(np.complex128)
        full = float(np.real(np.vdot(v, v)))
        sub = v[:, ix, iy, iz]
        F[a] = sub
        keep[a] = float(np.real(np.vdot(sub, sub))) / full
        del v, sub
        print(f"  restricted fine mode {a+1}/{len(lam_f)} "
              f"lam={lam_f[a]:.5f}  power inside the {G_C}^3 k-set "
              f"{keep[a]*100:.3f}%", flush=True)

    print(f"\n{'coarse lam':>11} {'fine lam':>10} {'|overlap|':>10} "
          f"{'d lam':>10} {'2nd':>8}")
    rows = []
    for i, gi in enumerate(sel):
        e = man[gi]
        c = np.asarray(np.load(Path(e["dir"]) / "window_vecs_spectral.npy",
                               mmap_mode="r")[e["index"]]).astype(np.complex128)
        c /= np.sqrt(float(np.real(np.vdot(c, c))))
        ov = np.array([abs(np.vdot(c, F[a])) / np.sqrt(
            float(np.real(np.vdot(F[a], F[a])))) for a in range(len(lam_f))])
        k = int(np.argmax(ov))
        second = float(np.sort(ov)[-2])
        rows.append({"lam_coarse": float(lam_c[i]), "lam_fine": float(lam_f[k]),
                     "overlap": float(ov[k]), "second": second,
                     "dlam": float(lam_f[k] - lam_c[i]),
                     "fine_power_in_coarse_kset": float(keep[k])})
        print(f"{lam_c[i]:11.5f} {lam_f[k]:10.5f} {ov[k]:10.4f} "
              f"{lam_f[k]-lam_c[i]:+10.5f} {second:8.4f}")
        del c

    o = np.array([r["overlap"] for r in rows])
    matched = int((o > 0.5).sum())
    uniq = len({r["lam_fine"] for r in rows if r["overlap"] > 0.5})
    print(f"\nmatched at |overlap| > 0.5: {matched} of {len(rows)}  "
          f"(distinct fine partners: {uniq})")
    print(f"overlap: min {o.min():.4f}  median {np.median(o):.4f}  "
          f"max {o.max():.4f}")
    print(f"fine-mode power inside the {G_C}^3 k-set: "
          f"{keep.min()*100:.3f}% .. {keep.max()*100:.3f}%")
    print("\nREAD: a one-to-one matching with high overlap means the two grids "
          "found the SAME states, not merely the same NUMBER of states.")
    (ROOT / "results" / "gates" / "crossgrid_match.json").write_text(
        json.dumps({"coarse": str(COARSE), "fine": str(FINE),
                    "window": [LO, HI], "pairs": rows,
                    "fine_power_in_coarse_kset": keep.tolist()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
