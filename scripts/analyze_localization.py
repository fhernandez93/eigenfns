#!/usr/bin/env python
"""Per-mode localization analysis: IPR/participation + envelope-decay ξ(ω),
with the finite-size ceiling ξ_max = L/2 stated explicitly and unresolved
fits reported as lower bounds only (never as extended).

    conda run -n lsu_ml python scripts/analyze_localization.py \
        results/n10k_G192_Sbelow results/n10k_G192_Sabove \
        --box 24.6467 --out results/n10k_localization

Each input dir needs window_eigenvalues.npy + window_energy_density.npy
(run_interior.py or run_modes.py layout). CPU only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eigenfns.localization import mode_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--box", type=float, required=True, help="box side L (um)")
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--a-norm", type=float, default=None,
                    help="normalization length a for nu = sqrt(lam)*a/2pi "
                         "(default: L/ (N/1000)^(1/3) convention not applied; "
                         "reports omega in sqrt(lam) units if unset)")
    args = ap.parse_args()

    rows = []
    for d in args.dirs:
        d = Path(d)
        vals = np.load(d / "window_eigenvalues.npy")
        ed = np.load(d / "window_energy_density.npy", mmap_mode="r")
        assert len(vals) == ed.shape[0], (d, len(vals), ed.shape)
        for i, lam in enumerate(vals):
            r = mode_report(np.asarray(ed[i]), args.box)
            r["lam"] = float(lam)
            r["source"] = str(d.name)
            rows.append(r)
            if i % 25 == 0:
                print(f"  {d.name}: {i}/{len(vals)}", flush=True)
    rows.sort(key=lambda r: r["lam"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}_modes.json", "w") as f:
        json.dump(rows, f, indent=1)

    lam = np.array([r["lam"] for r in rows])
    xi = np.array([r["xi_um"] for r in rows])
    unres = np.array([r["unresolved"] for r in rows])
    prf = np.array([r["pr_fraction"] for r in rows])
    ceil = rows[0]["xi_ceiling_um"]
    nres = int((~unres).sum())
    print(f"{len(rows)} modes; {nres} resolved xi, {len(rows)-nres} "
          f"ceiling-limited (>= {ceil:.2f} um lower bound only)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.sqrt(lam) * (args.a_norm / (2 * np.pi)) if args.a_norm else np.sqrt(lam)
    xlab = r"$\nu = \omega a/2\pi c$" if args.a_norm else r"$\sqrt{\lambda}\ (\mu m^{-1})$"
    fig, axs = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ok = ~unres
    axs[0].scatter(x[ok], xi[ok], s=14, c="#1f6feb", label="resolved ξ")
    axs[0].scatter(x[~ok], np.minimum(xi[~ok], ceil * 1.15), s=14, marker="^",
                   c="#d29922", label=f"unresolved (≥ ceiling, lower bound)")
    axs[0].axhline(ceil, ls="--", c="crimson",
                   label=f"finite-size ceiling L/2 = {ceil:.2f} µm")
    axs[0].set_ylabel(r"$\xi$ (µm)")
    axs[0].legend(loc="upper right", fontsize=8)
    axs[1].scatter(x, prf, s=14, c="#238636")
    axs[1].set_ylabel("participation fraction")
    axs[1].set_xlabel(xlab)
    axs[1].set_yscale("log")
    fig.suptitle("Gap-edge localization: ξ(ω) and participation "
                 f"(box L = {args.box:.2f} µm)")
    fig.tight_layout()
    fig.savefig(f"{out}_xi.png", dpi=150)
    print(f"wrote {out}_modes.json, {out}_xi.png")


if __name__ == "__main__":
    main()
