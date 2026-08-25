#!/usr/bin/env python
"""Verdict on the boundary-seam hypothesis: montage vs periodic rasterization.

    conda run -n lsu_ml python scripts/exp/exp_periodic_verdict.py

Compares the in-gap states of the montage-convention production window
against the periodic-rasterization re-solve of the same λ interval:

  * states flagged as seam-localized (high outer-shell energy) should VANISH
    or move substantially under periodic wrapping;
  * bulk-localized in-gap states should PERSIST at nearly the same λ.

For every state in both runs it reports λ, the outer-2-voxel-shell energy
fraction (volume fraction 6.1%), ξ, and the nearest counterpart in the other
run. Writes results/gates/gate_results.json entry + a summary table. CPU only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ROOT = Path(__file__).resolve().parents[2]
MONT = ROOT / "results" / "n10k_G192_window"
PERI = ROOT / "results" / "n10k_G192_gap_periodic"
GAP_LO, GAP_HI = 1.864, 1.996
SHELL_VOL_FRAC = None  # computed


def shell_mask(G, depth=1):
    idx = np.arange(G)
    d = np.minimum(idx, G - 1 - idx)
    D = np.minimum(np.minimum(d[:, None, None], d[None, :, None]),
                   d[None, None, :])
    return D <= depth


def load(run, lo=None, hi=None):
    lam = np.load(run / "window_eigenvalues.npy")
    ed = np.load(run / "window_energy_density.npy", mmap_mode="r")
    keep = np.arange(len(lam))
    if lo is not None:
        keep = keep[(lam[keep] >= lo) & (lam[keep] <= hi)]
    G = ed.shape[1]
    sh = shell_mask(G)
    rows = []
    xi = {}
    p = run / "localization_modes.json"
    if p.exists():
        for r in json.load(open(p)):
            xi[round(r["lam"], 6)] = r
    for i in keep:
        u = np.asarray(ed[i])
        f = float(u[sh].sum() / u.sum())
        r = xi.get(round(float(lam[i]), 6), {})
        rows.append({"lam": float(lam[i]), "shell_frac": f,
                     "xi_um": r.get("xi_um"), "unresolved": r.get("unresolved"),
                     "pr_fraction": r.get("pr_fraction")})
    return rows, float(sh.mean())


def main():
    if not (PERI / "window_eigenvalues.npy").exists():
        print(f"periodic run not finished yet: {PERI}")
        return 1
    m_rows, vf = load(MONT, GAP_LO, GAP_HI)
    p_rows, _ = load(PERI, GAP_LO, GAP_HI)
    print(f"outer 2-voxel shell = {100*vf:.2f}% of volume\n")

    def near(rows, lam, tol=2e-3):
        best, bd = None, 1e9
        for r in rows:
            d = abs(r["lam"] - lam)
            if d < bd:
                best, bd = r, d
        return (best, bd) if bd <= tol else (None, bd)

    print("MONTAGE-CONVENTION in-gap states -> nearest periodic counterpart")
    print(f"{'lam':>9} {'shell%':>7} {'x vol':>6} {'xi um':>7}  {'verdict':<28} {'peri lam':>9}")
    verdicts = []
    for r in m_rows:
        c, d = near(p_rows, r["lam"])
        seam = r["shell_frac"] / vf >= 2.0
        if c is None:
            v = "VANISHED (seam artifact)" if seam else "vanished (unexpected)"
        else:
            v = "persists" + (" (was flagged seam!)" if seam else "")
        verdicts.append({"lam": r["lam"], "shell_x": r["shell_frac"] / vf,
                         "flagged_seam": bool(seam),
                         "counterpart_lam": c["lam"] if c else None,
                         "delta": None if c is None else abs(c["lam"] - r["lam"]),
                         "verdict": v})
        print(f"{r['lam']:9.5f} {100*r['shell_frac']:6.1f}% {r['shell_frac']/vf:5.1f}x "
              f"{(r['xi_um'] or float('nan')):7.2f}  {v:<28} "
              f"{(c['lam'] if c else float('nan')):9.5f}")

    print("\nPERIODIC states with no montage counterpart (new):")
    for r in p_rows:
        c, d = near(m_rows, r["lam"])
        if c is None:
            print(f"  {r['lam']:9.5f}  shell {100*r['shell_frac']:5.1f}% "
                  f"({r['shell_frac']/vf:4.1f}x)")

    n_seam = sum(1 for v in verdicts if v["flagged_seam"])
    n_seam_gone = sum(1 for v in verdicts if v["flagged_seam"] and v["counterpart_lam"] is None)
    n_bulk = sum(1 for v in verdicts if not v["flagged_seam"])
    n_bulk_kept = sum(1 for v in verdicts
                      if not v["flagged_seam"] and v["counterpart_lam"] is not None)
    out = {
        "test": "boundary-seam hypothesis (montage vs periodic rasterization)",
        "montage_in_gap": len(m_rows), "periodic_in_gap": len(p_rows),
        "shell_volume_fraction": vf,
        "seam_flagged": n_seam, "seam_flagged_vanished": n_seam_gone,
        "bulk_flagged": n_bulk, "bulk_persisted": n_bulk_kept,
        "per_state": verdicts,
        "hypothesis_supported": bool(n_seam and n_seam_gone == n_seam
                                     and n_bulk_kept == n_bulk),
    }
    gates = ROOT / "results" / "gates" / "gate_results.json"
    data = json.loads(gates.read_text()) if gates.exists() else {}
    data["Boundary-seam test (in-gap states, periodic re-solve)"] = out
    gates.write_text(json.dumps(data, indent=1))
    print(f"\nseam-flagged {n_seam}, of which vanished {n_seam_gone}; "
          f"bulk {n_bulk}, of which persisted {n_bulk_kept}")
    print(f"hypothesis fully supported: {out['hypothesis_supported']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
