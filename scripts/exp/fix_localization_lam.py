#!/usr/bin/env python
"""Refresh the stale `lam` labels in localization_modes.json.

The round-3 F1 fix (unnormalised Rayleigh quotient) shifted every interior
eigenvalue by ~4.7e-5 relative, but the localization JSONs were written
before it and still carry the raw values.

Nothing else in them is stale. analyze_localization.py fits the envelope on
the energy density alone and only afterwards attaches `r["lam"] = float(lam)`
as a label, so xi, participation, r2 and the unresolved flag are all
independent of the eigenvalue and must NOT be recomputed -- doing so would
re-derive identical numbers at the cost of hours of I/O.

Each row is matched to window_eigenvalues_raw.npy EXACTLY (the value it was
written from) and replaced with the corrected value at that index. Matching
on the raw array rather than nearest-neighbour on the corrected one keeps the
mapping verifiable: an unmatched row is an error, not a silent guess.

    conda run -n lsu_ml python scripts/exp/fix_localization_lam.py results/<dir> [...]

CPU only, no GPU, no re-fit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def patch(run: Path) -> bool:
    js = run / "localization_modes.json"
    raw_p = run / "window_eigenvalues_raw.npy"
    cor_p = run / "window_eigenvalues.npy"
    if not js.exists():
        print(f"  {run.name}: no localization_modes.json, skip")
        return False
    if not raw_p.exists():
        print(f"  {run.name}: no raw eigenvalues -- never corrected, labels are "
              f"already right, skip")
        return False

    raw, cor = np.load(raw_p), np.load(cor_p)
    rows = json.loads(js.read_text())
    if all("lam_raw" in r for r in rows):
        print(f"  {run.name}: already patched, skip")
        return False

    n_fixed, worst = 0, 0.0
    for r in rows:
        hit = np.flatnonzero(raw == r["lam"])
        if len(hit) != 1:
            print(f"  {run.name}: lam={r['lam']!r} matched {len(hit)} raw "
                  f"entries -- ABORT, no rows written")
            return False
        i = int(hit[0])
        r["lam_raw"] = r["lam"]
        r["lam"] = float(cor[i])
        worst = max(worst, abs(cor[i] - raw[i]) / raw[i])
        n_fixed += 1
    rows.sort(key=lambda r: r["lam"])
    js.write_text(json.dumps(rows, indent=1))
    print(f"  {run.name}: {n_fixed} labels refreshed, max relative shift "
          f"{worst:.3e} (xi/PR/r2 untouched)")
    return True


if __name__ == "__main__":
    args = sys.argv[1:] or [
        "results/n10k_G192_window", "results/n10k_G192_Sbelow",
        "results/n10k_G192_Sabove", "results/n10k_G192_Sgap"]
    root = Path(__file__).resolve().parents[2]
    for a in args:
        patch(root / a if not Path(a).is_absolute() else Path(a))
