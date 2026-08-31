"""Shared paths, constants and the numbers-ledger helper for the PRL report.

Every script in report/scripts/ imports this. CPU only (numpy / scipy /
matplotlib). Nothing here re-derives a settled constant; constants are
re-verified against files by the individual scripts (see s03_structure.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]          # repo root
REPORT = ROOT / "report"
NUMBERS_DIR = REPORT / "numbers"
FIG = REPORT / "figures"
TAB = REPORT / "tables"
RES = ROOT / "results"
USB = Path("/media/francisco/EXTERNAL_USB/prod_N1000_G128")
STRUCT_N1K = Path("/home/francisco/Documents/Create LSU Structures  - Claude/"
                  "Example/N1000_lsu_example_ends.txt")
STRUCT_N10K = ROOT / "Structures" / "20260701_N10000_lsu_generated.txt"

# make the repo importable (eigenfns.structure / eigenfns.localization are
# numpy-only modules; eigenfns.operator needs JAX and is never imported here)
sys.path.insert(0, str(ROOT))

# ---- settled constants (re-verified from files by s03_structure.py) --------
D0_UM = 0.8
L_N1K = 11.44
L_N10K = (10000 / 1000) ** (1 / 3) * L_N1K          # 24.6467 um
N_IDX_ELL = 2.9275                                    # elliptical decoration
EPS_ELL = N_IDX_ELL ** 2                              # 8.5703
R_ELL, ASPECT_ELL = 0.2252, 2.5
N_IDX_CIRC = 2.9                                      # circular decoration
EPS_CIRC = N_IDX_CIRC ** 2                            # 8.41
R_CIRC, ASPECT_CIRC = 0.331836, 1.0
A_NORM_N1K = L_N1K / 5.0                              # a = 2.288 um (REPORT.md)

# KPM 10%-criterion nominal gap bracket of the N=10k structure (REPORT_N10K)
GAP_LO_10K, GAP_HI_10K = 1.864, 1.996


def nu_from_lam(lam, a=A_NORM_N1K):
    """nu = omega a / (2 pi c) = sqrt(lam) a / (2 pi) for lam=(omega/c)^2."""
    return np.sqrt(np.asarray(lam, float)) * a / (2 * np.pi)


def rel_gap(lo, hi):
    """Delta nu / nu_mid for gap edges given as lambda values."""
    wlo, whi = np.sqrt(lo), np.sqrt(hi)
    return 2 * (whi - wlo) / (whi + wlo)


class Ledger:
    """Collects {key: {value, unit, source_file, script, notes}} entries."""

    def __init__(self, script: str):
        self.script = script
        self.d: dict[str, dict] = {}

    def add(self, key, value, unit, source_file, notes=""):
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        if isinstance(value, np.ndarray):
            value = value.tolist()
        self.d[key] = {"value": value, "unit": unit,
                       "source_file": str(source_file), "script": self.script,
                       "notes": notes}
        return value

    def save(self):
        NUMBERS_DIR.mkdir(parents=True, exist_ok=True)
        out = NUMBERS_DIR / f"{Path(self.script).stem}.json"
        with open(out, "w") as f:
            json.dump(self.d, f, indent=1, sort_keys=True)
        print(f"[ledger] {len(self.d)} entries -> {out.relative_to(ROOT)}")


def rel(p) -> str:
    """Path relative to repo root when possible (for source_file fields)."""
    p = Path(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def load_json(p):
    with open(p) as f:
        return json.load(f)
