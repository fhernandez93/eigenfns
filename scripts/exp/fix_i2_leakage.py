#!/usr/bin/env python
"""Recompute the I2 v2 edge-leakage term piecewise (self-caught defect).

exp_i2_v2.py computed the leakage as

    np.trapezoid(rho[outside] * dos[outside], grid[outside])

where `outside` selects TWO DISJOINT intervals (below lam_lo, above lam_hi).
np.trapezoid does not know the mask is disconnected: it bridges the two
pieces with one trapezoid spanning the whole window -- width (lam_hi-lam_lo),
height ~ half the DOS at each edge. On the production window that phantom
segment contributed 199.98 of a reported 207.68 leakage, driving
missed_estimate to -198.2 (more states "missing" than the window contains).

The deflated Chebyshev measurement itself is unaffected -- it is the
expensive part and it is correct. Only the analytic leakage correction
subtracted from it was wrong, so this is pure post-processing: no re-solve,
no re-run of the 2.6 h moment accumulation.

    conda run -n lsu_ml python scripts/exp/fix_i2_leakage.py "<gate name>"

Rewrites that entry in results/gates/gate_results.json, keeping the original
under `results_raw_buggy_leakage` for provenance.  CPU only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "exp"))
GATES = ROOT / "results" / "gates" / "gate_results.json"
KPM = ROOT / "results" / "exp" / "n10k_G256_dos_kpm.npz"


def leakage(a, b, lam_max, degree, kpm=KPM):
    """(leak, leak_se) integrated on each side separately."""
    from exp_i2_v2 import _bandpass_rho
    from exp_kpm_analyze import jackson

    z = np.load(kpm, allow_pickle=True)
    mom, lmx = z["moments"], float(z["lam_max"])
    p = mom.shape[1] - 1
    gj = jackson(p)
    mu = mom.mean(0)[:p + 1] * gj
    se = mom.std(0, ddof=1)[:p + 1] / np.sqrt(mom.shape[0]) * gj
    grid = np.linspace(max(a - 0.6, 1e-3), b + 0.6, 6000)
    xg = np.clip(2 * grid / lmx - 1, -1 + 1e-12, 1 - 1e-12)
    thg = np.arccos(xg)
    Tg = np.cos(np.outer(np.arange(p + 1), thg))
    dos = (mu[0] + 2 * (mu[1:] @ Tg[1:])) / (np.pi * np.sin(thg)) * 2 / lmx
    dse = np.sqrt(se[0] ** 2 + 4 * ((se[1:] ** 2) @ (Tg[1:] ** 2))) \
        / (np.pi * np.sin(thg)) * 2 / lmx
    rho = _bandpass_rho(grid, a, b, lam_max, degree)
    lo, hi = grid < a, grid > b

    def piece(y):
        return (float(np.trapezoid(y[lo], grid[lo]))
                + float(np.trapezoid(y[hi], grid[hi])))

    return piece(rho * dos), piece(rho * dse), piece_detail(rho, dos, grid, lo, hi)


def piece_detail(rho, dos, grid, lo, hi):
    return {"below_edge": float(np.trapezoid((rho * dos)[lo], grid[lo])),
            "above_edge": float(np.trapezoid((rho * dos)[hi], grid[hi]))}


def main() -> int:
    name = sys.argv[1]
    data = json.loads(GATES.read_text())
    e = data[name]
    meta = json.loads((Path(e["rundir"]) / "interior_report.json").read_text())
    lam_max, degree = meta["lam_max"], e["degree"]
    e["results_raw_buggy_leakage"] = json.loads(json.dumps(e["results"]))

    for key, r in e["results"].items():
        a, b = r["interval"]
        leak, leak_se, detail = leakage(a, b, lam_max, degree)
        old_leak, old_missed = r["predicted_edge_leakage"], r["missed_estimate"]
        r["predicted_edge_leakage"] = leak
        r["leakage_se"] = leak_se
        r["leakage_by_edge"] = detail
        r["missed_estimate"] = r["deflated_estimate"] - leak
        print(f"{key:9s} [{a:.4f},{b:.4f}]  deflated {r['deflated_estimate']:8.4f}"
              f" +- {r['se']:.4f}")
        print(f"           leakage {old_leak:9.2f} -> {leak:7.3f} +- {leak_se:.3f}"
              f"   (below {detail['below_edge']:.3f}, above {detail['above_edge']:.3f})")
        print(f"           missed  {old_missed:9.2f} -> {r['missed_estimate']:+7.3f}"
              f" +- {np.hypot(r['se'], leak_se):.3f}")

    win, sub = e["results"]["window"], e["results"].get("sub_gap")
    tot_se = float(np.hypot(win["se"], win["leakage_se"]))
    e["missed_window"] = win["missed_estimate"]
    e["missed_window_se"] = tot_se
    if sub:
        e["missed_subgap"] = sub["missed_estimate"]
        e["missed_subgap_se"] = float(np.hypot(sub["se"], sub["leakage_se"]))
    # Acceptance per Amendment A3: the SUB-INTERVAL is the certifying test
    # (|missed| < 0.5). The full window is a consistency check only -- its
    # bias cannot be driven below one state at feasible degree.
    e["pass"] = bool(sub is not None and abs(sub["missed_estimate"]) < 0.5)
    e["certification"] = (
        "A3: sub-interval certifies (|missed|<0.5); window is a consistency "
        "check, NOT a certification. The window clause in exp_i2_v2.py uses "
        "max(0.5, 2*tot_se), which widens as the measurement gets noisier and "
        "must not be read as certifying.")
    e["leakage_fix"] = (
        "piecewise integration; the original summed one trapezoid over a "
        "disjoint mask, bridging the window (see fix_i2_leakage.py)")
    data[name] = e
    GATES.write_text(json.dumps(data, indent=1))
    print(f"\npass (sub-interval certifying): {e['pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
