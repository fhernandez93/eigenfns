#!/usr/bin/env python
"""Recompute the I2 v2 edge-leakage term. Three self-caught defects.

The expensive part of I2 v2 -- the deflated Chebyshev measurement, 2.6 h of
moment accumulation per interval -- is CORRECT and untouched. Only the
analytic leakage correction subtracted from it was wrong, in three separate
ways, so this is pure post-processing: no re-solve, no re-run.

(1) DISJOINT MASK. Leakage was

        np.trapezoid(rho[outside] * dos[outside], grid[outside])

    and `outside` selects TWO intervals (below lam_lo, above lam_hi).
    np.trapezoid cannot see the discontinuity and bridges them with a single
    trapezoid spanning the whole interval. On the production window that
    phantom segment was 199.98 of a reported 207.68, giving
    missed_estimate = -198.2 -- impossible, since the deflated measurement is
    an upper bound on leakage + missed and both are non-negative.
    Fix: integrate each connected segment separately.

(2) ERROR PROPAGATION. leakage_se integrated the POINTWISE standard error,
    int |rho| * dos_se, which assumes the DOS error is perfectly correlated
    AND same-signed across all lambda -- an upper bound, not a standard
    error. It returned ~2.3 for BOTH intervals despite their leakages
    differing 17x, because it is dominated by far-field DOS error where the
    true contributions cancel. Fix: compute the leakage once PER PROBE and
    take the paired standard error across probes, which respects the moment
    correlations. Window 2.285 -> 0.188; sub-interval 2.320 -> 0.026.

(3) DOUBLE-COUNTING THE DEFLATED STATES. The leakage model integrated
    rho * DOS over everything outside the interval -- but for a sub-interval
    nested inside the deflated window, the states between the sub-interval
    edge and the window edge HAVE ALREADY BEEN DEFLATED out of the probe.
    They cannot leak in. They contributed 0.4461 of the modelled 0.4478,
    i.e. 99.6% of it. Fix: the leakage integrand is zero wherever the
    spectrum is deflated. For the full window this is a no-op (nothing
    outside it is deflated), so one rule covers both intervals.

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


def _piecewise(y, grid, mask):
    """Integrate y over `mask`, each connected run separately (defect 1)."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return 0.0
    tot = 0.0
    for seg in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
        if len(seg) > 1:
            tot += float(np.trapezoid(y[seg], grid[seg]))
    return tot


def leakage(a, b, defl_lo, defl_hi, lam_max, degree, kpm=KPM):
    """(leak, se, detail) into [a,b] from states that are NOT deflated."""
    from exp_i2_v2 import _bandpass_rho
    from exp_kpm_analyze import jackson

    z = np.load(kpm, allow_pickle=True)
    mom, lmx = z["moments"], float(z["lam_max"])
    P, p = mom.shape[0], mom.shape[1] - 1
    gj = jackson(p)
    grid = np.linspace(max(a - 0.9, 1e-3), b + 0.9, 8000)
    xg = np.clip(2 * grid / lmx - 1, -1 + 1e-12, 1 - 1e-12)
    thg = np.arccos(xg)
    sn = np.pi * np.sin(thg)
    rho = _bandpass_rho(grid, a, b, lam_max, degree)

    def dos_of(mu):
        d = np.empty_like(grid)
        for s in range(0, len(grid), 1000):
            T = np.cos(np.outer(np.arange(p + 1), thg[s:s + 1000]))
            d[s:s + 1000] = (mu[0] + 2 * (mu[1:] @ T[1:])) / sn[s:s + 1000] * 2 / lmx
        return d

    outside = (grid < a) | (grid > b)
    deflated = (grid >= defl_lo) & (grid <= defl_hi)      # defect 3
    valid = outside & ~deflated

    dos = dos_of(mom.mean(0)[:p + 1] * gj)
    leak = _piecewise(rho * dos, grid, valid)
    per = np.array([_piecewise(rho * dos_of(mom[i, :p + 1] * gj), grid, valid)
                    for i in range(P)])                    # defect 2
    se = float(per.std(ddof=1) / np.sqrt(P))
    detail = {
        "modelled_all_outside": _piecewise(rho * dos, grid, outside),
        "from_already_deflated_states": _piecewise(rho * dos, grid,
                                                   outside & deflated),
        "from_non_deflated_states": leak,
    }
    return leak, se, detail


def main() -> int:
    name = sys.argv[1]
    data = json.loads(GATES.read_text())
    e = data[name]
    meta = json.loads((Path(e["rundir"]) / "interior_report.json").read_text())
    lam_max, degree = meta["lam_max"], e["degree"]
    dlo, dhi = meta["window"]           # the deflated set spans the full window
    if "results_raw_buggy_leakage" not in e:
        e["results_raw_buggy_leakage"] = json.loads(json.dumps(e["results"]))
    raw = e["results_raw_buggy_leakage"]

    for key, r in e["results"].items():
        a, b = r["interval"]
        leak, leak_se, detail = leakage(a, b, dlo, dhi, lam_max, degree)
        r["predicted_edge_leakage"] = leak
        r["leakage_se"] = leak_se
        r["leakage_detail"] = detail
        r["missed_estimate"] = r["deflated_estimate"] - leak
        tot = float(np.hypot(r["se"], leak_se))
        print(f"{key:8s} [{a:.4f},{b:.4f}]  deflated {r['deflated_estimate']:9.4f}"
              f" +- {r['se']:.4f}")
        print(f"          leakage {raw[key]['predicted_edge_leakage']:9.3f}"
              f" -> {leak:8.4f} +- {leak_se:.4f}")
        print(f"            (modelled-all {detail['modelled_all_outside']:.4f},"
              f" of which already-deflated {detail['from_already_deflated_states']:.4f})")
        print(f"          missed  {raw[key]['missed_estimate']:9.3f}"
              f" -> {r['missed_estimate']:+8.4f} +- {tot:.4f}")

    win, sub = e["results"]["window"], e["results"].get("sub_gap")
    e["missed_window"] = win["missed_estimate"]
    e["missed_window_se"] = float(np.hypot(win["se"], win["leakage_se"]))
    if sub:
        e["missed_subgap"] = sub["missed_estimate"]
        e["missed_subgap_se"] = float(np.hypot(sub["se"], sub["leakage_se"]))
    # Acceptance per Amendment A3: the SUB-INTERVAL certifies (|missed| < 0.5);
    # the full window is a consistency check only, since its bias cannot be
    # driven below one state at feasible degree.
    e["pass"] = bool(sub is not None and abs(sub["missed_estimate"]) < 0.5)
    e["certification"] = (
        "A3: sub-interval certifies (|missed|<0.5); window is a consistency "
        "check, NOT a certification. Note exp_i2_v2.py's window clause uses "
        "max(0.5, 2*tot_se), which widens as the measurement gets noisier -- "
        "it must never be read as certifying.")
    e["leakage_fix"] = ("three defects corrected post-hoc: disjoint-mask "
                        "trapezoid, pointwise-|se| error propagation, and "
                        "double-counting of already-deflated states "
                        "(see fix_i2_leakage.py)")
    data[name] = e
    GATES.write_text(json.dumps(data, indent=1))
    print(f"\nCERTIFYING (sub-interval) |missed| = "
          f"{abs(sub['missed_estimate']):.4f} < 0.5  ->  pass = {e['pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
