#!/usr/bin/env python
"""Level statistics of the certified spectra (CPU).

Adjacent-gap ratio r_n = min(s_n, s_{n-1}) / max(s_n, s_{n-1}) (Oganesyan &
Huse 2007; Atas, Bogomolny, Giraud & Roux 2013) per spectral band, with
bootstrap uncertainties; nearest-neighbour spacing distributions after local
unfolding; reference values Poisson <r> = 2 ln 2 - 1 = 0.38629 and GOE
<r> = 0.5307 (large-N numerical; 3x3 Wigner surmise 4 - 2 sqrt 3 = 0.53590).

Symmetry class: Theta = curl eps^-1 curl at k = 0 with real eps(r) is a real
symmetric operator on real H(r) (time-reversal invariant, no spin), so the
relevant Wigner-Dyson class is the orthogonal one (GOE), not GUE. The
disordered network has no point symmetry, so no symmetry-induced
degeneracies are expected (the near-degenerate pairs that do occur are
accidental).

Also computes: Poisson/GOE likelihood-ratio classification of each band's
r-sample, the spacing-distribution KS statistics, and the Thouless-ratio
statement (not obtainable from saved data). Writes report/numbers/s05_levelstats.json
and report/tables/levelstats.json.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import stats

from common import GAP_HI_10K, GAP_LO_10K, RES, TAB, USB, Ledger, rel

led = Ledger(__file__)
rng = np.random.default_rng(20260831)
R_POISSON = 2 * np.log(2) - 1                 # 0.386294
R_GOE = 0.5307                                # Atas et al. 2013, large N
R_GOE_SURMISE = 4 - 2 * np.sqrt(3)            # 0.535898
R_GUE = 0.6027                                # for reference only (not the class here)
led.add("r_poisson", R_POISSON, "dimensionless", "Atas et al. PRL 110, 084101 (2013)", "2 ln 2 - 1")
led.add("r_goe", R_GOE, "dimensionless", "Atas et al. PRL 110, 084101 (2013)", "large-N numerical value; surmise 0.5359")
led.add("r_goe_surmise", R_GOE_SURMISE, "dimensionless", "Atas et al. 2013", "4 - 2 sqrt(3), 3x3 Wigner surmise")
led.add("r_gue", R_GUE, "dimensionless", "Atas et al. 2013", "not the symmetry class of Theta at Gamma with real eps")


def r_ratios(lam):
    s = np.diff(np.sort(lam))
    s = s[s > 0]
    return np.minimum(s[1:], s[:-1]) / np.maximum(s[1:], s[:-1])


def r_surmise_poisson(r):
    return 2.0 / (1 + r) ** 2


def r_surmise_goe(r):
    # Atas et al. 2013, beta = 1: P(r) = (27/8)(r + r^2)/(1 + r + r^2)^(5/2) on r in [0, inf);
    # for the folded ratio r~ = min/max in [0, 1] the density doubles: prefactor 27/4 (integrates to 1).
    return 27.0 / 4.0 * (r + r ** 2) / (1 + r + r ** 2) ** 2.5


def band_stats(lam, name, n_boot=20000):
    lam = np.sort(np.asarray(lam, float))
    r = r_ratios(lam)
    if len(r) < 3:
        return {"band": name, "n_levels": int(len(lam)), "n_r": int(len(r)), "r_mean": None}
    boots = np.array([rng.choice(r, len(r), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [16, 84])
    # log-likelihood ratio GOE vs Poisson from the analytic surmises
    llr = float(np.sum(np.log(r_surmise_goe(r)) - np.log(r_surmise_poisson(r))))
    # KS distances to the two surmise CDFs (numerical CDFs on a grid)
    grid = np.linspace(0, 1, 4001)
    cdf_p = np.cumsum(r_surmise_poisson(grid)) * (grid[1] - grid[0])
    cdf_g = np.cumsum(r_surmise_goe(grid)) * (grid[1] - grid[0])
    cdf_p /= cdf_p[-1]
    cdf_g /= cdf_g[-1]
    emp = np.searchsorted(np.sort(r), grid, side="right") / len(r)
    ks_p, ks_g = float(np.max(np.abs(emp - cdf_p))), float(np.max(np.abs(emp - cdf_g)))
    # expected statistical error of <r> for a sample this size (from the surmise variances)
    var_p = float(np.sum(grid ** 2 * r_surmise_poisson(grid)) * (grid[1] - grid[0]) - (np.sum(grid * r_surmise_poisson(grid)) * (grid[1] - grid[0])) ** 2)
    var_g = float(np.sum(grid ** 2 * r_surmise_goe(grid)) * (grid[1] - grid[0]) - (np.sum(grid * r_surmise_goe(grid)) * (grid[1] - grid[0])) ** 2)
    # unfolded spacings (local mean spacing over a 9-level window) for <s^2> and the fraction of small spacings
    s = np.diff(lam)
    k = min(9, len(s))
    loc = np.convolve(s, np.ones(k) / k, mode="same")
    su = s / loc
    return {"band": name, "n_levels": int(len(lam)), "n_r": int(len(r)),
            "lam_range": [float(lam[0]), float(lam[-1])],
            "r_mean": float(r.mean()), "r_se_boot": float(boots.std(ddof=1)), "r_16_84": [float(lo), float(hi)],
            "r_median": float(np.median(r)),
            "llr_goe_minus_poisson": llr, "ks_poisson": ks_p, "ks_goe": ks_g,
            "sigma_r_expected_poisson": float(np.sqrt(var_p / len(r))), "sigma_r_expected_goe": float(np.sqrt(var_g / len(r))),
            "z_from_poisson": float((r.mean() - R_POISSON) / np.sqrt(var_p / len(r))),
            "z_from_goe": float((r.mean() - R_GOE) / np.sqrt(var_g / len(r))),
            "mean_spacing": float(s.mean()), "median_spacing": float(np.median(s)),
            "unfolded_s2": float(np.mean(su ** 2)), "frac_unfolded_s_lt_0p2": float(np.mean(su < 0.2)),
            "unfolded_spacings": su.tolist(), "r_values": r.tolist()}


out = {}
# ------------------------------------------------------------------ N=10k (133)
lam10 = np.load(RES / "n10k_G192_window" / "window_eigenvalues.npy")
src10 = rel(RES / "n10k_G192_window" / "window_eigenvalues.npy")
lo, hi = GAP_LO_10K, GAP_HI_10K
seam = [1.8707585792024861, 1.8730821374768227, 1.929596209673228, 1.947209447202747]
bands10 = {
    "n10k_below_edge": lam10[lam10 < lo],
    "n10k_in_gap_all10": lam10[(lam10 > lo) & (lam10 < hi)],
    "n10k_in_gap_bulk6": np.array([x for x in lam10 if lo < x < hi and not any(abs(x - s_) < 1e-9 for s_ in seam)]),
    "n10k_above_edge": lam10[lam10 > hi],
    "n10k_below_far": lam10[lam10 < 1.82],
    "n10k_below_near": lam10[(lam10 >= 1.82) & (lam10 < lo)],
    "n10k_above_near": lam10[(lam10 > hi) & (lam10 <= 2.05)],
    "n10k_above_far": lam10[lam10 > 2.05],
    "n10k_all133": lam10,
    "n10k_outside_gap_pooled_r": None,
}
for k, v in bands10.items():
    if v is None:
        continue
    out[k] = band_stats(v, k)
    out[k]["source"] = src10
# pooled r over the two outside bands (r computed within each band, then pooled)
rp = np.concatenate([r_ratios(bands10["n10k_below_edge"]), r_ratios(bands10["n10k_above_edge"])])
boots = np.array([rng.choice(rp, len(rp), replace=True).mean() for _ in range(20000)])
out["n10k_outside_gap_pooled_r"] = {"band": "n10k_outside_gap_pooled_r", "n_r": int(len(rp)), "r_mean": float(rp.mean()),
                                    "r_se_boot": float(boots.std(ddof=1)), "source": src10,
                                    "llr_goe_minus_poisson": float(np.sum(np.log(r_surmise_goe(rp)) - np.log(r_surmise_poisson(rp))))}
# the periodic re-solve: 7 states, too few -- record only
lp = np.load(RES / "n10k_G192_gap_periodic_v2" / "window_eigenvalues.npy")
out["periodic_v2_7"] = band_stats(lp, "periodic_v2_7")

# ------------------------------------------------------------------ N=1000 elliptical (611) and circular (611)
ev_e = np.load(USB / "eigenvalues_all.npy").astype(np.float64)
ev_c = np.load(RES / "i4_n1000_circ_G128" / "eigenvalues_all.npy")
srce = rel(USB / "eigenvalues_all.npy")
srcc = rel(RES / "i4_n1000_circ_G128" / "eigenvalues_all.npy")
# MPB band b is index b-3. Gap between 500|501 -> indices 497|498.
for tag, ev, src in (("n1k_ell", ev_e, srce), ("n1k_circ", ev_c, srcc)):
    bands = {
        f"{tag}_low_3_200": ev[:198],                       # MPB 3..200 (shell-structured low spectrum)
        f"{tag}_mid_201_400": ev[198:398],                  # MPB 201..400
        f"{tag}_below_edge_401_500": ev[398:498],           # MPB 401..500 (100 levels below the gap)
        f"{tag}_below_edge_451_500": ev[448:498],           # 50 nearest the gap
        f"{tag}_above_edge_501_600": ev[498:598],           # MPB 501..600
        f"{tag}_above_edge_501_550": ev[498:548],
        f"{tag}_window_398_607": ev[395:605],
        f"{tag}_all_611": ev,
    }
    for k, v in bands.items():
        out[k] = band_stats(v, k)
        out[k]["source"] = src
    # r across the gap itself (the gap spacing enters two r values; report them)
    s = np.diff(ev)
    ig = 497
    out[f"{tag}_r_at_gap"] = {"band": f"{tag}_r_at_gap", "r_500": float(min(s[ig - 1], s[ig]) / max(s[ig - 1], s[ig])),
                              "r_501": float(min(s[ig], s[ig + 1]) / max(s[ig, ], s[ig + 1])), "source": src,
                              "gap_over_neighbour_spacing": float(s[ig] / np.median(s[ig - 10:ig + 11]))}

# sliding-window <r>(lambda): consecutive-level windows, step 1 level
def sliding_r(lam, w):
    lam = np.sort(lam)
    res = []
    for i in range(0, len(lam) - w + 1):
        seg = lam[i:i + w]
        r = r_ratios(seg)
        res.append({"lam_centre": float(np.median(seg)), "lam_lo": float(seg[0]), "lam_hi": float(seg[-1]),
                    "r_mean": float(r.mean()), "r_se": float(r.std(ddof=1) / np.sqrt(len(r)))})
    return res
out["sliding_n10k_w15"] = {"window_levels": 15, "rows": sliding_r(lam10, 15), "source": src10}
out["sliding_n10k_w21"] = {"window_levels": 21, "rows": sliding_r(lam10, 21), "source": src10}
out["sliding_n1k_ell_w41"] = {"window_levels": 41, "rows": sliding_r(ev_e, 41), "source": srce}
out["sliding_n1k_circ_w41"] = {"window_levels": 41, "rows": sliding_r(ev_c, 41), "source": srcc}
with open(TAB / "levelstats.json", "w") as f:
    json.dump(out, f, indent=1)

# ------------------------------------------------------------------ ledger scalars
def put(k, note=""):
    o = out[k]
    if o.get("r_mean") is None:
        led.add(f"r_{k}", None, "dimensionless", o.get("source", ""), f"n_levels={o.get('n_levels')}: too few levels")
        return
    led.add(f"r_{k}", [o["r_mean"], o["r_se_boot"]], "dimensionless", o["source"],
            f"n_levels={o.get('n_levels')}, n_r={o['n_r']}; LLR(GOE-Poisson)={o.get('llr_goe_minus_poisson', float('nan')):.2f}; {note}")


for k in out:
    if "r_at_gap" in k or k == "periodic_v2_7" or k.startswith("sliding_"):
        continue
    put(k)
led.add("r_n1k_ell_at_gap", out["n1k_ell_r_at_gap"], "mixed", srce)
led.add("r_n1k_circ_at_gap", out["n1k_circ_r_at_gap"], "mixed", srcc)
led.add("r_periodic_v2_7_levels", out["periodic_v2_7"].get("r_mean"), "dimensionless", rel(RES / "n10k_G192_gap_periodic_v2" / "window_eigenvalues.npy"),
        "7 levels: not a statistic")
# near-degenerate pairs (accidental): fraction of unfolded spacings below 0.05 in each big sample
for k in ("n10k_all133", "n1k_ell_all_611", "n1k_circ_all_611"):
    su = np.array(out[k]["unfolded_spacings"])
    led.add(f"frac_unfolded_s_lt_0p05_{k}", float(np.mean(su < 0.05)), "fraction", out[k]["source"],
            "GOE expects ~0.4% (P(s)~ (pi/2) s), Poisson ~4.9%")
# Thouless ratio statement
led.add("thouless_ratio_available", False, "bool", "saved data",
        "requires dOmega/dk or twisted-boundary sensitivity; only k=Gamma eigenpairs are saved and no re-solve is permitted")
led.add("symmetry_class", "orthogonal (GOE)", "class", "eigenfns/operator.py",
        "real symmetric operator at k=0 with real scalar eps; eigenvectors can be chosen real; no spin, no magnetic field")
# summary numbers used in the text
led.add("n10k_r_below_edge_n", out["n10k_below_edge"]["n_levels"], "levels", src10)
led.add("n10k_r_above_edge_n", out["n10k_above_edge"]["n_levels"], "levels", src10)
led.add("n10k_r_ingap_n", out["n10k_in_gap_all10"]["n_levels"], "levels", src10)
led.save()
for k in ("n10k_below_edge", "n10k_above_edge", "n10k_in_gap_all10", "n10k_in_gap_bulk6", "n10k_outside_gap_pooled_r",
          "n1k_ell_below_edge_401_500", "n1k_ell_above_edge_501_600", "n1k_ell_low_3_200", "n1k_ell_all_611",
          "n1k_circ_below_edge_401_500", "n1k_circ_above_edge_501_600", "n1k_circ_all_611"):
    o = out[k]
    print(f"{k:32s} n={o.get('n_levels', o.get('n_r')):4}  <r> = {o['r_mean']:.3f} +- {o['r_se_boot']:.3f}   LLR = {o.get('llr_goe_minus_poisson', 0):+.1f}")
