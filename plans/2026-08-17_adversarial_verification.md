# Adversarial verification record — interior-solver project (N=10k)

Refutation-tasked subagent reviews; every pass recorded (including "found
nothing"). Fix status tracked inline.

---

## Round 1 — 2026-08-17, Phase 1 investigation claims

Scope: `plans/2026-08-17_interior_investigation.md` + `scripts/exp/exp_kpm_dos.py`,
`exp_kpm_analyze.py`, `exp_ff_calibration_n10k.py`, `eigenfns/interior.py`,
`eigenfns/localization.py`, saved KPM moments, calibration JSON, bake-off logs.
Reviewer recomputed from artifacts (CPU-only) — all recomputable figures
reproduced; "no fabricated numbers anywhere."

### Findings

**F1 — MAJOR — N=10k gap-edge precision overstated.** First-pass edges
[1.8877, 2.0262] were exactly points of the ~0.07-spaced Chebyshev-Gauss
analysis grid; criterion sensitivity (5%/10%/20% of local median) spans width
0.07–0.20. *Fixed:* fine uniform-grid re-analysis (step 5×10⁻⁴) now in report
§4 — width 0.13 ± 0.05 criterion-dominated, Δν/ν ≈ 3.4% range [2.2, 4.8]%;
N=1000 calibration added (§1); exact edges deferred to eigensolver + gate I5.
Downstream window derivation survives (≥0.1 λ margin beyond worst-case edges).

**F2 — MAJOR — folded-FAIL evidence base wrong (verdict survives on other
evidence).** Reviewer proved the folded relative residual is uninformative:
production-quality eigenvectors (Θ rel-res 6e-5) show folded rel-res ≈ 1.0 in
BOTH fp32 and fp64, since ‖(Θ−σ)²x−μx‖ ≈ λ_max·‖r_Θ‖ ≫ μ; the lock gate
tol=1e-3 was a priori unpassable. Implementation audited: no bugs (signs,
WZ diagonal, Θ-count, composition identity). *Fixed:* §6 re-based the FAIL on
the μ-trajectory (locked μ ∈ [0.096, 0.279] vs targets ≤ 0.0497 after 200
its, decelerating descent, preconditioner-insensitive) and states the folded
residual floor explicitly; future-attempt requirements recorded.

**F3 — MAJOR (doc) — bake-off slice mislabeled.** §6 said λ ∈ [1.858, 2.055]
(wrong — corresponds to no slice boundary); actual ev[473..522] =
[1.71129, 2.14583]. Script header MPB label off by one (475..524 → 476..525;
band = 0-based index + 3). *Fixed* in report + script header. Anchors
verified correct by reviewer: index 497|498 = gap 500|501 (1.88296|1.96277),
σ = 1.92286 = true mid-gap, 25 bands per side.

**F4 — MINOR — window-count error bar conflated.** Correct paired per-probe
difference: 317.09 ± 5.10 (probe-correlated thresholds cancel); the ±12.5
absolute se applies to band-index placement. *Fixed* in §4. Reviewer note:
with 12 probes the se itself carries ~21% uncertainty (χ², 11 dof).

**F5 — MINOR — bake-off scoring soft spots.** (i) 6/49 adjacent slice
spacings < combined match window (e.g. 513/514: Δλ=2.5e-4) — ghost detection
blind inside those clusters (~1/3 of slice λ-axis within ±1e-3 of some
reference band); projection criterion is the backstop. (ii) `targets_found`
did not enforce the projection criterion. *Fixed:* found = Δλ-match ∧
projection-pass. (iii) proj None (cluster outside reference window) treated
as pass — disclosed, latent only.

**F6 — MINOR — "128 days" labeled as measurement-scaled when it isn't.**
Production per-iteration cost is flat (11.5–12.1 s/iter, n_locked 20→600) —
the n² regime is not visible at N=1000 scale. First-principles PCIe bound
lands at the same order. *Fixed:* §0 relabels the time estimate
order-of-magnitude; the verified 1.38 TB storage wall (12× RAM+disk) is the
hard kill.

**F7 — note — N=1000 KPM gap validation partly grid luck.** Fine-grid 10%
criterion gives [1.886, 1.948] vs exact [1.883, 1.963] — edges biased by soft
band tails; recorded in §1 as calibration.

**F8 — note — `lobpcg_blocks` locks unconverged pairs at maxit** ("locked" ≠
converged in logs; production unaffected — all production blocks locked at
res ≤ 1.0e-4; bake-off protected by rr_extract + RES_TOL). §6 cell reworded
"0 verified".

### Refutations attempted and FAILED (= confirmations)

1. **KPM moments machinery**: doubling identities verified against dense
   moments (4e-17 agreement); Jackson/step coefficients match
   `kpm_count_below` + textbook; no off-by-one; probes independent; se logic
   correct incl. correlated differences. All shakeout numbers reproduced.
2. **λ_max safety** (both runs): no divergence signature in moment tails;
   N=10k λ_max cross-checked against 288³ Lanczos ×(256/288)² to 0.5%.
   Mid-gap 5010.34 ± 12.54 and edge DOS reproduce.
3. **ff calibration**: independent re-rasterization ff = 0.220004 at
   r = 0.331836/256³; strictly binary voxels; no last-midpoint bisection bug
   (returned pair was measured; trace terminates at |ff−0.22| = 4.4e-6);
   288³ independently re-bisects to the identical radius.
4. **Localization**: minimum-image arithmetic brute-force verified; synthetic
   e^{−2r/ξ} recovered exactly; factor 2 correct for ε|E|²; ξ=20 synthetic
   correctly flagged; band 500 → ξ = 1.822 µm (r² 0.973), bands 420/600
   correctly unresolved (dynamic-range and ceiling gates respectively).
5. **§0 arithmetic**: 1.38 TB, 16.4 h matvec, 568×, 128 d all arithmetically
   correct (premise caveat = F6).

---

## Round 2 — 2026-08-24, N=10k production results

Subagent pass launched against the production window, the in-gap claim, the
localization fits, the dedup, the 133-vs-139 tension and the hosted-solver
internals. **Both reviewer agents terminated early on an API session limit**,
leaving only fragments ("Independent verification confirms the structure
defect is real"; "N=1000 has BOTH exact eigenvalues and KPM moments at the
same grid — a zero-assumption validation of the bias model"). No reviewer
report exists; the fragments are recorded as *leads*, not findings. Both
leads were then chased directly, by hand:

### R2-F1 — MAJOR, CONFIRMED BY OWN MEASUREMENT: box-face material seam
hosting spurious in-gap states.

Measured on the rasterized ε(r) (N=10k, 192³, r=0.331836, aspect 1.0):

| shell distance from nearest face | ff |
|---|---|
| d = 0 (outermost voxel) | **0.1975** |
| d = 1 | 0.2144 |
| d = 2 | 0.2201 |
| interior d ≥ 10 | 0.2211 |

An 11% relative material deficit in the outermost shell, caused by the
inherited convention that rods whose *radius* pokes through a face are not
wrapped (documented in `structure.py` from the delivered project, never
before quantified). Consequences measured:

- 3 of 10 in-gap modes peak at box *edges* (two coordinates simultaneously at
  the outermost voxel; ~10⁻⁴ by chance), two of them at the identical voxel.
- Energy fraction in the outer 2-voxel shell (6.1% of volume): λ = 1.8709 →
  18%, 1.8732 → 24%, 1.9297 → 44%, 1.9473 → 42% (3–7× enhancement), versus
  0.6–0.9× for bulk control modes. **Those four are discounted as artifacts.**
- The other six in-gap states (2.5–10% shell energy, ≤1.7×) are bulk-localized
  and remain candidates.

**Fix implemented:** `rasterize_penlike(..., periodic=True)` — minimum-image
voxel wrapping, behind a flag because it is a CONVENTION CHANGE relative to
the reference montage (exactly as the original code comment required).
Verified: outer-shell ff 0.1975 → 0.2177, global 0.22011 → 0.22089, and only
5,528 voxels change (0.078%) — **every one within 3 voxels of a face**, none
in the bulk. All 12 tests still pass.

**Decisive test queued** (`chain_final_20260824.sh` step 1): re-solve the gap
window [1.855, 2.000] on the periodic structure. Boundary states must vanish;
genuine rare-region states must survive. Reported either way.

### R2-F2 — MAJOR, self-caught: a corroboration argument that was invalid.

The report claimed KPM/eigensolver agreement on the in-gap count (11.18 ± 0.58
vs 10) "ruled out an artifact". It does not: the KPM DOS is computed from the
*same rasterized ε(r)* and therefore sees the same seam. It rules out an
**eigensolver** artifact only. Corrected in REPORT_N10K.md, with the flawed
reasoning retained and marked, not deleted.

### Note on scope

The seam is a property of the *inherited montage convention*, so it is also
present in the delivered N=1000 project. There it was harmless in practice
(that gap is empty and its band-edge modes are bulk-localized), but the
finding should be carried back as an open item.

---

## Round 3 — 2026-08-25, full N=10k review (relaunched after the round-2 agents died on a session limit)

Reviewer ran read-only/CPU-only against the production window, the gates, and
the solver internals. Report at `/tmp/adv/FINDINGS.md`. **Four MAJOR findings,
three failed refutations.** All fixes applied same-day; independently verified
before acting on the largest one.

### R3-F1 — MAJOR (CONFIRMED by own fp64 recomputation) — unnormalized Rayleigh quotient biased every eigenvalue

`rr_extract` / `rr_extract_hosted` computed λ = ⟨x,Θx⟩ without dividing by
‖x‖², and SVQB leaves ‖x‖²−1 = **+2.77×10⁻⁵ (128³) / +4.73×10⁻⁵ (192³)**.
Verified: correlation between eigenvalue error and ‖x‖²−1 is **r = 0.969**;
dividing it out improves I1 ground-truth parity from 2.85×10⁻⁵ to
**2.35×10⁻⁷ (121×)**. *Verification trap encountered and recorded:* the same
check done in fp32 returns the wrong sign and magnitude (−3.4×10⁻⁴), because
an fp32 sum over 4.2M terms has its own ~10⁻⁴ error — the measurement must be
made in fp64.

Impact: all reported λ biased high by ~4.7×10⁻⁵ relative; reported residuals
carried a floor of the same size (≈ half the 10⁻⁴ gate budget); **no state
lost** (zero in-window unconverged in every slice); **no physics conclusion
moved** (shift is uniform, 9×10⁻⁵ absolute vs 1.4×10⁻³ level spacing).
Ironically the bug made the method look 100× *worse* than it is.

Fixed: normalized quotient in both code paths; all existing results corrected
retroactively via λ/‖x‖² (`scripts/exp/fix_rayleigh_norm.py`), originals kept
as `window_eigenvalues_raw.npy`. Also: `exp_i1_score.py`'s projection formula
measures ‖x‖²cos²θ, which is why `min_proj²` printed the impossible 1.0000197
— true value 0.999993, so **I1 still passes**, but the report had rounded an
impossible number to "1.0000" and hidden a live diagnostic.

### R3-F2 — MAJOR — KPM counts quoted with stochastic-only error bars; completeness never certified

Jackson smoothing biases interval counts by (σ²/2)ρ′ per edge; both window
edges sit on steep DOS shoulders and both bias **upward**: **+7.5 ± 0.3** for
the full window vs the ±2.8 stochastic bar quoted. Calibrated with zero grid
confound on N=1000 (exact spectrum + KPM moments at the same 128³): six nested
windows, positive error in all six. Consequences: the "69 ± 1.5 predicted vs
69 found" agreement was a **coincidence**; the 133-vs-139 tension **dissolves**
(and is equally not proof of completeness); the ledger's only I2 entry records
`pass: false`. Corrected in the report; I7/I9 ledger entries added (they had
been claimed PASS/DONE in the report without ledger records — against the
pre-registration's own rule).

### R3-F3 — MAJOR — a "rare-region candidate" that is an extended mode

λ = 1.9441 has ξ = 12.98 µm (**above** the 12.32 µm ceiling), r² = 0.325,
1.94 decades of decay, participation 0.56% (6–16× every other in-gap state);
the pipeline flags it `unresolved`. It was listed as a bulk-localized
rare-region candidate — the opposite of what the data says. Removed; in-gap
tally now **four seam artifacts, one extended, five candidates**. The quoted
in-gap ranges (ξ 1.8–2.1 µm, pr 0.03–0.09%, r² ≥ 0.96) were stale — written
when only the five S_below states existed, never updated after the gap slice.
Also fixed: "119 of 130 resolved" (stale pre-merge count) → 121 of 133, and
the conflation of "ceiling-limited" with "non-exponential" (only 1 of the 12
exceeds the ceiling; 11 fail the r² gate).

### R3-F4 — MAJOR (reproducibility) — the headline 133 was not reproducible from the documented command

`merge_slices.py` defaulted to `--dedup-rtol 1e-6`, but the two real
duplicates differ by 1.42×10⁻⁶ and 1.73×10⁻⁶ — so the README command yields
**135**, not 133. Default changed to 5e-6 (still 14× below the smallest
genuine spacing, 6.8×10⁻⁵) with the reasoning in-code. Separately: the script
is **untracked in git** — flagged to the user; committing awaits their say-so.

### Failed refutations (= confirmations)

1. **Hosted-basis solver** (used for every production run): ragged chunk
   boundaries, transposition, dead-row handling and rank deficiency all
   attacked at two grids × four block sizes. Matches the device path to fp32
   roundoff (λ to 2.9×10⁻⁷, overlaps 1.000000); dead-row accounting identical.
2. **Cross-slice dedup**: exhaustive scan within 1e-3 relative found exactly
   the two duplicates removed, no others. (Latent fragility recorded: the
   hard λ-window would miss a duplicate reproduced at I1-level accuracy.)
3. **Band numbering**: verified against MPB's own output, not project prose —
   MPB band = 0-based index + 3, to 7×10⁻⁷. Montage labels correct. Caveat
   added: N(1.757) is itself Jackson-biased low by ≈4 states, so the absolute
   index point estimate is biased, not merely uncertain.
4. **ξ measures hot-spot spacing, not decay** — refuted: spacing is flat at
   ~1 µm across all modes while ξ ranges over 7×. (But F7: the funnel
   *amplitude* is inflated ~2× by multi-blob structure; trend survives.)

*(Later rounds appended below as they run.)*

---

## Round 4 (2026-08-31) — final pass against the finished report

Fresh reviewer, given rounds 1–3 to avoid re-reporting, and pointed at the
newest and least-reviewed material: the I6 closure claim, the rare-region
statistics, gate arithmetic recomputed from raw data, and internal
consistency after five days of heavy editing.

### MAJOR

**R4-M1 — the rare-region |z| test is refuted; its caveat had the sign backwards.**
`|z| = 0.57` (candidates) vs `0.27` (controls) reproduces exactly, and on raw
|z| a permutation test gives p = 1.6e-4. But the comparison was never
extent-matched: controls were the six lowest and six highest window modes,
i.e. the MOST EXTENDED in the set (participation 0.72–12.1%), against the MOST
COMPACT candidates (0.034–0.33%) — a **36× median volume difference**
(independently verified here). Energy-weighting a coarse-grained field is a
shrinkage estimator, so wider modes must give smaller |z| with no anomaly
present. Against a translation null (each mode moved rigidly to all 192³
lattice sites, shape and extent fixed) the ordering REVERSES: |z|/σ_null is
2.77 for candidates vs 7.19 for controls, permutation p = 0.999 in the claimed
direction, and 3 of 5 candidates are not significant against their own null
(p = 0.06–0.09). The report's caveat claimed energy-weighting "understates the
anomaly"; it in fact **manufactures** the ratio. The band-character sign split
fails identically — all six low-λ controls z>0 and all six high-λ z<0 at
p ≤ 0.0025, a whole-window effect. RETRACTED in the report.
Untouched: the finite-size *statistics* argument (matched local distributions,
10× draws, deeper tail, P(zero at N=1000) = 0.61) never used this statistic.

**R4-M2 — "the resolution confound is CLOSED" is an overclaim; its statistic is saturated.**
Retention of a 256³ mode inside the 192³ k-set is 99.98%, but measured against
coarser cubes a hypothetical **96³ grid (half production resolution) still
retains 99.82%** (verified here), and the figure is flat at 99.975–99.984%
across all 14 modes, in-gap and band-edge alike. A statistic that does not vary
with mode character is not evidence about mode character; it also measures
band-limiting of E while the confound concerns ε(r) (the grids do rasterize
different dielectrics, ff 22.011% vs 22.000%). Counter-evidence already sat in
the ledger: mean inter-grid shift = 3.2× the smallest level spacing, 6 of 11
sorted pairings wrong, overlaps down to 0.873. The I6 **gate still passes** on
its registered per-band criterion (0.338% / 0.039% vs 0.6%); the "closed"
sentence is WITHDRAWN.

**R4-M3 — for λ = 1.94721, "vanished" is not separable from "not converged", and the ledger applies opposite standards to one pair of states.**
Overlap budget Σ_j|<m,p_j>|² over the found periodic set: 1.87076 → 0.0001,
1.92960 → 0.0087, 1.87308 → 0.0201, but **1.94721 → 0.0971**. Periodic state
1.94067 has budget 0.9984, of which 0.9969 lies inside
span{montage 1.94405, 1.94721} — one converged periodic vector in that 2D
subspace, the orthogonal direction unaccounted for, and I2 independently found
~2 states missing in the interval containing it. Meanwhile the I6 cross-grid
entry applies a 2D-subspace argument to *this same pair* to excuse overlaps of
0.879/0.873. Both cannot be right. Verdict for 1.94721 is now **undetermined**;
the other three seam states are unaffected (budgets ≤ 0.02, nothing between
0.30 and 0.95, so the 0.5 threshold does no work there).

### MINOR (all corrected)

m1 unannotated stale residuals inside the I3 entry's prose (conclusion survives:
1.94721 corrected 2.787e-5 vs median 3.251e-5); plus a superseded 130-mode
localization file at results/ top level.
m2 §0 per-slice residuals: S_above is 5.87e-5, not 7.4e-5 (nominal δ applied
instead of the per-vector one) — as written it exceeded §3's merged worst.
m3 fig_dos_spectrum caption restated the retired KPM-agreement claim.
m4 in-gap ξ range self-contradicting: asserted 1.8–2.1 µm while its own
parenthetical called that stale. Correct: 1.80–2.51 µm (candidates).
m5 "Nine residual-certified eigenstates" listed nine λ for a set of ten
(1.8690 omitted); ten elsewhere in three places.
m6 stale header ("one gate still running"), stale I4 row ("above-gap running"),
pre-correction window endpoints.
m7 "130 residual-certified eigenpairs" against a 69+61+5 = 135 → 133 list.
m8 unresolved-mode partition: all 12 fail r² < 0.7 and one ALSO exceeds the
ceiling; the criteria overlap rather than partition.

### Attacked and HELD (refutation failed)

1. **The cross-grid k-space embedding.** Both bases constructed and compared
   directly: `max|t_192 − t_256,sub| = 0.0` and `max|kn_192 − kn_256,sub| = 0.0`
   over all 7,077,888 coarse k-points — bit-identical frames, no sign flips, no
   differential triggering of the reference-axis switch. The overlaps are
   well-defined. (This was the assumption I reasoned from code but never
   verified; it holds.)
2. **I6 gate arithmetic** reproduces to five digits (0.33785% / 0.03920%).
3. **I2's certifying sub-interval.** Attacked as a difference of large numbers —
   wrong: leak is a direct piecewise integral. Robust to the leakage model
   entirely (with leak = 0 the number is 0.0014 vs a 0.5 gate), and the
   deflation premise cannot manufacture a pass.
4. **I4 and I8 arithmetic** reproduce (210/210, 4.11e-7, proj² 0.9961; I8 max
   0.0868% → "0.09%").
5. **Seam classification and periodic bookkeeping**: shell fractions, the five
   candidates' periodic counterparts, 10 → 7, S_gap 5 → 3, and the
   1.98401 ↔ 2.00879 band-edge reassignment all reproduce.
6. **The "8 PASS / 3 FAIL" header tally** matches the ledger exactly.
7. **Localization counts** (121 of 133 resolved) reproduce.

Raw artifacts: results/gates/zperm.json, results/gates/retention.json.
