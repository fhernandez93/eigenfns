# Interior gap-edge eigenmodes of the N=10,000 LSU network on a laptop GPU

**Project report — 2026-08 · RTX 4080 Laptop (12 GB), 62 GB RAM ·
env `lsu_ml` (JAX 0.10.0 cuda12)**

**Status (2026-08-28): the eigenmodes are delivered and certified; one gate
is still running.** 401 residual-certified eigenpairs are on disk with their
ε|E|² fields — 133 for the N=10k gap window at 192³, 45 at 160³, 216 across
two N=1000 interior slices, 7 for the periodic re-solve. Accuracy against
exact bottom-up ground truth is verified twice independently: **210/210
targets at Δλ/λ ≈ 3×10⁻⁷ with zero ghosts and zero missed** (I1, I4). Of the
nine registered interior gates, **8 entries PASS, 3 FAIL with recorded
diagnoses, 3 remain open**; I6 (grid convergence) is the only one still
waiting on computation. Every failure below is reported as a failure — the
pre-registration's rule was that a FAIL plus an honest explanation beats a
massaged PASS, and three of them are exactly that.

Companion records: `plans/2026-08-17_interior_investigation.md` (Phase 1,
all measurements), `plans/2026-08-17_interior_literature_survey.md`,
`plans/2026-08-18_interior_preregistration.md` (frozen + Amendment A1),
`plans/2026-08-17_adversarial_verification.md`, `results/gates/gate_results.json`.

---

## 0. Executive summary

We extended the delivered N=1000 solver to compute **interior gap-edge
Maxwell eigenmodes of the N=10,000 disordered LSU network** (periodic cube
L = 24.647 µm, ~5,000 bands below the window) on the same 12 GB laptop GPU —
a regime where the validated bottom-up solver is doubly infeasible (measured
extrapolation: 1.38 TB locked-set storage = 12× RAM+disk; O(100) days).

- **New decoration** (this project): circular rods (aspect 1.0), n = 2.9
  exactly (ε = 8.41), radius bisected to measured ff = 22.0%:
  **r = 0.331836 µm → ff = 22.000% (256³), 22.011% (192³)** (gate I7 PASS).
- **The spectrum**: full-bandwidth KPM DOS (validated first against the 611
  exact N=1000 eigenvalues) locates the N=10k gap and pins the mid-gap state
  count at **5010.3 ± 12.5 — the 0.5 bands/vertex rule confirmed by
  measurement** (gap between MPB bands ≈5012|5013 ± 13). KPM gap width
  0.13 ± 0.05 (criterion-dominated), Δν/ν ≈ 3.4% [2.2, 4.8]%. **Exact edges
  from the eigensolver** (these supersede the KPM criterion edges): the
  largest interior spacing in the certified 133-mode window runs from
  λ = 1.8860 to 1.9264, and the ten states inside the KPM bracket
  [1.864, 1.996] mean the "edges" are not a clean pair — the gap is a
  pseudogap populated by localized states, so §2 reports the state list
  rather than an edge pair.
- **Method** (chosen by measured bake-off, not opinion): **two-stage bandpass
  Chebyshev filtered subspace iteration** — build at degree ~3000, polish
  with the same filter at degree 8000–12000 on the trimmed basis; extraction
  exclusively by Rayleigh–Ritz on the original Θ + per-pair residual gate
  (≤ 1e-4) + cluster-projection checks. Ghosts are structurally excluded:
  **zero ghosts observed across every run of every method in the project.**
  Measured failures (recorded findings): folded spectrum (no preconditioner
  grips an ε-structure-dominated low spectrum; the ESCAN kinetic-diagonal
  analogy does not transfer), shift-invert PMINRES (inner solves
  tolerance-starved — reproduces Canning et al.'s verdict), and a
  LOBPCG-style expansion-RR polish (interior-Ritz ghost instability at scale,
  5 instrumented variants).
- **Ground-truth parity (gate I1, PASS)**: the production configuration
  reproduces a 50-band interior slice of the validated N=1000 spectrum
  (indices 473..522 straddling the gap) at **50/50 found, max Δω/ω =
  2.4×10⁻⁷ (median 6.3×10⁻⁸), 0 ghosts** — after Amendment A1 (polish cap
  4→6; a near-degenerate pair, Δλ = 7×10⁻⁴, needed outers 5–6; the first-run
  49/50 is recorded) and after the **Rayleigh-normalization correction
  below**, which is what took this figure from 2.8×10⁻⁵ to 2.4×10⁻⁷.
  Properly computed cluster projection² = 0.999993 (gate 0.99).
- **CORRECTION (2026-08-25, adversarial round 3, F1) — every eigenvalue was
  biased high by a missing normalization.** `rr_extract` reported the Ritz
  value as ⟨x,Θx⟩ without dividing by ‖x‖², and SVQB leaves ‖x‖² off unity
  by **+2.8×10⁻⁵ at 128³ and +4.7×10⁻⁵ at 192³** — blocked fp32 accumulation
  over millions of positive terms under-estimates the Gram diagonal, so the
  vectors come back slightly *over*-normalized. Verified independently in
  fp64: the eigenvalue error correlates with ‖x‖²−1 at r = 0.969 and dividing
  it out improves ground-truth parity **121×**. Note the sanity trap: the
  same check in fp32 gives the wrong sign and magnitude, because an fp32 sum
  over 4.2M terms carries its own ~10⁻⁴ error.
  - *Code fixed* (normalized quotient in both `rr_extract` and
    `rr_extract_hosted`).
  - *All existing results corrected retroactively* — λ_corrected =
    λ_raw/‖x‖² is exactly the Rayleigh quotient of the same saved vector, so
    no re-solve was needed (`scripts/exp/fix_rayleigh_norm.py`; originals
    kept as `window_eigenvalues_raw.npy`, norms as `window_norms.npy`).
  - *Effect on physics: none.* The shift is uniform at −4.7×10⁻⁵ relative
    (≈ 9×10⁻⁵ absolute at λ≈1.94) against a 1.4×10⁻³ median level spacing
    and a 0.13-wide gap. Gap widths, spacings, localization and DOS
    comparisons are unmoved.
  - *Effect on the residual gate:* reported residuals carry a floor of the
    same size (r_reported² = r_true² + δ²), so the true worst residuals are
    ≈ 4.4×10⁻⁵ (S_below) and 7.4×10⁻⁵ (S_above), not the quoted values —
    roughly half the 10⁻⁴ budget was being spent on this bug. **No state was
    lost to it**: all three slices report zero in-window unconverged pairs.
- **Cross-solver check on the new decoration (gate I4)**: bottom-up
  (validated machinery) solved N=1000/circular/ff22 at 128³ (8,876 s):
  gap again exactly at MPB 500|501, **Δν/ν = 5.07%** — the circular-rod
  decoration more than doubles the elliptical production gap (2.08%).
  **Interior-vs-bottom-up comparison: DONE and PASSING on both sides of the
  gap.** The interior solver reproduced the same structure from scratch —
  below-gap 107/107 converged, above-gap 109/109, zero in-window unconverged
  in either — and scored against the bottom-up reference gives
  **210/210 targets, 0 missed, 0 ghosts, max Δλ/λ = 4.11×10⁻⁷, min proj²
  0.9961**. Six further modes lie outside the range where the reference
  stores vectors and were checked on eigenvalues alone (max 1.87×10⁻⁷).
  This is also independent confirmation of the round-3 Rayleigh-norm fix: a
  fresh solve against a different reference lands at ~3×10⁻⁷, the corrected
  figure, not the 2.83×10⁻⁵ the unnormalised quotient produced.
- **N=10k production** (192³, window λ ∈ [1.757, 1.930] ∪ [1.980, 2.117],
  plus the Amendment-A2 gap-covering slice [1.925, 1.985]):
  **COMPLETE — 130 residual-certified eigenpairs**: S_below 69/69 (KPM
  predicted 69 ± 1.5; worst res 6.4×10⁻⁵; 6.87M Θ-apps; 63.3 h) and S_above
  61/61 (predicted 66 ± 2.4; worst res 8.7×10⁻⁵; 4.20M Θ-apps; 37.5 h) and
  the Amendment-A2 gap slice 5/5 (3.6 h); zero in-window pairs left
  unconverged in any slice. After cross-slice dedup: **133 distinct
  residual-certified eigenpairs, λ ∈ [1.75713, 2.11667]**, bulk median level
  spacing 1.35×10⁻³.
- **Internal accuracy check (no ground truth exists at N=10k).** Two states
  were found independently by two slices — different filter windows, degrees,
  and random starts. They agree to **Δλ/λ = 1.4×10⁻⁶ and 1.7×10⁻⁶** with
  eigenvector overlap **0.9988**, and were then deduplicated by the
  registered overlap rule. Honest caveat (round 3, F5): both solves carried
  the *same* normalization bias, which cancels in the comparison, so this
  measures reproducibility, not absolute accuracy — the absolute figure is
  the I1 parity above.
- **Physics finding — the gap is populated by localized states, not empty.**
  Nine residual-certified eigenstates lie inside the KPM 10%-criterion gap
  bracket [1.864, 1.996]: λ = 1.8709, 1.8732, 1.8861, 1.9265, 1.9297,
  **1.9441, 1.9473, 1.9739** (the last three from the dedicated gap slice,
  straddling gap centre 1.957) and 1.9902 — separated by spacings up to 80×
  the bulk spacing, with **ξ = 1.8–2.1 µm and participation fractions
  0.03–0.56%** (envelope-fit quality r² = 0.33–0.998 — see the correction on
  λ = 1.9441 below; the "1.8–2.1 µm / 0.03–0.09% / r² ≥ 0.96" ranges quoted in
  an earlier draft described only the five states known before the gap slice
  ran and were stale). The N=1000 network
  with the SAME decoration has a clean empty gap (bottom-up verified across
  611 bands, Δν/ν = 5.07%) — so these are large-box rare-configuration
  (Lifshitz-tail-type) states: the disorder physics that only appears once
  the box is big enough to contain rare local environments. This is the
  finite-size effect the N=10k computation was built to expose. The
  registered gate I5 clause "gap empty of converged pairs" accordingly
  **FAILS as registered** and is reported as this finding (Amendment A2,
  recorded before the gap slice ran); the measured in-gap DOS floor
  ≈ 60 states/unit-λ is consistent with the discrete count.
- **TERMINOLOGY (2026-08-25).** A frequency range containing states is **not a
  spectral gap** — a spectral gap means zero DOS, and this one is not empty.
  "Gap" is used throughout this report as shorthand for the *nominal* gap
  (the KPM 10%-criterion bracket [1.864, 1.996]) and should be read as
  **pseudogap with localized in-gap states**. The stronger term *mobility
  gap* is deliberately **not** claimed as measured: mobility is a transport
  statement, and what was computed is Γ-point modes of a finite periodic
  supercell, with no transport calculation anywhere in this project. What is
  measured is ξ = 1.80–2.51 µm in an L = 24.65 µm box (ξ/L ≈ 0.08), which
  makes the corresponding supercell defect bands flat and justifies treating
  these as isolated localized states — an inference from ξ ≪ L, not a
  measurement of mobility.
- **Is the in-gap DOS visible independently of the eigensolver?** Partly.
  KPM never asks the solver what it found, so it is the one completeness-
  independent handle — but the Jackson kernel smooths the band edges *into*
  the gap, so even a hard-zero gap reports a nonzero count. Measured
  (`scripts/exp/exp_gap_leakage.py`): convolving a hard-zero gap of the
  measured width with the measured kernel leaks only **0.3–1.9 states** into
  S_gap = [1.925, 1.985] (1.4–3.2 in the adversarial limit where the true gap
  is taken to be no wider than S_gap itself), against a measured KPM count of
  **4.87 ± 0.41**. So there is a genuine **~3–4.5-state excess** over edge
  leakage: the in-gap DOS is not a smoothing artifact. **But this does not
  make it physics** — KPM consumes the *same rasterized ε(r)*, seam included,
  so it counts seam states just as happily as bulk ones. It establishes that
  states are really there in the structure as rasterized; only the periodic
  re-solve can say whether they survive the fix.
  - **SUPERSEDED IN PART (2026-08-26).** The re-solve landed, so the numbers
    above must be read as **pre-fix**. S_gap held five montage states, and the
    overlap test proved **two of them (1.9296, 1.9472) were seam artifacts**;
    the periodic run finds **three** states in S_gap (1.94067, 1.97081,
    1.98401). So the eigensolver count this KPM excess was compared against
    has dropped 5 → 3. The KPM moments were computed on the seam-contaminated
    ε(r) and **a post-fix KPM run does not exist** — it is a full-bandwidth
    12,000-degree job and is not queued. The honest position: the
    ~3–4.5-state excess is a measurement on the *unfixed* structure and is
    **not** evidence for the post-fix in-gap count. Treat it as retired
    pending a periodic KPM run.
- **VERDICT (2026-08-26) — the seam test came back, and the registered
  prediction is CONFIRMED.** The gap window [1.855, 2.000] was re-solved at
  192³ against the periodically-wrapped structure (20.0 h, 7 converged pairs,
  1 in-window unconverged at λ = 1.9095, reported not dropped). Identity
  between the two runs is decided by **eigenvector overlap > 0.5 — the same
  rule as cross-slice dedup — not by a λ window**, and that distinction
  matters: `exp_periodic_verdict.py` matches with `tol = 2e-3` absolute,
  which is *smaller than the real physical shift*, so it mis-called four
  persisting states as "vanished (unexpected)" and their moved counterparts
  as "new". Widening the tolerance to fix that would have been gate-weakening;
  `exp_periodic_match.py` decides on physics instead.
  - **The four seam-flagged states are gone.** Their best overlap with *any*
    periodic state is **0.006, 0.13, 0.09, 0.30** — all below the rule. They
    were rasterization artifacts, as diagnosed.
  - **All six bulk states persist**, at overlap **0.95–0.9997**, shifted by
    Δλ = −0.0007…−0.0034. **Every shift is negative**, which is the sign
    required: wrapping *adds* the dielectric missing from the outer shell, and
    more dielectric pushes frequencies down. A physical shift of the right
    sign and a near-unit overlap is what "the same state, slightly perturbed"
    looks like.
  - **So the five rare-region candidates survive the fix** (λ = 1.8690,
    1.8860, 1.9264, 1.9738, 1.9901 → 1.8683, 1.8853, 1.9242, 1.9708, 1.9879),
    as does the one extended mode (1.9441 → 1.9407), which remains extended
    and remains excluded from the candidate list.
  - **The one apparently-new periodic state is not new.** Extending the
    overlap scan to all 133 montage modes (`--full`) finds 1.98401's partner
    at **2.00879, overlap 0.9149, Δλ = −0.02478** — a state that sat *above*
    the gap and was pulled **into** it by the added dielectric, shifting ~10×
    further than the localized ones, as a delocalized band-edge state should.
    So the gap edge moved inward under the fix. Usefully, **all seven periodic
    states have a montage partner above the rule**, so the montage run missed
    nothing the periodic run found — a small completeness datapoint that does
    not depend on I2.
  - Net in-gap population in the KPM bracket [1.864, 1.996]: **10 → 7** (four
    seam artifacts removed, one band-edge state pulled in).
  - **What this does NOT settle**: the resolution confound. This test changed
    the boundary convention at fixed 192³; it says nothing about whether
    7.79 vox/µm is enough. The 256³ anchors (I6) remain the open question,
    and completeness (I2) still bounds nothing.
- **Why does N=1000 have a clean gap and N=10k not?** Measured, not asserted
  (`scripts/exp/exp_rare_regions.py`, `exp_rare_region_modes.py`, CPU only).
  The two networks are density-matched with L scaling exactly as N^(1/3)
  (24.6467/11.4405 = 2.1543 = 10^(1/3)), and their **local statistics
  coincide**: filling fraction coarse-grained at the ξ ≈ 2 µm scale gives mean
  0.2204 vs 0.2209 and sd 0.0398 vs 0.0405 (ratio 1.018), with matching
  percentiles from 1% to 99%. What differs is the number of draws — 187 vs
  1871 independent ξ-cells — and therefore the **tail reach**: the extremes go
  −4.84σ…+4.00σ at N=1000 but −5.52σ…+4.51σ at N=10k. That is the rare-region
  (Lifshitz-tail) picture in one line: same physics per unit volume, ten times
  the chance of containing a gap-filling configuration. The Poisson arithmetic
  makes the contrast unremarkable — if N=10k truly holds 5 in-gap states, the
  rate is 0.5 per N=1000-volume and **P(zero at N=1000) = 0.61**, the single
  most likely outcome. No new physics is needed to explain the difference.
  Corroborating: with the same decoration the N=1000 gap is [1.8276, 2.0225]
  (Δν/ν = 5.07%) and the N=10k nominal gap is [1.864, 1.996] (3.4%) — the gap
  **narrowed by 32% and every N=10k in-gap state lies inside the N=1000 gap**,
  i.e. the gap did not vanish, its edges frayed inward, which is what band
  edges set by extreme-value statistics are supposed to do.
- **Direct test of the mechanism — where do the in-gap modes live?**
  Energy-weighted local ff, z-scored against the ξ-coarse-grained field:
  the five candidates reach **|z| = 0.57 (max 0.70), 2.1× the bulk controls'
  0.27**, and they split by band character exactly as the picture requires —
  the low-λ ones sit in dielectric-rich regions (z = +0.41…+0.64, peeled off
  the dielectric band) and the high-λ ones in air-rich regions (z = −0.61,
  −0.70, peeled off the air band). Independent corroboration of the seam
  classification falls out for free: **the four seam states score |z| = 0.22,
  statistically indistinguishable from the controls' 0.27** — they are not in
  anomalous material, which is what a boundary artifact should look like.
  **Honest weight**: a factor 2.1 is suggestive, not decisive. Energy-weighting
  over a mode whose extent (ξ ≈ 2 µm) matches the coarse-graining box regresses
  z toward zero, so this understates the anomaly, but the measurement as it
  stands does not by itself establish rare-region origin.
- **Two mundane explanations remain open** and are exactly what the queued
  runs test. (1) **Resolution**: N=10k runs at 7.79 vox/µm against N=1000's
  11.2 — 70% — and gate G5 has already failed once on convergence, so coarser
  rasterization manufacturing in-gap states is not excluded; the 256³ anchors
  address it. (2) **The seam**, which already accounts for 4 of the 10; the
  periodic re-solve settles it. Until both land, the finite-size explanation
  is the leading one, not the established one.
- **CORRECTION (2026-08-24) — four of the ten in-gap states are a
  rasterization artifact, not physics.** Self-audit of where the in-gap modes
  live (prompted by an adversarial pass) found three of them peaking at box
  *edges* — two coordinates simultaneously at the outermost voxel, a ~10⁻⁴
  coincidence. Direct measurement of the rasterized ε(r) confirms the cause:
  **the outermost voxel shell has ff = 0.1975 vs 0.2211 in the interior, an
  11% material deficit**, produced by the inherited convention that rods whose
  *radius* pokes through a face are not periodically wrapped (documented in
  `eigenfns/structure.py` since the delivered project). That thin seam acts as
  a planar defect in a gapped medium and hosts defect states. Energy fraction
  in the outer 2-voxel shell (6.1% of volume): modes at λ = 1.8709, 1.8732,
  1.9297, 1.9473 carry **18%, 24%, 44%, 42%** (3–7× enhancement), versus
  0.6–0.9× for bulk control modes. **Those four are discounted.**
- **SECOND CORRECTION (round 3, F3) — one more of the survivors is not a
  localized state at all.** λ = 1.9441 has ξ = 12.98 µm (*above* the 12.32 µm
  ceiling), r² = 0.325, only 1.94 decades of decay, and a participation
  fraction of 0.56% — 6–16× every other in-gap state. The project's own
  pipeline flags it `unresolved`. It is an extended, non-exponential mode, and
  calling it a rare-region candidate was the opposite of what the data said.
  **Removed from the candidate list.** That leaves **five** bulk-localized
  in-gap candidates (λ = 1.8690, 1.8860, 1.9264, 1.9738, 1.9901; shell
  fractions 2.5–10%, ξ = 1.80–2.51 µm, r² = 0.971–0.998) — pending the
  decisive test below. Of ten in-gap states found: four seam artifacts, one
  extended, five candidates.
- **CORRECTION (round 3, F2) — every KPM count in this report carried a
  stochastic-only error bar, with a larger systematic sitting on top.** The
  Jackson-damped estimator returns the *smoothed* counting function, so an
  interval count is biased by (σ²/2)·ρ′ at each edge. Both window edges sit on
  steep DOS shoulders (ρ ≈ 1286 below, 995 above; ρ′ = −1.8×10⁴, +1.1×10⁴) and
  **both push the count up**: predicted bias for the full window **+7.5 ± 0.3
  states**, against the ±2.8 stochastic bar that was quoted. Calibrated where
  no grid confound exists — N=1000 has exact eigenvalues *and* KPM moments at
  the same 128³ grid — six nested gap-straddling windows all show positive
  error (+2.2…+4.0), as the theory requires. Consequences, plainly:
  - "S_below 69 ± 1.5 predicted vs 69 found" was presented as a validating
    agreement. Under this bias model the prediction is ≈64.6, so **that
    agreement was a coincidence**, not a validation.
  - The **133-vs-139 tension dissolves**: the 6.3 deficit is the size of the
    +7.5 bias, so it is not evidence of missing states — and equally not
    evidence of completeness. Per-slice arithmetic under the same model is
    inconsistent at ±3–4, forbidding any point estimate either way.
  - **Nothing in this project yet certifies completeness.** The only I2 entry
    in the ledger records `pass: false` (the mis-designed v1 estimator,
    ±26 bands); the Amendment-A3 v2 estimator has not yet run. Earlier drafts
    implied a completeness they had not earned.
  - The in-gap count "11.18 ± 0.58 predicted vs 10 certified" carries the same
    bias (≈ +1 at that bracket) and is consistency, not proof.
- **A corroboration that does NOT discriminate (retracted reasoning).** An
  earlier draft argued that KPM agreement (11.18 ± 0.58 predicted vs 10
  certified) ruled out an artifact. That is wrong in an important way: the KPM
  DOS is computed from *the same rasterized ε(r)*, so it sees the same
  boundary seam. It rules out an **eigensolver** artifact — the states really
  are eigenstates of the operator we built — but says nothing about whether
  the operator faithfully represents the intended structure. Recorded rather
  than deleted, because the distinction is the whole point of the audit.
- **Decisive test (queued):** re-rasterize with periodic rod wrapping
  (`periodic=True`, new flag — the convention change the delivered project
  deliberately deferred), verify the shell deficit is gone, and re-solve the
  gap window. Boundary states must vanish; genuine rare-region states must
  survive. Result reported either way.
- Corroboration that stands: the gap-edge decay lengths agree across box
  sizes — ξ ≈ 1.5–2.3 µm (N=1000, L=11.44) vs 1.8–2.1 µm (N=10k, L=24.65).
- **ξ(ω) across the full window**: **121 of 133 modes have resolved decay
  lengths.** Of the 12 not resolved, **only one exceeds the ceiling**
  (ξ_max = L/2 = 12.32 µm); the other 11 fail the r² < 0.7 gate, i.e. their
  envelopes are not exponential — a different statement from "ceiling-limited
  lower bound", and the earlier draft conflated the two (and quoted 119/130,
  the stale pre-merge count). The trend is monotone and symmetric about the
  gap — median ξ by band: 5.00 µm (λ 1.757–1.80) → 3.01 → 1.91 µm at the
  lower gap edge, then 2.75 → 4.71 → 6.03 µm climbing away above.
  **Amplitude caveat (round 3, F7):** shrinking the fit range barely moves the
  compact gap-edge modes (10–15%) but collapses the window-edge ones by 2–3×,
  the signature of far radial bins catching *other* hot spots. The
  fit-range-robust contrast is therefore ≈2.4×, not the ≈5.8× the raw numbers
  imply; an independent PR-based estimator confirms the funnel shape (3–6×)
  at a different absolute scale. The *trend* is solid, the three-figure
  absolute ξ of extended modes is not. Refuted along the way: ξ is not
  measuring hot-spot spacing — that is flat at ~1 µm (the rod scale) across
  all modes while ξ ranges over 7×.
  Contrast the same decoration at N=1000: 168/210 modes ceiling-limited
  at L/2 = 5.72 µm — **the small box cannot resolve the localization the
  large box measures**, which is the quantitative case for the N=10k run.
- **Localization**: IPR + envelope-decay ξ with the finite-size ceiling
  ξ_max = L/2 built into the pipeline (validated on known N=1000 modes; every
  fit above ceiling / below 1 decade of decay / r² < 0.7 is flagged
  "unresolved — lower bound only"). N=1000 new-decoration gap-edge modes:
  ξ = 1.47 µm (band 500) / 2.30 µm (band 501), sharper than the elliptical
  decoration — wider gap, tighter confinement. **N=10k ξ(ω) is measured**:
  121 of 133 modes resolved, a clean funnel from ~8 µm at the lower window
  edge down to 1.8 µm at the gap and back out to ~10 µm above, with the
  remaining 12 flagged (one above the 12.32 µm ceiling, eleven failing the
  r² ≥ 0.7 exponential-shape test — a different statement, and stated
  separately). The pipeline itself is now cross-validated: **gate I8 compares
  ξ from the bottom-up and interior solvers on the same N=1000 modes and finds
  max 0.09% / median 0.01%** disagreement over the 20 gap-edge modes resolved
  in both. Read that for what it is — the two solvers agree on those
  eigenvalues to ~3×10⁻⁷ with near-unit overlap, so this proves the pipeline
  is deterministic and solver-independent; it does **not** validate the
  envelope-fit method, whose own fit-range sensitivity is tens of percent.

## 1. Why new machinery (measured, not asserted)

Bottom-up extrapolation from the production profile (611 bands, 128³,
19,908 s, ~130k Θ-applications): locked-set storage 5,150 × 0.268 GB =
**1.38 TB** (hard stop at 62 GB RAM + 54 GB disk); deflation/orthogonalization
time O(100) days (order-of-magnitude; the n² regime is not yet visible at
N=1000 scale — adversarial review F6); pure matvec alone 16.4 h. The lowpass
ChebSI alternative hits the same 1.38 TB wall.

## 2. Phase 1 measurements (details in the investigation report)

| item | value |
|---|---|
| KPM validation vs 611 exact eigenvalues | window count 213.4±2.6 vs 210; gap edges within smearing; se ≈ 2.7 bands @16 probes |
| N=10k DOS (256³, degree 12k, 12 probes, 8,477 s) | gap ⟨criterion range⟩; mid-gap count 5010.3±12.5; window derivation |
| matvec (ms/vector, chunk 4–8) | 23.2 (192³) · 38.2 (224³) · 53.9 (256³) · 79.2 (288³) |
| λ_max (Lanczos ×1.05) | 2209 (192³) · 3031 (224³) · 3975 (256³) · 5056 (288³) |
| bake-off (N=1000@128³, 50-band slice, ratio 9.4×10⁻⁵) | winner: two-stage bandpass; 24→50 verified pairs; competitors' failure modes measured |

Bake-off cost geometry: the full 317-band window at 256³ projects to 10–15
GPU-days → the pre-registered descope fired **at registration** (192³ +
128-band gap-edge window + I6 anchors at 160³/256³), not mid-run.

## 3. Gates (pre-registered `plans/2026-08-18_interior_preregistration.md`; ledger `results/gates/gate_results.json`)

| gate | status | measured |
|---|---|---|
| I1 ground-truth parity | **PASS** | 50/50 targets, 0 ghosts, worst res 5.18×10⁻⁵. Parity **max Δλ/λ = 2.35×10⁻⁷, median 6.29×10⁻⁸** over 55 pairs (gate 1e-4). The ledger entry still records the *pre-correction* 2.83×10⁻⁵ / min proj² 1.0000203 because the scorer has not been re-run — see the note on that entry. First run 49/50 → Amendment A1 |
| I2 completeness | **PASS — certified** | Certifying sub-interval [1.9063, 1.9606]: **missed = −0.00025 ± 0.00013** (gate \|missed\| < 0.5), i.e. 2000× inside it. Consistency check on the full window (explicitly *not* certifying, per A3): missed = +1.68 ± 0.32, within the acknowledged O(1) bias floor. Leakage term needed three corrections — see §2 |
| I3 residuals / orthonormality | **FAIL** | Residuals PASS: worst 5.88×10⁻⁵, median 3.26×10⁻⁵ (gate 1e-4). **Gram ‖G−I‖max = 4.86×10⁻⁴ vs gate 5×10⁻⁵ — fails by 9.7×**, entirely off-diagonal (diagonal 2.04×10⁻⁵). Diagnosed: **created by the merge, not the solver** — cross-slice max 4.86×10⁻⁴ vs same-slice max 8.58×10⁻⁶, so *each slice individually passes*. Dominated by λ=1.94721 (a seam artifact) in 6 of the 10 worst pairs. Two of my own explanations refuted by measurement; the specific mechanism is recorded **open** |
| I4 new-decoration cross-check | bottom-up DONE; interior below-gap DONE, above-gap running | Bottom-up: gap 500\|501, Δν/ν 5.07%, 8,876 s. Interior below-gap slice: **107/107 converged, 0 in-window unconverged**, worst res 9.57×10⁻⁵, λ ∈ [1.47016, 1.82759], 14,259 s |
| I5 spectrum consistency | **FAIL as registered** | `empty_gap_clause_pass: false` — ten converged pairs inside the KPM bracket [1.864, 1.996]. This is the clause failing exactly as Amendment A2 recorded *before* the gap slice ran, and it is the physics finding of §2, not a defect |
| Seam test (periodic rasterization) | **PASS** | Registered prediction confirmed: the four seam-flagged states have max overlap 0.006/0.13/0.09/0.30 with any periodic state (gone); all six bulk states persist at overlap 0.95–0.9997 with Δλ = −0.0007…−0.0034, every shift negative as added dielectric requires |
| I6 convergence (160³/192³ + 256³ anchors) | **PASS** | Per-band Δω/ω across grids, paired by **eigenvector overlap** (not sorted eigenvalue — at the low edge 6 of 11 sorted pairings were wrong, so a sorted comparison would have reported meaningless scatter). Low edge [1.84, 1.95]: 11 of 11 matched, |Δω/ω| max **0.338%**. High edge [1.99, 2.035]: 10 of 11, max **0.039%**. Both inside the registered 0.6% bound and consistent with the ~0.3% expectation. **The resolution confound is closed**: each 256³ mode keeps **99.98%** of its power at wavenumbers 192³ can represent, so the in-gap states are not rasterization artifacts. Caveats on the ledger: the one unmatched state (λ=1.99012) sits 1.2×10⁻⁴ above the window floor, inside the filter transition zone; the high-edge anchor certified only 3 of 11 pairs, the other 8 used as uncertified vectors for the overlap test only; the 160³ leg is edge-truncated to [1.8096, 2.0516] |
| I7 decoration | **PASS** | ff = 22.011% at 192³ (gate 22.0 ± 0.5); r = 0.331836 µm |
| I8 localization | **PASS** | Cross-solver ξ agreement on the 20 gap-edge modes resolved in *both* the bottom-up and interior solvers: **max 0.09%, median 0.01%** (gate ≤ 10%). Sample discipline: 210 of 216 modes matched, 42 resolved in both (the L/2 = 5.72 µm ceiling catches the rest in an 11.44 µm box), 20 of those gap-edge; ceiling-limited fits were *excluded*, since a lower bound cannot be compared. **Read correctly**: the two solvers agree on these eigenvalues to ~3×10⁻⁷ with near-unit overlap, so they hand the pipeline nearly the same field — this proves the pipeline is deterministic and solver-independent, but does **not** independently validate the envelope-fit method, whose own fit-range sensitivity is tens of percent (round 3, item 3). Ceiling ξ_max = L/2 stated on every figure; unresolved fits reported as lower bounds only |
| I9 montage | **DONE** (band-count agreement pends I2; absolute numbering carries the KPM ±13 placement caveat) | N=10k: 133 tiles, 15/row, 9 rows, 5250×3276 (`results/n10k_G192_window/band_montage_n10k_gapedge_15.png`); N=1000-circular: 210 tiles for the finite-size comparison |

**The montage as independent evidence.** Read top-to-bottom it shows the
localization transition directly, with no fitting involved: the first rows
(window bottom, λ ≈ 1.76–1.80) have energy spread over the whole 24.6 µm box
in many weak hot spots; the middle rows (gap edges and in-gap states) collapse
to *single compact blobs* a couple of µm across; the last rows (climbing away
above the gap) spread out again. That is the measured ξ(ω) curve — 5–6 µm →
1.9 µm → 5–6 µm — visible tile by tile, and it agrees with the numeric
envelope fits mode for mode.

## 3a. Figures (`results/figures/`)

| figure | what it shows |
|---|---|
| `fig_dos_spectrum.png` | the 133 converged eigenvalues drawn on the KPM DOS; the gap bracket, and the 10 in-gap states in red. The DOS floor inside the gap is *accounted for by those states* (11.18 ± 0.58 predicted vs 10 certified), not by kernel leakage |
| `fig_xi_omega.png` | ξ(ω) for N=10k and N=1000 side by side, same decoration, each with its own L/2 ceiling. The N=10k panel is a clean funnel — 8 µm → 1.8 µm → 10 µm across the gap, 121/133 resolved. The N=1000 panel shows the same physics unresolvable: 168/210 modes parked at the 5.72 µm ceiling |
| `fig_montage_sbs.png` | the two montages side by side (finite-size comparison, ceiling caveat printed on each) |
| `results/n10k_G192_window/band_montage_n10k_gapedge_15.png` | the N=10k gap-edge montage itself, 133 tiles, 15/row |

## 4. Engineering (the 12 GB discipline, continued)

Every rule below was a measured failure first (logs in `results/exp/` +
investigation §6):

- **Host-resident streamed basis** (`bandpass_subspace_hosted`): at 192³ the
  m=104 basis is 11.8 GB — device-resident OOMs instantly. All stages (filter,
  SVQB, RR assembly, rotation, residuals) stream ≤ 8-vector chunks; measured
  hosted-vs-device equivalence Δλ ≤ 4.5×10⁻⁸; streaming overhead ≈ 1.5× over
  raw matvec time.
- Whole-block operations at m ≥ 56 (128³) all needed chunking: Gram
  conjugate-copies, tensordot rotations, `combine` (peaks at 5 blocks),
  projection updates. A `blocks=[X,W]` list keeps dead blocks alive after
  `del X, W`; caller frames pin call arguments (holder-list pattern).
- MINRES async dispatch piles ~7 live blocks per in-flight iteration without
  periodic sync barriers.
- Two-stage degrees beat one-stage: build (transition ≈ half window) then
  polish (transition ≈ few band spacings) — measured ×12/outer polish damping
  at production vs ×0.62 for an undersized basis at low degree.
- Runs detach (`setsid nohup`) with per-outer rolling checkpoints; every
  interruption this project (5+ external kills, 1 crash, 1 self-inflicted
  race) cost ≤ 1 outer.
- **Don't gate a job queue on GPU *memory*.** A chain that waited for
  `nvidia-smi` free memory launched its next job on top of a running one: the
  running job's memory dipped below the threshold while it was writing a
  4.7 GB checkpoint, and both then contended and one died. Wait on the
  *process table*, and require the all-clear on consecutive samples
  (`scripts/exp/chain_i6resume.sh`). Cost of the lesson: one 13-hour run
  resumed from checkpoint, nothing lost.

## 5. Performance (measured)

| operation | cost |
|---|---|
| Θ apply, 192³ c64 chunk 8 (incl. mapped-filter axpy) | 23.2 ms/vector |
| S_below build outer (m=104, degree 3000) | 10,820 s (1.5× streaming overhead) |
| S_below polish outer (m=104, degree 12000) | 42,602 s |
| I1 slice (N=1000, 128³): build + 6 polish outers + I/O | 22,713 s + 9,930 s completion |
| I4 bottom-up reference (611 bands @128³, new decoration) | 8,876 s |
| N=10k KPM DOS (256³, degree 12k × 12 probes) | 8,477 s |
| N=1000 KPM moments (128³, degree 8k × 16 probes) | 731 s |

## 6. Honest limitations (running list; finalized at delivery)

**The four that matter most, stated first:**

- **I3 FAILS: the merged 133-mode set is not orthonormal to gate.**
  ‖G−I‖max = 4.86×10⁻⁴ against a registered ≤5×10⁻⁵. Diagnosed rather than
  excused: **cross-slice pairs max 4.86×10⁻⁴, same-slice pairs max
  8.58×10⁻⁶**, so *each slice individually passes* — SVQB enforces
  orthonormality within a slice and nothing enforces it across the three
  independent solves that were concatenated. The worst pairs are dominated by
  λ = 1.94721 (a seam artifact) in 6 of the top 10. Two explanations of mine
  were **refuted by measurement**: small eigenvalue spacing (the worst pair is
  Δλ = 0.167 apart) and poor convergence of the offending state (its residual
  ranks 73 of 133, better than median). The actual mechanism is **open**.
  Consequence: use the per-slice sets when mutual orthonormality matters; the
  merged set is fine for spectra, montages and localization, which do not.
- **The seam verdict's negative half is weakened.** "The four seam-flagged
  states vanish under periodic wrapping" rested on their having no partner
  above overlap 0.5 among the periodic solve's **seven** converged states.
  I2 later measured that solve to be **incomplete** (sub-interval missed
  +2.04 ± 0.42; it converged 7 in a window holding ~12), so absence of a
  partner is **not proof of absence**. The seam explanation remains the best
  available — those four states did carry 18–44% of their energy in 6.1% of
  the volume — but the alternative that they persist and were simply not
  found is no longer excluded. A re-solve at m = 48 with 8 polish outers is
  queued. **The positive half is unaffected**: six bulk states have partners
  at overlap 0.95–0.9997 with every Δλ negative, and incompleteness cannot
  manufacture a partner.
- **The full-window completeness estimator is biased and must not be read as
  a state count.** The N=1000 calibration — the cleanest possible test, with
  KPM moments and solve sharing structure, decoration, grid and λ_max — shows
  its leakage model **over-predicts by ≥1.23 ± 0.23 states**. The N=10k
  certification does not depend on it: that came from the sub-interval, where
  true leakage is 0.0017, three orders below this bias. This is exactly why
  Amendment A3 made the sub-interval certifying *before* any of it ran.
- **I6 is now answered and the resolution confound is CLOSED.** 192³ and 256³
  find the *same* states, matched one-to-one by eigenvector overlap, and each
  256³ mode keeps **99.98%** of its power at wavenumbers 192³ can represent.
  Per-band |Δω/ω| is 0.338% (low edge) and 0.039% (high edge), inside the
  registered 0.6%. The in-gap states are **not** rasterization artifacts.
  Residual caveats are on the ledger, not hidden: the high-edge anchor
  certified only 3 of its 11 pairs (the rest enter the overlap test as
  uncertified vectors), one state at the window floor sits in the filter
  transition zone, and the 160³ leg is edge-truncated.
- **The seam re-solve did not lift its caveat.** A second periodic solve at
  m = 48 — 60% more subspace — reproduced the verdict exactly (four
  seam-flagged states with no partner above 0.5; six bulk states persisting at
  0.952–0.9997, every Δλ negative) but found **the same seven states**, not the
  ~2 additional ones I2 said were missing. So the incompleteness is not a
  subspace-size problem, and the negative half of the seam claim — "the four
  vanished" versus "the four were not found" — remains unresolved. Suggestive
  detail: the two solves left *different* λ unconverged (1.9095 vs 1.9534),
  both in or near the certifying sub-interval, which is what several
  hard-to-converge states there would look like. Disclosed against myself: the
  re-solve was stopped after 4 of 8 requested polish outers on a two-outer
  plateau, so later outers are untested.

- **Window descoped from ~300 to ~139 bands** (64/side) by the measured cost
  wall (10–15 GPU-days at 256³ full window); recorded at registration with a
  stretch goal, not discovered mid-run. The full-spectrum picture comes from
  KPM DOS; exact eigenvalues cover the gap-edge window that carries the
  localization physics.
- **Production grid 192³ = 7.79 vox/µm** (70% of the N=1000 production
  resolution); the G5 lesson (gap-edge accuracy is rasterization-limited,
  ~0.3% scatter) applies with more force; I6 quantifies with 160³/192³ full
  comparison + 256³ gap-edge anchors. Sub-percent gap-edge claims are NOT
  made at 192³.
- KPM gap-edge criterion systematic (±0.03 λ) disclosed; eigensolver edges
  supersede KPM edges (I5 closes the loop).
- The ±2 band-numbering ambiguity vs the original cluster convention remains
  (inherited open item; MPB numbering used throughout, flag available).
- N=10k absolute band indices carry the KPM placement uncertainty (±13);
  I2 certifies the window population, not the absolute index. The montage
  tiles are labelled 4942–5074, from N(1.757) = 4939.1 (+3 for MPB's two
  ω = 0 Γ modes and 1-based numbering). **That point estimate is biased low
  by ≈ 4 states** (round 3, F2): the Jackson kernel smooths a steeply falling
  ρ, and the counting bias at a single endpoint is ≈ (σ²/2)ρ′ = (0.0219²/2)
  (−18451) = −4.4. The bias-corrected first index is ≈ 4946, i.e. the labels
  read ~4 low — inside the disclosed ±13, so the labels were left as rendered
  rather than re-cut to a second uncertain point estimate. The tile *count*
  (133) is exact; the absolute numbering is not certified either way.
  The mid-gap count 5010.3 ± 12.5 is **not** affected: ρ′ ≈ 800 inside the
  gap gives a bias of +0.2 states, three orders below the quoted uncertainty,
  so the 0.5-bands/vertex confirmation stands.
- fp16 storage was pre-registered as a contingency but NOT needed at 192³
  (fp32 window artifacts ≈ 20 GB).

## 7. References

As in `REPORT.md` §7 plus the interior-method survey (full citations in
`plans/2026-08-17_interior_literature_survey.md`): Wang & Zunger JCP 100,
2394 (1994); Canning et al. PARA'08; Fang & Saad SISC 34, A2220 (2012);
Pieper et al. JCP 325, 226 (2016) [ChebFD]; Li, Xi, Erlandson, Saad SISC 41,
C393 (2019) [EVSL]; Winkelmann, Springer, Di Napoli ACM TOMS 45, 21 (2019)
[ChASE]; Polizzi PRB 79, 115112 (2009) + Gavin & Polizzi NLAA 25, e2188
(2018) [IFEAST]; Lin, Saad, Yang SIAM Rev. 58, 34 (2016); Weiße et al. RMP
78, 275 (2006); Knyazev SISC 23, 517 (2001); Vecharynski & Yang
arXiv:1602.02306; Szyld, Vecharynski, Xue arXiv:1504.02811; R-ChFSI
arXiv:2503.22652 (2025); Kressner, Ma, Shao Numer. Alg. 94, 1653 (2023).
