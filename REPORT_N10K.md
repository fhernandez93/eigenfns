# Interior gap-edge eigenmodes of the N=10,000 LSU network on a laptop GPU

**Project report — 2026-08 (IN PROGRESS: production solve running; N=10k
results sections marked ⟨pending⟩) · RTX 4080 Laptop (12 GB), 62 GB RAM ·
env `lsu_ml` (JAX 0.10.0 cuda12)**

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
  0.13 ± 0.05 (criterion-dominated), Δν/ν ≈ 3.4% [2.2, 4.8]%. Exact edges:
  ⟨pending — production window solve⟩.
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
  Interior-vs-bottom-up comparison: ⟨pending — I4-interior run⟩.
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
  decoration — wider gap, tighter confinement. N=10k ξ(ω): ⟨pending⟩.

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
| I1 ground-truth parity | **PASS** | 50/50, max Δω/ω 2.83×10⁻⁵ (gate 1e-4), min proj² 1.0000 (gate 0.99), 0 ghosts, worst res 5.2×10⁻⁵; first run 49/50 → Amendment A1 recorded |
| I2 completeness | ⟨pending⟩ | deflated-probe KPM per slice |
| I3 residuals / no ghosts | ⟨pending — production⟩ | every reported pair ≤ 1e-4 by construction; Gram check pending |
| I4 new-decoration cross-check | bottom-up half DONE; interior half ⟨pending⟩ | gap 500\|501, Δν/ν 5.07%, 8,876 s |
| I5 spectrum consistency | ⟨pending⟩ | |
| I6 convergence (160³/192³ + 256³ anchors) | ⟨pending⟩ | |
| I7 decoration | **PASS** | ff = 22.011% at 192³ (gate 22.0 ± 0.5); r = 0.331836 µm |
| I8 localization | pipeline validated (N=1000); cross-solver + N=10k ⟨pending⟩ | band 500: ξ=1.82 µm r²=0.97 (production dec.) / 1.47 µm (circular); extended modes auto-flagged |
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
