# Phase 1 investigation — interior gap-edge eigenmodes of the N=10,000 LSU network

**2026-08-17 · RTX 4080 Laptop (12 GB), 62 GB RAM, 54 GB free disk at kickoff ·
env `lsu_ml` (JAX 0.10.0 cuda12)**

Companion: `docs/plans/2026-08-17_interior_literature_survey.md` (method literature,
full citations). Status: §1–§5 measured and final; §6 (bake-off) IN PROGRESS —
table filled as runs complete.

---

## 0. Why new machinery: bottom-up infeasibility, re-derived from measured profiling

Production run (measured, `results/prod_N1000_G128.log`): 611 bands at 128³,
19,908 s, ~130k Θ-applications (≈213 apps/band; the 96³ run measured 155,168
apps for 613 bands ≈ 253/band), 2.7 ms/vector matvec → pure matvec ≈ 350 s;
the other ~19.5 ks is deflation streaming + orthogonalization + RR, growing
with locked-set size (late 20-band blocks: 37 → 70 iterations, ~600–800 s each).

Extrapolation to N=10k at 256³ (window near band ~5,000 by the 0.5 bands/vertex
rule; measured matvec 53.9 ms/vector, vector size 0.268 GB):

| item | scaling | value |
|---|---|---|
| locked-set storage, ~5,150 vectors | 5,150 × 0.268 GB | **1.38 TB — 12× the combined RAM+free disk. Hard stop.** |
| pure matvec time | 213 apps/band × 5,150 × 53.9 ms | 16.4 h (the *cheap* part) |
| deflation/ortho time | order-of-magnitude estimate (adversarial review F6): naive n²-scaling of the measured 19.5 ks gives ≈ 128 days; a first-principles PCIe-traffic bound (streamed deflation ≈ 3 passes × 1.38 TB per iteration) lands at the same order or worse. NOT a measurement-scaled extrapolation — at N=1000 scale the per-iteration cost is flat (11.5–12.1 s/iter for n_locked 20→600), so the n² regime is not yet visible in the production profile | **O(100) days** |

The storage wall alone (a verified 12× overrun) is a hard kill; the time
estimate is corroborating order-of-magnitude, labeled as such. The
existing lowpass ChebSI (`eigenfns/chebyshev.py`) amplifies *everything* below
the cutoff — same ~5,150-vector basis, same 1.38 TB wall. Hence: an interior
solver targeting only the ~300 window modes.

## 1. KPM shakeout on N=1000 (validated against the 611 exact eigenvalues)

New machinery (`scripts/exp/exp_kpm_dos.py` + `exp_kpm_analyze.py`): stochastic
Chebyshev *moments* collected once per probe (even/odd doubling: degree-p
moments from p/2 matvecs), giving N(λ) and the DOS at every threshold from one
pass — unlike `kpm_count_below` (one recurrence per threshold). Per-probe
moments saved → honest stochastic error bars.

Run: production structure/decoration/grid (gold N=1000, 128³, r=0.2252,
aspect 2.5, ε=8.5703), degree 8,000, 16 probes, λ_max = 4633.6 (Lanczos×1.05).
731 s wall. Verdict vs the exact spectrum (`results/exp/kpm_n1000_G128_prod_kpm.npz`):

- N(λ) tracks the exact counting function over the full known range;
  discrepancies consistent with quoted (stochastic ⊕ smearing) errors. Largest
  normalized outlier: the Jackson tail leaking +0.10 ± 0.01 phantom band below
  the spectrum bottom (known effect, sub-band size).
- **Window count 213.4 ± 2.6 vs exact 210** (1.3σ). Mid-gap count 501.9 ± 2.7
  vs exact 498 (systematic +3–6 above the gap: correlated across thresholds —
  one shared probe set — consistent with a ~+1σ fluctuation, not bias;
  se ≈ 2.7 bands at 16 probes).
- **Gap located [1.895, 1.969] vs exact [1.883, 1.963]** — both edges within
  the Jackson smearing width 0.037. Adversarial review (F7): those numbers are
  quantized to the coarse Chebyshev-Gauss analysis grid (~0.074 near the gap);
  on a fine uniform grid the 10%-of-median criterion gives [1.886, 1.948]
  (Δν/ν 1.62%) and 20% gives [1.855, 1.982] (3.30%) vs true 2.08% — i.e. KPM
  gap *edges* carry a criterion-dominated systematic ~±0.02–0.03 in λ at this
  degree. Calibration retained for the N=10k interpretation below; exact
  N=10k edges come from the eigensolver (gate I5 closes the loop).

Conclusion: KPM counting/DOS validated *at its stated resolution*: window
placement to ±(few bands ⊕ smearing). Exact completeness certification is NOT
claimed from plain probes — that remains the deflated-probe mode (variance
collapse; production G6), pre-registered as gate I2.

## 2. Decoration calibration (Phase 1.3) — DONE

`scripts/exp/exp_ff_calibration_n10k.py`, `results/exp/ff_calibration_n10k.json`.
Bisection of `minor_radius` to measured ff on the N=10k structure (15,704 rod
rows, L = 24.6467 µm), aspect 1.0, ε_rod = 2.9² = 8.41:

- **r = 0.331836 µm → ff = 0.22000 (256³) / 0.22005 (288³)** — grid-insensitive.
- Back-of-envelope r ≈ 0.30 µm underestimates node overlaps; the measured value
  is the deliverable (gate I7: 22.0 ± 0.5% — headroom ×50).
- Same radius on gold N=1000: ff = 0.21908 (128³) / 0.21893 (144³) — Δ ≈ 0.1%
  absolute. Densities match; no anomaly.

## 3. Grid choice measurements (Phase 1.4)

N=10k box L = 24.6467 µm. Production N=1000 physical resolution 128/11.44 =
11.19 vox/µm → 276³ equivalent; FFT-friendly candidates measured (chunk 4,
includes the KPM map axpy):

| grid | vox/µm | ms/vector-matvec | λ_max (Lanczos×1.05) | vector size | 350-vec basis |
|---|---|---|---|---|---|
| 256³ | 10.39 | **53.9** | 3974.6 | 0.268 GB | 94 GB |
| 288³ | 11.68 | **79.2** (×1.47) | 5055.9 | 0.382 GB | 134 GB |

Note the λ_max growth: the window/λ_max ratio (filter difficulty) degrades
~λ_max ∝ G². Leaning: **256³ production** (0.8 vox/µm below the production
convention — G5 showed gap-edge accuracy is rasterization-limited ~0.3%
regardless) with the I6 convergence sweep on a gap-edge subset at ≥2 grids
(candidates 192³/224³/288³ — final registration in Phase 2 after the bake-off
fixes the per-band cost). ff at the pinned radius is grid-stable (22.000/22.005).

## 4. N=10k full-bandwidth DOS and window derivation — DONE

256³, new decoration, degree 12,000 (Jackson smearing at gap ≈ 0.023),
12 probes, λ_max 3974.6, 8,477 s wall (`results/exp/n10k_G256_dos_kpm.npz`,
figure `n10k_G256_dos.png`).

- **Gap located** (fine-grid re-analysis after adversarial review F1 — the
  first-pass numbers [1.8877, 2.0262] were quantized to a ~0.07 analysis
  grid): criterion range on a 5×10⁻⁴ uniform λ-grid —
  5% of local median: [1.885, 1.968] (width 0.083, Δν/ν 2.16%);
  10%: [1.864, 1.996] (0.132, 3.42%); 20%: [1.837, 2.022] (0.185, 4.80%).
  **KPM gap estimate: width 0.13 ± 0.05 (criterion-dominated systematic),
  Δν/ν ≈ 3.4% with honest range [2.2%, 4.8%]**; N=1000 calibration (§1)
  shows truth tends to sit between the 10% and 20% criteria. Wider than the
  production elliptical-rod gap under the matched 10% criterion (0.132 vs
  0.062) — the circular-rod decoration does open the gap. In-gap DOS floor is
  consistent with pure Jackson edge leakage (review: integrated in-gap weight
  16.1 ± 0.9 vs 1.6 ± 0.2 phantom bands in the *exactly empty* N=1000 gap at
  ~1/10 the state count) — no evidence of true in-gap states. Exact edges are
  an eigensolver deliverable (gate I5).
- **Mid-gap count 5010.3 ± 12.5** — the 0.5 bands/vertex rule (expected
  ~5000) *located by KPM, not assumed*; gap between MPB bands ≈ 5012|5013
  (±13 stochastic; exact index certified at I2 by deflated counting).
- **Derived window: λ ∈ [1.71, 2.19] — population 317.1 ± 5.1 bands**
  (paired per-probe difference; the ±12.5 absolute-count se applies to the
  band-index *placement*, ≈ MPB 4860–5177 — review F4). ~150 per side of the
  gap; window edges keep ≥ 0.1 λ margin beyond the worst-case (20%-criterion)
  gap edges. σ (folded/shift reference) = 1.957.
- DOS at the window edges: ≈ 2,150 (below) / 1,615 (above) bands per unit λ —
  sets the cluster density the solver must resolve (~5×10⁻⁴ mean λ-spacing).

## 5. Literature (full survey in companion doc)

Bake-off slate from the survey: (a) folded spectrum with Wang–Zunger
squared-kinetic diagonal preconditioner (σ in gap) — only method whose active
set fits VRAM; (b) bandpass Chebyshev filtered subspace iteration, gap as free
transition margin (Fang–Saad measured 600–4,100 matvecs/pair at our window
ratio) — strongest completeness story + only direct fp32 validation (R-ChFSI);
(c) shift-invert MINRES — literature-rejected in the closest analog
(ESCAN/Canning: "not effective in practice"), run small anyway per kickoff
(negative results are findings). IFEAST skipped: provably a polynomial filter
in disguise + preconditioner-incompatible. Unified ghost defense (all methods):
final extraction = Rayleigh–Ritz on the ORIGINAL Θ + explicit per-pair
residual gate + KPM count audit (checklist §5 of the survey).

## 6. Method bake-off on N=1000 @ 128³ (Phase 1.2) — IN PROGRESS

Protocol: 50-band interior slice straddling the gap — 0-based solver indices
473..522 = MPB bands 476..525 (25 per side of the 500|501 gap;
**λ ∈ [1.71129, 2.14583]** — corrected per adversarial review F3), ground
truth `results/prod_N1000_G128`. σ = mid-gap = 1.92286. Slice relative width
0.435/4634 = 9.4×10⁻⁵ of the spectrum — comparable to (slightly harder than)
the N=10k production window ratio 0.48/3975 = 1.2×10⁻⁴. All methods report:
converged-and-verified pairs (rel-res < 1e-4 on Θ), targets found/missed,
ghosts, max Δλ/λ vs reference, projection² onto reference cluster subspaces,
Θ-applications, wall-clock. Driver: `scripts/exp/exp_bakeoff.py`.

| method | params | targets found /50 | misses | ghosts | max Δλ/λ | min proj² | Θ-apps | wall (s) | Θ-apps/pair |
|---|---|---|---|---|---|---|---|---|---|
| (a) folded LOBPCG | m=32, WZ α²∈{0.05,1,20}σ², P² variant | **0 verified (FAIL)** | 50 | n/a | n/a | n/a | ~38k/block | 629/block | ∞ |
| (b) bandpass ChebSI (pure) | d=3300, m=80, 3 outers (stagnation break) | **0 at gate** (subspace captured: 50/50 within 2×10⁻³, 38/50 within 5×10⁻⁴) | 50 | 0 | — | — | 792,528 | 7,109 | ext. ~74k |
| (c) shift-invert PMINRES SI | m=64, inner tol 1e-2 / maxit 150, 2 outers | **0 at gate** (64 in-window, med-res 1.8×10⁻¹) | 50 | 0 | — | — | 32,336 (incl. precond @⅔) | 441 | ext. ≫10⁴ |
| (d) two-stage bandpass | build d=3300 m=80 ×2 → trim 56 → polish d=8000 ×6 | **24 verified** (26 = slice-edge + upper-gap-edge pairs, still descending) | 26 | **0** | **2.8×10⁻⁵** | **1.0000** | 3.217M | 23,100 total | ~134k at this config; trim-limited (see verdict) |

**Verdict — winner: two-stage bandpass ChebSI** (build at moderate degree →
polish with the *same* filter machinery at high degree; Rayleigh-Ritz on Θ +
1e-4 residual gate + cluster-projection check as the only extraction path).
Evidence: 24 pairs reproduced to Δλ/λ ≤ 2.8×10⁻⁵ with proj² = 1.0000 and zero
ghosts anywhere; every miss is a *slice-edge or gap-upper-edge* pair — the
measured signature of the m=56 trim (1.12× oversampling; ChebFD prescribes
2×, our margin held 3–6 spare vectors vs ~10 transition-zone bands).
Med-res trajectory ×3.7 → ×2.3 → ×1.4 per outer as edge-starved pairs come to
dominate the median. Production consequence, pre-registered: **m ≥ 1.5× slice
population**; polish outers budgeted 3–4 at that margin.

**Measured cost geometry for N=10k (drives the Phase 2 budget).** Cost per
slice-outer = m × degree; the degree is set by the transition-to-window-width
ratio in θ-space (δλ_t ≈ π·√(λ·λ_max)/p, √(λ·λ_max) ≈ 88 at 256³), and the
N=10k spectrum is 10× denser than N=1000's. Full 317-band window at 256³
projects to **~10–15 GPU-days — over any sane budget**; the kickoff's
sanctioned descope ladder therefore fires *at registration time*:
192³ (matvec ~22 ms est., λ_max ×0.56) brings the full window to **~3.8
GPU-days projected**, with the gap-edge subset re-solved at 256³ for the I6
convergence gate (~1.3 days) — both inside a 5-day cap, preserving deliverable
1 (full ~300-band window) AND production-resolution gap edges. Exact numbers
frozen in the pre-registration after the 192³ matvec measurement.

**(b) pure bandpass — works as a builder, uncompetitive as a closer.**
Residual damping measured ×0.62/outer (6.6→4.2→2.5 ×10⁻²): with the m=80
basis unable to contain targets + the ~24-band filter-transition population,
convergence is edge-contested. Yet the *subspace* is captured after 3 outers:
every target has a Ritz value within 2×10⁻³ rel, 38/50 within 5×10⁻⁴. Pure
filtering to res 1e-4 extrapolates to ~11 more outers ≈ 3.7M total Θ-apps
(~74k/pair) — rejected; but 1–2 build outers are the right subspace factory.

**(d1) expansion-RR polish — measured unstable at scale (recorded negative
finding).** A LOBPCG-flavored polish (per-pair preconditioned residuals W =
P(ΘX−λX), plain RR over [X, W], keep m pairs) was tried in five instrumented
variants on the 128³ problem: window-score selection (med-res 4.2e-2 → 0.9 in
one sweep — interior-Ritz ghosts assembled from W's P-boosted low-band
content), continuity selection without W filtering (→ 0.67), W stripped at
degree 300 (transition ~1.0 λ — wider than the whole window, so low/mid bands
survive the "strip"; oscillated), strip 900 + continuity 0.7 (oscillates
7–23×10⁻³, no convergence). First sweep always helps (×5); subsequent RR
truncations disturb what they refine. Diagnosis: plain interior RR over an
expansion containing any sub-window content is structurally ghost-prone
(checklist §8) and the fp32 basis truncation each sweep re-injects it. The
toy (32³) cannot reproduce this — its spectrum has no deep-low bands.
Consequence: the production polish uses NO new numerics — the same filtered
SI at high degree on the trimmed basis (transition ≈ 5 bands at degree 8000,
absorbed by ~6 spare basis vectors). All variants logged in
`results/exp/bakeoff_hybrid_m80_polish.log` history + this record.

**(c) shift-invert PMINRES — measured uncompetitive (matches the ESCAN
verdict).** Inner solves never reached tol 1e-2 in 150 its (the indefinite
interior solve; the MPB preconditioner helps Θ, not (Θ−σ)); after 2 outers
all 64 Ritz values sit in-window but at med-res 1.8×10⁻¹. Closing would need
~4× the inner budget × several more outers with weak damping for window-wing
modes — strictly dominated by (d). First attempt also OOM'd (async-dispatch
pileup at chunk 16; fixed chunk 8 + 4-iteration barriers — engineering
finding recorded).

**(a) folded spectrum — measured FAIL (recorded finding; evidence base
corrected per adversarial review F2).** LOBPCG on (Θ−σ)², σ = 1.92286, m=32,
200 its + three preconditioner probes at 50 its each (Wang–Zunger diagonal
α² = {0.05, 1, 20}·σ², structural P∘P ≈ Θ⁻²; 161–175 s per probe).
*Correction (review F2):* the folded relative residual is provably
uninformative in this configuration — even a production-quality eigenvector
(Θ rel-res 6×10⁻⁵) has folded rel-res ≈ 1, because ‖(Θ−σ)²x − μx‖ ≈
λ_max·‖r_Θ‖ ≫ μ (verified fp32 AND fp64 by the reviewer on production
vectors); the tol=1e-3 lock gate was unpassable a priori at any iteration
count, and "residuals pinned at 1.0" carries zero information about
preconditioner grip. **The FAIL rests on the fp32-measurable μ-trajectory:**
locked Rayleigh quotients μ ∈ [0.096, 0.279] after 200 its remain entirely
above the target range (all 50 targets have μ ≤ 0.0497), with the descent
decelerating (×0.75 per 10 its and slowing at 50 its across all four
preconditioner variants) — a geometric-fit extrapolation to μ ~ 10⁻²
(mid-target) needs ≥ several hundred further iterations *per block* with no
observed preconditioner sensitivity, and the subsequent residual convergence
inside the compressed folded spectrum (target separations ~10⁻⁴ in μ) is the
harder part. Implementation audited by the reviewer: no sign bugs, WZ diagonal
finite, Θ-count correct, composition identity unit-tested. Verdict: folded
spectrum is not viable *here* at practical budgets; a future attempt would
need folded-residual scoring on Θ via `rr_extract` (as (b)/(c) are scored)
and a spectral (not kinetic) preconditioner. Trajectories in
`results/exp/bakeoff_folded_m32.log` + probe log.

## 7. Memory & disk reality (feeds Phase 2 budget)

- 300-vector window basis at 256³ = 80 GB: exceeds free disk (54 GB) and RAM
  (62 GB). Full-window-at-once methods (ChebFD-style N_S = 2–4 N_T ⇒ 160–320 GB)
  are out. Design consequence: **slice the window** (~4–8 sub-windows of
  ~40–75 bands; per-slice active basis 17–26 GB, RAM-resident, GPU-streamed),
  sequential slices, cross-slice dedup by eigenvector overlap, per-slice
  checkpoint + prune.
- Final artifacts must be budgeted at reduced precision (fp16 spectral H ≈
  40 GB or per-mode ε|E|² fp16 ≈ 10 GB) — exact registration in Phase 2;
  `results/conv_N1000_G96` (12 GB, regenerable) is the pre-registered prune
  candidate if needed.
