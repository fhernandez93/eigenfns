# Pre-registration DRAFT — interior gap-edge eigenmodes, N=10,000 LSU network

**STATUS: DRAFT. Frozen only after Phase 1 completes (bake-off + N=10k DOS).
Placeholders marked ⟨⟩. On freeze this file is renamed without _DRAFT and no
gate may be weakened afterwards.**

## 1. Problem registration (already measured, final)

- Structure: `Structures/20260701_N10000_lsu_generated.txt`, 15,704 rod rows,
  N = 10,000, L = 24.6467 µm (`box_size_for_n`), ρ = 0.668 µm⁻³.
- Decoration (NEW, this project): circular rods aspect_ratio = 1.0,
  ε_rod = 2.9² = 8.41, ε_bg = 1, binary voxels (settled convention),
  **minor_radius = 0.331836 µm** (pinned 2026-08-17 by bisection to measured
  ff = 22.000% at 256³; 22.005% at 288³; N=1000 check 21.91%).
- Production grid: **256³** (10.39 vox/µm; 53.9 ms/vector matvec measured;
  vector = 0.268 GB c64). λ_max = 3974.6 (Lanczos ×1.05, to be re-bounded
  before production filtering).
- Window: ~300 bands straddling the gap. Exact [λ_lo, λ_hi] and band indices
  ⟨from N=10k KPM DOS: gap edges ± ~150 bands each side⟩. σ (folded/shift
  reference) = mid-gap from DOS.
- Band numbering: MPB convention (bands 1–2 = ω=0; solver mode n = band n+2).
  The ±2 open item vs the original cluster montage is noted in captions.

## 2. Method (fixed by bake-off — ⟨winner + parameters⟩)

Candidates measured on N=1000@128³ vs production ground truth (§6 of the
investigation report). Winner chosen on measured Θ-applications × 53.9 ms
extrapolated to 256³ with stated scaling, plus ghost/miss record.
- ⟨method, block size m, filter degree / σ placement / inner tol⟩
- Window sliced into ⟨k⟩ sub-windows of ⟨~n⟩ bands (memory: per-slice active
  basis must stay ≤ 26 GB host + ≤ 9 GB GPU transient); cross-slice dedup by
  eigenvector overlap (> 0.5 ⇒ duplicate).
- Precision policy: fp32 compute / fp64 host Gram+RR / precision=HIGHEST on
  all GPU matmuls (inherited; re-justified only if the bake-off shows
  otherwise).
- Every Ritz pair accepted ONLY with rel-res = ‖ΘH−λH‖/‖ΘH‖ < 1e-4 computed
  on the ORIGINAL Θ (ghost checklist item 1); out-of-window converged pairs
  recorded but excluded from the window deliverable (item 2).

## 3. Gate table (numbers final on freeze)

| gate | test | tolerance |
|---|---|---|
| I1 ground-truth parity | interior solver on production N=1000/128³/production-decoration reproduces the 50-band slice (modes 473..522): eigenvalues + degenerate-cluster principal angles | Δω/ω ≤ 1e-5 (eigenvalue, vs reference values); cluster projection² ≥ 0.999 (clusters = groups within rel 1e-3) |
| I2 completeness | # window Ritz pairs = deflated-probe KPM count in [λ_lo, λ_hi] (N=1000 exact check first, then N=10k) | count discrepancy = 0 (deflated mode, ±⟨se⟩); every discrepancy resolved |
| I3 residuals / no ghosts | per-pair rel-res, transversality (exact by construction — verified), window orthonormality; ghost checklist §5 of survey executed | rel-res ≤ 1e-4; ‖G−I‖_max ≤ 5e-5 |
| I4 new-decoration cross-check | N=1000 circular/2.9/ff22 at 128³: bottom-up (validated machinery) vs interior solver, full window | same as I1 |
| I5 spectrum consistency | KPM DOS gap edges vs window eigenvalue extremes, N=10k | within Jackson smearing (0.023 at degree 12k) ⊕ stochastic se |
| I6 convergence | gap edges + Δν/ν at 256³ vs ⟨192³ and/or 288³⟩ on a ⟨~40-band⟩ gap-edge subset; solver-tol sweep on ⟨8⟩ modes | expected scatter ~0.3% (G5 lesson); non-monotone ≤ ⟨0.5%⟩ or honest FAIL |
| I7 decoration | measured ff on production grid | 22.0 ± 0.5% absolute (measured 22.000%), radius recorded |
| I8 localization | IPR/ξ pipeline on N=1000 known modes, both solvers: band 500 (localized, ξ≈1.8 µm at ceiling 5.72) resolved & agreeing ⟨±10%⟩; deep-window modes flagged ceiling-limited | agreement bound ⟨±10%⟩; ceiling flags fire |
| I9 montage | N=10k gap-edge window, 15/row, MPB numbering, ±2 caveat in caption; side-by-side with N=1000 new-decoration montage (finite-size comparison) | band-count/window agreement with I2 |

All gates recorded in `results/gates/gate_results.json` (including I9 — G9's
omission was an open item).

## 4. Memory / disk budget (per phase; 54 GB free disk at kickoff, 62 GB RAM)

| item | size | policy |
|---|---|---|
| per-slice active basis (⟨m⟩ vectors × 0.268 GB) | ⟨17–26 GB⟩ | host RAM, GPU-streamed chunks ≤ 0.75 GB |
| per-slice checkpoint (converged H vectors) | ⟨~13 GB/slice fp32⟩ | disk, pruned to fp16 after slice verification (⟨~6.7 GB⟩) |
| final window H vectors (300) | 80 GB fp32 → **40 GB fp16** | fp16 on disk (parity loss ~1e-3 rel on vectors, eigenvalue/energy-density effect measured & reported); regenerable by script |
| final ε\|E\|² (300 × 256³) | 20 GB fp32 → **10 GB fp16** | fp16 |
| KPM moments/DOS | < 10 MB | keep |
| N=1000 new-decoration reference run (I4) | ~28 GB | prune vectors below window after gates, keep window (~7 GB) |
| prune-on-pressure list (pre-registered order) | `results/conv_N1000_G96` (12 GB) first, then old exp h5 (~0.4 GB) | only if < 10 GB free |

GPU transient budget per job ≤ 9.5 GB (measured headroom above desktop's
~1.7 GB). One heavy GPU job at a time (`gpu_is_busy` guard in every CLI).

## 5. Wall-clock budget + abort/descope (numbers final on freeze)

- N=10k production window solve: projected ⟨X⟩ h from bake-off Θ-apps/pair ×
  53.9 ms × 300 pairs × safety 2. **Abort criterion: if the measured first
  slice extrapolates the full window beyond ⟨3× projection⟩ or > 5 GPU-days,
  fall back (pre-registered now): descope to the ⟨150⟩-band core window at
  256³; if still over, drop to 192³ with the same gate table.**
- I4 bottom-up reference (new decoration, N=1000, 128³): ~5.5 h (measured
  analog); overnight job, resumable.
- I6 subset runs: ⟨~2–4 h each⟩.
- Every multi-hour run: per-slice/per-block checkpoints + auto-resume; assume
  CUDA death mid-run.

## 6. Deliverables checklist (unchanged from kickoff)

docs/REPORT_N10K.md; extended library + CLIs + tests; README one-command runs;
N=10k gap-edge montage + N=1000 new-decoration montage side-by-side; ξ(ω)
figure with ceiling marked; docs/plans/ records incl. adversarial verifications;
gate_results.json all gates.
