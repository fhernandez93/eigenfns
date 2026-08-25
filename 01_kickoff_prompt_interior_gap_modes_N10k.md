# Project kickoff prompt — interior gap-edge eigenmodes of the N=10,000 LSU network (JAX + CUDA)

## Mission

Extend the delivered `eigenfns/` project (this folder) to compute, **locally on this machine**,
the **gap-edge Maxwell eigenmodes and the spectrum of the N=10,000 disordered LSU network**

    Structures/20260701_N10000_lsu_generated.txt

— a periodic cube of side L = (10000/1000)^(1/3) · 11.44 ≈ 24.65 µm (canonical density
ρ ≈ 0.668 µm⁻³ recovered when using `box_size_for_n`; the raw endpoint extent ~26 µm includes
PBC-duplicate overhang), rasterized with a **new decoration**:

- **circular rods**: `aspect_ratio = 1.0` (not the 2.5 z-warp of the production run),
- **n = 2.9 exactly**: `eps_rod = 2.9**2` (not 2.9275²),
- **ff = 22%**: bisect `minor_radius` until the *measured* filling fraction on the production
  grid is 22.0%; back-of-envelope says r ≈ 0.30 µm, but the recorded, measured value is the
  deliverable. Pin the radius once, at production resolution, and freeze it.

Deliverables, in order of importance:

1. **The spectrum**: a full-bandwidth KPM density of states locating the gap of this
   structure/decoration, plus exact eigenvalues for a ~300-band window straddling the gap
   (expected near band ~5,000 by the 0.5 bands/vertex rule — but *located by KPM, not assumed*).
2. **Eigenfunction tiles**: a montage (same 15-per-row convention as
   `band_montage_398_607_15_non_ideal.png`) of ε|E|² for the gap-edge window modes.
3. **Localization analysis**: per-mode participation ratio / IPR and envelope-decay fits,
   reported as ξ(ω) across the window **with the finite-size ceiling stated explicitly**
   (a 24.65 µm periodic box resolves ξ up to only ~L/2 ≈ 12 µm; every mode whose fitted ξ
   exceeds that must be flagged "unresolved — lower bound only", never reported as extended).
4. **REPORT_N10K.md** + pre-registration and adversarial-verification records in `plans/`,
   in the same style as the delivered project.

## Why this needs new machinery (read before planning)

The production solver is bottom-up: it computes *every* band below the window, orthogonalizing
against all locked bands. That was 5.5 h for 611 bands at 128³ (N=1000). Here the window sits
near band ~5,000 at ~10× the DOF: extrapolated cost is **months** — infeasible, measured not
guessed (re-derive the scaling from the production profiling tables and say so in the plan).
The existing bottom-up Chebyshev filter (`eigenfns/chebyshev.py`) hits the same wall as
memory: it amplifies *everything* below the cutoff, so the basis would hold ~5,000 vectors
(~1 TB at production resolution) — also infeasible.

The project is therefore to build and validate an **interior-eigenpair solver** that targets
only the ~300 window modes directly. Candidate methods to evaluate **empirically, on this GPU,
at small scale where ground truth exists** (no method chosen by opinion):

(a) folded spectrum / shift-and-square, LOBPCG on (Θ − σ)² — squares the condition number
    but needs no new operator machinery;
(b) bandpass polynomial (Chebyshev) filtered subspace iteration — extend `chebyshev.py`
    from lowpass to bandpass; ChASE/FEAST-adjacent literature applies;
(c) shift-invert with an iterative inner solve (MINRES/CG on the shifted indefinite
    operator, matrix-free) — best convergence per outer iteration, cost dominated by inner
    solves;
(d) any hybrid the literature supports (polynomial-preconditioned folded spectrum, spectrum
    slicing with KPM counts as the slice oracle).

Interior methods have known failure modes the bottom-up solver never faced — **spurious/ghost
Ritz pairs, missed eigenvalues inside the window, filter-boundary pollution**. The validation
gates below exist specifically to catch these. Treat every window eigenpair as guilty until
its residual proves otherwise.

## Non-negotiables (inherited from the original kickoff, all still binding)

1. **Adversarial verification of all physics, numerics, and code** by refutation-tasked
   subagents; every pass (including "found nothing") recorded in `plans/`.
2. **Literature search online before committing to a method.** Read the primary sources on
   interior eigensolvers at scale: folded spectrum (Wang & Zunger / PARSEC-ESCAN lineage),
   Chebyshev filtered subspace iteration (Zhou, Saad et al.; ChASE), FEAST/contour methods,
   spectrum slicing + KPM counting (Weiße et al. KPM review), and polynomial-filter Maxwell
   or supercell photonics applications if any exist. Cite what the choice rests on.
3. **Fit this machine, by arithmetic done in advance.** RTX 4080 Laptop, 12 GB VRAM, 62 GB
   RAM, **54 GB free disk at kickoff (check again; heavy checkpoints must be budgeted,
   streamed, or pruned — the production run's `results/` are already ~tens of GB)**.
   At 256³ a single field vector (2 transverse components, complex64) is ~0.27 GB: a
   300-vector window basis is ~80 GB → **does not fit RAM**; the design must chunk the basis
   through GPU/RAM (the filter couples no vectors — exploit that, as `chebyshev.py` already
   does) and/or justify a smaller grid or window. Pre-register the full memory + disk budget
   per phase. One heavy GPU job at a time; checkpoint + auto-resume everything long.
4. **Faithful, not plausible.** Gates below, with pre-registered numeric tolerances, all
   adversarially verified before any PASS is claimed. Never weaken a gate to pass it; a
   FAIL + honest explanation beats a massaged PASS (the delivered project's G5 is the model).

## What already exists (reuse; do not reimplement)

| Asset | Where | Why it matters here |
|---|---|---|
| Working Γ-point Maxwell operator, bottom-up LOBPCG, checkpointing | `eigenfns/operator.py`, `solver.py`, `io.py`, `scripts/run_modes.py` | The operator and its GPU environment (platform allocator, cuBLASLt off, `precision=HIGHEST` for fp32 Gram — TF32 silently poisons it) are validated. Reuse verbatim. |
| Lowpass Chebyshev filter + Lanczos λ_max + KPM counting | `eigenfns/chebyshev.py` | Starting point for (b) and the DOS/counting oracle. Implemented, **never exercised at scale** — shake it out before trusting it. |
| Rasterizer with `aspect_ratio`, `eps_rod`, `minor_radius` parameters | `eigenfns/structure.py` (`rasterize_penlike`) | The new decoration is pure parameters. Keep the binary-voxel convention; its gap-edge sensitivity is *known* (G5: ~0.3% non-monotone) and must be re-measured, not re-litigated. |
| **Validated ground truth for the window**: N=1000 production run | `results/prod_N1000_G128/` (611 bands, window modes, ε\|E\|²), `REPORT.md`, `results/gates/` | **The single most valuable asset.** Any interior method must first reproduce these known interior eigenpairs on the same structure/grid/decoration before touching N=10k. MPB parity for the new run reduces to this chain — MPB itself cannot judge N=10k. |
| Montage + rendering | `scripts/make_montage.py`, `eigenfns/render.py` | Reuse for the tiles. |
| Gate framework + records | `scripts/validate.py`, `plans/`, `results/gates/` | Extend, same style. |
| Methodology + hardware quirk memories | project memory dir (`tier0-methodology`, `gpu-jax-quirks`, `project-final-state`) | TF32/Gram, fixed shapes under jit, orphaned-process hazard, ±2 band-numbering open item (we emit MPB numbering; bands 1–2 are ω=0), rfft-at-Γ ~2× headroom (optional, only with parity re-check). |

## Phase 1 — Investigate (deliverable: `plans/<date>_interior_investigation.md`)

1. **KPM shakeout + DOS of the target.** Validate KPM counting against the known N=1000
   spectrum (count in the production window vs the 611 known eigenvalues — exact agreement).
   Then run full-bandwidth KPM DOS on the rasterized N=10k structure (new decoration) at a
   feasible grid: locate the gap frequency, estimate the density of states at the intended
   window edges, and *derive the window* (band indices and [λ_lo, λ_hi]) from it. This also
   fixes σ for folded/shift methods. Report the KPM resolution (moments, stochastic vectors)
   with error bars.
2. **Method bake-off on N=1000 at 128³**, where `results/prod_N1000_G128/` supplies exact
   interior reference pairs: implement minimal versions of (a)/(b)/(c), measure wall-clock to
   reproduce a 50-band interior slice to the production residual level, count matvecs, and
   check for ghosts/misses against the known spectrum. Pick the winner on measured cost
   **extrapolated to N=10k with the scaling stated** (matvec cost × count, basis memory
   traffic). Losing methods are recorded findings.
3. **Decoration calibration.** Bisect `minor_radius` to measured ff = 22.0% on the production
   grid for the N=10k structure; record radius, achieved ff, and grid. Sanity-check the same
   radius on N=1000 (ff should land close; divergence means a density anomaly — investigate,
   don't ignore).
4. **Grid choice.** Production physical resolution is 128/11.44 ≈ 11.2 vox/µm → 276³ for
   L=24.65; FFT-friendly candidates 256³ (10.4 vox/µm) vs 288³ (11.7). Decide on measured
   matvec time + memory at both, and pre-register which grids the convergence sweep (gate I6)
   will use. The G5 lesson stands: gap-edge accuracy is rasterization-limited — expect ~0.3%
   scatter and design the sweep to *measure* it, not hide it.

## Phase 2 — Pre-register (deliverable: `plans/<date>_interior_preregistration.md`, frozen before build-out)

The chosen method and its parameters (filter degree / σ placement / inner-solve tolerances),
grid(s), window definition protocol, precision policy (inherit fp32-compute/fp64-Gram unless
the bake-off shows otherwise — any change re-justified), the complete gate table with numbers,
the memory/disk budget per phase, and a **wall-clock budget with an abort/descope criterion**
(e.g. if the bake-off extrapolation exceeds X GPU-days, the pre-registered fallback is a
narrower window / coarser grid, decided *now*, not mid-run).

## Phase 3 — Build

- Extend `eigenfns/` in place: interior driver alongside the existing solver
  (`eigenfns/interior.py` or extend `chebyshev.py`), thin CLI `scripts/run_interior.py`
  mirroring `run_modes.py` (`--resume`, single-GPU discipline, per-chunk checkpoints),
  KPM DOS CLI, `pytest` coverage for the new numerics at toy scale.
- Chunked basis streaming (GPU ↔ RAM ↔ disk if needed) with the budget from Phase 2.
- Montage + localization analysis scripts reusing `render.py`; IPR and envelope-decay fit
  with the ξ ceiling logic built in.

## Phase 4 — Validate (deliverable: `plans/<date>_interior_validation.md` + `results/gates/`)

| gate | test | tolerance (pre-register exact numbers) |
|---|---|---|
| I1 ground-truth parity | interior solver on the production N=1000 structure/grid/decoration reproduces `prod_N1000_G128` window: eigenvalues + subspace principal angles (degenerate clusters as subspaces) | Δω/ω ≤ ~1e-5; angle cosine ≥ 0.9999 (justify) |
| I2 completeness | # window Ritz pairs = KPM count in [λ_lo, λ_hi], on N=1000 (vs exact) then N=10k | exact match; every discrepancy resolved, not explained away |
| I3 residuals / no ghosts | per-pair ‖ΘH − λH‖/‖H‖, transversality, orthonormality; ghost-rejection procedure documented and exercised | ≤ production residual gate |
| I4 new-decoration cross-check | N=1000 with circular/2.9/ff22 solved BOTH bottom-up (validated machinery, ~5.5 h) and interior — full window compared | same as I1 |
| I5 spectrum consistency | KPM DOS gap edges vs window eigenvalue extremes, N=10k | within KPM resolution error bars |
| I6 convergence | gap edges + Δν/ν across ≥2 grids (from Phase 1.4) at N=10k; solver-tol sweep on a mode subset | reported with the rasterization-sensitivity framing; non-monotone ≤ pre-registered bound or honest FAIL |
| I7 decoration | measured ff on production grid | 22.0 ± 0.5% absolute, radius recorded |
| I8 localization metrics | IPR/ξ pipeline validated on a known-localized N=1000 gap-edge mode (both solvers agree) and a known-extended mode (flagged as ceiling-limited) | pre-registered agreement bound |
| I9 the montage | gap-edge window montage for N=10k, 15/row, MPB band numbering (bands 1–2 = ω=0; note the ±2 open item vs the original cluster montage in the caption) | band-count/window agreement + side-by-side in report |

## Final deliverables

1. `REPORT_N10K.md` — methods + literature + the bake-off table + performance breakdown
   (ms/matvec at each grid, total matvecs, wall-clock) + DOS figure + gap numbers + ξ(ω)
   figure with the ceiling marked + honest limitations.
2. Extended library + CLIs + tests; updated `README.md` (one-command runs for DOS, interior
   solve, montage, validation).
3. The N=10k gap-edge montage; the N=1000 new-decoration montage from I4 as a finite-size
   comparison (small-box vs 24.65 µm box localization, same decoration — state the ξ-ceiling
   caveat for both).
4. `plans/` records: investigation, pre-registration, adversarial verifications, gate results
   (all gates in `gate_results.json` this time — G9's omission was an open item).

## Rules

- Orient first: read `REPORT.md`, `plans/`, and the project memories before writing code;
  do not re-derive settled constants (d0 = 0.8 µm, L rule, density) or re-open settled
  conventions (binary voxels, MPB numbering) — parameter changes yes, convention changes no.
- Profile before optimizing; quote speedups at the size measured; label extrapolations.
- Negative results (a losing method, a failed precision policy, a ghost outbreak) are
  recorded findings.
- Heavy artifacts out of git, everything regenerable by script; watch the 54 GB free disk —
  pre-register what gets pruned when.
- One heavy GPU job at a time (`gpu_is_busy` discipline); every multi-hour run resumable;
  assume CUDA can die mid-run and design for it.
