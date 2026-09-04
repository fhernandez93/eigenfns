# Project kickoff prompt — lattice-constant convention, supercell band structures along Γ–X–M–R–Γ, and the resolution-matched in-gap test (JAX + CUDA, MPB judge)

## Mission

Extend the delivered `eigenfns/` project (this folder) with four pieces of work, **locally on this
machine** (RTX 4080 Laptop 12 GB, `lsu_ml` env; MPB on the CPU as the independent judge), in the
same working style as phases 1–3 (profile first, pre-register numeric gates, adversarial
verification, honest negatives, checkpoint + auto-resume, one GPU job at a time — see
`docs/plans/2026-08-12_orientation_and_experiments_log.md` and `docs/README.md`):

1. **Fix the definition of the lattice constant `a`.** The project currently quotes
   ν = ωa/2πc with a = L/5 = 2.288 µm (a "density-equivalent 8-vertex cell"). That convention
   is retired. Two admissible definitions replace it, both implemented and both reported:
   (a) **MPB's**: a is the periodicity of the computational cell, i.e. the supercell side L;
   (b) **the first peak of the radial autocorrelation of ε(r)**, exactly as
   `get_a_from_h5_eps` in
   https://github.com/fhernandez93/LoadingPermittivityDistToSlabTransmission/blob/main/AutomationModule/tools.py
   (the transmission project's convention).
2. **Treat the disordered network as a crystal with lattice constant L** (11.44 µm for N=1000,
   24.6467 µm for N=10k): every new run uses the periodically wrapped rasterization
   (`periodic=True` / `--periodic`), never the montage convention with its box-face seam.
3. **Band structures along Γ–X–M–R–Γ** of the supercell Brillouin zone — eigenvalues only,
   no wavefunctions — for the N=1000 network at low resolution **and** for the srs crystal,
   each computed by our solver and by MPB on the identical discrete ε grid, and compared
   band by band.
4. **Wavefunctions around the gap for N=1000 and N=10k at the same voxel size** (same
   resolution per wavelength), periodic rasterization, to decide whether the N=10k in-gap
   states survive when the resolution confound is removed, or whether they are a resolution
   artefact.
5. **A new PDF report in PRL style** (REVTeX 4.2 main text + complete Supplemental Material)
   presenting this study, built the way the phase-3 package was built: every number from the
   saved data through scripts, a number ledger, a fact-check ledger and adversarial rounds.
   See Part 5.

Deliverable records: a frozen pre-registration in `docs/plans/<date>_bandstructure_preregistration.md`
**before** any production run, gate results in `results/gates/gate_results.json` (new gates
B0–B5 below), an adversarial-verification record, `docs/REPORT_BANDS.md` (the working
record, same style as `docs/REPORT_N10K.md`), and the PRL-style package `report_bands/`
with its two PDFs (Part 5). Update `README.md`, `docs/README.md`, `results/README.md` and
`scripts/exp/README.md` indexes as you add runs and scripts. Do **not** edit the existing
`report/` (the phase-3 manuscript) — whether that paper adopts the new conventions is a
separate decision; the new report lists what would change there.

## Method of record — the phase-2 N=10k pipeline, for everything

Every new eigenpair computation in this study uses **the method of the N=10k work**, not the
phase-1 bottom-up solver: it is the interesting one, it is the one that scales, and this
study should deepen it rather than fall back. Concretely, the pipeline pre-registered in
`docs/plans/2026-08-18_interior_preregistration.md` and executed in phase 2:

1. **KPM density of states** (`scripts/exp/exp_kpm_dos.py`, `exp_kpm_analyze.py`) to locate
   the gap and derive the λ-window — re-run on the **periodic** ε, which `docs/REPORT_N10K.md`
   left pending ("retired pending a periodic KPM run");
2. **two-stage bandpass Chebyshev subspace iteration** (`eigenfns/interior.py`,
   `scripts/run_interior.py`: build filter degree ≈ 3000 with m ≈ 1.5× the expected count,
   polish at degree ≈ 12000 on the trimmed basis, residual certification, per-outer
   checkpoints, host-resident streamed basis where the device path OOMs);
3. **I2 completeness audit** on every slice and **I1-style parity scoring** against whatever
   ground truth exists (`scripts/exp/exp_i2_completeness.py`, `exp_i1_score.py`);
4. slice merge with overlap dedup (`scripts/merge_slices.py`), cross-grid overlap matching,
   localization with the L/2 ceiling, montage tiles.

This applies to the N=1000 legs of Part 4 as much as to N=10k (as gates I1 and I4-interior
already did at 128³), and to the per-k window solves of the band structures in Part 3: the
interior solver targets the λ-window around the gap at each k-point directly, which is
exactly what a band structure of bands ~480–520 needs. The bottom-up LOBPCG
(`scripts/run_modes.py`) is kept **only as the cheap cross-check** where it is affordable
(small grids, N=1000): it certifies band counts and provides the ground truth for the k ≠ Γ
parity gate, the way `prod_N1000_G128` did for gate I1. MPB is bottom-up by nature and
remains the independent judge.

## State of the code — read before planning

What exists (verify each claim by reading the code; do not trust this list blindly):

- **k ≠ Γ is supported by the operator but not by any entry point.**
  `eigenfns/operator.py::make_basis(grid_size, box_size, k_frac)` takes a Bloch vector in
  units of 2π/L and `MaxwellOperator(eps, L, k_frac=...)` accepts it; the preconditioner
  mask (`solver.py:162`) is built from `basis.kn`. But `scripts/run_modes.py` has no `--k`
  flag, and the only k ≠ Γ test (`tests/test_operator.py::test_bloch_k_reduces_to_shifted_planewaves`)
  is on a **homogeneous** medium. Nothing has ever been solved at k ≠ Γ on a disordered ε.
- **Band-numbering hazard at k ≠ Γ.** At Γ the G = 0 transverse pair is zeroed in the basis,
  which removes the two ω = 0 modes exactly, so solver index n ↔ MPB band n+2. At k ≠ Γ
  |k+G| > 0 everywhere, nothing is zeroed, and solver index n ↔ MPB band n. The number of
  bands below the gap is the same at every k (500 for N=1000 in MPB numbering, including the
  two Γ zero modes), so the gap is "500|501" at every k in MPB numbering but "498|499" in
  solver indices at Γ and "500|501" at k ≠ Γ. Get this right once, in one helper, and test it.
- **Rasterizer**: `eigenfns/structure.py::rasterize_penlike(..., periodic=False)`. The default
  reproduces the montage convention **including its box-face seam** (outer-shell ff 0.1975 vs
  0.2211 interior — a planar defect that hosted 4 spurious in-gap states in the N=10k window,
  `docs/REPORT_N10K.md`, correction of 2026-08-24). `periodic=True` fixes it (verified
  surgical: 0.078 % of voxels change). This study uses `periodic=True` **everywhere**.
- **Decoration** for everything new: circular rods, `aspect_ratio=1.0`, `minor_radius=0.331836`
  µm, `eps_rod=8.41` (n = 2.9), ff = 22.00 % at 256³ (21.91 % at 128³ for N=1000). The
  elliptical montage decoration (aspect 2.5, ε = 8.57) is not used here except where an
  existing artefact is reused for a cross-check.
- **MPB judge harness**: `scripts/exp/exp_parity32_mpb.sh` rasterizes with our code, writes
  the grid to HDF5 and feeds it to MPB via `epsilon-input-file` with `resolution` equal to our
  grid and `mesh-size 1`, so MPB solves the **identical binary ε** — parity has been 1e-4 in
  Δω/ω on 300 bands at 64³ (gate G3). `scripts/exp/exp_srs_literature.sh` runs the ideal srs
  crystal from MPB object geometry (MPB's own smoothing) along a cubic-cell k-path. Both
  scripts call `conda run -n mbpEnv mpb`, while `README.md` names the MPB environment
  `mpb_judge` — check which environment actually has `mpb` before running anything.
  MPB's `epsilon-input-file` reads the dataset named `data`; `get_a_from_h5_eps` reads a
  dataset named `epsilon`. Write both, or pass arrays directly.
- **srs generator**: `srs_crystal_rods(num_vertices, box, d0)` in the companion repo
  `/home/francisco/Documents/Create LSU Structures  - Claude/tools.py` (two spaces): 8
  vertices per conventional cubic cell; the N=1000-equivalent crystal is the 5×5×5 tiling
  (125 cells, 1000 vertices) in the same L = 11.44 µm box, hence cubic cell a_srs = L/5 =
  2.288 µm and d0 = a_srs·√2/4 = 0.809 µm (the disordered networks have mean rod length
  0.80 µm at the same vertex density 0.668 µm⁻³).
- **Existing runs to reuse** (all indexed in `results/README.md`): `i4_n1000_circ_G128`
  (N=1000, circular, montage convention, all 611 bands at Γ, gap 500|501 with exact edges
  λ = 1.8276 | 2.0225 µm⁻², Δν/ν = 5.07 %), `n10k_G192_window` (N=10k, montage convention,
  133 states in [1.757, 2.117], nominal gap 1.8860 | 1.9264), `n10k_G192_gap_periodic` and
  `_v2` (N=10k **periodic**, 192³, [1.855, 2.0], 7 converged states, 3 inside the S_gap
  bracket), `n10k_G256_edgelow` / `n10k_G256_edgehigh_narrow` (256³ anchors, montage
  convention, seam-contaminated), `results/exp/n10k_G256_dos_kpm.npz` (KPM DOS, montage
  convention). The cross-grid and periodic overlap matchers
  (`scripts/exp/exp_periodic_match.py`, `results/gates/crossgrid_match*.json`) are the
  tools for tracking one state across grids.
- **Open items this study closes**: the never-executed registered gate **G1** (crystal
  supercell k ≠ Γ parity), and the **resolution confound** left open in `docs/REPORT_N10K.md`
  ("N=10k runs at 7.79 vox/µm against N=1000's 11.2 … coarser rasterization manufacturing
  in-gap states is not excluded").
- Hardware/numerics rules that still apply: TF32 off in solver matmuls (`precision=HIGHEST`),
  fp64 dense algebra on the host, fixed shapes inside jitted loops, one heavy GPU job at a
  time, long chains detached with `setsid nohup` and polled through their logs (the harness
  kills background tasks), every run checkpointed and `--resume`-able.

## Part 1 — the lattice constant

Create `eigenfns/units.py` with two named conventions and use it everywhere a ν is printed:

- `a_cell(L) = L` — MPB's definition. With `geometry-lattice (size 1 1 1)` MPB's frequencies
  are ωa/2πc with a = L, and its k-points are in units of 2π/L, exactly the `k_frac` of
  `make_basis`. Under this convention the N=1000 gap centre sits at ν ≈ 2.58 and the N=10k
  one at ν ≈ 5.56 (the old 0.516 × L/2.288); ν is **not** comparable across sizes — say so
  in every table.
- `a_corr(eps, L)` — a faithful port of `get_a_from_h5_eps`: min–max normalise ε to [0, 1],
  subtract the mean, autocorrelation by FFT normalised to C(0) = 1 and `fftshift`ed
  (`compute_normalized_autocorrelation_fft`), radial average in 50 linear bins from 0 to the
  box half-diagonal (`radial_profile`), cubic interpolation on 20 000 points, first local
  maximum by `scipy.signal.argrelextrema(np.greater)`. Reproduce the reference
  implementation bit-for-bit first (same bins, same interpolation, same maximum rule) so that
  numbers agree with the transmission project, then characterise it:
  - **Binning quantisation.** With 50 bins the bin width is ≈ L√3/2/50 ≈ 0.2 µm for N=1000
    and ≈ 0.43 µm for N=10k, so the peak position is quantised at that scale. Report a_corr
    for bins ∈ {50, 100, 200, 400} and grids {64³, 96³, 128³} for N=1000 and {128³, 192³,
    256³} for N=10k, periodic rasterization, circular decoration; also for the elliptical
    montage decoration at 128³ and, once, for the montage convention (non-periodic) so the
    seam bias is on record. State which value the project adopts (the reference-implementation
    value with 50 bins, or a bin-converged one) and why.
  - The FFT autocorrelation is circular, i.e. it assumes ε is periodic — one more reason the
    periodic rasterization is the only admissible input.
  - Apply the same function to the rasterized srs 5³ supercell: a_corr should recover a
    length tied to the crystal (state which: the cubic cell 2.288 µm, its nearest-neighbour
    distance, or something else — measure, don't assume) and this is the sanity check of the
    port.
  - Expectation to test, not assume: for the disordered networks a_corr is of the order of
    the rod length (0.8 µm) or the vertex spacing, and the **same for N=1000 and N=10k**
    within statistical noise (same density, same decoration). If it is not, that is a
    finding, not a bug to hide.
- Every quoted frequency in this study carries the invariant (λ = (ω/c)² in µm⁻², or ω/c in
  µm⁻¹, and the vacuum wavelength) plus ν_cell and ν_corr. Use ν_cell for all MPB parity
  tables and band-structure plots (it is literally MPB's output), ν_corr for cross-size
  physics tables. Retire a = L/5: grep for `2.288`, `L/5`, `a_norm`, `ωa/2πc` in `eigenfns/`,
  `scripts/`, `docs/` and fix each site (leave `report/` alone, but list the sites).

## Part 2 — the disordered network as a crystal of period L

- All new ε grids: `rasterize_penlike(rods, G, L, minor_radius=0.331836, aspect_ratio=1.0,
  eps_rod=8.41, periodic=True)`, L from `box_size_for_n(N)` (11.44 and 24.6467 µm).
- One-time **re-validation of the periodic convention against MPB** (gate **B0**): N=1000 at
  32³ and 48³, Γ only, ≥ 300 bands, both codes on the identical periodic ε via
  `epsilon-input-file`; pass = max Δω/ω ≤ 1e-4 band by band (same criterion as G3). Record ff
  of the periodic grid at each resolution.
- **This is the same periodic supercell MPB uses, not an analogue of it.** MPB with
  `geometry-lattice (make lattice (size 1 1 1))` and `epsilon-input-file` at `resolution G`
  solves the Bloch-periodic problem on the cube of side L (its unit of length) in the plane-wave
  basis of the same G³ reciprocal vectors, with the same two-component transverse H basis and
  the same ε samples; its k-points in reciprocal-lattice units are our `k_frac` in units of
  2π/L; its frequencies in c/a are our ν_cell. The two codes differ only in the eigensolver
  and its precision, which is why the parity gates are exact comparisons at 1e-4 and not
  approximate ones. A uniform grid offset between the two codes (half-voxel conventions) is
  a translation and cannot change an eigenvalue. Write this down in the report's method
  section and verify it once in gate B0 by also checking one k ≠ Γ point at 32³.
- Say explicitly in the report that results here are **not** directly comparable to the
  montage-convention artefacts (`prod_N1000_G128`, `n10k_G192_window`) and quantify the
  difference where both exist (the N=1000 gap edges at 128³, periodic vs montage).

## Part 3 — band structures along Γ–X–M–R–Γ (eigenvalues only)

The supercell is the Bloch-periodic cube of side L of Part 2 — the one MPB solves — so it
is simple cubic and, in units of 2π/L: Γ = (0,0,0), X = (½,0,0), M = (½,½,0), R = (½,½,½). Time-reversal makes ω(k) = ω(−k); no need for both. Path
Γ→X→M→R→Γ with `n_interp` interior points per segment (MPB's `interpolate`); start at 3
(13 k-points with the closing Γ) and increase only if the cost table allows.

**3a — machinery (do first, cheap).**
- Add `--k kx ky kz` (units 2π/L) to `scripts/run_interior.py` (primary) and to
  `scripts/run_modes.py` (cross-check only), and a `scripts/run_bands.py` that walks a k-path,
  runs the interior window solve at each k (same window, same m, same filter degrees for every
  k), runs the I2 completeness audit per k, writes `bands.npz` (k-list, certified eigenvalues
  per k with their MPB band indices, residuals, I2 verdict, wall time) and is resumable
  **per k-point**. Eigenvectors are not stored beyond the per-k audit (a band structure is
  eigenvalues; see the report for the distinction). The Chebyshev filter needs its spectral
  bounds per k: re-estimate λ_max at every k-point (it shifts with k), never reuse the Γ value.
- The band **index** of each certified window eigenvalue at k comes from the count below the
  window: at Γ from the bottom-up cross-check or the I2-certified count, at k ≠ Γ from MPB's
  full band list at that k where MPB ran, and from a bottom-up cross-check at the production
  grid otherwise (cheap at 48³). Record which source was used at every k.
- Tests: (i) hermiticity and positive-definiteness of Θ_k on a small disordered ε at a
  general k; (ii) our eigenvalues vs dense diagonalisation on a 6³–8³ disordered ε at
  k = X, M, R and one general k (extend `test_operator.py`); (iii) the band-number helper:
  at Γ index n ↔ band n+2, at k ≠ Γ index n ↔ band n; (iv) DOF count 2G³−2 at Γ vs 2G³ at
  k ≠ Γ. Confirm no code path assumes real fields at Γ (the manuscript's "Θ real symmetric on
  real fields" statement is Γ-only; at k ≠ Γ fields are complex).
- Cost table (measured, not guessed): seconds per k-point for the interior window solve
  (bands ~480–520, state m and both degrees) at 32³, 48³, 64³; seconds per k-point for the
  bottom-up cross-check to band 520 at the same grids; and MPB wall time per k-point at the
  same grids with `tolerance 1e-7` (phase-1 calibration: MPB 300 bands/64³ = 2 h 22 min on
  the CPU; ours bottom-up 680 bands/64³ = 941 s; interior 50-band slice at 128³ = 2.6 h in
  gate I1). MPB must reach band ≥ 520 at every k it judges. Choose the production grid from
  the table;
  the expected answer is **48³** (dx = 0.238 µm, about 6.5 voxels per in-dielectric
  wavelength at the gap — crude, but both codes see the same grid, so the comparison is exact
  even though the physics is coarse). MPB may judge a subset of k-points (Γ, X, M, R and one
  general midpoint) if the full path would take more than ~3 CPU-days; it runs on the CPU
  concurrently with the GPU jobs.

**3b — N=1000 network, circular decoration, periodic, low resolution.**
- Interior window solve (bands ~480–520) at every k on the path with I2 per k; MPB bottom-up
  to band 520 on all k-points or the subset; bottom-up cross-check of the band count at the
  production grid at least at Γ, X, M, R.
- Gate **B1** (disordered k ≠ Γ parity, scored the I1 way): at every shared k, every MPB
  eigenvalue inside the window is matched by a certified interior eigenvalue with
  |Δω/ω| ≤ 1e-4, with zero misses and zero ghosts; the interior band indices agree with MPB's.
  This replaces the old unexecuted G1 for the disordered case, and is the first proof of the
  interior solver at k ≠ Γ — gate I1 only ever tested it at Γ.
- Report: the band structure plot of bands ~480–520 (ν_cell on the left axis, ω/c in µm⁻¹ on
  the right), the gap as max_k ω₅₀₀(k) | min_k ω₅₀₁(k) (indirect) versus the Γ-only values,
  and whether the gap is direct.
- Analysis worth having and cheap: the bandwidth W_n = max_k ω_n(k) − min_k ω_n(k) of every
  band near the gap, plotted against the mode's localization length ξ from the N=1000
  localization tables (`results/i4int_n1000_localization_modes.json`, Γ modes of the montage
  convention — match by eigenvalue, state the convention mismatch). Pre-register the
  expectation: modes with ξ ≪ L are insensitive to the Bloch phase across the box and should
  have W_n → 0; extended modes disperse. This is a Thouless-style localization diagnostic
  that needs no wavefunctions.

**3c — the srs crystal, two ways.**
- (i) **Same pipeline as the network**: rods from `srs_crystal_rods`, circular decoration
  (r = 0.331836 µm, ε = 8.41), binary periodic rasterization.
  - Single conventional cell, a_srs = 2.288 µm, grids 16³, 24³, 32³ (cheap; also report the
    ff and its distance from 22 %). Path Γ–X–M–R–Γ of the cubic cell. Expected gap between
    bands 4|5 (conventional cell; = 2|3 of the bcc primitive cell).
  - 5³ supercell (1000 vertices, L = 11.44 µm) at G_sc = 5 · G_cell (e.g. 80³ from the 16³
    cell) so that the voxel pattern is exactly the tiled cell pattern. Gate **B3** (the old
    G1): the supercell eigenvalues at k_sc must equal, within 1e-4 in ω, the union over the
    125 supercell reciprocal vectors G_sc of the single-cell bands at k_sc + G_sc (folding),
    and the gap must appear at 500|501. Ours on all k; MPB on the cell grid at all k and on
    the supercell at a subset.
  - Gate **B2**: cell parity vs MPB on the identical grid, ≤ 1e-4.
- (ii) **Literature anchor** from MPB object geometry (existing `exp_srs_literature.sh`):
  resolve the recorded r/a mismatch — Sellers's srs at ε = 13 gives 28.06 %; the surviving
  run at r/a = 0.2554 gave 2.90 % while the logged optimum 27.97 % was at r/a = 0.13. Establish
  the convention (cylinder radius vs cell edge, conventional vs primitive cell) and record
  it with a single reproducible MPB run whose output is kept.
- Side-by-side figure: srs cell bands (folded to the 5³ supercell zone) next to the N=1000
  network bands, same ν_cell axis, same k-path — the crystal-vs-disordered comparison the
  study is for.

## Part 4 — in-gap states vs resolution at matched voxel size (wavefunctions)

Design: **2 sizes × 3 matched voxel sizes**, all periodic, circular decoration. The box
ratio is L₁₀ₖ/L₁ₖ = 10^{1/3} = 2.1544. The reference resolution is **the N=1000 production
one, 128³ (dx = 0.0894 µm, 11.2 vox/µm)**; the N=10k grid that matches it is 275.8³, and the
FFT-friendly choices around it are 280³ (2³·5·7; dx = 0.0880, 1.5 % finer) and 270³ (2·3³·5;
dx = 0.0913, 2.1 % coarser). **Take 280³** — the mismatch is in the conservative direction
(N=10k slightly finer) and the N=1000 side stays identical to every existing 128³ artefact.
(The pair 125³ ↔ 270³ matches to 0.04 % and is ~10 % cheaper; it is the fallback if 280³ does
not fit, at the price of a new N=1000 grid.) Avoid 275³ and 276³: factors 11 and 23 are slow
in cuFFT. The full ladder:

| leg | N=1000 grid, dx (µm) | N=10k grid, dx (µm) | mismatch | status |
|---|---|---|---|---|
| coarse | 90³, 0.1271 | 192³, 0.1284 | 1.0 % | N=10k exists (`n10k_G192_gap_periodic`, `_v2`); N=1000 new (hours) |
| fine | 120³, 0.0953 | 256³, 0.0963 | 1.0 % | both new; N=10k ≈ 30–40 h detached (256³ anchors took 27 h at m=18) |
| **matched-128³ (required)** | 128³, 0.0894 | 280³, 0.0880 | 1.5 % | both new (the existing 128³ N=1000 runs are montage convention); N=10k ≈ 5–6 days detached, see below |

Feasibility of the 280³ leg on the 12 GB card (extrapolated from measurements — re-measure
before committing): one plane-wave vector is 2·280³ complex64 = 351 MB, so an m = 30 basis is
10.5 GB and must live on the host (the streamed-basis path `bandpass_subspace_hosted` already
does this at 192³); the filter chunk of 8 vectors plus the six-FFT workspace fits with
`--chunk 4–8`. Matvec: 23.2 ms at 192³, 53.9 at 256³, 79.2 at 288³ (measured,
`results/exp/n10k_G*_timing.json`), so ≈ 73 ms at 280³. λ_max: 2209 / 3975 / 5056 at
192³ / 256³ / 288³, so ≈ 4780 at 280³; the Chebyshev degree needed for a fixed window scales
with λ_max, so degrees ≈ 2.2× the 192³ values. The 192³ periodic gap-window run (m = 30,
build 4000 × 2, polish 12000 × 4) took 20.0 h; scaling by matvec (3.1×) and degree (2.2×)
gives ≈ 135 h ≈ 5.6 days for the same window and m. Narrowing the window does not lower the
cost per state (degree ∝ λ_max/Δλ while the count ∝ Δλ), so budget by the number of
states targeted, and use the resolution continuation of 4b to cut it. Run it detached with
per-outer checkpoints and poll the log; it is the longest single job of the study and the
one the verdict hinges on, so it starts as soon as the 256³ leg is done.

At the gap centre (√λ ≈ 1.39 µm⁻¹, vacuum wavelength ≈ 4.5 µm, in-dielectric ≈ 1.56 µm) the
coarse leg has ≈ 12 voxels per in-dielectric wavelength, the fine leg ≈ 16 and the
matched-128³ leg ≈ 17.5; state these in the report as "resolution per wavelength", and add
the helper that turns (L, vacuum wavelength, n_max, steps per in-medium wavelength) into an
FFT-friendly grid size, so this number is chosen once and printed with every run.

Runs:
- N=1000 legs: the same interior pipeline as N=10k (`run_interior.py --periodic`, same
  window, I2 on every slice), keeping vectors and ε|E|² for the window states — exactly as
  gates I1 and I4-interior did at 128³. Because bottom-up is cheap for N=1000 at ≤ 128³
  (5.5 h for 611 bands at 128³), run it once per leg as the cross-check that certifies the
  band count and scores the interior result I1-style; that cross-check is what makes the
  N=1000 legs a control for the N=10k legs, where no bottom-up ground truth can exist.
- N=10k legs: `run_interior.py --periodic` (two-stage bandpass ChebSI) on a window covering
  both gaps, λ ∈ [1.80, 2.05] µm⁻² (N=1000 edges 1.8276 | 2.0225; N=10k nominal 1.8860 |
  1.9264 at 192³ montage convention — the periodic edges moved inward, check
  `n10k_G192_gap_periodic_v2` before fixing the window), split into slices as in phase 2,
  with the **I2 completeness audit on every slice** (the periodic v2 run's I2 was "not
  resolved" — this time it must be). Per-outer checkpoints, detached, one at a time.

**4b — "dynamic resolution", Tidy3D-style: what transfers to a spectral solver.**
Tidy3D's automatic grid is a nonuniform *rectilinear* mesh whose step in each region is set
by a minimum number of steps per wavelength inside the local medium, refined where the index
is high and coarse where it is low. Take the idea seriously and sort it into what a
plane-wave (FFT) solver can and cannot do, then implement the parts that can:

1. **Automatic grid choice by steps per in-medium wavelength** — implement (the helper above).
   For a statistically homogeneous disordered medium at 22 % ff the high-index material is
   everywhere, so the wavelength-based rule gives one global dx; that *is* the uniform grid,
   and it should be said in the report that a Tidy3D-style graded mesh would refine
   essentially the whole box here.
2. **Resolution continuation (spectral prolongation)** — implement; this is the useful form
   of "dynamic resolution" for this solver. In the plane-wave basis a coarse-grid
   eigenvector prolongs *exactly* to a finer grid by zero-padding its spectral coefficients
   (restriction = truncation). Start every finer leg of the ladder from the prolonged
   converged basis of the coarser leg (192³ → 256³ → 280³) instead of random vectors: it
   should shorten the build stage and reduce the polish outers, and it gives the per-state
   tracking across dx (metric 3) for free, since each fine state descends from a named coarse
   one. Measure the saving in outers and wall time against a cold start at 256³ (one
   cold-start leg is needed anyway as the control; the 256³ anchors are montage convention
   and cannot serve). Also use it for the eigenvalue extrapolation across the three grids
   (state the order observed; binary voxels give first-order edges).
3. **Subpixel smoothing** (Kottke–Farjadpour–Johnson anisotropic ε averaging at rod
   surfaces, what MPB does for object geometry) is the standard way to buy effective
   resolution at interfaces on a fixed grid. It is a **decoration change** (a third
   decoration, not comparable to either binary one) — implement it in the rasterizer behind
   a flag, run it **only** as an N=1000 side check (gap edges and in-gap count vs the binary
   ladder at 90³/120³/128³), and report whether it moves the binary answer. Do not use it in
   the N=10k legs of the ladder.
4. **A true nonuniform mesh does not transfer.** The FFT basis needs a uniform grid; the only
   FFT-compatible route is a separable coordinate map x = f(u) (transformation optics), which
   turns the problem on a uniform u-grid into one with an anisotropic ε′ *and* a non-trivial
   μ′, i.e. a generalized eigenproblem the transverse-H operator does not solve. For this
   medium it would buy nothing (item 1). Record this reasoning in the report; do not build
   it. If a nonuniform mesh is ever wanted for a structure with distinct regions (a slab in
   air), that is a separate project.

Pre-registered metrics and verdict rules (write them down before any run starts):
1. **Count of states inside the N=1000 gap bracket** [1.8276, 2.0225] (and inside the
   narrower N=10k bracket) at each dx, for each size, periodic convention.
2. **The positive control**: does coarsening N=1000 from 120³ to 90³ create states inside
   its own gap, or move its edges by more than the local level spacing? If coarsening
   manufactures in-gap states in a structure known to be clean at 128³, then the N=10k
   in-gap states at 192³ are not trustworthy at that dx.
3. **Per-state tracking across dx** by eigenvector overlap after resampling (existing
   cross-grid matcher): a state that persists with overlap ≥ 0.9 and |Δω/ω| shrinking with dx
   is real at the level of this discretisation; one that disappears, or whose eigenvalue moves
   by more than the level spacing, is a discretisation artefact.
4. **Trend of the N=10k in-gap count with dx** (192³ → 256³ → 280³): monotone decrease
   toward zero → resolution artefact; stable count with stable eigenvalues and overlaps →
   physics of this finite structure (finite-size / disorder), to be stated with the ξ ≤ L/2
   ceiling as before.
5. Tiles (`make_montage.py`) and localization (`analyze_localization.py`, with the L/2
   ceiling) for every in-gap state at every dx; band-offset from the I2 count, with the
   ±2 numbering ambiguity carried, not resolved by fiat.

State the verdict in one sentence in `docs/REPORT_BANDS.md`, then the evidence. A negative
(the in-gap states are a resolution artefact) is a full result.

## Part 5 — the PRL-style report

A self-contained publication package in a **new folder `report_bands/`** at the repo root,
laid out and built exactly like the phase-3 package `report/` (read
`docs/kickoff/02_kickoff_prompt_PRL_report.md` in full first — its §1 layout, §5 working
method with the fact-check ledger, reference verification and adversarial rounds, and its §6
definition of done apply unchanged):

```
report_bands/
  main.tex          # \documentclass[prl,twocolumn,superscriptaddress,showpacs]{revtex4-2}
  supplement.tex    # Supplemental Material, REVTeX 4.2 preprint, complete, unlimited length
  refs.bib          # DOI on every entry, verified as in report/references_verified.md
  figures/          # PDF (vector where possible) + source PNG, all generated by scripts/
  tables/           # generated tables
  scripts/          # CPU-only generation scripts (lsu_ml env); reuse report/scripts/figstyle.py and common.py
  numbers.json      # every quoted number with its source file and generating script
  build.sh          # one command builds both PDFs (tectonic at /home/francisco/miniconda3/bin/tectonic; --figures regenerates everything from the saved data first)
  main.pdf, supplement.pdf
  PROGRESS.md       # decision log
  FACTCHECK.md      # claim-by-claim ledger with verifier verdicts
```

**Style target: Physical Review Letters.** Main text ≤ 3750 words-equivalent (about 4
two-column pages with figures), abstract ≤ 600 characters, at most 4 main-text figures,
numbered references. Flat, quantitative, claim-then-evidence register; no marketing
adjectives; every number with its uncertainty or the statement that it has none, plus its
grid. Author block `\author{F. Hernández}` with `\affiliation{[affiliation to be filled]}`;
invent no co-authors or affiliations. Every number is regenerated from saved data by a
script and logged in `numbers.json`; nothing is typed in by hand. No GPU solve runs from the
report scripts.

Required content, main text:
1. **Motivation**: the disordered network as a crystal of period L; why the band structure
   along the supercell path and a resolution-matched in-gap test are the natural next
   questions after the phase-3 result (pseudogap populated by localized states at N=10k,
   resolution confound left open).
2. **Conventions**: the two lattice-constant definitions, their measured values
   (a_corr with its bin/grid sensitivity), the periodic rasterization, the decoration, and
   the explicit statement that a = L/5 is retired. One sentence on what changes in the
   phase-3 numbers under the new conventions.
3. **Method**: the interior pipeline as method of record (KPM on the periodic ε, two-stage
   bandpass ChebSI, I2 completeness, I1-style parity), now at k ≠ Γ; the k ≠ Γ validation
   numbers (gates B0–B3) in one table.
4. **Band structures**: figure with the srs crystal (cell bands folded to the 5³ supercell
   zone) beside the N=1000 network along Γ–X–M–R–Γ on the same ν_cell axis; MPB parity;
   direct/indirect gap; the bandwidth-versus-ξ diagnostic.
5. **In-gap states versus resolution**: the 2 × 3 (sizes × matched voxel sizes) table of
   in-gap counts and gap edges, the positive control, the per-state tracking across dx, and
   the one-sentence verdict (artefact or physics) with the L/2 ceiling stated.
6. **Limitations** in 3–5 sentences (48³ physics is coarse; one disorder realization per
   size; binary rasterization; fp32 device arithmetic with fp64 host algebra; anything a gate
   failed).

Supplemental Material (complete): the a_corr port and its sensitivity tables; the k ≠ Γ
operator details and the band-numbering rule; the per-k cost tables; all gates B0–B5 with
their registered criteria and outcomes including failures; the full per-k eigenvalue tables
in the window; the srs r/a convention resolution; the complete in-gap state tables at every
dx with λ, ν_cell, ν_corr, residual, PR, ξ, r², overlap to the partner grid; extended figures
(full band structures, tiles of every in-gap state at every dx, KPM DOS periodic vs montage).

Verification before the report is called done: the fact-check ledger (every claim → source
file → verifier verdict), at least two adversarial rounds by subagents instructed to refute
(recorded in `docs/plans/<date>_adversarial_verification_bands.md` and summarized in
`FACTCHECK.md`), every DOI resolved, `bash report_bands/build.sh --figures` succeeding from
scratch with zero undefined references, and the page counts of both PDFs recorded in
`PROGRESS.md`.

## Order of execution and budget

Cheap and blocking first: Part 1 (CPU, hours) → Part 3a machinery + tests + cost table
(GPU hours) → Part 2 gate B0 → Part 3c srs cell (minutes) and 5³ supercell folding gate →
Part 3b N=1000 band structure (GPU hours; MPB path on the CPU in the background, days) →
Part 4 N=1000 legs (hours each) → Part 4 N=10k 256³ periodic leg (days, detached) → Part 4
N=10k 280³ leg started from the prolonged 256³ basis (about a week, detached) → Part 5 report (CPU only, from the saved data; draft the skeleton and the
convention/method sections while the long N=10k leg runs, fill the results last). Freeze the
pre-registration after the cost table and before Part 3b. Keep the gates as registered; a
failed gate with an honest explanation beats a massaged pass, in the report as much as in
the working record.

## Hazards, collected

- Off-by-two band numbering between Γ and k ≠ Γ (above). Test it.
- `mbpEnv` vs `mpb_judge`: check which environment has `mpb` before writing a script that
  calls it.
- HDF5 dataset names: MPB reads `data`; the autocorrelation reference reads `epsilon`.
- `radial_profile` quantises a_corr at the bin width; report the bin dependence.
- The FFT autocorrelation and the Bloch operator both assume a periodic ε; the montage
  convention is not periodic in the sense that matters (seam). Never mix conventions in one
  comparison without saying so.
- The k-path in MPB is in reciprocal-basis units; for `lattice (size 1 1 1)` that is 2π/L,
  identical to `k_frac`. Do not scale twice.
- At 48³ the physics is coarse; the parity gate is exact regardless, but do not quote 48³ gap
  widths as physics without the 128³ Γ value beside them.
- 12 GB card: chunk every whole-block operation at m ≥ 56 at 128³ (phase-2 lessons in
  `docs/REPORT_N10K.md`); N=10k at 256³ needs the host-resident streamed basis path.
- The interior solver has only ever been validated at Γ (gate I1). Its first k ≠ Γ use must
  be scored against a bottom-up or dense ground truth at a small grid before any production
  band-structure run; the Chebyshev spectral bounds are k-dependent.
- Grid sizes: keep every G a product of 2, 3, 5, 7 (cuFFT); 275 and 276 are not.
- Spectral prolongation is exact only between grids of the same box L and the same
  periodic ε sampled on both; never prolong across conventions or decorations.
- Do not drift back to the bottom-up solver for convenience: it is the cross-check, not the
  method. If the interior solver fails a gate, the record says so and the gate stays.
