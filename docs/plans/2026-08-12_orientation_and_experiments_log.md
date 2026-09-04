# Orientation + Phase-1 experiment log (2026-08-12, reconstructed after machine crash)

Everything below was measured on this machine (RTX 4080 Laptop 12 GB, JAX 0.10.0 cuda12,
env `lsu_ml`; MPB 1.11.1 CLI in env `mbpEnv`, pymeep in env `mpb_judge`). The scratchpad
copies of the scripts were lost in the crash; from here on all experiment scripts live in
`scripts/exp/` in this repo. Numbers quoted were captured before the crash.

## Settled facts about the target

- Montage `docs/reference/band_montage_398_607_15_non_ideal.png`: 210 tiles, 14 rows x 15 cols, bands
  398→607, grey network + orange field renders; tiles in rows ~6-7 (~bands 475-500) go
  nearly grey → the window straddles a gap-like DOS minimum there.
- ε(r) convention (from `20250903_create_h5_from_ends.ipynb` + user statement): binary
  voxels, flat-capped cylinders, elliptical cross-section via global z-warp aspect 2.5,
  eps_rod = 2.9275² = 8.5703, eps_bg = 1, MINOR_RADIUS = 0.2252 µm. **Measured: at
  256³, N=1000 gold structure → ff = 0.2172** ("ff ~22%" confirmed; parameters pinned).
  Reimplemented as `eigenfns/structure.py::rasterize_penlike` (validated vs notebook logic).
- **Band arithmetic (measured, MPB)**: ideal srs crystal, 8-vertex cubic cell a = 11.44/5,
  circular rods r = 0.2252, ε = 8.5703 → complete gap **between bands 4 and 5**, ~9.9%
  (quick settings: res 32 mesh 2, 5 k-points). → 0.5 bands/vertex → N = 1000 supercell:
  gap between bands 500/501. Montage window 398–607 straddles exactly that. Montage
  structure size = N = 1000. (Cross-check vs literature pending.)

## Operator (validated, in `eigenfns/operator.py`)

Transverse 2-component spectral representation (MPB's): curl diagonal in the (t̂₁,t̂₂)
frame, 6 FFTs per application; G=0 slots zeroed (exact Γ zero-mode deflation).
- Dense-validated vs analytic homogeneous spectrum at G=8: max rel err 2.6e-7 (c64),
  null dim exactly 2.
- **Measured cost (batched, c64)**: 128³: **2.7 ms/vector**; 256³: **28.6 ms/vector**
  (m≤4 chunks at 256³ — m=8 OOMs 12 GB). Naive 3-component version was 51 ms @128³
  (13 FFTs + poor fusion) — the transverse form is ~19× better. Gram (m x m) @128³:
  ~11 ms at m=32.
- Memory: 33.5 MiB/vector @128³ c64; 268 MiB @256³. 220-vector block: 7.4 GB @128³.

## Judge (MPB) facts

- `pymeep` needs python ≤3.11 (env `mpb_judge`); CLI `mpb` 1.11.1 works in `mbpEnv`.
- MPB CLI, 32³, 150 bands, tol 1e-9, epsilon-input-file: **6m40s** single process.
- **Object geometry → MPB applies tensor subpixel smoothing regardless of mesh-size**
  (epsilon.h5 has ε_xy up to 1.74). File-input → pure scalar: `data` == `epsilon.xx`,
  off-diagonals 0, `epsilon_inverse.xx` == 1/`data` exactly.
- MPB's file interpolation rule is NOT a simple half-offset 8-point average (delta-probe
  says out[i] = mean src{i-1,i}³, but that fails on a real grid — unresolved, and
  deliberately sidestepped): **judge protocol = feed MPB the file, then read back
  `*-epsilon.h5:data` (the exact scalar grid MPB used) and run OUR solver on that grid.**
  Both solvers then solve the identical discrete problem; interpolation becomes moot.
- Parity smoke (32³ disordered, MPB's grid): our band 1 λ=0.1290 vs MPB band-3 λ=0.1288
  (**dω/ω ≈ 8e-4**) where our block converged. Blocks beyond the first stalled (below).

## Block solver status (LOBPCG prototype — being rewritten as `eigenfns/solver.py`)

Bugs found and fixed (keep these in the library version):
1. SVQB/whitening drop thresholds must sit ABOVE the fp32 Gram noise floor (~1e-5·max),
   else noise directions get amplified 10³-10⁵× and poison RR with fake low Ritz values.
2. All small dense algebra (Gram eigh, RR) on host in float64 — c64 GEMM with c128
   accumulation is unsupported on GPU (cuBLAS "Unexpected GEMM dtype: c64 c64 c128").
3. Tracked H-products drift ~1e-3·||HW|| per c64 combine — swamps low eigenvalues.
   Recompute HX fresh every iteration (θ is cheap).
4. Re-deflate X every iteration: roundoff-reintroduced locked components regrow under
   RR minimization (leak confirmed: block 3 rediscovered locked-range eigenvalues).
5. Dead (rank-dropped) rows → +penalty on RR diagonal, not zero (zero → fake λ=0 pairs).
6. Guard empty P (full rank collapse) → set P=None.
7. Preconditioner: diagonal kinetic 1/(|k+G|²+σ). σ=median(λ) vs σ≈0 makes no difference
   to the block-2 stall (tested).

**RESOLVED 2026-08-12 (post-crash): the stall was the preconditioner + two block
hygiene bugs.** Three changes turned the stalled solver into an MPB-grade one:
1. **MPB's transverse-projection ("fancy") preconditioner** (JJ01 Eq. 14, from
   `maxwell_pre.c:maxwell_preconditioner2`): invert the diagonal curl (divide by
   |k+G| with the 90° rotation structure), IFFT, **multiply by ε(r)**, FFT, invert
   the curl again — same 6-FFT cost as Θ. Implemented as
   `MaxwellOperator.precondition`; the old diagonal kinetic one kept as
   `precondition_simple`.
2. Warm-start each block with the previous block's guard Ritz vectors.
3. Unit-normalize random block rows (norm-360 raw rows made SVQB's relative drop
   threshold discard the unit-norm carries), and mask dead rows out of the
   convergence/locking path (they report fake λ=0, res=0).

**Result (32³, MPB's own grid, 160 bands, tol 1e-4, c64):** blocks converge in
24/18/26/2 iterations; **parity vs MPB (tol 1e-9, fp64): max Δω/ω = 4.3e-6,
median 8.6e-7 over 148 bands**; 108 s on CPU. Commit b6256cc+1.

Key MPB-source facts (agent report, 2026-08-12): MPB 1.11.1 applies Kottke tensor
smoothing to *geometric objects* regardless of mesh-size (true mesh-size-1 bypass
only in MPB 1.12, PR#150, 2025-04); file input is scalar trilinear pixel-centered
interpolation with edge clamping (not periodic wrap) — explains the 'data' ≠ file
mismatch we measured; grid readback protocol remains the right judge design.
Consider upgrading judge env to MPB ≥1.12 for exact binary-grid parity runs.
MPB defaults: eigensolver-block-size -11, tolerance 1e-7, Fletcher-Reeves (PR
needs nwork≥4), Moré-Thuente exact line search, deflation against all previous
bands. Python ModeSolver: no MPI, no MaterialGrid; epsilon_input_file or callable
default_material.

**OLD open problem text (kept for the record; superseded above):** with blocks of m=96 (lock 48, guard 48),
block 1 converges (res ~1e-2 at 250-400 iters; its lowest bands agree with MPB to 8e-4),
but blocks 2+ stall at res ~0.9 with Ritz values far above the true continuation (e.g.
block 2 at λ≈0.64-0.75 when MPB says bands 49-150 live at 0.13-0.23). Deflation-related;
next step was per-iteration instrumentation of block 2 (Ritz trajectory, residual
quantiles, locked-leakage ‖⟨locked,X⟩‖). NOTE: MPB itself converges 150 bands at 32³
with 11-band blocks + deflation in ~7 min tol 1e-9, so block+deflation is fine in
principle — our implementation differs in preconditioner (MPB: transverse-projected
kinetic with ⟨ε⁻¹⟩ scaling), line search (MPB: exact trace-min CG), and possibly the
guard policy. Alternatives if the stall resists: (a) MPB-style small blocks (~16) with
deflation; (b) one full-width block (needs host streaming ≥128³); (c) Chebyshev filter
for the window + count verification.

## Literature-reproduction reference (srs, ε=13) — measured 2026-08-13

MPB CLI, conventional 8-vertex cubic cell, cylinder rods, res 32, tol 1e-6,
21-point k-sampling (quick settings): gap between bands 4|5 (=primitive 2|3):

| r/a_c | gap Δω/ω |
|---|---|
| 0.090 | 21.2% |
| 0.105 | 25.9% |
| 0.117 | 27.8% |
| 0.130 | **28.0%** |
| 0.145 | 26.9% |

Optimum ≈ 28.0% at r/a_c ≈ 0.12–0.13 — reproduces Sellers's published 28.06%
(SNG, ε=13); r/a_c≈0.117 gives ff≈18% matching their 17.88%. The research
agent's "r/a = 0.2554" is a different (unidentified) unit convention — at face
value it gives a 2.9% gap (measured) and >50% ff, inconsistent with their own
ff; our scan recovers their actual geometry. G2 gate material.

## TF32 lesson (2026-08-13)

On the RTX 4080 (Ada), XLA's default matmul precision runs fp32 GEMMs as TF32
(10-bit mantissa). Consequence, measured at 64³ GPU: Gram/rotation accuracy
floors at ~1e-3 and block residuals plateau at ~7e-3 — the 32³ parity run
converged only because it ran on CPU (true fp32). Foreseen verbatim by the
numerics survey ("Gram matrices in true fp32 — disable TF32"). Fix: every
matmul/tensordot in the solver now carries precision=HIGHEST. FFTs (cuFFT) are
unaffected by TF32.

## XLA fixed-shape lesson (2026-08-13, cost ~1 h of debugging)

The solver's P-block whitening originally SHRANK P to its kept rank — a
different array shape almost every iteration → XLA recompiled every downstream
kernel every iteration: GPU idle at 0%, ~22 CPU cores of compiler churn,
essentially zero progress at 64³ (the 32³ parity run escaped this only because
CPU compilation is cheap). Fix: all block shapes are static — dropped
directions become zero rows (the dead-row penalty machinery already handles
them). Rule for the library: **every jitted computation must see a fixed set
of shapes across the whole run**; rank adaptivity is expressed by zero rows,
never by shape changes.

## E3 — full 64³ spectrum of the N=1000 gold structure (COMPLETE, 2026-08-13)

680 bands, m=64, guard=24, tol 1e-4, c64 GPU: **941.8 s wall, 129,728 operator
applications** (17 blocks, 21–51 iters each, all locked ≤ 1e-4; monotone
spectrum, no out-of-order recoveries).

- **The largest interior gap falls between solver bands 498|499 = MPB bands
  500|501 — exactly as pre-registered** (λ 1.8831 → 1.9739; Δν/ν = 2.35% at
  64³, ν_center ≈ 0.516 in a = 2.288 µm units). Third independent confirmation
  of the band arithmetic, now on the actual disordered montage-convention ε.
- Montage window MPB 398–607 ↔ ν 0.4509–0.5674 (a=2.288); gap ~2.4% wide at
  64³ (vs crystal 28% — disorder + ε 8.57 + elongated rods narrow it heavily,
  consistent with the montage's ~1.5 rows of grey tiles).
- Low-band jumps (after solver bands 12, 36, 52, 64, 160: 37%, 18%, 11%, 6%,
  6%) are real finite-size shell structure. **KPM verification: count below
  λ=0.94 = 159.9 ± 0.8 vs 160 locked (exact); below λ=1.93 = 493 ± 3 vs 498
  locked** (consistent: Jackson smearing width 0.18 > gap width 0.09 at degree
  800 — sharp counts need the deflated-probe variant, as pre-registered).
- E4 (48³): c64 tol 1e-4 vs c128 tol 1e-6 over 96 bands: **max Δω/ω =
  2.4e-7, median 4.9e-8** — precision gate passed with 400× margin.

## 128³ RESOLVED (2026-08-14): the full memory campaign, final state

Smoke: 64 bands at 128³ in 771 s (4 blocks, m=32, guard=12, theta_chunk=8,
host-streamed locked storage), zero OOM. Production launched (bands→611).
The complete recipe that fits 12 GB at 128³:
- allocator: **XLA_PYTHON_CLIENT_ALLOCATOR=platform** (JAX's env var —
  TF_GPU_ALLOCATOR is a TensorFlow variable and was a silent no-op; BFC
  fragments; cuda_async's up-front reservation collides with the desktop's
  ~1.3 GB) + --xla_gpu_enable_cublaslt=false --xla_gpu_autotune_level=0
  (cublasLt autotune profiling wants ~2 GB scratch it can't get);
- m=32 block, theta_chunk=8 (bounds cuFFT batch workspace), deflate chunks
  ~0.75 GB with a **block_until_ready barrier per chunk** (async dispatch
  otherwise keeps every X-version live: ~10 GB transient);
- chunk-assembled Rayleigh-Ritz (no stored HW/HP: −2 persistent blocks);
- **block-boundary cleanup of ALL stale arrays (X, HX, Xl, Rf, R, W)** — the
  final-residual Rf and the last iteration's R/W retained ~3.2 GB invisibly;
  this was the last OOM standing (diagnosed by step-wise nvidia-smi probes
  after allocator stats proved misleading: platform reports 0 in_use).
Steady-state ~4.4 GB peak per iteration + streamed chunks; ~5.5 GB headroom.

## The 64³→680-band memory war (four real bugs, all now fixed)

1. Temporaries' lifetimes (R, W, HW, block lists) → ~2× peak → OOM; fixed with
   explicit del.
2. Growing locked array → new gram/deflate shapes per block → XLA recompile +
   late-run autotune scratch OOM at 440/680; fixed with fixed-capacity buffer.
3. gram(full locked buffer, X) conjugates a full 2.9 GB copy per call; fixed
   with chunked deflation (fixed 128-vector chunks — also the streamed-host
   architecture 128³ needs).
4. buf.at[].set() materializes a second full buffer copy per block; fixed with
   a donated dynamic_update_slice (true in-place).
   Allocation env for production: XLA_PYTHON_CLIENT_PREALLOCATE=false,
   XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 (desktop holds ~1-1.5 GB; the default
   0.75 fraction caps the pool at 9.2 GB).

## G3 disordered parity (64³, first 300 bands) — PASSED 2026-08-13

MPB CLI: 300 bands, 64³, tol 1e-7, file-input protocol: 2h22m CPU. Our solver
on MPB's exported grid (c64, tol 1e-4, m=40, host-streamed, CPU run): 33 min.
**Parity over 298 bands: max Δω/ω = 8.95e-6, median 1.43e-6, q99 7.5e-6 — gate
(≤1e-4 per band) passed 11×.** Worst bands are near-degenerate pairs (22/23,
132/133), still <1e-5. Mode-overlap subspace checks: Phase-4 batch.
Same-resolution calibration: MPB 300 bands = 2h22m CPU vs our 680 bands =
15.7 min GPU.

## G6 amendment (2026-08-14) — honest record

The gate as first implemented (deflated-probe KPM count below band 619's λ at
degree 800) measured **86.6 ± 0.9 phantom "missed" bands** — an artifact, not
a finding: the Jackson smearing half-width at that (λ_b, λ_max, degree) is
Δλ ≈ 0.38, which spans ~80 *unlocked* bands above the threshold, each leaking
partial weight into the count. (That 86 bands were NOT actually missing is
established independently: G3w matches MPB band-by-band through the window,
and the gap appears at locked index 498 — 86 missing bands would displace it
to ~414.) Amended gate (SC1a-style): threshold at MID-GAP — the transition
window then contains no eigenvalues — with degree chosen so the smearing fits
inside the measured gap width, and deflation against exactly the bands below
the gap. Expected count 0.

## Production run (2026-08-14)

611 bands at 128³: solve **19,908 s = 5h32m** (31 blocks, m=32, guard=12,
tol 1e-4, host-streamed locked set; over the 4 h target, inside the 12 h cap —
honest miss, driven by streamed-deflation growth late in the run). The
post-solve packaging was killed (host-RAM spike duplicating the 20.6 GB locked
set); all 31 block checkpoints survived and outputs were rebuilt from them
without re-solving (block-wise window gather; run_modes to be patched the same
way). Gap at MPB 500|501, Δν/ν = 2.08% at 128³ (G8 PASS); G7 residuals PASS.

## Environment / hazards

- `stage_b_resumable.py` from the ML repo auto-resumes on boot and takes ~9 GB GPU for
  ~10 min (λ sweep for the 200x200x32 slab). One heavy GPU job at a time — check
  `nvidia-smi` before launching solver runs.
- The machine crashed mid-session (16:xx 2026-08-12) — likely the multi-hour-CUDA-run
  fragility documented in the previous project. All long runs must checkpoint+resume.
- Scratchpad (/tmp) does not survive reboots: experiment scripts and intermediate
  results belong in the repo (`scripts/exp/`, `results/` — results gitignored).
