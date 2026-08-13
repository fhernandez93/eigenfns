# Orientation + Phase-1 experiment log (2026-08-12, reconstructed after machine crash)

Everything below was measured on this machine (RTX 4080 Laptop 12 GB, JAX 0.10.0 cuda12,
env `lsu_ml`; MPB 1.11.1 CLI in env `mbpEnv`, pymeep in env `mpb_judge`). The scratchpad
copies of the scripts were lost in the crash; from here on all experiment scripts live in
`scripts/exp/` in this repo. Numbers quoted were captured before the crash.

## Settled facts about the target

- Montage `band_montage_398_607_15_non_ideal.png`: 210 tiles, 14 rows x 15 cols, bands
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

## Environment / hazards

- `stage_b_resumable.py` from the ML repo auto-resumes on boot and takes ~9 GB GPU for
  ~10 min (λ sweep for the 200x200x32 slab). One heavy GPU job at a time — check
  `nvidia-smi` before launching solver runs.
- The machine crashed mid-session (16:xx 2026-08-12) — likely the multi-hour-CUDA-run
  fragility documented in the previous project. All long runs must checkpoint+resume.
- Scratchpad (/tmp) does not survive reboots: experiment scripts and intermediate
  results belong in the repo (`scripts/exp/`, `results/` — results gitignored).
