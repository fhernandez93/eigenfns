# Pre-registered plan — GPU Maxwell eigenmodes of disordered LSU networks

**Status: FROZEN 2026-08-13 (E3/E4/srs numbers in; user may amend).** Template:
the ML project's 2026-07-21 plan. Machine: RTX 4080 Laptop 12 GB, 62 GB RAM,
32 threads. Envs: `lsu_ml` (JAX 0.10.0 cuda12), `mbpEnv` (MPB 1.11.1 CLI),
`mpb_judge` (pymeep, py3.11).

**E-numbers backing this plan** (experiment log): E3: 680 bands @64³ in 941.8 s
(129,728 Θ applications), gap at MPB 500|501 as predicted (§2b PASSED), KPM
counts consistent; E4: c64 error ≤ 2.4e-7 (gate G9 PASSED at 48³); srs scan:
28.0% gap at optimum (G2 material); MPB parity 4.3e-6 @32³ (G1/G3 material).

## 0. What Phase 1 established (all adversarially verified 2026-08-12/13)

- Target: Γ-point eigenmodes, bands ~398–607 (MPB numbering) of N=1000 LSU
  networks; ε(r) = binary pen-like rasterization (r=0.2252 µm, z-warp s=2.5,
  ε=8.5703/1), 0.5 bands/vertex, gap at 500|501.
- Operator: transverse 2-component H-field, 6 FFTs/apply; validated to 2.6e-7
  vs analytics; 2.7 ms/vector @128³ c64 batched (28.6 ms @256³).
- Solver: deflated block LOBPCG + MPB fancy preconditioner + guard warm-start:
  18–26 iters/block; two-implementation parity vs MPB 4.3e-6 over 148 bands @32³.
- Judge protocol: MPB epsilon-input-file run → read back its `-epsilon.h5:data`
  → our solver on that exact grid (interpolation-proof).
- Field quantity: ε|E|² (Joannopoulos/MPB dpwr convention).

## 1. Frozen methodological choices

1. **Formulation**: transverse H-field plane-wave, matrix-free FFT application,
   MPB-equivalent discretization (pointwise scalar ε⁻¹ on the grid).
2. **Solver**: bottom-up deflated block LOBPCG (`eigenfns/solver.py`) with the
   MPB transverse-projection preconditioner, block m and guard g per the E3-
   informed table below; locked vectors on GPU up to VRAM budget, streamed from
   host beyond. Fallback (pre-registered, only if the 128³ production run
   exceeds 2× the projected wall-clock): ChebSI bottom-up (`eigenfns/chebyshev.py`).
3. **Grid resolution (production)**: **128³** for the N=1000 window solve
   (dx = 0.0894 µm; finer than Sellers's ~16/cell ≈ 80³ convention), with the
   256³-native ε consumed by solving on the 256³ grid's 2× box-average
   downsample **[decision: plain 2³ mean vs (r,s)-preserving re-rasterization
   at 128³ — re-rasterization chosen: it is the same convention at the target
   resolution, not a new smoothing scheme]**. Convergence gate: ω(G) sweep at
   G = 96/128/160(/192 if VRAM permits with streaming) on one structure for
   bands {398, 450, 500, 501, 550, 607}; report extrapolated Δω/ω.
4. **Precision policy**: vectors/operator complex64 on GPU; ALL ≤3m-dense
   algebra fp64 on host; locking tolerance rel-res ≤ 1e-4 (Weyl-guaranteed
   Δω/ω ≤ 5e-5). E4 quantifies the c64-vs-c128 spectrum error on 48³ — gate:
   c64 must match the c128 reference to Δω/ω ≤ 1e-4 over 96 bands, else the
   policy is revisited (options: fp64 Gram accumulation via chunked einsum on
   CPU, iterative refinement pass).
5. **Band numbering**: outputs use MPB-compatible numbering (the two ω=0 Γ
   modes are bands 1–2; our nth computed mode is band n+2), flagged in every
   artifact. (±2 ambiguity vs the original montage is recorded; one flag flips
   the convention.)
6. **Montage rendering**: ε|E|² per band, 15 tiles/row, band-index labels in
   filenames (+ optional on-tile labels), pyvista off-screen volume render,
   white background, camera/style tuned to the reference (qualitative gate).

## 2. Pre-registered validation gates (numbers frozen before production runs)

| gate | test | tolerance |
|---|---|---|
| G1 crystal parity | srs 8-vertex cell + 2×2×2 supercell (64 vertices), Γ + 3 k-points, judge protocol grids at 32³/64³: ours vs MPB | Δω/ω ≤ 1e-4 per band (measured margin 4.3e-6 at 32³ informs this; tighter than kickoff's 1e-3) |
| G2 literature reproduction | srs at Sellers SNG params (ε=13, r/a=0.2554, MPB object geometry, res 48): gap between conventional-cell bands 4|5 | gap Δω/ω = 28.06% ± 1.5 pp (their res-16 vs our res-48 discretization difference bounded by our own res sweep) |
| G3 disordered parity | N=1000 gold structure, judge-protocol grid at 64³, bands 1–300 + window sample up to ~650: ours (c64, tol 1e-4) vs MPB CLI (tol 1e-7) | Δω/ω ≤ 1e-4 per band; mode-overlap subspace check on 5 sampled clusters ≥ 0.99 (principal angles for degenerate clusters) |
| G4 degeneracy handling | clusters with spacing < 1e-3·λ compared as subspaces | principal angles: subspace overlap ≥ 0.99 |
| G5 convergence | ω(G) sweep (see §1.3) + ω(tol) at tol 1e-3/1e-4/1e-5 on 20 sampled window bands | monotone trend, reported extrapolation; tol-1e-4 vs 1e-5 shift ≤ 2e-5 |
| G6 completeness | (a) locked-value monotonicity (no out-of-order recoveries); (b) deflated-probe KPM count of missed eigenvalues below λ(band 620) | (a) zero violations; (b) estimate = 0 with SE ≤ 0.5 |
| G7 residuals | rel-res of every reported band; block orthonormality ‖V†V−I‖_max incl. locked set | ≤ 1e-4; ≤ 1e-3 (fp16/streaming floor if used: documented) |
| G8 montage | regenerate 398–607 montage for the montage-convention N=1000 structure at 256³ ε (solved at 128³ per §1.3) | 14×15 layout; grey-window position: the DOS-minimum band range must straddle 500|501 (±2 numbering flag); side-by-side figure in report |
| G9 precision | E4: c64 vs c128 (fixed fp64 basis) at 48³, 96 bands | Δω/ω ≤ 1e-4 |

Rules: no gate weakening — a fail is reported as a fail with diagnosis; every
gate's outcome goes through an adversarial pass before being claimed.

## 2b. Pre-registered E3 expectation (written before E3 finished)

At 64³ on the N=1000 gold structure, the largest spectral gap in the interior
should fall between **solver bands 498|499** (= MPB bands 500|501). If it
falls elsewhere by more than ±2, the band-arithmetic claim or the solver's
completeness is wrong and Phase 2 does not freeze until resolved. Note the
gold structure is a *well-annealed* network — Sellers's ensembles A–C show
in-gap defect modes for poorly-annealed samples; a handful of near-gap
localized modes narrowing the clean gap is acceptable, a displaced gap is not.

## 3. Performance target (pre-registered)

Full window (bottom-up to band ~620 + guard) for one N=1000 structure at 128³:
**target ≤ 4 h wall-clock; hard cap 12 h** (beyond that the ChebSI fallback is
triggered). Projection basis (measured E3): 941.8 s for 680 bands at 64³; the
per-iteration cost is Θ-and-GEMM dominated, both O(G³ log G / G³) → ~8× at
128³ ⇒ **~2.1 h projected** (locked storage: 680 × 33.5 MB = 22.8 GB exceeds
VRAM ⇒ host-resident locked buffer streamed through the fixed 128-vector
deflation chunks; PCIe adds ~0.2-0.5 s/iteration ≈ +25-50% ⇒ still ≤ 4 h).
Montage render: ≤ 30 min for 210 tiles. Checkpoint/resume: after every locked
block; auto-resume; single-GPU discipline (refuse to start over a foreign job).
Allocation env: XLA_PYTHON_CLIENT_PREALLOCATE=false, MEM_FRACTION=0.90.

## 4. Deliverables & layout (mirrors the previous project)

`eigenfns/` (operator, solver, chebyshev, structure, render, io), thin CLIs
(`scripts/run_modes.py`, `scripts/validate.py`, `scripts/make_montage.py`),
`notebooks/frontend.ipynb`, `tests/` (incl. solver-vs-dense at G=8, golden ff
with provenance + skipif, judge-protocol smoke), `README.md`, `docs/REPORT_N1000.md`,
`docs/plans/` records. Heavy artifacts in `results/` + `Structures/` (gitignored),
regenerable by script.
