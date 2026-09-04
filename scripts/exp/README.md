# scripts/exp/ — experiments, gates, fixes, run chains

All run in the `lsu_ml` conda env. Outputs go to `results/exp/` or
`results/gates/` unless a `--tag`/`--out` says otherwise. Gate scripts append
their verdict to `results/gates/gate_results.json`.

## Phase 1, N=1000 (2026-08-12 → 08-14)

| script | what |
|---|---|
| `exp_render_style.py` | prototype of the montage tile renderer (grey network volume + ε\|E\|² overlay) |
| `exp_parity32.py`, `exp_parity32_mpb.sh` | our solver vs MPB on MPB's effective 32³ grid (the .sh rasterizes and runs MPB) |
| `exp_parity64.py` | gate G3: disordered parity vs MPB, 64³, 300 bands |
| `exp_parity64w.py` | gate G3w: full-window parity, 660 bands, 64³ |
| `exp_e3_full.py` | E3: full bottom-up solve of the gold structure, binary montage convention |
| `exp_precision48.py` | E4: complex64 vs complex128 eigenvalues on a 48³ case (precision policy) |
| `exp_g4_degeneracy.py` | gate G4: degenerate-subspace principal angles vs MPB H-fields |
| `exp_srs_literature.sh` | gate G2: ideal srs crystal at the literature parameters, MPB |

## Phase 2, N=10k feasibility (2026-08-17 → 08-18)

| script | what |
|---|---|
| `exp_ff_calibration_n10k.py` | bisect the rod radius until the measured filling fraction is 22.0 % |
| `exp_kpm_dos.py` | full-bandwidth KPM density of states / eigenvalue counting (stochastic Chebyshev moments) |
| `exp_kpm_analyze.py` | from saved moments: counting function, DOS, gap location, window derivation, plot |
| `exp_bakeoff.py` | method bake-off (folded spectrum, shift-invert, hybrid, bandpass ChebSI) on N=1000 vs ground truth |
| `exp_gap_leakage.py` | is the KPM in-gap count distinguishable from Jackson-kernel edge leakage? |

## Interior gates (I1–I8) and the seam test

| script | what |
|---|---|
| `exp_i1_score.py` | gates I1 / I4: score an interior run against a bottom-up reference (parity, ghosts, missed) |
| `exp_i2_completeness.py` | gate I2 v1: deflated-probe KPM completeness audit (recorded as a failed design) |
| `exp_i2_v2.py` | gate I2 v2: completeness audit with the edge-leakage-corrected estimator |
| `exp_i3_i5_score.py` | gates I3 (residuals + orthonormality) and I5 (spectrum consistency) |
| `exp_i3_gram_diagnose.py` | why the merged window fails I3 (cross-slice Gram deviation, seam artefact) |
| `exp_i8_score.py` | gate I8: cross-solver localization (ξ) agreement |
| `exp_crossgrid_match.py` | gate I6: match 192³ and 256³ eigenmodes by eigenvector overlap |
| `exp_periodic_match.py` | seam test: match montage-convention in-gap states to periodic-rasterization ones |
| `exp_periodic_verdict.py` | seam test verdict (montage vs periodic rasterization) |
| `exp_rare_regions.py` | why N=1000 has a clean gap and N=10k a pseudogap: rare-region statistics |
| `exp_rare_region_modes.py` | do the in-gap modes live in the rare regions? |

## One-off fixes and recovery (applied once, kept for provenance)

| script | what |
|---|---|
| `fix_rayleigh_norm.py` | retroactively correct eigenvalues computed with an unnormalized Rayleigh quotient (writes `RAYLEIGH_CORRECTION.md` in each run dir) |
| `fix_i2_leakage.py` | recompute the I2 v2 edge-leakage term (three self-caught defects) |
| `fix_localization_lam.py` | refresh stale λ labels in `localization_modes.json` (no re-fit) |
| `recover_energy_density.py` | rebuild `window_energy_density.npy` + `interior_report.json` from a checkpoint after a save-stage crash |

## Detached run chains (historical; dated; run with `setsid nohup`)

| script | what |
|---|---|
| `overnight_20260817.sh` | hybrid polish continuation → I4 bottom-up reference |
| `chain_gpu_20260818.sh` | I4 resume → I1 run → I1 score |
| `chain_i1_20260818.sh` | gate I1 interior run + score, after the I4 run frees the GPU |
| `chain_prod_20260818.sh` | production N=10k S_below → S_above (first version) |
| `chain_prod2_20260818.sh` | Amendment A1: I1 completion with polish cap 6 → re-score → production |
| `chain_prod3_20260818.sh` | production v3 with the host-resident streamed basis (the one that ran) |
| `chain_post_20260821.sh` | post-production: S_gap addendum slice → merge → gates |
| `chain_final_20260824.sh` | reprioritised after the seam finding: periodic re-solve first |
| `chain_gates_20260824.sh` | I2 v2 + I3/I5 scoring after the post-production chain |
| `chain_i6resume.sh` | resume the 160³ I6 run killed by a GPU-wait race |
| `chain_post_20260826.sh` | I4-interior halves, 256³ anchors |
| `chain_periodic_redo_20260827.sh` | periodic re-solve v2 (larger basis) |
