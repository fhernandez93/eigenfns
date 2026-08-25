# eigenfns — GPU Maxwell eigenmodes of disordered LSU photonic networks

Computes the electromagnetic eigenmodes (eigenfrequencies + field
distributions) of 3-D disordered LSU photonic networks in a periodic
supercell, locally on a single consumer GPU (RTX 4080 Laptop, 12 GB), in
JAX + CUDA — the computation behind `band_montage_398_607_15_non_ideal.png`
(210 Γ-point modes, bands 398–607, straddling the photonic band gap of an
N=1000 network at bands 500|501).

Method: transverse H-field plane-wave formulation Θ H = ∇×(ε⁻¹∇×H) applied
matrix-free via FFTs (6 per application); deflated block LOBPCG with MPB's
transverse-projection preconditioner, guard warm-starting, and fp32-hardened
numerics; independent validation against MPB on identical discrete grids.
See `plans/` for the investigation report, pre-registered plan, adversarial
verification records, and `REPORT.md` (forthcoming) for results.

## Install

```bash
# solver env (JAX + CUDA)
conda env create -f environment.yml          # env name: eigenfns
# judge env (MPB, CPU) — used only by validation scripts
conda create -n mpb_judge -c conda-forge python=3.11 pymeep pymeep-extras numpy h5py -y
conda install -n mbpEnv -c conda-forge mpb -y   # CLI mpb (or use any env with the mpb binary)
```

The structure files come from the companion repos (`Create LSU Structures -
ML /Example/*_ends.txt` and `Create LSU Structures  - Claude/Example/`); paths
with trailing/double spaces are intentional.

## One-command run

```bash
# full band window of the montage, checkpointed + resumable:
conda run --no-capture-output -n lsu_ml python scripts/run_modes.py \
  "/home/francisco/Documents/Create LSU Structures  - Claude/Example/N1000_lsu_example_ends.txt" \
  --grid 128 --band-lo 398 --band-hi 607 --resume

# regenerate the montage from the run outputs:
conda run -n lsu_ml python scripts/make_montage.py results/<tag>

# validation gates (crystal parity, literature, disordered parity, ...):
conda run -n lsu_ml python scripts/validate.py --all
```

Band indices are MPB-numbered (bands 1–2 at Γ are the ω=0 modes). All long
runs checkpoint per locked block and auto-resume (`--resume`).

## Interior gap-window pipeline (N=10k project, 2026-08-17+)

For windows deep in the spectrum (e.g. ~band 5,000 of the N=10,000 network)
where bottom-up is infeasible (measured: 1.38 TB locked set), the two-stage
bandpass ChebSI solver targets a λ-window directly
(`plans/2026-08-18_interior_preregistration.md`):

```bash
# full-bandwidth KPM DOS + counting (locates the gap, derives the window):
conda run --no-capture-output -n lsu_ml python scripts/exp/exp_kpm_dos.py \
  Structures/20260701_N10000_lsu_generated.txt --grid 256 \
  --radius 0.331836 --aspect 1.0 --eps-rod 8.41 --degree 12000 --tag n10k_dos
conda run -n lsu_ml python scripts/exp/exp_kpm_analyze.py \
  results/exp/n10k_dos_kpm.npz --gap-guess 1.95 --plot dos.png

# interior window solve (per-outer checkpoints, resumable):
conda run --no-capture-output -n lsu_ml python scripts/run_interior.py \
  Structures/20260701_N10000_lsu_generated.txt --grid 192 \
  --lam-lo 1.757 --lam-hi 1.930 --m 104 --build-degree 3000 --build-outers 2 \
  --polish-degree 12000 --polish-outers 4 --tag n10k_G192_Sbelow --resume

# completeness audit (gate I2) + parity scoring vs a bottom-up reference:
conda run -n lsu_ml python scripts/exp/exp_i2_completeness.py \
  --rundir results/n10k_G192_Sbelow --gate-name "I2 (S_below)"
conda run -n lsu_ml python scripts/exp/exp_i1_score.py --interior results/<tag> \
  --reference results/prod_N1000_G128 --ref-lo 395 --slice-lo 473 --slice-hi 523 \
  --gate-name I1

# same solve on a PERIODICALLY WRAPPED structure (convention change — see below):
conda run --no-capture-output -n lsu_ml python scripts/run_interior.py \
  Structures/20260701_N10000_lsu_generated.txt --grid 192 \
  --lam-lo 1.855 --lam-hi 2.000 --m 30 --periodic --tag n10k_gap_periodic

# merge slices into one eigenvalue-ordered window (dedup by eigenvector overlap):
conda run -n lsu_ml python scripts/merge_slices.py --out results/n10k_G192_window \
  results/n10k_G192_Sbelow results/n10k_G192_Sgap results/n10k_G192_Sabove

# localization (IPR + xi with the L/2 finite-size ceiling):
conda run -n lsu_ml python scripts/analyze_localization.py \
  results/n10k_G192_Sbelow results/n10k_G192_Sabove --box 24.6467 \
  --out results/n10k_localization

# montage for interior runs (band offset from the I2-certified count):
conda run -n lsu_ml python scripts/make_montage.py results/n10k_G192_Sbelow \
  --band-offset <MPB band of stored mode 0> --band-lo <lo> --band-hi <hi>
```

### Rasterization conventions (read before comparing runs)

`rasterize_penlike` defaults to the **montage convention**, bit-for-bit as the
parent notebook: binary voxels, and rods whose *radius* pokes through a box
face are NOT wrapped. That last detail was quantified on 2026-08-24: it leaves
the outermost voxel shell with **ff = 0.1975 against 0.2211 in the interior**
(11% material deficit), a thin seam on the box faces. In a structure with a
photonic gap that seam behaves like a planar defect and hosts spurious
localized gap states — four were found and discounted in the N=10k window (see
`plans/2026-08-17_adversarial_verification.md`, round 2).

`periodic=True` (CLI `--periodic`) wraps the voxel indices so every rod
contributes its full volume. Verified surgical: only voxels within 3 of a face
change (0.078% of the grid), outer-shell ff → 0.2177. It is a **convention
change** relative to the reference montage, so results computed with it are not
directly comparable to montage-convention results without re-validation.

## Hardware notes

- One heavy GPU job at a time (12 GB card; `run_modes.py` refuses to start
  over a foreign >2 GB GPU process — `--force` to override).
- fp32 discipline: TF32 must stay disabled in solver matmuls
  (`precision=HIGHEST` — already in the code); all small dense algebra runs
  in fp64 on the host. See `plans/2026-08-12_orientation_and_experiments_log.md`.
- JAX 0.10.0 + cuda12; the `cuda_executor ... driver version` stderr warning
  is benign on this driver.

## Layout

| path | what |
|---|---|
| `eigenfns/` | library: operator, solver, chebyshev (window/counting), structure (rasterizer), render, io |
| `scripts/run_modes.py` | compute a band window for a structure (checkpointed) |
| `scripts/make_montage.py` | render per-band ε\|E\|² tiles + assemble the montage |
| `scripts/validate.py` | the validation gates vs MPB |
| `scripts/exp/` | Phase-1 experiment scripts (measurements in `plans/` log) |
| `tests/` | unit tests (operator analytics, rasterizer golden values, solver-vs-dense) |
| `plans/` | investigation report, pre-registered plan, adversarial verification, experiment log |
