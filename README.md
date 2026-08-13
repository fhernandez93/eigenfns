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
