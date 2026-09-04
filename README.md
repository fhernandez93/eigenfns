# eigenfns — GPU Maxwell eigenmodes of disordered LSU photonic networks

Computes the electromagnetic eigenmodes (eigenfrequencies + field distributions)
of 3-D disordered LSU photonic networks in a periodic supercell, locally on a
single consumer GPU (RTX 4080 Laptop, 12 GB), in JAX + CUDA.

Method: transverse H-field plane-wave formulation Θ H = ∇×(ε⁻¹∇×H) applied
matrix-free via FFTs (6 per application); deflated block LOBPCG with MPB's
transverse-projection preconditioner for bottom-up band windows, and a two-stage
bandpass Chebyshev subspace iteration for windows deep in the spectrum;
fp32-hardened numerics; independent validation against MPB on identical
discrete grids.

The work happened in three phases, each with its own record:

| phase | dates | what | record |
|---|---|---|---|
| 1 — N=1000 bottom-up | 2026-08-12 → 08-14 | reproduce the cluster montage `docs/reference/band_montage_398_607_15_non_ideal.png` (210 Γ-point modes, bands 398–607, gap at 500\|501) | `docs/REPORT_N1000.md` |
| 2 — N=10,000 interior | 2026-08-17 → 08-31 | KPM density of states + interior gap-edge modes near band 5,000; localization; the rasterizer boundary-seam finding | `docs/REPORT_N10K.md` |
| 3 — PRL manuscript | 2026-08-31 → 09-04 | main text + Supplemental Material, every quoted number regenerated from the saved data | `report/` |

## Repository map

```
README.md        this file
eigenfns/        the library: operator, solver, chebyshev, interior, structure (rasterizer), localization, render, io
scripts/         command-line entry points (table below); scripts/exp/ = experiments, gates, one-off chains
tests/           pytest unit tests (~20 s; the 18-min solver-vs-dense test is deselected by default)
notebooks/       frontend.ipynb — small phase-1 front-end notebook
Structures/      input rod networks (gitignored, 87 MB); only the N=10k file is used by this project
results/         all computed data (gitignored, ~66 GB) — every run indexed in results/README.md
report/          the PRL package: main.tex, supplement.tex, refs.bib, figures/, tables/, scripts/, build.sh
docs/            project records: the two phase reports, pre-registrations, adversarial reviews, kickoff prompts
```

Entry points in `scripts/`:

| script | what |
|---|---|
| `run_modes.py` | bottom-up solve of a band window (deflated block LOBPCG; per-block checkpoints, `--resume`) |
| `run_interior.py` | interior λ-window solve (two-stage bandpass ChebSI; per-outer checkpoints, `--resume`, `--periodic`) |
| `merge_slices.py` | merge interior slices into one eigenvalue-ordered window (dedup by eigenvector overlap) |
| `analyze_localization.py` | per-mode IPR / participation ratio + envelope-decay ξ, with the L/2 finite-size ceiling |
| `make_montage.py` | render ε\|E\|² tiles and assemble the 15-per-row montage |
| `make_report_figures.py` | figures for `docs/REPORT_N10K.md` (written to `results/figures/`) |
| `validate.py` | the pre-registered N=1000 validation gates vs MPB |
| `exp/` | phase-1 experiments, interior gates I1–I8, fixes, detached run chains — indexed in `scripts/exp/README.md` |

## Setup

Two conda environments, both present on this machine:

- `lsu_ml` — the solver environment (Python 3.12, JAX 0.10.0 + cuda12). Every command below runs in it. There is no `environment.yml`; `conda env export -n lsu_ml` produces one.
- `mpb_judge` — MPB 1.11.1 on the CPU, used only by the validation scripts.

Structure files: `Structures/20260701_N10000_lsu_generated.txt` is the N=10k network. The N=1000 gold structure lives in the companion repo at
`/home/francisco/Documents/Create LSU Structures  - Claude/Example/N1000_lsu_example_ends.txt` (the double space is intentional).

```bash
conda run -n lsu_ml python -m pytest          # unit tests; add "-m slow" for the solver-vs-dense test
```

## Workflows

### Bottom-up band window (phase 1, N=1000)

```bash
# full band window of the montage, checkpointed + resumable (5.5 h at 128^3):
conda run --no-capture-output -n lsu_ml python scripts/run_modes.py \
  "/home/francisco/Documents/Create LSU Structures  - Claude/Example/N1000_lsu_example_ends.txt" \
  --grid 128 --band-lo 398 --band-hi 607 --resume

# regenerate the montage from the run outputs:
conda run -n lsu_ml python scripts/make_montage.py results/<tag>

# validation gates (crystal parity, literature, disordered parity, ...):
conda run -n lsu_ml python scripts/validate.py --all
```

Band indices are MPB-numbered (bands 1–2 at Γ are the ω=0 modes). All long
runs checkpoint and auto-resume (`--resume`). The phase-1 production run
(`prod_N1000_G128`, 611 bands) is on the external drive at
`/media/francisco/EXTERNAL_USB/prod_N1000_G128` (path in `report/scripts/common.py`);
only its log remains under `results/`.

### Interior gap window (phase 2, N=10k)

For windows deep in the spectrum (band ~5,000 of the N=10,000 network) where
bottom-up is infeasible (measured: 1.38 TB locked set), the two-stage bandpass
ChebSI solver targets a λ-window directly
(`docs/plans/2026-08-18_interior_preregistration.md`):

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
  --reference /media/francisco/EXTERNAL_USB/prod_N1000_G128 --ref-lo 395 \
  --slice-lo 473 --slice-hi 523 --gate-name I1

# same solve on a PERIODICALLY WRAPPED structure (convention change — see below):
conda run --no-capture-output -n lsu_ml python scripts/run_interior.py \
  Structures/20260701_N10000_lsu_generated.txt --grid 192 \
  --lam-lo 1.855 --lam-hi 2.000 --m 30 --periodic --tag n10k_G192_gap_periodic

# merge slices into one eigenvalue-ordered window (dedup by eigenvector overlap):
conda run -n lsu_ml python scripts/merge_slices.py --out results/n10k_G192_window \
  results/n10k_G192_Sbelow results/n10k_G192_Sgap results/n10k_G192_Sabove

# localization (IPR + xi with the L/2 finite-size ceiling):
conda run -n lsu_ml python scripts/analyze_localization.py \
  results/n10k_G192_Sbelow results/n10k_G192_Sabove --box 24.6467 \
  --out results/n10k_G192_localization

# montage for interior runs (band offset from the I2-certified count):
conda run -n lsu_ml python scripts/make_montage.py results/n10k_G192_Sbelow \
  --band-offset <MPB band of stored mode 0> --band-lo <lo> --band-hi <hi>
```

### Manuscript (phase 3)

```bash
bash report/build.sh             # both PDFs via tectonic (~1 min); LaTeX intermediates are left behind, gitignored
bash report/build.sh --figures   # first regenerate every number, table and figure from the saved data (CPU, ~4 min)
```

`--figures` needs the external drive with `prod_N1000_G128` mounted. Inside
`report/`: `PROGRESS.md` is the decision log, `FACTCHECK.md` the claim-by-claim
ledger, `references_verified.md` the DOI check, `numbers.json` every quoted
number with the file and script it came from.

## Rasterization conventions (read before comparing runs)

`rasterize_penlike` defaults to the **montage convention**, bit-for-bit as the
parent notebook: binary voxels, and rods whose *radius* pokes through a box
face are NOT wrapped. That last detail was quantified on 2026-08-24: it leaves
the outermost voxel shell with **ff = 0.1975 against 0.2211 in the interior**
(11% material deficit), a thin seam on the box faces. In a structure with a
photonic gap that seam behaves like a planar defect and hosts spurious
localized gap states — four were found and discounted in the N=10k window (see
`docs/plans/2026-08-17_adversarial_verification.md`, round 2).

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
  in fp64 on the host. See `docs/plans/2026-08-12_orientation_and_experiments_log.md`.
- JAX 0.10.0 + cuda12; the `cuda_executor ... driver version` stderr warning
  is benign on this driver.
- Long chains were run detached (`setsid nohup`, see `scripts/exp/chain_*.sh`)
  and polled through their logs in `results/`.

## Where to read next

- `docs/README.md` — index of every record, in reading order.
- `results/README.md` — every run directory with its window, grid, outcome and size.
- `scripts/exp/README.md` — what each experiment, gate and chain script does.
