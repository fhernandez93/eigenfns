# results/ — computed data (gitignored, ~75 GB)

Each directory is one solver run and `<name>.log` beside it is that run's
stdout. The manuscript's number ledger (`report/numbers.json`) cites these
directories and logs by name, so keep the names. Only the JSON gate ledgers
and localization tables are tracked in git (`git ls-files results`).

Per-run contents: `interior_report.json` (all parameters, counts, wall time),
`window_eigenvalues.npy`, `window_residuals.npy`, `window_vecs_spectral.npy`
(eigenvectors in the plane-wave basis; the large file), and, where kept,
`window_energy_density.npy` (ε|E|² on the grid). `RAYLEIGH_CORRECTION.md`
and `PRUNED.md` record post-hoc edits to a directory.

## N=1000 runs (L = 11.44 µm)

| directory | what | grid | window | converged | wall | size |
|---|---|---|---|---|---|---|
| `prod_N1000_G128` | phase-1 production, bottom-up bands 398–607, elliptical rods (aspect 2.5, ε=8.57): all 611 eigenvalues, the 210 window modes (vectors + ε\|E\|²), tiles, regenerated montage and side-by-side. Moved back from the external drive 2026-09-04; its 20 G of solve blocks were removed then, see `PRUNED.md` | 128³ | bands 398–607 | 611 | 5.5 h | 8.4 G |
| `conv_N1000_G96` | phase-1 resolution check; data not kept in this tree, log only | 96³ | bands 398–607 | | | — |
| `i4_n1000_circ_G128` | bottom-up reference with the N=10k decoration (circular rods r=0.331836 µm, ε=8.41): all 611 eigenvalues, window modes, tiles, regenerated montage, localization. Solve blocks pruned, see `PRUNED.md` | 128³ | bands 398–607 | 611 | 5.5 h | 8.3 G |
| `i1_n1000_slice` | gate I1: interior solver on a 50-band slice vs the production ground truth — PASS | 128³ | λ∈[1.701, 2.156], m=80 | 55 (+1 unconv.) | 2.6 h | 2.2 G |
| `i4int_n1000_below` | gate I4-interior, below-gap half vs `i4_n1000_circ_G128` — PASS | 128³ | [1.47, 1.835], m=155 | 107 | 4.0 h | 4.2 G |
| `i4int_n1000_above` | gate I4-interior, above-gap half — PASS | 128³ | [2.015, 2.55], m=160 | 109 | 6.5 h | 4.3 G |

## N=10,000 runs (L = 24.6467 µm, circular rods r=0.331836 µm, ε=8.41, ff=22.0 %)

| directory | what | grid | window | converged | wall | size |
|---|---|---|---|---|---|---|
| `n10k_G192_window` | **the primary result**: merge of the three 192³ slices below (133 states after overlap dedup) with energy densities, tiles, the montage `band_montage_n10k_gapedge_15.png`, localization, the I3 Gram diagnosis and the rare-region audit | 192³ | [1.757, 2.117] | 133 | — | 3.6 G |
| `n10k_G192_Sbelow` | production slice below the gap (energy densities pruned; identical copies are in `n10k_G192_window`) | 192³ | [1.757, 1.93], m=104 | 69 | 63 h | 7.3 G |
| `n10k_G192_Sgap` | addendum slice across the gap | 192³ | [1.925, 1.985], m=16 | 5 | 3.6 h | 0.5 G |
| `n10k_G192_Sabove` | production slice above the gap (energy densities pruned, as above) | 192³ | [1.98, 2.117], m=100 | 61 | 38 h | 6.5 G |
| `n10k_G160_gapedge` | gate I6, coarser-grid leg; 18 in-window states unconverged (edge-truncated) | 160³ | [1.8, 2.06], m=72 | 45 (+18) | 18 h | 3.5 G |
| `n10k_G256_edgelow` | gate I6 anchor at the low gap edge, 11/11 converged; outputs recovered from the checkpoint after the save stage crashed | 256³ | [1.84, 1.95], m=18 | 11 | 27 h | 8.0 G |
| `n10k_G256_edgehigh` | first high-edge anchor, **abandoned after 9.1 h as misconfigured** (subspace-limited); only its checkpoint remains | 256³ | [1.99, 2.08], m=18 | — | 9.1 h | 4.6 G |
| `n10k_G256_edgehigh_narrow` | restarted high-edge anchor on a narrow window; 8 in-window states unconverged (their vectors are saved) | 256³ | [1.99, 2.035], m=18 | 3 (+8) | 27 h | 3.0 G |
| `n10k_G192_gap_periodic` | boundary-seam test v1: gap window re-solved with periodic rasterization | 192³ | [1.855, 2.0], m=30 | 7 (+1) | 20 h | 0.9 G |
| `n10k_G192_gap_periodic_v2` | seam test v2 with a larger basis: seam verdict reproduced, completeness (I2) not resolved | 192³ | [1.855, 2.0], m=48 | 7 (+2) | 4.0 h | 6.2 G |

## Supporting directories

- `exp/` — phase-1 experiments: solver bake-off (`bakeoff_*`), the KPM density
  of states used in the paper (`n10k_G256_dos_kpm.npz`), decoration
  calibration, MPB 32³ parity inputs/outputs, the srs-crystal literature run,
  timing probes, gate logs (`i2*`, `i3i5*`) and chain logs. 2.1 G, of which
  2.0 G is `bakeoff_shiftinv_m64b_subspace.npy`.
- `gates/` — `gate_results.json`, the gate ledger (tracked); MPB 64³ judge
  runs (`mpb64*`, 30 H-field files); parity arrays; cross-grid and periodic
  overlap matches (tracked).
- `figures/` — three figures for `docs/REPORT_N10K.md`, from
  `scripts/make_report_figures.py`.

## Loose files

- `*.log` — stdout of the run with the same name.
- `n10k_G192_localization_{xi.png,modes.json}` and
  `i4int_n1000_localization_{xi.png,modes.json}` — outputs of
  `scripts/analyze_localization.py` for the N=10k window and the N=1000
  interior halves.

## Disk-heavy leftovers (kept)

Nothing was deleted in the 2026-09-04 cleanup. The files below are read by no
report script; the checkpoints are used only by `run_interior.py --resume` and
`scripts/exp/recover_energy_density.py`. Together they hold about 17 GB.

| file | size | why it exists |
|---|---|---|
| `n10k_G192_gap_periodic_v2/interior_state.npz` | 5.4 G | checkpoint of a completed run, kept resumable |
| `n10k_G256_edgelow/interior_state.npz` | 4.8 G | the checkpoint the outputs were recovered from |
| `n10k_G256_edgehigh/interior_state.npz` | 4.6 G | checkpoint of the abandoned run |
| `exp/bakeoff_shiftinv_m64b_subspace.npy` | 2.0 G | subspace of the rejected shift-invert method |
