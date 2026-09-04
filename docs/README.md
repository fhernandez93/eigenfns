# docs/ — project records

Everything here is prose; the code is in `../eigenfns` and `../scripts`, the
data in `../results`, the manuscript in `../report`.

## Reports (one per phase)

- `REPORT_N1000.md` — phase 1 (2026-08-12 → 08-14). The N=1000 bottom-up
  solver: conventions, deflated block LOBPCG, gates G1–G9 and their outcomes,
  performance, honest limitations.
- `REPORT_N10K.md` — phase 2 (2026-08-17 → 08-31). The N=10,000 interior
  gap-edge project: KPM spectrum, two-stage bandpass ChebSI solver, gates
  I1–I9, localization, the rasterizer boundary-seam finding, the
  Rayleigh-normalization correction, everything retracted in adversarial
  rounds 1–4, honest limitations.
- Phase 3, the PRL manuscript, lives in `../report/` with its own
  `PROGRESS.md` (decision log) and `FACTCHECK.md` (claim ledger).

## plans/ — pre-registrations, investigations, adversarial reviews

Chronological. "FROZEN" files were fixed before the corresponding computation
started and never edited afterwards (amendments are appended, labelled).

| file | what |
|---|---|
| `2026-08-12_orientation_and_experiments_log.md` | phase-1 orientation and every measurement and failure (reconstructed after a machine crash) |
| `2026-08-12_investigation_report.md` | phase-1 investigation: montage analysis, conventions, solver candidates |
| `2026-08-12_adversarial_verification.md` | phase-1 refutation record |
| `2026-08-13_preregistered_plan.md` | FROZEN plan and gates G1–G9 for the N=1000 project |
| `2026-08-17_interior_investigation.md` | N=10k phase 1: feasibility measurements and the solver bake-off |
| `2026-08-17_interior_literature_survey.md` | interior-eigensolver literature and method selection |
| `2026-08-17_interior_preregistration_DRAFT.md` | draft, superseded by the frozen file below |
| `2026-08-18_interior_preregistration.md` | FROZEN pre-registration + Amendment A1, gates I1–I9 |
| `2026-08-17_adversarial_verification.md` | N=10k refutation record, rounds 1–4 (round 2 found the boundary seam) |

## kickoff/ — the task specification each phase started from

- `00_kickoff_prompt_eigenmodes.md` — phase 1
- `01_kickoff_prompt_interior_gap_modes_N10k.md` — phase 2
- `02_kickoff_prompt_PRL_report.md` — phase 3

Repository paths quoted inside these were rewritten to the current layout on
2026-09-04. Two names in the first prompt (`docs/plans/2026-07-21_preregistered_plan.md`,
`docs/plans/verification`) never existed; they were placeholders in the brief.

## reference/

- `band_montage_398_607_15_non_ideal.png` — the original cluster-computed
  montage (37 MB, 5250×5096, 210 tiles, bands 398–607) that phase 1 set out to
  reproduce. Our regenerations: `../results/i4_n1000_circ_G128/band_montage_398_607_15_non_ideal_regen.png`
  (circular decoration) and, for the elliptical decoration, the copy on the
  external drive next to `prod_N1000_G128`.
