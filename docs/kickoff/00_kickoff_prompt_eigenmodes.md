# Project kickoff prompt — local GPU Maxwell eigenmodes of disordered LSU networks (JAX + CUDA)

## Mission

Build a **new, self-contained project** in this folder (`/home/francisco/Documents/Eigenfuntions`)
that computes, **locally on this machine**, what someone else computed on a cluster:
the **electromagnetic eigenmodes (eigenfrequencies + field distributions) of a 3-D disordered
LSU photonic network in a periodic supercell**, given its permittivity distribution — i.e.
the photonic analogue of wavefunctions, most likely obtained originally with a supercell
plane-wave / iterative Maxwell eigensolver.

The evidence of what we must reproduce is in this folder:
`docs/reference/band_montage_398_607_15_non_ideal.png` — a 5250×5096 montage of ~210 tiles arranged 15 per
row, labelled by band index **398 → 607**. Each tile is a 3-D volume rendering of one
eigenmode: the dielectric network (grey, semi-transparent wireframe — the permittivity
distribution) with the mode's field intensity (orange/red isosurfaces or volume rendering)
overlaid. "non_ideal" almost certainly means a **disordered/imperfect** structure, as opposed
to an ideal crystal reference. Open and inspect the image yourself (downscale first — it is
37 MB) before assuming anything about it.

The image was calculated using one of the generated structures in `/home/francisco/Documents/Create LSU Structures - ML /TIER0_EXPLAINED.md`
with 1000 points, there are a few of these here: `/home/francisco/Documents/Create LSU Structures  - Claude/Example`
it was calculated from 256x256x256 arrays generated with the 20250903_create_h5_from_ends notebook with n=2.9 and elongated cylinders with ff ~22%

The end product is a **JAX + CUDA script (plus a small front-end notebook and a report)** that,
given a rod-network structure file from our previous projects, produces:

1. converged eigenfrequencies for a requested window of bands (e.g. the ~210 bands around the
   gap, like 398–607),
2. the corresponding field distributions on a real-space grid,
3. a montage figure equivalent to the reference image,
4. a validation report proving the results are faithful.

This is the **Tier-0 philosophy from the previous project applied to a new problem**: no ML,
no approximate shortcuts — exact numerics, engineered to be fast on this specific GPU. Read
`/home/francisco/Documents/Create LSU Structures - ML /TIER0_EXPLAINED.md` and `docs/REPORT_N1000.md`
(note: folder name has a trailing space) to absorb how that project worked: profile first,
rewrite exactly, validate against an independent judge, pre-register gates, verify
adversarially.

**Read this whole prompt before doing anything. Then orient, then investigate, then plan and
pre-register, and only then build.**

---

## Non-negotiables

These four are hard requirements. Violating any of them makes the deliverable worthless.

1. **Verify all physics, claims, and code against agents.** Every physics derivation, every
   methodological choice (operator formulation, eigensolver, smoothing scheme, precision
   policy), and every nontrivial piece of code must be independently checked by adversarial
   subagents whose explicit job is to refute it — separate agents for physics, for numerics,
   and for code correctness. Record what each verification pass found (including "nothing")
   in the docs/plans/verification files, as the previous project did.

2. **Search the literature online.** You are on an institutional network with access to many
   journals — use it. Find and read the primary sources before committing to a method: the
   original LSU paper (Sellers, Man, Sahba, Florescu, *Nat. Commun.* **8**, 14439 (2017)) and
   its supplementary methods (what solver, what supercell, what resolution, what rod radius
   and permittivity, how bands were indexed); the standard supercell/plane-wave references
   (Johnson & Joannopoulos, *Opt. Express* 2001 — the MPB paper; Joannopoulos et al.,
   *Photonic Crystals*, 2nd ed.); subpixel permittivity smoothing (Farjadpour et al. 2006 /
   Kottke, Farjadpour, Johnson 2008); interior-eigenpair methods (LOBPCG, folded spectrum,
   shift-invert, Chebyshev/polynomial filtering); amorphous-network photonics (Edagawa's
   amorphous diamond; Florescu/Torquato/Steinhardt hyperuniform PBG work; Imagawa et al.
   field-distribution/localization studies); and anything recent on GPU/JAX Maxwell
   eigensolvers. Cite what you rely on in the report.

3. **The script must run efficiently on this machine.** RTX 4080 Laptop GPU, **12 GB VRAM**, 62 GB RAM, 32 CPU threads,
   Linux, one heavy GPU job at a time. Do the memory arithmetic *before* building: a block of
   ~220 eigenvectors on an R³ grid with 2 transverse components in complex64 is ~7.4 GB at
   128³ — the design must fit, or stream/checkpoint deliberately. Profile before optimizing,
   as Tier-0 did. Long runs must checkpoint and auto-resume. The environment recipe from the
   previous project (`environment.yml`, JAX pinned with `jax[cuda12]`) is the starting point.

4. **The results must be faithful.** Not "plausible-looking" — validated. Faithfulness is
   established by the gates in the Validation section below (independent-solver parity,
   convergence, known-physics reproduction), not by visual similarity to the montage alone.
   Any approximation that trades accuracy for speed (single precision, truncated smoothing,
   loose solver tolerances) must be explicitly measured against a stricter reference and its
   error quantified and reported. No silent caps, no silent truncation.

---

## What already exists (reuse; do not reimplement)

| Asset | Where | What |
|---|---|---|
| Disordered LSU structures | `/home/francisco/Documents/Create LSU Structures - ML /Example/*_ends.txt` (trailing space in folder name) | Validated amorphous trivalent networks, 6-column rod endpoint files `x1 y1 z1 x2 y2 z2`, PBC-duplicated face-crossing rods. N=1000 examples and larger. |
| Gold reference structure | `/home/francisco/Documents/Create LSU Structures  - Claude/Example/N1000_lsu_example_ends.txt` (two spaces) | The Sellers et al. 2017 reference network, N=1000, L=11.44 µm, d0=0.8 µm. |
| Rod-file ↔ network tools | parent repo `tools.py` (`rods_to_network`, `srs_crystal_rods`), README §5 for the PBC convention | Use these to load structures; do not rewrite the PBC handling. |
| Crystal reference generator | `srs_crystal_rods` in parent `tools.py` | An ideal srs/single-gyroid-like crystal — your known-physics validation case. |
| Settled constants | both repos' READMEs | d0 = 0.8 µm, ρ ≈ 0.668 µm⁻³, L = 11.44 µm at N=1000. Never change these. |
| Tier-0 methodology | `.../Create LSU Structures - ML /TIER0_EXPLAINED.md`, `docs/REPORT_N1000.md`, `docs/plans/` | The working style to replicate: profiling tables, pre-registration, adversarial verification records, honest negatives. |
| The target | `docs/reference/band_montage_398_607_15_non_ideal.png` in this folder | What the deliverable must be able to reproduce for an equivalent structure. |

Also read the memory directories of both previous projects if present — they encode validated
findings and falsified dead ends. Do not re-derive what is settled there.

---

## Phase 1 — Investigate (deliverable: `docs/plans/<date>_investigation_report.md`)

Answer these before writing any solver code, with literature citations and agent verification:

1. **What exactly was computed.** The montage shows one field quantity per band — determine
   which (|E|², |H|², ε|E|² / electromagnetic energy density?) and at what k-point (almost
   certainly Γ of the supercell, but confirm what's standard for disordered supercells).
   What does "non_ideal" contrast with — find out if an "ideal" counterpart convention exists
   in the literature (e.g. crystal vs. disordered network).

2. **The permittivity distribution.** How is a rod network decorated into ε(r)? Pin down from
   Sellers et al. (and the standard in this literature) the rod radius (as a fraction of d0
   or of a), the dielectric constant (silicon, ε ≈ 11.56–11.9 — get the exact value used),
   background ε = 1, and how overlapping rods at vertices are handled. Decide and justify the
   rasterization + subpixel-averaging scheme (Kottke-style tensor smoothing vs. simple
   filling-fraction averaging) and quantify its effect on eigenfrequencies.

3. **Band-index arithmetic.** Derive the expected number of bands below the photonic band gap
   for a supercell of N trivalent vertices (from the primitive-cell band structure of the
   ideal srs/gyroid network and state counting). Use that to infer which structure size the
   montage's window 398–607 corresponds to, and confirm the window straddles the gap region.
   State this explicitly in the report — it determines the default band window of the script.

4. **The eigenproblem formulation and solver.** The standard is the transverse H-field
   formulation ∇×(ε⁻¹∇×H) = (ω/c)²H in a plane-wave basis, applied matrix-free with FFTs,
   solved iteratively (MPB uses preconditioned block conjugate gradient / Davidson).
   The hard part here is that we need **interior eigenpairs** (bands ~400–600, not the lowest
   210): evaluate honestly — with small-scale experiments, not opinion — the candidate
   strategies: (a) compute all bands from the bottom up in blocks with deflation, (b) folded
   spectrum / shift-and-square, (c) polynomial (Chebyshev) filtered subspace iteration,
   (d) shift-invert with an iterative inner solve. Pick based on measured cost on this GPU at
   the target grid size, and verify no bands in the window are missed (eigenvalue-count /
   spectrum-slicing check against the filter bounds).

5. **Precision policy.** JAX on a consumer GPU strongly favors fp32/complex64. Determine —
   empirically, on a small case with an fp64 CPU reference — what precision the operator
   application, orthogonalization, and Rayleigh–Ritz steps each need for the eigenfrequencies
   and mode shapes to pass the faithfulness gates. Mixed precision is fine **only** with
   measured error bounds.

6. **The independent judge.** Choose the reference implementation for parity checks —
   MPB (via `pymeeus`/`mpb` or the `meep`/`mpb` conda packages) on CPU is the natural choice,
   run on downsized cases (small supercell crystal, small disordered cell). The judge must be
   code we did not write.

## Phase 2 — Plan and pre-register (deliverable: `docs/plans/<date>_preregistered_plan.md`)

Before building, pre-register: the chosen formulation and solver, the grid resolution and its
convergence justification, the precision policy, the exact validation gates with numeric
tolerances (see below), and the performance target (wall-clock for the full 210-band window
on an N=1000 structure on this GPU). The previous project's
`docs/plans/2026-07-21_preregistered_plan.md` is the template.

## Phase 3 — Build

- Package layout mirroring the previous project: a library (`eigenfns/` or similar), thin
  CLI scripts (`scripts/run_modes.py`, `scripts/validate.py`, `scripts/make_montage.py`), a
  front-end notebook, `tests/`, `README.md`.
- Matrix-free operator via `jax.numpy.fft` (real-space ε, spectral curls), `jit`-compiled;
  block eigensolver with batched orthogonalization; deflation/filter machinery per the
  pre-registered choice.
- Checkpoint + auto-resume for the long solves; single-GPU discipline (one heavy job).
- Reproduce the montage: same tile layout (15 per row), band-index labels, network wireframe
  + field volume rendering, so the output is directly comparable to the reference image.

## Phase 4 — Validate (deliverable: `docs/plans/<date>_validation_report.md`)

Faithfulness gates — all must pass and be recorded:

| gate | test | tolerance (pre-register the final numbers) |
|---|---|---|
| Crystal parity | ideal srs crystal, primitive cell + small supercell: eigenfrequencies vs MPB | Δω/ω ≤ 10⁻³ per band (target; justify) |
| Literature reproduction | gap position/width of the ideal network crystal vs published values | within published/convergence error |
| Disordered parity | small disordered cell (e.g. N≈100–250 subcell or coarse grid): full band window vs MPB on CPU | Δω/ω per band + mode overlap `|⟨H_ours|H_ref⟩|` ≥ 0.99 up to degeneracy rotations |
| Degeneracy handling | degenerate/near-degenerate clusters compared as subspaces, not individual vectors | principal angles |
| Convergence | ω(resolution) and ω(solver tol) sweeps; smoothing on/off | monotone, extrapolated, reported |
| Completeness | eigenvalue count in the window matches spectrum-slicing bound count | exact — no missed/spurious bands |
| Residuals | ‖Â H − (ω/c)² H‖ / ‖H‖, transversality ‖k̂·H̃‖, block orthonormality | ≤ pre-registered thresholds |
| The montage | regenerate the 398–607 montage for the matching structure | qualitative + band-count/window agreement, shown side by side in the report |

Each gate's result goes through an adversarial verification pass (non-negotiable #1) before
being claimed as PASS.

## Final deliverables

1. `README.md` — install, one-command run, hardware notes.
2. The library + CLI scripts + notebook, tested (`pytest`).
3. `docs/REPORT_N1000.md` — physics + methods + literature + performance table (ms per operator
   application, wall-clock per band window vs N and resolution) + validation results +
   honest limitations.
4. The regenerated montage image(s).
5. `docs/plans/` — investigation report, pre-registration, verification records.

## Rules

- Orient first: read both previous repos' READMEs, TIER0_EXPLAINED.md, docs/REPORT_N1000.md, and
  memories before writing code.
- Profile before optimizing; put the measured breakdown table in the report, Tier-0 style.
- Never weaken a gate to make it pass; a failed gate plus an honest explanation beats a
  massaged PASS.
- Negative results (a solver strategy that loses, a precision that fails) are findings —
  record them, as the previous project recorded its falsified ML shortcuts.
- Keep heavy artifacts (fields, checkpoints) out of git; keep everything regenerable by
  script.
