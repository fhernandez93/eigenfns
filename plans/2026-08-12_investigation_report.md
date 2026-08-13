# Phase 1 — Investigation report (DRAFT; 3 of 4 literature agent reports pending)

Status: in progress 2026-08-12/13. Local-measurement sections are final pending
adversarial verification.

## 0. Sellers et al. 2017 ground truth (agent-verified from paper + SI, 2026-08-12)

- Solver: **MPB** ("Photonic band structures were calculated using MPB... plane wave
  expansion"), ε = 13 (n = 3.6), resolution "16 mesh points per a unit" (a = SNG
  primitive cell side), supercells N = 108–1000 vertices; disordered-supercell k-path
  Γ→X→M→R→Γ (folded); k-point density / band count / tolerances not published.
- Decoration (their convention, ≠ our montage convention): cylinders of equal radius
  **plus a sphere of the same radius at each vertex**, ff = 27.5% for CRNs; ε = 13.
- **Band counting, verbatim SI**: "a 216-vertex cubic supercell of single network
  gyroid exhibits its PBG between bands 108 and 109. Indeed a general N-vertex
  supercell has a PBG from bands N/2 to N/2+1." → N=1000: **gap between bands 500 and
  501** — confirms our independent MPB measurement (srs cubic cell: gap bands 4|5).
  Poorly-annealed samples show in-gap defect modes (their A,B,C ensembles).
- SNG crystal reference: gap **28.06% between bands 2 and 3** (primitive cell) at
  ε=13, r/a = 0.2554, ff 17.88% → literature-reproduction gate target. At ff 27.5%,
  ε13: SNG gap 0.1838 < a/λ < 0.2327. Amorphous N=1000 mean gap 16% (their ε/ff).
- No mode-field figures exist in Sellers 2017 (no montage precedent there); no
  "ideal/non-ideal" terminology. DLW literature (Haberko/Scheffold PRA 88, 043822)
  models the elliptical "laser-pen" voxel with aspect ≈ 3.0 → supports reading
  "non_ideal" = elongated-rod fabrication model (our aspect 2.5). **No published band
  structure of LSU/amorphous-gyroid networks with elliptical rods exists (2026)** —
  this project's computation is novel.
- Their follow-ups: Sellers PhD thesis (Surrey 2017); Siedentop et al. PNAS Nexus 3,
  pgae383 (2024) (tetravalent SHU network, N=1000 supercell, MPB+MEEP).

## 1. What exactly was computed (the target montage)

**Structure.** `band_montage_398_607_15_non_ideal.png`: 210 tiles (14×15), bands
398→607. The generating pipeline is pinned by the user's statement + the parent
notebook `20250903_create_h5_from_ends.ipynb`: a 256³ **binary** permittivity
grid from an N=1000 LSU rod network (box L=11.44 µm), "pen-like" cylinders —
circular radius 0.2252 µm in a z-unwarped space, global warp z=2.5·z′ giving
elliptical cross-sections (minor 0.2252 µm in-plane, major 0.563 µm along z),
ε_rod = 2.9275² = 8.5703, ε_bg = 1, flat caps, overlaps overwrite. **Measured
ff at 256³ = 0.2172** — matches the stated "ff ~22%"; parameters confirmed.

**"non_ideal"** = the elongated (aspect-2.5) elliptical cross-section — the
direct-laser-writing fabrication non-ideality — as opposed to an "ideal"
circular-rod decoration. (Literature cross-check pending; the parameter trail
in the notebook makes this reading concrete regardless.)

**Band window arithmetic (measured).** Ideal srs crystal in its 8-vertex cubic
cell (a = 11.44/5 µm, same ε, circular r=0.2252): complete PBG **between bands
4 and 5** (MPB, ~9.9% at quick settings) → 0.5 bands per vertex → for N=1000:
gap between bands **500|501**. The window 398–607 (210 bands) straddles it,
and the montage's nearly-grey tiles cluster around rows 6–7 ≈ bands ~475–505 —
consistent. The montage structure is an N=1000 network. [Literature
confirmation of srs band counting pending.]

**k-point.** Γ of the supercell (standard for disordered supercells; a single
k-point suffices for DOS/mode statistics at this supercell size — literature
section pending). The Γ-point spectrum of the supercell operator is what we
compute; the two ω=0 modes at Γ are removed exactly by the transverse basis.

**Field quantity.** Pending literature; hypothesis: electric energy density
ε|E|² (the Joannopoulos-book convention for dielectric-band modes) rendered as
volume/isosurface over the network wireframe. To be settled by comparing test
renders of ε|E|², |E|², |H|² of a gap-edge mode against the montage's visual
character during Phase 3.

## 2. The permittivity distribution

Settled (§1): the montage convention is binary voxelization, no subpixel
averaging. Consequences and decisions:

- **Production ε(r)**: reproduce the binary convention exactly (bit-compatible
  with the parent notebook's rasterizer) — this is what "faithful to the
  montage" means. Implemented and validated: `eigenfns/structure.py`.
- **Smoothing sensitivity gate**: quantify the eigenfrequency shift binary vs
  smoothed (filling-fraction-averaged and, if warranted, Kottke tensor) at
  128³/256³ on a small case — reported, not silently applied. [Farjadpour/
  Kottke literature specifics pending.]
- Resolution sensitivity: rods are 2.5 voxels across (minor axis) at 256³,
  1.26 at 128³ — convergence sweep ω(G) is a first-class validation gate.

## 3. Eigenproblem formulation (settled) and solver strategy (measured, in progress)

Formulation: transverse H-field, Θ H = ∇×(ε⁻¹∇×H), MPB's 2-transverse-
component spectral representation; curl diagonal in the per-(k+G) frame; one
application = 6 3-D FFTs + pointwise ε⁻¹. Validated against the analytic
homogeneous spectrum to 2.6e-7 (c64) with exactly the 2-dim Γ null space
removed. Measured: **2.7 ms/vector @128³** batched (m∈[8,32]), 28.6 ms @256³
(m≤4 fits 12 GB).

Solver candidates (kickoff Phase-1 item 4) — measured status:
- (a) bottom-up deflated block LOBPCG: implemented (`eigenfns/solver.py`) with
  seven fp32-hardening measures (experiment log). 32³ status: block 1 reaches
  MPB parity 8e-4 on its converged bands; later blocks stalled from cold random
  starts — **fix under test: warm-starting blocks with the previous block's
  guard Ritz vectors**. MPB's own success with 11-band blocks + deflation shows
  the approach is sound; remaining gap is convergence engineering.
- (b) folded spectrum, (c) Chebyshev window filter + count check, (d)
  shift-invert: to be costed after (a) is characterized; (c) is the fallback
  if deflated bottom-up proves too slow at 128³+.

Memory arithmetic @12 GB (measured per-vector sizes): 128³ c64 = 33.5 MiB →
660-band locked set = 22 GB ⇒ locked vectors stream from host RAM (62 GB) for
deflation, or fp16 storage (precision impact to be measured); working set
6 blocks × 96 ≈ 19 GB @128³ exceeds VRAM ⇒ working block m ≤ 48 at 128³ in
the [X,W,P,HX,HW,HP] scheme, or restructure. 256³: 268 MiB/vector ⇒ full
window infeasible on-GPU; strategy for 256³ (if needed): solve at 128³ and
refine, or chunked filtering. To be fixed in the pre-registered plan.

## 4. Precision policy (measured so far, experiments pending)

- Vectors/operator: complex64 on GPU (fp64 GEMM c64×c64→c128 unsupported on
  this stack; fp64 throughput 1:64 on RTX 4080).
- All ≤3m-dimensional dense algebra: host fp64 (measured necessity — fp32
  small-matrix paths caused catastrophic noise amplification; see log).
- Pending E4: fp64-CPU reference on a small case to bound c64 eigenvalue error
  vs the Δω/ω ≤ 1e-3 gate; c64-Gram noise floor measured implicitly ~1e-5.

## 5. The independent judge (settled + validated protocol)

MPB 1.11.1 (CLI, env `mbpEnv`; pymeep in `mpb_judge`). **Protocol**: binary ε
→ h5 → MPB `epsilon-input-file` run (file input = scalar ε, verified: `data` ==
`epsilon.xx`, ε_inv == 1/ε exactly) → read back MPB's `-epsilon.h5:data` → our
solver runs on that exact grid → band-by-band comparison. This makes the
comparison interpolation-proof (MPB's file resampling rule was probed and is
nontrivial; sidestepped by construction). Validated: 8e-4 agreement on the
first converged block at 32³. MPB cost point: 150 bands @32³, tol 1e-9: 6m40s.
Caveat for gates: MPB with *object* geometry always applies tensor subpixel
smoothing (measured ε_xy ≠ 0 even at mesh-size 1) — object-based MPB runs are
a *different discretization*, used only for literature-reproduction checks at
high resolution, not for grid-parity gates.

## 6. Band-index integrity plan

Bottom-up locking gives indexing by construction, but only if no band is
missed: completeness gate = (i) residuals + orthonormality of all locked
vectors, (ii) eigenvalue-count cross-check of the window against MPB on the
parity cases, (iii) spectrum-slicing count (stochastic Chebyshev step-function
trace) on production cases where MPB is unaffordable. [KPM details from the
eigensolver agent pending.]
