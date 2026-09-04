# Phase 1 — Investigation report

Status: **COMPLETE 2026-08-13.** All four literature agents reported; all
Phase-1 experiments (E1 operator cost, E2 band counting, E3 full 64³ spectrum,
E4 precision, srs literature scan, MPB parity) finished; one adversarial
verification round (physics/numerics/code) applied — see
`2026-08-12_adversarial_verification.md` and the experiment log for numbers.

Headline experimental results:
- E3 (64³, N=1000 gold structure, montage-convention ε): 680 bands in 941.8 s;
  **largest interior gap between MPB bands 500|501 exactly as pre-registered**;
  Δν/ν = 2.35% at 64³, ν_center 0.516 (a=2.288 µm); KPM count checks pass.
- E4 (48³): c64 vs c128 max Δω/ω = 2.4e-7 over 96 bands — precision policy
  confirmed with 400× margin.
- Parity vs MPB (32³, identical grid): max Δω/ω = 4.3e-6 over 148 bands.
- srs literature scan: gap optimum 28.0% at r/a≈0.12–0.13 (ff≈18%) —
  reproduces Sellers's published 28.06% / 17.88%.

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

**Structure.** `docs/reference/band_montage_398_607_15_non_ideal.png`: 210 tiles (14×15), bands
398→607. The generating pipeline is pinned by the user's statement + the parent
notebook `20250903_create_h5_from_ends.ipynb`: a 256³ **binary** permittivity
grid from an N=1000 LSU rod network (box L=11.44 µm), "pen-like" cylinders —
circular radius r=0.2252 µm in a z-unwarped space, global warp z=2.5·z′. The
resulting cross-section ⊥ each rod has semi-axes r and r·√(cos²θ+s²sin²θ) (θ =
rod angle from ẑ): horizontal rods get the full 0.2252×0.563 µm ellipse,
near-vertical rods stay circular — exactly the DLW "laser-pen" Minkowski-sweep
voxel model (adversarially verified, incl. measured cross-sections). ε_rod =
2.9275² = 8.5703, ε_bg = 1, flat caps, overlaps overwrite. **Measured ff at
256³ = 0.2172** — matches the stated "ff ~22%". (ff is one scalar: it rules
out circular rods at r=0.2252 (~10–11%) but does not pin (r,s) jointly; the
notebook's parameter cell is the primary evidence for r and s.)

**Open item (pre-registered decision): ±2 band numbering.** MPB counts the two
ω=0 Γ modes as bands 1–2; our transverse solver removes them exactly. If the
original montage used MPB numbering, its "band 398" is our 396th nonzero mode.
Not decidable from local artifacts. Decision: we emit **MPB-compatible
numbering** (our nth nonzero mode is labeled band n+2 at Γ) in all outputs,
flagged; the user can overturn with one flag if the original convention proves
otherwise.

**"non_ideal"** = the elongated (aspect-2.5) elliptical cross-section — the
direct-laser-writing fabrication non-ideality — as opposed to an "ideal"
circular-rod decoration. Literature-anchored: DLW-written network rods have
elliptical cross-sections from the elongated laser voxel, long axis along z,
aspect ≈ 2.8 (Muller, Haberko, Marichy, Scheffold, Optica 4, 361 (2017):
210×580 nm silicon rods; Haberko et al. PRA 88, 043822 (2013) model the
"laser-pen" voxel aspect ≈ 3); "ideal (lattice)" vs "nonidealities" is the
vocabulary of Peng et al., ACS Photonics 3, 1131 (2016) for DLW gyroids. No
published 3-D band-structure study with aspect-2–3 elliptical rods exists
(agent-verified through 2026) — this computation is novel.

**Band window arithmetic (measured + triple literature confirmation).** Ideal
srs crystal in its 8-vertex cubic cell (a = 11.44/5 µm, same ε, circular
r=0.2252): complete PBG **between bands 4 and 5** (MPB, ~9.9% at quick
settings) → 0.5 bands per vertex → for N=1000: gap between bands **500|501**.
Confirmed three independent ways: (i) Sellers SI verbatim: "a general N-vertex
supercell has a PBG from bands N/2 to N/2+1" (216-vertex SNG: bands 108|109);
(ii) Lu, Fu, Joannopoulos, Soljačić (Nat. Photonics 7, 294 (2013)): single
gyroid's 32% gap lies between bands 2 and 3 of the 4-vertex bcc primitive
cell; (iii) our MPB measurement above. The 398–607 window (210 bands)
straddles 500|501, and the montage's nearly-grey tiles cluster at rows 6–7 ≈
bands ~475–505 — consistent. The montage structure is an N=1000 network.
General band-counting context: 2-D TM gap between bands N and N+1 with N =
scatterers (Florescu PNAS 2009); 3-D tetravalent (diamond-family): 1 band per
vertex (diamond gap bands 2|3, 2 vertices/primitive cell — Ho/Chan/Soukoulis
PRL 65, 3152 (1990)); 3-D trivalent (srs-family): 0.5 bands per vertex.

**k-point.** Γ of the supercell. Literature conventions vary by purpose: gap
*measurement* papers use folded supercell paths (Sellers: Γ→X→M→R→Γ; Haberko
2020: 8 points on Γ-R-X-M-Γ at 128³ resolution with ~1000 bands; Man 2013 2-D:
full 64² BZ mesh for DOS) — but per-band *mode renders* are taken at a single
k, Γ (Klatt PNAS 2019 renders "a mode at Γ in the highest dielectric band").
The montage is a per-band sequence for one structure — a Γ-spectrum object; at
N=1000 (L=5a) the BZ is folded 125×, so Γ sampling is dense in the underlying
spectrum. We compute Γ (the two ω=0 modes removed exactly by the transverse
basis); the solver supports arbitrary k (validated at k≠0 vs analytics), which
the crystal-parity gate uses and which enables a folded-path gap check as a
validation extra.

**Field quantity — settled (literature-verified 2026-08-12).** The standard
rendered quantity for dielectric-band modes of 3-D networks is the
**time-averaged electric energy density ε|E|² = E·D** — the Joannopoulos-book
"dielectric band" convention, MPB's `output-dpwr`, and exactly what Klatt,
Steinhardt & Torquato (PNAS 116, 23480 (2019)) render for network modes at Γ.
Edagawa's PAD review states the physics the montage shows: E-field energy
concentrates in the dielectric below the gap and is expelled above it; Imagawa
et al. (PRB 82, 115116 (2010)) report rising inverse participation ratio
(localized states) at the band edges — matching the montage's nearly-grey
gap-region tiles with few bright spots. Default plot: ε|E|² per band at Γ
(|H|² montage as an option); mode-character confirmation against the montage
visual style happens at Phase-3 render time.

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
- (a) **bottom-up deflated block LOBPCG — SELECTED (pending 64³/128³ scaling
  confirmation)**: with MPB's transverse-projection preconditioner (JJ01 Eq. 14),
  guard warm-starting, and the fp32-hardening list, blocks converge in 18–26
  iterations and reproduce MPB to **Δω/ω ≤ 4.3e-6 over 148 bands at 32³** (judge
  at tol 1e-9 fp64; ours c64 tol 1e-4). Scope (adversarial review): this is
  *two-implementation parity on the identical discrete matrix* — it proves the
  operator implementation, solver convergence, and completeness of the lowest
  148 at 32³, with a magnitude independently predicted by Kato–Temple (~1.5e-6);
  it says nothing about discretization adequacy, the rasterizer, or 128³
  interior behavior (those have their own gates). Iteration counts are
  MPB-grade, which cuts the methods-survey PCIe estimate for 128³ deflation
  streaming from hours to ~tens of minutes; measured 64³/128³ numbers to be
  inserted from E3. Value-vs-index split (review): rel-res 1e-4 guarantees
  Δω/ω ≤ 5e-5 unconditionally (Weyl), but band-index attribution inside
  sub-1e-4 clusters needs the completeness gate — redesigned as
  **deflated-probe KPM** (count *missed* eigenvalues below λ_b, expected 0,
  variance O(1)) plus the solver's monotonicity check on locked values.
- (b) folded spectrum — rejected on the methods survey (condition-number
  squaring + fp32 digit loss in the dense interior; MPB's own target_freq mode
  carries the same caveat).
- (c) Chebyshev filtered subspace iteration + KPM counting — implemented
  (`eigenfns/chebyshev.py`); the designated route for 256³ (interior window
  without storing 400 lower bands) and the source of the completeness check
  (stochastic eigenvalue counting). Survey estimate: degree ~150–250, 30–90 min
  at 128³ for 660 bands; kept as alternative and as the counting machinery.
- (d) shift-invert with iterative inner solves — rejected (indefinite inner
  systems, no matrix-free preconditioner; survey estimate 10³–10⁵ inner
  iterations per shift on this hardware).
- Survey verdict (2026-08-12 agent report): no published GPU/JAX 3-D supercell
  interior-band Maxwell eigensolver exists (closest: FAME CUDA lowest-bands
  solver; ChASE dense-agnostic ChebSI; MPB CPU-only) — this build is novel.

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
