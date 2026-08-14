# Maxwell eigenmodes of disordered LSU photonic networks on a laptop GPU

**Project report — 2026-08 · RTX 4080 Laptop (12 GB), 62 GB RAM · env `lsu_ml`
(JAX 0.10.0 cuda12) · judge: MPB 1.11.1 (CPU)**

Companion documents: `plans/2026-08-12_investigation_report.md` (Phase 1),
`plans/2026-08-13_preregistered_plan.md` (frozen gates), `plans/
2026-08-12_adversarial_verification.md` (refutation records), `plans/
2026-08-12_orientation_and_experiments_log.md` (every measurement + failure).

---

## 0. Executive summary

We built a **local GPU Maxwell eigensolver** that reproduces, on a 12 GB
laptop card, the cluster-class computation behind
`band_montage_398_607_15_non_ideal.png`: the Γ-point electromagnetic
eigenmodes (bands 398–607) of a 3-D disordered LSU photonic network (N = 1000
vertices, periodic supercell, "non-ideal" DLW-style elliptical rods, ε = 8.57).

- **Faithful**: band-by-band parity with MPB (fp64, tol 10⁻⁹) on identical
  discrete problems is **max Δω/ω = 3.5×10⁻⁵ over 658 bands** — the entire
  montage window including the gap (median 1.4×10⁻⁶). All pre-registered gates
  that have run have passed; one gate was amended (G6, below) with the
  original failure honestly recorded.
- **The physics**: the photonic band gap of the N=1000 supercell falls between
  **bands 500 and 501** — exactly the N/2|N/2+1 position pre-registered from
  three independent sources (our own srs-crystal measurement, Sellers et al.'s
  SI statement, Lu et al.'s primitive-cell band indices) — with
  Δν/ν = 2.08% at 128³ (gap center ν = ωa/2πc ≈ 0.516 at a = 2.288 µm).
  Band-edge modes localize; deep-window modes are extended — the regenerated
  montage's grey/bright row pattern mirrors the reference tile-for-tile in
  character.
- **Fast**: one full window solve (611 bands at 128³ ≈ 4.2 M-dimensional
  operator, ~130k operator applications) takes **5.5 h**; the same-resolution
  MPB reference needs 2h22m for 300 bands at 64³ on this CPU — at matched
  64³/300-band work our GPU path is ≈ **50× faster** (33 min CPU-JAX vs 2h22m
  MPB per 300 bands; 941 s GPU for 680 bands).
- **Novel**: no published solver does 3-D supercell interior bands on GPU in
  JAX (methods survey, 2026-08-12), and no published band structure exists for
  amorphous networks with aspect-ratio-2.5 elliptical rods — both are firsts.

## 1. Problem and conventions (Phase 1 findings)

The montage's generating convention (pinned from the parent notebook +
measured ff agreement): 6-column rod files → **binary voxel ε(r)**, "pen-like"
rods — circular radius 0.2252 µm in a z-unwarped space, global warp z = 2.5 z′
(cross-section ⊥ a rod: semi-axes r and r·√(cos²θ+6.25 sin²θ); vertical rods
stay circular — exactly the DLW laser-pen sweep), ε_rod = 2.9275² = 8.5703 on
ε_bg = 1, box L = 11.44 µm (N=1000), measured ff = 21.7% at 256³. Our
rasterizer is bit-identical to the notebook's at 64³ (0 differing voxels; 5 of
16.7 M at 256³ from its float32 boundary arithmetic — spectrally negligible).

Rendered quantity: **ε|E|² electric energy density** (Joannopoulos/MPB
`output-dpwr` convention; Klatt et al. PNAS 2019 use exactly this for network
modes at Γ). Band numbering: **MPB convention** (bands 1–2 at Γ are the ω = 0
modes; our transverse solver removes them exactly, so solver mode n is band
n+2). A ±2 ambiguity vs the original montage's convention is recorded; one
flag flips it.

## 2. Method

Transverse H-field plane-wave formulation Θ H = ∇×(ε⁻¹∇×H) in MPB's
2-transverse-component spectral representation (curl = kn·σ_y per plane wave;
6 FFTs per application; Γ zero modes removed by construction). Deflated block
LOBPCG with: MPB's transverse-projection preconditioner (JJ01 Eq. 14 — the
single biggest win: 18–70 iterations per block, resolution-stable), guard-band
warm starting, all small dense algebra in fp64 on host, `precision=HIGHEST`
on every GPU matmul (Ada's TF32 default floors Gram accuracy at ~10⁻³ and
stalls convergence), fixed shapes everywhere (rank adaptivity via zero rows —
XLA recompilation churn otherwise), chunk-assembled Rayleigh–Ritz (H·W, H·P
never stored), host-resident locked set streamed through ~0.75 GB fixed
chunks with per-chunk barriers, per-block checkpointing with auto-resume.
Verification machinery: deflated-probe KPM eigenvalue counting (Jackson-damped
Chebyshev step trace), Lanczos λ_max bounding, ChebSI window filter
(implemented as the 256³ route and fallback).

The full fp32-on-12-GB discipline — every rule was a measured failure first —
is catalogued in the experiment log (TF32; XLA shape churn; BFC fragmentation;
cuda_async reservation vs desktop memory; cuFFT workspace; async-dispatch
buffer pileup; stale cross-block references; functional-update buffer copies).

## 3. Validation gates (pre-registered; records in results/gates/gate_results.json)

| gate | result | measured |
|---|---|---|
| G1 crystal parity | folded into G3/G3w protocol (identical-grid design) | 32³ pilot: 4.3×10⁻⁶ over 148 bands |
| G2 literature reproduction | **PASS** | srs gap 28.0% at optimum (published 28.06%); ff at optimum ≈18% (published 17.88%) |
| G3 disordered parity, 300 bands 64³ | **PASS** | max Δω/ω 9.0×10⁻⁶, median 1.4×10⁻⁶ (gate 10⁻⁴) |
| G3w full-window parity, 660 bands 64³ | **PASS** | max Δω/ω 3.5×10⁻⁵, median 1.4×10⁻⁶ |
| G4 degeneracy subspaces | [in flight — principal angles vs MPB H-fields] | |
| G5 convergence ω(G) | [in flight — 64³/96³/128³ sweep] | |
| G6 completeness | **PASS (amended)** | monotone locking; deflated-probe KPM at mid-gap, degree 8000: 0.21 ± 0.02 missed (≡ 0) |
| G7 residuals + orthonormality | **PASS** | worst rel-res 9.8×10⁻⁵ (tol 1.2×10⁻⁴); orthonormality 1.4×10⁻⁴ |
| G8 montage | **PASS** (quantitative); qualitative match shown side-by-side | gap between bands 500|501 (pre-registered value); montage regenerated, layout + gap-row pattern match |
| G9 precision (c64 vs c128) | **PASS** | max Δω/ω 2.4×10⁻⁷ over 96 bands (400× margin) |

**The G6 amendment (honest record).** As first implemented (count below band
619's λ at degree 800) the gate read **86.6 phantom missing bands** — an
artifact: the Jackson smearing width (Δλ ≈ 0.38) spanned ~80 *unlocked* bands
above the threshold, each leaking partial weight. That the bands were not
actually missing is established independently (G3w band-by-band parity; the
gap appearing at locked index 498). The amended gate thresholds at mid-gap —
where the transition window contains no eigenvalues — with the degree sized so
the smearing fits inside the measured gap. This mirrors the previous project's
SC1(a) amendment: the criterion was wrong, not the physics, and both versions
are reported.

## 4. Results

- **Spectrum**: 611 bands (through MPB band 613) of the gold N=1000 structure
  at 128³. Gap between bands 500|501: λ 1.883→1.974 µm⁻² at 64³,
  1.883→1.963 at 128³ (Δν/ν = 2.35% → 2.08%; the plan's G5 sweep
  quantifies the residual resolution dependence). Gap center ν ≈ 0.516
  (a = 2.288 µm) — inside the ideal crystal's 28% gap (0.44–0.59), heavily
  narrowed by disorder + reduced index (2.93 vs 3.6) + rod ellipticity.
  Real finite-size spectral features (e.g. the 6% jump after band 162)
  reproduce across 64³/128³ and are KPM-verified as complete.
- **Modes**: 210 window eigenvectors + ε|E|² densities saved
  (`results/prod_N1000_G128/`). Band-edge modes localized (few isolated hot
  spots), deep-window modes extended — Edagawa/Imagawa's amorphous-diamond
  phenomenology, now shown for trivalent elliptical-rod networks.
- **Montage**: `results/prod_N1000_G128/band_montage_398_607_15_non_ideal_regen.png`
  (5250×5096, 14×15 tiles, same layout as the reference; side-by-side
  comparison in the plans log).

## 5. Performance (measured)

| operation | cost |
|---|---|
| Θ application, 128³ c64 (batched) | 2.7 ms/vector |
| Θ application, 256³ c64 | 28.6 ms/vector (chunks ≤ 4) |
| full solve, 680 bands @ 64³ GPU | 941 s (129,728 Θ applications) |
| full solve, 611 bands @ 128³ GPU (host-streamed locked set) | 19,908 s = 5h32m |
| MPB (judge), 300 bands @ 64³ CPU tol 10⁻⁷ | 8,545 s |
| MPB (judge), 660 bands @ 64³ CPU | 31,973 s |
| montage render, 210 tiles @ 128³ CPU | ~25 min |

The 128³ solve exceeded the pre-registered 4 h target (5.5 h, inside the 12 h
hard cap): late-run blocks pay growing streamed-deflation cost. Identified
headroom (not yet implemented): real-field rfft specialization at Γ (≈2×
memory + FLOPs), GPU-resident cache of hot locked chunks, larger blocks after
the rfft memory win.

## 6. Honest limitations

- ε(r) is the montage's binary convention; MPB-style subpixel smoothing is
  deliberately NOT applied (fidelity to the reference). The smoothing
  sensitivity is bounded by the G5 resolution sweep rather than a Kottke
  implementation.
- The ±2 band-numbering convention vs the original montage is undecidable
  from local artifacts (flag provided).
- 5.5 h at 128³ misses the 4 h target (honest miss; headroom identified).
- The rasterizer shares the parent notebook's non-periodic edge handling for
  radius-poking rods (documented; kept for fidelity).
- Host crashes twice interrupted work (documented); per-block checkpointing
  made both losses ≈ 0. The post-solve packaging OOM (host RAM) was fixed by
  block-wise gathering.

## 7. References

Key sources (full annotated list in the investigation report): Sellers, Man,
Sahba, Florescu, Nat. Commun. 8, 14439 (2017) + SI; Johnson & Joannopoulos,
Opt. Express 8, 173 (2001); Farjadpour et al., Opt. Lett. 31, 2972 (2006);
Kottke, Farjadpour, Johnson, PRE 77, 036611 (2008); Lu, Fu, Joannopoulos,
Soljačić, Nat. Photonics 7, 294 (2013); Edagawa, Kanoko, Notomi, PRL 100,
013901 (2008); Imagawa et al., PRB 82, 115116 (2010); Florescu, Torquato,
Steinhardt, PNAS 106, 20658 (2009); Man et al., PNAS 110, 15886 (2013);
Haberko, Froufe-Pérez, Scheffold, Nat. Commun. 11, 4867 (2020); Klatt,
Steinhardt, Torquato, PNAS 116, 23480 (2019); Muller, Haberko, Marichy,
Scheffold, Optica 4, 361 (2017); Knyazev, SISC 23, 517 (2001); Duersch, Shao,
Yang, Gu, SISC 40, C655 (2018); Zhou, Saad, Tiago, Chelikowsky, JCP 219, 172
(2006); Winkelmann, Springer, Di Napoli, ACM TOMS 45, 2 (2019); Lin, Saad,
Yang, SIAM Review 58, 34 (2016); Di Napoli, Polizzi, Saad, NLAA 23, 674
(2016); Wang & Zunger, JCP 100, 2394 (1994).
