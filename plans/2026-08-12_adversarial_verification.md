# Adversarial verification record — Phase 1 (2026-08-12/13)

Three independent agents were instructed to REFUTE the Phase-1 claims (physics,
numerics, code). Full transcripts summarized; every finding and its disposition
listed. Verdict shorthand: CONFIRMED = survived independent derivation/numerics;
REFUTED = claim was wrong (as stated); FIXED = code changed in response.

## Physics reviewer

| claim | verdict | notes / disposition |
|---|---|---|
| Transverse 2-comp formulation, handedness, curl signs | CONFIRMED | Independent derivation + reviewer's own dense fp64 3-component Cartesian operator: nonzero spectra match to 1.2e-7 at random k on disordered ε. "Curl diagonal" is loose language → "2×2 block kn·σ_y" (report wording fixed). |
| Hermiticity/PSD by structure (frames may be inexact, symmetry never) | CONFIRMED | Hermiticity residual 5.6e-8; PD on retained space. |
| Γ zeroing of G=0 slots removes exactly the 2 ω=0 modes | CONFIRMED | Proof: λ>0 ⇒ H(G=0)=0 automatically. Numerics: exactly 2 zero eigs; 3-comp operator has G³+2 zeros; nonzero spectra agree 1.3e-7. |
| Unit conversions (λ=(ω/c)², ν=√λ·a/2π, MPB c/a units) | CONFIRMED | Parity at 8e-4 self-validates the 11.44 µm conversion. |
| Band arithmetic 0.5/vertex, N=1000 gap 500|501, montage 14×15 layout | CONFIRMED | Montage pixel geometry re-derived; rows 6–7 = bands 473–502 straddle 500|501. |
| **Montage band-numbering convention** | **UNCERTAIN — open item** | MPB counts the two ω=0 Γ modes as bands 1–2; our solver excludes them. If the original montage used MPB numbering, its "band 398" = our 396th nonzero mode (±2 shift on every label). Not decidable from local repo contents. DECISION (pre-registered): we emit MPB-compatible numbering (our band n reported as n+2 at Γ) as the default, flagged in all outputs; the user can confirm the original convention. |
| ε|E|² as the rendered quantity | CONFIRMED as convention; montage identification UNCERTAIN | Grey gap tiles don't discriminate ε|E|²/|E|²/|H|² (all show localization). Decidable at render time: ε|E|² concentrates in rods with interface discontinuities; |H|² is smooth. Poynting excluded a priori (≈0 for standing modes). |
| "Every cross-section is an r × s·r ellipse" | REFUTED as stated (implementation correct) | True cross-section: semi-axes r and r·√(cos²θ+s²sin²θ), θ = rod angle from ẑ; vertical rods stay circular. Measured: z-rod 0.224×0.224, x-rod 0.224×0.557, 45° 0.427 (predicted 0.429). This equals the DLW laser-pen Minkowski sweep — *more* fabrication-faithful than the wrong blanket claim. Docstring + report fixed. |
| ff≈22% as parameter confirmation | WEAKENED | One scalar, one constraint: rules out s=1 at r=0.2252 (would be ~10–11%) but (r,s) not jointly pinned by ff alone; the notebook parameter trail remains the primary evidence. Report wording adjusted. |
| `make_basis` rdtype | BUG FOUND → FIXED | fp64 mode silently ran an fp32 basis (would have blinded E4). Fixed; E4 relaunched. |

## Numerics reviewer

| claim | verdict | notes / disposition |
|---|---|---|
| 32³ parity 4.3e-6 | CONFIRMED as two-implementation parity on the identical matrix; REFUTED as evidence about discretization/rasterizer/interior-128³ behavior | Magnitude independently predicted by Kato–Temple (≈1.5e-6 typical) — an authenticity signature. Report now scopes the claim precisely. |
| rel-res 1e-4 ⇒ Δω/ω ≤ 1e-3 | CONFIRMED for values (Weyl: ≤5e-5 unconditional) | Typical interior error ~4e-6. |
| Band-index attribution in sub-1e-4 clusters | RISK IDENTIFIED → FIXED + gate redesigned | A missed cluster member shifts all later indices. Fixes: (i) final sort + monotonicity warning in solver (out-of-order recovery is a free miss detector); (ii) completeness gate redesigned to **deflated-probe KPM** (probes projected against locked set → expected count 0 → variance collapses to ±1 feasibility). Plain KPM REFUTED as a gate (±5 at 32³, ±30–50 at 128³ at stated parameters). |
| Single-pass c64 deflation | CONFIRMED adequate (leak floor ~1e-5–3e-5; λ-bias ~5e-10; 4× residual margin at L=660) | CGS2 only if tol<3e-5 or L≫10³. Production logging of locked-set orthogonality added to plan. |
| Fancy preconditioner Hermitian PD | CONFIRMED | −K⁻¹ at both ends cancels; kn floor provably inert (touches only masked slots). Comment fixed. |
| Chebyshev filter fp32 overflow hazard | BUG FOUND → FIXED | cosh growth overflows fp32 within a few hundred degrees at the λmax/4 cap; periodic in-recurrence rescale added. |
| "Certified" λ_max | REFUTED (wording) → FIXED | θ+β is not a certified upper bound; now 2-seed max + honest docstring + margin. |
| Memory arithmetic | CONFIRMED (MB↔MiB label fixed) ; 6-array peak model REFUTED | Realistic peak ~10–14 block-equivalents ⇒ m ≲ 30 at 128³/12 GB; E3/E5 must measure actual peak. |

## Code reviewer

| finding | severity | disposition |
|---|---|---|
| S1: final locking pass lacked the dead-row guard — a zero row could lock as fake λ=0, silently shifting all band indices | BUG-CRITICAL | FIXED (guard + rank-collapse RuntimeError). |
| O1: `rdtype` ignored | BUG-MINOR (fatal for E4's design) | FIXED. |
| C3: λ_max not certified | BUG-MINOR | FIXED (2 seeds + margin + docstring). |
| S5: dead-row penalty not spectrum-scaled | SMELL | FIXED (scales with max diag(A)). |
| S3/S4/C5: dead median-shift computation, discarded HP updates, unused n_dim | SMELL | FIXED (removed). |
| T2: float64 vs the notebook's float32 boundary membership (O(tens) voxels at 500³) | BUG-MINOR | Documented in structure.py; "exact" claim softened to "algorithmically exact". **Quantified 2026-08-13 against the notebook's literal function: 0 differing voxels at 64³, 5 of 16.7M (3.0e-7) at 256³ — spectrally negligible.** Golden ff 0.21733856 (64³) now carries notebook provenance in the test. |
| T3: radius-poking rods miss periodic wrap voxels — **shared with the parent notebook** | BUG-MINOR (convention) | Documented; deliberately NOT fixed (fidelity to the montage convention wins); any fix goes behind a flag + re-validation. |
| S2 (carry not deflated), C1 (locked-prefix aliasing), O3/O4 (fft/meshgrid, per-k ref) | hypothesized, NOT bugs | Confirmed correct by trace. |
| U2: no solver/chebyshev tests | GAP | Test vs dense operator added to the Phase-3 test plan (pre-registered). |
| U3: gold-path skipif, golden-ff provenance | GAP | To fix in Phase 3 tests: skipif + provenance note (notebook-function cross-check pending). |

## Net effect

Two bugs that would have corrupted results (S1 fake-λ0 locking; O1 fp32-basis
"fp64" reference) were found before any production run — both now fixed and the
affected experiments (E3, E4) relaunched on fixed code. One pre-registration
item was created (±2 numbering convention). The completeness gate was
redesigned around deflated-probe KPM. All core physics claims survived
refutation with independent derivations.
