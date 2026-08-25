# Interior Maxwell Eigenpairs at Scale: Literature Survey and Method Selection

*Record of the 2026-08-17 web survey (refutation-tasked subagent, primary sources
read where accessible; access level disclosed per source). Feeds Phase 1.2 bake-off
design and Phase 2 pre-registration.*

**Context.** Target: ~300 interior eigenpairs of the Hermitian PSD matrix-free operator Θ = ∇×(ε⁻¹∇×·) at 256³ (N ≈ 5×10⁷ dof), window near λ ≈ 2 at eigenvalue index ≈ 5,000, λ_max measured ≈ 4.6×10³ (128³ N=1000; grid-dependent), so the window sits at relative position λ/λ_max ≈ 4×10⁻⁴ with relative width δ/λ_max ~ 10⁻⁴–10⁻³. One Θ-apply ≈ 30 ms (FFT) at 256³. 12 GB VRAM; fp32 compute with fp64 reduced algebra. No factorization possible. A validated bottom-up LOBPCG exists; computing all ~5,000 lower bands is infeasible.

**Memory yardstick used throughout:** one field vector at 256³ with two transverse complex components in fp32 ≈ 0.27 GB. The 12 GB card holds roughly **30–40 vectors device-resident** (less FFT workspace). Any method needing ≥300 simultaneous basis vectors implies host-RAM offload/streaming regardless of algorithm; the discriminating quantity is the *active working set*.

**Source access disclosure.** Read in full or in key sections (PDF): Wang & Zunger 1994; Canning et al. PARA'08; Fang & Saad 2012 (complete, including all experiment tables); Pieper et al. ChebFD 2016 (Secs. 1–3 incl. all scaling formulas); EVSL 2019 (Maxwell experiments + recommendations + references). Read via abstract/HTML extraction: IFEAST 2018 (detailed HTML extraction incl. theorems and tables); Lin–Saad–Yang 2016; Weiße et al. 2006; Vecharynski–Yang 2016; Szyld–Vecharynski–Xue; R-ChFSI 2025; Kressner–Ma–Shao 2023; ChASE 2019/2022 (abstracts only); MPB developer docs. **Not accessible / not read:** AdaPolySI full text (ACM paywall; abstract via search), Di Napoli–Polizzi–Saad eigenvalue-count paper (cited from secondary sources), Zhou–Chelikowsky ChebSI full text (well-documented via abstracts and derivative works), the GPU photonic-crystal band-structure eigensolver papers (abstracts only). Claims from unread sources are marked as such; inferences are flagged "Inference".

---

## 1. Per-method summaries

### 1.1 Folded spectrum method (FSM): LOBPCG/CG/Jacobi–Davidson on (Θ−σ)²

**Primary sources (read):**
- L.-W. Wang, A. Zunger, "Solving Schrödinger's equation around a desired energy: Application to silicon quantum dots," *J. Chem. Phys.* **100**, 2394–2397 (1994). DOI: 10.1063/1.466486.
- A. Canning, J. Dongarra, J. Langou, O. Marques, S. Tomov, C. Voemel, L.-W. Wang, "Interior state computation of nano structures," PARA'08 proceedings (LBNL/UTK). (Related journal version: C. Vömel, S. Tomov, O. Marques, A. Canning, L.-W. Wang, J. Dongarra, *J. Comput. Phys.* **227**, 7113–7124 (2008) — not read.)

**What the papers show.** Wang–Zunger fold the spectrum at a reference σ inside the gap: the lowest eigenpairs of (H−σ)² are exactly the eigenpairs of H nearest σ, on *both* sides of it — precisely the HOMO/LUMO (for us: gap-edge) states — with **no orthogonalization against the thousands of lower states**. Plane-wave basis to 10⁵ orbitals; effort observed linear in system size. They state plainly that "(Ĥ−E_ref)² slows down considerably the convergence of standard minimization methods" (condition-number squaring) and that this is recovered by "a carefully preconditioned conjugate gradient approach" with the Fourier-space diagonal preconditioner **A = α² / ((½G² + v₀ − σ)² + α²)** — the squared shifted-kinetic diagonal, α of the order of the wavefunction kinetic energy. This is the direct analog of the MPB kinetic preconditioner squared; MPB itself ships this as its "targeted eigensolver" on (∇²+ω²)-type folding, with the docs candidly calling it "not really the best algorithm" versus hypothetical shift-invert Arnoldi/Jacobi–Davidson (MPB developer documentation, read).

Canning et al. (PARA'08) is the best measured-count source. ESCAN folded-spectrum with diagonal preconditioner P = (1 + (−½∇² + V_avg − σ)/E_k)², CdSe dots to n = 141,625, computing the 10 states at the VBM with σ in the gap. Matvec counts **on the folded operator** for Cd83Se81 (n = 34,143): Banded-PCG 15,096; LOBPCG 10,688; JDQMR 5,314; GD+1 4,084 — i.e. **≈ 400–1,500 folded-applies per converged pair** (≈ 800–3,000 Θ-applies). Findings: all methods robust, "find degenerate states without any particular problem"; Jacobi–Davidson variants ≈ 3× faster than Banded-CG; GD+1 always minimal in matvecs; JDQMR wins wall-clock when orthogonalization is relatively expensive. Critically for option (c): *"finding the extreme eigenvalues of (H−σ)⁻¹ requires very few iterations, [but] applying (H−σ)⁻¹ at each step is extremely expensive and we have not found this method effective in practice"* — shift-invert with inner iterative solves was tried and rejected in exactly this operator class. Their stopping practice: converge on the folded residual and verify on H; folded tolerance t implies ≤ 5t on H.

**Convergence drivers.** Per-iteration rate is the preconditioned-gradient rate on the folded operator: depends on the folded gap ratio (window-interior separations get quadratically compressed) and on how well P⁻¹ approximates (Θ−σ)². Knyazev's LOBPCG theory applies (*SIAM J. Sci. Comput.* **23**(2), 517–541 (2001)). The kinetic diagonal captures the entire high-λ tail of Θ, which is what makes folding viable despite squaring — this structural advantage is unavailable to polynomial filters.

**Cost/memory/failure modes.** Memory: block size + locked vectors; with hard locking, active set can be ≤ 3×(block of 32–64) vectors — the only approach in this survey whose active set fits entirely in 12 GB VRAM (locked vectors offloadable). Failure modes: (i) slow convergence for pairs far from σ inside the window (folded value grows quadratically — mitigate with 2–3 shifts covering the window); (ii) fp32 residual floor on the squared operator is worse (κ squared) — Inference, partially mitigated by Canning's verify-on-Θ practice and fp64 reduced algebra; (iii) missing states: FSM has no completeness guarantee — must be paired with a count oracle. Ghosts in the Lanczos sense do not arise (preconditioned minimization, not a long Krylov chain).

**fp32 experience.** No published fp32 FSM run found. Adjacent evidence: mixed-precision LOBPCG gives 1.4–2.0× speedups with fp32-preconditioner/orthogonalization (D. Kressner, Y. Ma, M. Shao, *Numer. Algorithms* **94**, 1653–1671 (2023), arXiv:2302.12528 — abstract-level).

### 1.2 Bandpass Chebyshev polynomial filtering (ChebSI / ChebFD / filtered Lanczos / EVSL / ChASE)

**Primary sources:**
- Y. Zhou, Y. Saad, M. L. Tiago, J. R. Chelikowsky, *J. Comput. Phys.* **219**, 172–184 (2006); *Phys. Rev. E* **74**, 066704 (2006). (Abstract-level; lowest-subspace filtering, not interior.)
- H.-R. Fang, Y. Saad, "A filtered Lanczos procedure for extreme and interior eigenvalue problems," *SIAM J. Sci. Comput.* **34**(4), A2220–A2246 (2012). **Read in full.**
- A. Pieper et al., "High-performance implementation of Chebyshev filter diagonalization for interior eigenvalue computations" (ChebFD), *J. Comput. Phys.* **325**, 226–243 (2016), arXiv:1510.04895. **Key sections read.**
- R. Li, Y. Xi, L. Erlandson, Y. Saad, "The Eigenvalues Slicing Library (EVSL)," *SIAM J. Sci. Comput.* **41**(4), C393–C415 (2019), arXiv:1802.05215. **Maxwell experiments + recommendations read.** Companion: Li, Xi, Vecharynski, Yang, Saad, TR-Lanczos with polynomial filtering, *SISC* **38**(4), A2512 (2016) (cited).
- J. Winkelmann, P. Springer, E. Di Napoli, ChASE, *ACM TOMS* **45**(2), 21 (2019); GPU arXiv:2205.02491 (abstracts). **ChASE targets extremal spectrum only** — usable here only in folded/shifted form.
- Ni et al., "AdaPolySI," ICS'26 (paywalled; abstract only). Adaptive degree scheduling; reported to outperform EVSL, SLEPc, CJ-FEAST on CPU clusters.

**Quantitative convergence law (ChebFD).** With target half-width δ, search-interval margin δ′, spectrum half-width S_w, filter quality η = N_p / (−log₁₀ σ_damp): **η_opt ≈ η₀ · (S_w/δ) · (δ/δ′)** and **N_p_opt ≈ N₀ · (S_w/δ) · (δ/δ′)**, with η₀ = 2.58, N₀ = 6.23 at mid-spectrum, falling to η₀ = 1.13, N₀ = 2.73 at c/S_w = 0.9 (near-edge windows are *cheaper*). **Total spMVM ≈ η · N_S · (−log₁₀ ε)**; recommended **N_S ≳ 2 N_T**, and for a linearly-vanishing DOS (pseudogap — the gap-edge case) the optimum is **N_S = 4 N_T**. Oversampling substitutes for filter degree. Measured anchors (Fig. 4, mid-spectrum, δ′=δ): δ/S_w = 10⁻³ → N_p_opt = 6,251 (Lanczos kernel), 7,899 (Jackson), 1,424 (no kernel); N_p_opt ∝ 1/δ. Demonstration: ~10² innermost pairs of a 10⁹-dim matrix on 512 SuperMUC nodes; "presently the only approach that can solve interior eigenvalue problems at this scale."

**Measured counts at our window ratio (Fang–Saad, read in full).** 3-D Laplacian, n = 10⁶, window ratio 8×10⁻⁴, 276 pairs: degree 600 → 840k matvecs; degree 1,000 → 950k; degree 1,600 → 1.14M — **≈ 3,000–4,100 matvecs per pair**, with 710–1,400 stored Lanczos vectors (partial reorthogonalization). Ga₄₁As₄₁H₇₂ (ratio 5×10⁻⁴, window near spectrum edge): degrees 100–400, ~600–700 matvecs/pair; low-pass filters at degree 10–50 beat mid-pass there. **Documented failure mode:** at d = 600 the run initially stopped with only 48/276 window eigenvalues converged — the rest had not yet *appeared* as filtered Ritz values; remedy: mandatory extra-iteration probes after apparent convergence (+30 iterations, repeated until no new window Ritz values emerge).

**EVSL on Maxwell (read).** Curl-curl FEM (to n = 2.6M, ~32% zero cluster): polynomial filtered Lanczos computed 96–121 interior pairs, degrees 15–570, factorization-free — filtering "skips the zero eigenvalues." Rational (shift-invert) filtering up to 90× faster in 2-D but needed Pardiso factorizations to **68 GB at n = 2.6M** — infeasible at N = 5×10⁷ matrix-free; EVSL declines iterative rational filters because the shifted systems are "highly indefinite… challenging."

**fp32 suitability.** Direct evidence: R-ChFSI reaches 10⁻⁸ residuals with **FP32/TF32 matvecs** at 85M grid points / 13,500 eigenpairs on GPUs (arXiv:2503.22652, 2025; extremal spectrum). The Chebyshev three-term recurrence on a mapped spectrum is numerically benign; practical fp32 limit is per-application damping floor (~ε_mach reinjection) — Inference: with fp64 RR this caps per-sweep damping ~10⁷ but not final accuracy (subspace iteration re-damps each sweep).

**Memory.** Filtered Lanczos: 3 active vectors + stored basis (2.5–5× N_T, host-resident, streamed for reorthogonalization). ChebFD-style SI: N_S ≈ 2–4 N_T simultaneous vectors, embarrassingly blockable (BLAS3, no cross-vector recurrence).

### 1.3 FEAST / contour integration

**Primary sources:** E. Polizzi, *PRB* **79**, 115112 (2009) [abstract]; B. Gavin, E. Polizzi, IFEAST, *NLAA* **25**(5), e2188 (2018), arXiv:1706.00692 [detailed extraction read]; FEAST v4 guide, arXiv:2002.04807.

**IFEAST specifics.** Convergence per outer governed by filter-value ratio degraded by inner-solve error; practical inner tolerance loose (10⁻²–0.5 relative to current eigenresidual). Na5 toy (n = 5,832, 50 pairs): 11 outer iterations, ~2,000 matvecs/pair. Gavin–Polizzi prove IFEAST is **a block Krylov/polynomial filter in disguise** — cannot asymptotically beat an optimal polynomial filter; advantages are memory and shift parallelism. Two direct warnings: (i) **preconditioning the inner solves is counterproductive** ("the right hand sides… converge to invariant subspaces of the matrix being diagonalized, but… not… of the preconditioned matrix") — the kinetic preconditioner cannot be exploited; (ii) IFEAST+FOM "unreliable for eigenvalues in the interior"; robust variant is IFEAST+GMRES ≈ harmonic Rayleigh–Ritz. m₀ ≈ 1.5× wanted.

### 1.4 Spectrum slicing and KPM counting oracles

**Primary sources:** Lin, Saad, Yang, *SIAM Review* **58**(1), 34–65 (2016) [abstract]; Weiße et al., *RMP* **78**, 275 (2006) [abstract]; Di Napoli, Polizzi, Saad, *NLAA* **23**(4), 674 (2016) [not read; secondary]; Vecharynski, Yang, arXiv:1602.02306 [summary read] — Chebyshev-projector counting "becomes computationally prohibitive for narrow intervals"; their preconditioned Lanczos-quadrature counting converges in a few iterations independent of condition number.

**Role here.** Not a standalone solver but the completeness oracle: fix N_T in [ξ,η] before solving, audit after. Cost ~10⁵ matvecs ≈ cheap. For very narrow windows prefer Lanczos quadrature, preconditioned counting, or counting on the folded operator (Inference).

### 1.5 Preconditioned interior methods without folding

**Primary sources:** Szyld, Vecharynski, Xue, PLMR (interior), arXiv:1504.02811 [abstract]; Vecharynski (MERL TR2016-165 w/ Knyazev), arXiv:1609.05407 — SPD-preconditioned PSD on an indefinite system ≡ restarted PMINRES [abstract]; Knyazev 2001; Loe–Morgan 2019 / Embree–Loe–Morgan 2020 polynomial preconditioning [abstracts]; Huang et al., *J. Comput. Appl. Math.* 2014 photonic-crystal shift-invert [abstract — relies on structured fast solves unavailable matrix-free].

**Literature verdict for FFT-applied operators.** The one group that tried inner-iterative shift-invert on an FFT-applied operator with exactly this preconditioner structure (ESCAN) explicitly abandoned it as uncompetitive versus preconditioned folded/JD methods. The literature-favored factorization-free routes are (1) preconditioned folded/JD and (2) polynomial filtering.

---

## 2. Comparison table

| Criterion | (a) Folded + precond. LOBPCG/JDQMR | (b) Bandpass Chebyshev (filtered Lanczos / ChebFD SI) | (c) IFEAST (contour + inner solves) | (d) Hybrids |
|---|---|---|---|---|
| Convergence driver | Preconditioned folded gap ratio; kinetic-diagonal quality on (Θ−σ)²; window-edge pairs slowest | Filter damping: η ≈ η₀(S_w/δ)(δ/δ′); oversampling substitutes for degree; adjacent spectral gap *increases* δ′ → cheaper | Filter ratio degraded by inner error; inner tol loose; can't exploit SPD precond | As per components |
| Published Θ-applies per pair | 800–3,000 (Canning; GD+1 best) | 600–4,100 measured at ratio 5–8×10⁻⁴ (Fang–Saad) | ~2,000 on toys; ≥ optimal poly filter asymptotically | oracle ~10⁵ once |
| Simultaneous vectors (×0.27 GB @256³) | **1–3 blocks of 32–64 + locked (offloadable) — fits VRAM** | Lanczos: 3 active + 700–1,500 stored (host-streamed); ChebFD: 600–1,200 active | m₀ ≈ 450 + workspace | ~1–40 |
| Failure modes | No completeness guarantee; window-edge slowdown; folded fp32 floor; multiple σ for wide windows | **Missed not-yet-appeared window pairs** (Fang–Saad d=600); ghosts from lost orthogonality; high degree at ≤10⁻⁴ | Interior Ritz unreliability (need GMRES/harmonic); precond incompatibility | inherited |
| Degeneracies | Block ≥ multiplicity; "without any particular problem" (Canning) | Clusters benign if N_S ≫ N_T | Fine if inside contour | — |
| fp32 evidence | None direct; mixed-precision LOBPCG 1.4–2× | **R-ChFSI: 10⁻⁸ with FP32/TF32 matvecs (2025)** | Mixed-precision FEAST exists (unread) | — |
| Matrix-free at 5×10⁷ | Proven (ESCAN lineage) | Proven (ChebFD 10⁹ dim; EVSL matrix-free) | No published run at this scale/regime | Proven (KPM) |

---

## 3. Answers to the regime-specific questions

**Does bandpass degrade at δ/λ_max ~ 10⁻³–10⁻⁴?** Yes, linearly in 1/δ: N_p_opt ≈ N₀(S_w/δ)(δ/δ′). Anchors: ratio 10⁻³ → degrees ≈1,400–7,900; ratio 8×10⁻⁴ → measured 600–1,600 (Fang–Saad); ratio 10⁻⁴ → extrapolated 1.4–8×10⁴ (no measured run found). Mitigations, all literature-supported: (i) near-edge windows are ~2.3× cheaper (ChebFD constants) and Fang–Saad's near-edge case ran at degree 100–400; (ii) **the spectral gap adjacent to the window is free transition margin δ′** — degree scales with δ/δ′; (iii) oversampling N_S = 2–4 N_T. Realistic total for 300 pairs at ratio 10⁻³: ~1–2×10⁶ Θ-applies ≈ 8–17 GPU-h at 30 ms; at 10⁻⁴: ~10⁷ ≈ 80+ h.

**Does preconditioned folded spectrum beat it here?** No published head-to-head at this exact regime (the gap our bake-off closes). Published: folded JD/LOBPCG at 800–3,000 Θ-applies/pair (Canning) vs filtered Lanczos 600–4,100/pair — **neither dominates on matvecs; folding wins on working-set memory, bandpass on completeness auditing + BLAS3**. Structural argument for folding here: the kinetic diagonal cancels the high-λ tail — the very thing that forces polynomial degree ∝ S_w/δ — and no polynomial in Θ can use that information. Wang–Zunger σ-in-gap placement is tailor-made for a gap-edge window. As the window grows to 300, folding's per-pair cost degrades (locking + folded-gap compression at the window's outer edge — Inference); hence the bake-off.

**Is preconditioned shift-invert MINRES literature-favored?** No. Theory sound (Vecharynski/Knyazev; PLMR), and IFEAST shows loose inner tolerances suffice; but ESCAN tried and rejected it on an FFT-applied operator, and EVSL declines iterative rational filters ("highly indefinite… challenging"). JDQMR ≈ its most robust published form.

---

## 4. Recommendation: bake-off slate and parameter ranges

**Arm 1 — Folded-spectrum block solver with Wang–Zunger squared-kinetic preconditioner (primary).** σ in the gap; P = α²/((kin−σ)²+α²), α ≈ mean kinetic energy of iterates, tune ±1 decade. Block 32–64, hard locking, fp64 RR, precision=HIGHEST. Expect 2–3 σ values for a 300-state window. Budget expectation: 0.8–3×10³ Θ-applies/pair. Converge on Θ residual, not folded (folded tol → ≤5× on Θ).

**Arm 2 — Polynomial filtered subspace with gap-aware mid-pass filter (co-primary).** Degree scan d ∈ {800, 1,600, 3,200}; δ′ = min(gap width, δ) as starting guess with near-edge N₀ ≈ 2.7–3; basis 2.5–5× N_T host-resident streamed; convergence stride 10; **mandatory** post-convergence probe iterations and unfiltered Rayleigh-quotient re-evaluation with out-of-window rejection.

**Arm 3 (conditional, cheap) — polynomial-preconditioned folded** (degree-10–30 Chebyshev in (Θ−σ)² as preconditioner if the diagonal underperforms at 256³). **Skip IFEAST** (polynomial filter in disguise, precond-incompatible, no published run near this scale).

**Common infrastructure:** KPM/Lanczos-quadrature DOS (20–40 stochastic vectors) to pin λ_max tightly (a loose bound inflates all degrees — Fang–Saad §4), fix N_T ± few, map gap edges for σ and δ′.

**Decision metric:** Θ-applies per converged-and-verified pair at ‖Θy−λy‖/λ ≤ tol, wall-clock incl. orthogonalization/streaming, completeness audit vs DOS count.

---

## 5. Ghost / spurious-pair detection checklist (literature-sourced)

1. **Re-evaluate every candidate on the original operator** (λ = ⟨y,Θy⟩, residual ≤ tol) — never accept convergence declared on ρ(Θ) or (Θ−σ)² alone (Fang–Saad Alg. 2; Canning ≤5× factor).
2. **Reject out-of-window Ritz pairs** after step 1, even with tiny filtered residual (Fang–Saad line 24).
3. **Filter-consistency cross-check:** filtered Ritz value θ_j vs ρ(λ_j); mismatch flags a leakage ghost (Inference from Fang–Saad/EVSL framework).
4. **Semi-orthogonality κ ≤ √ε_M with partial reorthogonalization** (Simon 1984 lineage); reorthogonalization frequency *increases* under filtering (Fang–Saad Fig. 3.1). fp32: run ω-recurrence + reorthogonalization in fp64.
5. **Independent count audit** vs KPM/Lanczos-quadrature count: deficit = missed, excess = ghosts/duplicates.
6. **Post-convergence probe iterations** (+30, repeated until quiescent) — the documented near-miss mode (48/276 at d=600).
7. **Cluster verification:** block ≥ largest multiplicity; N_S ≫ N_T; confirm degenerate subspaces by principal angles against an independent run (different seed/σ/degree).
8. **Interior extraction discipline:** refined/harmonic RR for any Lanczos-on-unfolded-Θ variant — plain Ritz in the interior is the classic ghost generator.
9. **Dedup across shifts/slices by eigenvector overlap**, not eigenvalue proximity (EVSL practice).
10. **Precision hygiene:** Gram/RR in fp64, TF32 disabled (precision=HIGHEST) — matches our G7 finding and the mixed-precision literature.

---

## Bibliography

1. Wang, Zunger, *J. Chem. Phys.* **100**, 2394 (1994). DOI: 10.1063/1.466486. [full PDF]
2. Canning, Dongarra, Langou, Marques, Tomov, Voemel, Wang, PARA'08. [full PDF] (Vömel et al., *JCP* **227**, 7113 (2008), not read.)
3. Zhou, Saad, Tiago, Chelikowsky, *JCP* **219**, 172 (2006); *PRE* **74**, 066704 (2006). [abstracts]
4. Winkelmann, Springer, Di Napoli, *ACM TOMS* **45**(2), 21 (2019); arXiv:2205.02491. [abstracts; extremal-only confirmed]
5. Polizzi, *PRB* **79**, 115112 (2009). [abstract]
6. Gavin, Polizzi, *NLAA* **25**(5), e2188 (2018); arXiv:1706.00692. [HTML full-text extraction]
7. Fang, Saad, *SISC* **34**(4), A2220 (2012). DOI: 10.1137/110836535. [full PDF]
8. Pieper et al., *JCP* **325**, 226 (2016); arXiv:1510.04895. [Secs. 1–3]
9. Li, Xi, Erlandson, Saad, *SISC* **41**(4), C393 (2019); arXiv:1802.05215. [Maxwell experiments, recommendations]
10. Li, Xi, Vecharynski, Yang, Saad, *SISC* **38**(4), A2512 (2016). [citation only]
11. Lin, Saad, Yang, *SIAM Review* **58**(1), 34 (2016). [abstract]
12. Weiße, Wellein, Alvermann, Fehske, *RMP* **78**, 275 (2006). [abstract]
13. Di Napoli, Polizzi, Saad, *NLAA* **23**(4), 674 (2016). [not read; secondary]
14. Vecharynski, Yang, arXiv:1602.02306. [summary]
15. Szyld, Vecharynski, Xue, arXiv:1504.02811. [abstract]
16. Vecharynski (MERL TR2016-165), arXiv:1609.05407. [abstract]
17. Knyazev, *SISC* **23**(2), 517 (2001). [citation]
18. Johnson, Joannopoulos, *Opt. Express* **8**, 173 (2001); MPB developer docs. [docs read]
19. DFT-FE group, R-ChFSI, arXiv:2503.22652 (2025). [abstract+summary]
20. Kressner, Ma, Shao, *Numer. Algorithms* **94**, 1653 (2023). [abstract]
21. Ni et al., AdaPolySI, ICS'26. DOI: 10.1145/3797905.3800553. [paywalled; abstract]
22. Loe, Morgan (2019); Embree, Loe, Morgan, *SISC* (2020). [abstracts]
23. Huang et al., *J. Comput. Appl. Math.* (2014). [abstract]

**Bottom line.** Bake off (1) folded-spectrum LOBPCG with the Wang–Zunger squared-kinetic diagonal preconditioner and σ in the gap, against (2) mid-pass filtered subspace at degrees ~800–3,200 exploiting the gap as free transition margin — with a KPM/DOS count oracle as shared infrastructure and the §5 checklist as the acceptance gate. Published evidence puts both at ~10³ Θ-applies per pair in this window-ratio regime; folding is the only option whose active set fits in 12 GB VRAM, while filtering has the stronger completeness story and the only direct published fp32 validation. Skip IFEAST and bare shift-invert MINRES as primary arms — though (c) is still run at small scale in the bake-off, because the kickoff requires the negative result to be measured, not assumed.
