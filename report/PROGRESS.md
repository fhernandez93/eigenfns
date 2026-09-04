# PROGRESS — PRL report package (`report/`)

Started 2026-08-31. Autonomous session; every routine decision recorded here.
Conventions: all Python via `/home/francisco/miniconda3/envs/lsu_ml/bin/python` (numpy/scipy/matplotlib, CPU only); the USB production data is read-only; nothing outside `report/` is modified.

## 0. Decisions taken up front
- **Numbers pipeline**: each `report/scripts/sNN_*.py` writes a ledger `report/numbers/<script>.json` ({value, unit, source_file, script, notes}); `build_numbers.py` merges them into `report/numbers.json` and refuses conflicting duplicate keys.
- **Every recomputable number was recomputed** (spectra, gaps, spacings, PR, ξ, seam fractions, ff, shell ff, changed voxels, KPM counts/brackets/bias, level statistics, MPB parity, G9, I1/I4 eigenvalue parity, cross-grid shifts, periodic overlaps). Timings, Θ-application counts and bake-off metrics are cited from their JSON/log files.
- **ν convention**: ν = √λ·a/2π with a = L/5 = 2.288 µm for N=1000; the density-matched N=10k box gives the same a (L/1250^{1/3} = 2.288 µm). Verified in `s06_tables.py`.
- **Nominal N=10k gap**: the registered KPM 10 %-criterion bracket [1.864, 1.996]. Its "local median" normalisation could not be reproduced (see §2.3); the bracket is reproduced *exactly* as the interval where the Jackson-smoothed DOS is below 160 states per unit λ, and the 5 %/20 % brackets as ρ < 80 / 320. The paper states the criterion in these absolute units.
- **Symmetry class for level statistics**: Θ at Γ with real scalar ε is real symmetric → orthogonal class (GOE ⟨r⟩ = 0.5307, Atas et al. 2013; Poisson 2 ln 2 − 1 = 0.3863). GUE (0.6027) is quoted only as "not the class".
- **Tile figures** reuse the project's rendered PNGs (pyvista volume render of ε|E|²; per-tile clip at the 99.9th percentile; translucent grey network). No re-rendering (pyvista is available but the tiles are the montage convention as published).
- The nested `report/.gitignore` un-ignores `*.png/*.pdf/*.npy/*.npz` under `report/` (the repo root ignores them).

## 1. Timeline
- Read docs/REPORT_N1000.md, docs/REPORT_N10K.md in full; docs/plans/ digested by a subagent (verbatim gate criteria, all four adversarial rounds); gate ledger, all interior_report.json, data shapes, code conventions (`eigenfns/localization.py`, `structure.py`, `operator.py`, `interior.py`, `render.py`) read directly.
- `s01_spectra.py` … `s06_tables.py`, `fig1`–`fig4`, `figs_sm.py` written and run (all CPU, total < 5 min).
- `main.tex` drafted; `refs.bib` drafted (46 entries, every one with a DOI); reference-checker subagent launched on `refs.bib` only.

## 2. Findings from the recomputation (disagreements with the markdown reports, all resolved in favour of the recomputed value)

### 2.1 Reproduced exactly (no action)
Gap 500|501 elliptical 2.076 % (report 2.08 %), series 2.354/1.929/2.076 %; circular 5.066 %; N=10k largest interior spacing 1.8860–1.9264; ten in-gap λ; Rayleigh shift mean −4.73×10⁻⁵; I1 parity 2.35×10⁻⁷ (raw 2.85×10⁻⁵, factor 121); G9 2.44×10⁻⁷; MPB parity 4.3×10⁻⁶ / 8.95×10⁻⁶ / 3.55×10⁻⁵ (32³/64³/64³-660); all PR and ξ values of the four localization JSONs (max relative difference 1.3×10⁻⁶ on ξ, 0 on PR); 121/133 resolved, 12 fail r² (1 also above ceiling); 168/210 N=1000 circular unresolved; band 500/501 ξ = 1.467/2.298 µm (circ), 1.822/2.921 µm (ell); seam shell fractions 18.2/23.6/43.7/42.1 %; shell ff 0.1975/0.2144/0.2201/0.2211; 5528 voxels (0.078 %) changed by wrapping, all within 3 voxels of a face; ff 0.22011 (192³), 0.22089 (periodic), 0.22000 (256³), 0.2172 (N=1000 ell 128³ and 256³), 0.2191 (circ 128³); KPM 5010.34±12.54 below λ=1.957, 139.28±2.81 in the production window, 317.09±5.10 in [1.71,2.19], 4.87±0.41 in S_gap, 11.18±0.58 in the bracket, 69.2±1.5 / 66.1±2.4 per slice, 213.3±2.6 for the N=1000 window, DOS 1281/996 at the window edges with slopes −1.9×10⁴/+1.1×10⁴, bias +7.7 (report +7.5, kernel-width convention), index bias −4.5 (report −4.4); cross-grid |Δω/ω| 0.338 %/0.039 %; I8 20 gap-edge modes, max 8.2×10⁻⁴ / median 6.4×10⁻⁵ (ledger 8.7×10⁻⁴ / 6.4×10⁻⁵ — different rounding of the "within 0.15" edge window).

### 2.2 Corrections / precisions adopted in the paper
1. **Δλ/λ vs Δω/ω.** REPORT_N10K labels the I1 parity "max Δω/ω = 2.4×10⁻⁷"; the scorer computes |Δλ|/λ. The paper writes Δλ/λ = 2.4×10⁻⁷ (Δω/ω = 1.2×10⁻⁷). Same for I4 (4.1×10⁻⁷ is Δλ/λ).
2. **Cross-slice duplicates.** REPORT_N10K: the two states found by both S_gap and S_below agree to 1.4×10⁻⁶ and 1.7×10⁻⁶ "because both carried the same bias". Recomputed: those are the *raw* differences; the corrected values agree to 7.7×10⁻⁹ and 1.2×10⁻⁸, because the two solves had *different* ‖x‖² (1.0000475 vs 1.0000461). The paper quotes the corrected figure; this is an independent confirmation of the normalisation fix.
3. **Gap centre ν.** docs/REPORT_N1000.md quotes ν ≈ 0.516 for the N=1000 gap centre "at a = 2.288 µm". √(1.9229)·2.288/2π = 0.505 (elliptical), 0.505 (circular). 0.516 is not reproducible with a = 2.288 (it would need a = 2.337 µm). The paper uses 0.505.
4. **"ξ = 1.5–2.3 µm (N=1000)"** in REPORT_N10K §0 is the band-500/501 pair (1.47 / 2.30 µm), not a range over edge modes: resolved modes within 0.15 of the N=1000 edges span 1.47–4.95 µm (20 modes, median 3.1). The paper quotes the pair explicitly as the two edge states.
5. **KPM bracket normalisation** (see §0): the project's `exp_kpm_analyze.py` uses the median DOS over a Chebyshev–Gauss grid in [0.5g, 1.6g]; reproducing that gives the *first-pass* bracket [1.8877, 2.0262] of INV17, not the fine-grid [1.864, 1.996]. The DOS at the registered edges is 158.9/160.2, so the registered brackets correspond to absolute thresholds 80/160/320 states per unit λ (5/10/20 % of ≈1600; the median over the production window is 302, over [1,3] it is 3735). Stated as a convention in the paper.
6. **Mid-gap count** 5010.3±12.5 is the count below λ = 1.957 (the project's reference σ); at the 10 %-bracket centre 1.930 it is 5008.3±12.6. Both are quoted with their λ.
7. **In-gap DOS floor**: recomputed 72 states per unit λ at λ = 1.93 (report "≈ 60"); the paper does not quote the floor.
8. **Fit-range sensitivity**: REPORT_N10K round-3 F7 quotes "10–15 % for compact modes, 2–3× for window-edge modes" and a "fit-range-robust contrast ≈ 2.4×". With r_hi 0.95 → 0.60 L/2 I find a median |Δξ|/ξ of 30 % over all resolved modes (compact modes median ratio 0.75, extended 0.66) and window-end/near-edge contrasts 1.7/2.2× (r_hi 0.95) vs 1.8/1.6× (r_hi 0.60); the PR-radius contrast is 1.8/1.7×. Different bins and range than the record; the paper quotes my definitions and numbers.
9. **Level statistics** are new: see `s05_levelstats.py` and the paper.

### 2.3 New findings (not in the reports) that the paper discloses
- **MPB judge grid vs binary raster at 64³.** On MPB's trilinear read-back of our 64³ grid (the G3w protocol), the elliptical 500|501 spacing is 0.60 % and the largest spacing sits at 501|502 (0.83 %), while on our binary 64³ raster it is 500|501 at 2.35 %. Same solver on both grids: gap-edge frequencies differ by −1.4 % to −3.3 % (bands 498–503). This does not affect the parity claim (the two codes agree to 3.5×10⁻⁵ on the same discrete problem), but the "gap exactly at 500|501" statement is convention-dependent at 64³. Disclosed in the main-text limitations and SM.
- **The N=1000 rasterizations carry the same seam** (outer-shell ff deficit 12.8 % elliptical, 11.3 % circular at 128³) and some N=1000 window modes have outer-2-voxel-shell enhancement up to 3.95× (ell) / 3.43× (circ). The N=1000 gaps are nevertheless empty. Disclosed in the SM seam section.
- 20 of the 133 N=10k modes have shell enhancement > 2 (not only the four in-gap seam states); the bulk controls used in the retracted z-test have 0.49–1.43× (report "0.6–0.9×").
- Corrected residual floor: the reported worst residual 8.7×10⁻⁵ becomes ≈ 5.9×10⁻⁵ after removing the normalisation floor (consistent with the ledger's direct fp64 recomputation 5.88×10⁻⁵).

## 3. Left out / blocked (running list)
- The Thouless-ratio / twisted-boundary sensitivity cannot be computed from the saved Γ-point data (no re-solve permitted). Stated in the paper.
- No re-render of tiles for the periodic re-solve (no tiles exist; energy densities exist but re-rendering would be a new artefact). The periodic-solve localization numbers are recomputed and tabulated instead.

## 4. What was left out and why
(filled at the end)

## 5. Draft complete (commit 78a0e86, 2026-08-31)
- `main.tex`: 5 pages in `prl,twocolumn` including ~1.3 pages of references; body+abstract 2385 words, captions 278, figure equivalents 667 (PRL formula) → ≈3330 words-equivalent ≤ 3750. Abstract ≤ 600 rendered characters (measured with pdftotext). 4 figures.
- `supplement.tex`: 34 pages (REVTeX preprint), 12 sections, 8 tables incl. the 133-row state table, 16 figures.
- `build.sh` builds both with tectonic (REVTeX 4.2 fetched by tectonic), greps for undefined references/citations, prints page counts; `--figures` regenerates everything from the saved data.
- Reference check: 41/41 DOIs resolved by an independent checker (Crossref + second sources); two author-list corrections applied (Yamilov 2023; Imagawa 2010). `references_verified.md` written.
- Fact-check round 1 launched: five SM section checkers (A–E) and three main-text checkers, each with the section text and the primary file list only; two adversarial reviewers (physicist, numerical analyst) launched on the full drafts.

## 6. Fact-check round 1 (2026-08-31, commits 78a0e86 → b24da21)
- Completed: SM checkers B (87 claims, 2 WRONG), C (134, 2 minor WRONG), D (73, 11 WRONG), E (78, 0 WRONG); main-text checker 2 (78, 8 WRONG). All WRONG items corrected; ledger in `FACTCHECK.md`.
- Substantive corrections: GOE surmise prefactor 27/8 → 27/4 for the folded ratio (affected LLRs, expected σ, the "1.7σ" → "2.2σ" statement, and the P(r) curve in Fig. S9); five mis-transcribed short-range ξ medians in the SM; candidate persistence overlaps 0.994–1.000 → 0.968–1.000 and shift range −0.0034 → −0.0030 (the −0.0034 belongs to the extended state); "≈5150 bands below the gap" → "up to the window top"; no basis trimming in production.
- The session usage limit terminated checker A (partial), main-text checkers 1 and 3 and both adversarial reviewers mid-run; all relaunched at low priority (A asked to deliver its partial ledger).

## 7. Fact-check round 1 completed; adversarial round 1 applied (commit 0f5fcae and after)
- Checker A (structure/operator/bottom-up): 121 claims, 5 WRONG (shell volume 9.1 % not 12.1 %; G4 "64 %" → 26 % under the registered criterion; G7 cross-overlap ≤2×10⁻⁶ by fp64 recomputation; I4 pooled median 7.9×10⁻⁸; the 128³ Θ-count is not logged). Main-text checkers 1/2/3: 3 + 8 + 0 WRONG. All corrected (FACTCHECK.md).
- Physicist reviewer: ten MAJOR findings, all accepted in substance (grid-refinement robustness restricted to the three tested candidates; P(empty N=1000 gap) recomputed over the empty interval → 0.15; seam contamination of the near-edge level-statistics band disclosed; "artefacts" → "seam-flagged"; verdict no longer says "Anderson-type"; the 12-level window declared post hoc and its above-gap counterpart's repulsion reported; "size-independent" removed from abstract/verdict; ten certified in-gap states in the abstract; 1.9441 called compact/non-exponential, not extended).
- Numerical-analyst reviewer: three MAJOR (sub-interval completeness precision → |missed| < 0.01; level-statistics accuracy sentence separates the Weyl guarantee from the N=1000 measurement; the I3 cross-slice Gram deviation is within the residual-gate bound, so the registered criterion was unattainable) plus eleven minors, all applied.
- New finding from the numerics review adopted as our own statement: the I3 "open mechanism" is explained by |⟨xᵢ,xⱼ⟩| ≤ (‖rᵢ‖+‖rⱼ‖)/|λᵢ−λⱼ| (verified for the worst pair and the top-5 list).
- The kickoff's suggested verdict wording ("consistent with Anderson-type localization") was tightened to "consistent with localization at the edges of a photonic pseudogap … whether interference-driven or trap-like cannot be decided from this data", following the adversarial review; the kickoff itself required writing only what the data support.
- Fig. 1(c) corrected: it had overlaid the circular-decoration exact spectrum on the elliptical-decoration KPM curve; it now shows the binned density of the same exact spectrum.
- Round 2 (fresh checkers on the corrected main text and SM) launched.

## 8. Fact-check round 2 and close-out
- Round-2 checkers (fresh agents on the corrected text): SM A+B 119 claims / 3 WRONG (minor); main text 84 / 5 WRONG (minor); SM C+D+E 146 / 8 WRONG (minor). All corrected (FACTCHECK.md). Two fp32-scored project numbers were replaced by fp64 recomputations made here: duplicate overlaps 0.9999999 (recorded 0.9988) and I4 min proj² 0.9999 (recorded 0.9961).
- Per the user's instruction (mid-session) no further agents were launched; the final corrections were verified directly by scripts.
- Final build: `bash build.sh --figures` regenerates all 501 ledger entries, 21 tables/CSVs and 20 figures from the saved data in ≈4 min and builds both PDFs with zero undefined references. main.pdf: 5 pages, body+abstract ≈2530 words, captions ≈310, figure equivalents ≈610 → ≈3450 words-equivalent (PRL limit 3750); abstract ≈ 600 rendered characters (612 by pdftotext including two page-header artefacts). supplement.pdf: 36 pages.
- `references_verified.md`: 41/41 DOIs resolved; two author corrections applied.

## 4. What was left out and why (final)
- **Thouless ratio / boundary-twist sensitivity**: not computable from the saved Γ-point data; no re-solve permitted. Stated in the paper.
- **Third independent fact-check round**: not run (user instruction to stop launching agents); replaced by direct script verification of the round-2 corrections. Round 2 had found only minor numerical/provenance items.
- **MPB parity at 128³/192³**: does not exist in the project; the paper states parity at 64³ only.
- **A post-fix (periodic) KPM run, an I2 audit of the 256³ anchors, and tiles for the periodic re-solve**: do not exist; not generated (no GPU work allowed).
- **The G2 srs-scan MPB output**: not retained by the project; the 27.97 % is quoted as a ledger value with that disclosure.
- **Acknowledgments**: placeholder removed to fit the page budget; author/affiliation placeholders kept as instructed.
