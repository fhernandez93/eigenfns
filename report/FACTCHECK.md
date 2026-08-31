# FACTCHECK — claim ledger

Protocol (kickoff §5.2): after the first complete draft, independent fact-checker subagents received one section of text each plus the list of primary source files, with the instruction to open the `.npy/.json/.log` files themselves and not to trust the markdown reports or `numbers.json` as evidence. Two adversarial reviewers (physicist; numerical analyst) received the full drafts with the instruction to refute. Every WRONG entry is listed with the correction applied; the "round 2" section records the fresh pass on the corrected text.

Verdict codes: VERIFIED / WRONG / UNVERIFIABLE (not decidable from files in the repository).

## Round 1 — Supplemental Material

### Part D: Localization methodology + Level statistics (checker D)
73 claims: 56 VERIFIED (4 with caveats), 11 WRONG, 6 UNVERIFIABLE. All WRONG items corrected in `supplement.tex` / `main.tex` / `s05_levelstats.py` / `figs_sm.py`:

| # | claim (as drafted) | source | verdict | correction applied |
|---|---|---|---|---|
| 1 | N=1000 circ.: "141 above ceiling, 77 fail r², overlapping" implying they exhaust the 168 | i4_n1000_circ_G128/window_energy_density.npy | WRONG (incomplete) | union = 153; 142 fail the one-decade test; text now lists all three criteria |
| 2–6 | short-range (r_hi 0.60) bin medians 2.32 / 1.76 / 2.05 / 3.66 / 3.79 µm | n10k energy densities | WRONG (transcription) | 2.01 / 1.70 / 2.45 / 3.35 / 4.04 µm (from the ledger `n10k_median_r_pr_by_bin`) |
| 7 | "ξ varies 7×" | n10k | WRONG | resolved ξ varies 6× (1.8–10.6 µm); 7× only with the unresolved 12.97 |
| 8 | I8: "both solvers hand it the same field to ∼10⁻⁷" | i4int vs i4 fields | WRONG (overstated) | eigenvalues agree to 4×10⁻⁷, projections ≥ 0.996; fields differ by 6×10⁻⁴ median rel. L2 |
| 9 | P_GOE(r) prefactor 27/8 | numeric integral | WRONG | 27/4 for the folded ratio on [0,1] (density doubles); fixed in `s05_levelstats.py`, `figs_sm.py`, SM |
| 10 | "GOE expects 0.4 %" of spacings < 0.05 | Wigner surmise | WRONG | 0.2 % |
| 11 | LLR "≥27 nats for every band ≥40 levels; −1.5 for the 12-level band" | recomputed with the corrected surmise | WRONG | +5.3…+19.9 for ≥40-level bands (+63.6/+85.2 for the 611-level spectra), −2.6 for the 12-level band, −2.3 for the ten in-bracket levels |
| c1 | "near-edge" contrast denominator | — | VERIFIED with caveat | the bin [1.80, 1.864] is now named explicitly |
| c2 | ">2.051" | — | VERIFIED with caveat | ">2.05" |
| c3 | "611 spacings" | — | VERIFIED with caveat | 610 |
| c4 | I8 "≤0.1 %" | — | VERIFIED with caveat | ≤0.08 % on 20 gap-edge modes, ≤0.22 % over all 42 |
| — | all 17 rows of Table S9 (⟨r⟩ ± s.e.) | recomputed | VERIFIED | — |
| — | 121/133, 12 fail r², 1 above ceiling; 42/210, 141, 77; 30/210; median |Δξ|/ξ = 30 %; compact/extended ratios; bin medians (full range); r_p and p by bin; 5 % local-spacing test | recomputed | VERIFIED | — |
| — | UNVERIFIABLE (6): thresholds "pre-registered" (docstring only), hot-spot spacing (no data in repo; source line quoted), ⟨r⟩_GOE = 0.5307 (literature), G4 statement, Thouless statement, sliding-window method | — | — | wording adjusted: "measured in adversarial round 3" |

Consequence in the main text: the near-edge band is "2.2σ below the GOE expectation for 12 levels" (was 1.7σ, computed with the mis-normalised surmise).

### Part E: Retracted claims + Limitations + Tables (checker E)
78 claims: 78 VERIFIED, 0 WRONG, 0 UNVERIFIABLE. Includes every row of the 133-state table (0 mismatches), the in-gap table, both N=1000 edge tables, all retraction statements against the adversarial record, `zperm.json` (|z|/σ_null 2.775 vs 7.191; permutation p = 0.9989 recomputed), `retention.json` (96³ cube: 0.99795–0.99884; 192³ set: 0.999755–0.999843 over 11 + 3 = 14 modes), all limitation numbers. Four notes adopted: "168/210 ceiling-limited" → "168 unresolved (142 at the ceiling)" (main + SM); "11–13 % shell deficit" → "10.7–12.8 %"; "+7.7"/"−4.5" flagged as the paper's recomputation (already so labelled); N=1000 edge-table eigenvalues are bottom-up values without the normalization bias (caption note added; MPB parity median 1.4×10⁻⁶ excludes a 2.8×10⁻⁵ shift).

### Part B: Interior solver — KPM, filter, bake-off, Rayleigh correction (checker B)
87 claims: 68 VERIFIED, 8 PARTIALLY VERIFIED, 6 UNVERIFIABLE, 2 WRONG. Corrections applied: (1) "≈5150 bands below the gap" → "up to the top of the derived window (the gap sits at ≈5010)"; (2) "polish on the (trimmed) basis" → no trimming in production (m = 104/100/16 throughout; the bake-off arm trimmed 80→56). Qualifications adopted: two-stage Θ count cumulative vs polish-only wall time (caption); "×12 per polish outer" → "up to ×12 (×2–7 in later outers)"; "≈27 states per edge" → "16–23 at the production-window edges"; duplicate agreement figures labelled relative; "26 unconverged" → "mostly edge pairs"; filtered-Lanczos memory argument quantified (39–79 GB / 90–180 GB vs 62 GB); Vömel 2008 attribution softened to "consistent with the ESCAN group's experience". Items only supported by the project record are now labelled "(project record)": hosted-vs-device 4.5×10⁻⁸, four (not five) expansion-RR variants, the folded α² probes, the fp32 −3.4×10⁻⁴ trap value (accumulation-order dependent).

### Part C: Interior runs, completeness, gates I1–I9, seam test (checker C)
134 claims: 129 VERIFIED (≈9 as ledger/plan quotes), 2 WRONG (minor), 3 UNVERIFIABLE. Corrections: "8 PASS, 3 FAIL" → explicit list (I1, I2 v2, I4, I6, I7, I8, seam PASS; I2 v1, I3, I5 FAIL; I9 done); "within 0.025" → "0.026". Caveats adopted: wall times of resumed runs are final-leg only (caption); edgelow wall 27.2 h vs ledger 20.5 h disclosed; seam budgets 5×10⁻⁵/0.020/0.009 and "5–2000×"; periodic state quoted with v1 rounding 1.9840; "six of ten worst pairs" attributed to the ledger diagnosis; the interruption tally reduced to what the records support. All rows of Tables S5 (runs), S7 (cross-grid) and S8 (periodic localization) reproduce to the printed digits.

## Round 1 — Main text

### Results N=1000 / N=10⁴ paragraphs, Fig. 2 and Fig. 3 captions (checker main-2)
78 claims: 68 VERIFIED, 8 WRONG, 2 UNVERIFIABLE (visual). Corrections applied: candidate persistence overlaps 0.994–1.000 → **0.968–1.000** (state 1.9901 has 0.968); shifts −0.0007…−0.0034 → **−0.0007…−0.0030** (−0.0034 is the extended state); "p = 12–37 % far from the gap" → 9–37 %; "168 hit the ceiling or fail the shape test" → 168 unresolved, 142 at the ceiling, rest decade/shape; "three decades" → 2.5 decades; 25× → 26×; 9× → 10×; Fig. 2(b) caption now names 1.8501/2.0052 as the nearest states outside the nominal gap rather than "gap edges". Every other number (KPM counts, brackets, ff, shell fractions, ten in-gap λ, candidate ranges, cross-grid, completeness, tile labels) reproduced.

### Interrupted checkers
Checker A (structure/operator/bottom-up), main-text parts 1 and 3 and both adversarial reviewers were terminated by the session limit; they were relaunched (see round 2 below).
