# Pre-registration (FROZEN 2026-08-18) — interior gap-edge eigenmodes, N=10,000 LSU network

Frozen before build-out, per methodology. Supersedes the 2026-08-17 DRAFT
(kept for the record). No gate below may be weakened after this point; a FAIL
+ explanation beats a massaged PASS. Basis: measured Phase 1 numbers
(`docs/plans/2026-08-17_interior_investigation.md`) and the completed bake-off.

## 1. Problem (final)

- Structure `Structures/20260701_N10000_lsu_generated.txt`; N = 10,000;
  L = 24.6467 µm; decoration circular aspect 1.0, ε_rod = 8.41,
  **minor_radius = 0.331836 µm** (measured ff 22.000–22.011% across all grids).
- **Production grid 192³** (7.79 vox/µm; matvec 23.24 ms/vec measured;
  λ_max = 2208.9 Lanczos×1.05; vector 0.113 GB c64). The full-window ask at
  256³ measured-projects to 10–15 GPU-days — the kickoff's sanctioned descope
  (grid) fires at registration, not mid-run. Production physical-resolution
  anchoring is provided by I6 (below).
- **Window: λ ∈ [1.757, 2.117] — 139 ± 3 bands** (KPM counting, paired-probe
  error; 64 bands per side of the 10%-criterion gap edges [1.864, 1.996] plus
  the smeared edge population; band indices ≈ MPB 4941–5080, placement ±13).
  This descopes deliverable 1's "~300" to the gap-edge window the measured
  cost wall supports (~2.5 GPU-days vs ~6 at 300); recorded as a shortfall
  with a stretch goal: widen symmetrically if actual cost lands ≤ 60% of cap.
- σ (reference only) = 1.957. Band numbering MPB (+2); ±2 caveat in captions.

## 2. Method (fixed by bake-off — two-stage bandpass ChebSI)

Subspace = Jackson step-difference bandpass filter of Θ applied by Chebyshev
recurrence (`eigenfns/interior.py`); extraction ONLY via Rayleigh–Ritz on the
original Θ (fp64 host) + per-pair rel-res gate ≤ 1e-4 + cluster-projection
dedup. Rejected by measurement (records in investigation §6): folded spectrum
(all preconditioners), shift-invert PMINRES, expansion-RR polish.

Production slices (filter windows; targets from KPM counts):
| slice | filter [λ_lo, λ_hi] | targets | m (=1.5×) | free margin side |
|---|---|---|---|---|
| S_below | [1.757, 1.930] | 69 ± 1.5 | 104 | upper edge in gap |
| S_above | [1.980, 2.117] | 66 ± 2.4 | 100 | lower edge in gap |

- Build: degree **3000**, 2 outers, from random start (seed recorded).
- Polish: same filter machinery, degree **12,000** (transition ≈ 0.017 λ ≈
  ~30 bands, absorbed by the ~35 spare vectors + gap margin), up to **4**
  outers; stop early when every in-slice pair < 1e-4.
- Oversampling m = 1.5× slice population (bake-off: 1.12× starves edge pairs).
- Precision: fp32 matvecs, fp64 host Gram/RR, precision=HIGHEST everywhere
  (unchanged; bake-off showed no fp32 filter floor down to ~1e-4).
- Cross-slice: gap-side boundaries need no overlap (empty gap); dedup any
  double-found pairs by eigenvector overlap > 0.5.
- Checkpoint after every outer (subspace npz); auto-resume; `gpu_is_busy`
  guard; assume CUDA death.

**Projected cost** (23.24 ms × m × degrees): build 2×0.31/0.30 M + polish
3–4×1.25/1.20 M apps per slice → **56 h nominal (3 polish outers), 73 h with
the 4th**. **Hard cap: 120 h GPU for the production stage. Abort/descope
(decided now): if after S_below's build + 2 polish outers the extrapolated
two-slice total exceeds 120 h, both slices narrow to 48 targets (m = 72)
symmetric about the gap edges; if still over, drop to 160³ with the same
window.** One heavy GPU job at a time.

## 3. Gates (exact numbers, frozen)

| gate | test | tolerance |
|---|---|---|
| I1 ground-truth parity | production-config interior solve (build d=3000×2, polish d=8000×≤4, m=80) of the N=1000/128³/production-decoration 50-band slice (0-based 473..522) vs `prod_N1000_G128` | all 50 found; Δω/ω ≤ 1e-4 each (expect ~3e-5, measured); cluster proj² ≥ 0.99 (cluster = rel λ within 1e-3); 0 ghosts (converged pair matching no reference band) |
| I2 completeness | deflated-probe KPM count (degree 12,000, ≥8 probes) of residual states in each slice window after deflating converged vectors — N=1000 slice first (vs exact), then both N=10k slices | count ≤ 0.5 (G6-style; expect ~0.2 leakage bias); any count ≥ 1 = missed state(s): resolve (more outers) or FAIL, never explain away |
| I3 residuals / no ghosts | every reported pair: rel-res ≤ 1e-4 on Θ; window Gram ‖G−I‖_max ≤ 5e-5; transversality exact by construction (assert); ghost checklist §5 (survey) executed and logged | as stated |
| I4 new-decoration cross-check | N=1000 circular/2.9/ff22 @128³: bottom-up (running, `i4_n1000_circ_G128`) vs interior solver, full montage window bands 398–607 | same numbers as I1 |
| I5 spectrum consistency | N=10k window eigenvalue extremes + empty-gap extent vs KPM DOS | eigen-edges inside the KPM 5%→20% criterion bracket ([1.885,1.968]→[1.837,2.022]); gap interval empty of converged pairs |
| I6 convergence | gap-edge 40-band subset (20/side nearest gap) at **160³ and 192³** full re-solve + **256³ mini-slices (~10 bands/side)** as production-resolution anchor; Δν/ν and per-band Δω/ω across grids | report with G5 framing; expected scatter ~0.3%; non-monotone ≤ 0.6% or honest FAIL (G5 precedent: 0.27% at N=1000) |
| I7 decoration | measured ff on 192³ production grid | 22.0 ± 0.5% (measured 22.011%); radius 0.331836 recorded |
| I8 localization | IPR/ξ pipeline (validated on N=1000: band 500 ξ=1.82 µm r²=0.97; extended modes auto-flagged) applied to I4's two solvers on the same modes; N=10k ceiling ξ_max = L/2 = 12.32 µm stated on every figure; any fitted ξ ≥ 12.32 or dyn-range < 1 decade or r² < 0.7 ⇒ "unresolved — lower bound only" | both-solver ξ agreement ≤ 10% on resolved N=1000 gap-edge modes; flags fire on known-extended modes |
| I9 montage | N=10k gap-edge window ε\|E\|², 15/row, MPB numbering + ±2 caveat; side-by-side with the I4 N=1000 new-decoration montage (finite-size comparison, both with ξ-ceiling caveat) | tile count = I2-certified band count; recorded in gate_results.json (G9's omission was an open item) |

All gates land in `results/gates/gate_results.json` + adversarial verification
round 2 before any PASS is claimed.

## 4. Memory / disk (54 GB free at kickoff; recheck before production)

Per-slice host basis ≤ 104 × 0.113 GB = 11.8 GB (RAM); GPU transient ≤ 9.5 GB
(all block ops 8-row-chunked — measured discipline from the bake-off OOM
series). Disk: per-outer checkpoint (one subspace, 11.8 GB, rolling — keep
latest only); final window vectors 139 × 0.113 = 15.7 GB fp32 + ε|E|² 3.9 GB
+ I4/I6 artifacts ~15 GB → ~35 GB new. Prune ladder (in order, only if
< 10 GB free): `results/conv_N1000_G96` (12 GB, regenerable), rolling
checkpoints of completed slices, `results/exp/bakeoff_*_subspace.npy`.

## Amendment A1 (2026-08-18 15:00, recorded BEFORE production execution)

I1 first run: 49/50 found, 0 ghosts, max Δλ/ω 2.8×10⁻⁵, proj² 1.0000 — but
one target missed: the near-degenerate pair ev[481]/ev[482] (Δλ = 7×10⁻⁴,
10× tighter than local spacing) was still unconverged (res > 1e-4, monotone
descending) when the registered polish cap of 4 outers ended the run;
unconverged pairs are never reported, hence the miss. **Amendment: polish
outer cap 4 → 6 (early stop unchanged: halt when all in-window pairs
converge), applied to the I1 completion and both production slices.** The
N=10k spectrum is 10× denser, so near-degenerate clusters are certain there.
No tolerance, window, or acceptance criterion changes; cost ceiling remains
120 h (worst case with 6 polish outers ≈ 94 h projected). I1 is scored
against the completed run; this first-run miss is recorded here, not hidden.

## Amendment A2 (2026-08-21 09:30, recorded during S_above, before further runs)

1. **In-gap states discovered (S_below FINAL):** λ = 1.8709, 1.8732, 1.8861,
   1.9265, 1.9297 lie at/inside the KPM gap region, isolated by ~0.04-wide
   spacings, residual-certified (≤ 6.4×10⁻⁵), ξ = 1.80–2.14 µm, PR ≤ 0.1%.
   The registration *assumed* an empty gap (KPM in-gap weight was
   leakage-consistent — review F1 bounded, did not prove). Consequences,
   decided now: (a) **gate I5's "gap interval empty of converged pairs"
   clause will be reported FAILED-as-registered** with the physics finding
   (real in-gap states), G5-precedent style; the DOS-consistency clause
   stands (measured in-gap DOS floor ≈ 60/unit-λ matches the discrete state
   density). (b) **Addendum slice S_gap [1.925, 1.985], m=16, build d=4000×2,
   polish d=12000×≤4** (projected ≈ 6 h) closes the previously uncovered
   interval (1.930, 1.980) so window completeness (I2) covers the entire
   region between the slice edges. Cross-slice dedup by eigenvector overlap
   at the boundaries as registered.
2. **Cost-cap accounting:** S_below consumed 63.3 h (5 polish outers vs 3
   nominal — the registered abort evaluation at build+2 polish outers
   projected 83 h total and passed; the extra outers emerged later).
   Two-slice extrapolation now ≈ 127 h vs the 120 h cap. Decided now:
   complete S_above at full size (descoping it would asymmetrize a
   half-completed window; the cap's purpose — bounding the burn — is served
   at ~+6%); the overrun is reported in REPORT_N10K, not silently absorbed.
   Hard stop remains: if S_above exceeds 6 polish outers it finalizes with
   whatever converged, unconverged pairs listed.

## Amendment A3 (2026-08-24, I2 estimator redesign — recorded with the failure)

**The registered I2 estimator was mis-designed and its first execution is
reported as a failure, not deleted.** As registered ("deflated-probe KPM
count in [λ_lo, λ_hi] … count ≤ 0.5") it was implemented as two independent
`kpm_count_below` calls differenced. Two defects, both measured:

1. *Precision*: each call counts ~5,000 states, so its stochastic error is
   ~√(2·5000/n_probe) ≈ 26 bands, and differencing two independent calls adds
   them. Recorded result (S_gap): missed = 0.66 ± 26.5 — an error bar 50×
   too wide to detect one missing state. The G6-style variance collapse only
   happens when *everything* contributing to the count is deflated; here only
   the ~69 window states were, while ~4,940 lower states dominate the variance.
2. *Memory*: the locked set was placed on the GPU (69 vectors at 192³ =
   7.8 GB) instead of streamed → OOM on the two larger slices. (`deflate`
   already streams host arrays; the script forced them to device.)

**A deeper issue the registered design ignored:** the window edges sit in
dense bulk, so the Jackson transition zone (width ~0.017 at degree 12,000)
covers ~27 states per edge, each contributing partial weight. This is exactly
the mechanism behind G6's "86.6 phantom missing bands" in the delivered
project. No amount of probes fixes a bias.

**v2 estimator (this amendment), `scripts/exp/exp_i2_v2.py`:** a single
Chebyshev recurrence per probe chunk accumulating the Jackson bandpass
ρ(Θ) directly (half the cost, per-probe values → paired standard error);
locked set host-resident and streamed; and the leakage handled two
independent ways — (A) predicted as ∫_outside ρ(λ)·DOS(λ)dλ from the measured
KPM DOS moments, with its own propagated error bar, and (B) re-measured on a
**sub-interval [1.9063, 1.9606] whose edges sit in the sparsest voids of the
gap region** (≥ 0.0133 from any converged state, vs a 0.0086 transition at
degree 24,000), where leakage is small by construction — the G6-amendment
logic applied honestly.

**Acceptance (unchanged in spirit, restated for the new estimator):** the
sub-interval test is the certifying one — |missed| < 0.5 required. The
full-window number is reported with its leakage correction and error bar as a
consistency check, not a certification, since its bias cannot be driven below
one state at feasible degree.

**Open tension this gate must address:** KPM predicted 139.3 ± 2.8 states in
[1.757, 2.117]; the solver converged 133 distinct pairs (2.2σ low). Either
~6 states are missing near the dense window edges, or the KPM interval count
carries an edge bias of that size. v2's leakage correction measures exactly
this. Whatever it says goes in the report.

## 5. Schedule / wall-clock

I4 finish (~2.5 h, running) → I1 + I4-interior (N=1000, ~3.4 + ~8 h) →
production S_below, S_above (56–73 h, checkpointed) → I2/I5 audits (~2 h) →
I6 (160³+192³ ~9 h; 256³ anchors ~20 h) → montage + localization (CPU) →
REPORT_N10K. Every run resumable; `gpu_is_busy` enforced.
