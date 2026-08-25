#!/bin/bash
# Post-production chain: waits for S_above -> S_gap addendum slice ->
# I2 audits (all three slices) -> I6 gap-edge subset runs (160^3, 192^3-anchor
# from production, 256^3 mini-anchors) -> I4-interior cross-solve.
set +e
cd /home/francisco/Documents/Eigenfuntions

# (resumed 2026-08-24: production already complete)


echo "=== $(date) S_above done; S_gap addendum slice (Amendment A2)"
conda run --no-capture-output -n lsu_ml python scripts/run_interior.py \
  Structures/20260701_N10000_lsu_generated.txt \
  --grid 192 --lam-lo 1.925 --lam-hi 1.985 --m 16 \
  --build-degree 4000 --build-outers 2 \
  --polish-degree 12000 --polish-outers 4 --tol 1e-4 \
  --radius 0.331836 --aspect 1.0 --eps-rod 8.41 \
  --lam-max 2208.8753 --chunk 8 \
  --tag n10k_G192_Sgap --resume \
  > results/n10k_G192_Sgap.log 2>&1

echo "=== $(date) I2 completeness audits"
for T in Sbelow Sgap Sabove; do
  conda run --no-capture-output -n lsu_ml python scripts/exp/exp_i2_completeness.py \
    --rundir results/n10k_G192_$T --degree 12000 --probes 8 \
    --gate-name "I2 completeness ($T)" --force \
    > results/exp/i2_$T.log 2>&1
done

echo "=== $(date) I6: gap-edge 40-band subset at 160^3"
conda run --no-capture-output -n lsu_ml python scripts/run_interior.py \
  Structures/20260701_N10000_lsu_generated.txt \
  --grid 160 --lam-lo 1.80 --lam-hi 2.06 --m 72 \
  --build-degree 3000 --build-outers 2 \
  --polish-degree 10000 --polish-outers 5 --tol 1e-4 \
  --radius 0.331836 --aspect 1.0 --eps-rod 8.41 \
  --chunk 8 --tag n10k_G160_gapedge --resume \
  > results/n10k_G160_gapedge.log 2>&1

echo "=== $(date) I6: 256^3 gap-edge mini-anchors"
conda run --no-capture-output -n lsu_ml python scripts/run_interior.py \
  Structures/20260701_N10000_lsu_generated.txt \
  --grid 256 --lam-lo 1.84 --lam-hi 1.95 --m 18 \
  --build-degree 4000 --build-outers 2 \
  --polish-degree 16000 --polish-outers 4 --tol 1e-4 \
  --radius 0.331836 --aspect 1.0 --eps-rod 8.41 \
  --chunk 4 --tag n10k_G256_edgelow --resume \
  > results/n10k_G256_edgelow.log 2>&1
conda run --no-capture-output -n lsu_ml python scripts/run_interior.py \
  Structures/20260701_N10000_lsu_generated.txt \
  --grid 256 --lam-lo 1.99 --lam-hi 2.08 --m 18 \
  --build-degree 4000 --build-outers 2 \
  --polish-degree 16000 --polish-outers 4 --tol 1e-4 \
  --radius 0.331836 --aspect 1.0 --eps-rod 8.41 \
  --chunk 4 --tag n10k_G256_edgehigh --resume \
  > results/n10k_G256_edgehigh.log 2>&1

echo "=== $(date) I4-interior: N=1000 new-decoration window, 2 slices"
conda run --no-capture-output -n lsu_ml python scripts/run_interior.py \
  "/home/francisco/Documents/Create LSU Structures  - Claude/Example/N1000_lsu_example_ends.txt" \
  --grid 128 --lam-lo 1.470 --lam-hi 1.835 --m 155 \
  --build-degree 2500 --build-outers 2 \
  --polish-degree 8000 --polish-outers 6 --tol 1e-4 \
  --radius 0.331836 --aspect 1.0 --eps-rod 8.41 \
  --chunk 8 --tag i4int_n1000_below --resume \
  > results/i4int_n1000_below.log 2>&1
conda run --no-capture-output -n lsu_ml python scripts/run_interior.py \
  "/home/francisco/Documents/Create LSU Structures  - Claude/Example/N1000_lsu_example_ends.txt" \
  --grid 128 --lam-lo 2.015 --lam-hi 2.55 --m 160 \
  --build-degree 2500 --build-outers 2 \
  --polish-degree 8000 --polish-outers 6 --tol 1e-4 \
  --radius 0.331836 --aspect 1.0 --eps-rod 8.41 \
  --chunk 8 --tag i4int_n1000_above --resume \
  > results/i4int_n1000_above.log 2>&1

echo "=== $(date) post-production chain done"
