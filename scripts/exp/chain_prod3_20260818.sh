#!/bin/bash
# Production chain v3 (hosted-basis solver): N=10k S_below -> S_above.
set -e
cd /home/francisco/Documents/Eigenfuntions

echo "=== $(date) launching N=10k S_below (hosted basis)"
conda run --no-capture-output -n lsu_ml python scripts/run_interior.py \
  Structures/20260701_N10000_lsu_generated.txt \
  --grid 192 --lam-lo 1.757 --lam-hi 1.930 --m 104 \
  --build-degree 3000 --build-outers 2 \
  --polish-degree 12000 --polish-outers 6 --tol 1e-4 \
  --radius 0.331836 --aspect 1.0 --eps-rod 8.41 \
  --lam-max 2208.8753 --chunk 8 \
  --tag n10k_G192_Sbelow --resume \
  > results/n10k_G192_Sbelow.log 2>&1

echo "=== $(date) S_below done; launching S_above"
conda run --no-capture-output -n lsu_ml python scripts/run_interior.py \
  Structures/20260701_N10000_lsu_generated.txt \
  --grid 192 --lam-lo 1.980 --lam-hi 2.117 --m 100 \
  --build-degree 3000 --build-outers 2 \
  --polish-degree 12000 --polish-outers 6 --tol 1e-4 \
  --radius 0.331836 --aspect 1.0 --eps-rod 8.41 \
  --lam-max 2208.8753 --chunk 8 \
  --tag n10k_G192_Sabove --resume \
  > results/n10k_G192_Sabove.log 2>&1

echo "=== $(date) production chain v3 done"
