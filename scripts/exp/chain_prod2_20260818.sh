#!/bin/bash
# Detached chain (Amendment A1): I1 completion (polish cap 6) -> re-score ->
# N=10k S_below -> S_above (both cap 6, early-stop).
set -e
cd /home/francisco/Documents/Eigenfuntions

echo "=== $(date) I1 completion (resume, polish outers -> 6)"
conda run --no-capture-output -n lsu_ml python scripts/run_interior.py \
  "/home/francisco/Documents/Create LSU Structures  - Claude/Example/N1000_lsu_example_ends.txt" \
  --grid 128 --lam-lo 1.70129 --lam-hi 2.15583 --m 80 \
  --build-degree 3000 --build-outers 2 \
  --polish-degree 8000 --polish-outers 6 --tol 1e-4 \
  --radius 0.2252 --aspect 2.5 --eps-rod 8.57025625 \
  --lam-max 4633.6 --tag i1_n1000_slice --resume \
  >> results/i1_n1000_slice.log 2>&1

echo "=== $(date) re-scoring I1"
conda run --no-capture-output -n lsu_ml python scripts/exp/exp_i1_score.py \
  --interior results/i1_n1000_slice --reference results/prod_N1000_G128 \
  --ref-lo 395 --slice-lo 473 --slice-hi 523 \
  --gate-name "I1 ground-truth parity (50-band slice, production config)" \
  >> results/i1_n1000_slice.log 2>&1

echo "=== $(date) I1 complete; launching N=10k S_below"
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

echo "=== $(date) production chain v2 done"
