#!/bin/bash
# Fires after the I4 bottom-up run frees the GPU: gate I1 interior run + score.
set -e
cd /home/francisco/Documents/Eigenfuntions

# wait for I4 to finish (its final line prints "saved bands")
until grep -q "saved bands" results/i4_n1000_circ_G128.log 2>/dev/null; do
  sleep 120
done
echo "=== $(date) I4 bottom-up done; starting I1 interior run"

conda run --no-capture-output -n lsu_ml python scripts/run_interior.py \
  "/home/francisco/Documents/Create LSU Structures  - Claude/Example/N1000_lsu_example_ends.txt" \
  --grid 128 --lam-lo 1.70129 --lam-hi 2.15583 --m 80 \
  --build-degree 3000 --build-outers 2 \
  --polish-degree 8000 --polish-outers 4 --tol 1e-4 \
  --radius 0.2252 --aspect 2.5 --eps-rod 8.57025625 \
  --lam-max 4633.6 --tag i1_n1000_slice --resume \
  > results/i1_n1000_slice.log 2>&1

echo "=== $(date) scoring I1"
conda run --no-capture-output -n lsu_ml python scripts/exp/exp_i1_score.py \
  --interior results/i1_n1000_slice --reference results/prod_N1000_G128 \
  --ref-lo 395 --slice-lo 473 --slice-hi 523 --gate-name "I1 ground-truth parity (50-band slice, production config)" \
  >> results/i1_n1000_slice.log 2>&1

echo "=== $(date) I1 chain done"
