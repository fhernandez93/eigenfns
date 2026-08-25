#!/bin/bash
# Detached GPU driver 2026-08-18: I4 resume -> I1 run -> I1 score.
# Run under setsid nohup; progress via results/*.log + this script's log.
set -e
cd /home/francisco/Documents/Eigenfuntions

echo "=== $(date) resuming I4 bottom-up (from block checkpoint)"
conda run --no-capture-output -n lsu_ml python scripts/run_modes.py \
  "/home/francisco/Documents/Create LSU Structures  - Claude/Example/N1000_lsu_example_ends.txt" \
  --grid 128 --band-lo 398 --band-hi 607 --m 32 --guard 12 --theta-chunk 8 \
  --radius 0.331836 --aspect 1.0 --eps-rod 8.41 \
  --tag i4_n1000_circ_G128 --resume \
  >> results/i4_n1000_circ_G128.log 2>&1

echo "=== $(date) I4 done; starting I1 interior run"
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
  --ref-lo 395 --slice-lo 473 --slice-hi 523 \
  --gate-name "I1 ground-truth parity (50-band slice, production config)" \
  >> results/i1_n1000_slice.log 2>&1

echo "=== $(date) GPU chain done"
