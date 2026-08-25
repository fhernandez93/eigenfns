#!/bin/bash
# Final chain, reprioritised 2026-08-24 after the boundary-seam finding.
# Order = scientific value: the periodic-rasterisation test (decides which
# in-gap states are real) -> completeness/residual gates -> I6 anchors ->
# cross-solver check -> estimator calibration.
set +e
cd /home/francisco/Documents/Eigenfuntions

wait_gpu () {
  while nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits \
        | awk '$1>2000{f=1} END{exit !f}'; do sleep 300; done
}

wait_gpu
echo "=== $(date) [1/6] PERIODIC-RASTERISATION TEST: gap window re-solve"
conda run --no-capture-output -n lsu_ml python scripts/run_interior.py \
  Structures/20260701_N10000_lsu_generated.txt \
  --grid 192 --lam-lo 1.855 --lam-hi 2.000 --m 30 \
  --build-degree 4000 --build-outers 2 \
  --polish-degree 12000 --polish-outers 5 --tol 1e-4 \
  --radius 0.331836 --aspect 1.0 --eps-rod 8.41 --periodic \
  --chunk 4 --tag n10k_G192_gap_periodic --resume \
  > results/n10k_G192_gap_periodic.log 2>&1

echo "=== $(date) [2/6] I2 v2 completeness (window + sub-gap)"
conda run --no-capture-output -n lsu_ml python scripts/exp/exp_i2_v2.py \
  --rundir results/n10k_G192_window --degree 24000 --probes 12 --chunk 4 \
  --kpm results/exp/n10k_G256_dos_kpm.npz --sub-lo 1.9063 --sub-hi 1.9606 \
  --gate-name "I2 completeness (N=10k window, v2 estimator)" --force \
  > results/exp/i2v2_n10k.log 2>&1

echo "=== $(date) [3/6] I3 + I5 scoring"
conda run --no-capture-output -n lsu_ml python scripts/exp/exp_i3_i5_score.py \
  --rundir results/n10k_G192_window --kpm results/exp/n10k_G256_dos_kpm.npz \
  --gate-suffix "(N=10k 192^3)" --force \
  > results/exp/i3i5_n10k.log 2>&1

echo "=== $(date) [4/6] I6 256^3 gap-edge anchors"
conda run --no-capture-output -n lsu_ml python scripts/run_interior.py \
  Structures/20260701_N10000_lsu_generated.txt \
  --grid 256 --lam-lo 1.84 --lam-hi 1.95 --m 18 \
  --build-degree 4000 --build-outers 2 --polish-degree 16000 --polish-outers 4 \
  --tol 1e-4 --radius 0.331836 --aspect 1.0 --eps-rod 8.41 --chunk 4 \
  --tag n10k_G256_edgelow --resume > results/n10k_G256_edgelow.log 2>&1
conda run --no-capture-output -n lsu_ml python scripts/run_interior.py \
  Structures/20260701_N10000_lsu_generated.txt \
  --grid 256 --lam-lo 1.99 --lam-hi 2.08 --m 18 \
  --build-degree 4000 --build-outers 2 --polish-degree 16000 --polish-outers 4 \
  --tol 1e-4 --radius 0.331836 --aspect 1.0 --eps-rod 8.41 --chunk 4 \
  --tag n10k_G256_edgehigh --resume > results/n10k_G256_edgehigh.log 2>&1

echo "=== $(date) [5/6] I4-interior cross-solve (N=1000 new decoration)"
conda run --no-capture-output -n lsu_ml python scripts/run_interior.py \
  "/home/francisco/Documents/Create LSU Structures  - Claude/Example/N1000_lsu_example_ends.txt" \
  --grid 128 --lam-lo 1.470 --lam-hi 1.835 --m 155 \
  --build-degree 2500 --build-outers 2 --polish-degree 8000 --polish-outers 6 \
  --tol 1e-4 --radius 0.331836 --aspect 1.0 --eps-rod 8.41 --chunk 8 \
  --tag i4int_n1000_below --resume > results/i4int_n1000_below.log 2>&1
conda run --no-capture-output -n lsu_ml python scripts/run_interior.py \
  "/home/francisco/Documents/Create LSU Structures  - Claude/Example/N1000_lsu_example_ends.txt" \
  --grid 128 --lam-lo 2.015 --lam-hi 2.55 --m 160 \
  --build-degree 2500 --build-outers 2 --polish-degree 8000 --polish-outers 6 \
  --tol 1e-4 --radius 0.331836 --aspect 1.0 --eps-rod 8.41 --chunk 8 \
  --tag i4int_n1000_above --resume > results/i4int_n1000_above.log 2>&1

echo "=== $(date) [6/6] I2 v2 calibration on the N=1000 I1 slice"
conda run --no-capture-output -n lsu_ml python scripts/exp/exp_i2_v2.py \
  --rundir results/i1_n1000_slice --degree 16000 --probes 12 --chunk 8 \
  --gate-name "I2 calibration (N=1000 I1 slice, v2 estimator)" --force \
  > results/exp/i2v2_n1000.log 2>&1

echo "=== $(date) final chain done"
