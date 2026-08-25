#!/bin/bash
# Gate chain: waits for the post-production chain, then I2 v2 + I3/I5 scoring.
set +e
cd /home/francisco/Documents/Eigenfuntions

until grep -q "post-production chain done" results/exp/chain_post2.log 2>/dev/null; do
  sleep 300
done

echo "=== $(date) I2 v2 (Amendment A3): window + sub-gap, degree 24000"
conda run --no-capture-output -n lsu_ml python scripts/exp/exp_i2_v2.py \
  --rundir results/n10k_G192_window --degree 24000 --probes 12 --chunk 4 \
  --kpm results/exp/n10k_G256_dos_kpm.npz \
  --sub-lo 1.9063 --sub-hi 1.9606 \
  --gate-name "I2 completeness (N=10k window, v2 estimator)" --force \
  > results/exp/i2v2_n10k.log 2>&1

echo "=== $(date) I3 + I5 scoring on the merged window"
conda run --no-capture-output -n lsu_ml python scripts/exp/exp_i3_i5_score.py \
  --rundir results/n10k_G192_window \
  --kpm results/exp/n10k_G256_dos_kpm.npz \
  --gate-suffix "(N=10k 192^3)" --force \
  > results/exp/i3i5_n10k.log 2>&1

echo "=== $(date) I2 v2 calibration on the N=1000 I1 slice (known ground truth)"
conda run --no-capture-output -n lsu_ml python scripts/exp/exp_i2_v2.py \
  --rundir results/i1_n1000_slice --degree 16000 --probes 12 --chunk 8 \
  --gate-name "I2 calibration (N=1000 I1 slice, v2 estimator)" --force \
  > results/exp/i2v2_n1000.log 2>&1

echo "=== $(date) gate chain done"
