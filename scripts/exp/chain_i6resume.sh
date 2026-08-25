#!/bin/bash
# Resume the 160^3 I6 run, killed 2026-08-25 06:01 by a GPU-wait race:
# chain_final's wait_gpu sampled `nvidia-smi` memory, which dipped below the
# 2 GB threshold while the 160^3 job was writing its 4.7 GB checkpoint, so the
# next job launched on top of it. Fix: wait on the PROCESS, not on GPU memory,
# and require the all-clear on consecutive samples.
#
# Sequenced after the final chain (deterministic; no lock needed).
set +e
cd /home/francisco/Documents/Eigenfuntions

until grep -q "final chain done" results/exp/chain_final.log 2>/dev/null; do
  sleep 300
done
# belt and braces: three consecutive clear samples of the process table
clear=0
while [ $clear -lt 3 ]; do
  if pgrep -f "scripts/run_interior.py" >/dev/null || \
     pgrep -f "exp_i2_v2.py" >/dev/null || \
     pgrep -f "exp_i3_i5_score.py" >/dev/null; then clear=0; else clear=$((clear+1)); fi
  sleep 60
done

echo "=== $(date) resuming I6 160^3 gap-edge run from its polish-outer-2 checkpoint"
conda run --no-capture-output -n lsu_ml python scripts/run_interior.py \
  Structures/20260701_N10000_lsu_generated.txt \
  --grid 160 --lam-lo 1.80 --lam-hi 2.06 --m 72 \
  --build-degree 3000 --build-outers 2 \
  --polish-degree 10000 --polish-outers 5 --tol 1e-4 \
  --radius 0.331836 --aspect 1.0 --eps-rod 8.41 \
  --chunk 8 --tag n10k_G160_gapedge --resume \
  >> results/n10k_G160_gapedge.log 2>&1

echo "=== $(date) I6 160^3 done"
