#!/bin/bash
# Queued 2026-08-26 on the user's decision. Runs AFTER everything currently
# scheduled: chain_final_20260824.sh (steps 5-6) and then chain_i6resume.sh
# (the 160^3 resume). Two items, cheap one first so a result lands sooner:
#
#   [1/3] I2 v2 on the PERIODIC re-solve      (~5 h)  -- certifies the
#         post-fix in-gap population, which is currently only a lower bound.
#         The montage window is certified (missed = -0.0002 +- 0.0001); the
#         periodic run's own population is not.
#   [2/3] I6 256^3 edgelow anchor             (~32 h)
#   [3/3] I6 256^3 edgehigh anchor            (~32 h)
#         Both re-run from scratch: the 2026-08-26 attempts OOM'd at 1h32m
#         each and neither wrote a checkpoint. ff85ee8 caps theta_chunk to 4
#         at 256^3, which is what they died on. Cost estimate: ~1.30M Theta
#         applications each at ~88 ms (2.5x the 192^3 FFT), so ~32 h apiece.
#
# GPU serialisation waits on the PROCESS TABLE, never on nvidia-smi memory --
# sampling free VRAM raced against a checkpoint write on 2026-08-25 and
# launched a second job on top of a running one, costing ~5 h of contention.
set +e
cd /home/francisco/Documents/Eigenfuntions
LOG=results/exp/chain_post_20260826.log

# Wait for both predecessors to exit, then require three consecutive clear
# samples of the process table before touching the GPU.
until ! pgrep -f "chain_final_20260824.sh" >/dev/null \
   && ! pgrep -f "chain_i6resume.sh" >/dev/null; do
  sleep 300
done
clear=0
while [ $clear -lt 3 ]; do
  if pgrep -f "scripts/run_interior.py" >/dev/null || \
     pgrep -f "exp_i2_v2.py" >/dev/null || \
     pgrep -f "exp_i3_i5_score.py" >/dev/null; then clear=0; else clear=$((clear+1)); fi
  sleep 60
done

echo "=== $(date) [1/3] I2 v2 completeness of the PERIODIC re-solve" >> $LOG
conda run --no-capture-output -n lsu_ml python scripts/exp/exp_i2_v2.py \
  --rundir results/n10k_G192_gap_periodic --degree 24000 --probes 12 --chunk 4 \
  --kpm results/exp/n10k_G256_dos_kpm.npz --sub-lo 1.9063 --sub-hi 1.9606 \
  --gate-name "I2 completeness (N=10k periodic gap window, v2 estimator)" --force \
  > results/exp/i2v2_periodic.log 2>&1
# leakage correction is applied post-hoc; exp_i2_v2.py's own numbers carry the
# three defects fixed in a513054 (disjoint mask, pointwise-|se|, deflated
# double-count) for the run that produced them.
conda run --no-capture-output -n lsu_ml python scripts/exp/fix_i2_leakage.py \
  "I2 completeness (N=10k periodic gap window, v2 estimator)" >> $LOG 2>&1

echo "=== $(date) [2/3] I6 256^3 edgelow anchor (retry, theta_chunk capped)" >> $LOG
conda run --no-capture-output -n lsu_ml python scripts/run_interior.py \
  Structures/20260701_N10000_lsu_generated.txt \
  --grid 256 --lam-lo 1.84 --lam-hi 1.95 --m 18 \
  --build-degree 4000 --build-outers 2 --polish-degree 16000 --polish-outers 4 \
  --tol 1e-4 --radius 0.331836 --aspect 1.0 --eps-rod 8.41 --chunk 4 \
  --tag n10k_G256_edgelow --resume > results/n10k_G256_edgelow.log 2>&1

echo "=== $(date) [3/3] I6 256^3 edgehigh anchor (retry)" >> $LOG
conda run --no-capture-output -n lsu_ml python scripts/run_interior.py \
  Structures/20260701_N10000_lsu_generated.txt \
  --grid 256 --lam-lo 1.99 --lam-hi 2.08 --m 18 \
  --build-degree 4000 --build-outers 2 --polish-degree 16000 --polish-outers 4 \
  --tol 1e-4 --radius 0.331836 --aspect 1.0 --eps-rod 8.41 --chunk 4 \
  --tag n10k_G256_edgehigh --resume > results/n10k_G256_edgehigh.log 2>&1

echo "=== $(date) post chain done" >> $LOG
