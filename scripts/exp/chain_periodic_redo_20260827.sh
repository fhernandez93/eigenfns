#!/bin/bash
# Queued 2026-08-27 on the user's decision, AFTER chain_post_20260826.sh
# (the two 256^3 I6 anchors). I6 keeps priority: it is the last unaddressed
# explanation for the in-gap states.
#
# WHY: I2 on the periodic re-solve FAILED -- sub-interval missed = +2.04 +-
# 0.42 against a gate of |missed| < 0.5. The solve converged 7 pairs in
# [1.855, 2.000] where ~12 exist. That incompleteness limits what the seam
# verdict's NEGATIVE half can claim: "the four seam states vanished" rested
# on their having no partner above overlap 0.5 among only those 7 states, and
# absence of a partner is not proof of absence when the solve demonstrably
# misses states. (The POSITIVE half -- six bulk states persisting at overlap
# 0.95-0.9997 with every shift negative -- is unaffected; incompleteness
# cannot manufacture a partner.)
#
# Specific target: the unconverged pair at lambda = 1.9095 sits 0.0201 below
# the seam state at 1.92960. Bulk states shifted -0.0007..-0.0034 and the
# band-edge state -0.0248, so a state living in the shell whose material
# actually changed would be expected to shift hard. 1.9095 is therefore a
# plausible heavily-shifted seam counterpart -- or it is not, and the seam
# verdict is clean. Either answer is worth 20 h.
#
# Changes vs the original run: m 30 -> 48 (the old subspace was too small for
# a ~12-state window), polish outers 5 -> 8, and --keep-checkpoint. The
# unconverged vectors are now saved by run_interior.py itself, so the vector
# that would have settled this cannot be lost again.
set +e
cd /home/francisco/Documents/Eigenfuntions
LOG=results/exp/chain_periodic_redo.log

until ! pgrep -f "chain_post_20260826.sh" >/dev/null; do sleep 300; done
clear=0
while [ $clear -lt 3 ]; do
  if pgrep -f "scripts/run_interior.py" >/dev/null || \
     pgrep -f "exp_i2_v2.py" >/dev/null; then clear=0; else clear=$((clear+1)); fi
  sleep 60
done

echo "=== $(date) periodic gap window re-solve, m=48, 8 polish outers" >> $LOG
conda run --no-capture-output -n lsu_ml python scripts/run_interior.py \
  Structures/20260701_N10000_lsu_generated.txt \
  --grid 192 --lam-lo 1.855 --lam-hi 2.000 --m 48 \
  --build-degree 4000 --build-outers 2 \
  --polish-degree 12000 --polish-outers 8 --tol 1e-4 \
  --radius 0.331836 --aspect 1.0 --eps-rod 8.41 --periodic \
  --chunk 4 --keep-checkpoint \
  --tag n10k_G192_gap_periodic_v2 > results/n10k_G192_gap_periodic_v2.log 2>&1

echo "=== $(date) re-running the overlap match against the new solve" >> $LOG
conda run --no-capture-output -n lsu_ml python scripts/exp/exp_periodic_match.py \
  --full >> $LOG 2>&1

echo "=== $(date) I2 completeness of the re-solve" >> $LOG
conda run --no-capture-output -n lsu_ml python scripts/exp/exp_i2_v2.py \
  --rundir results/n10k_G192_gap_periodic_v2 --degree 24000 --probes 12 --chunk 4 \
  --kpm results/exp/n10k_G256_dos_kpm.npz --sub-lo 1.9063 --sub-hi 1.9606 \
  --gate-name "I2 completeness (N=10k periodic v2, v2 estimator)" --force \
  > results/exp/i2v2_periodic_v2.log 2>&1
conda run --no-capture-output -n lsu_ml python scripts/exp/fix_i2_leakage.py \
  "I2 completeness (N=10k periodic v2, v2 estimator)" >> $LOG 2>&1

echo "=== $(date) periodic redo done" >> $LOG
