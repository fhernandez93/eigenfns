#!/bin/bash
# Overnight chain 2026-08-17: hybrid polish continuation -> I4 bottom-up reference.
# Launched only after the current bake-off run exits (caller ensures GPU free).
set -e
cd /home/francisco/Documents/Eigenfuntions

echo "=== $(date) wrap continuation checkpoint"
conda run --no-capture-output -n lsu_ml python - <<'EOF'
import numpy as np, json
X = np.load('results/exp/bakeoff_hybrid_m80_subspace.npy')
prev = json.loads(open('results/exp/bakeoff_hybrid_m80.json').read())
np.savez_compressed('results/exp/bakeoff_hybrid_m80_cont.npz', X=X,
                    n_theta=prev['theta_applications'])
print('wrapped', X.shape, 'prev theta', prev['theta_applications'])
EOF

echo "=== $(date) continuation polish (3 outers, d=8000)"
conda run --no-capture-output -n lsu_ml python scripts/exp/exp_bakeoff.py \
  --method hybrid --m 80 --degree 3300 --polish-sweeps 3 --strip-degree 8000 \
  --resume-build results/exp/bakeoff_hybrid_m80_cont.npz --tag hybrid_m80c \
  > results/exp/bakeoff_hybrid_m80c.log 2>&1

echo "=== $(date) I4 bottom-up reference: N=1000 circular/2.9/ff22 @128^3 (~5.5h)"
conda run --no-capture-output -n lsu_ml python scripts/run_modes.py \
  "/home/francisco/Documents/Create LSU Structures  - Claude/Example/N1000_lsu_example_ends.txt" \
  --grid 128 --band-lo 398 --band-hi 607 --m 32 --guard 12 --theta-chunk 8 \
  --radius 0.331836 --aspect 1.0 --eps-rod 8.41 \
  --tag i4_n1000_circ_G128 --resume \
  > results/i4_n1000_circ_G128.log 2>&1

echo "=== $(date) overnight chain done"
