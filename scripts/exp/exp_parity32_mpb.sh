#!/bin/bash
# Rasterize the N=1000 gold structure at 32^3 (binary, montage convention),
# run MPB on it (150 bands, tol 1e-9), leaving results/exp/mpb32-epsilon.h5 + mpb32.out.
set -e
cd "$(dirname "$0")/../.."
mkdir -p results/exp
conda run --no-capture-output -n mpb_judge python - <<'EOF'
import sys; sys.path.insert(0, ".")
import numpy as np, h5py
from eigenfns.structure import load_rods, rasterize_penlike
rods, N, L = load_rods("/home/francisco/Documents/Create LSU Structures  - Claude/Example/N1000_lsu_example_ends.txt")
eps = rasterize_penlike(rods, 32, L).astype(np.float64)
with h5py.File("results/exp/eps32.h5", "w") as f:
    f.create_dataset("data", data=eps)
print("wrote results/exp/eps32.h5, ff =", (eps != 1).mean())
EOF
cat > results/exp/mpb32.ctl <<'CTL'
(set! geometry-lattice (make lattice (size 1 1 1)))
(set! epsilon-input-file "eps32.h5")
(set! k-points (list (vector3 0 0 0)))
(set! resolution 32)
(set! mesh-size 1)
(set! num-bands 150)
(set! tolerance 1e-9)
(run)
CTL
(cd results/exp && conda run -n mbpEnv mpb mpb32.ctl > mpb32.out 2>&1)
grep -c "^freqs:" results/exp/mpb32.out
echo done
