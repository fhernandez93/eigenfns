#!/bin/bash
# Literature-reproduction reference: ideal srs crystal at Sellers's SNG parameters
# (eps=13, cylinder radius r/a=0.2554 in conventional-cubic-cell units, published
# gap 28.06% between primitive-cell bands 2|3 = conventional-cell bands 4|5).
# MPB object geometry (its own Kottke smoothing), resolution 48/cell, k-path in the
# conventional cubic BZ + the folded bcc P point.
set -e
cd "$(dirname "$0")/../.."
mkdir -p results/exp
python3 - <<'EOF'
import sys
sys.path.insert(0, "/home/francisco/Documents/Create LSU Structures  - Claude")
import numpy as np
from tools import srs_crystal_rods
a = 1.0
rods = srs_crystal_rods(num_vertices=8, box=a, d0=a*np.sqrt(2)/4)
r = 0.2554
lines = []
for rod in rods:
    p1, p2 = rod[:3], rod[3:]
    c = (p1 + p2) / 2; v = p2 - p1
    L = np.linalg.norm(v); v = v / L
    lines.append(f"(make cylinder (center {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}) "
                 f"(radius {r:.6f}) (height {L:.6f}) (axis {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}) "
                 f"(material (make dielectric (epsilon 13))))")
geom = "\n    ".join(lines)
ctl = f"""(set! geometry-lattice (make lattice (size 1 1 1)))
(set! geometry (list
    {geom}))
(set! k-points (interpolate 6 (list (vector3 0 0 0) (vector3 0.5 0 0) (vector3 0.5 0.5 0)
    (vector3 0 0 0) (vector3 0.5 0.5 0.5) (vector3 0.5 0.5 0) (vector3 0.25 0.25 0.25))))
(set! resolution 48)
(set! num-bands 12)
(set! tolerance 1e-7)
(run)
"""
open("results/exp/srs_lit.ctl", "w").write(ctl)
print("ctl written")
EOF
(cd results/exp && conda run -n mbpEnv mpb srs_lit.ctl > srs_lit.out 2>&1)
grep -E "Gap from band" results/exp/srs_lit.out
grep -m1 "filling fraction" results/exp/srs_lit.out || true
echo done
