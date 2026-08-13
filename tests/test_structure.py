"""Rasterizer golden values: ff of the gold N=1000 reference at fixed settings."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from eigenfns.structure import box_size_for_n, load_rods, rasterize_penlike

GOLD = ("/home/francisco/Documents/Create LSU Structures  - Claude/"
        "Example/N1000_lsu_example_ends.txt")

pytestmark = pytest.mark.skipif(
    not Path(GOLD).exists(),
    reason="gold reference structure (parent repo) not present on this machine")


@pytest.fixture(scope="module")
def gold():
    return load_rods(GOLD)


def test_load_gold(gold):
    rods, n, box = gold
    assert rods.shape == (1653, 6)  # 1500 edges + 153 PBC duplicates
    assert n == 1000
    assert abs(box - 11.44) < 1e-9


def test_box_size_scaling():
    assert abs(box_size_for_n(8000) - 2 * 11.44) < 1e-9


def test_ff_montage_convention_64(gold):
    """Golden ff at 64^3 with montage parameters (b=0.2252, s=2.5).

    Provenance: 0.21733856 was produced by the parent notebook's literal
    `create_permittivity_grid_penlike` (20250903_create_h5_from_ends.ipynb),
    2026-08-13; our rasterizer matched it with ZERO differing voxels at 64^3.
    The 256^3 value (0.2172, matches the stated ~22%) is too slow for CI;
    64^3 is the pinned regression point.
    """
    rods, n, box = gold
    eps = rasterize_penlike(rods, 64, box)
    ff = float((eps != 1.0).mean())
    assert eps.min() == 1.0 and abs(eps.max() - 8.5703) < 1e-3
    assert abs(ff - 0.217339) < 1e-4, ff


def test_binary_values_only(gold):
    rods, n, box = gold
    eps = rasterize_penlike(rods, 32, box)
    u = np.unique(eps)
    assert len(u) == 2 and u[0] == 1.0 and abs(u[1] - 2.9275**2) < 1e-4
