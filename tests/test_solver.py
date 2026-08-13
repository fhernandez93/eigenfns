"""Solver correctness against dense eigendecomposition (G=8, disordered ε)."""
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import jax.numpy as jnp
import pytest

from eigenfns.operator import MaxwellOperator
from eigenfns.solver import lobpcg_blocks


@pytest.fixture(scope="module")
def small_problem():
    rng = np.random.default_rng(3)
    G, L = 8, 1.0
    eps = np.asarray(1.0 + 7.57 * (rng.random((G, G, G)) > 0.7), np.float32)
    op = MaxwellOperator(eps, L)
    n = 2 * G**3
    M = np.zeros((n, n), np.complex64)
    I = np.eye(n, dtype=np.complex64)
    for i in range(n):
        H = jnp.asarray(I[:, i].reshape(1, 2, G, G, G))
        M[:, i] = np.asarray(op.theta(H)).ravel()
    w = np.linalg.eigvalsh((M + M.conj().T) / 2)
    return op, np.sort(w)[2:]  # drop the two zeroed Γ slots


def test_lobpcg_matches_dense(small_problem):
    op, w_true = small_problem
    nev = 48
    vals, vecs, stats = lobpcg_blocks(op, nev, m=24, guard=8, tol=1e-6,
                                      maxit=400, verbose=False)
    rel = np.abs(vals - w_true[:nev]) / w_true[:nev]
    assert rel.max() < 1e-4, rel.max()
    # monotone, no fake zeros, orthonormal
    assert (np.diff(vals) > -1e-8).all()
    assert vals[0] > 1e-3
    V = np.asarray(vecs).reshape(nev, -1)
    G = V.conj() @ V.T
    assert np.abs(G - np.eye(nev)).max() < 5e-4


def test_lobpcg_host_storage_matches_gpu_storage(small_problem):
    op, w_true = small_problem
    v1, _, _ = lobpcg_blocks(op, 24, m=16, guard=8, tol=1e-6, maxit=400,
                             verbose=False, locked_storage="gpu")
    v2, _, _ = lobpcg_blocks(op, 24, m=16, guard=8, tol=1e-6, maxit=400,
                             verbose=False, locked_storage="host")
    assert np.abs(v1 - v2).max() < 1e-6 * w_true[23]
