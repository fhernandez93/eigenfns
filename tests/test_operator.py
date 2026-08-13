"""Operator correctness: analytic homogeneous spectrum, hermiticity, positivity."""
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import jax.numpy as jnp
import pytest

from eigenfns.operator import MaxwellOperator


def dense_theta(op):
    G = op.grid_size
    n = 2 * G**3
    M = np.zeros((n, n), np.complex64)
    I = np.eye(n, dtype=np.complex64)
    for i in range(n):
        H = jnp.asarray(I[:, i].reshape(1, 2, G, G, G))
        M[:, i] = np.asarray(op.theta(H)).ravel()
    return M


def test_homogeneous_matches_analytic():
    G, L, eps0 = 8, 1.0, 4.0
    op = MaxwellOperator(np.full((G, G, G), eps0, np.float32), L)
    M = dense_theta(op)
    w = np.linalg.eigvalsh((M + M.conj().T) / 2)
    k = 2 * np.pi * np.fft.fftfreq(G, d=L / G)
    g2 = (np.array(np.meshgrid(k, k, k, indexing="ij")) ** 2).sum(0).ravel()
    ana = np.sort(np.concatenate([g2[g2 > 0] / eps0] * 2))
    assert (np.abs(w) < 1e-5).sum() == 2  # exactly the two zeroed G=0 slots
    num = np.sort(w)[2:]
    rel = np.abs(num - ana) / ana
    assert rel.max() < 5e-6


def test_hermitian_and_psd_on_disordered():
    rng = np.random.default_rng(0)
    G, L = 8, 1.0
    eps = np.asarray(1.0 + 7.57 * (rng.random((G, G, G)) > 0.7), np.float32)
    op = MaxwellOperator(eps, L)
    M = dense_theta(op)
    herm = np.abs(M - M.conj().T).max() / np.abs(M).max()
    assert herm < 1e-5
    w = np.linalg.eigvalsh((M + M.conj().T) / 2)
    assert w.min() > -1e-4 * abs(w).max()


def test_bloch_k_reduces_to_shifted_planewaves():
    G, L, eps0 = 6, 1.0, 2.25
    kf = (0.5, 0.0, 0.0)
    op = MaxwellOperator(np.full((G, G, G), eps0, np.float32), L, k_frac=kf)
    M = dense_theta(op)
    w = np.sort(np.linalg.eigvalsh((M + M.conj().T) / 2))
    k1 = 2 * np.pi * np.fft.fftfreq(G, d=L / G)
    k0 = 2 * np.pi * np.asarray(kf) / L
    KX, KY, KZ = np.meshgrid(k1 + k0[0], k1 + k0[1], k1 + k0[2], indexing="ij")
    g2 = (KX**2 + KY**2 + KZ**2).ravel()
    ana = np.sort(np.concatenate([g2 / eps0] * 2))
    rel = np.abs(w - ana) / np.maximum(ana, 1e-12)
    assert rel.max() < 5e-6
