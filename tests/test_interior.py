"""Toy-scale tests for eigenfns.interior (CPU, small grids).

Run: JAX_PLATFORMS=cpu conda run -n lsu_ml python -m pytest tests/test_interior.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from eigenfns.operator import MaxwellOperator
from eigenfns.interior import (FoldedOperator, minres_batched, rr_extract,
                               bandpass_coeffs, _apply_cheb_poly)
from eigenfns.solver import _flat


G = 16


@pytest.fixture(scope="module")
def op():
    # a few random rods on a tiny grid — spectrum irrelevant, operator real
    rng = np.random.default_rng(0)
    eps = np.ones((G, G, G), np.float32)
    idx = rng.integers(0, G, (60, 3))
    eps[idx[:, 0], idx[:, 1], idx[:, 2]] = 8.41
    return MaxwellOperator(eps, 2.0)


def _rand_block(op, m, seed=1):
    key = jax.random.PRNGKey(seed)
    ka, kb = jax.random.split(key)
    mask = (op.basis.kn > 0).astype(jnp.float32)[None, None]
    Z = ((jax.random.normal(ka, (m, 2, G, G, G), jnp.float32)
          + 1j * jax.random.normal(kb, (m, 2, G, G, G), jnp.float32))
         .astype(jnp.complex64) * mask)
    return Z / jnp.linalg.norm(_flat(Z), axis=1)[:, None, None, None, None]


def test_folded_operator_identity(op):
    """(Θ−σ)²x computed by FoldedOperator equals the composition."""
    sigma = 1.3
    fop = FoldedOperator(op, sigma)
    X = _rand_block(op, 3)
    Y1 = fop.theta(X)
    Z = op.theta(X) - sigma * X
    Y2 = op.theta(Z) - sigma * Z
    err = float(jnp.linalg.norm(_flat(Y1 - Y2)) / jnp.linalg.norm(_flat(Y2)))
    assert err < 5e-6


def test_minres_solves_spd(op):
    """MINRES on Θ (σ=0, SPD) reaches its tolerance quickly."""
    B = _rand_block(op, 2)
    Y, it, _ = minres_batched(op, B, 0.0, tol=1e-5, maxit=200)
    R = op.theta(Y) - B
    rel = np.asarray(jnp.linalg.norm(_flat(R), axis=1))
    assert rel.max() < 1e-3
    assert it < 100


def test_minres_indefinite_reduces_residual(op):
    """Indefinite shift: residual decreases monotonically-ish vs maxit."""
    B = _rand_block(op, 2)
    sigma = 5.0
    rels = []
    for maxit in (10, 80):
        Y, _, _ = minres_batched(op, B, sigma, tol=1e-10, maxit=maxit)
        R = op.theta(Y) - sigma * Y - B
        rels.append(float(jnp.linalg.norm(_flat(R)) / jnp.linalg.norm(_flat(B))))
    assert rels[1] < rels[0]


def test_rr_extract_reproduces_invariant_subspace(op):
    """Feed rr_extract a span of true eigenvectors (from dense-ish LOBPCG on
    the toy op via power-iteration refinement) — residuals must be tiny."""
    # cheap reference: run inverse-free block power on exp(-tΘ)-like filter:
    # simplest is many Θ-orthogonal iterations — instead use lobpcg from solver
    from eigenfns.solver import lobpcg_blocks
    vals, vecs, _ = lobpcg_blocks(op, 6, m=10, guard=4, tol=1e-6, maxit=200,
                                  verbose=False)
    lam, Xr, rn = rr_extract(op, jnp.asarray(vecs[:6]))
    assert np.isfinite(lam[:6]).all()
    assert rn[:6].max() < 1e-4
    # eigenvalues agree with the solver's
    assert np.allclose(np.sort(lam[:6]), np.sort(vals[:6]), rtol=1e-4)


def test_bandpass_filter_selects_window(op):
    """The bandpass polynomial concentrates a random vector onto the window.

    Window sized to the filter's resolving power: at degree 800 the Jackson
    transition width near the toy spectrum's bottom is ~0.6 in λ, so the test
    window ([5, 15], containing the 8-fold low cluster at λ≈9.5) is ~15×
    wider than the transition — a fair separation test. Selecting individual
    bands inside a 0.04-wide cluster would need degree ~5×10⁴ (the production
    regime), not a toy test.
    """
    from eigenfns.solver import lobpcg_blocks
    vals, vecs, _ = lobpcg_blocks(op, 8, m=12, guard=4, tol=1e-6, maxit=200,
                                  verbose=False)
    from eigenfns.chebyshev import lanczos_lambda_max
    lam_max = 1.05 * lanczos_lambda_max(op)  # an underestimate makes T_k diverge
    assert float(vals[7]) < 15 < lam_max
    coef = bandpass_coeffs(5.0, 15.0, lam_max, 800)
    X = _rand_block(op, 2, seed=3)
    Y = _apply_cheb_poly(op, X, coef, lam_max)
    V = jnp.asarray(vecs)
    P2 = np.abs(np.asarray(jnp.matmul(_flat(V).conj(), _flat(Y).T,
                                      precision=jax.lax.Precision.HIGHEST))) ** 2
    P2_raw = np.abs(np.asarray(jnp.matmul(_flat(V).conj(), _flat(X).T,
                                          precision=jax.lax.Precision.HIGHEST))) ** 2
    # energy captured by the 8 lowest eigenvectors (the window holds more
    # eigenvalues than we computed, so capture < 1 is expected)
    assert P2.sum(0).min() > 100 * P2_raw.sum(0).max()
    # decisive: the Rayleigh quotient of the filtered vector lands in/near the
    # window, while the raw random vector sits at mid-spectrum scale
    def rq(Z):
        return np.asarray(jnp.real(jnp.sum(_flat(Z).conj() * _flat(op.theta(Z)),
                                           axis=1)))
    assert rq(Y).max() < 15.0 + 2.0      # window top + transition margin
    assert rq(X).min() > 100.0
