"""Chebyshev filtered subspace iteration (ChASE-style) + KPM eigenvalue counting.

Bottom-up variant: a degree-p Chebyshev polynomial mapped to [λ_c, λ_max]
amplifies everything below the cutoff λ_c and damps [λ_c, λ_max] to ≤ 1.
Filtering couples no vectors, so the basis tiles through GPU memory in chunks.

fp32 rules baked in (from the 2026-08-12 methods survey + our own failures):
- λ_max from Lanczos with certified upper bound θ_max + ‖r‖, ×1.05 margin —
  never underestimate (T_p explodes above the mapped interval);
- per-chunk normalization after each filter application (T_p reaches 1e4–1e6);
- Gram/RR dense algebra in fp64 on host;
- convergence judged on subspaces (residual per Ritz pair, locking), clusters
  accepted up to internal rotation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from .operator import MaxwellOperator
from .solver import _flat, gram, ortho_block, deflate, combine, deflate_chunk_rows


def lanczos_lambda_max(op: MaxwellOperator, iters: int = 24, seed: int = 0,
                       n_seeds: int = 2) -> float:
    """Probable upper bound on λ_max: max over seeds of θ_max + ‖r‖.

    NOT a certified bound (θ_max + β only brackets *some* eigenvalue); the
    caller must add a safety margin — an underestimate makes the Chebyshev
    filter and KPM diverge above the mapped interval. Multiple independent
    seeds reduce the unlucky-start risk."""
    if n_seeds > 1:
        return max(lanczos_lambda_max(op, iters, seed + 1000 * s, n_seeds=1)
                   for s in range(n_seeds))
    key = jax.random.PRNGKey(seed)
    G3 = (1, 2, op.grid_size, op.grid_size, op.grid_size)
    ka, kb = jax.random.split(key)
    mask = (op.basis.kn > 0).astype(jnp.float32)[None, None]
    v = ((jax.random.normal(ka, G3, jnp.float32)
          + 1j * jax.random.normal(kb, G3, jnp.float32)).astype(jnp.complex64) * mask)
    v = v / jnp.linalg.norm(_flat(v))
    alphas, betas = [], []
    v_prev = None
    beta = 0.0
    for _ in range(iters):
        w = op.theta(v)
        alpha = float(jnp.real(jnp.sum(_flat(v).conj() * _flat(w))))
        w = w - alpha * v - (beta * v_prev if v_prev is not None else 0.0)
        beta_new = float(jnp.linalg.norm(_flat(w)))
        alphas.append(alpha)
        betas.append(beta_new)
        if beta_new < 1e-20:
            break
        v_prev = v
        v = w / beta_new
        beta = beta_new
    T = np.diag(alphas) + np.diag(betas[:-1], 1) + np.diag(betas[:-1], -1)
    theta = np.linalg.eigvalsh(T)
    upper = theta[-1] + betas[-1]
    return float(upper)


def _cheb_filter_chunk(op, X, degree: int, lam_c: float, lam_max: float):
    """Apply the [λ_c, λ_max]-damping Chebyshev filter of given degree to a chunk.

    Standard ChebSI recurrence with the affine map t(λ) = (2λ − (λ_max+λ_c)) /
    (λ_max − λ_c); wanted eigenvalues (λ < λ_c) map below −1 where |T_p| grows
    exponentially. Chunk re-normalized after the recurrence.
    """
    e = (lam_max - lam_c) / 2.0
    c = (lam_max + lam_c) / 2.0
    # Y = (A X - c X)/e ; T_{k+1} = 2/e (A - c) T_k - T_{k-1}
    Y = (op.theta(X) - c * X) / e
    Xk_prev, Xk = X, Y
    for j in range(degree - 1):
        Xn = (op.theta(Xk) - c * Xk) * (2.0 / e) - Xk_prev
        Xk_prev, Xk = Xk, Xn
        # rescale periodically: T_p grows like cosh(p·arccosh(t0)) on wanted
        # components and overflows fp32 within a few hundred degrees
        if (j + 1) % 32 == 0:
            s = jnp.max(jnp.abs(_flat(Xk)), axis=1)
            sc = jnp.where(s > 1e30, 1.0 / jnp.maximum(s, 1e-30), 1.0)
            sc = jnp.where(s > 1e30, sc, 1.0)
            scale = sc[:, None, None, None, None]
            Xk = Xk * scale
            Xk_prev = Xk_prev * scale
    ns = 1.0 / jnp.maximum(jnp.linalg.norm(_flat(Xk), axis=1), 1e-30)
    return Xk * ns[:, None, None, None, None]


@dataclass
class ChebStats:
    theta_applications: int = 0
    outer: list = field(default_factory=list)
    lambda_max: float = 0.0
    wall_seconds: float = 0.0


def chebsi_bottom_up(
    op: MaxwellOperator,
    nev: int,
    guard: int = 48,
    degree: int = 200,
    tol: float = 1e-4,
    max_outer: int = 30,
    chunk: int = 64,
    seed: int = 0,
    lam_max: float | None = None,
    verbose: bool = True,
):
    """Compute the lowest `nev` eigenpairs via bottom-up ChebSI with locking.

    Basis of nev+guard vectors; each outer iteration filters the *unlocked*
    part chunk-by-chunk, orthonormalizes against locked + itself, Rayleigh-
    Ritzes, and locks converged leading Ritz pairs. Returns (vals, vecs, stats).
    """
    t0 = time.perf_counter()
    m = nev + guard
    key = jax.random.PRNGKey(seed)
    G3 = (2, op.grid_size, op.grid_size, op.grid_size)
    mask = (op.basis.kn > 0).astype(jnp.float32)[None, None]
    stats = ChebStats()

    if lam_max is None:
        lam_max = 1.05 * lanczos_lambda_max(op)
        stats.theta_applications += 24
    stats.lambda_max = lam_max

    ka, kb = jax.random.split(key)
    X = ((jax.random.normal(ka, (m,) + G3, jnp.float32)
          + 1j * jax.random.normal(kb, (m,) + G3, jnp.float32)).astype(jnp.complex64)
         * mask)
    X = ortho_block(X)

    n_locked = 0
    locked_vals = np.empty((0,))
    lam_c = None  # filter cutoff; set from current Ritz spectrum
    lam = None

    for outer in range(max_outer):
        # --- set cutoff: just above the current estimate of λ_{nev+guard/2}
        if lam is None:
            # cheap first estimate: Rayleigh quotients of the random block are
            # ~uniform over the spectrum; use λ_max/50 as a safe generous start
            lam_c = lam_max / 50.0
        else:
            hi = lam[min(m - 1, nev + guard // 2 - 1)]
            lam_c = float(min(hi * 1.1, lam_max / 4))
        # --- filter unlocked part, chunked
        act = X[n_locked:]
        outs = []
        for s in range(0, act.shape[0], chunk):
            piece = act[s:s + chunk]
            outs.append(_cheb_filter_chunk(op, piece, degree, lam_c, lam_max))
            stats.theta_applications += degree * piece.shape[0]
        act = jnp.concatenate(outs, axis=0)
        # --- orthonormalize: against locked, then internally
        if n_locked:
            act = deflate(act, X[:n_locked])
            act = deflate(act, X[:n_locked])
        act = ortho_block(act)
        # --- Rayleigh-Ritz on the active part
        Hact = []
        for s in range(0, act.shape[0], chunk):
            Hact.append(op.theta(act[s:s + chunk]))
            stats.theta_applications += act[s:s + chunk].shape[0]
        Hact = jnp.concatenate(Hact, axis=0)
        A = np.asarray(gram(act, Hact)).astype(np.complex128)
        A = (A + A.conj().T) / 2
        rown = np.asarray(jnp.linalg.norm(_flat(act), axis=1))
        dead = rown < 0.5
        A[np.diag_indices_from(A)] += np.where(dead, 1e4, 0.0)
        wA, VA = np.linalg.eigh(A)
        C = jnp.asarray(VA, jnp.complex64)
        act = combine(C, act)
        Hact = combine(C, Hact)
        lam_act = wA
        # --- residuals of the active Ritz pairs
        R = Hact - jnp.asarray(lam_act)[:, None, None, None, None] * act
        rn = np.asarray(jnp.linalg.norm(_flat(R), axis=1)
                        / np.maximum(np.asarray(jnp.linalg.norm(_flat(Hact), axis=1)), 1e-30))
        # --- lock leading converged prefix
        want = nev - n_locked
        conv_prefix = 0
        for i in range(min(want, len(rn))):
            if rn[i] < tol:
                conv_prefix += 1
            else:
                break
        X = jnp.concatenate([X[:n_locked], act], axis=0)
        lam_full = np.concatenate([locked_vals, lam_act])
        if conv_prefix:
            locked_vals = lam_full[:n_locked + conv_prefix]
            n_locked += conv_prefix
        lam = lam_full
        stats.outer.append({"outer": outer, "locked": n_locked,
                            "lam_c": lam_c, "median_res": float(np.median(rn))})
        if verbose:
            print(f"  outer {outer:2d}: locked {n_locked:4d}/{nev}  lam_c {lam_c:8.4f}  "
                  f"res q10/q50/q90 {np.quantile(rn,0.1):.1e}/{np.quantile(rn,0.5):.1e}/"
                  f"{np.quantile(rn,0.9):.1e}  elapsed {time.perf_counter()-t0:6.1f}s",
                  flush=True)
        if n_locked >= nev:
            break
    stats.wall_seconds = time.perf_counter() - t0
    return lam[:nev], X[:nev], stats


def kpm_count_below(op: MaxwellOperator, lam_b: float, lam_max: float,
                    degree: int = 500, n_probe: int = 30, seed: int = 1,
                    locked=None) -> tuple[float, float]:
    """Stochastic estimate of #eigenvalues below λ_b (Jackson-damped KPM step trace).

    Returns (estimate, standard_error). Counts over the transverse space minus
    the 2 zeroed Γ slots (excluded by the probe mask).

    **Completeness-gate mode**: pass `locked` (the converged eigenvectors with
    λ < λ_b). Probes are then deflated against them, so the estimator counts
    only *missed* eigenvalues below λ_b — expected 0 — and the stochastic
    variance collapses from ~√(2·count) to ~O(1), making ±1 certification
    feasible (plain-count mode is only a ±few-% sanity check; adversarial
    review 2026-08-12).
    """
    G = op.grid_size
    n_dim = 2 * G**3 - 2
    # Chebyshev coefficients of the step function 1_{λ<λ_b} on [0, lam_max]
    # mapped to x∈[-1,1]: step at x_b = (2λ_b - lam_max)/lam_max
    xb = (2 * lam_b - lam_max) / lam_max
    ks = np.arange(degree + 1)
    tb = np.arccos(np.clip(xb, -1, 1))
    mu = np.empty(degree + 1)
    mu[0] = 1 - tb / np.pi
    mu[1:] = -2 * np.sin(ks[1:] * tb) / (ks[1:] * np.pi)
    # Jackson damping
    g = ((degree - ks + 1) * np.cos(np.pi * ks / (degree + 1))
         + np.sin(np.pi * ks / (degree + 1)) / np.tan(np.pi / (degree + 1)))
    g /= (degree + 1)
    coef = jnp.asarray(mu * g, jnp.float32)

    key = jax.random.PRNGKey(seed)
    mask = (op.basis.kn > 0).astype(jnp.float32)[None, None]
    G3 = (n_probe, 2, G, G, G)
    kr, = jax.random.split(key, 1)
    Z = jnp.where(jax.random.bernoulli(kr, 0.5, G3), 1.0, -1.0).astype(jnp.complex64) * mask
    if locked is not None:
        cr = deflate_chunk_rows(G)
        Z = deflate(Z, locked, cr)
        Z = deflate(Z, locked, cr)
    # three-term recurrence on the mapped operator B = (2A - lam_max)/lam_max
    def Bx(V):
        return (2.0 / lam_max) * op.theta(V) - V
    T0, T1 = Z, Bx(Z)
    z_flat = _flat(Z)
    est = coef[0] * jnp.real(jnp.sum(z_flat.conj() * _flat(T0), axis=1))
    est = est + coef[1] * jnp.real(jnp.sum(z_flat.conj() * _flat(T1), axis=1))
    for k in range(2, degree + 1):
        T2 = 2.0 * Bx(T1) - T0
        est = est + coef[k] * jnp.real(jnp.sum(z_flat.conj() * _flat(T2), axis=1))
        T0, T1 = T1, T2
    per_probe = np.asarray(est)  # each ~ tr estimate (z† P z), E = count
    mean = float(per_probe.mean())
    se = float(per_probe.std(ddof=1) / np.sqrt(n_probe))
    return mean, se
