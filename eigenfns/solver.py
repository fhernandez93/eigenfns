"""Deflated block LOBPCG for the transverse Maxwell operator.

Numerically-hardened design (each point was a measured failure mode at fp32,
see plans/2026-08-12_orientation_and_experiments_log.md):

- all small dense algebra (Gram eigendecompositions, Rayleigh-Ritz) on host fp64;
- SVQB drop thresholds above the fp32 Gram noise floor (1e-5 relative);
- HX recomputed fresh every iteration (tracked products drift ~1e-3*||HW||);
- X re-deflated against the locked set every iteration (leaked components regrow);
- rank-dropped (zero) rows penalized on the RR diagonal, never left at fake λ=0;
- empty-P guard;
- blocks warm-started from the previous block's guard Ritz vectors.

Vectors are (m, 2, G, G, G) complex64 transverse spectral fields.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from .operator import MaxwellOperator

_SVQB_DROP = 1e-5   # relative Gram-eigenvalue drop threshold (fp32 noise floor)
_RR_DEAD_PENALTY = 1e4  # added to RR diagonal for dead directions (backward-stable in fp64)


def _flat(X):
    return X.reshape(X.shape[0], -1)


@jax.jit
def gram(A, B):
    """<A_i, B_j> Gram block in complex64 on device."""
    return _flat(A).conj() @ _flat(B).T


def combine(C, *blocks):
    """Linear combination out_i = sum_j C[j, i] S_j over the concatenated blocks."""
    m = blocks[0].shape[0]
    out = jnp.tensordot(C[:m].T, blocks[0], axes=(1, 0))
    off = m
    for Bk in blocks[1:]:
        out = out + jnp.tensordot(C[off:off + Bk.shape[0]].T, Bk, axes=(1, 0))
        off += Bk.shape[0]
    return out


def deflate(X, locked):
    """Project out the locked subspace (single c64 pass)."""
    if locked is None or locked.shape[0] == 0:
        return X
    C = gram(locked, X)
    return X - jnp.tensordot(C.T, locked, axes=(1, 0))


def ortho_block(X, passes=2, drop=_SVQB_DROP):
    """SVQB orthonormalization, fp64 Gram algebra on host, `passes` sweeps.

    Rank-dropped directions come back as zero rows (caller handles via the
    dead-row penalty in Rayleigh-Ritz)."""
    for _ in range(passes):
        G = np.asarray(gram(X, X)).astype(np.complex128)
        G = (G + G.conj().T) / 2
        w, V = np.linalg.eigh(G)
        keep = w > drop * max(w.max(), 1e-300)
        scale = np.where(keep, 1.0 / np.sqrt(np.where(keep, w, 1.0)), 0.0)
        C = jnp.asarray(V * scale[None, :], jnp.complex64)
        X = jnp.tensordot(C.T, X, axes=(1, 0))
    return X


@dataclass
class SolveStats:
    theta_applications: int = 0
    rounds: list = field(default_factory=list)
    wall_seconds: float = 0.0


def lobpcg_blocks(
    op: MaxwellOperator,
    nev: int,
    m: int = 96,
    guard: int = 48,
    tol: float = 1e-4,
    maxit: int = 400,
    seed: int = 0,
    precond_shift: str = "median",
    verbose: bool = True,
    log_every: int = 0,
):
    """Bottom-up deflated block LOBPCG. Returns (eigenvalues, locked_vectors, stats).

    Locks the lowest (m - guard) Ritz pairs of each block once their relative
    residuals pass `tol`; the guard Ritz vectors seed the next block.
    """
    key = jax.random.PRNGKey(seed)
    G3 = (2, op.grid_size, op.grid_size, op.grid_size)
    mask = (op.basis.kn > 0).astype(jnp.float32)[None, None]
    locked = None
    locked_vals = np.empty((0,))
    stats = SolveStats()
    t0 = time.perf_counter()
    carry = None  # guard Ritz vectors from the previous block

    def rand_block(k, n):
        ka, kb = jax.random.split(k)
        Z = (jax.random.normal(ka, (n,) + G3, jnp.float32)
             + 1j * jax.random.normal(kb, (n,) + G3, jnp.float32))
        Z = Z.astype(jnp.complex64) * mask
        # unit rows: mixed-norm blocks poison SVQB's relative drop threshold
        # (norm-360 random rows made it drop the unit-norm warm-start carries)
        ns = 1.0 / jnp.maximum(jnp.linalg.norm(_flat(Z), axis=1), 1e-30)
        return Z * ns[:, None, None, None, None]

    while locked_vals.size < nev:
        key, k2 = jax.random.split(key)
        if carry is not None and carry.shape[0] > 0:
            X = jnp.concatenate([carry, rand_block(k2, m - carry.shape[0])], axis=0)
        else:
            X = rand_block(k2, m)
        X = ortho_block(deflate(X, locked))
        HX = op.theta(X); stats.theta_applications += m
        P = HP = None
        n_lock = min(m - guard, nev - locked_vals.size)
        it = 0
        rn = np.full(m, np.inf)
        for it in range(maxit):
            lam = jnp.real(jnp.sum(_flat(X).conj() * _flat(HX), axis=1))
            R = HX - lam[:, None, None, None, None] * X
            rn = np.asarray(jnp.linalg.norm(_flat(R), axis=1)
                            / jnp.maximum(jnp.linalg.norm(_flat(HX), axis=1), 1e-30))
            lam_h = np.asarray(lam)
            # dead rows (rank-dropped: ~zero norm) report fake lam=0, rn=0 —
            # push them to the end so they are never counted converged/locked
            xnorm = np.asarray(jnp.linalg.norm(_flat(X), axis=1))
            dead_rows = xnorm < 0.5
            lam_h = np.where(dead_rows, np.inf, lam_h)
            rn = np.where(dead_rows, np.inf, rn)
            order = np.argsort(lam_h)
            if (rn[order[:n_lock]] < tol).all():
                break
            if log_every and it % log_every == 0:
                leak = 0.0
                if locked is not None:
                    leak = float(jnp.abs(gram(locked, X)).max())
                lo = np.sort(lam_h)
                print(f"    it {it:3d}  lam[0,{n_lock-1},{m-1}] = {lo[0]:.4f} {lo[n_lock-1]:.4f} "
                      f"{lo[-1]:.4f}  res q50/q90 {np.quantile(rn,0.5):.1e}/{np.quantile(rn,0.9):.1e}"
                      f"  leak {leak:.1e}", flush=True)
            W = op.precondition(R) * mask
            W = deflate(W, locked)
            for _ in range(2):
                Cxw = gram(X, W)
                W = W - jnp.tensordot(Cxw.T, X, axes=(1, 0))
            W = ortho_block(W)
            HW = op.theta(W); stats.theta_applications += m
            if P is not None:
                for _ in range(2):
                    Cxp = gram(X, P); Cwp = gram(W, P)
                    P = (P - jnp.tensordot(Cxp.T, X, axes=(1, 0))
                           - jnp.tensordot(Cwp.T, W, axes=(1, 0)))
                Gp = np.asarray(gram(P, P)).astype(np.complex128)
                Gp = (Gp + Gp.conj().T) / 2
                wp, Vp = np.linalg.eigh(Gp)
                keepp = wp > _SVQB_DROP * max(wp.max(), 1e-300)
                if not keepp.any():
                    P = HP = None
                else:
                    # FIXED-SHAPE whitening: dropped directions become zero
                    # rows (handled by the dead-row penalty) instead of
                    # shrinking P — a varying P shape forces XLA to recompile
                    # every downstream kernel every iteration (measured: GPU
                    # idle, 22 CPU cores of compiler churn, ~no progress).
                    scale = np.where(keepp, 1.0 / np.sqrt(np.where(keepp, wp, 1.0)), 0.0)
                    Tp = jnp.asarray(Vp * scale[None, :], jnp.complex64)
                    P = jnp.tensordot(Tp.T, P, axes=(1, 0))
                    HP = op.theta(P); stats.theta_applications += int(P.shape[0])
            blocks = [X, W] if P is None else [X, W, P]
            hblocks = [HX, HW] if P is None else [HX, HW, HP]
            A = np.asarray(
                jnp.concatenate([jnp.concatenate([gram(a, hb) for hb in hblocks], axis=1)
                                 for a in blocks], axis=0)).astype(np.complex128)
            A = (A + A.conj().T) / 2
            rownorm = np.concatenate(
                [np.asarray(jnp.linalg.norm(_flat(Bk), axis=1)) for Bk in blocks])
            dead = rownorm < 0.5
            # penalty must sit above anything the subspace can reach (S5)
            penalty = max(_RR_DEAD_PENALTY, 100.0 * float(np.abs(np.diag(A)).max()))
            A[np.diag_indices_from(A)] += np.where(dead, penalty, 0.0)
            wA, VA = np.linalg.eigh(A)
            C = jnp.asarray(VA[:, :m], jnp.complex64)
            Xn = combine(C, *blocks)
            Xn = deflate(Xn, locked)
            xs = 1.0 / jnp.maximum(jnp.linalg.norm(_flat(Xn), axis=1), 1e-20)
            Xn = Xn * xs[:, None, None, None, None]
            HXn = op.theta(Xn); stats.theta_applications += m
            Cp = jnp.asarray(np.asarray(C).copy(), jnp.complex64).at[:m].set(0)
            P = combine(Cp, *blocks)
            HP = combine(Cp, *hblocks)
            ps = 1.0 / jnp.maximum(jnp.linalg.norm(_flat(P), axis=1), 1e-20)
            P = P * ps[:, None, None, None, None]
            HP = HP * ps[:, None, None, None, None]
            X, HX = Xn, HXn
        # honest final values with fresh HX
        HX = op.theta(X); stats.theta_applications += m
        lam = np.asarray(jnp.real(jnp.sum(_flat(X).conj() * _flat(HX), axis=1)))
        Rf = HX - jnp.asarray(lam)[:, None, None, None, None] * X
        rn = np.asarray(jnp.linalg.norm(_flat(Rf), axis=1)
                        / jnp.maximum(jnp.linalg.norm(_flat(HX), axis=1), 1e-30))
        # dead rows report fake lam=0, rn=0 — exclude them here too (S1 fix:
        # a locked dead row would silently shift every subsequent band index)
        xnorm = np.asarray(jnp.linalg.norm(_flat(X), axis=1))
        lam = np.where(xnorm < 0.5, np.inf, lam)
        rn = np.where(xnorm < 0.5, np.inf, rn)
        order = np.argsort(lam)
        if not np.isfinite(lam[order[:n_lock]]).all():
            raise RuntimeError(
                f"block produced only {int(np.isfinite(lam).sum())} live Ritz "
                f"pairs but needs {n_lock} to lock — rank collapse")
        lock_idx = np.asarray(order[:n_lock])
        carry_idx = np.asarray(order[n_lock:])
        Xl = X[lock_idx]
        carry = X[carry_idx]
        locked = Xl if locked is None else jnp.concatenate([locked, Xl], axis=0)
        locked_vals = np.concatenate([locked_vals, lam[lock_idx]])
        stats.rounds.append({
            "iters": it + 1,
            "locked": int(locked_vals.size),
            "max_res_locked": float(rn[lock_idx].max()),
        })
        if verbose:
            print(f"  locked {locked_vals.size:4d}/{nev} (+{n_lock})  iters {it+1:3d}  "
                  f"lam [{lam[lock_idx][0]:.4f}..{lam[lock_idx][-1]:.4f}]  "
                  f"worst res {rn[lock_idx].max():.1e}  "
                  f"elapsed {time.perf_counter()-t0:6.1f}s", flush=True)
    stats.wall_seconds = time.perf_counter() - t0
    # Band-index integrity: locked_vals must be monotone across blocks (a
    # violation means a lower band was found late — a previously missed
    # eigenvalue). Sort values+vectors and surface the event loudly.
    if (np.diff(locked_vals) < -1e-6 * np.abs(locked_vals[1:])).any():
        worst = float(np.min(np.diff(locked_vals)))
        print(f"WARNING: locked eigenvalues non-monotone (worst step {worst:.3e}) — "
              f"a band was recovered out of order; re-sorting. Completeness gate "
              f"must confirm the final count.", flush=True)
    order = np.argsort(locked_vals, kind="stable")
    locked_vals = locked_vals[order]
    locked = locked[np.asarray(order)]
    return locked_vals[:nev], locked, stats
