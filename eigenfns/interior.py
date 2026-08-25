"""Interior eigenpair machinery — Phase 1 bake-off scaffolding.

Three candidate subspace builders for a ~300-band window deep inside the
spectrum (window/lam_max ~ 4e-4 — the extreme low edge of the mapped
Chebyshev interval):

  (a) folded spectrum: LOBPCG on (Θ−σ)² (reuses `solver.lobpcg_blocks`);
  (b) bandpass Chebyshev filtered subspace iteration (Jackson step-difference
      filter accumulated during the KPM recurrence);
  (c) shift-invert subspace iteration: X ← (Θ−σ)⁻¹X by preconditioned
      batched MINRES (MPB transverse-projection preconditioner).

Every method funnels through `rr_extract`: orthonormalize, Rayleigh-Ritz on
the ORIGINAL Θ (fp64 host algebra), explicit per-pair relative residuals.
Ritz pairs failing the residual gate are never reported — ghosts are excluded
by construction; completeness (no misses) is checked against KPM counts and,
on N=1000, the exact production spectrum.

fp32 discipline inherited from solver.py: precision=HIGHEST Gram, fp64 host
RR, fixed-shape chunked operator application.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from .operator import MaxwellOperator
from .solver import _flat, gram, ortho_block, combine, apply_chunked, _HI


# ---------------------------------------------------------------- RR extract

def rr_extract(op: MaxwellOperator, X, theta_chunk: int = 8):
    """Orthonormalize X, Rayleigh-Ritz on Θ, return (lam, Xr, relres).

    Memory-lean: HX is never stored (chunk-assembled Gram, then a second
    chunked Θ pass for residuals after rotation — peak is 2 blocks, not 4;
    m=80 at 128³ was a measured OOM with the naive version). Costs 2m Θ-apps.
    All pairs are returned (sorted by λ); the caller applies the residual
    gate. Dead (rank-dropped) rows come back with λ=inf, res=inf.
    """
    if isinstance(X, list):
        X = X.pop()  # holder: caller's frame drops its ref (memory)
    m = int(X.shape[0])
    X = ortho_block_lowmem(X) if m >= 48 else ortho_block(X)
    A = np.zeros((m, m), np.complex128)
    for s in range(0, m, theta_chunk):
        Hc = op.theta(X[s:s + theta_chunk])
        A[:, s:s + Hc.shape[0]] = gram_chunked(X, Hc)
        del Hc
    A = (A + A.conj().T) / 2
    rown = np.asarray(jnp.linalg.norm(_flat(X), axis=1))
    dead = rown < 0.5
    A[np.diag_indices_from(A)] += np.where(dead, 1e8, 0.0)
    wA, VA = np.linalg.eigh(A)
    C = jnp.asarray(VA, jnp.complex64)
    Xr = combine(C, X)
    Xr.block_until_ready()
    del X
    lam = np.empty(m)
    rn = np.empty(m)
    for s in range(0, m, theta_chunk):
        Xc = Xr[s:s + theta_chunk]
        Hc = op.theta(Xc)
        # NORMALIZED Rayleigh quotient. SVQB leaves ‖x‖² off unity by ~3e-5
        # at 128³ and ~5e-5 at 192³ (blocked fp32 accumulation over millions
        # of positive terms under-estimates the Gram diagonal), and an
        # unnormalized ⟨x,Θx⟩ passes that error straight into λ: measured
        # 2.8e-5 relative bias vs exact ground truth, removed to 2.4e-7 by
        # this division (adversarial round 3, F1).
        xn2 = jnp.real(jnp.sum(jnp.abs(_flat(Xc)) ** 2, axis=1))
        lc = (jnp.real(jnp.sum(_flat(Xc).conj() * _flat(Hc), axis=1))
              / jnp.maximum(xn2, 1e-30))
        Rc = Hc - lc[:, None, None, None, None] * Xc
        lam[s:s + Xc.shape[0]] = np.asarray(lc)
        rn[s:s + Xc.shape[0]] = np.asarray(
            jnp.linalg.norm(_flat(Rc), axis=1)
            / jnp.maximum(jnp.linalg.norm(_flat(Hc), axis=1), 1e-30))
        del Hc, Rc
    xnorm = np.asarray(jnp.linalg.norm(_flat(Xr), axis=1))
    bad = xnorm < 0.5
    lam = np.where(bad, np.inf, lam)
    rn = np.where(bad, np.inf, rn)
    return lam, Xr, rn


# ----------------------------- host-resident basis (192³+: basis > VRAM) ----

def _stream_rotate(C: np.ndarray, Xh: np.ndarray, rows: int = 8) -> np.ndarray:
    """out[i] = Σ_j C[j, i] Xh[j] with Xh host-resident, GPU chunk transient
    ≤ 2×rows vectors. C: (m_in, m_out) complex."""
    m_in, m_out = C.shape
    out = np.empty((m_out,) + Xh.shape[1:], np.complex64)
    for so in range(0, m_out, rows):
        ro = min(rows, m_out - so)
        acc = jnp.zeros((ro,) + Xh.shape[1:], jnp.complex64)
        for si in range(0, m_in, rows):
            Xc = jnp.asarray(Xh[si:si + rows])
            Cb = jnp.asarray(C[si:si + rows, so:so + ro], jnp.complex64)
            acc = acc + jnp.tensordot(Cb.T, Xc, axes=(1, 0), precision=_HI)
            acc.block_until_ready()
            del Xc
        out[so:so + ro] = np.asarray(acc)
        del acc
    return out


def gram_hosted(Ah: np.ndarray, Bh: np.ndarray | None = None,
                rows: int = 8) -> np.ndarray:
    """Full fp64 Gram <A_i, B_j> with host-resident blocks (B defaults to A)."""
    if Bh is None:
        Bh = Ah
    ma, mb = Ah.shape[0], Bh.shape[0]
    G = np.zeros((ma, mb), np.complex128)
    for si in range(0, ma, rows):
        Ac = jnp.asarray(Ah[si:si + rows])
        for sj in range(0, mb, rows):
            Bc = jnp.asarray(Bh[sj:sj + rows])
            G[si:si + Ac.shape[0], sj:sj + Bc.shape[0]] = np.asarray(gram(Ac, Bc))
            del Bc
        del Ac
    return G


def svqb_hosted(Xh: np.ndarray, passes: int = 2, rows: int = 8) -> np.ndarray:
    from .solver import _SVQB_DROP
    for _ in range(passes):
        G = gram_hosted(Xh, rows=rows)
        G = (G + G.conj().T) / 2
        w, V = np.linalg.eigh(G)
        keep = w > _SVQB_DROP * max(w.max(), 1e-300)
        scale = np.where(keep, 1.0 / np.sqrt(np.where(keep, w, 1.0)), 0.0)
        Xh = _stream_rotate((V * scale[None, :]), Xh, rows)
    return Xh


def rr_extract_hosted(op: MaxwellOperator, Xh: np.ndarray,
                      theta_chunk: int = 8):
    """rr_extract for a host-resident basis. Returns (lam, Xh_rot, relres)."""
    m = int(Xh.shape[0])
    Xh = svqb_hosted(Xh, rows=theta_chunk)
    A = np.zeros((m, m), np.complex128)
    for s in range(0, m, theta_chunk):
        Hc = op.theta(jnp.asarray(Xh[s:s + theta_chunk]))
        for sj in range(0, m, theta_chunk):
            Bc = jnp.asarray(Xh[sj:sj + theta_chunk])
            A[sj:sj + Bc.shape[0], s:s + Hc.shape[0]] = np.asarray(gram(Bc, Hc))
            del Bc
        del Hc
    A = (A + A.conj().T) / 2
    rown = np.array([np.linalg.norm(Xh[i].ravel()) for i in range(m)])
    dead = rown < 0.5
    A[np.diag_indices_from(A)] += np.where(dead, 1e8, 0.0)
    wA, VA = np.linalg.eigh(A)
    Xh = _stream_rotate(VA, Xh, theta_chunk)
    lam = np.empty(m)
    rn = np.empty(m)
    for s in range(0, m, theta_chunk):
        Xc = jnp.asarray(Xh[s:s + theta_chunk])
        Hc = op.theta(Xc)
        # normalized Rayleigh quotient — see rr_extract (adversarial round 3, F1)
        xn2 = jnp.real(jnp.sum(jnp.abs(_flat(Xc)) ** 2, axis=1))
        lc = (jnp.real(jnp.sum(_flat(Xc).conj() * _flat(Hc), axis=1))
              / jnp.maximum(xn2, 1e-30))
        Rc = Hc - lc[:, None, None, None, None] * Xc
        lam[s:s + Xc.shape[0]] = np.asarray(lc)
        rn[s:s + Xc.shape[0]] = np.asarray(
            jnp.linalg.norm(_flat(Rc), axis=1)
            / jnp.maximum(jnp.linalg.norm(_flat(Hc), axis=1), 1e-30))
        del Xc, Hc, Rc
    bad = np.array([np.linalg.norm(Xh[i].ravel()) for i in range(m)]) < 0.5
    lam = np.where(bad, np.inf, lam)
    rn = np.where(bad, np.inf, rn)
    return lam, Xh, rn


def bandpass_subspace_hosted(op: MaxwellOperator, lam_lo: float, lam_hi: float,
                             lam_max: float, m: int = 104, degree: int = 3000,
                             max_outer: int = 2, res_tol: float = 1e-4,
                             chunk: int = 8, theta_chunk: int = 8,
                             seed: int = 0, X0h: np.ndarray | None = None,
                             verbose: bool = True):
    """Filtered SI with a HOST-resident basis (m × vecsize may exceed VRAM).

    Same algorithm as `bandpass_subspace`; every stage streams ≤ `chunk`
    vectors through the GPU. Required at 192³/m≈100 where the basis is
    ~12 GB (measured OOM with the device-resident path)."""
    key = jax.random.PRNGKey(seed)
    G3 = (2, op.grid_size, op.grid_size, op.grid_size)
    mask = (op.basis.kn > 0).astype(jnp.float32)[None, None]
    coef = bandpass_coeffs(lam_lo, lam_hi, lam_max, degree)
    if X0h is not None:
        Xh = X0h
        m = int(Xh.shape[0])
    else:
        Xh = np.empty((m,) + G3, np.complex64)
        for s in range(0, m, chunk):
            ka, kb = jax.random.split(jax.random.fold_in(key, s))
            c = min(chunk, m - s)
            Z = ((jax.random.normal(ka, (c,) + G3, jnp.float32)
                  + 1j * jax.random.normal(kb, (c,) + G3, jnp.float32))
                 .astype(jnp.complex64) * mask)
            ns = 1.0 / jnp.maximum(jnp.linalg.norm(_flat(Z), axis=1), 1e-30)
            Xh[s:s + c] = np.asarray(Z * ns[:, None, None, None, None])
            del Z
    n_theta = 0
    t0 = time.perf_counter()
    hist = []
    lam = rn = None
    for outer in range(max_outer):
        for s in range(0, m, chunk):
            piece = _apply_cheb_poly(op, jnp.asarray(Xh[s:s + chunk]), coef,
                                     lam_max)
            Xh[s:s + piece.shape[0]] = np.asarray(piece)
            n_theta += degree * int(piece.shape[0])
            del piece
        lam, Xh, rn = rr_extract_hosted(op, Xh, theta_chunk)
        n_theta += 2 * m
        inwin = (lam >= lam_lo) & (lam <= lam_hi) & np.isfinite(lam)
        conv = inwin & (rn < res_tol)
        hist.append({"outer": outer, "in_window": int(inwin.sum()),
                     "converged": int(conv.sum()),
                     "median_res_inwin": float(np.median(rn[inwin])) if inwin.any() else np.nan})
        if verbose:
            print(f"  bandpass-hosted outer {outer}: in-window {inwin.sum():3d} "
                  f"converged {conv.sum():3d} med-res "
                  f"{hist[-1]['median_res_inwin']:.1e} "
                  f"elapsed {time.perf_counter()-t0:.0f}s", flush=True)
        if inwin.any() and conv.sum() == inwin.sum():
            break
    return Xh, {"method": "bandpass_hosted", "degree": degree, "m": m,
                "lam_window": (lam_lo, lam_hi), "theta_applications": n_theta,
                "wall_seconds": time.perf_counter() - t0, "outer_stats": hist,
                "lam": lam, "res": rn}


# ------------------------------------------------------- (a) folded spectrum

class FoldedOperator:
    """(Θ−σ)² with a folded diagonal-kinetic preconditioner.

    Quacks like MaxwellOperator for `solver.lobpcg_blocks` (theta,
    precondition, basis, grid_size). One theta = 2 Θ applications.
    """

    def __init__(self, op: MaxwellOperator, sigma: float,
                 precond_reg: float | None = None):
        self.op = op
        self.sigma = float(sigma)
        self.basis = op.basis
        self.grid_size = op.grid_size
        self.box_size = op.box_size
        self.dtype = op.dtype
        # Wang-Zunger folded preconditioner alpha^2/((kin-σ)^2+alpha^2) with
        # kin = kn^2 <1/eps> the diagonal kinetic model; alpha of the order of
        # the target states' kinetic energy, i.e. alpha ~ σ (WZ 1994; tune ±1
        # decade per the survey). Overall scale is irrelevant to LOBPCG.
        if precond_reg is None:
            precond_reg = self.sigma ** 2
        self.precond_reg = float(precond_reg)
        self.precond_kind = "wz"   # "wz" diagonal | "psq" MPB projection twice
        kin = np.asarray(op.basis.kn) ** 2 * op.eps_mean_inv
        self._fold_diag = jnp.asarray(
            1.0 / ((kin - self.sigma) ** 2 + self.precond_reg), jnp.float32)
        self.n_theta = 0

    def theta(self, X):
        self.n_theta += 2 * int(X.shape[0])
        s = self.sigma
        Y = self.op.theta(X)
        return self.op.theta(Y) - (2.0 * s) * Y + (s * s) * X

    def precondition(self, R, target: float = 0.0):
        if self.precond_kind == "psq":
            # MPB transverse-projection preconditioner applied twice ≈ Θ⁻²:
            # matches (Θ−σ)⁻² in the high tail with full ε structure (2/3
            # matvec cost each); wrong near/below σ but SPD, so only speed —
            # never correctness — is at stake.
            return self.op.precondition(self.op.precondition(R))
        return R * self._fold_diag[None, None]


def folded_subspace(op: MaxwellOperator, sigma: float, nev: int,
                    m: int = 64, guard: int = 14, tol: float = 1e-3,
                    maxit: int = 400, theta_chunk: int = 8, seed: int = 0,
                    precond_reg: float | None = None, log_every: int = 0,
                    verbose: bool = True):
    """Method (a): LOBPCG on (Θ−σ)²; returns (subspace, stats dict).

    `tol` is on the FOLDED residual (fp32 floor: folded eigenvalues ~1e-2
    scale vs λ_max² ~ 2e7 — the practical floor is measured, not assumed).
    """
    from .solver import lobpcg_blocks
    fop = FoldedOperator(op, sigma, precond_reg=precond_reg)
    t0 = time.perf_counter()
    vals, vecs, st = lobpcg_blocks(
        fop, nev, m=m, guard=guard, tol=tol, maxit=maxit,
        theta_chunk=theta_chunk, seed=seed, verbose=verbose,
        log_every=log_every, locked_storage="host")
    wall = time.perf_counter() - t0
    return vecs, {"method": "folded", "sigma": sigma, "mu": np.asarray(vals),
                  "theta_applications": fop.n_theta, "wall_seconds": wall,
                  "outer_stats": st.rounds}


# --------------------------------------- (b) bandpass Chebyshev filtered SI

def _jackson(p: int) -> np.ndarray:
    k = np.arange(p + 1)
    g = ((p - k + 1) * np.cos(np.pi * k / (p + 1))
         + np.sin(np.pi * k / (p + 1)) / np.tan(np.pi / (p + 1)))
    return g / (p + 1)


def _step_coeffs(xb: float, p: int) -> np.ndarray:
    k = np.arange(1, p + 1)
    tb = np.arccos(np.clip(xb, -1.0, 1.0))
    c = np.empty(p + 1)
    c[0] = 1 - tb / np.pi
    c[1:] = -2 * np.sin(k * tb) / (k * np.pi)
    return c


def bandpass_coeffs(lam_lo: float, lam_hi: float, lam_max: float,
                    degree: int) -> np.ndarray:
    """Jackson-damped Chebyshev coefficients of 1_{lam_lo < λ < lam_hi}."""
    x_lo = 2 * lam_lo / lam_max - 1
    x_hi = 2 * lam_hi / lam_max - 1
    return _jackson(degree) * (_step_coeffs(x_hi, degree) - _step_coeffs(x_lo, degree))


def _apply_cheb_poly(op, X, coef: np.ndarray, lam_max: float):
    """Y = Σ_k coef_k T_k(B) X, B = (2Θ − λ_max)/λ_max. len(coef)-1 matvecs."""
    inv_l = 2.0 / lam_max
    c = jnp.asarray(coef, jnp.float32)

    def Bx(V):
        return inv_l * op.theta(V) - V

    T0 = X
    T1 = Bx(X)
    Y = c[0] * T0 + c[1] * T1
    for k in range(2, len(coef)):
        T2 = 2.0 * Bx(T1) - T0
        Y = Y + c[k] * T2
        T0, T1 = T1, T2
        if k % 512 == 0:
            Y.block_until_ready()  # async pileup guard
    ns = 1.0 / jnp.maximum(jnp.linalg.norm(_flat(Y), axis=1), 1e-30)
    return Y * ns[:, None, None, None, None]


def bandpass_subspace(op: MaxwellOperator, lam_lo: float, lam_hi: float,
                      lam_max: float, m: int = 64, degree: int = 2000,
                      max_outer: int = 8, res_tol: float = 1e-4,
                      target_count: int | None = None,
                      chunk: int = 16, theta_chunk: int = 8, seed: int = 0,
                      X0=None, verbose: bool = True):
    """Method (b): filtered subspace iteration with the bandpass polynomial.

    Each outer: filter (degree matvecs/vector, chunked), RR on Θ, count
    converged pairs inside [lam_lo, lam_hi]. Stops when the count of
    converged in-window pairs stops growing (or target_count reached).
    `X0`: warm-start subspace (e.g. a lower-degree build) — overrides m.
    """
    key = jax.random.PRNGKey(seed)
    G3 = (2, op.grid_size, op.grid_size, op.grid_size)
    mask = (op.basis.kn > 0).astype(jnp.float32)[None, None]
    if X0 is not None:
        if isinstance(X0, list):
            X0 = X0.pop()
        X = X0
        del X0
        m = int(X.shape[0])
    else:
        ka, kb = jax.random.split(key)
        X = ((jax.random.normal(ka, (m,) + G3, jnp.float32)
              + 1j * jax.random.normal(kb, (m,) + G3, jnp.float32)).astype(jnp.complex64)
             * mask)
    coef = bandpass_coeffs(lam_lo, lam_hi, lam_max, degree)
    n_theta = 0
    t0 = time.perf_counter()
    hist = []
    best = -1
    lam = Xr = rn = None
    for outer in range(max_outer):
        outs = []
        for s in range(0, X.shape[0], chunk):
            piece = _apply_cheb_poly(op, X[s:s + chunk], coef, lam_max)
            piece.block_until_ready()
            outs.append(piece)
            n_theta += degree * int(X[s:s + chunk].shape[0])
        # release the pre-filter block and the chunk list BEFORE RR — keeping
        # them alive was a measured 5.4 GB OOM at m=80, 128^3
        del X
        X = jnp.concatenate(outs, axis=0)
        X.block_until_ready()
        del outs, piece
        lam, Xr, rn = rr_extract(op, X, theta_chunk)
        del X
        n_theta += m + m  # HX in rr_extract counted approx (m), rotation reuse
        inwin = (lam >= lam_lo) & (lam <= lam_hi) & np.isfinite(lam)
        conv = inwin & (rn < res_tol)
        hist.append({"outer": outer, "in_window": int(inwin.sum()),
                     "converged": int(conv.sum()),
                     "median_res_inwin": float(np.median(rn[inwin])) if inwin.any() else np.nan})
        if verbose:
            print(f"  bandpass outer {outer}: in-window {inwin.sum():3d} "
                  f"converged {conv.sum():3d} med-res "
                  f"{hist[-1]['median_res_inwin']:.1e} "
                  f"elapsed {time.perf_counter()-t0:.0f}s", flush=True)
        if target_count is not None and conv.sum() >= target_count:
            break
        if int(conv.sum()) <= best and outer >= 2:
            break  # stagnated
        best = max(best, int(conv.sum()))
        X = Xr
        if outer < max_outer - 1:
            del Xr
    return Xr, {"method": "bandpass", "degree": degree, "m": m,
                "lam_window": (lam_lo, lam_hi), "theta_applications": n_theta,
                "wall_seconds": time.perf_counter() - t0, "outer_stats": hist,
                "lam": lam, "res": rn}


# ---------------------- (d) hybrid polish: preconditioned residual expansion

def gram_chunked(A, B, rows: int = 8):
    """<A_i,B_j> without materializing a full conjugated copy of A.

    `gram` conj()s its whole first argument (a full extra block — measured
    OOM contributor at m≈75, 128³); chunking keeps the transient at `rows`."""
    m = int(A.shape[0])
    out = np.zeros((m, int(B.shape[0])), np.complex128)
    for s in range(0, m, rows):
        out[s:s + min(rows, m - s)] = np.asarray(gram(A[s:s + rows], B))
    return out


def ortho_block_lowmem(X, passes: int = 2, rows: int = 8):
    """SVQB like solver.ortho_block but with chunk-assembled Gram (fp64 host).

    Peak = 2 blocks (X + rotated X) instead of 3+."""
    from .solver import _SVQB_DROP
    for _ in range(passes):
        G = gram_chunked(X, X, rows)
        G = (G + G.conj().T) / 2
        w, V = np.linalg.eigh(G)
        keep = w > _SVQB_DROP * max(w.max(), 1e-300)
        scale = np.where(keep, 1.0 / np.sqrt(np.where(keep, w, 1.0)), 0.0)
        C = jnp.asarray(V * scale[None, :], jnp.complex64)
        outs = []
        for s in range(0, C.shape[1], rows):
            piece = jnp.tensordot(C[:, s:s + rows].T, X, axes=(1, 0),
                                  precision=_HI)
            piece.block_until_ready()
            outs.append(piece)
        del X
        X = jnp.concatenate(outs, axis=0)
        X.block_until_ready()
        del outs, piece
    return X

def polish_subspace(op: MaxwellOperator, X, lam_lo: float, lam_hi: float,
                    lam_max: float, max_sweeps: int = 8, res_tol: float = 1e-4,
                    strip_degree: int = 300, strip_every: int = 2,
                    strip_w: bool = False,
                    theta_chunk: int = 8, verbose: bool = True):
    """Drive window Ritz pairs to res_tol by preconditioned-residual expansion.

    Each sweep: R = ΘX − λX for unconverged pairs, W = mask·P(R) (MPB
    transverse-projection preconditioner), RR on span[X, W] (2m basis, plain
    Ritz), keep the m pairs continuing the window. Plain interior Ritz is the
    classic ghost generator (checklist §8), so every `strip_every` sweeps the
    basis is re-filtered with a cheap degree-`strip_degree` bandpass to shear
    off accumulated out-of-window drift; final acceptance stays with the
    caller's rr_extract + residual gate + KPM count audit.

    Returns (X, lam, res, stats). `X` may be passed as a single-element list
    (holder) — it is emptied, so the caller's frame drops its reference and
    the pre-polish block can be freed (a caller-held ref was a measured OOM).
    """
    if isinstance(X, list):
        X = X.pop()
    m = int(X.shape[0])
    mask = (op.basis.kn > 0).astype(jnp.float32)[None, None]
    coef = bandpass_coeffs(lam_lo, lam_hi, lam_max, strip_degree)
    n_theta = 0
    t0 = time.perf_counter()
    hist = []
    holder = [X]
    del X
    lam, X, rn = rr_extract(op, holder, theta_chunk)
    n_theta += 2 * m
    for sweep in range(max_sweeps):
        conv = np.isfinite(lam) & (rn < res_tol) & (lam >= lam_lo) & (lam <= lam_hi)
        inwin = np.isfinite(lam) & (lam >= lam_lo) & (lam <= lam_hi)
        if verbose:
            print(f"  polish sweep {sweep}: in-window {int(inwin.sum()):3d} "
                  f"converged {int(conv.sum()):3d} med-res "
                  f"{float(np.median(rn[inwin])) if inwin.any() else np.nan:.1e} "
                  f"elapsed {time.perf_counter()-t0:.0f}s", flush=True)
        hist.append({"sweep": sweep, "in_window": int(inwin.sum()),
                     "converged": int(conv.sum()),
                     "med_res": float(np.median(rn[inwin])) if inwin.any() else None})
        if inwin.any() and conv.sum() == inwin.sum():
            break
        # W = strip(P(R)) for ALL pairs (fixed shape; converged rows cost
        # little extra and keep XLA shapes static). Dead rows have lam=inf
        # from rr_extract — 0·inf = NaN would poison the whole Gram; zero
        # them. The strip is NOT optional: P ≈ Θ⁻¹ boosts the lowest bands
        # ~10×, and plain interior RR then manufactures in-window ghost Ritz
        # values from that low-band content (measured: med-res 4.2e-2 → 0.9
        # in one sweep at 128³ with 472 bands below the window).
        Ws = []
        lam_j = jnp.asarray(np.where(np.isfinite(lam), lam, 0.0).astype(np.float32))
        for s in range(0, m, theta_chunk):
            Xc = X[s:s + theta_chunk]
            Rc = op.theta(Xc) - lam_j[s:s + theta_chunk, None, None, None, None] * Xc
            Wc = op.precondition(Rc) * mask
            n_theta += int(Xc.shape[0]) * 2
            if strip_w:
                Wc = _apply_cheb_poly(op, Wc, coef, lam_max)
                n_theta += int(Xc.shape[0]) * strip_degree
            Wc.block_until_ready()
            Ws.append(Wc)
            del Rc, Wc
        W = jnp.concatenate(Ws, axis=0)
        del Ws
        # orthonormalize W against X, then RR on [X, W] — all Grams AND the
        # projection update chunk-assembled (whole-block versions each peaked
        # at 4+ blocks — measured OOMs)
        for _ in range(2):
            C = jnp.asarray(gram_chunked(X, W), jnp.complex64)
            outs = []
            for s in range(0, m, 8):
                piece = W[s:s + 8] - jnp.tensordot(C[:, s:s + 8].T, X,
                                                   axes=(1, 0), precision=_HI)
                piece.block_until_ready()
                outs.append(piece)
            del W
            W = jnp.concatenate(outs, axis=0)
            W.block_until_ready()
            del outs, piece
        W = ortho_block_lowmem(W)
        nb = 2 * m
        A = np.zeros((nb, nb), np.complex128)
        blocks = [X, W]
        col = 0
        for Bsrc in blocks:
            for s in range(0, m, theta_chunk):
                piece = Bsrc[s:s + theta_chunk]
                Hc = op.theta(piece)
                n_theta += int(piece.shape[0])
                for bi, Bk in enumerate(blocks):
                    A[bi * m:(bi + 1) * m, col + s:col + s + piece.shape[0]] = (
                        gram_chunked(Bk, Hc))
                del Hc
            col += m
        A = (A + A.conj().T) / 2
        rownorm = np.concatenate([np.asarray(jnp.linalg.norm(_flat(Bk), axis=1))
                                  for Bk in blocks])
        dead = rownorm < 0.5
        A[np.diag_indices_from(A)] += np.where(dead, 1e8, 0.0)
        wA, VA = np.linalg.eigh(A)
        # CONTINUITY selection (adversarial lesson: window-score selection
        # kept junk in-window RR pairs assembled from W's low-band content
        # and diverged — med-res 8.4e-3 → 7.9e-2 over two sweeps). Each RR
        # pair's weight in the old X-space is ‖VA[:m, j]‖²; true refinements
        # have weight ≈ 1, ghost directions live in the W block. Keep
        # continuations first (weight ≥ 0.5) ranked by window score, fill any
        # remainder by window score alone.
        cen = 0.5 * (lam_lo + lam_hi)
        half = 0.5 * (lam_hi - lam_lo)
        score = np.maximum(np.abs(wA - cen) - half, 0.0)  # 0 inside window
        xw = (np.abs(VA[:m, :]) ** 2).sum(axis=0)
        score = score + np.where(xw >= 0.7, 0.0, 1e3)
        keep = np.argsort(score, kind="stable")[:m]
        keep = keep[np.argsort(wA[keep])]
        C = jnp.asarray(VA[:, keep], jnp.complex64)
        # rotation output built in 8-row slices: a whole-block combine peaks
        # at ~5 blocks (X, W, two tensordot temps, sum) — measured OOM; and
        # the blocks LIST keeps X/W alive even after `del X, W`
        outs = []
        for s in range(0, C.shape[1], 8):
            piece = combine(C[:, s:s + 8], X, W)
            piece.block_until_ready()
            outs.append(piece)
        del blocks, X, W
        X = jnp.concatenate(outs, axis=0)
        X.block_until_ready()
        del outs, piece
        if strip_every and (sweep + 1) % strip_every == 0:
            outs = []
            for s in range(0, m, 16):
                piece = _apply_cheb_poly(op, X[s:s + 16], coef, lam_max)
                piece.block_until_ready()
                outs.append(piece)
                n_theta += strip_degree * int(X[s:s + 16].shape[0])
            del X
            X = jnp.concatenate(outs, axis=0)
            del outs
        holder = [X]
        del X
        lam, X, rn = rr_extract(op, holder, theta_chunk)
        n_theta += 2 * m
    return X, lam, rn, {"method": "polish", "theta_applications": n_theta,
                        "wall_seconds": time.perf_counter() - t0,
                        "sweeps": hist}


# ------------------------------------- (c) shift-invert (batched PMINRES) SI

def minres_batched(op: MaxwellOperator, B, sigma: float, tol: float = 1e-3,
                   maxit: int = 200, use_precond: bool = True):
    """Solve (Θ−σ) Y = B per vector, preconditioned MINRES (ESW Alg. 2.4).

    Returns (Y, iters_used, n_theta). All per-vector scalars are batched.
    The preconditioner is MPB's transverse projection ≈ Θ⁻¹ (SPD).
    Stopping: |eta| (preconditioned residual norm) < tol * gamma1, per vector;
    iterates for the max over the batch (converged vectors keep polishing).
    """
    m = int(B.shape[0])
    n_theta = 0
    n_precond = 0

    def A(V):
        return op.theta(V) - sigma * V

    def M(V):
        nonlocal n_precond
        if not use_precond:
            return V
        n_precond += int(V.shape[0])  # 4 FFTs vs theta's 6 — real cost ~2/3 matvec
        return op.precondition(V)

    def dot(P, Q):  # per-vector Re<p,q>
        return jnp.real(jnp.sum(_flat(P).conj() * _flat(Q), axis=1))

    def bcast(a):
        return a[:, None, None, None, None].astype(jnp.float32)

    X = jnp.zeros_like(B)
    v_old = jnp.zeros_like(B)
    v = B
    z = M(v)
    gamma = jnp.sqrt(jnp.maximum(dot(z, v), 1e-38))
    gamma_old = jnp.ones_like(gamma)
    eta = gamma
    gamma1 = gamma
    s_old = jnp.zeros(m); s = jnp.zeros(m)
    c_old = jnp.ones(m); c = jnp.ones(m)
    w = jnp.zeros_like(B); w_old = jnp.zeros_like(B)
    it_used = 0
    for j in range(maxit):
        zt = z * bcast(1.0 / gamma)
        Az = A(zt); n_theta += m
        delta = dot(Az, zt)
        v_new = Az - v * bcast(delta / gamma) - v_old * bcast(gamma / gamma_old)
        z_new = M(v_new)
        gamma_new = jnp.sqrt(jnp.maximum(dot(z_new, v_new), 1e-38))
        a0 = c * delta - c_old * s * gamma
        a1 = jnp.sqrt(a0**2 + gamma_new**2)
        a2 = s * delta + c_old * c * gamma
        a3 = s_old * gamma
        c_new = a0 / a1
        s_new = gamma_new / a1
        w_new = (zt - w_old * bcast(a3) - w * bcast(a2)) * bcast(1.0 / a1)
        X = X + w_new * bcast(c_new * eta)
        eta = -s_new * eta
        v_old, v, z = v, v_new, z_new
        gamma_old, gamma = gamma, gamma_new
        s_old, s, c_old, c = s, s_new, c, c_new
        w_old, w = w, w_new
        it_used = j + 1
        if (j + 1) % 4 == 0:
            # the residual read doubles as a dispatch barrier: without it the
            # async queue keeps ~7 live blocks per in-flight iteration and
            # OOMs at chunk 16 (measured)
            r = np.asarray(jnp.abs(eta) / gamma1)
            if (r < tol).all():
                break
    return X, it_used, n_theta + (2 * n_precond) // 3


def shift_invert_subspace(op: MaxwellOperator, sigma: float, m: int = 64,
                          max_outer: int = 12, inner_tol: float = 1e-2,
                          inner_maxit: int = 200, res_tol: float = 1e-4,
                          lam_window: tuple[float, float] | None = None,
                          target_count: int | None = None,
                          chunk: int = 8, theta_chunk: int = 8, seed: int = 0,
                          verbose: bool = True):
    """Method (c): subspace iteration on (Θ−σ)⁻¹ via batched PMINRES.

    Each outer: Y = (Θ−σ)⁻¹X (chunked batched MINRES), orthonormalize, RR on
    Θ, residual bookkeeping. Krylov memory beyond X is deliberately NOT kept
    (minimal version); convergence per outer = |λ−σ|-ratio damping.
    """
    key = jax.random.PRNGKey(seed)
    G3 = (2, op.grid_size, op.grid_size, op.grid_size)
    mask = (op.basis.kn > 0).astype(jnp.float32)[None, None]
    ka, kb = jax.random.split(key)
    X = ((jax.random.normal(ka, (m,) + G3, jnp.float32)
          + 1j * jax.random.normal(kb, (m,) + G3, jnp.float32)).astype(jnp.complex64)
         * mask)
    X = ortho_block(X)
    n_theta = 0
    t0 = time.perf_counter()
    hist = []
    best = -1
    lam = Xr = rn = None
    for outer in range(max_outer):
        outs, iters = [], []
        for s in range(0, X.shape[0], chunk):
            Y, it, nt = minres_batched(op, X[s:s + chunk], sigma,
                                       tol=inner_tol, maxit=inner_maxit)
            Y.block_until_ready()
            outs.append(Y); iters.append(it); n_theta += nt
        del X, Y
        X = jnp.concatenate(outs, axis=0)
        X.block_until_ready()
        del outs
        lam, Xr, rn = rr_extract(op, X, theta_chunk)
        del X
        n_theta += 2 * m
        if lam_window:
            inwin = (lam >= lam_window[0]) & (lam <= lam_window[1]) & np.isfinite(lam)
        else:
            inwin = np.isfinite(lam)
        conv = inwin & (rn < res_tol)
        hist.append({"outer": outer, "inner_iters": iters,
                     "in_window": int(inwin.sum()), "converged": int(conv.sum()),
                     "median_res_inwin": float(np.median(rn[inwin])) if inwin.any() else np.nan})
        if verbose:
            print(f"  shift-invert outer {outer}: inner {iters} in-window "
                  f"{inwin.sum():3d} converged {conv.sum():3d} med-res "
                  f"{hist[-1]['median_res_inwin']:.1e} elapsed "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)
        if target_count is not None and conv.sum() >= target_count:
            break
        if int(conv.sum()) <= best and outer >= 3:
            break
        best = max(best, int(conv.sum()))
        X = Xr
        if outer < max_outer - 1:
            del Xr
    return Xr, {"method": "shift_invert", "sigma": sigma, "m": m,
                "theta_applications": n_theta,
                "wall_seconds": time.perf_counter() - t0, "outer_stats": hist,
                "lam": lam, "res": rn}
