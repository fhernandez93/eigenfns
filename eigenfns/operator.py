"""The Maxwell operator in the transverse H-field plane-wave formulation.

    Θ H = ∇ × (ε⁻¹ ∇ × H),   ∇·H = 0,   eigenvalues (ω/c)².

Representation (MPB's): each field is a pair of complex amplitudes on the two
transverse unit vectors (t̂₁, t̂₂) ⊥ (k+G) for every reciprocal vector G on the
grid. Transversality is exact by construction; the curl is diagonal in this
basis: with a right-handed (k̂, t̂₁, t̂₂) triad,  i(k+G)×(a t̂₁ + b t̂₂) =
i|k+G| (a t̂₂ − b t̂₁). One operator application costs 6 3-D FFTs:
spectral curl → Cartesian → IFFT → multiply ε⁻¹(r) → FFT → project curl back.

At k = Γ the G = 0 plane wave has no transverse pair; its two amplitudes are
identically zero (both basis vectors are zeroed), which removes the two ω = 0
Bloch modes exactly — the operator is positive definite on the remaining space.

Units: lengths in µm; eigenvalue λ = (ω/c)² in µm⁻²; ν = ω a /(2πc) = √λ·a/(2π)
for any chosen normalization length a.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class TransverseBasis:
    """Precomputed per-(k+G) transverse frame and |k+G| for one k-point."""

    grid_size: int
    box_size: float
    k_frac: tuple[float, float, float]  # Bloch k in units of 2π/L
    kn: jax.Array          # (G,G,G) float32 — |k+G|
    t: jax.Array           # (2,3,G,G,G) float32 — t̂₁, t̂₂ (zeroed where |k+G|=0)

    @property
    def n_dof(self) -> int:
        return 2 * self.grid_size**3


def make_basis(grid_size: int, box_size: float, k_frac=(0.0, 0.0, 0.0)) -> TransverseBasis:
    G, L = int(grid_size), float(box_size)
    g1 = 2 * np.pi * np.fft.fftfreq(G, d=L / G)  # reciprocal grid, rad/µm
    k0 = 2 * np.pi * np.asarray(k_frac, np.float64) / L
    KX, KY, KZ = np.meshgrid(g1 + k0[0], g1 + k0[1], g1 + k0[2], indexing="ij")
    K = np.stack([KX, KY, KZ])
    kn = np.sqrt((K**2).sum(0))

    # Reference axis per k: ẑ, switched to x̂ where k+G is nearly parallel to ẑ,
    # so the cross product never degenerates.
    ref = np.zeros_like(K)
    ref[2] = 1.0
    swap = np.abs(K[2]) > 0.9 * np.maximum(kn, 1e-300)
    ref[2][swap] = 0.0
    ref[0][swap] = 1.0

    t1 = np.cross(K, ref, axis=0)
    n1 = np.linalg.norm(t1, axis=0)
    sing = n1 == 0  # only where k+G = 0
    n1[sing] = 1.0
    t1 /= n1
    t2 = np.cross(K, t1, axis=0)
    n2 = np.linalg.norm(t2, axis=0)
    n2[n2 == 0] = 1.0
    t2 /= n2
    t1[:, sing] = 0.0
    t2[:, sing] = 0.0

    return TransverseBasis(
        grid_size=G,
        box_size=L,
        k_frac=tuple(float(x) for x in k_frac),
        kn=jnp.asarray(kn, jnp.float32),
        t=jnp.asarray(np.stack([t1, t2]), jnp.float32),
    )


@partial(jax.jit, static_argnames=())
def _theta_block(Hs, kn, t, inv_eps):
    """Θ applied to a block. Hs: (m, 2, G, G, G) complex spectral amplitudes."""
    a, b = Hs[:, 0], Hs[:, 1]
    # curl H in the transverse frame: i|k+G|(a t̂₂ − b t̂₁)
    c1 = -1j * kn * b
    c2 = 1j * kn * a
    Ec = c1[:, None] * t[0][None] + c2[:, None] * t[1][None]   # Cartesian spectral
    E = jnp.fft.ifftn(Ec, axes=(2, 3, 4))
    D = inv_eps * E
    Dc = jnp.fft.fftn(D, axes=(2, 3, 4))
    f1 = (t[0][None] * Dc).sum(1)
    f2 = (t[1][None] * Dc).sum(1)
    return jnp.stack([-1j * kn * f2, 1j * kn * f1], axis=1)


@partial(jax.jit, static_argnames=())
def _precondition_block(Rs, kn, scale):
    """MPB-style kinetic preconditioner: divide by (|k+G|² + s) in the transverse basis."""
    return Rs / (kn**2 + scale)


class MaxwellOperator:
    """Matrix-free Θ for a fixed ε(r) grid and k-point, batched over vectors."""

    def __init__(self, eps: np.ndarray, box_size: float, k_frac=(0.0, 0.0, 0.0)):
        eps = np.asarray(eps)
        if eps.ndim != 3 or len(set(eps.shape)) != 1:
            raise ValueError(f"eps must be a cube, got {eps.shape}")
        self.grid_size = eps.shape[0]
        self.box_size = float(box_size)
        self.basis = make_basis(self.grid_size, self.box_size, k_frac)
        self.inv_eps = jnp.asarray(1.0 / eps, jnp.float32)
        self.eps_mean_inv = float(np.mean(1.0 / eps))

    def theta(self, Hs: jax.Array) -> jax.Array:
        """Θ Hs for a block (m, 2, G, G, G) of spectral transverse fields."""
        return _theta_block(Hs, self.basis.kn, self.basis.t, self.inv_eps)

    def precondition(self, Rs: jax.Array, target: float = 0.0) -> jax.Array:
        """Approximate (Θ − target)⁻¹ R via the diagonal kinetic term."""
        scale = jnp.float32(max(target / self.eps_mean_inv, 1e-6) * self.eps_mean_inv + 1e-6)
        return _precondition_block(Rs, self.basis.kn, scale)

    # ---- conversions -------------------------------------------------------
    def to_cartesian_spectral(self, Hs: jax.Array) -> jax.Array:
        """(m,2,G,G,G) transverse → (m,3,G,G,G) Cartesian spectral H(k+G)."""
        t = self.basis.t
        return Hs[:, 0:1] * t[0][None] + Hs[:, 1:2] * t[1][None]

    def h_realspace(self, Hs: jax.Array) -> jax.Array:
        """Real-space Cartesian H(r) for a block."""
        return jnp.fft.ifftn(self.to_cartesian_spectral(Hs), axes=(2, 3, 4))

    def e_realspace(self, Hs: jax.Array, eigvals: jax.Array) -> jax.Array:
        """Real-space E(r) ∝ ε⁻¹ ∇×H / λ^{1/2} for a block (unnormalized)."""
        kn, t = self.basis.kn, self.basis.t
        a, b = Hs[:, 0], Hs[:, 1]
        Ec = (-1j * kn * b)[:, None] * t[0][None] + (1j * kn * a)[:, None] * t[1][None]
        E = self.inv_eps * jnp.fft.ifftn(Ec, axes=(2, 3, 4))
        return E / jnp.sqrt(eigvals)[:, None, None, None, None]
