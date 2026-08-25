#!/usr/bin/env python
"""Analyze KPM moments: counting function N(lam), DOS, gap location, and
(optionally) validation against a known exact spectrum.

    conda run -n lsu_ml python scripts/exp/exp_kpm_analyze.py \
        results/exp/<tag>_kpm.npz [--exact results/.../eigenvalues_all.npy] \
        [--window-count LO HI] [--gap-guess 1.9] [--plot out.png]

Counting function: N(b) = sum_k g_k mu_k c_k(x_b), c_k the Chebyshev
coefficients of the step 1_{x < x_b}, Jackson-damped. Error bars from the
per-probe spread (moments are saved per probe). All CPU/numpy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def jackson(p):
    k = np.arange(p + 1)
    g = ((p - k + 1) * np.cos(np.pi * k / (p + 1))
         + np.sin(np.pi * k / (p + 1)) / np.tan(np.pi / (p + 1)))
    return g / (p + 1)


def step_coeffs(xb, p):
    """Chebyshev coefficients of 1_{x < xb} on [-1,1], orders 0..p."""
    k = np.arange(1, p + 1)
    tb = np.arccos(np.clip(xb, -1.0, 1.0))
    c = np.empty(p + 1)
    c[0] = 1 - tb / np.pi
    c[1:] = -2 * np.sin(k * tb) / (k * np.pi)
    return c


def counting(mom, lam_max, lams, degree=None):
    """N(lam) per probe -> (mean, se) arrays over thresholds `lams`."""
    p = (mom.shape[1] - 1) if degree is None else degree
    g = jackson(p)
    xs = (2.0 * np.asarray(lams) - lam_max) / lam_max
    est = np.empty((mom.shape[0], len(xs)))
    for j, xb in enumerate(xs):
        c = step_coeffs(xb, p) * g
        est[:, j] = mom[:, :p + 1] @ c
    return est.mean(0), est.std(0, ddof=1) / np.sqrt(mom.shape[0]), est


def dos_curve(mom, lam_max, n_pts=4000, degree=None):
    """Jackson-damped DOS on a lambda grid (per-probe mean)."""
    p = (mom.shape[1] - 1) if degree is None else degree
    g = jackson(p)
    mu = mom.mean(0)[:p + 1] * g
    # Chebyshev-Gauss grid avoids the 1/sqrt(1-x^2) endpoint blowup
    theta = (np.arange(n_pts) + 0.5) * np.pi / n_pts
    x = np.cos(theta)
    T = np.cos(np.outer(np.arange(p + 1), theta))
    rho_x = (mu[0] + 2 * (mu[1:] @ T[1:])) / (np.pi * np.sin(theta))
    lam = (x + 1) * lam_max / 2
    rho_lam = rho_x * 2 / lam_max
    order = np.argsort(lam)
    return lam[order], rho_lam[order]


def smearing_width(lam, lam_max, degree):
    """Jackson kernel width in lambda units at position lam (pi/p in theta)."""
    x = np.clip(2 * lam / lam_max - 1, -1 + 1e-12, 1 - 1e-12)
    return np.pi / degree * np.sqrt(1 - x**2) * lam_max / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--exact", default=None)
    ap.add_argument("--window-count", nargs=2, type=float, default=None)
    ap.add_argument("--gap-guess", type=float, default=None)
    ap.add_argument("--plot", default=None)
    ap.add_argument("--degree", type=int, default=None,
                    help="truncate moments (resolution study)")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    mom, lam_max = z["moments"], float(z["lam_max"])
    meta = json.loads(str(z["meta"]))
    p = args.degree or (mom.shape[1] - 1)
    print(f"tag meta: N={meta['N']} grid={meta['grid']} ff={meta['ff']:.5f} "
          f"degree(moments)={mom.shape[1]-1} used={p} probes={mom.shape[0]} "
          f"lam_max={lam_max:.2f} ms/vec={meta['ms_per_vec']:.2f}")

    if args.exact:
        ev = np.load(args.exact)
        # thresholds: midpoints between consecutive exact eigenvalues over the
        # known range + a mid-gap point; compare KPM count vs exact count
        lo, hi = ev[0], ev[-1]
        ts = np.linspace(lo * 0.5, hi * 0.995, 60)
        mean, se, _ = counting(mom, lam_max, ts, p)
        exact = np.searchsorted(ev, ts)
        err = mean - exact
        sw = smearing_width(ts, lam_max, p)
        # local exact density to convert smearing width -> expected count blur
        dens = np.gradient(np.searchsorted(ev, ts).astype(float), ts)
        blur = dens * sw
        print("\n lam      exact  kpm      +-se    err     smear-blur(bands)")
        for i in range(len(ts)):
            print(f" {ts[i]:7.4f} {exact[i]:5d}  {mean[i]:8.2f} {se[i]:6.2f} "
                  f"{err[i]:+7.2f}  {blur[i]:6.2f}")
        chi = np.abs(err) / np.maximum(np.hypot(se, blur), 1e-9)
        print(f"\nmax |err| = {np.abs(err).max():.2f} bands; "
              f"max |err|/sqrt(se^2+blur^2) = {chi.max():.2f}")

    if args.window_count:
        a, b = args.window_count
        mean, se, est = counting(mom, lam_max, [a, b], p)
        diff = est[:, 1] - est[:, 0]
        print(f"\nwindow count in [{a}, {b}]: {diff.mean():.2f} "
              f"+- {diff.std(ddof=1)/np.sqrt(len(diff)):.2f}")

    if args.gap_guess:
        lam, rho = dos_curve(mom, lam_max, degree=p)
        m = (lam > args.gap_guess * 0.5) & (lam < args.gap_guess * 1.6)
        lam_m, rho_m = lam[m], rho[m]
        # normalize by median local dos, find the widest low-DOS stretch
        med = np.median(rho_m)
        low = rho_m < 0.10 * med
        best, cur, s0 = None, None, None
        for i, fl in enumerate(low):
            if fl and cur is None:
                cur, s0 = 0, i
            elif fl:
                cur += 1
            elif cur is not None:
                if best is None or cur > best[0]:
                    best = (cur, s0, i)
                cur = None
        if cur is not None and (best is None or cur > best[0]):
            best = (cur, s0, len(low))
        if best:
            gl, gh = lam_m[best[1]], lam_m[best[2] - 1]
            sw = smearing_width(np.array([gl, gh]), lam_max, p)
            print(f"\nDOS gap (rho < 10% of local median): [{gl:.4f}, {gh:.4f}] "
                  f"width {gh-gl:.4f}; Jackson smearing at edges "
                  f"{sw[0]:.4f}/{sw[1]:.4f}")
        else:
            print("\nno low-DOS stretch found near gap guess")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        lam, rho = dos_curve(mom, lam_max, degree=p)
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        axs[0].plot(lam, rho, lw=0.8)
        axs[0].set(xlabel=r"$\lambda\ (\mu m^{-2})$", ylabel="DOS (per unit lam)",
                   title="KPM DOS (full bandwidth)")
        axs[0].set_yscale("log")
        m = lam < (args.gap_guess or 2.5) * 2
        axs[1].plot(lam[m], rho[m], lw=1.0)
        axs[1].set(xlabel=r"$\lambda\ (\mu m^{-2})$", title="low-spectrum zoom")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=140)
        print(f"plot -> {args.plot}")


if __name__ == "__main__":
    main()
