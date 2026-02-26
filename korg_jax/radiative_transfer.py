"""Radiative transfer formal solution.

Ported from Korg.jl/src/RadiativeTransfer/RadiativeTransfer.jl.
"""
from __future__ import annotations

import math
import numpy as np
import jax.numpy as jnp
from numpy.polynomial.legendre import leggauss

from .constants import c_cgs
from .atmosphere import ModelAtmosphere


# ── μ grid ──────────────────────────────────────────────────────────────────

def generate_mu_grid(mu_points):
    if isinstance(mu_points, (int, np.integer)):
        mu_grid, mu_weights = leggauss(int(mu_points))
        mu_grid = mu_grid / 2.0 + 0.5
        mu_weights = mu_weights / 2.0
        return mu_grid, mu_weights

    mu_grid = np.asarray(mu_points, dtype=float)
    if len(mu_grid) == 1:
        return mu_grid, np.array([1.0])

    if (not np.all(np.diff(mu_grid) >= 0)) or mu_grid[0] < 0 or mu_grid[-1] > 1:
        raise ValueError("mu_grid must be sorted and bounded between 0 and 1")

    d = np.diff(mu_grid)
    mu_weights = 0.5 * np.concatenate([[d[0]], d[:-1] + d[1:], [d[-1]]])
    return mu_grid, mu_weights


# ── Rays ────────────────────────────────────────────────────────────────────

def calculate_rays(mu_surface_grid, spatial_coord, spherical):
    rays = []
    spatial_coord = np.asarray(spatial_coord)

    if spherical:
        for mu in mu_surface_grid:
            b = spatial_coord[0] * math.sqrt(1 - mu ** 2)
            if b < spatial_coord[-1]:
                lowest = len(spatial_coord)
            else:
                lowest = int(np.argmin(np.abs(spatial_coord - b)))
                if spatial_coord[lowest] < b:
                    lowest -= 1
                lowest += 1
            s = np.sqrt(spatial_coord[:lowest] ** 2 - b ** 2)
            dsdr = spatial_coord[:lowest] / s
            rays.append((s, dsdr))
    else:
        for mu in mu_surface_grid:
            rays.append((spatial_coord / mu, np.ones_like(spatial_coord) / mu))

    return rays


# ── Tau schemes ─────────────────────────────────────────────────────────────

def compute_tau_anchored(alpha, integrand_factor, log_tau_ref):
    integrand = alpha * integrand_factor
    dlog = log_tau_ref[1:] - log_tau_ref[:-1]
    trap = 0.5 * (integrand[1:] + integrand[:-1]) * dlog
    tau = jnp.concatenate([jnp.zeros(1), jnp.cumsum(trap)])
    return tau


def compute_tau_bezier(s, alpha):
    # Not recommended; kept for parity
    tau = np.zeros_like(alpha)
    tau[0] = 1e-5
    C = fritsch_butland_C(s, alpha)
    C = np.clip(C, 0.5 * np.min(alpha), 2.0 * np.max(alpha))
    for i in range(1, len(alpha)):
        tau[i] = tau[i-1] + (s[i-1] - s[i]) / 3.0 * (alpha[i] + alpha[i-1] + C[i-1])
    return jnp.asarray(tau)


# ── Intensity schemes ───────────────────────────────────────────────────────

def compute_I_linear_flux_only(tau, S):
    if len(tau) == 1:
        return 0.0
    I = 0.0
    next_exp = jnp.exp(-tau[0])
    for i in range(len(tau) - 1):
        d = tau[i+1] - tau[i]
        d = d + (d == 0)
        m = (S[i+1] - S[i]) / d
        cur_exp = next_exp
        next_exp = jnp.exp(-tau[i+1])
        I = I + (-next_exp * (S[i+1] + m) + cur_exp * (S[i] + m))
    return I


def compute_I_linear(tau, S):
    I = np.zeros_like(S)
    if len(tau) <= 1:
        return jnp.asarray(I)
    for k in range(len(tau) - 2, -1, -1):
        d = float(tau[k+1] - tau[k])
        m = float(S[k+1] - S[k]) / d
        I[k] = (I[k+1] - float(S[k]) - m * (d + 1)) * np.exp(-d) + m + float(S[k])
    return jnp.asarray(I)


def compute_I_bezier(tau, S):
    I = np.zeros_like(S)
    I[-1] = 0.0
    if len(tau) <= 1:
        return jnp.asarray(I)
    C = fritsch_butland_C(tau, S)
    for k in range(len(tau) - 2, -1, -1):
        d = float(tau[k+1] - tau[k])
        ed = np.exp(-d)
        alpha = (2 + d ** 2 - 2 * d - 2 * ed) / d ** 2
        beta = (2 - (2 + 2 * d + d ** 2) * ed) / d ** 2
        gamma = (2 * d - 4 + (2 * d + 4) * ed) / d ** 2
        I[k] = I[k+1] * ed + alpha * float(S[k]) + beta * float(S[k+1]) + gamma * C[k]
    I[0] *= np.exp(-float(tau[0]))
    return jnp.asarray(I)


# ── Exponential integrals ───────────────────────────────────────────────────

def expint_transfer_integral_core(tau, m, b):
    return (1.0 / 6.0 * (tau * exponential_integral_2(tau) * (3 * b + 2 * m * tau)
                         - jnp.exp(-tau) * (3 * b + 2 * m * (tau + 1))))


def compute_F_flux_only_expint(tau, S):
    I = 0.0
    for i in range(len(tau) - 1):
        m = (S[i+1] - S[i]) / (tau[i+1] - tau[i])
        b = S[i] - m * tau[i]
        I = I + (expint_transfer_integral_core(tau[i+1], m, b) -
                 expint_transfer_integral_core(tau[i], m, b))
    return I


def exponential_integral_2(x):
    x = jnp.asarray(x)
    r = _expint_large(x)
    r = jnp.where(x < 9.0, _expint_8(x), r)
    r = jnp.where(x < 7.5, _expint_7(x), r)
    r = jnp.where(x < 6.5, _expint_6(x), r)
    r = jnp.where(x < 5.5, _expint_5(x), r)
    r = jnp.where(x < 4.5, _expint_4(x), r)
    r = jnp.where(x < 3.5, _expint_3(x), r)
    r = jnp.where(x < 2.5, _expint_2(x), r)
    r = jnp.where(x < 1.1, _expint_small(x), r)
    r = jnp.where(x == 0, 1.0, r)
    return r


def _expint_small(x):
    gamma = 0.5772156649015329
    return (1 + ((jnp.log(x) + gamma - 1)
                 + (-0.5 + (0.08333333333333333 + (-0.013888888888888888
                 + 0.0020833333333333333 * x) * x) * x) * x) * x)


def _expint_large(x):
    invx = 1.0 / x
    return jnp.exp(-x) * (1 + (-2 + (6 + (-24 + 120 * invx) * invx) * invx) * invx) * invx


def _expint_2(x):
    x = x - 2
    return (0.037534261820486914
            + (-0.04890051070806112
               + (0.033833820809153176
                  + (-0.016916910404576574
                     + (0.007048712668573576
                        - 0.0026785108140579598 * x) * x) * x) * x) * x)


def _expint_3(x):
    x = x - 3
    return (0.010641925085272673
            + (-0.013048381094197039
               + (0.008297844727977323
                  + (-0.003687930990212144
                     + (0.0013061422257001345
                        - 0.0003995258572729822 * x) * x) * x) * x) * x)


def _expint_4(x):
    x = x - 4
    return (0.0031982292493385146
            + (-0.0037793524098489054
               + (0.0022894548610917728
                  + (-0.0009539395254549051
                     + (0.00031003034577284415
                        - 8.466213288412284e-5 * x) * x) * x) * x) * x)


def _expint_5(x):
    x = x - 5
    return (0.000996469042708825
            + (-0.0011482955912753257
               + (0.0006737946999085467
                  + (-0.00026951787996341863
                     + (8.310134632205409e-5
                        - 2.1202073223788938e-5 * x) * x) * x) * x) * x)


def _expint_6(x):
    x = x - 6
    return (0.0003182574636904001
            + (-0.0003600824521626587
               + (0.00020656268138886323
                  + (-8.032993165122457e-5
                     + (2.390771775334065e-5
                        - 5.8334831318151185e-6 * x) * x) * x) * x) * x)


def _expint_7(x):
    x = x - 7
    return (0.00010350984428214624
            + (-0.00011548173161033826
               + (6.513442611103688e-5
                  + (-2.4813114708966427e-5
                     + (7.200234178941151e-6
                        - 1.7027366981408086e-6 * x) * x) * x) * x) * x)


def _expint_8(x):
    x = x - 8
    return (3.413764515111217e-5
            + (-3.76656228439249e-5
               + (2.096641424390699e-5
                  + (-7.862405341465122e-6
                     + (2.2386015208338193e-6
                        - 5.173353514609864e-7 * x) * x) * x) * x) * x)


# ── Bezier helper ───────────────────────────────────────────────────────────

def fritsch_butland_C(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    h = np.diff(x)
    alpha = 1.0 / 3.0 * (1 + h[1:] / (h[1:] + h[:-1]))
    d = (y[1:] - y[:-1]) / h
    yprime = (d[:-1] * d[1:]) / (alpha * d[1:] + (1 - alpha) * d[:-1])

    C0 = y[1:-1] + h[:-1] * yprime / 2.0
    C1 = y[1:-1] - h[1:] * yprime / 2.0

    return (np.concatenate([C0, [C1[-1]]]) + np.concatenate([[C0[0]], C1])) / 2.0


# ── Main API ────────────────────────────────────────────────────────────────

def radiative_transfer(atm: ModelAtmosphere, alpha, S, mu_points,
                        alpha_ref=None, tau_ref=None,
                        include_inward_rays=False,
                        tau_scheme="anchored", I_scheme="linear_flux_only"):
    if tau_ref is None:
        tau_ref = atm.tau_ref

    if alpha_ref is None:
        alpha_ref = np.ones_like(tau_ref)

    if I_scheme == "linear_flux_only" and tau_scheme == "anchored" and not atm.is_spherical:
        I_scheme = "linear_flux_only_expint"
        mu_surface_grid, mu_weights = np.array([1.0]), np.array([1.0])
    else:
        mu_surface_grid, mu_weights = generate_mu_grid(mu_points)

    spatial_coord = atm.z if not atm.is_spherical else (atm.R + atm.z)
    rays = calculate_rays(mu_surface_grid, spatial_coord, atm.is_spherical)

    inward_mu = -mu_surface_grid if include_inward_rays else -mu_surface_grid[
        [len(r[0]) < len(spatial_coord) for r in rays]
    ]
    n_inward = len(inward_mu)

    n_mu = n_inward + len(mu_surface_grid)
    n_layers, n_wl = alpha.shape

    if I_scheme.startswith("linear_flux_only"):
        I = np.zeros((n_mu, n_wl))
    else:
        I = np.zeros((n_mu, n_wl, n_layers))

    log_tau_ref = np.log(tau_ref)

    # inward rays
    # Julia passes -log_tau_ref (negated, NOT reversed), and tau_ref/alpha_ref as-is.
    # layer_inds handles the reversal when indexing into these arrays.
    for i in range(n_inward):
        path, dsdz = rays[i]
        path = path[::-1]
        dsdz = dsdz[::-1]
        layer_inds = np.arange(len(path))[::-1]
        _radiative_transfer_core(i, layer_inds, n_inward, -path, dsdz, -log_tau_ref,
                                 alpha, S, I, tau_ref, alpha_ref,
                                 tau_scheme, I_scheme)

    # outward rays
    for i in range(len(mu_surface_grid)):
        path, dsdz = rays[i]
        layer_inds = np.arange(len(path))
        _radiative_transfer_core(n_inward + i, layer_inds, n_inward, path, dsdz,
                                 log_tau_ref, alpha, S, I, tau_ref, alpha_ref,
                                 tau_scheme, I_scheme)

    surface_I = I[n_inward:, :] if I_scheme.startswith("linear_flux_only") else I[n_inward:, :, 0]
    F = 2 * math.pi * (surface_I.T @ (mu_weights * mu_surface_grid))

    # Photosphere correction for spherical atmospheres
    # Julia: photosphere_correction = radii[1]^2 / atm.R^2
    if atm.is_spherical:
        R_surface = spatial_coord[0]  # radius of outermost layer
        photosphere_correction = R_surface ** 2 / atm.R ** 2
        F = F * photosphere_correction

    return jnp.asarray(F), jnp.asarray(I), mu_surface_grid, mu_weights


def _radiative_transfer_core(mu_ind, layer_inds, n_inward, path, dsdz, log_tau_ref,
                             alpha, S, I, tau_ref, alpha_ref, tau_scheme, I_scheme):
    if len(path) == 1 and (I_scheme == "bezier" or tau_scheme == "bezier"):
        if I_scheme.startswith("linear_flux_only"):
            I[mu_ind, :] = 0.0
        else:
            I[mu_ind, :, 0] = 0.0
        return

    integrand_factor = tau_ref[layer_inds] / alpha_ref[layer_inds] * dsdz

    for wl in range(alpha.shape[1]):
        alpha_slice = alpha[layer_inds, wl]
        if tau_scheme == "anchored":
            tau = compute_tau_anchored(alpha_slice, integrand_factor, log_tau_ref[layer_inds])
        elif tau_scheme == "bezier":
            tau = compute_tau_bezier(path, alpha_slice)
        else:
            raise ValueError("tau_scheme must be 'anchored' or 'bezier'")

        S_slice = S[layer_inds, wl]
        if I_scheme == "linear":
            I[mu_ind, wl, layer_inds] = np.asarray(compute_I_linear(tau, S_slice))
        elif I_scheme == "linear_flux_only":
            I[mu_ind, wl] += float(compute_I_linear_flux_only(tau, S_slice))
        elif I_scheme == "linear_flux_only_expint":
            I[mu_ind, wl] += float(compute_F_flux_only_expint(tau, S_slice))
        elif I_scheme == "bezier":
            I[mu_ind, wl, layer_inds] = np.asarray(compute_I_bezier(tau, S_slice))
        else:
            raise ValueError("I_scheme must be 'linear', 'bezier', or 'linear_flux_only'")

        if mu_ind < n_inward:
            if I_scheme.startswith("linear_flux_only"):
                I[mu_ind + n_inward, wl] = I[mu_ind, wl] * np.exp(-float(tau[-1]))
            else:
                I[mu_ind + n_inward, wl, len(path) - 1] = I[mu_ind, wl, len(path) - 1]
