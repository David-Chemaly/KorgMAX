"""Hydrogen line profiles (Stark broadening for Lyman/Balmer/Paschen/Brackett).

Ported from Korg.jl/src/hydrogen_line_absorption.jl.
"""
from __future__ import annotations

import math
import os
import numpy as np
import jax.numpy as jnp

from .constants import (
    c_cgs, hplanck_cgs, hplanck_eV, kboltz_cgs, kboltz_eV,
    RydbergH_eV, electron_mass_cgs, electron_charge_cgs, bohr_radius_cgs,
    amu_cgs,
)
from .voigt import voigt_hjerting
from .statmech import hummer_mihalas_w


# ── Helpers ────────────────────────────────────────────────────────────────

def _zero2epsilon(x):
    return x + (x == 0) * np.finfo(float).tiny


def normal_pdf(delta, sigma):
    return np.exp(-0.5 * (delta / sigma) ** 2) / (math.sqrt(2.0 * math.pi) * sigma)


def doppler_width(lambda0, T, mass, xi):
    return lambda0 * np.sqrt(kboltz_cgs * T / mass + (xi ** 2) / 2.0) / c_cgs


def scaled_vdW(vdW, mass, T):
    # vdW = (sigma, alpha) or (gamma, -1)
    sigma, alpha = vdW
    if alpha == -1:
        return sigma * (T / 10000.0) ** 0.3

    v0 = 1e6  # cm/s
    inv_mu = 1.0 / (1.008 * amu_cgs) + 1.0 / mass
    vbar = np.sqrt(8.0 * kboltz_cgs * T / math.pi * inv_mu)
    return 2.0 * (4.0 / math.pi) ** (alpha / 2.0) * math.gamma((4.0 - alpha) / 2.0) \
        * v0 * sigma * (vbar / v0) ** (1.0 - alpha)


def sigma_line(lambda0):
    return (math.pi * electron_charge_cgs ** 2 / (electron_mass_cgs * c_cgs)) \
        * (lambda0 ** 2 / c_cgs)


def line_profile(lambda0, sigma, gamma, amplitude, lambdas):
    inv = 1.0 / (sigma * math.sqrt(2.0))
    scaling = inv / math.sqrt(math.pi) * amplitude
    v = np.abs(lambdas - lambda0) * inv
    return np.asarray(voigt_hjerting(gamma * inv, v)) * scaling


def exponential_integral_1(x):
    if x < 0.0:
        return 0.0
    if x <= 0.01:
        return -math.log(x) - 0.577215 + x
    if x <= 1.0:
        return (-math.log(x) - 0.57721566 +
                x * (0.99999193 + x * (-0.24991055 + x * (0.05519968 +
                x * (-0.00976004 + x * 0.00107857)))))
    if x <= 30.0:
        return ((x * (x + 2.334733) + 0.25062) /
                (x * (x + 3.330657) + 1.681534) / x * math.exp(-x))
    return 0.0


# ── Stark profiles from Stehle & Hutcheon (1999) ────────────────────────────


def _load_stark_profiles(fname):
    import h5py

    profiles = []
    with h5py.File(fname, "r") as f:
        for name in f.keys():
            g = f[name]
            temps = g["temps"][()]
            nes = g["electron_number_densities"][()]
            delta = g["delta_nu_over_F0"][()]
            prof = g["profile"][()]

            logP = np.log(prof)
            logP[~np.isfinite(logP)] = -700.0

            log_delta = np.empty_like(delta)
            log_delta[0] = -np.finfo(float).max
            log_delta[1:] = np.log(delta[1:])

            # Reorder to (temp, ne, delta)
            logP = np.transpose(logP, (2, 1, 0))

            lambda0 = g["lambda0"][()] * 1e-8
            if lambda0.shape == (len(nes), len(temps)):
                lambda0 = lambda0.T

            profiles.append({
                "temps": temps,
                "nes": nes,
                "log_delta": log_delta,
                "logP": logP,
                "lambda0": lambda0,
                "lower": int(g.attrs["lower"]),
                "upper": int(g.attrs["upper"]),
                "Kalpha": float(g.attrs["Kalpha"]),
                "log_gf": float(g.attrs["log_gf"]),
            })
    return profiles


try:
    _data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _hline_stark_profiles = _load_stark_profiles(
        os.path.join(_data_dir, "data", "Stehle-Hutchson-hydrogen-profiles.h5")
    )
except Exception:
    _hline_stark_profiles = None

# Cache the packed Stark arrays for faster access
_stark_cache = None


def _pack_stark_profiles(stark_profiles):
    """Pack Stark profiles into arrays for faster looping."""
    if stark_profiles is None:
        return None
    n_lines = len(stark_profiles)
    temps = np.array(stark_profiles[0]["temps"], dtype=float)
    nes = np.array(stark_profiles[0]["nes"], dtype=float)
    log_delta = np.array(stark_profiles[0]["log_delta"], dtype=float)

    nT = len(temps)
    nNe = len(nes)
    nD = len(log_delta)

    logP = np.zeros((n_lines, nT, nNe, nD), dtype=float)
    lambda0 = np.zeros((n_lines, nT, nNe), dtype=float)
    lower = np.zeros(n_lines, dtype=np.int32)
    upper = np.zeros(n_lines, dtype=np.int32)
    log_gf = np.zeros(n_lines, dtype=float)

    for i, line in enumerate(stark_profiles):
        logP[i] = line["logP"]
        lambda0[i] = line["lambda0"]
        lower[i] = line["lower"]
        upper[i] = line["upper"]
        log_gf[i] = line["log_gf"]

    return {
        "temps": temps,
        "nes": nes,
        "log_delta": log_delta,
        "logP": logP,
        "lambda0": lambda0,
        "lower": lower,
        "upper": upper,
        "log_gf": log_gf,
    }


def _bilinear_interp(x, y, x_grid, y_grid, table):
    x = float(np.clip(x, x_grid[0], x_grid[-1]))
    y = float(np.clip(y, y_grid[0], y_grid[-1]))

    ix = np.clip(np.searchsorted(x_grid, x, side="right") - 1, 0, len(x_grid) - 2)
    iy = np.clip(np.searchsorted(y_grid, y, side="right") - 1, 0, len(y_grid) - 2)

    x0, x1 = x_grid[ix], x_grid[ix + 1]
    y0, y1 = y_grid[iy], y_grid[iy + 1]

    tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)

    f00 = table[ix, iy]
    f10 = table[ix + 1, iy]
    f01 = table[ix, iy + 1]
    f11 = table[ix + 1, iy + 1]

    return (1 - tx) * (1 - ty) * f00 + tx * (1 - ty) * f10 + (1 - tx) * ty * f01 + tx * ty * f11


def _trilinear_interp(x, y, z, x_grid, y_grid, z_grid, table):
    x = float(np.clip(x, x_grid[0], x_grid[-1]))
    y = float(np.clip(y, y_grid[0], y_grid[-1]))
    z = np.clip(z, z_grid[0], z_grid[-1])

    ix = np.clip(np.searchsorted(x_grid, x, side="right") - 1, 0, len(x_grid) - 2)
    iy = np.clip(np.searchsorted(y_grid, y, side="right") - 1, 0, len(y_grid) - 2)
    iz = np.clip(np.searchsorted(z_grid, z, side="right") - 1, 0, len(z_grid) - 2)

    x0, x1 = x_grid[ix], x_grid[ix + 1]
    y0, y1 = y_grid[iy], y_grid[iy + 1]
    z0, z1 = z_grid[iz], z_grid[iz + 1]

    tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
    tz = np.where(z1 == z0, 0.0, (z - z0) / (z1 - z0))

    f000 = table[ix, iy, iz]
    f100 = table[ix + 1, iy, iz]
    f010 = table[ix, iy + 1, iz]
    f110 = table[ix + 1, iy + 1, iz]
    f001 = table[ix, iy, iz + 1]
    f101 = table[ix + 1, iy, iz + 1]
    f011 = table[ix, iy + 1, iz + 1]
    f111 = table[ix + 1, iy + 1, iz + 1]

    c00 = f000 * (1 - tx) + f100 * tx
    c10 = f010 * (1 - tx) + f110 * tx
    c01 = f001 * (1 - tx) + f101 * tx
    c11 = f011 * (1 - tx) + f111 * tx

    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty

    return c0 * (1 - tz) + c1 * tz


# ── Brackett series ─────────────────────────────────────────────────────────


_greim_Kmn_table = np.array([
    [0.0001716, 0.0090190, 0.1001000, 0.5820000],
    [0.0005235, 0.0177200, 0.1710000, 0.8660000],
    [0.0008912, 0.0250700, 0.2230000, 1.0200000],
], dtype=float)


def greim_1960_Knm(n, m):
    if (m - n <= 3) and (n <= 4):
        return _greim_Kmn_table[m - n - 1, n - 1]
    return 5.5e-5 * n ** 4 * m ** 4 / (m ** 2 - n ** 2) / (1.0 + 0.13 / (m - n))


_holtsmark_PROB7 = np.array([
    [0.005, 0.128, 0.260, 0.389, 0.504],
    [0.004, 0.109, 0.220, 0.318, 0.389],
    [-0.007, 0.079, 0.162, 0.222, 0.244],
    [-0.018, 0.041, 0.089, 0.106, 0.080],
    [-0.026, -0.003, 0.003, -0.023, -0.086],
    [-0.025, -0.048, -0.087, -0.148, -0.234],
    [-0.008, -0.085, -0.165, -0.251, -0.343],
    [0.018, -0.111, -0.223, -0.321, -0.407],
    [0.032, -0.130, -0.255, -0.354, -0.431],
    [0.014, -0.148, -0.269, -0.359, -0.427],
    [-0.005, -0.140, -0.243, -0.323, -0.386],
    [0.005, -0.095, -0.178, -0.248, -0.307],
    [-0.002, -0.068, -0.129, -0.187, -0.241],
    [-0.007, -0.049, -0.094, -0.139, -0.186],
    [-0.010, -0.036, -0.067, -0.103, -0.143],
], dtype=float)
_holtsmark_C7 = np.array([511.318, 1.532, 4.044, 19.266, 41.812], dtype=float)
_holtsmark_D7 = np.array([-6.070, -4.528, -8.759, -14.984, -23.956], dtype=float)
_holtsmark_PP = np.array([0.0, 0.2, 0.4, 0.6, 0.8], dtype=float)
_holtsmark_beta_knots = np.array([
    1.0, 1.259, 1.585, 1.995, 2.512, 3.162, 3.981, 5.012, 6.310, 7.943,
    10.0, 12.59, 15.85, 19.95, 25.12
], dtype=float)


def holtsmark_profile(beta, P):
    if beta > 500:
        return (1.5 / math.sqrt(beta) + 27.0 / beta ** 2) / beta ** 2

    IM = min(int(math.floor(5 * P + 1)), 4)
    IP = IM + 1
    WTPP = 5 * (P - _holtsmark_PP[IM - 1])
    WTPM = 1 - WTPP

    if beta <= 25.12:
        JP = max(2, int(np.searchsorted(_holtsmark_beta_knots, beta, side="right")))
        JM = JP - 1

        WTBP = ((beta - _holtsmark_beta_knots[JM - 1]) /
                (_holtsmark_beta_knots[JP - 1] - _holtsmark_beta_knots[JM - 1]))
        WTBM = 1 - WTBP

        CBP = _holtsmark_PROB7[JP - 1, IP - 1] * WTPP + _holtsmark_PROB7[JP - 1, IM - 1] * WTPM
        CBM = _holtsmark_PROB7[JM - 1, IP - 1] * WTPP + _holtsmark_PROB7[JM - 1, IM - 1] * WTPM
        CORR = 1 + CBP * WTBP + CBM * WTBM

        WT = max(min(0.5 * (10 - beta), 1), 0)

        PR1 = 8 / (83 + (2 + 0.95 * beta ** 2) * beta) if beta <= 10 else 0.0
        PR2 = (1.5 / math.sqrt(beta) + 27 / beta ** 2) / beta ** 2 if beta >= 8 else 0.0

        return (PR1 * WT + PR2 * (1 - WT)) * CORR

    CC = _holtsmark_C7[IP - 1] * WTPP + _holtsmark_C7[IM - 1] * WTPM
    DD = _holtsmark_D7[IP - 1] * WTPP + _holtsmark_D7[IM - 1] * WTPM
    CORR = 1 + DD / (CC + beta * math.sqrt(beta))
    return (1.5 / math.sqrt(beta) + 27 / beta ** 2) / beta ** 2 * CORR


def brackett_oscillator_strength(n, m):
    GINF = 0.2027 / n ** 0.71
    GCA = 0.124 / n
    FKN = 1.9603 * n
    WTC = 0.45 - 2.4 / n ** 3 * (n - 1)
    FK = FKN * (m / ((m - n) * (m + n))) ** 3
    XMN12 = (m - n) ** 1.2
    WT = (XMN12 - 1) / (XMN12 + WTC)
    return FK * (1 - WT * GINF - (0.222 + GCA / m) * (1 - WT))


def brackett_line_stark_profiles(m, lambdas, lambda0, T, ne):
    n = 4
    nus = c_cgs / lambdas
    nu0 = c_cgs / lambda0

    ne_1_6 = ne ** (1 / 6)
    F0 = 1.25e-9 * ne ** (2 / 3)
    # Matches Julia literal 10_0004 (100004)
    GCON1 = 0.2 + 0.09 * math.sqrt(T / 100004.0) / (1 + ne / 1.0e13)
    GCON2 = 0.2 / (1 + ne / 1.0e15)

    Knm = greim_1960_Knm(n, m)

    Y1WHT = 1e14 if (m - n <= 3) else 1e13
    WTY1 = 1 / (1 + ne / Y1WHT)
    Y1B = 2 / (1 + 0.012 / T * math.sqrt(ne / T))
    C1CON = Knm / lambda0 * (m ** 2 - n ** 2) ** 2 / (n ** 2 * m ** 2) * 1e-8
    Y1NUM = 320
    Y1SCAL = Y1NUM * ((T / 10000.0) ** 0.3 / ne_1_6) * WTY1 + Y1B * (1 - WTY1)
    C1 = F0 * 78940 / T * C1CON * Y1SCAL

    C2 = F0 ** 2 / (5.96e-23 * ne) * (Knm / lambda0) ** 2 * 1e-16

    betas = np.abs(lambdas - lambda0) / F0 / Knm * 1e8
    y1 = C1 * betas
    y2 = C2 * betas ** 2

    G1 = 6.77 * math.sqrt(C1)

    impact = np.zeros_like(betas)
    for i, (yy1, yy2, bb) in enumerate(zip(y1, y2, betas)):
        if (yy2 <= 1e-4) and (yy1 <= 1e-5):
            width = G1 * max(0.0, 0.2114 + math.log(math.sqrt(C2) / C1)) * (1 - GCON1 - GCON2)
        else:
            GAM = (G1 * (0.5 * math.exp(-min(80.0, yy1)) +
                         exponential_integral_1(yy1) - 0.5 * exponential_integral_1(yy2)) *
                   (1 - GCON1 / (1 + (90 * yy1) ** 3) - GCON2 / (1 + 2000 * yy1)))
            width = 0.0 if GAM <= 1e-20 else GAM

        impact[i] = width / (math.pi * (width ** 2 + bb ** 2)) if width > 0 else 0.0

    shielding_parameter = ne_1_6 * 0.08989 / math.sqrt(T)
    quasistatic_ion = np.vectorize(lambda b: holtsmark_profile(b, shielding_parameter))(betas)

    ps = (0.9 * y1) ** 2
    quasistatic_e = (ps + 0.03 * np.sqrt(y1)) / (ps + 1.0)
    quasistatic_e = np.where(np.isfinite(quasistatic_e), quasistatic_e, 0.0)

    total_quasi = quasistatic_ion * (1 + quasistatic_e)

    dβ_dλ = 1e8 / (Knm * F0)
    for prof in (impact, total_quasi):
        prof *= np.sqrt(lambdas / lambda0)
        # Match Julia behavior: apply from start to last index where nu < nu0
        red = np.where(nus < nu0)[0]
        if red.size:
            last = red[-1]
            prof[:last + 1] *= np.exp((hplanck_cgs * (nus[:last + 1] - nu0)) / kboltz_cgs / T)
        prof *= dβ_dλ

    return impact, total_quasi


def bracket_line_interpolator(m, lambda0, T, ne, xi, lambda_min=0.0, lambda_max=np.inf,
                              n_wavelength_points=201, window_size=5,
                              include_doppler_threshold=0.25):
    n = 4
    F0 = 1.25e-9 * ne ** (2 / 3)
    Knm = greim_1960_Knm(n, m)
    stark_width = 1.6678e-18 * Knm * F0 * c_cgs
    H_mass = 1.008 * amu_cgs
    sigma_dop = doppler_width(lambda0, T, H_mass, xi)

    window = window_size * max(sigma_dop, stark_width)
    lambda_start = max(lambda_min, lambda0 - window)
    lambda_end = min(lambda0 + window, lambda_max)
    if lambda_start > lambda_max or lambda_end < lambda_min or lambda_start == lambda_end:
        return lambda lam: np.zeros_like(lam), 0.0

    wls = np.linspace(lambda_start, lambda_end, n_wavelength_points)
    step = wls[1] - wls[0]
    start_ind = (n_wavelength_points - 1) // 2

    phi_impact, phi_quasi = brackett_line_stark_profiles(m, wls, lambda0, T, ne)

    if sigma_dop / stark_width > include_doppler_threshold:
        phi_dop = normal_pdf(wls - lambda0, sigma_dop)
        phi_quasi = np.convolve(phi_quasi, phi_dop, mode="full") * step
        phi_quasi = phi_quasi[start_ind:start_ind + n_wavelength_points]

    phi_conv = np.convolve(phi_impact, phi_quasi, mode="full") * step
    phi_conv = phi_conv[start_ind:start_ind + n_wavelength_points]

    def itp(lam):
        return np.interp(lam, wls, phi_conv, left=0.0, right=0.0)

    return itp, window


# ── Main API ────────────────────────────────────────────────────────────────


def hydrogen_line_absorption(wl_grid, T, ne, nH_I, nHe_I, U_H_I, xi, window_size,
                              stark_profiles=None, use_MHD=True):
    """Compute hydrogen line absorption on a wavelength grid.

    Parameters
    ----------
    wl_grid : (n_wl,) wavelength array in cm
    T : temperature (K)
    ne : electron number density
    nH_I : neutral hydrogen number density
    nHe_I : neutral helium number density
    U_H_I : partition function for H I
    xi : microturbulent velocity (cm/s)
    window_size : max distance from line center (cm)
    """
    if stark_profiles is None:
        stark_profiles = _hline_stark_profiles

    if stark_profiles is None:
        # Fall back to approximate Stark/Lorentz profiles if tables are unavailable
        return _hydrogen_line_absorption_approx(wl_grid, T, ne, nH_I, 1.0 / U_H_I)

    global _stark_cache
    if _stark_cache is None:
        _stark_cache = _pack_stark_profiles(stark_profiles)
    packed = _stark_cache
    if packed is None:
        return _hydrogen_line_absorption_approx(wl_grid, T, ne, nH_I, 1.0 / U_H_I)

    wls = np.asarray(wl_grid)
    alpha = np.zeros_like(wls)

    nus = c_cgs / wls
    dnu_dlambda = c_cgs / wls ** 2

    n_max = int(np.max(packed["upper"]))
    ws = np.ones(n_max + 1)
    if use_MHD:
        for n in range(1, n_max + 1):
            ws[n] = hummer_mihalas_w(T, n, nH_I, nHe_I, ne)

    beta = 1.0 / (kboltz_eV * T)
    F0 = 1.25e-9 * ne ** (2 / 3)

    temps = packed["temps"]
    nes = packed["nes"]
    log_delta = packed["log_delta"]

    if T >= temps[0] and T <= temps[-1] and ne >= nes[0] and ne <= nes[-1]:
        for i in range(packed["logP"].shape[0]):
            line_lower = packed["lower"][i]
            line_upper = packed["upper"][i]
            line_log_gf = packed["log_gf"][i]

            lambda0 = _bilinear_interp(T, ne, temps, nes, packed["lambda0"][i])

            Elo = RydbergH_eV * (1 - 1 / line_lower ** 2)
            Eup = RydbergH_eV * (1 - 1 / line_upper ** 2)

            levels_factor = ws[line_upper] * (math.exp(-beta * Elo) - math.exp(-beta * Eup)) / U_H_I
            amplitude = 10.0 ** line_log_gf * nH_I * sigma_line(lambda0) * levels_factor

            lb = np.searchsorted(wls, lambda0 - window_size, side="left")
            ub = np.searchsorted(wls, lambda0 + window_size, side="right")
            if lb >= ub:
                continue

            if line_lower == 2 and line_upper in (3, 4, 5):
                if line_upper == 3:
                    lambda0_pd, sigma_ABO, alpha_ABO = 6.56460998e-5, 1180.0, 0.677
                elif line_upper == 4:
                    lambda0_pd, sigma_ABO, alpha_ABO = 4.8626810200000004e-5, 2320.0, 0.455
                else:
                    lambda0_pd, sigma_ABO, alpha_ABO = 4.34168232e-5, 4208.0, 0.380

                Gamma = scaled_vdW((sigma_ABO * bohr_radius_cgs ** 2, alpha_ABO), amu_cgs, T) * nH_I
                gamma = Gamma * lambda0_pd ** 2 / (c_cgs * 4.0 * math.pi)
                H_mass = 1.008 * amu_cgs
                sigma = doppler_width(lambda0_pd, T, H_mass, xi)

                alpha[lb:ub] += line_profile(lambda0_pd, sigma, gamma, amplitude, wls[lb:ub])

            nu0 = c_cgs / lambda0
            scaled_dnu = _zero2epsilon(np.abs(nus[lb:ub] - nu0) / F0)
            log_scaled = np.log(scaled_dnu)

            logP = _trilinear_interp(T, ne, log_scaled, temps, nes,
                                     log_delta, packed["logP"][i])
            dIdnu = np.exp(logP)

            alpha[lb:ub] += dIdnu * dnu_dlambda[lb:ub] * amplitude

    # Brackett series (n=4)
    n = 4
    E_low = RydbergH_eV * (1 - 1 / n ** 2)
    for m in range(5, n_max + 1):
        E = RydbergH_eV * (1 / n ** 2 - 1 / m ** 2)
        lambda0 = hplanck_eV * c_cgs / E
        levels_factor = ws[m] * math.exp(-beta * E_low) * (1 - math.exp(-beta * E)) / U_H_I
        gf = 2 * n ** 2 * brackett_oscillator_strength(n, m)
        amplitude = gf * nH_I * sigma_line(lambda0) * levels_factor

        itp, window = bracket_line_interpolator(m, lambda0, T, ne, xi, wls[0], wls[-1])
        if window == 0.0:
            continue

        lb = np.searchsorted(wls, lambda0 - window, side="left")
        ub = np.searchsorted(wls, lambda0 + window, side="right")
        alpha[lb:ub] += itp(wls[lb:ub]) * amplitude

    return jnp.asarray(alpha)


def hydrogen_line_absorption_layers(wl_grid, T, ne, nH_I, nHe_I, U_H_I, xi, window_size,
                                    stark_profiles=None, use_MHD=True):
    """Vectorized hydrogen line absorption across layers (Python loop)."""
    T = np.asarray(T)
    ne = np.asarray(ne)
    nH_I = np.asarray(nH_I)
    nHe_I = np.asarray(nHe_I)
    U_H_I = np.asarray(U_H_I)
    xi = np.asarray(xi) if np.ndim(xi) > 0 else np.full_like(T, xi)

    out = np.zeros((len(T), len(wl_grid)), dtype=float)
    for i in range(len(T)):
        out[i] = np.asarray(hydrogen_line_absorption(
            wl_grid, T[i], ne[i], nH_I[i], nHe_I[i], U_H_I[i], xi[i], window_size,
            stark_profiles=stark_profiles, use_MHD=use_MHD
        ))
    return jnp.asarray(out)


# ── Approximate fallback (previous implementation) ──────────────────────────


def _H_level_energy(n):
    return RydbergH_eV * (1.0 - 1.0 / n ** 2)


def _H_line_wavelength(n_lower, n_upper):
    dE = RydbergH_eV * (1.0 / n_lower ** 2 - 1.0 / n_upper ** 2)
    return hplanck_eV * c_cgs / dE


def _approximate_stark_width(n_lower, n_upper, ne, T):
    n2 = n_upper ** 2
    F0 = 2.61 * ne ** (2.0 / 3.0)
    delta_nu = 0.7 * n2 * F0 * electron_charge_cgs / (electron_mass_cgs * c_cgs)
    wl0 = _H_line_wavelength(n_lower, n_upper)
    delta_lam = wl0 ** 2 * delta_nu / c_cgs
    return delta_lam


def _hydrogen_line_absorption_approx(wl_grid, T, ne, nH_I, inv_U_H, series="balmer"):
    wl_grid = jnp.asarray(wl_grid)
    alpha = jnp.zeros_like(wl_grid)

    kT = kboltz_eV * T

    series_lines = {
        "lyman": [(1, n) for n in range(2, 31)],
        "balmer": [(2, n) for n in range(3, 31)],
        "paschen": [(3, n) for n in range(4, 21)],
        "brackett": [(4, n) for n in range(5, 21)],
    }[series]

    for n_lo, n_hi in series_lines:
        wl0 = _H_line_wavelength(n_lo, n_hi)
        E_lo = _H_level_energy(n_lo)
        g_lo = 2.0 * n_lo ** 2

        pop = nH_I * inv_U_H * g_lo * jnp.exp(-E_lo / kT)

        dn = 1.0 / n_lo ** 2 - 1.0 / n_hi ** 2
        f_osc = 0.6407 / (n_lo ** 2 * dn ** 3 * n_hi ** 3)

        strength = (math.pi * electron_charge_cgs ** 2 / (electron_mass_cgs * c_cgs)
                    * f_osc * float(pop))

        delta_lam = _approximate_stark_width(n_lo, n_hi, ne, T)
        delta_D = wl0 * jnp.sqrt(2.0 * kboltz_cgs * T
                                 / (1.008 * 1.6605e-24)) / c_cgs
        delta_tot = jnp.sqrt(delta_D ** 2 + delta_lam ** 2)

        dw = wl_grid - wl0
        profile = (delta_tot / jnp.pi) / (dw ** 2 + delta_tot ** 2)

        nu0 = c_cgs / wl0
        stim = 1.0 - jnp.exp(-hplanck_eV * nu0 / kT)

        alpha = alpha + strength * stim * profile

    return alpha
