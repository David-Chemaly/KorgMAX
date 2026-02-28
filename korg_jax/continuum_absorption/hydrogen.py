"""Hydrogen continuum absorption: H I bf, H- bf/ff, H2+ bf/ff.

Ported from Korg.jl/src/ContinuumAbsorption/absorption_H.jl and Stancil1994.jl.
"""
from __future__ import annotations

import os
import math
import warnings
import numpy as np

from ..constants import (
    c_cgs, hplanck_cgs, hplanck_eV, kboltz_cgs, kboltz_eV,
    electron_mass_cgs, electron_charge_cgs, RydbergH_eV, Rydberg_eV,
    bohr_radius_cgs, eV_to_cgs,
)
from ..statmech import hummer_mihalas_w, hummer_mihalas_w_vec

# ── Data directory ────────────────────────────────────────────────────────────
_data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")

# ── H- ion energy ────────────────────────────────────────────────────────────
_Hminus_ion_energy = 0.754204  # eV, used by McLaughlin+ 2017


# ── Load H I bound-free cross-sections (Nahar 2021) ──────────────────────────
# Each entry is (n, interp_E, interp_sigma) for n=1..6
# E in eV, sigma in megabarns (Mb)
_H_I_bf_cross_sections = None
try:
    import h5py
    _h5_path = os.path.join(_data_dir, "bf_cross-sections",
                            "individual_H_cross-sections.h5")
    if os.path.exists(_h5_path):
        with h5py.File(_h5_path, "r") as f:
            ns = f["n"][:]
            Es_all = f["E"][:]   # shape (n_points, n_levels)
            sigmas_all = f["sigma"][:]  # same shape
        _H_I_bf_cross_sections = []
        for col_idx in range(len(ns)):
            n = int(ns[col_idx])
            Es = Es_all[:, col_idx].copy()
            sigs = sigmas_all[:, col_idx].copy()
            # Remove NaN entries (padding)
            mask = np.isfinite(Es) & np.isfinite(sigs)
            _H_I_bf_cross_sections.append((n, Es[mask], sigs[mask]))
except ImportError:
    pass


# ── Load H- bound-free cross-sections (McLaughlin 2017) ──────────────────────
_Hminus_bf_nu = None
_Hminus_bf_sigma = None
try:
    import h5py
    _h5_Hminus_path = os.path.join(_data_dir, "McLaughlin2017Hminusbf.h5")
    if os.path.exists(_h5_Hminus_path):
        with h5py.File(_h5_Hminus_path, "r") as f:
            _Hminus_bf_nu = f["nu"][:].copy()
            _Hminus_bf_sigma = f["sigma"][:].copy()
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════════════
#  H I bound-free
# ══════════════════════════════════════════════════════════════════════════════

def simple_hydrogen_bf_cross_section(n, nu):
    """Hydrogenic bf cross-section in megabarns (Kurucz 1970, eqn 5.5 corrected).

    Used for n > 6 where detailed cross-sections are unavailable.
    Returns 0 when the photon energy is below the ionization threshold.
    """
    inv_n = 1.0 / n
    inv_n2 = inv_n * inv_n
    threshold = RydbergH_eV * inv_n2
    # 64*pi^4*e^10*m_e / (c * h^6 * 3*sqrt(3))
    bf_sigma_const = 2.815e29
    sigma = bf_sigma_const * (inv_n2 * inv_n2 * inv_n) * (1.0 / nu) ** 3 * 1e18  # Mb
    return np.where(hplanck_eV * nu >= threshold, sigma, 0.0)


def H_I_bf(nus, T, nH_I, nHe_I, ne, invU_H, n_max_MHD=6,
           use_MHD_for_Lyman=False):
    """H I bound-free absorption including MHD occupation probability.

    For n=1..n_max_MHD, uses Nahar 2021 cross-sections (if available) with
    MHD occupation probabilities and dissolved-fraction calculation.
    For n > n_max_MHD up to 40, uses the hydrogenic approximation.

    Parameters
    ----------
    nus : (N,) sorted frequency array (Hz)
    T : temperature (K)
    nH_I : neutral H number density (cm^-3)
    nHe_I : neutral He number density (cm^-3)
    ne : electron number density (cm^-3)
    invU_H : 1 / U(H I), inverse partition function
    n_max_MHD : int, max level for detailed cross-sections (default 6)
    use_MHD_for_Lyman : bool, apply MHD to n=1 (default False)
    """
    nus = np.asarray(nus, dtype=np.float64)
    chi = RydbergH_eV  # 13.598... eV
    kT = kboltz_eV * T

    total_cross_section = np.zeros_like(nus)

    # --- Levels n=1..n_max_MHD with detailed cross-sections ---
    if _H_I_bf_cross_sections is not None:
        for (n, Es_np, sigs_np) in _H_I_bf_cross_sections[:n_max_MHD]:
            w_lower = hummer_mihalas_w(T, n, nH_I, nHe_I, ne)
            # Degeneracy is already in the Nahar cross-sections
            occupation_prob = w_lower * math.exp(-chi * (1 - 1.0 / n**2) / kT)

            nu_break = chi / (n**2 * hplanck_eV)  # ionization freq for level n

            # --- Cross-section ---
            # For nu >= nu_break: interpolate from tabulated data
            # For nu < nu_break: extrapolate as sigma ~ nu^{-3}
            E_tab = np.asarray(Es_np, dtype=np.float64)
            sig_tab = np.asarray(sigs_np, dtype=np.float64)

            photon_E = hplanck_eV * nus  # eV
            # Linear interpolation with no extrapolation (clamp to range)
            sigma_interp = np.interp(photon_E, E_tab, sig_tab,
                                      left=sig_tab[0], right=sig_tab[-1])

            # Cross-section at nu_break for extrapolation below threshold
            sigma_at_break = float(np.interp(nu_break * hplanck_eV, Es_np, sigs_np))
            scaling_factor = sigma_at_break / nu_break**(-3)

            cross_section = np.where(
                nus > nu_break,
                sigma_interp,
                nus**(-3) * scaling_factor,
            )

            # --- Dissolved fraction ---
            # For nu >= nu_break: dissolved_fraction = 1.0
            # For nu < nu_break: depends on MHD of upper level
            if not use_MHD_for_Lyman and n == 1:
                dissolved_fraction = np.where(nus > nu_break, 1.0, 0.0)
            else:
                # For frequencies below the ionization threshold, compute
                # the effective quantum number and MHD probability
                n_eff_arr = 1.0 / np.sqrt(np.maximum(
                    1.0 / n**2 - hplanck_eV * nus / chi, 1e-30))

                # Vectorized MHD occupation probability for all frequencies
                n_eff_np = np.array(n_eff_arr)
                valid = (n_eff_np > 0) & np.isfinite(n_eff_np)
                safe_n_eff = np.where(valid, n_eff_np, 1.0)
                w_upper_np = np.where(valid, hummer_mihalas_w_vec(T, safe_n_eff, nH_I, nHe_I, ne), 0.0)
                w_upper = np.asarray(w_upper_np)

                frac = np.where(
                    w_lower > 0,
                    1.0 - w_upper / w_lower,
                    0.0,
                )
                dissolved_fraction = np.where(nus > nu_break, 1.0, frac)

            total_cross_section = total_cross_section + (
                occupation_prob * cross_section * dissolved_fraction
            )
    else:
        # Fallback: use hydrogenic approximation for all levels
        for n in range(1, n_max_MHD + 1):
            w_lower = hummer_mihalas_w(T, n, nH_I, nHe_I, ne)
            occupation_prob = (2.0 * n**2 * w_lower
                               * math.exp(-chi * (1 - 1.0 / n**2) / kT))
            total_cross_section = total_cross_section + (
                occupation_prob * simple_hydrogen_bf_cross_section(n, nus)
            )

    # --- Levels n > n_max_MHD using hydrogenic approximation ---
    for n in range(n_max_MHD + 1, 41):
        w_lower = hummer_mihalas_w(T, n, nH_I, nHe_I, ne)
        if w_lower < 1e-5:
            break
        occupation_prob = (2.0 * n**2 * w_lower
                           * math.exp(-chi * (1 - 1.0 / n**2) / kT))
        total_cross_section = total_cross_section + (
            occupation_prob * simple_hydrogen_bf_cross_section(n, nus)
        )

    # Factor 1e-18 converts cross-sections from megabarns to cm^2
    return (nH_I * invU_H * total_cross_section
            * (1.0 - np.exp(-hplanck_eV * nus / kT)) * 1e-18)


# ══════════════════════════════════════════════════════════════════════════════
#  H- number density
# ══════════════════════════════════════════════════════════════════════════════

def _ndens_Hminus(nH_I_div_partition, ne, T, ion_energy=_Hminus_ion_energy):
    """Number density of H- via Saha equation (Kurucz 1970, eqn 5.10)."""
    nHI_groundstate = 2.0 * nH_I_div_partition
    coef = 3.31283018e-22  # (h^2/(2*pi*m))^1.5, cm^3 * eV^1.5
    beta = 1.0 / (kboltz_eV * T)
    return 0.25 * nHI_groundstate * ne * coef * beta**1.5 * np.exp(ion_energy * beta)


# ══════════════════════════════════════════════════════════════════════════════
#  H- bound-free  (McLaughlin 2017)
# ══════════════════════════════════════════════════════════════════════════════

# Fallback Wishart 1979 table (used only if HDF5 not found)
_Hminus_bf_lambda_wishart = np.array([
    1250., 1500., 1750., 2000., 2500., 3000., 3500., 4000., 4500.,
    5000., 5500., 6000., 6500., 7000., 7500., 8000., 8500., 9000.,
    9500., 10000., 10500., 11000., 11500., 12000., 12500., 13000.,
    13500., 14000., 14500., 15000., 15500., 16000., 16500.,
])
_Hminus_bf_sigma_wishart = np.array([
    0.166, 0.566, 1.099, 1.706, 2.935, 3.841, 4.229, 4.158, 3.873,
    3.500, 3.114, 2.739, 2.389, 2.067, 1.776, 1.515, 1.282, 1.076,
    0.893, 0.731, 0.589, 0.464, 0.354, 0.258, 0.175, 0.103, 0.042,
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
]) * 1e-18  # cm^2


def _Hminus_bf_cross_section(nu):
    """H- bf cross-section in cm^2 (excluding stimulated emission).

    Uses McLaughlin 2017 data if available, otherwise Wishart 1979.
    Includes (nu - ion_nu)^1.5 power law extrapolation at the low-energy end.
    """
    ion_nu = _Hminus_ion_energy / hplanck_eV

    if _Hminus_bf_nu is not None and _Hminus_bf_sigma is not None:
        # McLaughlin 2017 from HDF5
        nu_tab = np.asarray(_Hminus_bf_nu)
        sig_tab = np.asarray(_Hminus_bf_sigma)
        min_interp_nu = float(_Hminus_bf_nu.min())

        # Low-energy power law: sigma = coef * (nu - ion_nu)^1.5
        sig_at_min = float(np.interp(min_interp_nu, _Hminus_bf_nu, _Hminus_bf_sigma))
        low_nu_coef = sig_at_min / (min_interp_nu - ion_nu)**1.5

        sigma_interp = np.interp(nu, nu_tab, sig_tab,
                                  left=sig_tab[0], right=sig_tab[-1])
        sigma_low = low_nu_coef * np.maximum(nu - ion_nu, 0.0)**1.5

        # Choose: below table range use power law, in range use table, above ion use 0
        sigma = np.where(
            nu <= ion_nu,
            0.0,
            np.where(nu < min_interp_nu, sigma_low, sigma_interp),
        )
    else:
        # Wishart 1979 fallback: convert nu to wavelength in Angstroms
        lam_A = c_cgs * 1e8 / np.maximum(nu, 1.0)
        sigma = np.interp(lam_A,
                           np.asarray(_Hminus_bf_lambda_wishart),
                           np.asarray(_Hminus_bf_sigma_wishart),
                           left=0.0, right=0.0)
        sigma = np.where(nu <= ion_nu, 0.0, sigma)

    return sigma


def Hminus_bf(nus, T, nH_I_div_U, ne):
    """H- bound-free absorption coefficient.

    Uses McLaughlin 2017 cross-sections (from HDF5) if available,
    otherwise falls back to Wishart 1979.
    """
    nus = np.asarray(nus)
    sigma = _Hminus_bf_cross_section(nus)  # cm^2
    stim = 1.0 - np.exp(-hplanck_cgs * nus / (kboltz_cgs * T))
    n_Hminus = _ndens_Hminus(nH_I_div_U, ne, T, _Hminus_ion_energy)
    return n_Hminus * sigma * stim


# ══════════════════════════════════════════════════════════════════════════════
#  H- free-free  (Bell & Berrington 1987, Table 1)
# ══════════════════════════════════════════════════════════════════════════════

# Table from Bell & Berrington (1987) https://doi.org/10.1088/0022-3700/20/4/019
# K(lambda, theta) in units of 1e-26 cm^4/dyn (factor built into table values)
_Hminus_ff_theta = np.array([0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.8, 3.6])
_Hminus_ff_lambda = np.array([
    1823, 2278, 2604, 3038, 3645, 4557, 5063, 5696, 6510, 7595, 9113,
    10126, 11392, 13019, 15189, 18227, 22784, 30378, 45567, 91134,
    113918, 151890,
], dtype=float)
_Hminus_ff_K = np.array([
    [0.0178, 0.0222, 0.0308, 0.0402, 0.0498, 0.0596, 0.0695, 0.0795, 0.0896, 0.131, 0.172],
    [0.0228, 0.0280, 0.0388, 0.0499, 0.0614, 0.0732, 0.0851, 0.0972, 0.110, 0.160, 0.211],
    [0.0277, 0.0342, 0.0476, 0.0615, 0.0760, 0.0908, 0.105, 0.121, 0.136, 0.199, 0.262],
    [0.0364, 0.0447, 0.0616, 0.0789, 0.0966, 0.114, 0.132, 0.150, 0.169, 0.243, 0.318],
    [0.0520, 0.0633, 0.0859, 0.108, 0.131, 0.154, 0.178, 0.201, 0.225, 0.321, 0.418],
    [0.0791, 0.0959, 0.129, 0.161, 0.194, 0.227, 0.260, 0.293, 0.327, 0.463, 0.602],
    [0.0965, 0.117, 0.157, 0.195, 0.234, 0.272, 0.311, 0.351, 0.390, 0.549, 0.711],
    [0.121, 0.146, 0.195, 0.241, 0.288, 0.334, 0.381, 0.428, 0.475, 0.667, 0.861],
    [0.154, 0.188, 0.249, 0.309, 0.367, 0.424, 0.482, 0.539, 0.597, 0.830, 1.07],
    [0.208, 0.250, 0.332, 0.409, 0.484, 0.557, 0.630, 0.702, 0.774, 1.06, 1.36],
    [0.293, 0.354, 0.468, 0.576, 0.677, 0.777, 0.874, 0.969, 1.06, 1.45, 1.83],
    [0.358, 0.432, 0.572, 0.702, 0.825, 0.943, 1.06, 1.17, 1.28, 1.73, 2.17],
    [0.448, 0.539, 0.711, 0.871, 1.02, 1.16, 1.29, 1.43, 1.57, 2.09, 2.60],
    [0.579, 0.699, 0.924, 1.13, 1.33, 1.51, 1.69, 1.86, 2.02, 2.67, 3.31],
    [0.781, 0.940, 1.24, 1.52, 1.78, 2.02, 2.26, 2.48, 2.69, 3.52, 4.31],
    [1.11, 1.34, 1.77, 2.17, 2.53, 2.87, 3.20, 3.51, 3.80, 4.92, 5.97],
    [1.73, 2.08, 2.74, 3.37, 3.90, 4.50, 5.01, 5.50, 5.95, 7.59, 9.06],
    [3.04, 3.65, 4.80, 5.86, 6.86, 7.79, 8.67, 9.50, 10.3, 13.2, 15.6],
    [6.79, 8.16, 10.7, 13.1, 15.3, 17.4, 19.4, 21.2, 23.0, 29.5, 35.0],
    [27.0, 32.4, 42.6, 51.9, 60.7, 68.9, 76.8, 84.2, 91.4, 117.0, 140.0],
    [42.3, 50.6, 66.4, 80.8, 94.5, 107.0, 120.0, 131.0, 142.0, 183.0, 219.0],
    [75.1, 90.0, 118.0, 144.0, 168.0, 191.0, 212.0, 234.0, 253.0, 325.0, 388.0],
], dtype=float)


def Hminus_ff(nus, T, nH_I_div_U, ne):
    """H- free-free absorption coefficient.

    Uses Bell & Berrington (1987) Table 1, which tabulates K(lambda, theta)
    in units of cm^4/dyn (with 1e-26 built in). Multiply by Pe and ground-state
    H I number density to get alpha.

    Valid range: 1823-151890 Angstroms, T = 1400-10080 K (theta 0.5-3.6).
    """
    nus = np.asarray(nus)
    lam_A = c_cgs * 1e8 / nus  # Angstroms
    theta = 5040.0 / T

    # Bilinear interpolation into Bell & Berrington table
    K = 1e-26 * _bilinear_interp_jax(
        lam_A, np.full_like(nus, theta),
        _Hminus_ff_lambda, _Hminus_ff_theta, _Hminus_ff_K,
    )

    Pe = ne * kboltz_cgs * T
    nHI_gs = 2.0 * nH_I_div_U  # ground state n=1, g=2, E=0

    return K * Pe * nHI_gs


# ══════════════════════════════════════════════════════════════════════════════
#  H2+ bound-free and free-free  (Stancil 1994)
# ══════════════════════════════════════════════════════════════════════════════

# Temperature grids
_H2plus_Ts = np.array([3150, 4200, 5040, 6300, 8400, 12600, 16800, 25200], dtype=float)
_H2plus_K_vals = 1e19 * np.array([
    0.9600, 9.7683, 29.997, 89.599, 265.32, 845.01, 1685.3, 4289.5,
], dtype=float)

# Wavelength grids (Angstroms, converted from nm)
_H2plus_ff_lambda = 10.0 * np.array([
    70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220,
    230, 240, 250, 260, 270, 280, 290, 295, 300, 350, 400, 450, 500, 600, 700,
    800, 900, 1000, 2000, 3000, 4000, 5000, 11000, 15000, 20000,
], dtype=float)

_H2plus_bf_lambda = 10.0 * np.array([
    50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
    210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 350, 400, 450, 500, 600,
    700, 800, 900, 1000, 2000, 3000, 4500, 5000, 10000, 15000, 20000,
], dtype=float)

# Cross-section tables (converted to cgs: bf * 1e-18, ff * 1e-39)
_H2plus_bf_sigma = 1e-18 * np.array([
    [7.34e-5, 1.43e-4, 2.04e-4, 2.87e-4, 3.89e-4, 4.97e-4, 5.49e-4, 5.98e-4],
    [0.0100, 0.0150, 0.0186, 0.0230, 0.0276, 0.0319, 0.0337, 0.0353],
    [0.1676, 0.1965, 0.2105, 0.2215, 0.2266, 0.2246, 0.2211, 0.2163],
    [0.8477, 0.8199, 0.7797, 0.7183, 0.6376, 0.5477, 0.5037, 0.4622],
    [2.1113, 1.8166, 1.6135, 1.3823, 1.1403, 0.9157, 0.8176, 0.7313],
    [3.4427, 2.8069, 2.4213, 2.0136, 1.6137, 1.2616, 1.1129, 0.9845],
    [4.3470, 3.5155, 3.0224, 2.5070, 2.0062, 1.5685, 1.3846, 1.2262],
    [4.6981, 3.8841, 3.3806, 2.8402, 2.3019, 1.8203, 1.6147, 1.4358],
    [4.6169, 3.9763, 3.5361, 3.0358, 2.5120, 2.0239, 1.8096, 1.6202],
    [4.2811, 3.8840, 3.5476, 3.1272, 2.6535, 2.1858, 1.9729, 1.7809],
    [3.8331, 3.6850, 3.4660, 3.1434, 2.7388, 2.3082, 2.1033, 1.9138],
    [3.3624, 3.4344, 3.3301, 3.1098, 2.7840, 2.4020, 2.2106, 2.0287],
    [2.9167, 3.1670, 3.1662, 3.0438, 2.7985, 2.4707, 2.2958, 2.1244],
    [2.5172, 2.9031, 2.9909, 2.9573, 2.7898, 2.5175, 2.3610, 2.2019],
    [2.1697, 2.6538, 2.8149, 2.8600, 2.7657, 2.5495, 2.4128, 2.2682],
    [1.8726, 2.4243, 2.6444, 2.7573, 2.7300, 2.5682, 2.4516, 2.3222],
    [1.6208, 2.2160, 2.4821, 2.6522, 2.6843, 2.5725, 2.4748, 2.3597],
    [1.4085, 2.0289, 2.3297, 2.5473, 2.6314, 2.5653, 2.4852, 2.3839],
    [1.2295, 1.8617, 2.1883, 2.4453, 2.5755, 2.5517, 2.4889, 2.4012],
    [1.0785, 1.7129, 2.0581, 2.3479, 2.5188, 2.5347, 2.4885, 2.4144],
    [0.9508, 1.5804, 1.9387, 2.2553, 2.4622, 2.5145, 2.4840, 2.4230],
    [0.8424, 1.4623, 1.8293, 2.1680, 2.4063, 2.4920, 2.4762, 2.4277],
    [0.7501, 1.3568, 1.7296, 2.0865, 2.3530, 2.4701, 2.4687, 2.4327],
    [0.6712, 1.2626, 1.6392, 2.0118, 2.3045, 2.4525, 2.4660, 2.4433],
    [0.6033, 1.1783, 1.5577, 1.9443, 2.2620, 2.4415, 2.4707, 2.4626],
    [0.5448, 1.1029, 1.4843, 1.8834, 2.2256, 2.4369, 2.4825, 2.4901],
    [0.3477, 0.8233, 1.2037, 1.6442, 2.0832, 2.4343, 2.5567, 2.6400],
    [0.2412, 0.6469, 1.0069, 1.4533, 1.9363, 2.3659, 2.5349, 2.6658],
    [0.1782, 0.5283, 0.8605, 1.2951, 1.7920, 2.2630, 2.4604, 2.6222],
    [0.1389, 0.4487, 0.7547, 1.1780, 1.6859, 2.1963, 2.4228, 2.6179],
    [0.0927, 0.3396, 0.6120, 1.0109, 1.5255, 2.0874, 2.3564, 2.6031],
    [0.0685, 0.2749, 0.5175, 0.8894, 1.3923, 1.9700, 2.2587, 2.5323],
    [0.0542, 0.2335, 0.4534, 0.8033, 1.2940, 1.8800, 2.1824, 2.4760],
    [0.0448, 0.2041, 0.4046, 0.7334, 1.2067, 1.7876, 2.0941, 2.3965],
    [0.0382, 0.1813, 0.3635, 0.6701, 1.1194, 1.6816, 1.9827, 2.2828],
    [0.0159, 0.0901, 0.1982, 0.3951, 0.7108, 1.1448, 1.3953, 1.6590],
    [0.0100, 0.0596, 0.1325, 0.2699, 0.4954, 0.8132, 1.0003, 1.1999],
    [6.88e-3, 0.0425, 0.0962, 0.1994, 0.3723, 0.6216, 0.7710, 0.9325],
    [6.41e-3, 0.0400, 0.0908, 0.1889, 0.3540, 0.5932, 0.7371, 0.8932],
    [3.56e-3, 0.0229, 0.0526, 0.1110, 0.2109, 0.3582, 0.4479, 0.5463],
    [2.50e-3, 0.0161, 0.0373, 0.0790, 0.1506, 0.2567, 0.3216, 0.3929],
    [1.90e-3, 0.0123, 0.0286, 0.0607, 0.1161, 0.1982, 0.2487, 0.3042],
], dtype=float)

_H2plus_ff_sigma = 1e-39 * np.array([
    [0.0174, 0.0154, 0.0142, 0.0130, 0.0116, 0.0100, 9.10e-3, 8.08e-3],
    [0.0280, 0.0246, 0.0227, 0.0207, 0.0184, 0.0158, 0.0143, 0.0126],
    [0.0394, 0.0346, 0.0319, 0.0290, 0.0257, 0.0220, 0.0199, 0.0175],
    [0.0514, 0.0451, 0.0416, 0.0378, 0.0336, 0.0287, 0.0259, 0.0227],
    [0.0640, 0.0562, 0.0519, 0.0471, 0.0418, 0.0357, 0.0322, 0.0283],
    [0.0770, 0.0676, 0.0624, 0.0567, 0.0504, 0.0431, 0.0389, 0.0341],
    [0.0903, 0.0794, 0.0733, 0.0666, 0.0592, 0.0506, 0.0457, 0.0401],
    [0.1040, 0.0914, 0.0843, 0.0767, 0.0682, 0.0584, 0.0527, 0.0464],
    [0.1177, 0.1035, 0.0956, 0.0869, 0.0773, 0.0663, 0.0599, 0.0527],
    [0.1317, 0.1158, 0.1070, 0.0973, 0.0866, 0.0743, 0.0672, 0.0592],
    [0.1456, 0.1281, 0.1184, 0.1078, 0.0960, 0.0824, 0.0746, 0.0658],
    [0.1597, 0.1405, 0.1299, 0.1183, 0.1054, 0.0906, 0.0821, 0.0725],
    [0.1737, 0.1530, 0.1414, 0.1288, 0.1149, 0.0988, 0.0896, 0.0793],
    [0.1877, 0.1654, 0.1529, 0.1394, 0.1243, 0.1071, 0.0972, 0.0861],
    [0.2017, 0.1777, 0.1644, 0.1499, 0.1338, 0.1154, 0.1048, 0.0930],
    [0.2156, 0.1901, 0.1759, 0.1605, 0.1433, 0.1237, 0.1125, 0.0998],
    [0.2294, 0.2023, 0.1873, 0.1709, 0.1527, 0.1319, 0.1201, 0.1068],
    [0.2431, 0.2145, 0.1987, 0.1814, 0.1622, 0.1402, 0.1277, 0.1137],
    [0.2568, 0.2266, 0.2099, 0.1917, 0.1716, 0.1485, 0.1354, 0.1206],
    [0.2703, 0.2387, 0.2211, 0.2020, 0.1809, 0.1567, 0.1430, 0.1276],
    [0.2836, 0.2506, 0.2322, 0.2123, 0.1902, 0.1649, 0.1506, 0.1345],
    [0.2969, 0.2624, 0.2433, 0.2225, 0.1994, 0.1731, 0.1582, 0.1414],
    [0.3100, 0.2741, 0.2542, 0.2325, 0.2086, 0.1812, 0.1657, 0.1483],
    [0.3165, 0.2799, 0.2596, 0.2376, 0.2131, 0.1853, 0.1695, 0.1518],
    [0.3230, 0.2857, 0.2650, 0.2425, 0.2177, 0.1893, 0.1732, 0.1552],
    [0.3858, 0.3419, 0.3176, 0.2913, 0.2621, 0.2290, 0.2103, 0.1894],
    [0.4451, 0.3952, 0.3677, 0.3378, 0.3048, 0.2674, 0.2463, 0.2228],
    [0.5011, 0.4457, 0.4152, 0.3821, 0.3456, 0.3044, 0.2812, 0.2555],
    [0.5539, 0.4935, 0.4603, 0.4243, 0.3848, 0.3401, 0.3150, 0.2873],
    [0.6511, 0.5821, 0.5442, 0.5032, 0.4583, 0.4078, 0.3795, 0.3484],
    [0.7388, 0.6625, 0.6207, 0.5756, 0.5262, 0.4710, 0.4402, 0.4064],
    [0.8186, 0.7362, 0.6911, 0.6425, 0.5895, 0.5303, 0.4975, 0.4615],
    [0.8918, 0.8042, 0.7563, 0.7049, 0.6488, 0.5864, 0.5518, 0.5141],
    [0.9596, 0.8675, 0.8173, 0.7633, 0.7047, 0.6395, 0.6036, 0.5644],
    [1.4600, 1.3450, 1.2830, 1.2170, 1.1460, 1.0680, 1.0260, 0.9815],
    [1.8050, 1.6820, 1.6170, 1.5470, 1.4740, 1.3940, 1.3510, 1.3060],
    [2.2000, 2.0750, 2.0090, 1.9390, 1.8650, 1.7860, 1.7450, 1.7000],
    [2.3130, 2.1880, 2.1220, 2.0520, 1.9790, 1.9010, 1.8590, 1.8160],
    [3.1800, 3.0620, 3.0000, 2.9360, 2.8690, 2.7980, 2.7610, 2.7230],
    [3.8110, 3.7000, 3.6420, 3.5820, 3.5200, 3.4560, 3.4220, 3.3870],
    [4.3230, 4.2180, 4.1640, 4.1080, 4.0500, 3.9900, 3.9580, 3.9260],
], dtype=float)


def H2plus_bf_ff(nus, T, nH_I, nH_II):
    """H2+ bound-free + free-free absorption (Stancil 1994).

    Uses tabulated cross-sections with linear extrapolation outside grid.
    """
    nus = np.asarray(nus)
    lambdas_A = c_cgs * 1e8 / nus

    # Interpolate equilibrium constant (linear extrapolation)
    K = _linear_interp_extrap(T, _H2plus_Ts, _H2plus_K_vals)

    # Interpolate cross-sections (bilinear with linear extrapolation)
    sigma_bf = _bilinear_interp_extrap(lambdas_A, T,
                                       _H2plus_bf_lambda, _H2plus_Ts,
                                       _H2plus_bf_sigma)
    sigma_ff = _bilinear_interp_extrap(lambdas_A, T,
                                       _H2plus_ff_lambda, _H2plus_Ts,
                                       _H2plus_ff_sigma)

    # Stimulated emission correction
    beta = 1.0 / (kboltz_eV * T)
    stim = 1.0 - np.exp(-hplanck_eV * nus * beta)

    return (sigma_bf / K + sigma_ff) * nH_I * nH_II * stim


# ══════════════════════════════════════════════════════════════════════════════
#  Interpolation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _linear_interp_extrap(x, xp, fp):
    """1D linear interpolation with linear extrapolation (matching Julia Line()).

    Parameters: x is scalar or array, xp/fp are numpy arrays (grid/values).
    """
    xp = np.asarray(xp)
    fp = np.asarray(fp)
    x = np.asarray(x)

    # Linear extrapolation below
    slope_lo = (fp[1] - fp[0]) / (xp[1] - xp[0])
    extrap_lo = fp[0] + slope_lo * (x - xp[0])

    # Linear extrapolation above
    slope_hi = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    extrap_hi = fp[-1] + slope_hi * (x - xp[-1])

    # Interior interpolation
    interp_val = np.interp(x, xp, fp)

    return np.where(x < xp[0], extrap_lo,
                     np.where(x > xp[-1], extrap_hi, interp_val))


def _bilinear_interp_jax(x, y, x_grid, y_grid, table):
    """Bilinear interpolation (no extrapolation -- clamps to grid edges).

    Used for H- ff where Julia uses Throw() extrapolation (caller must
    ensure inputs are in range).
    """
    xg = np.asarray(x_grid)
    yg = np.asarray(y_grid)
    tab = np.asarray(table)

    x = np.asarray(x)
    y = np.asarray(y)

    x = np.clip(x, xg[0], xg[-1])
    y = np.clip(y, yg[0], yg[-1])

    ix = np.clip(np.searchsorted(xg, x, side="right") - 1, 0, len(xg) - 2)
    iy = np.clip(np.searchsorted(yg, y, side="right") - 1, 0, len(yg) - 2)

    x0 = xg[ix]; x1 = xg[ix + 1]
    y0 = yg[iy]; y1 = yg[iy + 1]

    tx = np.where(x1 == x0, 0.0, (x - x0) / (x1 - x0))
    ty = np.where(y1 == y0, 0.0, (y - y0) / (y1 - y0))

    f00 = tab[ix, iy]
    f10 = tab[ix + 1, iy]
    f01 = tab[ix, iy + 1]
    f11 = tab[ix + 1, iy + 1]

    return (1 - tx) * (1 - ty) * f00 + tx * (1 - ty) * f10 + (1 - tx) * ty * f01 + tx * ty * f11


def _bilinear_interp_extrap(x, y_scalar, x_grid, y_grid, table):
    """Bilinear interpolation with linear extrapolation (matching Julia Line()).

    y_scalar is a scalar (temperature), x is a 1D array (wavelengths).
    table shape is (n_x, n_y).
    """
    xg = np.asarray(x_grid)
    yg = np.asarray(y_grid)
    tab = np.asarray(table)
    x = np.asarray(x)
    y = np.asarray(y_scalar)

    # y (temperature) index and weight -- with linear extrapolation
    iy = np.clip(np.searchsorted(yg, y, side="right") - 1, 0, len(yg) - 2)
    y0 = yg[iy]; y1 = yg[iy + 1]
    ty = (y - y0) / (y1 - y0)  # can be <0 or >1 for extrapolation

    # Interpolate along y for each x grid point -> 1D function of x
    vals_at_y = (1 - ty) * tab[:, iy] + ty * tab[:, iy + 1]

    # Now interpolate in x with linear extrapolation
    return _linear_interp_extrap(x, xg, vals_at_y)
