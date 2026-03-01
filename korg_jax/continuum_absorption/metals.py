"""Metal bound-free and positive ion free-free absorption.

Ported from Korg.jl/src/ContinuumAbsorption/absorption_metals_bf.jl,
absorption_ff_positive_ion.jl, and Peach1970.jl.
"""
from __future__ import annotations

import os
import numpy as np
import jax.numpy as jnp

from ..species import Species, Formula
from ..constants import hplanck_eV, Rydberg_eV
from .hydrogenic import hydrogenic_ff_absorption, hydrogenic_ff_absorption_layers


# ==============================================================================
# Peach 1970 departure coefficients
# ==============================================================================
# These are hardcoded from Table III of Peach (1970).  The dictionary maps
# the species that *participates* in the interaction (i.e. the ion) to a
# tuple (T_vals, sigma_vals, table_vals).  sigma is the photon energy in
# units of Rydberg_eV * Zeff^2.
#
# The tables were OCR'd; see the warning in the Julia source.
# ==============================================================================

def _build_peach1970_tables():
    """Return dict mapping Species -> (T_vals, sigma_vals, table_vals) as np arrays."""
    tables = {}

    # --- He II (Table III, neutral Helium) ---
    He_II = Species(Formula.from_Z(2), 1)
    sigma_vals = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    T_vals = np.array([
        10000.0, 11000.0, 12000.0, 13000.0, 14000.0, 15000.0, 16000.0, 17000.0, 18000.0,
        19000.0, 20000.0, 21000.0, 22000.0, 23000.0, 24000.0, 25000.0, 26000.0, 27000.0,
        28000.0, 29000.0, 30000.0, 32000.0, 34000.0, 36000.0, 38000.0, 40000.0, 42000.0,
        44000.0, 46000.0, 48000.0,
    ])
    # shape: (n_T, n_sigma)
    table_vals = np.array([
        [0.016, 0.039, 0.069, 0.100, 0.135, 0.169],
        [0.018, 0.041, 0.071, 0.103, 0.137, 0.172],
        [0.020, 0.043, 0.073, 0.105, 0.139, 0.174],
        [0.022, 0.045, 0.075, 0.107, 0.142, 0.176],
        [0.024, 0.047, 0.078, 0.109, 0.144, 0.179],
        [0.026, 0.050, 0.080, 0.112, 0.146, 0.181],
        [0.028, 0.052, 0.082, 0.114, 0.148, 0.183],
        [0.029, 0.054, 0.084, 0.116, 0.151, 0.185],
        [0.031, 0.056, 0.086, 0.118, 0.153, 0.187],
        [0.033, 0.058, 0.088, 0.120, 0.155, 0.190],
        [0.035, 0.060, 0.090, 0.122, 0.157, 0.192],
        [0.037, 0.062, 0.092, 0.125, 0.159, 0.194],
        [0.039, 0.064, 0.095, 0.127, 0.162, 0.196],
        [0.041, 0.066, 0.097, 0.129, 0.164, 0.198],
        [0.043, 0.068, 0.099, 0.131, 0.166, 0.201],
        [0.045, 0.070, 0.101, 0.133, 0.168, 0.203],
        [0.047, 0.072, 0.103, 0.135, 0.170, 0.205],
        [0.049, 0.074, 0.105, 0.138, 0.173, 0.207],
        [0.050, 0.076, 0.107, 0.140, 0.175, 0.209],
        [0.052, 0.079, 0.109, 0.142, 0.177, 0.211],
        [0.054, 0.081, 0.111, 0.144, 0.179, 0.214],
        [0.058, 0.085, 0.115, 0.148, 0.183, 0.218],
        [0.062, 0.089, 0.119, 0.153, 0.188, 0.222],
        [0.065, 0.093, 0.124, 0.157, 0.102, 0.226],
        [0.069, 0.096, 0.128, 0.161, 0.196, 0.230],
        [0.072, 0.100, 0.132, 0.165, 0.200, 0.235],
        [0.076, 0.104, 0.135, 0.169, 0.204, 0.239],
        [0.079, 0.108, 0.139, 0.173, 0.208, 0.243],
        [0.082, 0.111, 0.143, 0.177, 0.212, 0.247],
        [0.085, 0.115, 0.147, 0.181, 0.216, 0.251],
    ])
    tables[He_II] = (T_vals, sigma_vals, table_vals)

    # --- C II (Table III, neutral Carbon) ---
    C_II = Species(Formula.from_Z(6), 1)
    sigma_vals_C = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    T_vals_C = np.array([
        4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0, 11000.0, 12000.0, 13000.0,
        14000.0, 15000.0, 16000.0, 17000.0, 18000.0, 19000.0, 20000.0, 21000.0, 22000.0,
        23000.0, 24000.0, 25000.0, 26000.0, 27000.0, 28000.0, 29000.0, 30000.0, 32000.0,
        34000.0, 36000.0,
    ])
    table_vals_C = np.array([
        [-0.145, -0.144, -0.068,  0.054,  0.200,  0.394],
        [-0.132, -0.124, -0.045,  0.077,  0.222,  0.415],
        [-0.121, -0.109, -0.027,  0.097,  0.244,  0.438],
        [-0.112, -0.095, -0.010,  0.115,  0.264,  0.461],
        [-0.104, -0.082,  0.005,  0.133,  0.284,  0.484],
        [-0.095, -0.070,  0.020,  0.150,  0.303,  0.507],
        [-0.087, -0.058,  0.034,  0.166,  0.321,  0.529],
        [-0.079, -0.047,  0.048,  0.181,  0.339,  0.550],
        [-0.071, -0.036,  0.061,  0.196,  0.356,  0.570],
        [-0.063, -0.025,  0.074,  0.210,  0.372,  0.590],
        [-0.055, -0.015,  0.086,  0.223,  0.388,  0.609],
        [-0.047, -0.005,  0.098,  0.237,  0.403,  0.628],
        [-0.040,  0.005,  0.109,  0.249,  0.418,  0.646],
        [-0.032,  0.015,  0.120,  0.261,  0.432,  0.664],
        [-0.025,  0.024,  0.131,  0.273,  0.446,  0.680],
        [-0.017,  0.034,  0.141,  0.285,  0.459,  0.697],
        [-0.010,  0.043,  0.152,  0.296,  0.472,  0.713],
        [-0.003,  0.051,  0.161,  0.307,  0.485,  0.728],
        [ 0.004,  0.060,  0.171,  0.317,  0.497,  0.744],
        [ 0.011,  0.069,  0.181,  0.327,  0.509,  0.758],
        [ 0.018,  0.077,  0.100,  0.337,  0.521,  0.773],
        [ 0.025,  0.085,  0.109,  0.347,  0.532,  0.787],
        [ 0.032,  0.093,  0.208,  0.356,  0.543,  0.800],
        [ 0.039,  0.101,  0.216,  0.365,  0.554,  0.814],
        [ 0.046,  0.109,  0.225,  0.374,  0.564,  0.827],
        [ 0.052,  0.117,  0.233,  0.383,  0.574,  0.839],
        [ 0.059,  0.124,  0.241,  0.391,  0.585,  0.852],
        [ 0.072,  0.139,  0.257,  0.408,  0.604,  0.876],
        [ 0.085,  0.154,  0.273,  0.424,  0.623,  0.900],
        [ 0.097,  0.168,  0.288,  0.439,  0.641,  0.923],
    ])
    tables[C_II] = (T_vals_C, sigma_vals_C, table_vals_C)

    # --- Si II (Table III, neutral Silicon) ---
    Si_II = Species(Formula.from_Z(14), 1)
    sigma_vals_Si = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    T_vals_Si = np.array([
        4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0, 11000.0, 12000.0, 13000.0,
        14000.0, 15000.0, 16000.0, 17000.0, 18000.0, 19000.0, 20000.0, 21000.0, 22000.0,
        23000.0, 24000.0, 25000.0, 26000.0, 27000.0, 28000.0, 29000.0, 30000.0, 32000.0,
        34000.0, 36000.0,
    ])
    table_vals_Si = np.array([
        [-0.079,  0.033,  0.214,  0.434,  0.650,  0.973],
        [-0.066,  0.042,  0.216,  0.429,  0.642,  0.062],
        [-0.056,  0.050,  0.220,  0.430,  0.643,  0.965],
        [-0.048,  0.057,  0.224,  0.433,  0.648,  0.974],
        [-0.040,  0.063,  0.229,  0.436,  0.653,  0.081],
        [-0.033,  0.069,  0.233,  0.440,  0.659,  0.995],
        [-0.027,  0.074,  0.238,  0.444,  0.666,  1.007],
        [-0.021,  0.080,  0.242,  0.448,  0.672,  1.019],
        [-0.015,  0.085,  0.246,  0.452,  0.679,  1.031],
        [-0.010,  0.089,  0.250,  0.456,  0.685,  1.042],
        [-0.004,  0.094,  0.254,  0.459,  0.692,  1.054],
        [ 0.001,  0.009,  0.258,  0.463,  0.698,  1.065],
        [ 0.006,  0.103,  0.262,  0.467,  0.705,  1.076],
        [ 0.011,  0.107,  0.265,  0.471,  0.711,  1.087],
        [ 0.016,  0.112,  0.269,  0.474,  0.717,  1.097],
        [ 0.021,  0.116,  0.273,  0.478,  0.724,  1.108],
        [ 0.026,  0.120,  0.277,  0.482,  0.730,  1.118],
        [ 0.030,  0.125,  0.281,  0.486,  0.736,  1.127],
        [ 0.035,  0.129,  0.285,  0.490,  0.742,  1.137],
        [ 0.040,  0.134,  0.289,  0.493,  0.747,  1.146],
        [ 0.045,  0.138,  0.293,  0.497,  0.753,  1.155],
        [ 0.050,  0.143,  0.297,  0.501,  0.759,  1.164],
        [ 0.055,  0.147,  0.301,  0.505,  0.765,  1.173],
        [ 0.060,  0.152,  0.305,  0.509,  0.770,  1.181],
        [ 0.065,  0.156,  0.310,  0.513,  0.776,  1.189],
        [ 0.071,  0.161,  0.314,  0.517,  0.781,  1.197],
        [ 0.076,  0.166,  0.318,  0.520,  0.787,  1.205],
        [ 0.087,  0.176,  0.328,  0.528,  0.798,  1.221],
        [ 0.008,  0.186,  0.317,  0.537,  0.809,  1.236],
        [ 0.109,  0.196,  0.346,  0.545,  0.819,  1.251],
    ])
    tables[Si_II] = (T_vals_Si, sigma_vals_Si, table_vals_Si)

    # --- Mg II (Table III, neutral Magnesium) ---
    Mg_II = Species(Formula.from_Z(12), 1)
    sigma_vals_Mg = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    T_vals_Mg = np.array([
        4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0, 11000.0, 12000.0, 13000.0,
        14000.0, 15000.0, 16000.0, 17000.0, 18000.0, 19000.0, 20000.0, 21000.0, 22000.0,
        23000.0, 24000.0, 25000.0, 26000.0, 27000.0, 28000.0, 29000.0, 30000.0, 32000.0,
        34000.0,
    ])
    table_vals_Mg = np.array([
        [-0.070,  0.008,  0.121,  0.221,  0.274,  0.356],
        [-0.067,  0.003,  0.104,  0.105,  0.244,  0.325],
        [-0.066, -0.002,  0.091,  0.175,  0.221,  0.302],
        [-0.065, -0.007,  0.080,  0.157,  0.201,  0.282],
        [-0.065, -0.012,  0.069,  0.141,  0.183,  0.264],
        [-0.065, -0.016,  0.059,  0.126,  0.166,  0.248],
        [-0.065, -0.020,  0.049,  0.113,  0.151,  0.232],
        [-0.066, -0.024,  0.040,  0.100,  0.137,  0.218],
        [-0.066, -0.028,  0.032,  0.088,  0.124,  0.205],
        [-0.066, -0.032,  0.025,  0.077,  0.112,  0.194],
        [-0.066, -0.035,  0.018,  0.067,  0.101,  0.183],
        [-0.066, -0.037,  0.012,  0.058,  0.091,  0.173],
        [-0.066, -0.040,  0.006,  0.049,  0.082,  0.164],
        [-0.066, -0.042,  0.001,  0.042,  0.074,  0.157],
        [-0.066, -0.044, -0.004,  0.036,  0.067,  0.150],
        [-0.065, -0.045, -0.007,  0.030,  0.061,  0.144],
        [-0.064, -0.046, -0.011,  0.025,  0.056,  0.139],
        [-0.063, -0.047, -0.014,  0.020,  0.051,  0.135],
        [-0.062, -0.048, -0.016,  0.017,  0.048,  0.131],
        [-0.061, -0.048, -0.018,  0.014,  0.045,  0.128],
        [-0.059, -0.047, -0.019,  0.011,  0.042,  0.126],
        [-0.057, -0.047, -0.020,  0.009,  0.040,  0.124],
        [-0.055, -0.046, -0.020,  0.008,  0.039,  0.123],
        [-0.053, -0.045, -0.021,  0.007,  0.038,  0.123],
        [-0.051, -0.044, -0.020,  0.006,  0.038,  0.123],
        [-0.048, -0.042, -0.020,  0.006,  0.038,  0.123],
        [-0.045, -0.040, -0.019,  0.006,  0.039,  0.124],
        [-0.039, -0.035, -0.016,  0.008,  0.042,  0.128],
        [-0.032, -0.030, -0.012,  0.011,  0.046,  0.133],
    ])
    tables[Mg_II] = (T_vals_Mg, sigma_vals_Mg, table_vals_Mg)

    return tables


# Build once at import time
_peach1970_tables = _build_peach1970_tables()


def _peach1970_departure(T, sigma, T_vals, sigma_vals, table_vals):
    """Bilinear interpolation of Peach 1970 departure coefficient D(T, sigma).

    Returns 0 outside the table domain (extrapolation_bc=0 in Julia).

    Parameters
    ----------
    T : JAX scalar -- temperature in K
    sigma : JAX array -- photon energy in units of Rydberg_eV * Zeff^2
    T_vals : 1-D numpy array of temperature knots
    sigma_vals : 1-D numpy array of sigma knots
    table_vals : 2-D numpy array, shape (n_T, n_sigma)

    Returns
    -------
    D : JAX array, same shape as sigma
    """
    T_arr = np.asarray(T_vals)
    s_arr = np.asarray(sigma_vals)
    tab = np.asarray(table_vals)

    sigma = np.asarray(sigma)

    # Determine whether T and sigma are inside the table domain
    T_in = (T >= T_arr[0]) & (T <= T_arr[-1])
    sigma_in = (sigma >= s_arr[0]) & (sigma <= s_arr[-1])

    # Clamp for interpolation (result will be zeroed outside domain)
    T_c = np.clip(T, T_arr[0], T_arr[-1])
    sigma_c = np.clip(sigma, s_arr[0], s_arr[-1])

    # T index
    iT = np.clip(np.searchsorted(T_arr, T_c, side='right') - 1, 0, len(T_arr) - 2)
    T0 = T_arr[iT]
    T1 = T_arr[iT + 1]
    tT = np.where(T1 == T0, 0.0, (T_c - T0) / (T1 - T0))

    # sigma index (vectorized)
    iS = np.clip(np.searchsorted(s_arr, sigma_c, side='right') - 1, 0, len(s_arr) - 2)
    s0 = s_arr[iS]
    s1 = s_arr[iS + 1]
    tS = np.where(s1 == s0, 0.0, (sigma_c - s0) / (s1 - s0))

    # Bilinear interpolation
    f00 = tab[iT, iS]
    f10 = tab[iT + 1, iS]
    f01 = tab[iT, iS + 1]
    f11 = tab[iT + 1, iS + 1]

    D = (1 - tT) * (1 - tS) * f00 + tT * (1 - tS) * f10 + (1 - tT) * tS * f01 + tT * tS * f11

    # Zero outside domain (extrapolation_bc=0)
    D = np.where(T_in & sigma_in, D, 0.0)
    return D


# ==============================================================================
# Positive ion free-free absorption
# ==============================================================================

def positive_ion_ff_absorption(nus, T, number_densities, ne):
    """Free-free absorption from positive ions.

    Iterates over ALL positively charged species in number_densities.
    For species with Peach 1970 departure coefficients (He II, C II, Si II,
    Mg II), applies the correction factor (1 + D(T, sigma)).  For all other
    species, accumulates number densities by charge and uses the uncorrected
    hydrogenic approximation.

    This matches the Julia function positive_ion_ff_absorption! in
    absorption_ff_positive_ion.jl.

    Parameters
    ----------
    nus : frequency array (Hz)
    T : temperature (K)
    number_densities : dict mapping Species -> number density
    ne : electron number density
    """
    nus = np.asarray(nus)
    alpha = np.zeros_like(nus)

    # Accumulate number densities of species WITHOUT departure coefficients,
    # grouped by net charge Z.  This avoids calling hydrogenic_ff_absorption
    # once per species (they share the same cross-section for a given Z).
    ndens_Z1 = 0.0
    ndens_Z2 = 0.0

    for spec, ndens in number_densities.items():
        if not isinstance(spec, Species):
            continue
        if spec.charge <= 0:
            # Neutral and negative species don't participate in ff
            continue

        if spec in _peach1970_tables:
            # Apply Peach 1970 departure correction
            T_vals, sigma_vals, table_vals = _peach1970_tables[spec]
            Z = spec.charge
            # photon energy in Rydberg*Zeff^2: see eq (5) in Peach 1967
            sigma = nus / (Z * Z) * (hplanck_eV / Rydberg_eV)
            D = _peach1970_departure(T, sigma, T_vals, sigma_vals, table_vals)
            alpha = alpha + hydrogenic_ff_absorption(nus, T, Z, ndens, ne) * (1.0 + D)
        else:
            # Accumulate for uncorrected hydrogenic approximation
            if spec.charge == 1:
                ndens_Z1 += ndens
            elif spec.charge == 2:
                ndens_Z2 += ndens
            # Note: Julia errors on charge >= 3; we silently skip here to
            # maintain JAX compatibility (no runtime errors).

    # Add contributions from species using the uncorrected hydrogenic approx
    if ndens_Z1 > 0:
        alpha = alpha + hydrogenic_ff_absorption(nus, T, 1, ndens_Z1, ne)
    if ndens_Z2 > 0:
        alpha = alpha + hydrogenic_ff_absorption(nus, T, 2, ndens_Z2, ne)

    return alpha


# ==============================================================================
# Metal bound-free absorption
# ==============================================================================

def metal_bf_absorption(nus, T, number_densities):
    """Bound-free absorption by metals (TOPBase + NORAD cross-sections).

    Uses precomputed cross-section tables in data/bf_cross-sections/bf_cross-sections.h5.
    The Julia source stores tables with axes (nu, logT) -- i.e. the HDF5 datasets have
    shape (n_nu, n_logT).  We interpolate bilinearly in (nu, logT) space.
    """
    if _metal_bf_data is None:
        return np.zeros_like(np.asarray(nus))

    nus = np.asarray(nus)
    logT = np.log10(T)
    alpha = np.zeros_like(nus)

    H_I = Species(Formula.from_Z(1), 0)
    He_I = Species(Formula.from_Z(2), 0)
    H_II = Species(Formula.from_Z(1), 1)

    for spec, data in _metal_bf_data.items():
        # These are handled with other (analytic) approximations
        if spec in (H_I, He_I, H_II):
            continue
        n_spec = number_densities.get(spec, 0.0)
        if n_spec == 0.0:
            continue

        nu_grid, logT_grid, log_sigma = data
        log_sigma_vals = _bilinear_interp_jax(nus, logT, nu_grid, logT_grid, log_sigma)
        # Mask -inf values that arise from log(0) in the cross-section tables
        mask = np.isfinite(log_sigma_vals)
        alpha = alpha + np.where(
            mask,
            np.exp(np.log(n_spec) + log_sigma_vals) * 1e-18,
            0.0,
        )

    return alpha


# -- Metal bf tables -----------------------------------------------------------

def _load_metal_bf_tables():
    """Load precomputed bound-free cross-section tables from HDF5.

    The Julia source reads the HDF5 file with axes (nu_grid, T_grid), i.e.
    the first axis is frequency and the second is log10(T).  We ensure the
    same convention here.
    """
    try:
        import h5py
    except Exception:
        return None

    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    fname = os.path.join(base, "data", "bf_cross-sections", "bf_cross-sections.h5")
    if not os.path.exists(fname):
        return None

    data = {}
    with h5py.File(fname, "r") as f:
        logT_min = float(f["logT_min"][()])
        logT_step = float(f["logT_step"][()])
        logT_max = float(f["logT_max"][()])
        nu_min = float(f["nu_min"][()])
        nu_step = float(f["nu_step"][()])
        nu_max = float(f["nu_max"][()])

        logT_grid = np.arange(logT_min, logT_max + 0.5 * logT_step, logT_step)
        nu_grid = np.arange(nu_min, nu_max + 0.5 * nu_step, nu_step)

        for name, ds in f["cross-sections"].items():
            spec = Species.from_string(name)
            sigma = ds[()]  # log cross-section table

            # Julia stores (nu, logT). If the HDF5 dataset has shape
            # (len(logT_grid), len(nu_grid)), it was stored transposed.
            if sigma.shape == (len(logT_grid), len(nu_grid)):
                sigma = sigma.T
            data[spec] = (np.asarray(nu_grid), np.asarray(logT_grid), np.asarray(sigma))

    return data


def _bilinear_interp_jax(x, y, x_grid, y_grid, table):
    """Bilinear interpolation with flat extrapolation (Flat() in Julia).

    Parameters
    ----------
    x : JAX array of query points along axis 0 (nu)
    y : JAX scalar query point along axis 1 (logT)
    x_grid : 1-D array of grid knots for axis 0
    y_grid : 1-D array of grid knots for axis 1
    table : 2-D array, shape (len(x_grid), len(y_grid))
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

    x0 = xg[ix]
    x1 = xg[ix + 1]
    y0 = yg[iy]
    y1 = yg[iy + 1]

    tx = np.where(x1 == x0, 0.0, (x - x0) / (x1 - x0))
    ty = np.where(y1 == y0, 0.0, (y - y0) / (y1 - y0))

    f00 = tab[ix, iy]
    f10 = tab[ix + 1, iy]
    f01 = tab[ix, iy + 1]
    f11 = tab[ix + 1, iy + 1]

    return (1 - tx) * (1 - ty) * f00 + tx * (1 - ty) * f10 + (1 - tx) * ty * f01 + tx * ty * f11


try:
    _metal_bf_data = _load_metal_bf_tables()
except Exception:
    _metal_bf_data = None

# JAX versions of the metal bf tables (transferred to device once at import)
if _metal_bf_data is not None:
    _metal_bf_data_jnp = {
        spec: (jnp.asarray(nu_g), jnp.asarray(logT_g), jnp.asarray(log_s))
        for spec, (nu_g, logT_g, log_s) in _metal_bf_data.items()
    }
else:
    _metal_bf_data_jnp = None


# ==============================================================================
# Batch (layers) versions
# ==============================================================================

def _bilinear_interp_layers(x, y, x_grid, y_grid, table):
    """Bilinear interp (flat extrap): x=(n_freq,), y=(n_layers,) → (n_layers, n_freq). JAX."""
    xg = jnp.asarray(x_grid); yg = jnp.asarray(y_grid); tab = jnp.asarray(table)
    x = jnp.clip(jnp.asarray(x), xg[0], xg[-1])   # (n_freq,)
    y = jnp.clip(jnp.asarray(y), yg[0], yg[-1])   # (n_layers,)

    ix = jnp.clip(jnp.searchsorted(xg, x, side="right") - 1, 0, len(xg) - 2)  # (n_freq,)
    iy = jnp.clip(jnp.searchsorted(yg, y, side="right") - 1, 0, len(yg) - 2)  # (n_layers,)

    tx = jnp.where(xg[ix + 1] == xg[ix], 0.0, (x - xg[ix]) / (xg[ix + 1] - xg[ix]))
    ty = jnp.where(yg[iy + 1] == yg[iy], 0.0, (y - yg[iy]) / (yg[iy + 1] - yg[iy]))

    ix_ = ix[None, :]; iy_ = iy[:, None]
    tx_ = tx[None, :]; ty_ = ty[:, None]

    f00 = tab[ix_,     iy_    ]; f10 = tab[ix_ + 1, iy_    ]
    f01 = tab[ix_,     iy_ + 1]; f11 = tab[ix_ + 1, iy_ + 1]
    return (1 - tx_) * (1 - ty_) * f00 + tx_ * (1 - ty_) * f10 \
         + (1 - tx_) * ty_ * f01 + tx_ * ty_ * f11


def _peach1970_departure_layers(T, sigma, T_vals, sigma_vals, table_vals):
    """Peach 1970 departure coefficient: T=(n_layers,), sigma=(n_freq,) → (n_layers, n_freq). JAX."""
    T_arr = jnp.asarray(T_vals); s_arr = jnp.asarray(sigma_vals); tab = jnp.asarray(table_vals)
    T     = jnp.asarray(T);     sigma = jnp.asarray(sigma)

    T_in     = (T     >= T_arr[0]) & (T     <= T_arr[-1])   # (n_layers,)
    sigma_in = (sigma >= s_arr[0]) & (sigma <= s_arr[-1])   # (n_freq,)

    T_c     = jnp.clip(T,     T_arr[0], T_arr[-1])
    sigma_c = jnp.clip(sigma, s_arr[0], s_arr[-1])

    iT = jnp.clip(jnp.searchsorted(T_arr, T_c,     side="right") - 1, 0, len(T_arr) - 2)
    iS = jnp.clip(jnp.searchsorted(s_arr, sigma_c, side="right") - 1, 0, len(s_arr) - 2)

    tT = jnp.where(T_arr[iT + 1] == T_arr[iT], 0.0,
                   (T_c - T_arr[iT]) / (T_arr[iT + 1] - T_arr[iT]))   # (n_layers,)
    tS = jnp.where(s_arr[iS + 1] == s_arr[iS], 0.0,
                   (sigma_c - s_arr[iS]) / (s_arr[iS + 1] - s_arr[iS]))   # (n_freq,)

    iT_ = iT[:, None]; tT_ = tT[:, None]   # (n_layers, 1)
    iS_ = iS[None, :]; tS_ = tS[None, :]   # (1, n_freq)

    f00 = tab[iT_,     iS_    ]; f10 = tab[iT_ + 1, iS_    ]
    f01 = tab[iT_,     iS_ + 1]; f11 = tab[iT_ + 1, iS_ + 1]

    D = (1 - tT_) * (1 - tS_) * f00 + tT_ * (1 - tS_) * f10 \
      + (1 - tT_) * tS_ * f01 + tT_ * tS_ * f11

    return jnp.where(T_in[:, None] & sigma_in[None, :], D, 0.0)


def positive_ion_ff_absorption_layers(nus, T, number_densities, ne):
    """Batch positive-ion ff: T/ne are (n_layers,), densities are arrays. Returns (n_layers, n_freq)."""
    nus = jnp.asarray(nus)
    T   = jnp.asarray(T, dtype=jnp.float64)
    ne  = jnp.asarray(ne, dtype=jnp.float64)
    n_layers = len(T)

    alpha    = jnp.zeros((n_layers, len(nus)))
    ndens_Z1 = np.zeros(n_layers)   # numpy accumulators (small, not on GPU)
    ndens_Z2 = np.zeros(n_layers)

    for spec, ndens in number_densities.items():
        if not isinstance(spec, Species):
            continue
        if spec.charge <= 0:
            continue

        ndens = np.asarray(ndens, dtype=np.float64)   # (n_layers,)

        if spec in _peach1970_tables:
            T_vals, sigma_vals, table_vals = _peach1970_tables[spec]
            Z = spec.charge
            sigma = nus / (Z * Z) * (hplanck_eV / Rydberg_eV)   # (n_freq,)
            D = _peach1970_departure_layers(T, sigma, T_vals, sigma_vals, table_vals)
            alpha = alpha + hydrogenic_ff_absorption_layers(nus, T, Z, ndens, ne) * (1.0 + D)
        else:
            if spec.charge == 1:
                ndens_Z1 += ndens
            elif spec.charge == 2:
                ndens_Z2 += ndens

    if np.any(ndens_Z1 > 0):
        alpha = alpha + hydrogenic_ff_absorption_layers(nus, T, 1, ndens_Z1, ne)
    if np.any(ndens_Z2 > 0):
        alpha = alpha + hydrogenic_ff_absorption_layers(nus, T, 2, ndens_Z2, ne)

    return alpha


_H_I_metals  = Species(Formula.from_Z(1), 0)
_He_I_metals = Species(Formula.from_Z(2), 0)
_H_II_metals = Species(Formula.from_Z(1), 1)


def metal_bf_absorption_layers(nus, T, number_densities):
    """Batch metal bf: T is (n_layers,), densities are arrays. Returns (n_layers, n_freq)."""
    if _metal_bf_data_jnp is None:
        return jnp.zeros((len(T), len(nus)))

    nus  = jnp.asarray(nus)
    T    = jnp.asarray(T, dtype=jnp.float64)
    logT = jnp.log10(T)   # (n_layers,)
    alpha = jnp.zeros((len(T), len(nus)))

    for spec, data in _metal_bf_data_jnp.items():
        if spec in (_H_I_metals, _He_I_metals, _H_II_metals):
            continue
        n_spec = number_densities.get(spec, None)
        if n_spec is None:
            continue
        n_spec = jnp.asarray(n_spec, dtype=jnp.float64)   # (n_layers,)
        if jnp.all(n_spec == 0.0):
            continue

        nu_grid, logT_grid, log_sigma = data
        log_sigma_vals = _bilinear_interp_layers(nus, logT, nu_grid, logT_grid, log_sigma)
        mask = jnp.isfinite(log_sigma_vals)
        safe_log_n = jnp.where(n_spec > 0, jnp.log(jnp.maximum(n_spec, 1e-300)),
                                jnp.full_like(n_spec, -jnp.inf))
        alpha = alpha + jnp.where(
            mask & (n_spec[:, None] > 0),
            jnp.exp(safe_log_n[:, None] + log_sigma_vals) * 1e-18,
            0.0,
        )

    return alpha
