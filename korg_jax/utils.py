"""Utility functions: blackbody, LSF, rotation, and wavelength conversions.

Ported from Korg.jl/src/utils.jl.
"""
from __future__ import annotations

import math
import numpy as np
import jax.numpy as jnp

from .constants import c_cgs, hplanck_cgs, kboltz_cgs, amu_cgs
from .wavelengths import Wavelengths


# ── Basic PDFs ───────────────────────────────────────────────────────────────

def normal_pdf(delta, sigma):
    return jnp.exp(-0.5 * (delta / sigma) ** 2) / (jnp.sqrt(2.0 * jnp.pi) * sigma)


# ── Bounds helpers ──────────────────────────────────────────────────────────

def merge_bounds(bounds, merge_distance=0.0):
    """Sort and merge overlapping (lo, hi) ranges.

    Returns (merged_bounds, index_groups).
    """
    if len(bounds) == 0:
        return [], []

    indices = list(range(len(bounds)))
    s = sorted(indices, key=lambda i: bounds[i][0])
    bounds = [bounds[i] for i in s]
    indices = [indices[i] for i in s]

    new_bounds = [bounds[0]]
    groups = [[indices[0]]]
    for i in range(1, len(bounds)):
        lo, hi = bounds[i]
        prev_lo, prev_hi = new_bounds[-1]
        if lo <= prev_hi + merge_distance:
            new_bounds[-1] = (prev_lo, max(prev_hi, hi))
            groups[-1].append(indices[i])
        else:
            new_bounds.append((lo, hi))
            groups.append([indices[i]])
    return new_bounds, groups


# ── LSF convolution ─────────────────────────────────────────────────────────

def _resolve_R(R, lam0):
    return R(lam0 * 1e8) if callable(R) else R


def _lsf_bounds_and_kernel(wls: Wavelengths, lam0, R, window_size):
    R_val = _resolve_R(R, lam0)
    sigma = lam0 / R_val / (2.0 * math.sqrt(2.0 * math.log(2.0)))

    lb = np.searchsorted(wls.all_wls, lam0 - window_size * sigma, side="left")
    ub = np.searchsorted(wls.all_wls, lam0 + window_size * sigma, side="right")

    phi = normal_pdf(jnp.asarray(wls.all_wls[lb:ub]) - lam0, sigma)
    phi = phi / jnp.sum(phi)
    return lb, ub, np.asarray(phi)


def apply_LSF(flux, wls, R, window_size=4):
    """Apply a Gaussian LSF to a spectrum."""
    if R == math.inf:
        return np.array(flux, copy=True)

    wls = Wavelengths.from_array(wls) if not isinstance(wls, Wavelengths) else wls
    mat = compute_LSF_matrix(wls, wls.all_wls, R, window_size, verbose=False)
    return mat @ np.asarray(flux)


def compute_LSF_matrix(synth_wls, obs_wls, R, window_size=4, verbose=True):
    """Return a dense matrix applying the LSF and resampling.

    For a fixed (scalar) R the matrix is built with pure NumPy broadcasting
    in O(n_obs * n_synth) without any Python loop.  For a callable R the
    per-row loop is retained.
    """
    obs_wls = np.asarray(obs_wls)
    if obs_wls[0] >= 1:
        obs_wls = obs_wls * 1e-8

    synth_wls = Wavelengths.from_array(synth_wls) if not isinstance(synth_wls, Wavelengths) else synth_wls
    synth_arr = synth_wls.all_wls  # (n_synth,)

    if callable(R):
        # Callable R: fall back to per-row loop (R varies per wavelength)
        mat = np.zeros((len(obs_wls), len(synth_arr)))
        for i, lam0 in enumerate(obs_wls):
            lb, ub, kernel = _lsf_bounds_and_kernel(synth_wls, lam0, R, window_size)
            mat[i, lb:ub] += kernel
        return mat

    # Fixed R: fully vectorised NumPy broadcast (no Python loop)
    FWHM_factor = 2.0 * math.sqrt(2.0 * math.log(2.0))
    sigma = obs_wls / R / FWHM_factor            # (n_obs,)
    half_win = window_size * sigma               # (n_obs,)

    # delta[i, j] = synth_arr[j] - obs_wls[i]
    delta = synth_arr[np.newaxis, :] - obs_wls[:, np.newaxis]   # (n_obs, n_synth)

    mask = np.abs(delta) <= half_win[:, np.newaxis]
    phi = np.where(mask, np.exp(-0.5 * (delta / sigma[:, np.newaxis]) ** 2), 0.0)
    row_sums = phi.sum(axis=1, keepdims=True)
    phi = np.where(row_sums > 0, phi / row_sums, 0.0)
    return phi


# ── Rotation ────────────────────────────────────────────────────────────────

def _rotation_kernel_integral(c1, c2, c3, detuning, dlam_rot):
    if abs(detuning) == dlam_rot:
        return math.copysign(0.5, detuning)
    return (0.5 * c1 * detuning * math.sqrt(1 - detuning ** 2 / dlam_rot ** 2)
            + 0.5 * c1 * dlam_rot * math.asin(detuning / dlam_rot)
            + c2 * (detuning - detuning ** 3 / (3 * dlam_rot ** 2))) / c3


def _apply_rotation_core(flux, wls, vsini, eps=0.6):
    if vsini == 0:
        return np.array(flux, copy=True)

    wls = np.asarray(wls)
    if wls[0] > 1:
        wls = wls * 1e-8

    vsini = vsini * 1e5
    new_flux = np.zeros_like(flux)

    c1 = 2.0 * (1 - eps)
    c2 = math.pi * eps / 2.0
    c3 = math.pi * (1 - eps / 3.0)

    step = wls[1] - wls[0]
    for i in range(len(flux)):
        dlam_rot = wls[i] * vsini / c_cgs
        lb = np.searchsorted(wls, wls[i] - dlam_rot, side="left")
        ub = np.searchsorted(wls, wls[i] + dlam_rot, side="right")
        window = flux[lb:ub]

        detunings = np.concatenate([
            [-dlam_rot],
            (np.arange(lb - i + 0.5, ub - i - 0.5 + 1e-12) * step),
            [dlam_rot],
        ])

        ks = np.array([_rotation_kernel_integral(c1, c2, c3 * dlam_rot, d, dlam_rot) for d in detunings])
        new_flux[i] = np.sum(ks[1:] * window) - np.sum(ks[:-1] * window)

    return new_flux


def apply_rotation(flux, wls, vsini, eps=0.6):
    wls = Wavelengths.from_array(wls) if not isinstance(wls, Wavelengths) else wls
    new_flux = np.zeros_like(flux)

    idx = 0
    for r in wls.wl_ranges:
        n = len(r)
        new_flux[idx:idx + n] = _apply_rotation_core(flux[idx:idx + n], r, vsini, eps)
        idx += n

    return new_flux


# ── Air/vacuum conversions ──────────────────────────────────────────────────

def air_to_vacuum(lam, cgs=None):
    lam = float(lam)
    if cgs is None:
        cgs = lam < 1
    if cgs:
        lam *= 1e8
    s = 1e4 / lam
    n = (1.0
         + 0.00008336624212083
         + 0.02408926869968 / (130.1065924522 - s ** 2)
         + 0.0001599740894897 / (38.92568793293 - s ** 2))
    lam_vac = lam * n
    if cgs:
        lam_vac *= 1e-8
    return lam_vac


def vacuum_to_air(lam, cgs=None):
    lam = float(lam)
    if cgs is None:
        cgs = lam < 1
    if cgs:
        lam *= 1e8
    s = 1e4 / lam
    n = (1.0
         + 0.0000834254
         + 0.02406147 / (130 - s ** 2)
         + 0.00015998 / (38.9 - s ** 2))
    lam_air = lam / n
    if cgs:
        lam_air *= 1e-8
    return lam_air


# ── Interval helpers ────────────────────────────────────────────────────────

def _nextfloat_skipsubnorm(v):
    v = float(v)
    fmin = np.finfo(float).tiny
    if -fmin <= v < 0:
        return 0.0
    if 0 <= v < fmin:
        return fmin
    return np.nextafter(v, np.inf)


def _prevfloat_skipsubnorm(v):
    v = float(v)
    fmin = np.finfo(float).tiny
    if -fmin < v <= 0:
        return -fmin
    if 0 < v <= fmin:
        return 0.0
    return np.nextafter(v, -np.inf)


class Interval:
    """Exclusive interval (lower, upper)."""

    def __init__(self, lower, upper, exclusive_lower=True, exclusive_upper=True):
        if not lower < upper:
            raise ValueError("upper bound must exceed lower bound")
        lower = float(lower)
        upper = float(upper)
        self.lower = lower if (exclusive_lower or math.isinf(lower)) else _prevfloat_skipsubnorm(lower)
        self.upper = upper if (exclusive_upper or math.isinf(upper)) else _nextfloat_skipsubnorm(upper)


def closed_interval(lo, up):
    return Interval(lo, up, exclusive_lower=False, exclusive_upper=False)


def contained(value, interval: Interval):
    return interval.lower < value < interval.upper


def contained_slice(vals, interval: Interval):
    vals = np.asarray(vals)
    lb = np.searchsorted(vals, interval.lower, side="left")
    ub = np.searchsorted(vals, interval.upper, side="right")
    return slice(lb, ub)


def _convert_lambda_endpoint(lam_endpoint, lam_lower_bound):
    if lam_lower_bound:
        inbound, outbound, rel = np.nextafter, np.nextafter, lambda a, b: a > b
        inbound_dir = -np.inf
        outbound_dir = np.inf
    else:
        inbound, outbound, rel = np.nextafter, np.nextafter, lambda a, b: a < b
        inbound_dir = np.inf
        outbound_dir = -np.inf

    nu_endpoint = math.inf if lam_endpoint == 0 else c_cgs / lam_endpoint

    if math.isfinite(nu_endpoint) and nu_endpoint != 0:
        while not rel(c_cgs / inbound(nu_endpoint, inbound_dir), lam_endpoint):
            nu_endpoint = inbound(nu_endpoint, inbound_dir)
        while rel(c_cgs / nu_endpoint, lam_endpoint):
            nu_endpoint = outbound(nu_endpoint, outbound_dir)
    return nu_endpoint


def lambda_to_nu_bound(lam_bound: Interval):
    return Interval(
        _convert_lambda_endpoint(lam_bound.upper, False),
        _convert_lambda_endpoint(lam_bound.lower, True),
    )


# ── Blackbody ───────────────────────────────────────────────────────────────

def blackbody(T, lam):
    h = hplanck_cgs
    c = c_cgs
    k = kboltz_cgs
    return 2 * h * c ** 2 / lam ** 5 / (jnp.expm1(h * c / lam / k / T))
