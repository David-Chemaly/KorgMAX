"""Line opacity computation (chunked vmap approach).

Ported from Korg.jl/src/line_absorption.jl.

Processes lines in chunks of ~500 via jax.vmap, accumulates into the
absorption array with jax.lax.scan over chunks.
"""
from __future__ import annotations

import math
import jax
import jax.numpy as jnp

from .constants import (
    c_cgs, kboltz_cgs, kboltz_eV, hplanck_cgs, hplanck_eV,
    electron_mass_cgs, electron_charge_cgs, amu_cgs,
)
from .voigt import voigt_hjerting

# Precompute log of gamma function values for ABO broadening.
# gamma((4-alpha)/2) for alpha in typical range [0.2, 0.4].
# We use jax.scipy for the JAX-compatible version.
from jax.scipy.special import gammaln as _gammaln

# Default chunk size for processing lines
CHUNK_SIZE = 500


@jax.jit
def _doppler_width(wl, T, mass, xi):
    """Doppler width sigma (NOT sqrt(2)*sigma) in cm.

    Matches Julia: doppler_width(λ₀, T, m, ξ) = λ₀ * sqrt(kT/m + ξ²/2) / c
    where ξ is microturbulent velocity in cm/s.
    """
    return wl * jnp.sqrt(kboltz_cgs * T / mass + (xi ** 2) / 2.0) / c_cgs


@jax.jit
def _scaled_vdW(vdW_1, vdW_2, mass, T):
    """Compute the per-perturber vdW broadening rate (gamma / n_H_I).

    Matches Julia: scaled_vdW(vdW, m, T)
    Two modes based on vdW_2:
      vdW_2 < 0: simple scaling, vdW_1 * (T/10000)^0.3
      vdW_2 >= 0: ABO theory with reduced mass, gamma function, v0 normalization
    """
    is_gamma_form = vdW_2 < 0.0

    # Simple form: scale with T^0.3
    gamma_simple = vdW_1 * (T / 10000.0) ** 0.3

    # ABO form: 2 * (4/pi)^(alpha/2) * Gamma((4-alpha)/2) * v0 * sigma * (vbar/v0)^(1-alpha)
    v0 = 1e6  # sigma is given at 10000 m/s = 1e6 cm/s
    sigma = vdW_1
    alpha = vdW_2

    # Inverse reduced mass: 1/(1.008*amu) + 1/mass
    inv_mu = 1.0 / (1.008 * amu_cgs) + 1.0 / mass
    vbar = jnp.sqrt(8.0 * kboltz_cgs * T / jnp.pi * inv_mu)

    # gamma function via exp(gammaln) for JAX compatibility
    gamma_val = jnp.exp(_gammaln((4.0 - alpha) / 2.0))

    gamma_abo = jnp.where(
        vdW_1 > 0,
        2.0 * (4.0 / jnp.pi) ** (alpha / 2.0) * gamma_val * v0 * sigma * (vbar / v0) ** (1.0 - alpha),
        0.0,
    )

    return jnp.where(is_gamma_form, gamma_simple, gamma_abo)


@jax.jit
def _lorentz_width(gamma_rad, gamma_stark, vdW_1, vdW_2, mass, T, ne, nH_I, is_molecule):
    """Total Lorentzian FWHM (Gamma) in angular frequency (s^-1).

    Combines radiative, Stark, and van der Waals broadening.
    Matches Julia line_absorption.jl lines 76-80.
    """
    # Radiative: gamma_rad is already FWHM in s^-1
    gamma = gamma_rad

    # Stark + vdW apply only to atoms in Julia
    atomic_mask = jnp.where(is_molecule, 0.0, 1.0)

    # Stark: gamma_stark is per-perturber at 10000 K, scale with T and ne
    gamma = gamma + atomic_mask * gamma_stark * ne * (T / 10000.0) ** (1.0 / 6.0)

    # van der Waals: scaled_vdW returns per-perturber rate, multiply by nH_I
    gamma = gamma + atomic_mask * _scaled_vdW(vdW_1, vdW_2, mass, T) * nH_I

    return gamma


@jax.jit
def _inverse_gaussian_density(rho, sigma):
    """Inverse of a centered Gaussian PDF with stddev sigma."""
    sigma = jnp.maximum(sigma, 1e-300)
    max_rho = 1.0 / (jnp.sqrt(2.0 * jnp.pi) * sigma)
    return jnp.where(
        rho > max_rho,
        0.0,
        sigma * jnp.sqrt(-2.0 * jnp.log(jnp.sqrt(2.0 * jnp.pi) * sigma * rho)),
    )


@jax.jit
def _inverse_lorentz_density(rho, gamma):
    """Inverse of a centered Lorentz PDF with HWHM gamma."""
    gamma = jnp.maximum(gamma, 1e-300)
    max_rho = 1.0 / (jnp.pi * gamma)
    return jnp.where(
        rho > max_rho,
        0.0,
        jnp.sqrt(gamma / (jnp.pi * rho) - gamma * gamma),
    )


@jax.jit
def _line_absorption_single(wl_grid, line_wl, log_gf, E_lower, mass,
                             gamma_rad, gamma_stark, vdW_1, vdW_2,
                             T, ne, nH_I, n_absorber, xi,
                             is_molecule, alpha_cntm_line, cutoff_threshold):
    """Compute absorption profile for a single line on the wavelength grid.

    Matches Julia line_absorption.jl line_profile() and the per-line loop.
    Returns alpha contribution array of same shape as wl_grid.
    """
    valid = line_wl > 0
    line_wl = jnp.where(valid, line_wl, 1.0)
    n_absorber = jnp.where(valid, n_absorber, 0.0)

    beta = 1.0 / (kboltz_eV * T)

    # Doppler width sigma (NOT sqrt(2)*sigma)
    sigma = _doppler_width(line_wl, T, mass, xi)

    # Lorentz width Gamma (FWHM in angular frequency)
    Gamma = _lorentz_width(gamma_rad, gamma_stark, vdW_1, vdW_2, mass, T, ne, nH_I, is_molecule)

    # Lorentz HWHM in wavelength units: gamma = Gamma * lambda^2 / (c * 4*pi)
    # Factor breakdown: lambda^2/c is |dlambda/dnu|, 1/(2*pi) for angular->cyclical,
    # 1/2 for FWHM->HWHM
    gamma = Gamma * line_wl ** 2 / (c_cgs * 4.0 * jnp.pi)

    # Line cross-section: sigma_line(wl) = pi*e^2/(m_e*c) * wl^2/c
    # The wl^2/c factor converts from frequency to wavelength space
    sigma_line = (jnp.pi * electron_charge_cgs ** 2
                  / (electron_mass_cgs * c_cgs)
                  * line_wl ** 2 / c_cgs)

    # Levels factor: exp(-beta*E_lower) - exp(-beta*E_upper)
    # = exp(-beta*E_lower) * (1 - exp(-h*nu/kT))
    E_upper = E_lower + c_cgs * hplanck_eV / line_wl
    levels_factor = jnp.exp(-beta * E_lower) - jnp.exp(-beta * E_upper)

    # Total wavelength-integrated absorption coefficient
    # amplitude = 10^log_gf * sigma_line * levels_factor * n_absorber
    # where n_absorber = n(species) / U(species) (number density / partition function)
    amplitude = 10.0 ** log_gf * sigma_line * levels_factor * n_absorber

    # Voigt profile: H(a, v) / (sigma * sqrt(2*pi))
    # where a = gamma / (sigma * sqrt(2)), v = |lambda - lambda0| / (sigma * sqrt(2))
    inv_sigma_sqrt2 = 1.0 / (sigma * jnp.sqrt(2.0))
    a = gamma * inv_sigma_sqrt2
    v = jnp.abs(wl_grid - line_wl) * inv_sigma_sqrt2
    H = voigt_hjerting(a, v)

    # Scaling: amplitude / (sigma * sqrt(2) * sqrt(pi)) = amplitude * inv_sigma_sqrt2 / sqrt(pi)
    scaling = inv_sigma_sqrt2 / jnp.sqrt(jnp.pi) * amplitude

    # Apply window cutoff if alpha_cntm_line provided (Julia behavior)
    safe_amp = jnp.maximum(amplitude, 1e-300)
    rho_crit = alpha_cntm_line * cutoff_threshold / safe_amp
    doppler_window = _inverse_gaussian_density(rho_crit, sigma)
    lorentz_window = _inverse_lorentz_density(rho_crit, gamma)
    window_size = jnp.sqrt(doppler_window * doppler_window + lorentz_window * lorentz_window)
    mask = jnp.abs(wl_grid - line_wl) <= window_size

    return scaling * H * mask * valid


@jax.jit
def _line_absorption_chunks(wl_grid, wl_c, lgf_c, el_c, gr_c, gs_c, v1_c, v2_c, m_c,
                            mol_c, na_c, alpha_c,
                            T, ne, nH_I, xi, cutoff_threshold):
    """Compute line absorption with pre-chunked arrays for one layer."""
    n_chunks = wl_c.shape[0]
    n_wl = wl_grid.shape[0]

    # Process one chunk: vmap over lines within the chunk, sum contributions
    def process_chunk(chunk_data):
        (wl_ch, lgf_ch, el_ch, gr_ch, gs_ch, v1_ch, v2_ch, m_ch, na_ch, mol_ch, ac_ch) = chunk_data

        def single_line(lw, lg, el, gr, gs, v1, v2, m, na, mol, ac):
            return _line_absorption_single(
                wl_grid, lw, lg, el, m, gr, gs, v1, v2, T, ne, nH_I, na, xi,
                mol, ac, cutoff_threshold
            )

        alphas = jax.vmap(single_line)(
            wl_ch, lgf_ch, el_ch, gr_ch, gs_ch, v1_ch, v2_ch, m_ch, na_ch, mol_ch, ac_ch
        )
        return jnp.sum(alphas, axis=0)  # sum over lines in chunk

    # Scan over chunks, accumulating alpha
    def scan_fn(alpha_acc, chunk_idx):
        chunk_data = (wl_c[chunk_idx], lgf_c[chunk_idx], el_c[chunk_idx],
                      gr_c[chunk_idx], gs_c[chunk_idx], v1_c[chunk_idx],
                      v2_c[chunk_idx], m_c[chunk_idx], na_c[chunk_idx],
                      mol_c[chunk_idx], alpha_c[chunk_idx])
        alpha_chunk = process_chunk(chunk_data)
        return alpha_acc + alpha_chunk, None

    alpha, _ = jax.lax.scan(scan_fn, jnp.zeros(n_wl), jnp.arange(n_chunks))
    return alpha


def build_linelist_chunks(linelist_jax, chunk_size=CHUNK_SIZE):
    """Pre-chunk and pad a linelist for repeated fast absorption calls."""
    n_lines = len(linelist_jax['wl'])
    n_chunks = math.ceil(n_lines / chunk_size)
    n_padded = n_chunks * chunk_size
    pad = n_padded - n_lines

    def pad_arr(a):
        return jnp.concatenate([a, jnp.zeros(pad, dtype=a.dtype)])

    wl = pad_arr(linelist_jax['wl'])
    log_gf = pad_arr(linelist_jax['log_gf'])
    E_lower = pad_arr(linelist_jax['E_lower'])
    gamma_rad = pad_arr(linelist_jax['gamma_rad'])
    gamma_stark = pad_arr(linelist_jax['gamma_stark'])
    vdW_1 = pad_arr(linelist_jax['vdW_1'])
    vdW_2 = pad_arr(linelist_jax['vdW_2'])
    mass = pad_arr(linelist_jax['mass'])
    is_molecule = pad_arr(linelist_jax['is_molecule']).astype(jnp.bool_)

    def reshape(a):
        return a.reshape(n_chunks, chunk_size)

    return {
        "n_lines": n_lines,
        "chunk_size": chunk_size,
        "n_chunks": n_chunks,
        "pad": pad,
        "wl_c": reshape(wl),
        "lgf_c": reshape(log_gf),
        "el_c": reshape(E_lower),
        "gr_c": reshape(gamma_rad),
        "gs_c": reshape(gamma_stark),
        "v1_c": reshape(vdW_1),
        "v2_c": reshape(vdW_2),
        "m_c": reshape(mass),
        "mol_c": reshape(is_molecule),
    }


def line_absorption_with_chunks(wl_grid, chunks, T, ne, nH_I, n_absorbers, xi,
                                alpha_cntm_at_line=None, cutoff_threshold=3e-4):
    """Line absorption using pre-chunked linelist for speed."""
    n_lines = chunks["n_lines"]
    pad = chunks["pad"]
    chunk_size = chunks["chunk_size"]
    n_chunks = chunks["n_chunks"]

    n_abs = jnp.concatenate([n_absorbers, jnp.zeros(pad)])
    if alpha_cntm_at_line is None:
        alpha_cntm_at_line = jnp.zeros(n_lines)
    alpha_cntm_at_line = jnp.asarray(alpha_cntm_at_line)
    alpha_cntm = jnp.concatenate([alpha_cntm_at_line, jnp.zeros(pad)])

    def reshape(a):
        return a.reshape(n_chunks, chunk_size)

    na_c = reshape(n_abs)
    alpha_c = reshape(alpha_cntm)

    return _line_absorption_chunks(
        jnp.asarray(wl_grid),
        chunks["wl_c"], chunks["lgf_c"], chunks["el_c"], chunks["gr_c"], chunks["gs_c"],
        chunks["v1_c"], chunks["v2_c"], chunks["m_c"], chunks["mol_c"],
        na_c, alpha_c,
        T, ne, nH_I, xi, cutoff_threshold
    )


def line_absorption(wl_grid, linelist_jax, T, ne, nH_I, n_absorbers, xi,
                    alpha_cntm_at_line=None, cutoff_threshold=3e-4,
                    chunk_size=CHUNK_SIZE):
    """Compute total line absorption coefficient on a wavelength grid.

    Parameters
    ----------
    wl_grid : (n_wl,) JAX array of wavelengths in cm.
    linelist_jax : dict with keys 'wl', 'log_gf', 'species_idx', 'E_lower',
        'gamma_rad', 'gamma_stark', 'vdW_1', 'vdW_2', 'mass', 'is_molecule'
        — all (n_lines,).
    T : scalar temperature (K).
    ne : scalar electron number density.
    nH_I : scalar neutral hydrogen number density.
    n_absorbers : (n_lines,) number density of the absorbing species
        (n(species) / U(species), i.e. number density divided by partition fn).
    xi : scalar microturbulent velocity in cm/s (NOT km/s).
    alpha_cntm_at_line : (n_lines,) continuum opacity at each line center.
    cutoff_threshold : threshold for line window cutoff (Julia default: 3e-4).
    chunk_size : number of lines per chunk.

    Returns
    -------
    alpha : (n_wl,) total line absorption coefficient.
    """
    chunks = build_linelist_chunks(linelist_jax, chunk_size=chunk_size)
    return line_absorption_with_chunks(
        wl_grid, chunks, T, ne, nH_I, n_absorbers, xi,
        alpha_cntm_at_line=alpha_cntm_at_line, cutoff_threshold=cutoff_threshold
    )


def line_absorption_layers(wl_grid, linelist_jax, T, ne, nH_I, n_absorbers, xi,
                           alpha_cntm_at_line=None, cutoff_threshold=3e-4,
                           chunk_size=CHUNK_SIZE, chunks=None):
    """Vectorized line absorption across layers.

    T, ne, nH_I: (n_layers,)
    n_absorbers: (n_layers, n_lines)
    xi: scalar or (n_layers,)
    alpha_cntm_at_line: (n_layers, n_lines) or None
    """
    if chunks is None:
        chunks = build_linelist_chunks(linelist_jax, chunk_size=chunk_size)
    n_lines = chunks["n_lines"]
    pad = chunks["pad"]
    chunk_size = chunks["chunk_size"]
    n_chunks = chunks["n_chunks"]

    n_abs = jnp.concatenate([n_absorbers, jnp.zeros((n_absorbers.shape[0], pad))], axis=1)
    if alpha_cntm_at_line is None:
        alpha_cntm_at_line = jnp.zeros((n_absorbers.shape[0], n_lines))
    alpha_cntm = jnp.concatenate([alpha_cntm_at_line, jnp.zeros((n_absorbers.shape[0], pad))], axis=1)

    # Reshape per-layer inputs to (n_layers, n_chunks, chunk_size)
    def reshape_layer(a):
        return a.reshape(a.shape[0], n_chunks, chunk_size)

    na_c = reshape_layer(n_abs)
    alpha_c = reshape_layer(alpha_cntm)

    T = jnp.asarray(T)
    ne = jnp.asarray(ne)
    nH_I = jnp.asarray(nH_I)
    xi = jnp.asarray(xi) if jnp.ndim(xi) > 0 else jnp.full_like(T, xi)

    def per_layer(t, ne_i, nH_i, xi_i, na_i, ac_i):
        return _line_absorption_chunks(
            jnp.asarray(wl_grid),
            chunks["wl_c"], chunks["lgf_c"], chunks["el_c"], chunks["gr_c"], chunks["gs_c"],
            chunks["v1_c"], chunks["v2_c"], chunks["m_c"], chunks["mol_c"],
            na_i, ac_i,
            t, ne_i, nH_i, xi_i, cutoff_threshold
        )

    return jax.vmap(per_layer)(T, ne, nH_I, xi, na_c, alpha_c)
