"""Hydrogenic free-free absorption with Gaunt factor interpolation.

Ported from Korg.jl/src/ContinuumAbsorption/hydrogenic_bf_ff.jl.
"""
from __future__ import annotations

import os
import math
import numpy as np
import jax.numpy as jnp

from ..constants import (
    hplanck_eV, hplanck_cgs, kboltz_eV, kboltz_cgs, c_cgs, Rydberg_eV,
)


# ── Gaunt factor table (loaded once at import time) ─────────────────────────

def _load_gaunt_table(fname=None):
    """Load van Hoof et al. (2014) non-relativistic free-free Gaunt factors."""
    if fname is None:
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        fname = os.path.join(base, "data", "vanHoof2014-nr-gauntff.dat")

    log10_gamma2_vals = None
    log10_u_vals = None
    table = None

    with open(fname) as f:
        # Skip initial comments
        for line in f:
            if not line.startswith("#"):
                break
        # First non-comment line: magic number
        magic = int(line.split("#")[0].split()[0])
        assert magic == 20140210

        line = next(f)
        num_g2, num_u = [int(x) for x in line.split("#")[0].split()]
        line = next(f)
        log10_g2_start = float(line.split("#")[0].split()[0])
        line = next(f)
        log10_u_start = float(line.split("#")[0].split()[0])
        line = next(f)
        step = float(line.split("#")[0].split()[0])

        log10_gamma2_vals = np.arange(num_g2) * step + log10_g2_start
        log10_u_vals = np.arange(num_u) * step + log10_u_start

        # Skip second comment block
        for line in f:
            if not line.startswith("#"):
                break

        rows = [list(map(float, line.split()))]
        for line in f:
            if line.startswith("#"):
                break
            vals = line.split()
            if not vals:
                break
            rows.append(list(map(float, vals)))
            if len(rows) == num_u:
                break

        table = np.array(rows)  # shape (num_u, num_g2)

    return table, log10_gamma2_vals, log10_u_vals


try:
    _gaunt_table, _gaunt_log_g2, _gaunt_log_u = _load_gaunt_table()
    _gaunt_available = True
except Exception:
    _gaunt_available = False
    _gaunt_table = None
    _gaunt_log_g2 = None
    _gaunt_log_u = None


def _gaunt_ff_interp(log_u, log_g2):
    """Bilinear interpolation in the Gaunt factor table (NumPy)."""
    if not _gaunt_available:
        return 1.0  # fallback

    g2 = _gaunt_log_g2
    u = _gaunt_log_u

    # Clamp to table bounds
    log_u = np.clip(log_u, u[0], u[-1])
    log_g2 = np.clip(log_g2, g2[0], g2[-1])

    # Find indices
    step_u = u[1] - u[0]
    step_g = g2[1] - g2[0]
    fi = (log_u - u[0]) / step_u
    fj = (log_g2 - g2[0]) / step_g
    i = int(np.clip(np.floor(fi), 0, len(u) - 2))
    j = int(np.clip(np.floor(fj), 0, len(g2) - 2))
    di = fi - i
    dj = fj - j

    # Bilinear
    val = (_gaunt_table[i, j] * (1 - di) * (1 - dj)
           + _gaunt_table[i + 1, j] * di * (1 - dj)
           + _gaunt_table[i, j + 1] * (1 - di) * dj
           + _gaunt_table[i + 1, j + 1] * di * dj)
    return float(val)


def gaunt_ff_jax(log_u, log_g2):
    """JAX-compatible Gaunt factor via bilinear interp on pre-loaded table."""
    if not _gaunt_available:
        return jnp.ones_like(log_u)

    g2 = jnp.array(_gaunt_log_g2)
    u = jnp.array(_gaunt_log_u)
    table = jnp.array(_gaunt_table)

    step_u = u[1] - u[0]
    step_g = g2[1] - g2[0]

    fi = jnp.clip((log_u - u[0]) / step_u, 0.0, len(u) - 1.001)
    fj = jnp.clip((log_g2 - g2[0]) / step_g, 0.0, len(g2) - 1.001)
    i = jnp.floor(fi).astype(jnp.int32)
    j = jnp.floor(fj).astype(jnp.int32)
    di = fi - i
    dj = fj - j

    val = (table[i, j] * (1 - di) * (1 - dj)
           + table[i + 1, j] * di * (1 - dj)
           + table[i, j + 1] * (1 - di) * dj
           + table[i + 1, j + 1] * di * dj)
    return val


# ── Main function ────────────────────────────────────────────────────────────

def hydrogenic_ff_absorption(nu, T, Z, ni, ne):
    """Free-free absorption coefficient for a hydrogenic ion.

    Parameters
    ----------
    nu : frequency (Hz) — scalar or JAX array
    T : temperature (K)
    Z : charge of the ion (int)
    ni : ion number density (cm^-3)
    ne : electron number density
    """
    inv_T = 1.0 / T
    Z2 = Z * Z
    hnu_div_kT = (hplanck_eV / kboltz_eV) * nu * inv_T
    log_u = jnp.log10(hnu_div_kT)
    log_g2 = jnp.log10((Rydberg_eV / kboltz_eV) * Z2 * inv_T)

    gaunt = gaunt_ff_jax(log_u, log_g2)
    F_nu = 3.6919e8 * gaunt * Z2 * jnp.sqrt(inv_T) / (nu * nu * nu)

    return ni * ne * F_nu * (1.0 - jnp.exp(-hnu_div_kT))
