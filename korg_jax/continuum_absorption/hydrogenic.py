"""Hydrogenic free-free absorption with Gaunt factor interpolation.

Ported from Korg.jl/src/ContinuumAbsorption/hydrogenic_bf_ff.jl.
"""
from __future__ import annotations

import os
import numpy as np
import jax.numpy as jnp

from ..constants import hplanck_eV, kboltz_eV, Rydberg_eV


# ── Gaunt factor table (loaded once at import time) ─────────────────────────

def _load_gaunt_table(fname=None):
    """Load van Hoof et al. (2014) non-relativistic free-free Gaunt factors."""
    if fname is None:
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        fname = os.path.join(base, "data", "vanHoof2014-nr-gauntff.dat")

    with open(fname) as f:
        for line in f:
            if not line.startswith("#"):
                break
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

        table = np.array(rows)

    return table, log10_gamma2_vals, log10_u_vals


try:
    _gaunt_table, _gaunt_log_g2, _gaunt_log_u = _load_gaunt_table()
    _gaunt_available = True
    _gaunt_step_u = float(_gaunt_log_u[1] - _gaunt_log_u[0])
    _gaunt_step_g = float(_gaunt_log_g2[1] - _gaunt_log_g2[0])
    _gaunt_nu = len(_gaunt_log_u)
    _gaunt_ng = len(_gaunt_log_g2)
    # JAX version of the table (transferred to device once at import)
    _gaunt_table_jnp = jnp.asarray(_gaunt_table)
except Exception:
    _gaunt_available = False
    _gaunt_table = None
    _gaunt_log_g2 = None
    _gaunt_log_u = None
    _gaunt_step_u = None
    _gaunt_step_g = None
    _gaunt_nu = 0
    _gaunt_ng = 0
    _gaunt_table_jnp = None


def gaunt_ff(log_u, log_g2):
    """Vectorized Gaunt factor via bilinear interpolation (JAX)."""
    if not _gaunt_available:
        return jnp.ones_like(jnp.asarray(log_u))

    log_u  = jnp.asarray(log_u,  dtype=jnp.float64)
    log_g2 = jnp.asarray(log_g2, dtype=jnp.float64)

    fi = jnp.clip((log_u  - _gaunt_log_u[0])  / _gaunt_step_u, 0.0, _gaunt_nu - 1.001)
    fj = jnp.clip((log_g2 - _gaunt_log_g2[0]) / _gaunt_step_g, 0.0, _gaunt_ng - 1.001)
    i = jnp.floor(fi).astype(jnp.int32)
    j = jnp.floor(fj).astype(jnp.int32)
    di = fi - i
    dj = fj - j

    return (_gaunt_table_jnp[i, j]         * (1 - di) * (1 - dj)
            + _gaunt_table_jnp[i + 1, j]   *      di  * (1 - dj)
            + _gaunt_table_jnp[i, j + 1]   * (1 - di) *      dj
            + _gaunt_table_jnp[i + 1, j + 1] *    di  *      dj)


# ── Main function ────────────────────────────────────────────────────────────

def hydrogenic_ff_absorption(nu, T, Z, ni, ne):
    """Free-free absorption coefficient for a hydrogenic ion (NumPy).

    Parameters
    ----------
    nu : frequency (Hz) — scalar or array
    T : temperature (K)
    Z : charge of the ion (int)
    ni : ion number density (cm^-3)
    ne : electron number density
    """
    nu = np.asarray(nu, dtype=np.float64)
    inv_T = 1.0 / T
    Z2 = Z * Z
    hnu_div_kT = (hplanck_eV / kboltz_eV) * nu * inv_T
    log_u = np.log10(hnu_div_kT)
    log_g2 = np.log10((Rydberg_eV / kboltz_eV) * Z2 * inv_T)

    gaunt = gaunt_ff(log_u, log_g2)
    F_nu = 3.6919e8 * gaunt * Z2 * np.sqrt(inv_T) / (nu * nu * nu)

    return ni * ne * F_nu * (1.0 - np.exp(-hnu_div_kT))


def hydrogenic_ff_absorption_layers(nu, T, Z, ni, ne):
    """Batch hydrogenic free-free: T/ni/ne are (n_layers,), nu is (n_freq,).

    Returns (n_layers, n_freq).
    """
    nu  = jnp.asarray(nu,  dtype=jnp.float64)                         # (n_freq,)
    T   = jnp.asarray(T,   dtype=jnp.float64)[:, None]                # (n_layers, 1)
    ni  = jnp.asarray(ni,  dtype=jnp.float64)[:, None]
    ne  = jnp.asarray(ne,  dtype=jnp.float64)[:, None]

    Z2     = Z * Z
    inv_T  = 1.0 / T                                                   # (n_layers, 1)
    nu2d   = nu[None, :]                                               # (1, n_freq)

    hnu_div_kT = (hplanck_eV / kboltz_eV) * nu2d * inv_T             # (n_layers, n_freq)
    log_u  = jnp.log10(hnu_div_kT)                                    # (n_layers, n_freq)
    log_g2 = jnp.log10((Rydberg_eV / kboltz_eV) * Z2 * inv_T)        # (n_layers, 1)

    gaunt = gaunt_ff(log_u, log_g2)                                    # (n_layers, n_freq)
    F_nu  = 3.6919e8 * gaunt * Z2 * jnp.sqrt(inv_T) / (nu2d ** 3)    # (n_layers, n_freq)

    return ni * ne * F_nu * (1.0 - jnp.exp(-hnu_div_kT))
