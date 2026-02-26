"""Precomputed molecular cross sections.

Ported from Korg.jl/src/molecular_cross_sections.jl.
"""
from __future__ import annotations

import os
import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass

from .wavelengths import Wavelengths
from .linelist import Linelist
from .species import Species
from .line_absorption import line_absorption


@dataclass
class MolecularCrossSection:
    wls: Wavelengths
    vmic_vals: np.ndarray
    log_temp_vals: np.ndarray
    values: np.ndarray
    species: object

    def interpolate(self, vmic, logT, wls):
        return _trilinear_interp(vmic, logT, wls, self.vmic_vals, self.log_temp_vals,
                                 self.wls.all_wls, self.values)


# ── Interpolation ───────────────────────────────────────────────────────────

def _trilinear_interp(x, y, z, xg, yg, zg, table):
    xg = jnp.asarray(xg)
    yg = jnp.asarray(yg)
    zg = jnp.asarray(zg)
    tab = jnp.asarray(table)

    x = jnp.asarray(x)
    y = jnp.asarray(y)
    z = jnp.asarray(z)

    x = jnp.clip(x, xg[0], xg[-1])
    y = jnp.clip(y, yg[0], yg[-1])
    z = jnp.clip(z, zg[0], zg[-1])

    ix = jnp.clip(jnp.searchsorted(xg, x, side="right") - 1, 0, len(xg) - 2)
    iy = jnp.clip(jnp.searchsorted(yg, y, side="right") - 1, 0, len(yg) - 2)
    iz = jnp.clip(jnp.searchsorted(zg, z, side="right") - 1, 0, len(zg) - 2)

    x0, x1 = xg[ix], xg[ix + 1]
    y0, y1 = yg[iy], yg[iy + 1]
    z0, z1 = zg[iz], zg[iz + 1]

    tx = jnp.where(x1 == x0, 0.0, (x - x0) / (x1 - x0))
    ty = jnp.where(y1 == y0, 0.0, (y - y0) / (y1 - y0))
    tz = jnp.where(z1 == z0, 0.0, (z - z0) / (z1 - z0))

    f000 = tab[ix, iy, iz]
    f100 = tab[ix + 1, iy, iz]
    f010 = tab[ix, iy + 1, iz]
    f110 = tab[ix + 1, iy + 1, iz]
    f001 = tab[ix, iy, iz + 1]
    f101 = tab[ix + 1, iy, iz + 1]
    f011 = tab[ix, iy + 1, iz + 1]
    f111 = tab[ix + 1, iy + 1, iz + 1]

    c00 = f000 * (1 - tx) + f100 * tx
    c10 = f010 * (1 - tx) + f110 * tx
    c01 = f001 * (1 - tx) + f101 * tx
    c11 = f011 * (1 - tx) + f111 * tx

    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty

    return c0 * (1 - tz) + c1 * tz


# ── API ─────────────────────────────────────────────────────────────────────

def build_molecular_cross_section(linelist: Linelist, wl_params,
                                  cutoff_alpha=1e-32,
                                  vmic_vals=None, log_temp_vals=None,
                                  partition_funcs=None):
    """Precompute a molecular cross section for a single species linelist."""
    if vmic_vals is None:
        vmic_vals = np.concatenate([
            np.arange(0.0, 1.0 + 1e-12, 1/3),
            [1.5],
            np.arange(2.0, 5.3334 + 1e-12, 2/3),
        ])
    if log_temp_vals is None:
        log_temp_vals = np.arange(3.0, 5.0 + 1e-12, 0.04)

    wls = Wavelengths(wl_params) if not isinstance(wl_params, Wavelengths) else wl_params
    all_specs = linelist.species
    if not all(s == all_specs[0] for s in all_specs):
        raise ValueError("All lines must be of the same species")
    species = all_specs[0]

    n_v = len(vmic_vals)
    n_t = len(log_temp_vals)
    n_w = len(wls)
    alpha = np.zeros((n_v, n_t, n_w), dtype=float)

    Ts = 10 ** log_temp_vals
    ne = np.zeros(n_t)
    n_dict = {species: 1.0 / cutoff_alpha}

    for i, vmic in enumerate(vmic_vals):
        xi = vmic * 1e5
        for t_idx, T in enumerate(Ts):
            n_abs = np.full(linelist.n_lines, n_dict[species])
            linelist_jax = {
                "wl": jnp.asarray(linelist.wl),
                "log_gf": jnp.asarray(linelist.log_gf),
                "species_idx": jnp.zeros(linelist.n_lines, dtype=jnp.int32),
                "E_lower": jnp.asarray(linelist.E_lower),
                "gamma_rad": jnp.asarray(linelist.gamma_rad),
                "gamma_stark": jnp.asarray(linelist.gamma_stark),
                "vdW_1": jnp.asarray(linelist.vdW_1),
                "vdW_2": jnp.asarray(linelist.vdW_2),
                "mass": jnp.asarray([s.get_mass() for s in linelist.species]),
                "is_molecule": jnp.asarray([s.ismolecule() for s in linelist.species]),
            }
            alpha[i, t_idx, :] = np.asarray(line_absorption(
                jnp.asarray(wls.all_wls),
                linelist_jax,
                T, 0.0, 0.0, jnp.asarray(n_abs), xi
            ))

    alpha = alpha * cutoff_alpha
    return MolecularCrossSection(wls, np.array(vmic_vals), np.array(log_temp_vals), alpha, species)


def interpolate_molecular_cross_sections(alpha, molecular_cross_sections, wls, Ts, vmic, number_densities):
    if molecular_cross_sections is None or len(molecular_cross_sections) == 0:
        return alpha

    wls = Wavelengths(wls) if not isinstance(wls, Wavelengths) else wls
    Ts = np.asarray(Ts)
    vmic_vals = vmic
    out = alpha

    for sigma in molecular_cross_sections:
        for i in range(alpha.shape[0]):
            vm = vmic_vals if np.isscalar(vmic_vals) else vmic_vals[i]
            logT = np.log10(Ts[i])
            vals = _trilinear_interp(vm, logT, wls.all_wls,
                                     sigma.vmic_vals, sigma.log_temp_vals, sigma.wls.all_wls,
                                     sigma.values)
            out = out.at[i, :].add(vals * number_densities[sigma.species][i])

    return out


def save_molecular_cross_section(filename, cross_section: MolecularCrossSection):
    import h5py
    wls = cross_section.wls
    with h5py.File(filename, "w") as f:
        f.create_dataset("wls", data=np.array([(r[0], r[1]-r[0], r[-1]) for r in wls.wl_ranges]))
        f.create_dataset("vmic_vals", data=cross_section.vmic_vals)
        f.create_dataset("T_vals", data=cross_section.log_temp_vals)
        f.create_dataset("vals", data=cross_section.values)
        f.create_dataset("species", data=str(cross_section.species))


def read_molecular_cross_section(filename):
    import h5py
    with h5py.File(filename, "r") as f:
        wls_ranges = [np.arange(start, stop + step * 0.5, step) * 1e8
                      for start, step, stop in f["wls"][()]]
        wls = Wavelengths([r for r in wls_ranges])
        vmic_vals = f["vmic_vals"][()]
        logT_vals = f["T_vals"][()]
        values = f["vals"][()]
        species_val = f["species"][()]
        if hasattr(species_val, "decode"):
            species_val = species_val.decode()
        species = Species.from_string(species_val)

    return MolecularCrossSection(wls, vmic_vals, logT_vals, values, species)
