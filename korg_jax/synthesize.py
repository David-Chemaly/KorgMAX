"""Main synthesis pipeline.

Ported from Korg.jl/src/synthesize.jl.
"""
from __future__ import annotations

import os
import math
import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass

from .constants import c_cgs
from .wavelengths import Wavelengths
from .species import Species, Formula
from .abundances import format_A_X
from .linelist import Linelist
from .continuum_absorption import total_continuum_absorption
from .line_absorption import line_absorption, line_absorption_layers, build_linelist_chunks
from .hydrogen_lines import hydrogen_line_absorption, hydrogen_line_absorption_layers
from .radiative_transfer import radiative_transfer
from .utils import merge_bounds, blackbody
from .statmech import chemical_equilibrium, _ChemEqContext
from .read_statmech import setup_all
from .cubic_splines import CubicSpline as _CubicSpline, cubic_spline_build, cubic_spline_eval
from .molecular_cross_sections import interpolate_molecular_cross_sections

_FILTER_CACHE = {}
_LINE_PREP_CACHE = {}
_CACHE_MAXSIZE = 16
_STATMECH_SETUP_CACHE = None
_PF_CACHE = {}
_LEC_CACHE = {}


@dataclass
class SynthesisResult:
    flux: np.ndarray
    cntm: np.ndarray | None
    intensity: np.ndarray
    alpha: np.ndarray
    mu_grid: list
    number_densities: dict
    electron_number_density: np.ndarray
    wavelengths: np.ndarray
    subspectra: list


# ── Helpers ────────────────────────────────────────────────────────────────

def _ensure_callable(data):
    """Convert (knots, values) tuple to callable CubicSpline if needed."""
    if callable(data):
        return data
    lnTs, vals = data
    return _CubicSpline(np.asarray(lnTs), np.asarray(vals), extrapolate=True)


def _build_partition_funcs(partition_funcs):
    key = id(partition_funcs)
    cached = _PF_CACHE.get(key)
    if cached is not None:
        return cached
    out = {spec: _ensure_callable(data) for spec, data in partition_funcs.items()}
    _cache_put(_PF_CACHE, key, out)
    return out


def _build_log_equilibrium_constants(log_equilibrium_constants):
    key = id(log_equilibrium_constants)
    cached = _LEC_CACHE.get(key)
    if cached is not None:
        return cached
    out = {spec: _ensure_callable(data) for spec, data in log_equilibrium_constants.items()}
    _cache_put(_LEC_CACHE, key, out)
    return out


def _get_statmech_setup_cached():
    global _STATMECH_SETUP_CACHE
    if _STATMECH_SETUP_CACHE is None:
        ionization_energies, partition_funcs, log_equilibrium_constants = setup_all()
        _STATMECH_SETUP_CACHE = (
            ionization_energies,
            _build_partition_funcs(partition_funcs),
            _build_log_equilibrium_constants(log_equilibrium_constants),
        )
    return _STATMECH_SETUP_CACHE


def _cache_put(cache, key, value):
    if key in cache:
        cache[key] = value
        return
    if len(cache) >= _CACHE_MAXSIZE:
        oldest = next(iter(cache))
        del cache[oldest]
    cache[key] = value


def _wavelength_cache_key(wls: Wavelengths):
    windows = tuple((float(lo), float(hi)) for lo, hi in wls.eachwindow())
    return (float(wls.all_wls[0]), float(wls.all_wls[-1]), int(len(wls.all_wls)), windows)


def _get_filtered_linelist(linelist: Linelist, wls: Wavelengths, line_buffer):
    key = (id(linelist), _wavelength_cache_key(wls), float(line_buffer))
    cached = _FILTER_CACHE.get(key)
    if cached is not None:
        return cached
    filtered = filter_linelist(linelist, wls, line_buffer)
    _cache_put(_FILTER_CACHE, key, filtered)
    return filtered


def _prepare_linelist_fast(linelist: Linelist, chunk_size: int):
    key = (id(linelist), int(chunk_size))
    cached = _LINE_PREP_CACHE.get(key)
    if cached is not None:
        return cached

    species = linelist.species
    unique_species = list(dict.fromkeys(species))
    species_to_idx = {s: i for i, s in enumerate(unique_species)}
    line_species_idx = np.array([species_to_idx[s] for s in species], dtype=np.int32)

    linelist_jax = {
        "wl": jnp.asarray(linelist.wl),
        "log_gf": jnp.asarray(linelist.log_gf),
        "species_idx": jnp.zeros(linelist.n_lines, dtype=jnp.int32),
        "E_lower": jnp.asarray(linelist.E_lower),
        "gamma_rad": jnp.asarray(linelist.gamma_rad),
        "gamma_stark": jnp.asarray(linelist.gamma_stark),
        "vdW_1": jnp.asarray(linelist.vdW_1),
        "vdW_2": jnp.asarray(linelist.vdW_2),
        "mass": jnp.asarray([s.get_mass() for s in species]),
        "is_molecule": jnp.asarray([s.ismolecule() for s in species]),
    }
    chunks = build_linelist_chunks(linelist_jax, chunk_size=chunk_size)
    out = (linelist_jax, chunks, unique_species, line_species_idx)
    _cache_put(_LINE_PREP_CACHE, key, out)
    return out


def _build_n_abs_matrix(n_dicts, unique_species, line_species_idx, partition_funcs, temps):
    n_layers = len(n_dicts)
    n_unique = len(unique_species)
    if n_unique == 0:
        return np.zeros((n_layers, 0), dtype=float)

    dens = np.zeros((n_layers, n_unique), dtype=float)
    for j, spec in enumerate(unique_species):
        dens[:, j] = [nd.get(spec, 0.0) for nd in n_dicts]

    lnT = np.log(np.asarray(temps))
    part = np.empty((n_layers, n_unique), dtype=float)
    for j, spec in enumerate(unique_species):
        pf = partition_funcs[spec]
        try:
            vals = np.asarray(pf(lnT), dtype=float)
            if vals.shape == ():
                vals = np.full(n_layers, float(vals))
            elif vals.shape[0] != n_layers:
                vals = np.array([pf(float(lt)) for lt in lnT], dtype=float)
        except Exception:
            vals = np.array([pf(float(lt)) for lt in lnT], dtype=float)
        part[:, j] = vals

    n_unique_scaled = np.divide(dens, part, out=np.zeros_like(dens), where=part != 0.0)
    return n_unique_scaled[:, line_species_idx]


def filter_linelist(linelist: Linelist, wls: Wavelengths, line_buffer, warn_empty=True):
    nlines_before = linelist.n_lines
    last_line_index = 0

    ranges = []
    for lam_start, lam_stop in wls.eachwindow():
        first_line_index = np.searchsorted(linelist.wl, lam_start - line_buffer, side="left")
        first_line_index = max(first_line_index, last_line_index)
        last_line_index = np.searchsorted(linelist.wl, lam_stop + line_buffer, side="right")
        ranges.append((first_line_index, last_line_index))

    idxs = np.concatenate([np.arange(a, b) for a, b in ranges if b > a])
    if len(idxs) == 0:
        if nlines_before != 0 and warn_empty:
            pass
        return Linelist(
            wl=np.array([]), log_gf=np.array([]), species=[], E_lower=np.array([]),
            gamma_rad=np.array([]), gamma_stark=np.array([]), vdW_1=np.array([]), vdW_2=np.array([])
        )

    return Linelist(
        wl=linelist.wl[idxs], log_gf=linelist.log_gf[idxs],
        species=[linelist.species[i] for i in idxs],
        E_lower=linelist.E_lower[idxs],
        gamma_rad=linelist.gamma_rad[idxs], gamma_stark=linelist.gamma_stark[idxs],
        vdW_1=linelist.vdW_1[idxs], vdW_2=linelist.vdW_2[idxs],
    )


def _load_alpha_5000_default_linelist():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fname = os.path.join(base, "data", "linelists", "alpha_5000", "alpha_5000_lines.csv")
    if not os.path.exists(fname):
        return None

    wl = []
    log_gf = []
    species = []
    E_lower = []
    gamma_rad = []
    gamma_stark = []
    vdW_1 = []
    vdW_2 = []

    import csv
    with open(fname, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for parts in reader:
            if len(parts) < 7:
                continue
            wl.append(float(parts[0]))
            log_gf.append(float(parts[1]))
            species.append(Species.from_string(parts[2]))
            E_lower.append(float(parts[3]))
            gamma_rad.append(float(parts[4]))
            gamma_stark.append(float(parts[5]))
            vdW = parts[6].strip().strip('"')
            if not vdW:
                vdW_1.append(-1.0)
                vdW_2.append(-1.0)
            elif vdW.startswith("("):
                vdW = vdW.strip("()").replace(",", " ")
                vals = [v for v in vdW.split() if v]
                v1 = vals[0] if len(vals) >= 1 else "-1.0"
                v2 = vals[1] if len(vals) >= 2 else "-1.0"
                vdW_1.append(float(v1))
                vdW_2.append(float(v2))
            else:
                vdW_1.append(float(vdW))
                vdW_2.append(-1.0)

    return Linelist(
        wl=np.array(wl), log_gf=np.array(log_gf), species=species,
        E_lower=np.array(E_lower), gamma_rad=np.array(gamma_rad),
        gamma_stark=np.array(gamma_stark), vdW_1=np.array(vdW_1), vdW_2=np.array(vdW_2)
    )

def _concat_linelist(a: Linelist, b: Linelist) -> Linelist:
    if a.n_lines == 0:
        return b
    if b.n_lines == 0:
        return a
    return Linelist(
        wl=np.concatenate([a.wl, b.wl]),
        log_gf=np.concatenate([a.log_gf, b.log_gf]),
        species=a.species + b.species,
        E_lower=np.concatenate([a.E_lower, b.E_lower]),
        gamma_rad=np.concatenate([a.gamma_rad, b.gamma_rad]),
        gamma_stark=np.concatenate([a.gamma_stark, b.gamma_stark]),
        vdW_1=np.concatenate([a.vdW_1, b.vdW_1]),
        vdW_2=np.concatenate([a.vdW_2, b.vdW_2]),
    )


def get_reference_wavelength_linelist(linelist, reference_wavelength, use_internal_reference_linelist=True):
    if reference_wavelength == 5e-5 and use_internal_reference_linelist:
        ll = _load_alpha_5000_default_linelist()
        if ll is not None:
            return ll

    filtered = filter_linelist(linelist, Wavelengths.from_tuple(5000, 5000), 21e-8, warn_empty=False)
    if reference_wavelength != 5e-5:
        if filtered.n_lines == 0:
            raise ValueError("Linelist must contain lines near the reference wavelength")
        return filtered

    # reference_wavelength == 5000 Å and user linelist (or fallback) should cover it
    if filtered.n_lines == 0:
        ll = _load_alpha_5000_default_linelist()
        return ll if ll is not None else linelist

    if filtered.wl[0] > 5e-5:
        ll = _load_alpha_5000_default_linelist()
        if ll is None:
            return filtered
        mask = ll.wl < filtered.wl[0]
        left = Linelist(
            wl=ll.wl[mask],
            log_gf=ll.log_gf[mask],
            species=[s for s, m in zip(ll.species, mask) if m],
            E_lower=ll.E_lower[mask],
            gamma_rad=ll.gamma_rad[mask],
            gamma_stark=ll.gamma_stark[mask],
            vdW_1=ll.vdW_1[mask],
            vdW_2=ll.vdW_2[mask],
        )
        return _concat_linelist(left, filtered)

    if filtered.wl[-1] < 5e-5:
        ll = _load_alpha_5000_default_linelist()
        if ll is None:
            return filtered
        mask = ll.wl > filtered.wl[-1]
        right = Linelist(
            wl=ll.wl[mask],
            log_gf=ll.log_gf[mask],
            species=[s for s, m in zip(ll.species, mask) if m],
            E_lower=ll.E_lower[mask],
            gamma_rad=ll.gamma_rad[mask],
            gamma_stark=ll.gamma_stark[mask],
            vdW_1=ll.vdW_1[mask],
            vdW_2=ll.vdW_2[mask],
        )
        return _concat_linelist(filtered, right)

    return filtered


# ── Main API ────────────────────────────────────────────────────────────────

def synthesize(atm, linelist: Linelist, A_X, *wavelength_params,
               vmic=1.0,
               line_buffer=10.0,
               cntm_step=1.0,
               hydrogen_lines=True,
               use_MHD_for_hydrogen_lines=None,
               hydrogen_line_window_size=150,
               mu_values=20,
               line_cutoff_threshold=3e-4,
               electron_number_density_warn_threshold=np.inf,
               electron_number_density_warn_min_value=1e-4,
               return_cntm=True,
               use_internal_reference_linelist=True,
               I_scheme="linear_flux_only",
               tau_scheme="anchored",
               ionization_energies=None,
               partition_funcs=None,
               log_equilibrium_constants=None,
               molecular_cross_sections=None,
               use_chemical_equilibrium_from=None,
               prefer_jit=True,
               chunk_size=None):

    if len(wavelength_params) == 1 and isinstance(wavelength_params[0], Wavelengths):
        wls = wavelength_params[0]
    elif len(wavelength_params) == 1 and isinstance(wavelength_params[0], tuple) and len(wavelength_params[0]) in (2, 3):
        wls = Wavelengths.from_tuple(*wavelength_params[0])
    elif len(wavelength_params) == 1 and isinstance(wavelength_params[0], (list, tuple)):
        wls = Wavelengths(wavelength_params[0])
    elif len(wavelength_params) in (2, 3):
        wls = Wavelengths.from_tuple(*wavelength_params)
    elif len(wavelength_params) == 1:
        wls = Wavelengths(wavelength_params[0])
    else:
        wls = Wavelengths(wavelength_params)

    if use_MHD_for_hydrogen_lines is None:
        use_MHD_for_hydrogen_lines = wls.all_wls[-1] < 13000 * 1e-8

    min_allowed_wl = 1300.0 * 1e-8
    if wls.all_wls[0] < min_allowed_wl:
        raise ValueError("Wavelength range extends blueward of 1300 Å")

    cntm_step = cntm_step * 1e-8
    line_buffer = line_buffer * 1e-8

    cntm_windows = [(lo - line_buffer - cntm_step, hi + line_buffer + cntm_step)
                    for lo, hi in wls.eachwindow()]
    cntm_windows, _ = merge_bounds(cntm_windows)
    cntm_wls = Wavelengths([np.arange(a, b + cntm_step * 0.5, cntm_step) for a, b in cntm_windows])

    if not np.all(np.diff(linelist.wl) >= 0):
        linelist = linelist.sort_by_wavelength()

    if tau_scheme == "anchored":
        linelist5 = get_reference_wavelength_linelist(linelist, atm.reference_wavelength,
                                                      use_internal_reference_linelist)

    linelist = _get_filtered_linelist(linelist, wls, line_buffer)

    if len(A_X) < 92 or A_X[0] != 12:
        raise ValueError("A(H) must be a 92-element vector with A[1] == 12")

    abs_abundances = 10 ** (np.asarray(A_X) - 12)
    abs_abundances = abs_abundances / np.sum(abs_abundances)

    if ionization_energies is None or partition_funcs is None or log_equilibrium_constants is None:
        ionization_energies, partition_funcs, log_equilibrium_constants = _get_statmech_setup_cached()
    else:
        partition_funcs = _build_partition_funcs(partition_funcs)
        log_equilibrium_constants = _build_log_equilibrium_constants(log_equilibrium_constants)

    n_layers = atm.n_layers
    alpha_np = np.zeros((n_layers, len(wls)))
    alpha_ref = np.zeros(n_layers)
    n_e_vals = np.zeros(n_layers)
    n_dicts = []
    alpha_cntm_funcs = []

    # Merge reference frequency into continuum grid to halve the number of calls
    if tau_scheme == "anchored":
        ref_freq = c_cgs / atm.reference_wavelength
        combined_freqs = np.append(cntm_wls.all_freqs, ref_freq)
    else:
        combined_freqs = np.asarray(cntm_wls.all_freqs)

    # Pre-compute chemical equilibrium context once (avoids rebuilding
    # Species objects and molecule data for every layer)
    cheq_ctx = None
    if use_chemical_equilibrium_from is None:
        cheq_ctx = _ChemEqContext(abs_abundances, ionization_energies,
                                  partition_funcs, log_equilibrium_constants)

    for i in range(n_layers):
        if use_chemical_equilibrium_from is None:
            ne, n_dict = chemical_equilibrium(
                atm.temp[i], atm.number_density[i], atm.electron_number_density[i],
                abs_abundances, ionization_energies, partition_funcs,
                log_equilibrium_constants,
                _ctx=cheq_ctx,
            )
        else:
            sol = use_chemical_equilibrium_from
            ne = sol.electron_number_density[i]
            n_dict = {s: sol.number_densities[s][i] for s in sol.number_densities}

        n_e_vals[i] = ne
        n_dicts.append(n_dict)

        alpha_combined = total_continuum_absorption(
            combined_freqs, atm.temp[i], ne, n_dict, partition_funcs
        )

        if tau_scheme == "anchored":
            alpha_ref[i] = alpha_combined[-1]
            alpha_cntm_vals = alpha_combined[:-1][::-1]
        else:
            alpha_cntm_vals = alpha_combined[::-1]

        alpha_cntm = np.interp(wls.all_wls, cntm_wls.all_wls, alpha_cntm_vals, left=alpha_cntm_vals[0], right=alpha_cntm_vals[-1])
        alpha_np[i, :] = alpha_cntm
        alpha_cntm_funcs.append(alpha_cntm)

    # Keep alpha as numpy throughout; convert to/from JAX only for line absorption
    alpha = alpha_np

    # Julia filters out H III from number_densities (synthesize.jl line 248-249)
    H_I = Species(Formula.from_Z(1), 0)
    He_I = Species(Formula.from_Z(2), 0)
    H_III = Species(Formula.from_Z(1), 2)
    number_densities = {spec: np.array([n[spec] for n in n_dicts])
                        for spec in n_dicts[0] if spec != H_III}

    linelist5_chunks = None
    if chunk_size is None:
        chunk_size = 1024 if prefer_jit else 500

    linelist_jax, linelist_chunks, unique_species, line_species_idx = _prepare_linelist_fast(
        linelist, chunk_size
    )

    if tau_scheme == "anchored":
        linelist5_jax, linelist5_chunks, unique_species5, line5_species_idx = _prepare_linelist_fast(
            linelist5, chunk_size
        )
        # vmic in cm/s for line absorption (Julia passes vmic * 1e5)
        xi_cms = vmic * 1e5
        # Vectorized line absorption for reference wavelength
        n_abs_ref = _build_n_abs_matrix(
            n_dicts, unique_species5, line5_species_idx, partition_funcs, atm.temp
        )
        alpha_cntm_ref = np.zeros((n_layers, linelist5.n_lines), dtype=float)
        nH_I_vals_ref = number_densities.get(H_I, np.zeros(n_layers, dtype=float))
        for i in range(n_layers):
            alpha_cntm_ref[i, :] = alpha_ref[i]

        alpha_ref_lines = line_absorption_layers(
            jnp.asarray([atm.reference_wavelength]),
            linelist5_jax,
            jnp.asarray(atm.temp), jnp.asarray(n_e_vals), jnp.asarray(nH_I_vals_ref),
            jnp.asarray(n_abs_ref), xi_cms,
            alpha_cntm_at_line=jnp.asarray(alpha_cntm_ref),
            cutoff_threshold=line_cutoff_threshold,
            chunks=linelist5_chunks,
        )
        alpha_ref += np.asarray(alpha_ref_lines).reshape(-1)
        if molecular_cross_sections:
            alpha_ref_arr = jnp.asarray(alpha_ref).reshape(-1, 1)
            alpha_ref_arr = interpolate_molecular_cross_sections(
                alpha_ref_arr, molecular_cross_sections,
                Wavelengths.from_array([atm.reference_wavelength * 1e8]),
                atm.temp, vmic, number_densities
            )
            alpha_ref = np.asarray(alpha_ref_arr).reshape(-1)

    source_fn = np.array([blackbody(t, wls.all_wls) for t in atm.temp])

    cntm = None
    if return_cntm:
        cntm, _, _, _ = radiative_transfer(atm, alpha, source_fn, mu_values,
                                           alpha_ref=alpha_ref, tau_ref=atm.tau_ref,
                                           I_scheme=I_scheme, tau_scheme=tau_scheme)

    if hydrogen_lines:
        nH_I_vals = number_densities.get(H_I, np.zeros(n_layers, dtype=float))
        nHe_I_vals = number_densities.get(He_I, np.zeros(n_layers, dtype=float))
        U_H_I_vals = np.array([partition_funcs[Species(Formula.from_Z(1), 0)](math.log(t)) for t in atm.temp])
        xi_vals = (vmic if np.isscalar(vmic) else np.asarray(vmic)) * 1e5
        h_alpha = hydrogen_line_absorption_layers(
            wls.all_wls, atm.temp, n_e_vals, nH_I_vals, nHe_I_vals, U_H_I_vals,
            xi_vals, hydrogen_line_window_size * 1e-8, use_MHD=use_MHD_for_hydrogen_lines
        )
        alpha = alpha + np.asarray(h_alpha)

    # vmic in cm/s for line absorption (Julia passes vmic * 1e5)
    xi_cms = vmic * 1e5
    # Vectorized line absorption across layers
    n_abs_matrix = _build_n_abs_matrix(
        n_dicts, unique_species, line_species_idx, partition_funcs, atm.temp
    )
    alpha_cntm_at_lines = np.vstack([
        np.interp(linelist.wl, wls.all_wls, alpha_cntm)
        for alpha_cntm in alpha_cntm_funcs
    ]) if linelist.n_lines > 0 else np.zeros((n_layers, 0), dtype=float)
    nH_I_vals = number_densities.get(H_I, np.zeros(n_layers, dtype=float))

    alpha_lines = line_absorption_layers(
        jnp.asarray(wls.all_wls), linelist_jax,
        jnp.asarray(atm.temp), jnp.asarray(n_e_vals), jnp.asarray(nH_I_vals),
        jnp.asarray(n_abs_matrix), xi_cms,
        alpha_cntm_at_line=jnp.asarray(alpha_cntm_at_lines),
        cutoff_threshold=line_cutoff_threshold,
        chunks=linelist_chunks,
    )
    alpha = alpha + np.asarray(alpha_lines)

    if molecular_cross_sections:
        alpha = np.asarray(interpolate_molecular_cross_sections(
            jnp.asarray(alpha), molecular_cross_sections, wls,
            atm.temp, vmic, number_densities))

    flux, intensity, mu_grid, mu_weights = radiative_transfer(
        atm, alpha, source_fn, mu_values,
        alpha_ref=alpha_ref, tau_ref=atm.tau_ref,
        I_scheme=I_scheme, tau_scheme=tau_scheme
    )

    flux = np.asarray(flux) * 1e-8
    if cntm is not None:
        cntm = np.asarray(cntm) * 1e-8

    return SynthesisResult(
        flux=flux,
        cntm=cntm,
        intensity=np.asarray(intensity),
        alpha=np.asarray(alpha),
        mu_grid=list(zip(mu_grid, mu_weights)),
        number_densities=number_densities,
        electron_number_density=n_e_vals,
        wavelengths=wls.all_wls * 1e8,
        subspectra=wls.subspectrum_indices(),
    )
