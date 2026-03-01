"""Continuum (bound-free + free-free + scattering) opacity.

Ported from Korg.jl/src/ContinuumAbsorption/.
"""
from __future__ import annotations
import math
import numpy as np
import jax.numpy as jnp

from ..species import Species, Formula
from ..constants import electron_mass_cgs, electron_charge_cgs, c_cgs
from .scattering import electron_scattering, rayleigh, rayleigh_layers
from .hydrogenic import hydrogenic_ff_absorption
from .hydrogen import (H_I_bf, Hminus_bf, Hminus_ff, H2plus_bf_ff,
                       H_I_bf_layers, Hminus_bf_layers, Hminus_ff_layers, H2plus_bf_ff_layers)
from .helium import Heminus_ff as He_minus_ff, Heminus_ff_layers as He_minus_ff_layers
from .metals import (metal_bf_absorption, positive_ion_ff_absorption,
                     metal_bf_absorption_layers, positive_ion_ff_absorption_layers)

# Cached Species objects (avoid repeated construction per call)
_H_I = Species(Formula.from_Z(1), 0)
_H_II = Species(Formula.from_Z(1), 1)
_He_I = Species(Formula.from_Z(2), 0)
try:
    _H2 = Species.from_string("H2")
except (ValueError, KeyError):
    _H2 = None


def total_continuum_absorption(nus, T, ne, number_densities, partition_funcs):
    """Total continuum linear absorption coefficient alpha(nu).

    Parameters
    ----------
    nus : (n_freq,) sorted frequency array (Hz)
    T : scalar temperature (K)
    ne : scalar electron number density
    number_densities : dict Species -> float
    partition_funcs : dict Species -> callable(logT)

    Returns
    -------
    alpha : (n_freq,) numpy array
    """
    nus = np.asarray(nus, dtype=np.float64)

    nH_I = number_densities[_H_I]
    nH_II = number_densities[_H_II]
    nHe_I = number_densities[_He_I]

    logT = math.log(T)
    U_H_I = partition_funcs[_H_I](logT)
    U_He_I = partition_funcs[_He_I](logT)

    nH_I_div_U = nH_I / U_H_I
    nHe_I_div_U = nHe_I / U_He_I

    nH2 = number_densities.get(_H2, 0.0) if _H2 is not None else 0.0

    alpha = np.zeros_like(nus)

    # H I bound-free
    alpha += H_I_bf(nus, T, nH_I, nHe_I, ne, 1.0 / U_H_I)

    # H- bound-free and free-free
    alpha += Hminus_bf(nus, T, nH_I_div_U, ne)
    alpha += Hminus_ff(nus, T, nH_I_div_U, ne)

    # H2+ bound-free + free-free
    alpha += H2plus_bf_ff(nus, T, nH_I, nH_II)

    # He- free-free
    alpha += He_minus_ff(nus, T, nHe_I_div_U, ne)

    # Positive ion free-free (hydrogenic)
    alpha += positive_ion_ff_absorption(nus, T, number_densities, ne)

    # Metal bound-free
    alpha += metal_bf_absorption(nus, T, number_densities)

    # Scattering
    alpha += electron_scattering(ne)
    alpha += rayleigh(nus, nH_I, nHe_I, nH2)

    return alpha


def total_continuum_absorption_layers(nus, T_arr, ne_arr, number_densities, partition_funcs):
    """Vectorised total continuum absorption over all atmosphere layers.

    Parameters
    ----------
    nus : (n_freq,) frequency array (Hz), sorted ascending
    T_arr : (n_layers,) temperature array (K)
    ne_arr : (n_layers,) electron number density array
    number_densities : dict Species -> (n_layers,) array
    partition_funcs : dict Species -> callable with .batch() method

    Returns
    -------
    alpha : (n_layers, n_freq) linear absorption coefficient
    """
    nus    = np.asarray(nus,    dtype=np.float64)
    T_arr  = np.asarray(T_arr,  dtype=np.float64)
    ne_arr = np.asarray(ne_arr, dtype=np.float64)
    n_layers = len(T_arr)
    n_freq   = len(nus)

    nH_I  = np.asarray(number_densities.get(_H_I,  np.zeros(n_layers)))
    nH_II = np.asarray(number_densities.get(_H_II, np.zeros(n_layers)))
    nHe_I = np.asarray(number_densities.get(_He_I, np.zeros(n_layers)))
    nH2   = (np.asarray(number_densities.get(_H2, np.zeros(n_layers)))
             if _H2 is not None else np.zeros(n_layers))

    # Batch partition function evaluation
    logT_jax  = jnp.log(jnp.asarray(T_arr))
    U_H_I  = np.asarray(partition_funcs[_H_I].batch(logT_jax))    # (n_layers,)
    U_He_I = np.asarray(partition_funcs[_He_I].batch(logT_jax))   # (n_layers,)

    nH_I_div_U  = nH_I  / U_H_I    # (n_layers,)
    nHe_I_div_U = nHe_I / U_He_I
    invU_H      = 1.0   / U_H_I

    thomson_coef = (8.0 * np.pi / 3.0
                    * (electron_charge_cgs ** 2
                       / (electron_mass_cgs * c_cgs ** 2)) ** 2)

    # All _layers functions now return JAX arrays → sum stays on device
    alpha = (H_I_bf_layers(nus, T_arr, nH_I, nHe_I, ne_arr, invU_H)
             + Hminus_bf_layers(nus, T_arr, nH_I_div_U, ne_arr)
             + Hminus_ff_layers(nus, T_arr, nH_I_div_U, ne_arr)
             + H2plus_bf_ff_layers(nus, T_arr, nH_I, nH_II)
             + He_minus_ff_layers(nus, T_arr, nHe_I_div_U, ne_arr)
             + positive_ion_ff_absorption_layers(nus, T_arr, number_densities, ne_arr)
             + metal_bf_absorption_layers(nus, T_arr, number_densities)
             + (thomson_coef * ne_arr)[:, np.newaxis]
             + rayleigh_layers(nus, nH_I, nHe_I, nH2))

    return alpha
