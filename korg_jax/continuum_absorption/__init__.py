"""Continuum (bound-free + free-free + scattering) opacity.

Ported from Korg.jl/src/ContinuumAbsorption/.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from functools import partial

from .scattering import electron_scattering, rayleigh
from .hydrogenic import hydrogenic_ff_absorption
from .hydrogen import H_I_bf, Hminus_bf, Hminus_ff, H2plus_bf_ff
from .helium import Heminus_ff as He_minus_ff
from .metals import metal_bf_absorption, positive_ion_ff_absorption


def total_continuum_absorption(nus, T, ne, number_densities, partition_funcs):
    """Total continuum linear absorption coefficient alpha(nu).

    Parameters
    ----------
    nus : (n_freq,) sorted frequency array (Hz)
    T : scalar temperature (K)
    ne : scalar electron number density
    number_densities : dict Species -> float  (or JAX-array indexed by registry)
    partition_funcs : dict Species -> callable(logT)

    Returns
    -------
    alpha : (n_freq,) JAX array
    """
    import math
    from ..species import Species, Formula

    H_I = Species(Formula.from_Z(1), 0)
    H_II = Species(Formula.from_Z(1), 1)
    He_I = Species(Formula.from_Z(2), 0)

    nH_I = number_densities[H_I]
    nH_II = number_densities[H_II]
    nHe_I = number_densities[He_I]

    logT = math.log(T)
    U_H_I = partition_funcs[H_I](logT)
    U_He_I = partition_funcs[He_I](logT)

    nH_I_div_U = nH_I / U_H_I
    nHe_I_div_U = nHe_I / U_He_I

    # Get H2 number density if available
    try:
        H2 = Species.from_string("H2")
        nH2 = number_densities.get(H2, 0.0)
    except (ValueError, KeyError):
        nH2 = 0.0

    nus = jnp.asarray(nus)
    alpha = jnp.zeros_like(nus)

    # H I bound-free
    alpha = alpha + H_I_bf(nus, T, nH_I, nHe_I, ne, 1.0 / U_H_I)

    # H- bound-free and free-free
    alpha = alpha + Hminus_bf(nus, T, nH_I_div_U, ne)
    alpha = alpha + Hminus_ff(nus, T, nH_I_div_U, ne)

    # H2+ bound-free + free-free
    alpha = alpha + H2plus_bf_ff(nus, T, nH_I, nH_II)

    # He- free-free
    alpha = alpha + He_minus_ff(nus, T, nHe_I_div_U, ne)

    # Positive ion free-free (hydrogenic)
    alpha = alpha + positive_ion_ff_absorption(nus, T, number_densities, ne)

    # Metal bound-free
    alpha = alpha + metal_bf_absorption(nus, T, number_densities)

    # Scattering
    alpha = alpha + electron_scattering(ne)
    alpha = alpha + rayleigh(nus, nH_I, nHe_I, nH2)

    return alpha
