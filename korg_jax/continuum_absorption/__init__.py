"""Continuum (bound-free + free-free + scattering) opacity.

Ported from Korg.jl/src/ContinuumAbsorption/.
"""
from __future__ import annotations
import math
import numpy as np

from ..species import Species, Formula
from .scattering import electron_scattering, rayleigh
from .hydrogenic import hydrogenic_ff_absorption
from .hydrogen import H_I_bf, Hminus_bf, Hminus_ff, H2plus_bf_ff
from .helium import Heminus_ff as He_minus_ff
from .metals import metal_bf_absorption, positive_ion_ff_absorption

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
