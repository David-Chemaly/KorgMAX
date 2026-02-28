"""Thomson + Rayleigh scattering.

Ported from Korg.jl/src/ContinuumAbsorption/scattering.jl.
"""
import math
import numpy as np

from ..constants import (
    electron_charge_cgs, electron_mass_cgs, c_cgs,
    hplanck_eV, Rydberg_eV,
)

# Thomson cross-section (cm^2)
_sigma_th = 6.65246e-25


def electron_scattering(ne):
    """Thomson scattering coefficient (wavelength-independent scalar)."""
    return (8.0 * math.pi / 3.0
            * (electron_charge_cgs ** 2 / (electron_mass_cgs * c_cgs ** 2)) ** 2
            * ne)


def rayleigh(nus, nH_I, nHe_I, nH2):
    """Rayleigh scattering by H I, He I, and H2."""
    nus = np.asarray(nus)

    E_2Ryd_2 = (hplanck_eV * nus / (2.0 * Rydberg_eV)) ** 2
    E_2Ryd_4 = E_2Ryd_2 ** 2
    E_2Ryd_6 = E_2Ryd_2 * E_2Ryd_4
    E_2Ryd_8 = E_2Ryd_4 ** 2

    sigma_H = (20.24 * E_2Ryd_4 + 239.2 * E_2Ryd_6 + 2256.0 * E_2Ryd_8) * _sigma_th
    sigma_He = (1.913 * E_2Ryd_4 + 4.52 * E_2Ryd_6 + 7.90 * E_2Ryd_8) * _sigma_th

    alpha_H_He = nH_I * sigma_H + nHe_I * sigma_He

    inv_lam2 = (nus / (1e8 * c_cgs)) ** 2
    inv_lam4 = inv_lam2 ** 2
    inv_lam6 = inv_lam2 * inv_lam4
    inv_lam8 = inv_lam4 ** 2
    alpha_H2 = (8.14e-13 * inv_lam4 + 1.28e-6 * inv_lam6 + 1.61 * inv_lam8) * nH2

    return alpha_H_He + alpha_H2
