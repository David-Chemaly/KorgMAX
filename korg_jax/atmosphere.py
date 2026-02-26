"""Model atmosphere dataclass and MARCS/PHOENIX readers.

Ported from Korg.jl/src/atmosphere.jl.
"""
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from .constants import kboltz_cgs, G_cgs, solar_mass_cgs
from .abundances import get_metals_H, get_alpha_H, default_alpha_elements
from .interpolation import lazy_multilinear_interpolation


@dataclass
class ModelAtmosphere:
    """Model atmosphere with arrays for each layer quantity.

    All arrays are shape (n_layers,).
    """
    tau_ref: np.ndarray          # reference optical depth (dimensionless)
    z: np.ndarray                # height relative to photosphere (cm)
    temp: np.ndarray             # temperature (K)
    electron_number_density: np.ndarray  # cm^-3
    number_density: np.ndarray   # total number density cm^-3
    reference_wavelength: float  # cm (usually 5e-5 = 5000 Å)
    is_spherical: bool = False
    R: Optional[float] = None   # photospheric radius for spherical (cm)

    @property
    def n_layers(self) -> int:
        return len(self.temp)

    def to_jax(self):
        """Convert arrays to JAX arrays."""
        import jax.numpy as jnp
        return {
            'tau_ref': jnp.array(self.tau_ref),
            'z': jnp.array(self.z),
            'temp': jnp.array(self.temp),
            'electron_number_density': jnp.array(self.electron_number_density),
            'number_density': jnp.array(self.number_density),
        }


def read_model_atmosphere(fname, format="marcs"):
    """Parse a model atmosphere file. Returns ModelAtmosphere."""
    if format == "marcs" or fname.endswith(".mod"):
        return _read_marcs(fname)
    elif format == "phoenix" or fname.endswith(".fits"):
        return _read_phoenix(fname)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _read_marcs(fname):
    """Read a MARCS .mod file."""
    with open(fname, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip('\n') for l in lines]

    # Find radius
    R_line = None
    for i, l in enumerate(lines):
        if 'adius' in l:  # {rR}adius
            R_line = i
            break
    if R_line is None:
        raise ValueError("Can't parse .mod file: can't detect radius.")
    R = float(lines[R_line].split()[0])
    planar = (R == 1.0)

    # Find number of layers
    nlayers = None
    for i, l in enumerate(lines):
        if 'Number of depth points' in l:
            nlayers = int(l.split()[0])
            break
    if nlayers is None:
        raise ValueError("Can't parse .mod file: can't detect number of layers.")

    # Find header
    header = None
    for i, l in enumerate(lines):
        if 'lgTauR' in l:
            header = i
            break
    if header is None:
        raise ValueError("Can't parse .mod file: can't find header.")

    tau_refs = []
    zs = []
    temps = []
    n_es = []
    ns = []

    for line in lines[header + 1:header + 1 + nlayers]:
        logτ5 = float(line[10:17])
        depth = float(line[18:28])
        temp = float(line[29:36])
        Pe = float(line[38:48])
        Pg = float(line[48:60])

        Pe = max(Pe, 0.0)
        Pg = max(Pg, 0.0)

        ne = Pe / (temp * kboltz_cgs)
        n = Pg / (temp * kboltz_cgs)

        tau_refs.append(10**logτ5)
        zs.append(-depth)
        temps.append(temp)
        n_es.append(ne)
        ns.append(n)

    atm = ModelAtmosphere(
        tau_ref=np.array(tau_refs),
        z=np.array(zs),
        temp=np.array(temps),
        electron_number_density=np.array(n_es),
        number_density=np.array(ns),
        reference_wavelength=5e-5,  # 5000 Å
        is_spherical=not planar,
        R=None if planar else R,
    )
    return atm


def _read_phoenix(fname):
    """Read a PHOENIX FITS atmosphere file."""
    from astropy.io import fits
    with fits.open(fname) as hdul:
        Teff = hdul[0].header.get('PHXTEFF', 5777)
        data = hdul[1].data
        tau = data['tau']
        T = data['temp']
        Pgas = data['pgas']
        Pe = data['pe']

    n = Pgas / (kboltz_cgs * T)
    ne = Pe / (kboltz_cgs * T)

    ref_wl = 12e-5 if Teff < 5000 else 5e-5

    # Skip first layer (tau=0)
    return ModelAtmosphere(
        tau_ref=tau[1:],
        z=np.full(len(tau) - 1, np.nan),
        temp=T[1:],
        electron_number_density=ne[1:],
        number_density=n[1:],
        reference_wavelength=ref_wl,
    )


# ── MARCS interpolation (requires external grids) ───────────────────────────

def interpolate_marcs(Teff, logg, A_X=None, M_H=0.0, alpha_m=0.0, C_m=0.0,
                      spherical=None, clamp_abundances=False,
                      perturb_at_grid_values=True, archives=None):
    """Interpolate MARCS atmospheres from precomputed grids.

    This function requires external MARCS grids (not included in this repo).
    Pass `archives=(nodes, grid)` or a tuple of archives matching the Julia API.
    """
    if archives is None:
        raise FileNotFoundError("MARCS atmosphere grids are not bundled. Provide `archives=` with grid data.")

    if A_X is not None:
        M_H = get_metals_H(A_X, alpha_elements=[6] + default_alpha_elements)
        alpha_H = get_alpha_H(A_X)
        alpha_m = alpha_H - M_H
        C_H = A_X[5] - A_X[5]
        C_m = C_H - M_H

    if spherical is None:
        spherical = logg < 3.5

    reference_wavelength = 5e-5
    nodes, grid = archives
    params = [Teff, logg, M_H, alpha_m, C_m]
    param_names = ["Teff", "log(g)", "[M/H]", "[alpha/M]", "[C/metals]"]

    atm_quants = lazy_multilinear_interpolation(
        params, nodes, grid,
        param_names=param_names,
        perturb_at_grid_values=perturb_at_grid_values,
    )

    nanmask = ~np.isnan(atm_quants[:, 3])
    tau_ref = atm_quants[nanmask, 3]
    z = np.sinh(atm_quants[nanmask, 4])
    temp = atm_quants[nanmask, 0]
    ne = np.exp(atm_quants[nanmask, 1])
    n = np.exp(atm_quants[nanmask, 2])

    if spherical:
        R = math.sqrt(G_cgs * solar_mass_cgs / (10 ** logg))
        return ModelAtmosphere(
            tau_ref=tau_ref,
            z=z,
            temp=temp,
            electron_number_density=ne,
            number_density=n,
            reference_wavelength=reference_wavelength,
            is_spherical=True,
            R=R,
        )

    return ModelAtmosphere(
        tau_ref=tau_ref,
        z=z,
        temp=temp,
        electron_number_density=ne,
        number_density=n,
        reference_wavelength=reference_wavelength,
    )
