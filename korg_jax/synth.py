"""High-level synthesis wrapper.

Ported from Korg.jl/src/synth.jl.
"""
from __future__ import annotations

from .synthesize import synthesize
from .abundances import format_A_X
from .wavelengths import Wavelengths
from .utils import apply_LSF, apply_rotation
from .atomic_data import atomic_symbols


def synth(*, Teff=None, logg=None, M_H=0.0, alpha_H=None,
          linelist=None,
          wavelengths=(5000, 6000),
          rectify=True,
          R=float("inf"),
          vsini=0.0,
          vmic=1.0,
          synthesize_kwargs=None,
          format_A_X_kwargs=None,
          **abundances):
    if alpha_H is None:
        alpha_H = M_H
    if synthesize_kwargs is None:
        synthesize_kwargs = {}
    if format_A_X_kwargs is None:
        format_A_X_kwargs = {}

    if Teff is None or logg is None:
        raise ValueError("Teff and logg must be provided")

    for key in abundances:
        if key not in atomic_symbols:
            msg = f"{key} is not a valid elemental symbol"
            if key.endswith("_H"):
                msg += "; use the bare symbol (e.g., Ca)"
            raise ValueError(msg)

    A_X = format_A_X(M_H, alpha_H, abundances, **format_A_X_kwargs)

    from .atmosphere import interpolate_marcs
    atm = interpolate_marcs(Teff, logg, A_X)

    if linelist is None:
        raise ValueError("linelist must be provided")

    wls = Wavelengths(wavelengths) if not isinstance(wavelengths, Wavelengths) else wavelengths
    spectrum = synthesize(atm, linelist, A_X, wls, vmic=vmic, **synthesize_kwargs)

    flux = spectrum.flux / spectrum.cntm if rectify else spectrum.flux
    flux = apply_LSF(flux, spectrum.wavelengths, R)
    if vsini > 0:
        flux = apply_rotation(flux, wls, vsini)

    return spectrum.wavelengths, flux, spectrum.cntm
