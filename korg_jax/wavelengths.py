"""Wavelength grid for spectral synthesis.

Ported from Korg.jl/src/wavelengths.jl.
"""
from __future__ import annotations
import numpy as np
from .constants import c_cgs


def _air_to_vacuum_angstrom(lam_A):
    """Air -> vacuum conversion (Birch & Downs 1994). Input & output in Angstrom."""
    s = 1e4 / lam_A
    n = (1.0
         + 0.00008336624212083
         + 0.02408926869968 / (130.1065924522 - s * s)
         + 0.0001599740894897 / (38.92568793293 - s * s))
    return lam_A * n


class Wavelengths:
    """Represents a (possibly non-contiguous) wavelength grid.

    Internally everything is stored in **cm**.
    """

    def __init__(self, wl_ranges, air_wavelengths=False):
        """
        Parameters
        ----------
        wl_ranges : list of np.ndarray
            Each array is a uniformly-spaced wavelength grid.
            If the first value >= 1 the grids are assumed to be in Angstrom
            and will be converted to cm.
        air_wavelengths : bool
            If True, convert air -> vacuum before storing.
        """
        # Detect Angstrom vs cm
        if wl_ranges[0][0] >= 1.0:
            wl_ranges = [r * 1e-8 for r in wl_ranges]

        if air_wavelengths:
            new_ranges = []
            for wls in wl_ranges:
                wls_A = wls * 1e8
                vac_start = _air_to_vacuum_angstrom(wls_A[0])
                vac_stop = _air_to_vacuum_angstrom(wls_A[-1])
                new_ranges.append(
                    np.linspace(vac_start, vac_stop, len(wls)) * 1e-8
                )
            wl_ranges = new_ranges

        self.wl_ranges = wl_ranges
        self.all_wls = np.concatenate(wl_ranges)
        if not np.all(np.diff(self.all_wls) > 0):
            raise ValueError("wl_ranges must be sorted and non-overlapping")
        self.all_freqs = c_cgs / self.all_wls[::-1]

    # ── convenience constructors ──

    @classmethod
    def from_tuple(cls, start, stop, step=None, air_wavelengths=False):
        """Create from (start, stop[, step]) in Angstrom.

        Default step is 0.01 Å for Angstrom input, 1e-10 cm for cm input.
        """
        if step is None:
            step = 0.01 if start >= 1 else 1e-10
        wls = np.arange(start, stop + step * 0.5, step)
        return cls([wls], air_wavelengths=air_wavelengths)

    @classmethod
    def from_array(cls, wls, air_wavelengths=False):
        """Wrap a pre-built 1-D wavelength array."""
        return cls([np.asarray(wls, dtype=np.float64)],
                   air_wavelengths=air_wavelengths)

    # ── array-like interface ──

    def __len__(self):
        return len(self.all_wls)

    def __getitem__(self, i):
        return self.all_wls[i]

    def __repr__(self):
        parts = []
        for r in self.wl_ranges:
            lo = int(round(r[0] * 1e8))
            hi = int(round(r[-1] * 1e8))
            parts.append(f"{lo}-{hi}")
        return f"Wavelengths({', '.join(parts)} Å)"

    # ── helpers ──

    def eachwindow(self):
        """Iterate (lambda_low, lambda_hi) in cm for each sub-range."""
        for r in self.wl_ranges:
            yield (r[0], r[-1])

    def subspectrum_indices(self):
        """Return list of (start, stop) index pairs into all_wls."""
        idx = 0
        out = []
        for r in self.wl_ranges:
            out.append((idx, idx + len(r)))
            idx += len(r)
        return out

    def to_jax(self):
        """Return JAX arrays of wavelengths and frequencies."""
        import jax.numpy as jnp
        return jnp.array(self.all_wls), jnp.array(self.all_freqs)
