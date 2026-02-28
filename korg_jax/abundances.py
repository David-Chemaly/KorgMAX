"""Abundance formatting utilities.

Ported from Korg.jl/src/abundances.jl.
"""
import numpy as np
from .atomic_data import (
    MAX_ATOMIC_NUMBER, atomic_numbers, default_solar_abundances,
)

# Alpha elements: O, Ne, Mg, Si, S, Ar, Ca, Ti (even Z from 8-22)
default_alpha_elements = [8, 10, 12, 14, 16, 18, 20, 22]


def format_A_X(default_metals_H=0.0, default_alpha_H=None, abundances=None,
               solar_relative=True, solar_abundances=None, alpha_elements=None):
    """Return a 92-element A(X) abundance vector.

    Parameters
    ----------
    default_metals_H : float or dict
        [metals/H] for elements heavier than He.
        If a dict is passed, it is treated as per-element abundance overrides
        (convenience shorthand for ``format_A_X(0.0, abundances=dict)``).
    default_alpha_H : float or None
        [alpha/H]. Defaults to default_metals_H.
    abundances : dict or None
        Per-element overrides. Keys can be int (Z) or str (symbol).
        Values are [X/H] if solar_relative else A(X).
    solar_relative : bool
        Interpret *abundances* values as [X/H] (True) or A(X) (False).
    solar_abundances : array-like or None
        Solar scale (length-92). Defaults to Bergemann 2025.
    alpha_elements : list[int] or None
        Atomic numbers of alpha elements.

    Returns
    -------
    np.ndarray of shape (92,)
    """
    # Convenience: format_A_X({"Ni": 1.0}) -> format_A_X(0.0, abundances={"Ni": 1.0})
    if isinstance(default_metals_H, dict):
        abundances = default_metals_H
        default_metals_H = 0.0
    if default_alpha_H is None:
        default_alpha_H = default_metals_H
    if abundances is None:
        abundances = {}
    if solar_abundances is None:
        solar_abundances = default_solar_abundances
    if alpha_elements is None:
        alpha_elements = default_alpha_elements

    # Normalise keys to int (Z)
    clean = {}
    for el, val in abundances.items():
        if isinstance(el, str):
            if el not in atomic_numbers:
                raise ValueError(f"{el} is not a valid atomic symbol.")
            Z = atomic_numbers[el]
            if Z in abundances:
                raise ValueError(
                    f"Abundance of {el} specified by both symbol and Z.")
            clean[Z] = val
        elif isinstance(el, (int, np.integer)):
            if not 1 <= el <= MAX_ATOMIC_NUMBER:
                raise ValueError(f"Z={el} is not a supported atomic number.")
            clean[int(el)] = val
        else:
            raise ValueError(f"{el} isn't a valid element key.")

    correct_H = 0.0 if solar_relative else 12.0
    if 1 in clean and clean[1] != correct_H:
        raise ValueError("Cannot set H abundance directly; adjust metallicity instead.")

    A_X = np.empty(MAX_ATOMIC_NUMBER, dtype=np.float64)
    alpha_set = set(alpha_elements)

    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        if Z == 1:
            A_X[0] = 12.0
        elif Z in clean:
            if solar_relative:
                A_X[Z - 1] = clean[Z] + solar_abundances[Z - 1]
            else:
                A_X[Z - 1] = clean[Z]
        elif Z in alpha_set:
            A_X[Z - 1] = solar_abundances[Z - 1] + default_alpha_H
        else:
            delta = default_metals_H if Z >= 3 else 0.0
            A_X[Z - 1] = solar_abundances[Z - 1] + delta

    return A_X


def _get_multi_X_H(A_X, Zs, solar_abundances):
    """Compute [I+J+.../H] for a set of atomic numbers *Zs*."""
    A_mX = np.log10(sum(10 ** A_X[Z - 1] for Z in Zs))
    A_mX_solar = np.log10(sum(10 ** solar_abundances[Z - 1] for Z in Zs))
    return A_mX - A_mX_solar


def get_metals_H(A_X, solar_abundances=None, ignore_alpha=True,
                 alpha_elements=None):
    """Calculate [metals/H]."""
    if solar_abundances is None:
        solar_abundances = default_solar_abundances
    if alpha_elements is None:
        alpha_elements = default_alpha_elements
    alpha_set = set(alpha_elements)
    if ignore_alpha:
        els = [Z for Z in range(3, MAX_ATOMIC_NUMBER + 1) if Z not in alpha_set]
    else:
        els = list(range(3, MAX_ATOMIC_NUMBER + 1))
    return _get_multi_X_H(A_X, els, solar_abundances)


def get_alpha_H(A_X, solar_abundances=None, alpha_elements=None):
    """Calculate [alpha/H]."""
    if solar_abundances is None:
        solar_abundances = default_solar_abundances
    if alpha_elements is None:
        alpha_elements = default_alpha_elements
    return _get_multi_X_H(A_X, alpha_elements, solar_abundances)
