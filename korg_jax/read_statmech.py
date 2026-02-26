"""Load partition functions and equilibrium constants from Korg data files.

Ported from Korg.jl/src/read_statmech_quantities.jl.

Returns CubicSpline objects (callable) matching the Julia behaviour:
- Atomic partition functions: CubicSpline over ln(T), flat extrapolation
- Molecular partition functions: CubicSpline over ln(T), flat extrapolation
- Equilibrium constants (Barklem-Collet): CubicSpline over ln(T), flat extrapolation
- Equilibrium constants (polyatomic): callable closures computing logK(lnT)
"""
from __future__ import annotations
import os
import csv
import math
import numpy as np
from typing import Dict, Optional
from .species import Species, Formula
from .atomic_data import atomic_symbols, atomic_masses, MAX_ATOMIC_NUMBER
from .constants import kboltz_cgs, kboltz_eV, hplanck_cgs, c_cgs, hplanck_eV
from .cubic_splines import CubicSpline


# Nuclear spin degeneracy for the most abundant isotope of each element.
# Used to convert ExoMol partition functions from "physics" to "astrophysics" convention.
# Computed from NIST isotopic data (isotopic_data.jl).
_nuclear_spin_degeneracy_most_abundant = {
    1: 2, 2: 1, 3: 4, 4: 4, 5: 4, 6: 1, 7: 3, 8: 1, 9: 2, 10: 1,
    11: 4, 12: 1, 13: 6, 14: 1, 15: 2, 16: 1, 17: 4, 18: 1, 19: 4, 20: 1,
    21: 8, 22: 1, 23: 8, 24: 1, 25: 6, 26: 1, 27: 8, 28: 1, 29: 4, 30: 1,
    31: 4, 32: 1, 33: 4, 34: 1, 35: 4, 36: 1, 37: 6, 38: 1, 39: 2, 40: 1,
    41: 10, 42: 1, 44: 1, 45: 2, 46: 1, 47: 2, 48: 1, 49: 10, 50: 1,
    51: 6, 52: 1, 53: 6, 54: 1, 55: 8, 56: 1, 57: 8, 58: 1, 59: 6, 60: 1,
    62: 1, 63: 6, 64: 1, 65: 4, 66: 1, 67: 8, 68: 1, 69: 2, 70: 1,
    71: 8, 72: 1, 73: 8, 74: 1, 75: 6, 76: 1, 77: 4, 78: 1, 79: 4, 80: 1,
    81: 2, 82: 1, 83: 10, 90: 1, 91: 4, 92: 1,
}


def _resolve_data_dir():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(base, "data")
    if os.path.exists(candidate):
        return candidate
    return os.path.join(base, os.pardir, "data")


def setup_ionization_energies(fname=None):
    """Parse ionization energies table. Returns dict: Z -> [chi1, chi2, chi3] in eV."""
    if fname is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fname = os.path.join(base, "data", "barklem_collet_2016",
                             "BarklemCollet2016-ionization_energies.dat")

    result = {}
    with open(fname, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            toks = line.split()
            if len(toks) >= 3:
                Z = int(toks[0])
                result[Z] = [float(x) for x in toks[2:]]
    return result


def load_atomic_partition_functions(fname=None):
    """Load tabulated atomic partition functions from HDF5.

    Returns dict: Species -> CubicSpline over ln(T).
    Matches Julia: ``CubicSpline(logTs, values)`` with flat extrapolation.
    """
    if fname is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fname = os.path.join(base, "data", "atomic_partition_funcs", "partition_funcs.h5")

    import h5py
    partition_funcs = {}

    with h5py.File(fname, 'r') as f:
        logT_min = float(f['logT_min'][()])
        logT_step = float(f['logT_step'][()])
        logT_max = float(f['logT_max'][()])
        logTs = np.arange(logT_min, logT_max + logT_step / 2, logT_step)

        for elem in atomic_symbols:
            for ion_str in ["I", "II", "III"]:
                if elem == "H" and ion_str != "I":
                    continue
                if elem == "He" and ion_str == "III":
                    continue
                spec_str = f"{elem} {ion_str}"
                spec = Species.from_string(spec_str)
                if spec_str in f:
                    vals = np.array(f[spec_str])
                    # Julia uses CubicSpline(logTs, values) — no extrapolation (throws error).
                    # JAX version uses flat extrapolation (safe, no error).
                    partition_funcs[spec] = CubicSpline(logTs, vals)

        # Bare nuclei: partition function = 1
        all_ones = np.ones(len(logTs))
        partition_funcs[Species.from_string("H II")] = CubicSpline(logTs, all_ones)
        partition_funcs[Species.from_string("He III")] = CubicSpline(logTs, all_ones)

    return partition_funcs


def load_molecular_partition_functions(fname=None):
    """Load Barklem & Collet molecular partition functions.

    Returns dict: Species -> CubicSpline over ln(T).
    Matches Julia: ``CubicSpline(log.(temperatures), vals; extrapolate=true)``.
    """
    if fname is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fname = os.path.join(base, "data", "barklem_collet_2016",
                             "BarklemCollet2016-molecular_partition.dat")

    temperatures = []
    data_pairs = []

    with open(fname, 'r') as f:
        for line in f:
            if len(line) >= 9 and "T [K]" in line:
                temps = [float(x) for x in line[9:].split()]
                temperatures.extend(temps)
            elif line.startswith('#'):
                continue
            else:
                toks = line.split()
                if not toks:
                    continue
                species_code = toks[0]
                if species_code.startswith("D_"):
                    continue
                try:
                    spec = Species.from_string(species_code)
                    vals = [float(x) for x in toks[1:]]
                    data_pairs.append((spec, vals))
                except (ValueError, KeyError):
                    continue

    lnTs = np.log(np.array(temperatures))
    result = {}
    for spec, vals in data_pairs:
        # Julia uses extrapolate=true for molecular partition funcs because they
        # are only defined up to ~10,000 K. Flat extrapolation is safe since
        # molecules are destroyed at higher temperatures anyway.
        result[spec] = CubicSpline(lnTs, np.array(vals), extrapolate=True)
    return result


def load_exomol_partition_functions(fname=None):
    """Load ExoMol polyatomic partition functions from HDF5.

    Returns dict: Species -> CubicSpline over ln(T).
    Divides by total nuclear spin degeneracy to convert from "physics"
    to "astrophysics" convention (matching Julia load_exomol_partition_functions).
    """
    if fname is None:
        data_dir = _resolve_data_dir()
        fname = os.path.join(data_dir, "polyatomic_partition_funcs",
                             "polyatomic_partition_funcs.h5")

    if not os.path.exists(fname):
        return {}

    import h5py
    result = {}

    with h5py.File(fname, 'r') as f:
        for group_name in f:
            group = f[group_name]
            spec = Species.from_string(group_name)

            # Total nuclear spin degeneracy for the most abundant isotopologue
            atoms = spec.formula.get_atoms()
            total_g_ns = 1
            for Z in atoms:
                g_ns = _nuclear_spin_degeneracy_most_abundant.get(Z, 1)
                total_g_ns *= g_ns

            Ts = np.array(group['temp'])
            Us = np.array(group['partition_function'])

            # Filter out T=0 if present (log(0) = -inf)
            mask = Ts > 0
            Ts = Ts[mask]
            Us = Us[mask]

            lnTs = np.log(Ts)
            result[spec] = CubicSpline(lnTs, Us / total_g_ns, extrapolate=True)

    return result


def load_equilibrium_constants(fname=None):
    """Load Barklem-Collet molecular equilibrium constants from HDF5.

    Returns dict: Species -> CubicSpline over ln(T), giving log10(K) in
    partial pressure form.

    Matches Julia: ``CubicSpline(lnTs, logKs; extrapolate=true)``.
    As recommended by Aquilina+ 2024, the C2 equilibrium constant is corrected
    to reflect the dissociation energy from Visser+ 2019.
    """
    if fname is None:
        data_dir = _resolve_data_dir()
        fname = os.path.join(data_dir, "barklem_collet_2016", "barklem_collet_ks.h5")

    import h5py
    result = {}
    with h5py.File(fname, 'r') as f:
        mols = [s.decode() if isinstance(s, bytes) else s for s in f['mols'][:]]
        lnTs_all = f['lnTs'][:]
        logKs_all = f['logKs'][:]

        mols_len = len(mols)
        lnTs_mols_axis = 0 if lnTs_all.shape[0] == mols_len else 1
        logKs_mols_axis = 0 if logKs_all.shape[0] == mols_len else 1

        # Correct C2 equilibrium constant (Visser+ 2019, Aquilina+ 2024)
        C2_str = None
        for i, mol_str in enumerate(mols):
            try:
                spec = Species.from_string(mol_str)
                if str(spec) == "C2":
                    C2_str = mol_str
                    break
            except (ValueError, KeyError):
                pass

        if C2_str is not None:
            C2_idx = mols.index(C2_str)
            if lnTs_mols_axis == 0:
                C2_lnTs = lnTs_all[C2_idx]
            else:
                C2_lnTs = lnTs_all[:, C2_idx]
            BC_C2_E0 = 6.371
            Visser_C2_E0 = 6.24
            correction = (np.log10(np.e) / (kboltz_eV * np.exp(C2_lnTs))
                          * (Visser_C2_E0 - BC_C2_E0))
            if logKs_mols_axis == 0:
                logKs_all[C2_idx] += correction
            else:
                logKs_all[:, C2_idx] += correction

        for i, mol_str in enumerate(mols):
            try:
                spec = Species.from_string(mol_str)
            except (ValueError, KeyError):
                continue

            if lnTs_mols_axis == 0:
                lnTs_row = lnTs_all[i]
            else:
                lnTs_row = lnTs_all[:, i]

            if logKs_mols_axis == 0:
                logKs_row = logKs_all[i]
            else:
                logKs_row = logKs_all[:, i]

            mask = np.isfinite(lnTs_row)
            result[spec] = CubicSpline(lnTs_row[mask], logKs_row[mask], extrapolate=True)

    return result


def _compute_polyatomic_equilibrium_constants(partition_funcs, data_dir=None):
    """Compute equilibrium constants for polyatomic molecules from atomization energies.

    Matches Julia: polyatomic_Ks computation in setup_partition_funcs_and_equilibrium_constants().

    Returns dict: Species -> callable(lnT) -> log10(K_p)
    """
    if data_dir is None:
        data_dir = _resolve_data_dir()

    csv_path = os.path.join(data_dir, "polyatomic_partition_funcs", "atomization_energies.csv")
    if not os.path.exists(csv_path):
        return {}

    result = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            spec_str = row['spec']
            D00_kJmol = float(row['energy'])
            D00_eV = D00_kJmol * 0.01036  # kJ/mol -> eV

            try:
                spec = Species.from_string(spec_str)
            except (ValueError, KeyError):
                continue

            # Verify we have partition functions for this molecule and its atoms
            Zs = spec.formula.get_atoms()
            atom_specs = [Species(Formula.from_Z(Z), 0) for Z in Zs]
            if spec not in partition_funcs:
                continue
            if not all(s in partition_funcs for s in atom_specs):
                continue

            # Build closure matching Julia's polyatomic logK calculation
            def make_logK(spec=spec, atom_specs=atom_specs, Zs=Zs, D00=D00_eV,
                          pfuncs=partition_funcs):
                def logK(lnT):
                    T = math.exp(lnT)

                    # Ratio of partition functions: prod(U_atoms) / U_molecule
                    U_atoms = 1.0
                    for s in atom_specs:
                        U_atoms *= float(pfuncs[s](lnT))
                    U_mol = float(pfuncs[spec](lnT))
                    log_Us_ratio = math.log10(U_atoms / U_mol)

                    # Mass ratio
                    log_masses_ratio = (sum(math.log10(atomic_masses[Z - 1]) for Z in Zs)
                                        - math.log10(spec.get_mass()))

                    # Translational partition function factor
                    log_translational_U_factor = 1.5 * math.log10(
                        2.0 * math.pi * kboltz_cgs * T / hplanck_cgs ** 2)

                    # log10 number-density equilibrium constant
                    n_at = len(Zs)
                    log_nK = ((n_at - 1) * log_translational_U_factor
                              + 1.5 * log_masses_ratio
                              + log_Us_ratio
                              - D00 / (kboltz_eV * T * math.log(10)))

                    # Convert to partial-pressure form
                    log_pK = log_nK + (n_at - 1) * math.log10(kboltz_cgs * T)
                    return log_pK
                return logK

            result[spec] = make_logK()

    return result


def setup_all(data_dir=None):
    """Load all statmech data.

    Returns (ionization_energies, partition_funcs, equilibrium_constants).

    partition_funcs: dict Species -> CubicSpline (callable)
    equilibrium_constants: dict Species -> callable(lnT) -> log10(K_p)
    """
    if data_dir is None:
        data_dir = _resolve_data_dir()

    ion_energies = setup_ionization_energies(
        os.path.join(data_dir, "barklem_collet_2016",
                     "BarklemCollet2016-ionization_energies.dat"))

    atomic_pf = load_atomic_partition_functions(
        os.path.join(data_dir, "atomic_partition_funcs", "partition_funcs.h5"))

    mol_pf = load_molecular_partition_functions(
        os.path.join(data_dir, "barklem_collet_2016",
                     "BarklemCollet2016-molecular_partition.dat"))

    exomol_pf = load_exomol_partition_functions(
        os.path.join(data_dir, "polyatomic_partition_funcs",
                     "polyatomic_partition_funcs.h5"))

    partition_funcs = {**atomic_pf, **mol_pf, **exomol_pf}

    # Barklem-Collet diatomic equilibrium constants
    bc_eq = load_equilibrium_constants(
        os.path.join(data_dir, "barklem_collet_2016", "barklem_collet_ks.h5"))

    # Polyatomic equilibrium constants (computed from partition functions)
    poly_eq = _compute_polyatomic_equilibrium_constants(partition_funcs, data_dir)

    equilibrium_constants = {**bc_eq, **poly_eq}

    return ion_energies, partition_funcs, equilibrium_constants
