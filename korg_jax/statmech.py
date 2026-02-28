"""Chemical equilibrium solver (Saha + molecular equilibrium).

Ported from Korg.jl/src/statmech.jl.

Uses pure NumPy with a finite-difference Newton solver.  A pre-computed
context (``_ChemEqContext``) avoids redundant Species/Formula object creation
and molecule-data packing when the solver is called for every atmosphere layer.
"""
from __future__ import annotations

import math
import numpy as np

from .constants import (
    kboltz_cgs, kboltz_eV, hplanck_cgs, electron_mass_cgs,
    bohr_radius_cgs, RydbergH_eV, eV_to_cgs, electron_charge_cgs,
)
from .atomic_data import MAX_ATOMIC_NUMBER


# ── Helpers ───────────────────────────────────────────────────────────────

def translational_U(m, T):
    """Translational partition function factor (pure Python, scalar)."""
    k = kboltz_cgs
    h = hplanck_cgs
    return (2.0 * math.pi * m * k * T / h ** 2) ** 1.5


def saha_ion_weights_np(T, ne, atom_Z, ionization_energies, partition_funcs):
    """Compute (wII, wIII) = ratios of ionised/neutral number densities."""
    from .species import Species, Formula
    chi = ionization_energies[atom_Z]
    chi_I = chi[0]
    f = Formula.from_Z(atom_Z)
    logT = math.log(T)
    UI = partition_funcs[Species(f, 0)](logT)
    UII = partition_funcs[Species(f, 1)](logT)
    k = kboltz_eV
    transU = translational_U(electron_mass_cgs, T)
    wII = 2.0 / ne * (UII / UI) * transU * math.exp(-chi_I / (k * T))
    if atom_Z == 1:
        wIII = 0.0
    else:
        chi_II = chi[1]
        UIII = partition_funcs[Species(f, 2)](logT)
        wIII = wII * 2.0 / ne * (UIII / UII) * transU * math.exp(-chi_II / (k * T))
    return wII, wIII


# ── Pre-computed context ──────────────────────────────────────────────────

class _ChemEqContext:
    """Invariant data for chemical equilibrium (build once, reuse per layer)."""

    def __init__(self, abs_ab, ionization_energies, partition_funcs,
                 log_equilibrium_constants):
        from .species import Species, Formula

        self.abs_ab = np.maximum(
            np.asarray(abs_ab, dtype=np.float64), 1e-99
        )

        # Pre-build Species/Formula objects for all 92 elements
        formulas = [Formula.from_Z(Z) for Z in range(1, MAX_ATOMIC_NUMBER + 1)]
        self.species_I = [Species(f, 0) for f in formulas]
        self.species_II = [Species(f, 1) for f in formulas]
        self.species_III = [Species(f, 2) for f in formulas]

        # Pre-extract partition function callables (avoid dict lookup per call)
        # H III doesn't exist in partition functions; use None sentinel
        self.pf_I = [partition_funcs[s] for s in self.species_I]
        self.pf_II = [partition_funcs[s] for s in self.species_II]
        self.pf_III = [partition_funcs.get(s) for s in self.species_III]

        # Ionization energies as flat arrays
        self.chi_I = np.array(
            [ionization_energies[Z][0] for Z in range(1, MAX_ATOMIC_NUMBER + 1)],
            dtype=np.float64,
        )
        self.chi_II = np.zeros(MAX_ATOMIC_NUMBER, dtype=np.float64)
        for Z in range(2, MAX_ATOMIC_NUMBER + 1):
            self.chi_II[Z - 1] = ionization_energies[Z][1]

        # Molecule invariant data (atoms, charges — same across all layers)
        molecules = list(log_equilibrium_constants.keys())
        n_mol = len(molecules)
        self.molecules = molecules
        self.n_mol = n_mol

        mol_atom_Z = np.zeros((n_mol, 6), dtype=np.int32)
        mol_charges = np.zeros(n_mol, dtype=np.int32)
        mol_atom_counts = np.zeros(n_mol, dtype=np.int32)
        lec_fns = []

        for i, mol in enumerate(molecules):
            atoms = mol.formula.get_atoms()
            mol_charges[i] = mol.charge
            mol_atom_counts[i] = len(atoms)
            for j, a in enumerate(atoms):
                mol_atom_Z[i, j] = a
            lec_fns.append(log_equilibrium_constants[mol])

        self.mol_atom_Z = mol_atom_Z
        self.mol_charges = mol_charges
        self.mol_atom_counts = mol_atom_counts
        self.lec_fns = lec_fns

        # Pre-compute masks for vectorized residual
        self.mol_active = mol_atom_Z > 0                          # (n_mol, 6)
        self.mol_safe_idx = np.where(self.mol_active,
                                     mol_atom_Z - 1, 0)           # (n_mol, 6)
        self.mol_first_Z_idx = np.maximum(mol_atom_Z[:, 0] - 1, 0)  # (n_mol,)
        self.mol_charged_mask = mol_charges == 1                   # (n_mol,)


# ── Vectorized residual ──────────────────────────────────────────────────

def _residual_vec(x, n_tot, abs_ab, wII_ne, wIII_ne2, log_nKs, ctx):
    """Pure NumPy residual with vectorized molecule contributions."""
    N = MAX_ATOMIC_NUMBER
    ne = abs(x[N]) * n_tot * 1e-5
    atom_nd = abs_ab * (n_tot - ne)
    neutral_nd = atom_nd * np.abs(x[:N])

    wII = wII_ne / ne
    wIII = wIII_ne2 / ne ** 2

    F_atoms = atom_nd - (1.0 + wII + wIII) * neutral_nd
    F_elec = np.sum((wII + 2.0 * wIII) * neutral_nd) - ne

    if ctx.n_mol > 0:
        log_neutral = np.log10(np.maximum(neutral_nd, 1e-300))

        # Vectorized: gather log(neutral) for each atom slot
        log_contribs = np.where(ctx.mol_active,
                                log_neutral[ctx.mol_safe_idx], 0.0)
        log_sums = log_contribs.sum(axis=1)

        # Charged diatomic correction
        charged_corr = np.where(
            ctx.mol_charged_mask,
            np.log10(np.maximum(wII_ne[ctx.mol_first_Z_idx] / ne, 1e-300)),
            0.0,
        )
        log_sums += charged_corr

        n_mol_vals = 10.0 ** (log_sums - log_nKs)

        # Scatter-subtract from F_atoms (only 6 iterations max)
        for k in range(6):
            Z_k = ctx.mol_atom_Z[:, k]
            mask = Z_k > 0
            if not np.any(mask):
                break
            np.subtract.at(F_atoms, Z_k[mask] - 1, n_mol_vals[mask])

        F_elec += np.sum(n_mol_vals[ctx.mol_charged_mask])

    # Normalise
    F_atoms = np.where(atom_nd > 0, F_atoms / atom_nd, 0.0)
    F_elec = F_elec / (ne * 1e-5)

    result = np.empty(N + 1, dtype=np.float64)
    result[:N] = F_atoms
    result[N] = F_elec
    return result


# ── Newton solver ─────────────────────────────────────────────────────────

def _newton_solve_np(residual_fn, x0, max_iter=1000, tol=1e-8):
    """Damped Newton–Broyden solver (pure NumPy).

    Computes a full finite-difference Jacobian on the first iteration, then
    uses Broyden rank-1 updates on subsequent iterations to avoid the
    expensive 93-column finite-difference recomputation.
    """
    N = len(x0)
    x = x0.copy()
    F = residual_fn(x)
    eps = 1.5e-8
    J = None

    for _ in range(max_iter):
        if np.max(np.abs(F)) <= tol:
            break

        if J is None:
            # First iteration: full finite-difference Jacobian
            J = np.empty((N, N), dtype=np.float64)
            for j in range(N):
                h = eps * (1.0 + abs(x[j]))
                xj_orig = x[j]
                x[j] = xj_orig + h
                Fp = residual_fn(x)
                J[:, j] = (Fp - F) / h
                x[j] = xj_orig

        dx = np.linalg.solve(J, -F)

        max_step = np.max(np.abs(dx))
        if max_step > 1.0:
            dx *= 1.0 / max_step

        x_new = x + dx
        F_new = residual_fn(x_new)

        # Broyden rank-1 update
        dF = F_new - F - J @ dx
        dx_dot = np.dot(dx, dx)
        if dx_dot > 0:
            J += np.outer(dF, dx) / dx_dot

        x = x_new
        F = F_new

    return x


# ── Fast per-layer solver (uses pre-computed context) ────────────────────

def _solve_layer(ctx, T, n_tot, model_ne):
    """Solve chemical equilibrium for one atmosphere layer.

    Returns (ne, number_densities_dict).
    """
    logT = math.log(T)
    transU = translational_U(electron_mass_cgs, T)
    kT = kboltz_eV * T

    # Saha weights at ne=1 (use pre-extracted partition functions)
    wII_ne = np.empty(MAX_ATOMIC_NUMBER, dtype=np.float64)
    wIII_ne2 = np.zeros(MAX_ATOMIC_NUMBER, dtype=np.float64)

    for idx in range(MAX_ATOMIC_NUMBER):
        UI = ctx.pf_I[idx](logT)
        UII = ctx.pf_II[idx](logT)
        wII = 2.0 * (UII / UI) * transU * math.exp(-ctx.chi_I[idx] / kT)
        wII_ne[idx] = wII
        pf3 = ctx.pf_III[idx]
        if pf3 is not None:
            UIII = pf3(logT)
            wIII_ne2[idx] = wII * 2.0 * (UIII / UII) * transU * math.exp(-ctx.chi_II[idx] / kT)

    # Compute log_nKs (only T-dependent quantity for molecules)
    log_kT = math.log10(kboltz_cgs * T)
    log_nKs = np.empty(ctx.n_mol, dtype=np.float64)
    for i in range(ctx.n_mol):
        log_pK = float(ctx.lec_fns[i](logT))
        log_nKs[i] = log_pK - (ctx.mol_atom_counts[i] - 1) * log_kT

    # Initial guess from Saha weights
    wII_model = wII_ne / model_ne
    wIII_model = wIII_ne2 / model_ne ** 2
    neutral_frac_guess = 1.0 / (1.0 + wII_model + wIII_model)

    x0 = np.empty(MAX_ATOMIC_NUMBER + 1, dtype=np.float64)
    x0[:MAX_ATOMIC_NUMBER] = neutral_frac_guess
    x0[MAX_ATOMIC_NUMBER] = model_ne / n_tot * 1e5

    # Solve
    abs_ab = ctx.abs_ab

    def res_fn(x):
        return _residual_vec(x, n_tot, abs_ab, wII_ne, wIII_ne2,
                             log_nKs, ctx)

    x_sol = _newton_solve_np(res_fn, x0)

    # Unpack
    ne = abs(x_sol[MAX_ATOMIC_NUMBER]) * n_tot * 1e-5
    neutral_fracs = np.abs(x_sol[:MAX_ATOMIC_NUMBER])
    wII_sol = wII_ne / ne
    wIII_sol = wIII_ne2 / ne ** 2

    number_densities = {}
    for idx in range(MAX_ATOMIC_NUMBER):
        n_neutral = (n_tot - ne) * abs_ab[idx] * neutral_fracs[idx]
        number_densities[ctx.species_I[idx]] = n_neutral
        number_densities[ctx.species_II[idx]] = wII_sol[idx] * n_neutral
        number_densities[ctx.species_III[idx]] = wIII_sol[idx] * n_neutral

    # Molecules
    for i, mol in enumerate(ctx.molecules):
        atoms = mol.formula.get_atoms()
        if mol.charge == 0:
            log_n = sum(
                math.log10(max(number_densities[ctx.species_I[el - 1]], 0.0))
                for el in atoms
            )
        else:
            Z1, Z2 = atoms[0], atoms[1]
            log_n = (math.log10(max(number_densities[ctx.species_II[Z1 - 1]], 0.0))
                     + math.log10(max(number_densities[ctx.species_I[Z2 - 1]], 0.0)))
        number_densities[mol] = 10.0 ** (log_n - log_nKs[i])

    return ne, number_densities


# ── Public API (backward compatible) ─────────────────────────────────────

def chemical_equilibrium(T, n_tot, model_ne, absolute_abundances,
                         ionization_energies, partition_funcs,
                         log_equilibrium_constants,
                         registry=None,
                         electron_number_density_warn_threshold=None,
                         electron_number_density_warn_min_value=None,
                         _ctx=None):
    """Solve for the number density of every species.

    Parameters
    ----------
    T : float — temperature (K)
    n_tot : float — total number density (cm^-3)
    model_ne : float — electron number density from model atmosphere
    absolute_abundances : dict or array (Z -> N_X/N_total)
    ionization_energies : dict (Z -> [chi1, chi2, chi3])
    partition_funcs : dict (Species -> callable(logT))
    log_equilibrium_constants : dict (Species -> callable(logT))
    _ctx : optional pre-computed _ChemEqContext (for batch use)

    Returns
    -------
    ne : float — electron number density
    number_densities : dict (Species -> float)
    """
    if _ctx is not None:
        return _solve_layer(_ctx, T, n_tot, model_ne)

    # Build absolute abundance array
    if isinstance(absolute_abundances, dict):
        abs_ab = np.zeros(MAX_ATOMIC_NUMBER, dtype=np.float64)
        for Z, val in absolute_abundances.items():
            abs_ab[Z - 1] = val
    else:
        abs_ab = np.asarray(absolute_abundances, dtype=np.float64)

    ctx = _ChemEqContext(abs_ab, ionization_energies, partition_funcs,
                         log_equilibrium_constants)
    return _solve_layer(ctx, T, n_tot, model_ne)


# ── Hummer-Mihalas occupation probability ────────────────────────────────

def hummer_mihalas_w(T, n_eff, nH, nHe, ne):
    """MHD occupation probability *w* for a hydrogen level."""
    r_level = math.sqrt(2.5 * n_eff ** 4 + 0.5 * n_eff ** 2) * bohr_radius_cgs
    neutral_term = (nH * (r_level + math.sqrt(3) * bohr_radius_cgs) ** 3
                    + nHe * (r_level + 1.02 * bohr_radius_cgs) ** 3)
    K = 1.0
    if n_eff > 3:
        K = (16.0 / 3.0
             * (n_eff / (n_eff + 1)) ** 2
             * ((n_eff + 7.0 / 6.0) / (n_eff ** 2 + n_eff + 0.5)))
    chi = RydbergH_eV / n_eff ** 2 * eV_to_cgs
    e = electron_charge_cgs
    charged_term = 16.0 * ((e ** 2) / (chi * math.sqrt(K))) ** 3 * ne
    return math.exp(-4.0 * math.pi / 3.0 * (neutral_term + charged_term))


def hummer_mihalas_w_vec(T, n_eff, nH, nHe, ne):
    """Vectorized MHD occupation probability for arrays of n_eff (NumPy)."""
    n_eff = np.asarray(n_eff, dtype=np.float64)
    r_level = np.sqrt(2.5 * n_eff ** 4 + 0.5 * n_eff ** 2) * bohr_radius_cgs
    neutral_term = (nH * (r_level + np.sqrt(3.0) * bohr_radius_cgs) ** 3
                    + nHe * (r_level + 1.02 * bohr_radius_cgs) ** 3)
    K = np.where(
        n_eff > 3,
        16.0 / 3.0 * (n_eff / (n_eff + 1)) ** 2
        * ((n_eff + 7.0 / 6.0) / (n_eff ** 2 + n_eff + 0.5)),
        1.0,
    )
    chi = RydbergH_eV / n_eff ** 2 * eV_to_cgs
    e = electron_charge_cgs
    charged_term = 16.0 * ((e ** 2) / (chi * np.sqrt(K))) ** 3 * ne
    return np.exp(-4.0 * np.pi / 3.0 * (neutral_term + charged_term))
