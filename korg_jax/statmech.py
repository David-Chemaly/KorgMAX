"""Chemical equilibrium solver (Saha + molecular equilibrium).

Ported from Korg.jl/src/statmech.jl.

The solver uses a JAX Newton iteration with ``jax.lax.while_loop``
and ``jax.jacfwd`` for the Jacobian so it is fully jit-compatible.
"""
from __future__ import annotations

import math
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from .constants import (
    kboltz_cgs, kboltz_eV, hplanck_cgs, electron_mass_cgs,
    bohr_radius_cgs, RydbergH_eV, eV_to_cgs, electron_charge_cgs,
)
from .atomic_data import MAX_ATOMIC_NUMBER


# ── Helpers (pure JAX, jit-compatible) ───────────────────────────────────────

@jax.jit
def translational_U(m, T):
    """Translational partition function factor."""
    k = kboltz_cgs
    h = hplanck_cgs
    return (2.0 * jnp.pi * m * k * T / h ** 2) ** 1.5


def saha_ion_weights_np(T, ne, atom_Z, ionization_energies, partition_funcs):
    """Compute (wII, wIII) = ratios of ionised/neutral number densities.

    *partition_funcs* maps Species -> callable(logT) returning U.
    Uses NumPy (called during setup, not inside JIT).
    """
    from .species import Species, Formula

    chi = ionization_energies[atom_Z]
    chi_I = chi[0]
    f = Formula.from_Z(atom_Z)

    logT = math.log(T)
    UI = partition_funcs[Species(f, 0)](logT)
    UII = partition_funcs[Species(f, 1)](logT)

    k = kboltz_eV
    transU = float(translational_U(electron_mass_cgs, T))

    wII = 2.0 / ne * (UII / UI) * transU * math.exp(-chi_I / (k * T))

    if atom_Z == 1:
        wIII = 0.0
    else:
        chi_II = chi[1]
        UIII = partition_funcs[Species(f, 2)](logT)
        wIII = wII * 2.0 / ne * (UIII / UII) * transU * math.exp(-chi_II / (k * T))

    return wII, wIII


# ── Vectorised Saha (JAX arrays, for use inside JIT) ────────────────────────

@partial(jax.jit, static_argnums=())
def _saha_ion_factors(neutral_nd, ne, wII_over_ne, wIII_over_ne2):
    """Compute ion factors for all elements simultaneously.

    Parameters
    ----------
    neutral_nd : (92,) neutral number densities
    ne : scalar electron number density
    wII_over_ne, wIII_over_ne2 : (92,) pre-computed Saha weights divided
        by ne^1 and ne^2 respectively (computed at ne=1).

    Returns
    -------
    wII, wIII : (92,) ratios n(X II)/n(X I), n(X III)/n(X I)
    """
    wII = wII_over_ne / ne
    wIII = wIII_over_ne2 / ne ** 2
    return wII, wIII


# ── Residual function builder (returns a pure function for JAX) ──────────────

def build_residual_fn(T, n_tot, absolute_abundances,
                      wII_ne, wIII_ne2,
                      mol_atom_Z, mol_n_atoms, mol_charges,
                      log_nKs):
    """Return a pure JAX function ``residuals(x) -> F`` for Newton's method.

    ``x[0:92]`` = neutral fraction of each element.
    ``x[92]`` = ne / n_tot * 1e5.

    *mol_atom_Z* : (n_mol, MAX_ATOMS) int array of atomic numbers (0-padded).
    *mol_n_atoms* : (n_mol,) int.
    *mol_charges* : (n_mol,) int.
    *log_nKs* : (n_mol,) float64 — log10 number-density equilibrium constants.
    """
    N = MAX_ATOMIC_NUMBER
    abs_ab = jnp.array(absolute_abundances)
    wII_ne_j = jnp.array(wII_ne)
    wIII_ne2_j = jnp.array(wIII_ne2)
    mol_Z = jnp.array(mol_atom_Z)          # (n_mol, 6)
    mol_na = jnp.array(mol_n_atoms)        # (n_mol,)
    mol_ch = jnp.array(mol_charges)        # (n_mol,)
    log_nKs_j = jnp.array(log_nKs)         # (n_mol,)

    def residuals(x):
        ne = jnp.abs(x[N]) * n_tot * 1e-5
        atom_nd = abs_ab * (n_tot - ne)               # total atoms
        neutral_nd = atom_nd * jnp.abs(x[:N])          # neutral number densities

        wII = wII_ne_j / ne
        wIII = wIII_ne2_j / ne ** 2

        # F[Z] = atom_nd[Z] - (1 + wII + wIII) * neutral_nd[Z] - molecule contributions
        F_atoms = atom_nd - (1.0 + wII + wIII) * neutral_nd

        # Electron balance: sum of freed electrons - ne
        F_elec = jnp.sum((wII + 2.0 * wIII) * neutral_nd) - ne

        # Molecule contributions via log-space
        # Clamp to a tiny positive value to avoid -inf in log10 and NaN in
        # gradients (inf * 0 from d(log10(0))/dx when abundance is zero).
        # jnp.maximum ensures the tangent is 0 for clamped entries.
        log_neutral = jnp.log10(jnp.maximum(neutral_nd, 1e-300))

        def mol_contribution(carry, mol_idx):
            F_a, F_e = carry
            # Sum log(neutral[Z]) for each constituent atom
            atom_Zs = mol_Z[mol_idx]  # (6,) 0-padded, 1-based Z
            na = mol_na[mol_idx]
            charge = mol_ch[mol_idx]

            # For neutral molecules: sum log n(Z I) for each atom
            # For charged diatomics: first atom is ionised
            log_sum = jnp.float64(0.0)
            for k in range(6):
                # Only count non-zero entries
                Z_k = atom_Zs[k]
                is_active = Z_k > 0
                # Safe index: use 0 when Z_k=0 (padding) to avoid
                # wrapping to index 91
                safe_idx = jnp.where(is_active, Z_k - 1, 0)
                contrib = jnp.where(is_active, log_neutral[safe_idx], 0.0)
                log_sum = log_sum + contrib

            # Charged diatomic correction: first atom (lower Z) is ionised
            # Add log(wII[Z1]) and subtract one neutral log
            # (Korg: Z1 is the first non-zero atom)
            first_Z = atom_Zs[jnp.argmax(atom_Zs > 0)]
            safe_first = jnp.where(first_Z > 0, first_Z - 1, 0)
            charged_corr = jnp.where(
                charge == 1,
                jnp.log10(jnp.maximum(wII_ne_j[safe_first] / ne, 1e-300)),
                0.0
            )
            log_sum = log_sum + charged_corr

            n_mol = 10.0 ** (log_sum - log_nKs_j[mol_idx])

            # Subtract from each constituent element
            for k in range(6):
                Z_k = atom_Zs[k]
                is_active = Z_k > 0
                safe_idx = jnp.where(is_active, Z_k - 1, 0)
                F_a = F_a.at[safe_idx].add(jnp.where(is_active, -n_mol, 0.0))

            # Charged molecules contribute to electron balance
            F_e = F_e + jnp.where(charge == 1, n_mol, 0.0)

            return (F_a, F_e), None

        n_mol = mol_Z.shape[0]
        (F_atoms, F_elec), _ = jax.lax.scan(
            mol_contribution, (F_atoms, F_elec), jnp.arange(n_mol)
        )

        # Normalise (safe division: zero-abundance elements get residual 0)
        F_atoms = jnp.where(atom_nd > 0, F_atoms / atom_nd, 0.0)
        F_elec = F_elec / (ne * 1e-5)

        return jnp.concatenate([F_atoms, F_elec[None]])

    return residuals


# ── Newton solver ────────────────────────────────────────────────────────────

def newton_solve(residual_fn, x0, max_iter=1000, tol=1e-8):
    """Damped Newton solver using ``jax.lax.while_loop``.

    Parameters
    ----------
    residual_fn : callable x -> F  (both length-93 arrays)
    x0 : initial guess (93,)

    Returns
    -------
    x_solution : (93,) array
    """
    jac_fn = jax.jacfwd(residual_fn)

    def cond(state):
        x, F, i = state
        return (jnp.max(jnp.abs(F)) > tol) & (i < max_iter)

    def body(state):
        x, F, i = state
        J = jac_fn(x)
        # Solve J @ dx = -F
        dx = jnp.linalg.solve(J, -F)
        # Damping: limit step size
        max_step = jnp.max(jnp.abs(dx))
        scale = jnp.where(max_step > 1.0, 1.0 / max_step, 1.0)
        x_new = x + scale * dx
        F_new = residual_fn(x_new)
        return x_new, F_new, i + 1

    F0 = residual_fn(x0)
    x_sol, F_sol, n_iter = jax.lax.while_loop(cond, body, (x0, F0, 0))
    return x_sol


# ── High-level API ───────────────────────────────────────────────────────────

def chemical_equilibrium(T, n_tot, model_ne, absolute_abundances,
                         ionization_energies, partition_funcs,
                         log_equilibrium_constants,
                         registry=None,
                         electron_number_density_warn_threshold=None,
                         electron_number_density_warn_min_value=None):
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

    Returns
    -------
    ne : float — electron number density
    number_densities : dict (Species -> float)
    """
    from .species import Species, Formula

    # Build absolute abundance array (length 92)
    if isinstance(absolute_abundances, dict):
        abs_ab = np.zeros(MAX_ATOMIC_NUMBER, dtype=np.float64)
        for Z, val in absolute_abundances.items():
            abs_ab[Z - 1] = val
    else:
        abs_ab = np.asarray(absolute_abundances, dtype=np.float64)

    # Ensure all abundances are positive.  Julia always passes full solar
    # abundances for all 92 elements; zero rows make the Jacobian singular
    # and crash jnp.linalg.solve.  A tiny floor (1e-99) has no effect on
    # spectral quantities but keeps the system well-conditioned.
    abs_ab = np.maximum(abs_ab, 1e-99)

    # Pre-compute Saha weights at ne=1
    wII_ne = np.zeros(MAX_ATOMIC_NUMBER, dtype=np.float64)
    wIII_ne2 = np.zeros(MAX_ATOMIC_NUMBER, dtype=np.float64)
    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        w2, w3 = saha_ion_weights_np(T, 1.0, Z, ionization_energies, partition_funcs)
        wII_ne[Z - 1] = w2
        wIII_ne2[Z - 1] = w3

    # Neutral fraction first guess (neglecting molecules)
    neutral_frac_guess = np.zeros(MAX_ATOMIC_NUMBER, dtype=np.float64)
    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        w2, w3 = saha_ion_weights_np(T, model_ne, Z, ionization_energies, partition_funcs)
        neutral_frac_guess[Z - 1] = 1.0 / (1.0 + w2 + w3)

    # Pack molecule data
    molecules = list(log_equilibrium_constants.keys())
    n_mol = len(molecules)

    MAX_ATOMS = 6
    mol_atom_Z = np.zeros((n_mol, MAX_ATOMS), dtype=np.int32)
    mol_n_atoms = np.zeros(n_mol, dtype=np.int32)
    mol_charges = np.zeros(n_mol, dtype=np.int32)
    log_nKs = np.zeros(n_mol, dtype=np.float64)

    for i, mol in enumerate(molecules):
        atoms = mol.formula.get_atoms()
        mol_n_atoms[i] = len(atoms)
        mol_charges[i] = mol.charge
        for j, a in enumerate(atoms):
            mol_atom_Z[i, j] = a
        # Compute log10(nK) from log10(pK)
        logK_fn = log_equilibrium_constants[mol]
        log_pK = float(logK_fn(math.log(T)))
        log_nKs[i] = log_pK - (len(atoms) - 1) * math.log10(kboltz_cgs * T)

    # Build residual function and solve
    res_fn = build_residual_fn(
        T, n_tot, abs_ab, wII_ne, wIII_ne2,
        mol_atom_Z, mol_n_atoms, mol_charges, log_nKs
    )

    x0 = jnp.concatenate([
        jnp.array(neutral_frac_guess),
        jnp.array([model_ne / n_tot * 1e5]),
    ])

    x_sol = newton_solve(res_fn, x0)

    # Unpack solution
    ne = float(jnp.abs(x_sol[MAX_ATOMIC_NUMBER]) * n_tot * 1e-5)
    neutral_fracs = np.array(jnp.abs(x_sol[:MAX_ATOMIC_NUMBER]))

    # Build number density dict
    number_densities = {}
    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        n_neutral = (n_tot - ne) * abs_ab[Z - 1] * neutral_fracs[Z - 1]
        spec_I = Species(Formula.from_Z(Z), 0)
        number_densities[spec_I] = n_neutral

        w2, w3 = saha_ion_weights_np(T, ne, Z, ionization_energies, partition_funcs)
        number_densities[Species(Formula.from_Z(Z), 1)] = w2 * n_neutral
        number_densities[Species(Formula.from_Z(Z), 2)] = w3 * n_neutral

    # Molecules
    # Use np.log10 which returns -inf for 0 (matching Julia's log10(0) = -Inf)
    for i, mol in enumerate(molecules):
        atoms = mol.formula.get_atoms()
        if mol.charge == 0:
            log_n = sum(
                float(np.log10(max(number_densities[Species(Formula.from_Z(el), 0)], 0.0)))
                for el in atoms
            )
        else:
            Z1, Z2 = atoms[0], atoms[1]
            log_n = (float(np.log10(max(number_densities[Species(Formula.from_Z(Z1), 1)], 0.0)))
                     + float(np.log10(max(number_densities[Species(Formula.from_Z(Z2), 0)], 0.0))))
        number_densities[mol] = 10.0 ** (log_n - log_nKs[i])

    return ne, number_densities


# ── Hummer-Mihalas occupation probability ────────────────────────────────────

def hummer_mihalas_w(T, n_eff, nH, nHe, ne):
    """MHD occupation probability *w* for a hydrogen level.

    Parameters
    ----------
    T : temperature (K) — unused in default mode
    n_eff : effective quantum number
    nH, nHe : neutral H and He number densities
    ne : electron number density
    """
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


@jax.jit
def hummer_mihalas_w_jax(n_eff, nH, nHe, ne):
    """JAX-jittable version of MHD occupation probability."""
    r_level = jnp.sqrt(2.5 * n_eff ** 4 + 0.5 * n_eff ** 2) * bohr_radius_cgs
    neutral_term = (nH * (r_level + jnp.sqrt(3.0) * bohr_radius_cgs) ** 3
                    + nHe * (r_level + 1.02 * bohr_radius_cgs) ** 3)

    K = jnp.where(
        n_eff > 3,
        16.0 / 3.0 * (n_eff / (n_eff + 1)) ** 2
        * ((n_eff + 7.0 / 6.0) / (n_eff ** 2 + n_eff + 0.5)),
        1.0,
    )
    chi = RydbergH_eV / n_eff ** 2 * eV_to_cgs
    e = electron_charge_cgs
    charged_term = 16.0 * ((e ** 2) / (chi * jnp.sqrt(K))) ** 3 * ne

    return jnp.exp(-4.0 * jnp.pi / 3.0 * (neutral_term + charged_term))
