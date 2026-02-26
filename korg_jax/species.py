"""Species and Formula classes, plus SpeciesRegistry for JAX-compatible indexing.

Ported from Korg.jl/src/species.jl.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from .atomic_data import MAX_ATOMIC_NUMBER, atomic_masses, atomic_numbers, atomic_symbols

MAX_ATOMS_PER_MOLECULE = 6

_roman_numerals = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


# ── Formula ──────────────────────────────────────────────────────────────────

class Formula:
    """Represents an atom or molecule irrespective of charge.

    Internally stores a sorted tuple of up to MAX_ATOMS_PER_MOLECULE atomic
    numbers (zero-padded on the left, like the Julia SVector{6,UInt8}).
    """

    __slots__ = ("atoms",)

    def __init__(self, atoms: Tuple[int, ...]):
        """Low-level constructor: *atoms* must be a sorted tuple of length 6,
        zero-padded on the left.  Prefer the classmethods for user input."""
        assert len(atoms) == MAX_ATOMS_PER_MOLECULE
        self.atoms = atoms

    # ── factory classmethods ──

    @classmethod
    def from_Z(cls, Z: int) -> "Formula":
        assert 1 <= Z <= MAX_ATOMIC_NUMBER
        return cls((0,) * (MAX_ATOMS_PER_MOLECULE - 1) + (Z,))

    @classmethod
    def from_Zs(cls, Zs: List[int]) -> "Formula":
        n = len(Zs)
        if n == 0:
            raise ValueError("Can't construct an empty Formula")
        if n > MAX_ATOMS_PER_MOLECULE:
            raise ValueError(
                f"Can't construct Formula with atoms {Zs}. "
                f"Up to {MAX_ATOMS_PER_MOLECULE} atoms are supported."
            )
        Zs = sorted(Zs)
        padded = [0] * (MAX_ATOMS_PER_MOLECULE - n) + Zs
        return cls(tuple(padded))

    @classmethod
    def from_string(cls, code: str) -> "Formula":
        """Parse a formula string like 'Fe', 'OH', 'C2', 'FeH', or numeric
        codes like '0801' for OH."""
        # Quick-parse single element symbols
        if code in atomic_numbers:
            return cls.from_Z(atomic_numbers[code])

        # Numeric codes (e.g. '0801' -> OH)
        if code.isdigit():
            if len(code) <= 2:
                return cls.from_Z(int(code))
            elif len(code) <= 4:
                el1 = int(code[:-2])
                el2 = int(code[-2:])
                return cls.from_Zs([el1, el2])
            else:
                if len(code) % 2 == 1:
                    code = "0" + code
                els = [int(code[i:i+2]) for i in range(0, len(code), 2)]
                return cls.from_Zs(els)

        # Symbolic codes: 'OH', 'FeH', 'C2', 'H2O', etc.
        inds = [i for i, c in enumerate(code) if c.isdigit() or c.isupper()]
        inds.append(len(code))
        subcodes = [code[inds[j]:inds[j+1]] for j in range(len(inds) - 1)]

        atoms: List[int] = []
        for s in subcodes:
            try:
                num = int(s)
                # Number means repeat previous atom
                prev = atoms[-1]
                for _ in range(num - 1):
                    atoms.append(prev)
            except ValueError:
                atoms.append(atomic_numbers[s])

        return cls.from_Zs(atoms)

    # ── accessors ──

    def get_atoms(self) -> Tuple[int, ...]:
        """Return the non-zero atomic numbers."""
        first = 0
        for i, a in enumerate(self.atoms):
            if a != 0:
                first = i
                break
        return self.atoms[first:]

    def get_atom(self) -> int:
        if self.ismolecule():
            raise ValueError("Can't get single atom from a molecule; use get_atoms().")
        return self.get_atoms()[0]

    def n_atoms(self) -> int:
        return len(self.get_atoms())

    def ismolecule(self) -> bool:
        return self.atoms[MAX_ATOMS_PER_MOLECULE - 2] != 0

    def get_mass(self) -> float:
        return sum(atomic_masses[a - 1] for a in self.get_atoms())

    # ── dunder methods ──

    def __eq__(self, other):
        if not isinstance(other, Formula):
            return NotImplemented
        return self.atoms == other.atoms

    def __hash__(self):
        return hash(self.atoms)

    def __repr__(self):
        parts = []
        atoms = self.get_atoms()
        i = 0
        while i < len(atoms):
            a = atoms[i]
            count = 1
            while i + count < len(atoms) and atoms[i + count] == a:
                count += 1
            sym = atomic_symbols[a - 1]
            parts.append(sym if count == 1 else f"{sym}{count}")
            i += count
        return "".join(parts)


# ── Species ──────────────────────────────────────────────────────────────────

class Species:
    """Represents a Formula with a particular charge (ionization state)."""

    __slots__ = ("formula", "charge")

    def __init__(self, formula: Formula, charge: int):
        if charge < -1:
            raise ValueError(f"Can't construct species with charge < -1: {formula} charge={charge}")
        self.formula = formula
        self.charge = charge

    @classmethod
    def from_string(cls, code: str) -> "Species":
        """Parse species codes like 'H I', 'Fe II', 'OH', '01.00', '26.01', etc."""
        code = code.strip().lstrip("0").strip()
        if not code:
            raise ValueError("Empty species code")

        # Handle trailing +/-
        if code.endswith("+"):
            code = code[:-1] + " 2"
        elif code.endswith("-"):
            code = code[:-1] + " 0"

        toks = [t for t in re.split(r"[\s._]+", code) if t]

        if len(toks) > 2:
            raise ValueError(f"{code!r} isn't a valid species code")

        formula = Formula.from_string(toks[0])

        if len(toks) == 1 or toks[1] == "":
            charge = 0
        else:
            # Try roman numeral first
            charge_tok = toks[1]
            if charge_tok in _roman_numerals:
                charge = _roman_numerals.index(charge_tok)  # I->0, II->1, ...
            else:
                charge = int(charge_tok)
                # If original code is NOT a pure float (Kurucz-style), subtract 1
                try:
                    float(code.replace(" ", ".").replace("_", "."))
                except ValueError:
                    charge -= 1
        return cls(formula, charge)

    # Convenience delegations
    def ismolecule(self) -> bool:
        return self.formula.ismolecule()

    def get_mass(self) -> float:
        return self.formula.get_mass()

    def get_atoms(self) -> Tuple[int, ...]:
        return self.formula.get_atoms()

    def get_atom(self) -> int:
        return self.formula.get_atom()

    def n_atoms(self) -> int:
        return self.formula.n_atoms()

    # ── dunder ──

    def __eq__(self, other):
        if not isinstance(other, Species):
            return NotImplemented
        return self.formula == other.formula and self.charge == other.charge

    def __hash__(self):
        return hash((self.formula, self.charge))

    def __repr__(self):
        f_str = repr(self.formula)
        if self.ismolecule() and self.charge == 1:
            return f_str + "+"
        if self.ismolecule() and self.charge == 0:
            return f_str
        if 0 <= self.charge <= len(_roman_numerals) - 1:
            return f"{f_str} {_roman_numerals[self.charge]}"
        if self.charge == -1:
            return f_str + "-"
        return f"{f_str} {self.charge}"


def all_atomic_species():
    """Yield all atomic species supported by Korg (Z=1..92, charge 0..2)."""
    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        for charge in range(min(Z, 2) + 1):
            yield Species(Formula.from_Z(Z), charge)


# ── SpeciesRegistry ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RegistryData:
    """Pre-computed named indices and arrays for vectorized equilibrium."""
    H_I_idx: int
    He_I_idx: int
    # molecule info: packed arrays for vectorized equilibrium
    mol_species_indices: np.ndarray   # (n_molecules,) int
    mol_atom_indices: np.ndarray      # (n_molecules, MAX_ATOMS_PER_MOLECULE) int, 0-padded
    mol_n_atoms: np.ndarray           # (n_molecules,) int
    mol_charges: np.ndarray           # (n_molecules,) int


class SpeciesRegistry:
    """Maps every Species to a fixed integer index for JAX array indexing.

    Atomic: Z=1..92 x charges {-1,0,1,2} -> first ~277 slots
    Molecular: from Barklem & Collet + ExoMol  -> next ~40 slots
    """

    def __init__(self):
        self._spec_to_idx: Dict[Species, int] = {}
        self._idx_to_spec: Dict[int, Species] = {}
        self._next_idx = 0

        # Register all atomic species: charges -1, 0, 1, 2 for Z=1..92
        for Z in range(1, MAX_ATOMIC_NUMBER + 1):
            for charge in [-1, 0, 1, 2]:
                if charge > Z:
                    continue
                spec = Species(Formula.from_Z(Z), charge)
                self._register(spec)

    def _register(self, spec: Species) -> int:
        if spec in self._spec_to_idx:
            return self._spec_to_idx[spec]
        idx = self._next_idx
        self._spec_to_idx[spec] = idx
        self._idx_to_spec[idx] = spec
        self._next_idx += 1
        return idx

    def register(self, spec: Species) -> int:
        """Register a species (e.g. a molecule) and return its index."""
        return self._register(spec)

    def index(self, spec: Species) -> int:
        """Return the index of an already-registered species."""
        return self._spec_to_idx[spec]

    def __contains__(self, spec: Species) -> bool:
        return spec in self._spec_to_idx

    def __len__(self) -> int:
        return self._next_idx

    @property
    def n_species(self) -> int:
        return self._next_idx

    def spec_at(self, idx: int) -> Species:
        return self._idx_to_spec[idx]

    def build_registry_data(self, molecules: List[Species]) -> RegistryData:
        """Build packed arrays for vectorized chemical equilibrium.

        *molecules* should be the list of molecular species whose equilibrium
        constants are available (keys of log_equilibrium_constants).
        """
        H_I = Species(Formula.from_Z(1), 0)
        He_I = Species(Formula.from_Z(2), 0)

        n_mol = len(molecules)
        mol_species_indices = np.zeros(n_mol, dtype=np.int32)
        mol_atom_indices = np.zeros((n_mol, MAX_ATOMS_PER_MOLECULE), dtype=np.int32)
        mol_n_atoms = np.zeros(n_mol, dtype=np.int32)
        mol_charges = np.zeros(n_mol, dtype=np.int32)

        for i, mol in enumerate(molecules):
            # Ensure molecule is registered
            self._register(mol)
            mol_species_indices[i] = self.index(mol)
            atoms = mol.formula.get_atoms()
            mol_n_atoms[i] = len(atoms)
            mol_charges[i] = mol.charge
            for j, a in enumerate(atoms):
                mol_atom_indices[i, j] = a  # 1-based Z

        return RegistryData(
            H_I_idx=self.index(H_I),
            He_I_idx=self.index(He_I),
            mol_species_indices=mol_species_indices,
            mol_atom_indices=mol_atom_indices,
            mol_n_atoms=mol_n_atoms,
            mol_charges=mol_charges,
        )
