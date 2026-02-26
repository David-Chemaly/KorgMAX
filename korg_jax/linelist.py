"""Linelist: Structure-of-Arrays representation and I/O parsers.

Ported from Korg.jl/src/linelist.jl.
"""
from __future__ import annotations

import math
import re
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .constants import (
    electron_charge_cgs, electron_mass_cgs, c_cgs,
    hplanck_eV, kboltz_cgs, kboltz_eV, bohr_radius_cgs,
    RydbergH_eV, Rydberg_eV,
)
from .species import Species, Formula
from .atomic_data import MAX_ATOMIC_NUMBER


# ── Linelist dataclass ───────────────────────────────────────────────────────

@dataclass
class Linelist:
    """Structure-of-Arrays line list.

    All arrays have shape ``(n_lines,)``.
    """
    wl: np.ndarray           # wavelength in cm
    log_gf: np.ndarray       # log10(gf)
    species: list            # list[Species]
    E_lower: np.ndarray      # lower level energy (eV)
    gamma_rad: np.ndarray    # radiative damping (s^-1)
    gamma_stark: np.ndarray  # Stark broadening (s^-1)
    vdW_1: np.ndarray        # vdW param 1
    vdW_2: np.ndarray        # vdW param 2 (-1 ⇒ γ_vdW form)

    @property
    def n_lines(self) -> int:
        return len(self.wl)

    def sort_by_wavelength(self) -> "Linelist":
        idx = np.argsort(self.wl)
        return Linelist(
            wl=self.wl[idx], log_gf=self.log_gf[idx],
            species=[self.species[i] for i in idx],
            E_lower=self.E_lower[idx], gamma_rad=self.gamma_rad[idx],
            gamma_stark=self.gamma_stark[idx],
            vdW_1=self.vdW_1[idx], vdW_2=self.vdW_2[idx],
        )

    def filter_species(self, exclude=None):
        """Drop triply+ ionised and specific species (default: H I)."""
        if exclude is None:
            exclude = {Species(Formula.from_Z(1), 0)}
        mask = np.array([
            (0 <= s.charge <= 2) and s not in exclude
            for s in self.species
        ])
        return Linelist(
            wl=self.wl[mask], log_gf=self.log_gf[mask],
            species=[s for s, m in zip(self.species, mask) if m],
            E_lower=self.E_lower[mask], gamma_rad=self.gamma_rad[mask],
            gamma_stark=self.gamma_stark[mask],
            vdW_1=self.vdW_1[mask], vdW_2=self.vdW_2[mask],
        )

    def to_jax(self, registry):
        """Convert to a dict of JAX arrays indexed by *registry*."""
        import jax.numpy as jnp
        species_idx = np.array([registry.index(s) for s in self.species],
                               dtype=np.int32)
        mass = np.array([s.get_mass() for s in self.species], dtype=np.float64)
        return {
            "wl": jnp.array(self.wl),
            "log_gf": jnp.array(self.log_gf),
            "species_idx": jnp.array(species_idx),
            "E_lower": jnp.array(self.E_lower),
            "gamma_rad": jnp.array(self.gamma_rad),
            "gamma_stark": jnp.array(self.gamma_stark),
            "vdW_1": jnp.array(self.vdW_1),
            "vdW_2": jnp.array(self.vdW_2),
            "mass": jnp.array(mass),
        }


# ── Broadening helpers ───────────────────────────────────────────────────────

def approximate_radiative_gamma(wl, log_gf):
    """Approximate radiative broadening (FWHM, s^-1)."""
    e = electron_charge_cgs
    m = electron_mass_cgs
    c = c_cgs
    return 8.0 * math.pi ** 2 * e ** 2 / (m * c * wl ** 2) * 10.0 ** log_gf


def approximate_gammas(wl, species, E_lower, ionization_energies=None):
    """Unsöld vdW + Cowley Stark at 10 000 K.

    Returns ``(gamma_stark, log10(gamma_vdW))``.
    """
    Z_ion = species.charge + 1
    if species.ismolecule() or Z_ion > 3:
        return 0.0, 0.0
    if ionization_energies is None:
        return 0.0, 0.0
    atom_Z = species.formula.get_atoms()[0]
    if atom_Z not in ionization_energies:
        return 0.0, 0.0
    chi = ionization_energies[atom_Z][Z_ion - 1]

    h = hplanck_eV
    c = c_cgs
    k = kboltz_cgs
    E_upper = E_lower + h * c / wl

    nstar4 = (Z_ion ** 2 * RydbergH_eV / (chi - E_upper)) ** 2
    if Z_ion == 1:
        gs = 2.25910152e-7 * nstar4
    else:
        gs = 5.42184365e-7 * nstar4 / (Z_ion + 1) ** 2

    dr2 = 2.5 * Rydberg_eV ** 2 * Z_ion ** 2 * (
        1.0 / (chi - E_upper) ** 2 - 1.0 / (chi - E_lower) ** 2
    )
    if chi < E_upper:
        log_gvdW = 0.0
    else:
        log_gvdW = (6.33 + 0.4 * math.log10(dr2)
                    + 0.3 * math.log10(10_000) + math.log10(k))
    return gs, log_gvdW


def _process_vdW(raw, wl, species, E_lower, ion_e=None):
    """Convert raw vdW value to ``(vdW_1, vdW_2)`` tuple."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        _, raw = approximate_gammas(wl, species, E_lower, ion_e)
    if isinstance(raw, tuple):
        return raw
    if raw < 0:
        return (10.0 ** raw, -1.0)
    if raw == 0:
        return (0.0, -1.0)
    if raw < 20:
        _, log_v = approximate_gammas(wl, species, E_lower, ion_e)
        return (raw * 10.0 ** log_v, -1.0)
    sigma = math.floor(raw) * bohr_radius_cgs ** 2
    alpha = raw - math.floor(raw)
    return (sigma, alpha)


def _ten_or_none(x):
    return None if x == 0 else 10.0 ** x

def _id_or_none(x):
    return None if x == 0 else x


def _make_line(wl, log_gf, species, E_lower,
               gamma_rad=None, gamma_stark=None, vdW=None, ion_e=None):
    """Assemble one line dict (used during parsing)."""
    if wl >= 1:
        wl *= 1e-8
    if gamma_rad is None or (isinstance(gamma_rad, float) and math.isnan(gamma_rad)):
        gamma_rad = approximate_radiative_gamma(wl, log_gf)
    if gamma_stark is None or (isinstance(gamma_stark, float) and math.isnan(gamma_stark)):
        gamma_stark, _ = approximate_gammas(wl, species, E_lower, ion_e)
    vdW_1, vdW_2 = _process_vdW(vdW, wl, species, E_lower, ion_e)
    return dict(wl=wl, log_gf=log_gf, species=species, E_lower=E_lower,
                gamma_rad=gamma_rad, gamma_stark=gamma_stark,
                vdW_1=vdW_1, vdW_2=vdW_2)


def _dicts_to_linelist(lines):
    if not lines:
        empty = np.array([], dtype=np.float64)
        return Linelist(empty, empty, [], empty, empty, empty, empty, empty)
    return Linelist(
        wl=np.array([l["wl"] for l in lines]),
        log_gf=np.array([l["log_gf"] for l in lines]),
        species=[l["species"] for l in lines],
        E_lower=np.array([l["E_lower"] for l in lines]),
        gamma_rad=np.array([l["gamma_rad"] for l in lines]),
        gamma_stark=np.array([l["gamma_stark"] for l in lines]),
        vdW_1=np.array([l["vdW_1"] for l in lines]),
        vdW_2=np.array([l["vdW_2"] for l in lines]),
    )


# ── Air/vacuum (local copy to avoid circular imports) ────────────────────────

def _air_to_vacuum_cm(lam_cm):
    lam_A = lam_cm * 1e8
    s = 1e4 / lam_A
    n = (1.0
         + 0.00008336624212083
         + 0.02408926869968 / (130.1065924522 - s * s)
         + 0.0001599740894897 / (38.92568793293 - s * s))
    return lam_cm * n


# ── Public read entry-point ──────────────────────────────────────────────────

def read_linelist(fname, format=None, ionization_energies=None):
    """Parse a linelist file and return a :class:`Linelist`.

    Supported formats: ``vald``, ``kurucz``, ``kurucz_vac``, ``moog``,
    ``moog_air``, ``turbospectrum``, ``turbospectrum_vac``, ``korg``.
    """
    if format is None:
        format = "korg" if fname.endswith(".h5") else "vald"
    fmt = format.lower()

    if fmt == "korg":
        return _read_korg(fname)

    with open(fname) as fh:
        text = fh.read()

    ie = ionization_energies
    if fmt.startswith("kurucz"):
        lines = _parse_kurucz(text, vacuum=fmt.endswith("_vac"), ion_e=ie)
    elif fmt == "vald":
        lines = _parse_vald(text, ion_e=ie)
    elif fmt == "moog":
        lines = _parse_moog(text, vacuum=True)
    elif fmt == "moog_air":
        lines = _parse_moog(text, vacuum=False)
    elif fmt == "turbospectrum":
        lines = _parse_turbospectrum(text, vacuum=False)
    elif fmt == "turbospectrum_vac":
        lines = _parse_turbospectrum(text, vacuum=True)
    else:
        raise ValueError(f"{format!r} is not a supported linelist format")

    ll = _dicts_to_linelist(lines)
    ll = ll.filter_species()
    return ll.sort_by_wavelength()


# ── Korg HDF5 format ─────────────────────────────────────────────────────────

def _read_korg(fname):
    import h5py
    with h5py.File(fname, "r") as f:
        wl = f["wl"][:]
        log_gf = f["log_gf"][:]
        E_lower = f["E_lower"][:]
        gamma_rad = f["gamma_rad"][:]
        gamma_stark = f["gamma_stark"][:]
        vdW_1 = f["vdW_1"][:]
        vdW_2 = f["vdW_2"][:]
        formula_arr = f["formula"][:]   # (6, n_lines)
        charge_arr = f["charge"][:]

    species_list = []
    for i in range(len(wl)):
        atoms = formula_arr[:, i]
        first = next(j for j in range(len(atoms)) if atoms[j] != 0)
        formula = Formula.from_Zs(list(atoms[first:].astype(int)))
        species_list.append(Species(formula, int(charge_arr[i])))

    return Linelist(wl=wl, log_gf=log_gf, species=species_list,
                    E_lower=E_lower, gamma_rad=gamma_rad,
                    gamma_stark=gamma_stark, vdW_1=vdW_1, vdW_2=vdW_2)


# ── VALD parser ──────────────────────────────────────────────────────────────

def _parse_vald(text, ion_e=None):
    raw = text.split("\n")
    lines = [l for l in raw if l and l[0] != "#"]
    if not lines:
        return []
    if lines[0].startswith(" WARNING"):
        lines = lines[1:]
    lines = [l.replace("'", '"') for l in lines]

    extractall = not bool(re.match(r"^\s+\d", lines[0]))
    first = 2 if extractall else 3
    header = lines[first - 1]
    short = first + 1 < len(lines) and not bool(re.match(r'^"? ', lines[first + 1]))
    step = 1 if short else 4
    body = lines[first::step]

    data = []
    for l in body:
        if len(l) > 1 and l[0] == '"' and l[1].isupper():
            data.append(l)
        else:
            break

    is_air = "air" in header.lower()
    is_cm = "cm" in header.lower()
    result = []
    for row in data:
        toks = [t.strip().strip('"') for t in row.split(",")]
        if len(toks) < 8:
            continue
        try:
            spec = Species.from_string(toks[0])
            if short and extractall:
                wl_A, E_low, loggf = float(toks[1]), float(toks[2]), float(toks[3])
                gr, gs, gv = float(toks[4]), float(toks[5]), float(toks[6])
            elif short:
                wl_A, E_low, loggf = float(toks[1]), float(toks[2]), float(toks[4])
                gr, gs, gv = float(toks[5]), float(toks[6]), float(toks[7])
            else:
                wl_A, loggf, E_low = float(toks[1]), float(toks[2]), float(toks[3])
                gr = float(toks[10]) if len(toks) > 10 else 0
                gs = float(toks[11]) if len(toks) > 11 else 0
                gv = float(toks[12]) if len(toks) > 12 else 0

            if is_cm:
                E_low *= c_cgs * hplanck_eV
            wl_cm = wl_A * 1e-8
            if is_air:
                wl_cm = _air_to_vacuum_cm(wl_cm)

            grad = approximate_radiative_gamma(wl_cm, loggf) if gr == 0 else 10.0 ** gr
            result.append(_make_line(
                wl_cm, loggf, spec, E_low,
                gamma_rad=grad, gamma_stark=_ten_or_none(gs),
                vdW=_id_or_none(gv), ion_e=ion_e))
        except (ValueError, KeyError, IndexError):
            continue
    return result


# ── MOOG parser ──────────────────────────────────────────────────────────────

def _parse_moog(text, vacuum=True):
    rows = text.strip().split("\n")
    result = []
    for row in rows[1:]:
        toks = row.split()
        if len(toks) < 4:
            continue
        try:
            wl_cm = float(toks[0]) * 1e-8
            if not vacuum:
                wl_cm = _air_to_vacuum_cm(wl_cm)
            dot = toks[1].index(".")
            spec = Species.from_string(toks[1][: dot + 2])
            result.append(_make_line(wl_cm, float(toks[3]), spec, float(toks[2])))
        except (ValueError, KeyError, IndexError):
            continue
    return result


# ── Kurucz parser ────────────────────────────────────────────────────────────

def _parse_kurucz(text, vacuum=False, ion_e=None):
    result = []
    for row in text.split("\n"):
        if not row.strip():
            continue
        if len(row) == 159:
            row = " " + row
        if len(row) < 100:
            continue
        try:
            E1 = abs(float(row[24:36])) * c_cgs * hplanck_eV
            E2 = abs(float(row[52:64])) * c_cgs * hplanck_eV
            spec = Species.from_string(row[18:24])
            loggf = float(row[11:18])
            if len(row) > 115:
                loggf += float(row[109:115])

            wl_cm = float(row[0:11]) * 1e-7   # nm -> cm
            if not vacuum:
                wl_cm = _air_to_vacuum_cm(wl_cm)

            gr = float(row[80:86]) if len(row) > 86 else 0
            gs = float(row[86:92]) if len(row) > 92 else 0
            gv = float(row[92:98]) if len(row) > 98 else 0

            result.append(_make_line(
                wl_cm, loggf, spec, min(E1, E2),
                gamma_rad=_ten_or_none(gr), gamma_stark=_ten_or_none(gs),
                vdW=_id_or_none(gv), ion_e=ion_e))
        except (ValueError, KeyError, IndexError):
            continue
    return result


# ── Turbospectrum parser ─────────────────────────────────────────────────────

def _parse_turbospectrum(text, vacuum=False):
    lines = text.split("\n")
    headers = [
        i for i in range(len(lines) - 1)
        if lines[i].startswith("'") and lines[i + 1].startswith("'")
    ]
    result = []
    for h_idx, h_line in enumerate(headers):
        m = re.match(r"'\s*(\d+)\.(\d+)\s+'\s+(\d+)\s+(\d+)", lines[h_line])
        if not m:
            continue
        formula = Formula.from_string(m.group(1))
        charge = int(m.group(3)) - 1
        spec = Species(formula, charge)
        end = headers[h_idx + 1] if h_idx + 1 < len(headers) else len(lines)

        for row in lines[h_line + 2: end]:
            toks = row.split()
            if len(toks) < 6:
                continue
            try:
                loggf = float(toks[2])
                wl_cm = float(toks[0]) * 1e-8
                if not vacuum:
                    wl_cm = _air_to_vacuum_cm(wl_cm)
                gr = float(toks[5])
                if gr in (0, 1):
                    gr = approximate_radiative_gamma(wl_cm, loggf)
                gs_val = None
                if len(toks) >= 7:
                    try:
                        gs_val = _ten_or_none(float(toks[6]))
                    except ValueError:
                        pass
                result.append(_make_line(
                    wl_cm, loggf, spec, float(toks[1]),
                    gamma_rad=gr, gamma_stark=gs_val, vdW=float(toks[3])))
            except (ValueError, KeyError, IndexError):
                continue
    return result
