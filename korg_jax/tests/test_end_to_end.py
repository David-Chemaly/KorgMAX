import numpy as np

from korg_jax.atmosphere import ModelAtmosphere
from korg_jax.abundances import format_A_X
from korg_jax.hydrogen_lines import hydrogen_line_absorption
from korg_jax.linelist import Linelist
from korg_jax.read_statmech import setup_all
from korg_jax.species import Species, Formula
from korg_jax.synthesize import synthesize
from korg_jax.wavelengths import Wavelengths
from korg_jax.line_absorption import line_absorption


def _simple_atmosphere():
    return ModelAtmosphere(
        tau_ref=np.array([1e-4, 1e-2, 1.0], dtype=float),
        z=np.array([2e7, 1e7, 0.0], dtype=float),
        temp=np.array([5200.0, 5600.0, 6000.0], dtype=float),
        electron_number_density=np.array([5e12, 8e12, 1.2e13], dtype=float),
        number_density=np.array([1e17, 1e17, 1e17], dtype=float),
        reference_wavelength=5e-5,
        is_spherical=False,
        R=None,
    )


def _simple_linelist():
    return Linelist(
        wl=np.array([5000.0e-8, 5005.0e-8], dtype=float),
        log_gf=np.array([-1.0, -2.0], dtype=float),
        species=[Species(Formula.from_Z(26), 0), Species(Formula.from_Z(12), 0)],
        E_lower=np.array([2.5, 3.1], dtype=float),
        gamma_rad=np.array([1e8, 5e7], dtype=float),
        gamma_stark=np.array([0.0, 0.0], dtype=float),
        vdW_1=np.array([0.0, 0.0], dtype=float),
        vdW_2=np.array([-1.0, -1.0], dtype=float),
    )


def test_hydrogen_line_absorption_finite():
    ionization_energies, partition_funcs, log_eq = setup_all()
    H_I = Species(Formula.from_Z(1), 0)
    U_H_I = partition_funcs[H_I](np.log(6000.0))

    wls = np.linspace(4800.0, 4900.0, 200) * 1e-8
    alpha = hydrogen_line_absorption(
        wls,
        T=6000.0,
        ne=1e13,
        nH_I=1e17,
        nHe_I=1e16,
        U_H_I=U_H_I,
        xi=1e5,
        window_size=150.0e-8,
        use_MHD=True,
    )
    assert np.all(np.isfinite(np.asarray(alpha)))


def test_line_absorption_basic():
    ll = _simple_linelist()
    wls = np.linspace(4999.0, 5006.0, 200) * 1e-8

    linelist_jax = {
        "wl": ll.wl,
        "log_gf": ll.log_gf,
        "species_idx": np.zeros(ll.n_lines, dtype=np.int32),
        "E_lower": ll.E_lower,
        "gamma_rad": ll.gamma_rad,
        "gamma_stark": ll.gamma_stark,
        "vdW_1": ll.vdW_1,
        "vdW_2": ll.vdW_2,
        "mass": np.array([s.get_mass() for s in ll.species]),
        "is_molecule": np.array([s.ismolecule() for s in ll.species]),
    }

    alpha = line_absorption(
        wls,
        linelist_jax,
        T=5800.0,
        ne=1e13,
        nH_I=1e17,
        n_absorbers=np.array([1e12, 5e11]),
        xi=1e5,
        alpha_cntm_at_line=np.zeros(ll.n_lines),
        cutoff_threshold=3e-4,
    )
    alpha = np.asarray(alpha)
    assert alpha.shape == wls.shape
    assert np.all(np.isfinite(alpha))
    assert np.max(alpha) >= 0.0


def test_synthesize_smoke():
    atm = _simple_atmosphere()
    ll = _simple_linelist()
    A_X = format_A_X(0.0)

    wls = Wavelengths.from_tuple(4995.0, 5010.0, 0.05)
    result = synthesize(
        atm,
        ll,
        A_X,
        wls,
        vmic=1.0,
        hydrogen_lines=True,
        return_cntm=True,
        line_cutoff_threshold=3e-4,
    )

    assert result.flux.shape == (len(wls),)
    assert result.cntm.shape == (len(wls),)
    assert result.alpha.shape == (atm.n_layers, len(wls))
    assert np.all(np.isfinite(result.flux))
    assert np.all(np.isfinite(result.cntm))
