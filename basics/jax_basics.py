import os
import sys
import time
import numpy as np

# Ensure repo root on path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import korg_jax as KJ
from korg_jax.linelist import read_linelist
from korg_jax.atmosphere import read_model_atmosphere
from korg_jax.abundances import format_A_X

FAST_MODE = os.environ.get("KORGMAX_FAST", "1") != "0"


def main():
    linelist_path = os.path.join(repo_root, "basics", "linelist.vald")
    atm_path = os.path.join(
        repo_root,
        "basics",
        "s6000_g+1.0_m0.5_t05_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.mod",
    )

    print("Repo root:", repo_root)
    print("Linelist:", linelist_path)
    print("Atmosphere:", atm_path)

    lines = read_linelist(linelist_path, format="vald")
    print("Linelist n_lines:", lines.n_lines)

    atm = read_model_atmosphere(atm_path)
    print("Atmosphere layers:", atm.n_layers)

    A_X = format_A_X(0)

    if FAST_MODE:
        print("FAST MODE enabled (set KORGMAX_FAST=0 for full run)")
        wl_a = (4000, 4008)
        wl_b = (5000, 5008)
        wl_abund = (5015, 5018)
        prefer_jit = False
        hydrogen_lines = False
    else:
        wl_a = (4000, 4030)
        wl_b = (5000, 5030)
        wl_abund = (5015, 5025)
        prefer_jit = True
        hydrogen_lines = True

    # Timing window A
    t0 = time.perf_counter()
    sol_4000 = KJ.synthesize.synthesize(
        atm, lines, A_X, wl_a[0], wl_a[1],
        prefer_jit=prefer_jit, hydrogen_lines=hydrogen_lines
    )
    t1 = time.perf_counter() - t0
    print(f"synthesize {wl_a[0]}-{wl_a[1]} time:", t1)

    # Timing window B
    t0 = time.perf_counter()
    sol = KJ.synthesize.synthesize(
        atm, lines, A_X, wl_b[0], wl_b[1],
        prefer_jit=prefer_jit, hydrogen_lines=hydrogen_lines,
        use_chemical_equilibrium_from=sol_4000,
    )
    t2 = time.perf_counter() - t0
    print(f"synthesize {wl_b[0]}-{wl_b[1]} time:", t2)

    # Print key outputs for comparison
    flux = np.array(sol.flux)
    cntm = np.array(sol.cntm)
    wls = np.array(sol.wavelengths)

    print("flux len:", len(flux))
    print("flux head:", flux[:10])
    print("flux tail:", flux[-10:])
    print("cntm head:", cntm[:10])
    print("wls head:", wls[:10])

    # Abundance variations
    metal_poor_A_X = format_A_X(-0.5)
    alpha_rich_A_X = format_A_X(0, 0.5)
    Ni_enriched_A_X = format_A_X({"Ni": 1.0})

    metal_poor_sol = KJ.synthesize.synthesize(
        atm, lines, metal_poor_A_X, wl_abund[0], wl_abund[1],
        prefer_jit=prefer_jit, hydrogen_lines=hydrogen_lines
    )
    alpha_rich_sol = KJ.synthesize.synthesize(
        atm, lines, alpha_rich_A_X, wl_abund[0], wl_abund[1],
        prefer_jit=prefer_jit, hydrogen_lines=hydrogen_lines
    )
    Ni_enriched_sol = KJ.synthesize.synthesize(
        atm, lines, Ni_enriched_A_X, wl_abund[0], wl_abund[1],
        prefer_jit=prefer_jit, hydrogen_lines=hydrogen_lines
    )

    # Print summary stats for comparisons
    for name, soln in [
        ("metal_poor", metal_poor_sol),
        ("alpha_rich", alpha_rich_sol),
        ("Ni_enriched", Ni_enriched_sol),
    ]:
        f = np.array(soln.flux)
        print(name, "flux len", len(f), "min", f.min(), "max", f.max(), "mean", f.mean())

    # Alpha matrix summary
    alpha = np.array(sol.alpha)
    print("alpha shape:", alpha.shape, "min", alpha.min(), "max", alpha.max())

    # Number densities for a few species
    from korg_jax.species import Species
    for spec in ["H I", "H II", "O I", "OH"]:
        s = Species.from_string(spec)
        arr = np.array(sol.number_densities[s])
        print(spec, "n min", arr.min(), "max", arr.max())


if __name__ == "__main__":
    main()
