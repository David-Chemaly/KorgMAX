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

    # Timing 4000-4030
    t0 = time.process_time()
    sol_4000 = KJ.synthesize.synthesize(atm, lines, A_X, 4000, 4030, prefer_jit=True)
    t1 = time.process_time() - t0
    print("synthesize 4000-4030 time:", t1)

    # Timing 5000-5030
    t0 = time.process_time()
    sol = KJ.synthesize.synthesize(atm, lines, A_X, 5000, 5030, prefer_jit=True)
    t2 = time.process_time() - t0
    print("synthesize 5000-5030 time:", t2)

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

    metal_poor_sol = KJ.synthesize.synthesize(atm, lines, metal_poor_A_X, 5015, 5025, prefer_jit=True)
    alpha_rich_sol = KJ.synthesize.synthesize(atm, lines, alpha_rich_A_X, 5015, 5025, prefer_jit=True)
    Ni_enriched_sol = KJ.synthesize.synthesize(atm, lines, Ni_enriched_A_X, 5015, 5025, prefer_jit=True)

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
