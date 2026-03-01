"""Float32 vs float64 precision and speed benchmark for korg_jax.

Architecture:
  1. Parent loads Julia atmosphere + linelist once, serialises to a temp .npz.
  2. Two subprocesses (float32 / float64) each read the .npz, reconstruct
     Python objects, and call synthesize() with the correct JAX precision
     set *before* any imports.
  3. Results (timing + flux) come back as a single JSON line on stdout.

Usage:
    python benchmark/float32_test.py
"""
import subprocess, sys, json, tempfile, os
import numpy as np

# ── Inline worker (executed in a fresh subprocess) ───────────────────────────

WORKER = r"""
import os, sys, time, json
import numpy as np

mode      = sys.argv[1]   # "float32" or "float64"
data_file = sys.argv[2]   # path to .npz with atmosphere + linelist

if mode == "float32":
    os.environ["KORGMAX_FLOAT32"] = "1"

import korg_jax
from korg_jax.atmosphere import ModelAtmosphere
from korg_jax.linelist   import Linelist
from korg_jax.species    import Species
from korg_jax.synthesize import synthesize

d = np.load(data_file)

atm_py = ModelAtmosphere(
    tau_ref                 = d["tau_ref"],
    z                       = d["z"],
    temp                    = d["temp"],
    electron_number_density = d["ne"],
    number_density          = d["ntot"],
    reference_wavelength    = float(d["ref_wl"]),
    is_spherical            = bool(d["is_spherical"]),
    R                       = float(d["R"]) if bool(d["has_R"]) else None,
)

linelist_py = Linelist(
    wl          = d["ll_wl"],
    log_gf      = d["ll_log_gf"],
    species     = [Species.from_string(str(s)) for s in d["ll_species"]],
    E_lower     = d["ll_E_lower"],
    gamma_rad   = d["ll_gamma_rad"],
    gamma_stark = d["ll_gamma_stark"],
    vdW_1       = d["ll_vdW_1"],
    vdW_2       = d["ll_vdW_2"],
)

A_X      = d["A_X"]
WL_RANGE = (5000.0, 5050.0)

# warm-up (triggers JIT compilation)
_ = synthesize(atm_py, linelist_py, A_X, WL_RANGE)

# timed run
t0  = time.perf_counter()
sol = synthesize(atm_py, linelist_py, A_X, WL_RANGE)
elapsed = time.perf_counter() - t0

out = {
    "mode": mode,
    "time": elapsed,
    "flux": np.asarray(sol.flux).tolist(),
    "wls":  np.asarray(sol.wavelengths).tolist(),
}
print(json.dumps(out))
"""


def run_mode(mode, data_file):
    result = subprocess.run(
        [sys.executable, "-c", WORKER, mode, data_file],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    if result.returncode != 0:
        print(f"[{mode}] STDERR:\n{result.stderr[-4000:]}")
        raise RuntimeError(f"Subprocess failed for mode={mode}")
    for line in reversed(result.stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"No JSON output for mode={mode}\n{result.stdout[-2000:]}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from juliacall import Main as jl

    jl.seval("using Korg")
    Korg = jl.Korg

    # ── Linelist ──────────────────────────────────────────────────────────────
    jl.seval("""
    function _extract_linelist_arrays(lines)
        wl          = [l.wl for l in lines]
        log_gf      = [l.log_gf for l in lines]
        species     = [string(l.species) for l in lines]
        E_lower     = [l.E_lower for l in lines]
        gamma_rad   = [l.gamma_rad for l in lines]
        gamma_stark = [l.gamma_stark for l in lines]
        vdW_1 = Float64[]; vdW_2 = Float64[]
        for l in lines
            if l.vdW isa Tuple
                push!(vdW_1, l.vdW[1]); push!(vdW_2, l.vdW[2])
            else
                push!(vdW_1, l.vdW);    push!(vdW_2, -1.0)
            end
        end
        return (wl, log_gf, species, E_lower, gamma_rad, gamma_stark, vdW_1, vdW_2)
    end
    """)

    print("Loading VALD solar linelist (5000–5050 Å) via Korg...")
    lines = Korg.get_VALD_solar_linelist()
    wl, log_gf, species, E_lower, gamma_rad, gamma_stark, vdW_1, vdW_2 = \
        jl._extract_linelist_arrays(lines)

    # ── Atmosphere ────────────────────────────────────────────────────────────
    jl.seval("""
    get_tau_refs(atm)     = Korg.get_tau_refs(atm)
    get_zs(atm)           = Korg.get_zs(atm)
    get_temps(atm)        = Korg.get_temps(atm)
    get_nes(atm)          = Korg.get_electron_number_densities(atm)
    get_ns(atm)           = Korg.get_number_densities(atm)
    get_ref_wl(atm)       = atm.reference_wavelength
    get_is_spherical(atm) = atm isa Korg.ShellAtmosphere
    get_R(atm)            = get_is_spherical(atm) ? atm.R : nothing
    """)

    print("Interpolating solar atmosphere (Teff=5778, logg=4.437) via Korg...")
    atm = Korg.interpolate_marcs(5778, 4.437)

    tau_ref      = np.array(jl.get_tau_refs(atm), dtype=float)
    z            = np.array(jl.get_zs(atm),        dtype=float)
    temp         = np.array(jl.get_temps(atm),     dtype=float)
    ne           = np.array(jl.get_nes(atm),        dtype=float)
    ntot         = np.array(jl.get_ns(atm),         dtype=float)
    ref_wl       = float(jl.get_ref_wl(atm))
    is_spherical = bool(jl.get_is_spherical(atm))
    R_julia      = jl.get_R(atm)
    R_val        = float(R_julia) if R_julia is not None else 0.0
    has_R        = R_julia is not None

    A_X = np.array(Korg.format_A_X(0.0), dtype=float)

    # ── Serialise to a temp .npz ──────────────────────────────────────────────
    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    tmp.close()
    np.savez(
        tmp.name,
        tau_ref        = tau_ref,
        z              = z,
        temp           = temp,
        ne             = ne,
        ntot           = ntot,
        ref_wl         = np.array(ref_wl),
        is_spherical   = np.array(is_spherical),
        R              = np.array(R_val),
        has_R          = np.array(has_R),
        A_X            = A_X,
        ll_wl          = np.array(wl,          dtype=float),
        ll_log_gf      = np.array(log_gf,      dtype=float),
        ll_species     = np.array([str(s) for s in species]),   # Unicode array
        ll_E_lower     = np.array(E_lower,     dtype=float),
        ll_gamma_rad   = np.array(gamma_rad,   dtype=float),
        ll_gamma_stark = np.array(gamma_stark, dtype=float),
        ll_vdW_1       = np.array(vdW_1,       dtype=float),
        ll_vdW_2       = np.array(vdW_2,       dtype=float),
    )
    print(f"  Data serialised → {tmp.name}\n")

    try:
        print("Running float64 synthesis (warm-up + timed)...")
        f64 = run_mode("float64", tmp.name)
        print(f"  float64 time: {f64['time']:.2f}s")

        print("Running float32 synthesis (warm-up + timed)...")
        f32 = run_mode("float32", tmp.name)
        print(f"  float32 time: {f32['time']:.2f}s")
    finally:
        os.unlink(tmp.name)

    flux64 = np.array(f64["flux"])
    flux32 = np.array(f32["flux"])

    n_nan32  = int(np.isnan(flux32).sum())
    n_inf32  = int(np.isinf(flux32).sum())
    overflow = n_nan32 > 0 or n_inf32 > 0

    print(f"\n{'─'*60}")
    print(f"  float64 time : {f64['time']:.3f}s")
    print(f"  float32 time : {f32['time']:.3f}s")
    print(f"  Speedup      : {f64['time']/f32['time']:.2f}x  (CPU; GPU benefit would be larger)")

    if overflow:
        print(f"\n  float32 numerical overflow detected!")
        print(f"    NaN pixels : {n_nan32}/{len(flux32)}")
        print(f"    Inf pixels : {n_inf32}/{len(flux32)}")
        print(f"\n  Root cause: nu^3 at optical frequencies (~6e14 Hz)")
        print(f"    nu^3 ≈ {(6e14)**3:.1e}  >  float32 max ≈ {np.finfo(np.float32).max:.1e}")
        print(f"    This overflows in hydrogenic free-free absorption.")
        print(f"\n  VERDICT: float32 NOT viable for stellar synthesis.")
        print(f"           Code restructuring (rescaled nu) required to support float32.")
        print(f"           Use float64 (default) for all science runs.")
    else:
        rel_diff = np.abs(flux64 - flux32) / (np.abs(flux64) + 1e-30)
        print(f"\n  float32 vs float64 flux differences (5000–5050 Å):")
        print(f"    max  rel diff    : {rel_diff.max():.3e}")
        print(f"    mean rel diff    : {rel_diff.mean():.3e}")
        print(f"    % pixels < 1e-4  : {(rel_diff < 1e-4).mean()*100:.1f}%")
        print(f"    % pixels < 1e-3  : {(rel_diff < 1e-3).mean()*100:.1f}%")
        print(f"    % pixels < 1e-2  : {(rel_diff < 1e-2).mean()*100:.1f}%")
        verdict = "acceptable (<1%)" if rel_diff.max() < 0.01 else ">1% error — use float64"
        print(f"\n  VERDICT: float32 precision {verdict}")

    print(f"{'─'*60}")
