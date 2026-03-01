"""Profile each phase of korg_jax.synthesize to identify GPU conversion priority.

Measures wall-clock time for:
  - chemical equilibrium (per-layer Newton-Broyden, pure NumPy)
  - continuum absorption (batch JAX)
  - reference-wavelength line absorption
  - hydrogen line absorption
  - main line absorption (chunked JAX vmap)
  - radiative transfer (JAX)

Usage:
    uv run python benchmark/profile_synthesis.py
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Load Julia data ───────────────────────────────────────────────────────────
from juliacall import Main as jl
jl.seval("using Korg")
Korg = jl.Korg

jl.seval("""
function _extract_linelist_arrays(lines)
    wl=[l.wl for l in lines]; log_gf=[l.log_gf for l in lines]
    species=[string(l.species) for l in lines]; E_lower=[l.E_lower for l in lines]
    gamma_rad=[l.gamma_rad for l in lines]; gamma_stark=[l.gamma_stark for l in lines]
    vdW_1=Float64[]; vdW_2=Float64[]
    for l in lines
        if l.vdW isa Tuple; push!(vdW_1,l.vdW[1]); push!(vdW_2,l.vdW[2])
        else; push!(vdW_1,l.vdW); push!(vdW_2,-1.0); end
    end
    return (wl,log_gf,species,E_lower,gamma_rad,gamma_stark,vdW_1,vdW_2)
end
get_tau_refs(a)=Korg.get_tau_refs(a); get_zs(a)=Korg.get_zs(a)
get_temps(a)=Korg.get_temps(a); get_nes(a)=Korg.get_electron_number_densities(a)
get_ns(a)=Korg.get_number_densities(a); get_ref_wl(a)=a.reference_wavelength
get_is_spherical(a)=a isa Korg.ShellAtmosphere
""")

print("Loading VALD solar linelist via Korg...")
lines = Korg.get_VALD_solar_linelist()
wl, log_gf, species, E_lower, gamma_rad, gamma_stark, vdW_1, vdW_2 = \
    jl._extract_linelist_arrays(lines)

print("Interpolating solar atmosphere via Korg...")
atm_jl = Korg.interpolate_marcs(5778, 4.437)

tau_ref      = np.array(jl.get_tau_refs(atm_jl), dtype=float)
z            = np.array(jl.get_zs(atm_jl),        dtype=float)
temp         = np.array(jl.get_temps(atm_jl),     dtype=float)
ne_atm       = np.array(jl.get_nes(atm_jl),        dtype=float)
ntot         = np.array(jl.get_ns(atm_jl),         dtype=float)
ref_wl       = float(jl.get_ref_wl(atm_jl))
is_spherical = bool(jl.get_is_spherical(atm_jl))
A_X          = np.array(Korg.format_A_X(0.0), dtype=float)

# ── Build Python objects ──────────────────────────────────────────────────────
import korg_jax
import jax.numpy as jnp
from korg_jax.atmosphere import ModelAtmosphere
from korg_jax.linelist   import Linelist
from korg_jax.species    import Species, Formula
from korg_jax.wavelengths import Wavelengths
from korg_jax.synthesize  import (
    _get_statmech_setup_cached, _ChemEqContext,
    _prepare_linelist_fast, _batch_interp, _build_n_abs_matrix,
    get_reference_wavelength_linelist, _get_filtered_linelist,
    merge_bounds,
)
from korg_jax.statmech   import chemical_equilibrium
from korg_jax.continuum_absorption import total_continuum_absorption_layers
from korg_jax.line_absorption      import line_absorption_layers
from korg_jax.hydrogen_lines       import hydrogen_line_absorption_layers
from korg_jax.radiative_transfer   import radiative_transfer
from korg_jax.synthesize           import blackbody

atm = ModelAtmosphere(
    tau_ref=tau_ref, z=z, temp=temp,
    electron_number_density=ne_atm, number_density=ntot,
    reference_wavelength=ref_wl, is_spherical=is_spherical,
)
linelist_py = Linelist(
    wl=np.array(wl, dtype=float), log_gf=np.array(log_gf, dtype=float),
    species=[Species.from_string(str(s)) for s in species],
    E_lower=np.array(E_lower, dtype=float), gamma_rad=np.array(gamma_rad, dtype=float),
    gamma_stark=np.array(gamma_stark, dtype=float),
    vdW_1=np.array(vdW_1, dtype=float), vdW_2=np.array(vdW_2, dtype=float),
)
abs_ab = 10 ** (A_X - 12); abs_ab /= abs_ab.sum()

WL_RANGE       = (5000.0, 5050.0)
LINE_BUFFER    = 10.0 * 1e-8
CNTM_STEP      = 1.0  * 1e-8
LINE_CUTOFF    = 3e-4
VMIC           = 1.0
XI_CMS         = VMIC * 1e5
MU_VALUES      = 20
CHUNK_SIZE     = 1024

# ── Warm-up: full synthesize to trigger JIT ───────────────────────────────────
print("\nWarm-up run (triggers JAX JIT compilation)...")
from korg_jax.synthesize import synthesize
_ = synthesize(atm, linelist_py, A_X, WL_RANGE)
print("  done.\n")

# ── Setup (shared across timed runs) ─────────────────────────────────────────
ie, pf, leq = _get_statmech_setup_cached()
ctx = _ChemEqContext(abs_ab, ie, pf, leq)

wls = Wavelengths.from_tuple(*WL_RANGE)
use_MHD = wls.all_wls[-1] < 13000 * 1e-8

cntm_windows = [(lo - LINE_BUFFER - CNTM_STEP, hi + LINE_BUFFER + CNTM_STEP)
                for lo, hi in wls.eachwindow()]
cntm_windows, _ = merge_bounds(cntm_windows)
cntm_wls = Wavelengths([np.arange(a, b + CNTM_STEP * 0.5, CNTM_STEP)
                        for a, b in cntm_windows])

linelist_filt  = _get_filtered_linelist(linelist_py, wls, LINE_BUFFER)
linelist5      = get_reference_wavelength_linelist(linelist_py, atm.reference_wavelength)

linelist_jax, linelist_chunks, unique_sp, line_sp_idx = _prepare_linelist_fast(
    linelist_filt, CHUNK_SIZE)
linelist5_jax, linelist5_chunks, unique_sp5, line5_sp_idx = _prepare_linelist_fast(
    linelist5, CHUNK_SIZE)

from korg_jax.constants import c_cgs
ref_freq      = c_cgs / atm.reference_wavelength
combined_freqs = np.append(cntm_wls.all_freqs, ref_freq)

H_I   = Species(Formula.from_Z(1), 0)
He_I  = Species(Formula.from_Z(2), 0)
H_III = Species(Formula.from_Z(1), 2)

N_REPEATS = 5

def timeit(label, fn, n=N_REPEATS):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    med = np.median(times)
    print(f"  {label:<45s}  {med*1000:7.1f} ms  (median of {n})")
    return result, med

print("=" * 70)
print(f"Phase timing  (median of {N_REPEATS} runs, post-JIT)")
print("=" * 70)

# ── 1. Chemical equilibrium (statmech) ───────────────────────────────────────
def run_cheq():
    n_dicts = []
    n_e_vals = np.zeros(atm.n_layers)
    for i in range(atm.n_layers):
        ne_i, nd_i = chemical_equilibrium(
            atm.temp[i], atm.number_density[i], atm.electron_number_density[i],
            abs_ab, ie, pf, leq, _ctx=ctx)
        n_dicts.append(nd_i)
        n_e_vals[i] = ne_i
    return n_dicts, n_e_vals

(n_dicts, n_e_vals), t_cheq = timeit("Chemical equilibrium (56 layers, NumPy)", run_cheq)

number_densities = {spec: np.array([n[spec] for n in n_dicts])
                    for spec in n_dicts[0] if spec != H_III}
nH_I_vals  = number_densities.get(H_I,  np.zeros(atm.n_layers))
nHe_I_vals = number_densities.get(He_I, np.zeros(atm.n_layers))

# ── 2. Continuum absorption ───────────────────────────────────────────────────
_, t_cntm = timeit(
    "Continuum absorption (batch JAX)",
    lambda: total_continuum_absorption_layers(
        combined_freqs, atm.temp, n_e_vals, number_densities, pf))

alpha_combined = np.asarray(total_continuum_absorption_layers(
    combined_freqs, atm.temp, n_e_vals, number_densities, pf))
alpha_ref_cntm = alpha_combined[:, -1]
alpha_cntm_vals = alpha_combined[:, :-1][:, ::-1]

# ── 3. Continuum interpolation ────────────────────────────────────────────────
_, t_interp = timeit(
    "Continuum interpolation to synthesis grid",
    lambda: _batch_interp(wls.all_wls, cntm_wls.all_wls, alpha_cntm_vals))

alpha_np = _batch_interp(wls.all_wls, cntm_wls.all_wls, alpha_cntm_vals)

# ── 4. Reference-wavelength line absorption ───────────────────────────────────
n_abs_ref = _build_n_abs_matrix(n_dicts, unique_sp5, line5_sp_idx, pf, atm.temp)
alpha_cntm_ref = np.broadcast_to(
    alpha_ref_cntm[:, np.newaxis], (atm.n_layers, linelist5.n_lines)).copy()

_, t_reflines = timeit(
    "Ref-wl line absorption (anchored tau)",
    lambda: line_absorption_layers(
        jnp.asarray([atm.reference_wavelength]), linelist5_jax,
        jnp.asarray(atm.temp), jnp.asarray(n_e_vals), jnp.asarray(nH_I_vals),
        jnp.asarray(n_abs_ref), XI_CMS,
        alpha_cntm_at_line=jnp.asarray(alpha_cntm_ref),
        cutoff_threshold=LINE_CUTOFF, chunks=linelist5_chunks).block_until_ready())

# ── 5. Hydrogen line absorption ───────────────────────────────────────────────
_pf_H_I   = pf[Species(Formula.from_Z(1), 0)]
U_H_I_vals = np.asarray(_pf_H_I.batch(jnp.log(jnp.asarray(atm.temp))))

_, t_hlines = timeit(
    "Hydrogen line absorption",
    lambda: np.asarray(hydrogen_line_absorption_layers(
        wls.all_wls, atm.temp, n_e_vals, nH_I_vals, nHe_I_vals, U_H_I_vals,
        XI_CMS, 150 * 1e-8, use_MHD=use_MHD)))

h_alpha = np.asarray(hydrogen_line_absorption_layers(
    wls.all_wls, atm.temp, n_e_vals, nH_I_vals, nHe_I_vals, U_H_I_vals,
    XI_CMS, 150 * 1e-8, use_MHD=use_MHD))
alpha = alpha_np + h_alpha

# ── 6. Main line absorption ───────────────────────────────────────────────────
n_abs_matrix = _build_n_abs_matrix(n_dicts, unique_sp, line_sp_idx, pf, atm.temp)
alpha_cntm_at_lines = (_batch_interp(linelist_filt.wl, wls.all_wls, alpha_np)
                       if linelist_filt.n_lines > 0
                       else np.zeros((atm.n_layers, 0)))

_, t_lines = timeit(
    "Main line absorption (chunked JAX vmap)",
    lambda: line_absorption_layers(
        jnp.asarray(wls.all_wls), linelist_jax,
        jnp.asarray(atm.temp), jnp.asarray(n_e_vals), jnp.asarray(nH_I_vals),
        jnp.asarray(n_abs_matrix), XI_CMS,
        alpha_cntm_at_line=jnp.asarray(alpha_cntm_at_lines),
        cutoff_threshold=LINE_CUTOFF, chunks=linelist_chunks).block_until_ready())

alpha_lines = np.asarray(line_absorption_layers(
    jnp.asarray(wls.all_wls), linelist_jax,
    jnp.asarray(atm.temp), jnp.asarray(n_e_vals), jnp.asarray(nH_I_vals),
    jnp.asarray(n_abs_matrix), XI_CMS,
    alpha_cntm_at_line=jnp.asarray(alpha_cntm_at_lines),
    cutoff_threshold=LINE_CUTOFF, chunks=linelist_chunks))
alpha = alpha + alpha_lines

# ── 7. Source function ────────────────────────────────────────────────────────
_, t_src = timeit(
    "Source function (Planck / blackbody)",
    lambda: np.asarray(blackbody(jnp.asarray(atm.temp)[:, None],
                                  jnp.asarray(wls.all_wls)[None, :])))

source_fn = np.asarray(blackbody(jnp.asarray(atm.temp)[:, None],
                                  jnp.asarray(wls.all_wls)[None, :]))

# ── 8. Radiative transfer ─────────────────────────────────────────────────────
_, t_rt = timeit(
    "Radiative transfer (JAX)",
    lambda: radiative_transfer(atm, alpha, source_fn, MU_VALUES,
                               alpha_ref=alpha_ref_cntm, tau_ref=atm.tau_ref))

# ── Summary ───────────────────────────────────────────────────────────────────
total = t_cheq + t_cntm + t_interp + t_reflines + t_hlines + t_lines + t_src + t_rt

print()
print("=" * 70)
print(f"{'Phase':<45s}  {'Time':>8s}  {'Share':>6s}  Backend")
print("-" * 70)
rows = [
    ("Chemical equilibrium (statmech)",    t_cheq,    "NumPy"),
    ("Continuum absorption",               t_cntm,    "JAX"),
    ("Continuum interpolation",            t_interp,  "NumPy"),
    ("Ref-wl line absorption",             t_reflines,"JAX"),
    ("Hydrogen line absorption",           t_hlines,  "NumPy"),
    ("Main line absorption",               t_lines,   "JAX"),
    ("Source function",                    t_src,     "JAX"),
    ("Radiative transfer",                 t_rt,      "JAX"),
]
for name, t, backend in rows:
    bar = "█" * int(round(t / total * 30))
    print(f"  {name:<43s}  {t*1000:7.1f} ms  {t/total*100:5.1f}%  {backend}  {bar}")

print("-" * 70)
print(f"  {'TOTAL (accounted phases)':<43s}  {total*1000:7.1f} ms")
print()

numpy_total = t_cheq + t_interp + t_hlines
jax_total   = t_cntm + t_reflines + t_lines + t_src + t_rt
print(f"  NumPy phases total : {numpy_total*1000:.1f} ms  ({numpy_total/total*100:.1f}%)")
print(f"  JAX   phases total : {jax_total*1000:.1f} ms  ({jax_total/total*100:.1f}%)")
print()
print(f"  n_layers = {atm.n_layers},  n_wl = {len(wls.all_wls)},  n_lines = {linelist_filt.n_lines}")
print("=" * 70)
