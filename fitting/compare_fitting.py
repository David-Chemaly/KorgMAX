"""
compare_fitting.py — Validate korg_jax against Julia Korg using the
Griffith et al. (2022) fitting dataset.

Star: 2MASS J03443498+0553014
Published parameters: Teff=5456 K, logg=3.86, [M/H]=-1.22, vmic=1.23 km/s

This script:
  1. Parses the Griffith linelist (lines.csv) into both Julia Korg and korg_jax
  2. Interpolates a MARCS atmosphere at the published stellar parameters
  3. Synthesizes a spectral region with both codes
  4. Compares linelist properties, continuum, flux, and normalized flux
  5. Reads the observed spectrum and overlays fitting windows

Usage:
    python fitting/compare_fitting.py
"""
import numpy as np
import pandas as pd
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

# ── Configuration ─────────────────────────────────────────────────────────────

# Griffith et al. (2022) published stellar parameters
TEFF = 5456.0
LOGG = 3.86
M_H = -1.22
VMIC = 1.23  # km/s

# Synthesis wavelength range (Angstroms)
# 4600-4650 A has 21 fitting windows (11 Fe 1) and observed data coverage
WL_START, WL_STOP = 4600.0, 4650.0

# Paths
LINES_CSV = os.path.join(SCRIPT_DIR, "lines.csv")
WINDOWS_TSV = os.path.join(SCRIPT_DIR, "windows.tsv")
OBSERVED_CSV = os.path.join(SCRIPT_DIR, "2MASS_J03443498+0553014.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: JULIA KORG
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("FITTING COMPARISON: Julia Korg vs korg_jax")
print("Star: 2MASS J03443498+0553014 (Griffith et al. 2022)")
print(f"Teff={TEFF} K, logg={LOGG}, [M/H]={M_H}, vmic={VMIC} km/s")
print(f"Wavelength range: {WL_START:.0f}-{WL_STOP:.0f} A")
print("=" * 70)

print("\n[1/6] Loading Julia Korg...")
from juliacall import Main as jl
jl.seval("using Korg")
Korg = jl.Korg

# ── Read linelist ─────────────────────────────────────────────────────────────
print("[2/6] Reading and building linelists...")
linetable = pd.read_csv(LINES_CSV)
n_total = len(linetable)

# Filter out H 1 lines (Korg has built-in hydrogen line treatment)
linetable = linetable[linetable.element != "H 1"]
n_filtered = len(linetable)
print(f"  lines.csv: {n_total} total, {n_filtered} after H 1 filter")

# Extract arrays for both Julia and Python usage
wave_A = linetable.wave_A.values.astype(float)
loggf_arr = linetable.loggf.values.astype(float)
E_lower_arr = linetable.lower_state_eV.values.astype(float)
element_arr = linetable.element.values
rad_arr = linetable.rad.values.astype(float)    # NaN where empty
stark_arr = linetable.stark.values.astype(float)
waals_arr = linetable.waals.values.astype(float)

# Build Julia linelist (same approach as the fitting notebook)
print("  Building Julia linelist via Korg.Line...")
jl_lines = jl.broadcast(
    Korg.Line,
    jl.broadcast(Korg.air_to_vacuum, wave_A),
    loggf_arr,
    jl.broadcast(Korg.Species, element_arr),
    E_lower_arr, rad_arr, stark_arr, waals_arr,
)

# Extract Julia linelist arrays for line-by-line comparison
jl.seval("""
function _extract_linelist_arrays(lines)
    wl = [l.wl for l in lines]
    log_gf = [l.log_gf for l in lines]
    species = [string(l.species) for l in lines]
    E_lower = [l.E_lower for l in lines]
    gamma_rad = [l.gamma_rad for l in lines]
    gamma_stark = [l.gamma_stark for l in lines]
    vdW_1 = Float64[]
    vdW_2 = Float64[]
    for l in lines
        if l.vdW isa Tuple
            push!(vdW_1, l.vdW[1])
            push!(vdW_2, l.vdW[2])
        else
            push!(vdW_1, l.vdW)
            push!(vdW_2, -1.0)
        end
    end
    return (wl, log_gf, species, E_lower, gamma_rad, gamma_stark, vdW_1, vdW_2)
end
""")
(jl_wl_raw, jl_loggf_raw, jl_species_raw, jl_Elower_raw,
 jl_grad_raw, jl_gstark_raw, jl_vdW1_raw, jl_vdW2_raw) = \
    jl._extract_linelist_arrays(jl_lines)

jl_wl = np.array(jl_wl_raw, dtype=float)
jl_loggf = np.array(jl_loggf_raw, dtype=float)
jl_Elower = np.array(jl_Elower_raw, dtype=float)
jl_grad = np.array(jl_grad_raw, dtype=float)
jl_gstark = np.array(jl_gstark_raw, dtype=float)
jl_vdW1 = np.array(jl_vdW1_raw, dtype=float)
jl_vdW2 = np.array(jl_vdW2_raw, dtype=float)
print(f"  Julia linelist: {len(jl_wl)} lines extracted")

# ── Atmosphere from Korg ──────────────────────────────────────────────────────
print(f"[3/6] Interpolating MARCS atmosphere...")
A_X_jl = np.array(Korg.format_A_X(M_H), dtype=float)
atm_jl = Korg.interpolate_marcs(TEFF, LOGG, Korg.format_A_X(M_H))

# Extract atmosphere arrays for korg_jax
jl.seval("""
get_tau_refs(atm) = Korg.get_tau_refs(atm)
get_zs(atm) = Korg.get_zs(atm)
get_temps(atm) = Korg.get_temps(atm)
get_nes(atm) = Korg.get_electron_number_densities(atm)
get_ns(atm) = Korg.get_number_densities(atm)
get_ref_wl(atm) = atm.reference_wavelength
get_is_spherical(atm) = atm isa Korg.ShellAtmosphere
get_R(atm) = get_is_spherical(atm) ? atm.R : nothing
""")
tau_ref = np.array(jl.get_tau_refs(atm_jl), dtype=float)
z_arr = np.array(jl.get_zs(atm_jl), dtype=float)
temp_arr = np.array(jl.get_temps(atm_jl), dtype=float)
ne_arr = np.array(jl.get_nes(atm_jl), dtype=float)
ntot_arr = np.array(jl.get_ns(atm_jl), dtype=float)
ref_wl = float(jl.get_ref_wl(atm_jl))
is_spherical = bool(jl.get_is_spherical(atm_jl))
R_phot = jl.get_R(atm_jl)
R_phot = float(R_phot) if R_phot is not None else None

print(f"  {len(tau_ref)} layers, "
      f"{'spherical' if is_spherical else 'plane-parallel'}, "
      f"T range: {temp_arr.min():.0f}-{temp_arr.max():.0f} K")

# ── Synthesize with Julia Korg ────────────────────────────────────────────────
print(f"[4/6] Synthesizing with Julia Korg...")
t0 = time.perf_counter()
sol_korg = Korg.synthesize(
    atm_jl, jl_lines, Korg.format_A_X(M_H), (WL_START, WL_STOP), vmic=VMIC
)
t_korg = time.perf_counter() - t0
flux_korg = np.array(sol_korg.flux, dtype=float)
wl_korg = np.array(sol_korg.wavelengths, dtype=float)
cntm_korg = np.array(sol_korg.cntm, dtype=float)
print(f"  {t_korg:.2f}s, {len(wl_korg)} wavelength pixels")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: KORG_JAX
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[5/6] Building korg_jax objects and synthesizing...")
from korg_jax.atmosphere import ModelAtmosphere
from korg_jax.linelist import (
    Linelist, _make_line, _dicts_to_linelist, _air_to_vacuum_cm,
)
from korg_jax.species import Species
from korg_jax.synthesize import synthesize
from korg_jax.abundances import format_A_X
from korg_jax.read_statmech import setup_ionization_energies

# Build korg_jax atmosphere from the extracted arrays
atm_py = ModelAtmosphere(
    tau_ref=tau_ref, z=z_arr, temp=temp_arr,
    electron_number_density=ne_arr, number_density=ntot_arr,
    reference_wavelength=ref_wl, is_spherical=is_spherical, R=R_phot,
)

# Use Julia's A_X to eliminate solar abundance differences
A_X = A_X_jl.copy()

# Also compute korg_jax's A_X for comparison
A_X_py = format_A_X(M_H)
ax_diff = np.max(np.abs(A_X_jl - A_X_py))
if ax_diff > 1e-10:
    print(f"  NOTE: A_X differs by up to {ax_diff:.2e} between Julia and Python")
    print(f"        Using Julia A_X for fair synthesis comparison")
else:
    print(f"  A_X agrees to <1e-10 between Julia and Python")

# Load ionization energies for accurate broadening approximation
ion_e = setup_ionization_energies()

# Parse linelist from CSV into korg_jax format
print("  Parsing linelist into korg_jax format...")
t_parse = time.perf_counter()

# Vectorized air-to-vacuum conversion
wave_cm = wave_A * 1e-8
lam_A_for_conv = wave_cm * 1e8
s = 1e4 / lam_A_for_conv
n_refrac = (1.0
            + 0.00008336624212083
            + 0.02408926869968 / (130.1065924522 - s * s)
            + 0.0001599740894897 / (38.92568793293 - s * s))
wave_vac_cm = wave_cm * n_refrac

# Pre-build Species objects
species_list = [Species.from_string(str(e)) for e in element_arr]

# Build line dicts via _make_line (handles broadening approximation)
lines_py = []
for i in range(len(wave_A)):
    gamma_rad = float(rad_arr[i]) if not np.isnan(rad_arr[i]) else None
    gamma_stark = float(stark_arr[i]) if not np.isnan(stark_arr[i]) else None
    vdW = float(waals_arr[i]) if not np.isnan(waals_arr[i]) else None

    line = _make_line(
        wave_vac_cm[i], loggf_arr[i], species_list[i], E_lower_arr[i],
        gamma_rad=gamma_rad, gamma_stark=gamma_stark, vdW=vdW,
        ion_e=ion_e,
    )
    lines_py.append(line)

linelist_py = _dicts_to_linelist(lines_py).sort_by_wavelength()
t_parse = time.perf_counter() - t_parse
print(f"  korg_jax linelist: {len(linelist_py.wl)} lines ({t_parse:.1f}s)")

# Synthesize with korg_jax
print("  Synthesizing with korg_jax...")
t0 = time.perf_counter()
sol_jax = synthesize(atm_py, linelist_py, A_X, (WL_START, WL_STOP), vmic=VMIC)
t_jax = time.perf_counter() - t0
flux_jax = np.array(sol_jax.flux, dtype=float)
wl_jax = np.array(sol_jax.wavelengths, dtype=float)
cntm_jax = np.array(sol_jax.cntm, dtype=float)
print(f"  {t_jax:.2f}s, {len(wl_jax)} wavelength pixels")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[6/6] Comparing results...")
print()
print("=" * 70)
print("RESULTS")
print("=" * 70)

# ── Linelist comparison ───────────────────────────────────────────────────────

# Both are sorted by wavelength
jl_sort = np.argsort(jl_wl)
py_wl = linelist_py.wl  # already sorted by sort_by_wavelength()

print(f"\n--- Linelist ({n_filtered} lines) ---")
print(f"  Python: {len(py_wl)} lines, Julia: {len(jl_wl)} lines")

if len(py_wl) == len(jl_wl):
    jl_wl_sorted = jl_wl[jl_sort]
    wl_diff = np.abs(py_wl - jl_wl_sorted)
    rel_wl_diff = wl_diff / jl_wl_sorted
    print(f"  Wavelengths:")
    print(f"    max |dWL|:     {wl_diff.max():.2e} cm")
    print(f"    max rel:       {rel_wl_diff.max():.2e}")

    # log_gf
    jl_loggf_sorted = jl_loggf[jl_sort]
    py_loggf = linelist_py.log_gf
    loggf_diff = np.abs(py_loggf - jl_loggf_sorted)
    print(f"  log_gf:")
    print(f"    max |diff|:    {loggf_diff.max():.2e}")

    # E_lower
    jl_Elower_sorted = jl_Elower[jl_sort]
    py_Elower = linelist_py.E_lower
    el_diff = np.abs(py_Elower - jl_Elower_sorted)
    print(f"  E_lower:")
    print(f"    max |diff|:    {el_diff.max():.2e} eV")

    # gamma_rad
    jl_grad_sorted = jl_grad[jl_sort]
    py_grad = linelist_py.gamma_rad
    grad_mask = (jl_grad_sorted > 0) & (py_grad > 0)
    if grad_mask.any():
        rel_grad = np.abs(jl_grad_sorted[grad_mask] - py_grad[grad_mask]) / \
                   np.maximum(1e-30, jl_grad_sorted[grad_mask])
        n_exact = (rel_grad < 1e-6).sum()
        print(f"  gamma_rad:")
        print(f"    exact (<1e-6): {n_exact}/{grad_mask.sum()}")
        print(f"    max rel diff:  {rel_grad.max():.2e}")

    # gamma_stark
    jl_gstark_sorted = jl_gstark[jl_sort]
    py_gstark = linelist_py.gamma_stark
    gs_mask = (jl_gstark_sorted > 0) & (py_gstark > 0)
    if gs_mask.any():
        rel_gs = np.abs(jl_gstark_sorted[gs_mask] - py_gstark[gs_mask]) / \
                 np.maximum(1e-30, jl_gstark_sorted[gs_mask])
        n_exact_gs = (rel_gs < 1e-6).sum()
        print(f"  gamma_stark:")
        print(f"    exact (<1e-6): {n_exact_gs}/{gs_mask.sum()}")
        print(f"    max rel diff:  {rel_gs.max():.2e}")

    # vdW_1
    jl_vdW1_sorted = jl_vdW1[jl_sort]
    py_vdW1 = linelist_py.vdW_1
    vdw_mask = (np.abs(jl_vdW1_sorted) > 0) & (np.abs(py_vdW1) > 0)
    if vdw_mask.any():
        rel_vdw = np.abs(jl_vdW1_sorted[vdw_mask] - py_vdW1[vdw_mask]) / \
                  np.maximum(1e-30, np.abs(jl_vdW1_sorted[vdw_mask]))
        n_exact_vdw = (rel_vdw < 1e-6).sum()
        print(f"  vdW_1:")
        print(f"    exact (<1e-6): {n_exact_vdw}/{vdw_mask.sum()}")
        print(f"    max rel diff:  {rel_vdw.max():.2e}")

    # vdW_2
    jl_vdW2_sorted = jl_vdW2[jl_sort]
    py_vdW2 = linelist_py.vdW_2
    vdw2_mask = (jl_vdW2_sorted > 0) & (py_vdW2 > 0)
    if vdw2_mask.any():
        rel_vdw2 = np.abs(jl_vdW2_sorted[vdw2_mask] - py_vdW2[vdw2_mask]) / \
                   np.maximum(1e-30, np.abs(jl_vdW2_sorted[vdw2_mask]))
        n_exact_vdw2 = (rel_vdw2 < 1e-6).sum()
        print(f"  vdW_2:")
        print(f"    exact (<1e-6): {n_exact_vdw2}/{vdw2_mask.sum()}")
        print(f"    max rel diff:  {rel_vdw2.max():.2e}")
else:
    print(f"  WARNING: line count mismatch!")

# ── Wavelength grid comparison ────────────────────────────────────────────────

print(f"\n--- Wavelength grid ---")
if len(wl_korg) == len(wl_jax):
    wl_grid_diff = np.max(np.abs(wl_korg - wl_jax))
    print(f"  {len(wl_korg)} pixels")
    print(f"  max |dWL| = {wl_grid_diff:.2e} A")
else:
    print(f"  WARNING: grid size mismatch (Julia={len(wl_korg)}, Python={len(wl_jax)})")

# ── Continuum comparison ─────────────────────────────────────────────────────

print(f"\n--- Continuum ---")
rel_cntm = np.abs(cntm_korg - cntm_jax) / np.maximum(1e-30, np.abs(cntm_korg))
print(f"  max  rel diff: {rel_cntm.max():.6e}")
print(f"  mean rel diff: {rel_cntm.mean():.6e}")

# ── Flux comparison ──────────────────────────────────────────────────────────

print(f"\n--- Flux ---")
rel_flux = np.abs(flux_korg - flux_jax) / np.maximum(1e-30, np.abs(flux_korg))
print(f"  max  rel diff: {rel_flux.max():.6e}")
print(f"  mean rel diff: {rel_flux.mean():.6e}")
idx_max = np.argmax(rel_flux)
print(f"  worst pixel:   WL={wl_jax[idx_max]:.2f} A, "
      f"Julia={flux_korg[idx_max]:.6f}, Python={flux_jax[idx_max]:.6f}")

print(f"\n--- Error distribution ---")
for thresh in [1e-5, 1e-4, 1e-3, 1e-2]:
    pct = np.mean(rel_flux < thresh) * 100
    print(f"  {pct:6.1f}% of pixels have rel error < {thresh:.0e}")

# ── Normalized flux comparison ───────────────────────────────────────────────

print(f"\n--- Normalized flux (F/Fcntm) ---")
norm_korg = flux_korg / cntm_korg
norm_jax = flux_jax / cntm_jax
abs_norm = np.abs(norm_korg - norm_jax)
print(f"  max  abs diff: {abs_norm.max():.6e}")
print(f"  mean abs diff: {abs_norm.mean():.6e}")
print(f"  (This is what matters for fitting spectra)")

# ── Observed spectrum comparison ─────────────────────────────────────────────

print(f"\n--- Observed spectrum ---")
obs = pd.read_csv(OBSERVED_CSV, delimiter="\t")
obs_wl_nm = obs.waveobs.values
obs_flux_raw = obs.flux.values
obs_err_raw = obs.err.values

# Convert observed wavelengths from air nm to vacuum Angstroms
from korg_jax.utils import air_to_vacuum
obs_wl_vac_A = np.array([air_to_vacuum(w * 10.0) for w in obs_wl_nm])

# Find observed pixels within our synthesis range
obs_mask = (obs_wl_vac_A >= WL_START) & (obs_wl_vac_A <= WL_STOP)
n_obs = obs_mask.sum()
print(f"  Observed spectrum: {len(obs_wl_nm)} total pixels")
print(f"  Pixels in {WL_START:.0f}-{WL_STOP:.0f} A: {n_obs}")

if n_obs > 0:
    obs_wl_in = obs_wl_vac_A[obs_mask]
    obs_flux_in = obs_flux_raw[obs_mask]

    # Interpolate model normalized fluxes to observed wavelengths
    from scipy.interpolate import interp1d
    f_korg = interp1d(wl_korg, norm_korg, kind='linear',
                      bounds_error=False, fill_value=np.nan)
    f_jax = interp1d(wl_jax, norm_jax, kind='linear',
                     bounds_error=False, fill_value=np.nan)

    model_korg_at_obs = f_korg(obs_wl_in)
    model_jax_at_obs = f_jax(obs_wl_in)

    valid = np.isfinite(model_korg_at_obs) & np.isfinite(model_jax_at_obs)
    if valid.any():
        resid_korg = np.abs(obs_flux_in[valid] - model_korg_at_obs[valid])
        resid_jax = np.abs(obs_flux_in[valid] - model_jax_at_obs[valid])
        resid_model = np.abs(model_korg_at_obs[valid] - model_jax_at_obs[valid])
        print(f"\n  Model vs observed (unconvolved, for reference only):")
        print(f"    Julia Korg:  mean |resid| = {resid_korg.mean():.4f}, "
              f"max = {resid_korg.max():.4f}")
        print(f"    korg_jax:    mean |resid| = {resid_jax.mean():.4f}, "
              f"max = {resid_jax.max():.4f}")
        print(f"    Korg vs jax: mean |diff|  = {resid_model.mean():.6f}, "
              f"max = {resid_model.max():.6f}")
        print(f"    (Model is unconvolved; observed spectrum is at R~50000)")
else:
    print(f"  No observed pixels in synthesis range.")

# ── Fitting windows in synthesis range ────────────────────────────────────────

print(f"\n--- Fitting windows in {WL_START:.0f}-{WL_STOP:.0f} A ---")
wintable = pd.read_csv(WINDOWS_TSV, delimiter="\t")
win_lo = np.array([air_to_vacuum(w * 10.0) for w in wintable.wave_base.values])
win_hi = np.array([air_to_vacuum(w * 10.0) for w in wintable.wave_top.values])
win_species = wintable.species.values

win_mask = (win_hi >= WL_START) & (win_lo <= WL_STOP)
n_windows = win_mask.sum()
print(f"  Overlapping windows: {n_windows}")

if n_windows > 0:
    for idx in np.where(win_mask)[0]:
        wlo, whi = win_lo[idx], win_hi[idx]
        sp = win_species[idx]
        # Compare model flux in this specific window
        wmask = (wl_korg >= wlo) & (wl_korg <= whi)
        if wmask.any():
            local_rel = rel_flux[wmask]
            local_norm_diff = abs_norm[wmask]
            print(f"    {wlo:.1f}-{whi:.1f} A ({sp:6s}): "
                  f"rel flux diff: max={local_rel.max():.2e} mean={local_rel.mean():.2e}, "
                  f"norm diff: max={local_norm_diff.max():.2e}")
        else:
            print(f"    {wlo:.1f}-{whi:.1f} A ({sp:6s}): no synthesis pixels")

# ── Timing summary ───────────────────────────────────────────────────────────

print(f"\n--- Timing ---")
print(f"  Julia Korg synthesis: {t_korg:.2f}s")
print(f"  korg_jax synthesis:   {t_jax:.2f}s")
print(f"  Linelist parsing:     {t_parse:.1f}s")

# ── Verdict ───────────────────────────────────────────────────────────────────

print()
print("=" * 70)
all_pass = True

if rel_flux.max() < 0.01:
    print("PASS: all flux values agree to <1%")
elif rel_flux.max() < 0.05:
    print("PASS (marginal): all flux values agree to <5%")
else:
    print(f"WARNING: some flux values differ by >{rel_flux.max()*100:.1f}%")
    all_pass = False

if abs_norm.max() < 0.01:
    print("PASS: normalized flux agrees to <0.01")
else:
    print(f"WARNING: normalized flux differs by up to {abs_norm.max():.4f}")
    all_pass = False

if rel_cntm.max() < 0.01:
    print("PASS: continuum agrees to <1%")
else:
    print(f"WARNING: continuum differs by up to {rel_cntm.max()*100:.2f}%")
    all_pass = False

if len(py_wl) == len(jl_wl):
    print("PASS: linelist sizes match")
else:
    print("WARNING: linelist sizes differ")
    all_pass = False

if all_pass:
    print("\nALL CHECKS PASSED")
else:
    print("\nSOME CHECKS FAILED — investigate above warnings")

print("=" * 70)
