import numpy as np
from juliacall import Main as jl

# Load Korg
jl.seval("using Korg")
Korg = jl.Korg

# Build linelist in Julia and extract arrays
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

print("Loading VALD solar linelist via Korg...")
lines = Korg.get_VALD_solar_linelist()
wl, log_gf, species, E_lower, gamma_rad, gamma_stark, vdW_1, vdW_2 = jl._extract_linelist_arrays(lines)

# Atmosphere from Korg
print("Interpolating atmosphere via Korg...")
dwarf_atm = Korg.interpolate_marcs(5221, 4.32)

# Extract atmosphere arrays
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

tau_ref = np.array(jl.get_tau_refs(dwarf_atm), dtype=float)
z = np.array(jl.get_zs(dwarf_atm), dtype=float)
temp = np.array(jl.get_temps(dwarf_atm), dtype=float)
ne = np.array(jl.get_nes(dwarf_atm), dtype=float)
ntot = np.array(jl.get_ns(dwarf_atm), dtype=float)
ref_wl = float(jl.get_ref_wl(dwarf_atm))
is_spherical = bool(jl.get_is_spherical(dwarf_atm))
R = jl.get_R(dwarf_atm)
R = float(R) if R is not None else None

# A_X from Korg
A_X = np.array(Korg.format_A_X(0.0), dtype=float)

# Run Korg synthesize
print("Running Korg.synthesize...")
sol = Korg.synthesize(dwarf_atm, lines, A_X, 5000, 5030)
flux_korg = np.array(sol.flux, dtype=float)
wl_korg = np.array(sol.wavelengths, dtype=float)

# Build korg_jax objects
print("Running korg_jax.synthesize...")
from korg_jax.atmosphere import ModelAtmosphere
from korg_jax.linelist import Linelist
from korg_jax.species import Species
from korg_jax.synthesize import synthesize

linelist_py = Linelist(
    wl=np.array(wl, dtype=float),
    log_gf=np.array(log_gf, dtype=float),
    species=[Species.from_string(s) for s in species],
    E_lower=np.array(E_lower, dtype=float),
    gamma_rad=np.array(gamma_rad, dtype=float),
    gamma_stark=np.array(gamma_stark, dtype=float),
    vdW_1=np.array(vdW_1, dtype=float),
    vdW_2=np.array(vdW_2, dtype=float),
)

atm_py = ModelAtmosphere(
    tau_ref=tau_ref,
    z=z,
    temp=temp,
    electron_number_density=ne,
    number_density=ntot,
    reference_wavelength=ref_wl,
    is_spherical=is_spherical,
    R=R,
)

sol_jax = synthesize(atm_py, linelist_py, A_X, (5000, 5030))
flux_jax = np.array(sol_jax.flux, dtype=float)
wl_jax = np.array(sol_jax.wavelengths, dtype=float)

# Compare
print("Comparing wavelengths...")
wl_diff = np.max(np.abs(wl_korg - wl_jax))
print("max |Δλ| =", wl_diff)

print("Comparing flux...")
abs_diff = np.max(np.abs(flux_korg - flux_jax))
rel_diff = np.max(np.abs(flux_korg - flux_jax) / np.maximum(1e-30, np.abs(flux_korg)))
print("max |ΔF| =", abs_diff)
print("max rel ΔF =", rel_diff)

if wl_diff == 0 and abs_diff == 0:
    print("Exact match.")
else:
    print("Not exact. See diffs above.")
