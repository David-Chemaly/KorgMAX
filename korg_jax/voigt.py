"""JAX Voigt-Hjerting function H(a, v).

Ported from Korg.jl/src/line_absorption.jl (Hunger 1965 approximation).
All branches use ``jnp.where`` for JIT compatibility.
"""
import jax
import jax.numpy as jnp


@jax.jit
def _harris_series(v):
    v2 = v * v
    H0 = jnp.exp(-v2)

    h1_low = -1.12470432 + (-0.15516677 + (3.288675912 + (-2.34357915 + 0.42139162 * v) * v) * v) * v
    h1_mid = -4.48480194 + (9.39456063 + (-6.61487486 + (1.98919585 - 0.22041650 * v) * v) * v) * v
    h1_high = (
        (0.554153432 +
         (0.278711796 + (-0.1883256872 + (0.042991293 - 0.003278278 * v) * v) * v) * v)
        / (v2 - 1.5)
    )
    H1 = jnp.where(v < 1.3, h1_low, jnp.where(v < 2.4, h1_mid, h1_high))
    H2 = (1.0 - 2.0 * v2) * H0
    return H0, H1, H2


@jax.jit
def voigt_hjerting(a, v):
    """Compute the Voigt-Hjerting function ``H(a, v)``."""
    a = jnp.asarray(a, dtype=jnp.float64)
    v = jnp.asarray(v, dtype=jnp.float64)

    v2 = v * v

    # Branch 1: a <= 0.2 and v >= 5
    invv2 = 1.0 / v2
    branch1 = (a / jnp.sqrt(jnp.pi) * invv2) * (1.0 + 1.5 * invv2 + 3.75 * invv2 * invv2)

    # Branch 2: a <= 0.2 and v < 5 (Harris series)
    H0, H1, H2 = _harris_series(v)
    branch2 = H0 + (H1 + H2 * a) * a

    # Branch 3: (a <= 1.4) and (a + v < 3.2)
    M0 = H0
    M1 = H1 + 2.0 / jnp.sqrt(jnp.pi) * M0
    M2 = H2 - M0 + 2.0 / jnp.sqrt(jnp.pi) * M1
    M3 = 2.0 / (3.0 * jnp.sqrt(jnp.pi)) * (1.0 - H2) - (2.0 / 3.0) * v2 * M1 + 2.0 / jnp.sqrt(jnp.pi) * M2
    M4 = 2.0 / 3.0 * v2 * v2 * M0 - 2.0 / (3.0 * jnp.sqrt(jnp.pi)) * M1 + 2.0 / jnp.sqrt(jnp.pi) * M3
    psi = 0.979895023 + (-0.962846325 + (0.532770573 - 0.122727278 * a) * a) * a
    branch3 = psi * (M0 + (M1 + (M2 + (M3 + M4 * a) * a) * a) * a)

    # Branch 4: a > 1.4 or (a > 0.2 and a + v > 3.2)
    r2 = v2 / (a * a)
    a_invu = 1.0 / jnp.sqrt(2.0) / ((r2 + 1.0) * a)
    a2_invu2 = a_invu * a_invu
    branch4 = jnp.sqrt(2.0 / jnp.pi) * a_invu * (
        1.0 + (3.0 * r2 - 1.0 + ((r2 - 2.0) * 15.0 * r2 + 2.0) * a2_invu2) * a2_invu2
    )

    cond1 = (a <= 0.2) & (v >= 5.0)
    cond2 = (a <= 0.2)
    cond3 = (a <= 1.4) & ((a + v) < 3.2)

    out = jnp.where(cond1, branch1, jnp.where(cond2, branch2, jnp.where(cond3, branch3, branch4)))
    return out
