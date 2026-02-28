"""JAX-compatible cubic spline interpolation.

Ported from Korg.jl/src/CubicSplines.jl.

Two-phase design:
  - ``cubic_spline_build(t, u)`` — pure NumPy, called once at init time.
  - ``cubic_spline_eval(t, u, h, z, x)`` — JAX jit-compatible evaluation.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp


# ── Build (NumPy, one-time) ──────────────────────────────────────────────────

def cubic_spline_build(t, u, extrapolate=False):
    """Compute spline coefficients from knots *t* and values *u*.

    Returns ``(t, u, h, z, extrapolate)`` — all NumPy arrays (or scalars).
    """
    t = np.asarray(t, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    n = len(t) - 1

    h_full = np.zeros(n + 2, dtype=np.float64)
    h_full[1:-1] = np.diff(t)
    # h_full[0] = h_full[-1] = 0 already

    dl = h_full[1:n + 1].copy()
    du = h_full[1:n + 1].copy()
    d_diag = 2.0 * (h_full[:n + 1] + h_full[1:n + 2])

    rhs = np.zeros(n + 1, dtype=np.float64)
    for i in range(1, n):
        rhs[i] = (6.0 * (u[i + 1] - u[i]) / h_full[i + 1]
                  - 6.0 * (u[i] - u[i - 1]) / h_full[i])

    # Solve tridiagonal system
    from scipy.linalg import solve_banded
    ab = np.zeros((3, n + 1), dtype=np.float64)
    ab[0, 1:] = du          # super-diagonal
    ab[1, :] = d_diag       # diagonal
    ab[2, :-1] = dl         # sub-diagonal
    z = solve_banded((1, 1), ab, rhs)

    h = h_full[:n + 1].copy()  # length n+1, h[0]=0

    return t, u, h, z, extrapolate


# ── Eval (JAX, jit-compatible) ───────────────────────────────────────────────

@jax.jit
def cubic_spline_eval(t, u, h, z, x):
    """Evaluate the cubic spline at point *x*.

    Uses ``jnp.where`` chains so it is branchless and jit-safe.
    Flat extrapolation beyond the knot range.
    """
    n = len(t) - 1
    # Clamp x to the valid range for flat extrapolation
    x_clamped = jnp.clip(x, t[0], t[-1])

    i = jnp.searchsorted(t, x_clamped, side="right") - 1
    i = jnp.clip(i, 0, n - 1)

    h_ip1 = t[i + 1] - t[i]
    dt_right = t[i + 1] - x_clamped
    dt_left = x_clamped - t[i]

    I = z[i] * dt_right ** 3 / (6.0 * h_ip1) + z[i + 1] * dt_left ** 3 / (6.0 * h_ip1)
    C = (u[i + 1] / h_ip1 - z[i + 1] * h_ip1 / 6.0) * dt_left
    D = (u[i] / h_ip1 - z[i] * h_ip1 / 6.0) * dt_right

    val = I + C + D

    # Flat extrapolation
    val = jnp.where(x < t[0], u[0], val)
    val = jnp.where(x > t[-1], u[-1], val)
    return val


def batch_eval(t, u, h, z, xs):
    """Evaluate the spline at a batch of points via vmap."""
    return jax.vmap(lambda x: cubic_spline_eval(t, u, h, z, x))(xs)


# ── Convenience wrapper ──────────────────────────────────────────────────────

class CubicSpline:
    """Convenience class wrapping build + eval."""

    def __init__(self, t, u, extrapolate=False):
        t_np, u_np, h_np, z_np, self._extrapolate = cubic_spline_build(
            t, u, extrapolate=extrapolate
        )
        # Keep NumPy arrays for fast scalar evaluation
        self._t_np = t_np
        self._u_np = u_np
        self._z_np = z_np
        # JAX arrays for JIT-compatible evaluation
        self.t = jnp.array(t_np)
        self.u = jnp.array(u_np)
        self.h = jnp.array(h_np)
        self.z = jnp.array(z_np)

    def __call__(self, x):
        if isinstance(x, (int, float)):
            return self._eval_scalar(x)
        return cubic_spline_eval(self.t, self.u, self.h, self.z, x)

    def _eval_scalar(self, x):
        """Fast scalar evaluation using pure Python/NumPy (no JAX dispatch)."""
        t = self._t_np
        u = self._u_np
        z = self._z_np
        n = len(t) - 1
        if x <= t[0]:
            return float(u[0])
        if x >= t[-1]:
            return float(u[-1])
        i = int(np.searchsorted(t, x, side='right')) - 1
        if i < 0:
            i = 0
        elif i > n - 1:
            i = n - 1
        h_ip1 = t[i + 1] - t[i]
        dr = t[i + 1] - x
        dl = x - t[i]
        val = (z[i] * dr ** 3 / (6.0 * h_ip1)
               + z[i + 1] * dl ** 3 / (6.0 * h_ip1)
               + (u[i + 1] / h_ip1 - z[i + 1] * h_ip1 / 6.0) * dl
               + (u[i] / h_ip1 - z[i] * h_ip1 / 6.0) * dr)
        return float(val)

    def batch(self, xs):
        return batch_eval(self.t, self.u, self.h, self.z, xs)
