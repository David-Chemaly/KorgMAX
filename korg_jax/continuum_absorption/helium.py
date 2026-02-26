"""He- free-free absorption from John (1994).

Ported from Korg.jl/src/ContinuumAbsorption/absorption_He.jl.
"""
import numpy as np
import jax.numpy as jnp
from ..constants import c_cgs, kboltz_cgs, kboltz_eV


# John (1994) Table — OCR'd from the paper
_theta_nodes = np.array([0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.8, 3.6])
_lambda_nodes = np.array([
    5063., 5695., 6509., 7594., 9113., 11391., 15188.,
    18225., 22782., 30376., 36451., 45564., 60751., 91127., 113900., 151878.,
])

_ff_table = np.array([
    [0.033, 0.036, 0.043, 0.049, 0.055, 0.061, 0.066, 0.072, 0.078, 0.100, 0.121],
    [0.041, 0.045, 0.053, 0.061, 0.067, 0.074, 0.081, 0.087, 0.094, 0.120, 0.145],
    [0.053, 0.059, 0.069, 0.077, 0.086, 0.094, 0.102, 0.109, 0.117, 0.148, 0.178],
    [0.072, 0.079, 0.092, 0.103, 0.114, 0.124, 0.133, 0.143, 0.152, 0.190, 0.227],
    [0.102, 0.113, 0.131, 0.147, 0.160, 0.173, 0.186, 0.198, 0.210, 0.258, 0.305],
    [0.159, 0.176, 0.204, 0.227, 0.247, 0.266, 0.283, 0.300, 0.316, 0.380, 0.444],
    [0.282, 0.311, 0.360, 0.400, 0.435, 0.466, 0.495, 0.522, 0.547, 0.643, 0.737],
    [0.405, 0.447, 0.518, 0.576, 0.625, 0.670, 0.710, 0.747, 0.782, 0.910, 1.030],
    [0.632, 0.698, 0.808, 0.899, 0.977, 1.045, 1.108, 1.165, 1.218, 1.405, 1.574],
    [1.121, 1.239, 1.435, 1.597, 1.737, 1.860, 1.971, 2.073, 2.167, 2.490, 2.765],
    [1.614, 1.783, 2.065, 2.299, 2.502, 2.681, 2.842, 2.990, 3.126, 3.592, 3.979],
    [2.520, 2.784, 3.226, 3.593, 3.910, 4.193, 4.448, 4.681, 4.897, 5.632, 6.234],
    [4.479, 4.947, 5.733, 6.387, 6.955, 7.460, 7.918, 8.338, 8.728, 10.059, 11.147],
    [10.074, 11.128, 12.897, 14.372, 15.653, 16.798, 17.838, 18.795, 19.685, 22.747, 25.268],
    [15.739, 17.386, 20.151, 22.456, 24.461, 26.252, 27.882, 29.384, 30.782, 35.606, 39.598],
    [27.979, 30.907, 35.822, 39.921, 43.488, 46.678, 49.583, 52.262, 54.757, 63.395, 70.580],
], dtype=np.float64)  # shape (16, 11)


def _interp_ff(lam_A, theta):
    """Bilinear interpolation in the John (1994) table. NumPy."""
    lam_A = np.clip(lam_A, _lambda_nodes[0], _lambda_nodes[-1])
    theta = np.clip(theta, _theta_nodes[0], _theta_nodes[-1])

    i = np.searchsorted(_lambda_nodes, lam_A, side="right") - 1
    i = np.clip(i, 0, len(_lambda_nodes) - 2)
    j = np.searchsorted(_theta_nodes, theta, side="right") - 1
    j = np.clip(j, 0, len(_theta_nodes) - 2)

    di = (lam_A - _lambda_nodes[i]) / (_lambda_nodes[i + 1] - _lambda_nodes[i])
    dj = (theta - _theta_nodes[j]) / (_theta_nodes[j + 1] - _theta_nodes[j])

    return (_ff_table[i, j] * (1 - di) * (1 - dj)
            + _ff_table[i + 1, j] * di * (1 - dj)
            + _ff_table[i, j + 1] * (1 - di) * dj
            + _ff_table[i + 1, j + 1] * di * dj)


def Heminus_ff(nus, T, nHe_I_div_U, ne):
    """He- free-free absorption coefficient.

    Parameters
    ----------
    nus : (n,) frequency array (Hz)
    T : temperature (K)
    nHe_I_div_U : n(He I) / U(He I)
    ne : electron number density
    """
    nus = jnp.asarray(nus)
    theta = 5040.0 / T
    lambdas_A = c_cgs * 1e8 / nus  # Angstrom

    # Bounds check: only valid for certain lambda and theta
    lam_min, lam_max = _lambda_nodes[0], _lambda_nodes[-1]
    th_min, th_max = _theta_nodes[0], _theta_nodes[-1]
    in_bounds = ((lambdas_A >= lam_min) & (lambdas_A <= lam_max)
                 & (theta >= th_min) & (theta <= th_max))

    # Interpolate using NumPy, then convert
    lam_np = np.asarray(lambdas_A)
    K_vals = np.zeros_like(lam_np)
    for k in range(len(lam_np)):
        if lam_min <= lam_np[k] <= lam_max and th_min <= theta <= th_max:
            K_vals[k] = _interp_ff(lam_np[k], theta)

    K = jnp.array(K_vals) * 1e-26  # cm^4/dyn

    Pe = ne * kboltz_cgs * T
    # Ground state of He I: g=1, E=0
    nHe_I_gs = nHe_I_div_U * 1.0  # g_1 * exp(-0/kT) = 1

    return K * nHe_I_gs * Pe
