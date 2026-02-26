"""Lazy multilinear interpolation for MARCS grids.

Ported from Korg.jl/src/lazy_multilinear_interpolation.jl.
"""
from __future__ import annotations

import numpy as np


class LazyMultilinearInterpError(Exception):
    pass


def lazy_multilinear_interpolation(params, nodes, grid, param_names=None,
                                   perturb_at_grid_values=False):
    """Multilinear interpolation on a grid whose first two dims are values.

    params: list of parameter values
    nodes: list of 1D node arrays (same length as params)
    grid: ndarray with shape (n_vals1, n_vals2, len(node1), len(node2), ...)
    """
    params = np.array(params, dtype=float)
    nodes = [np.asarray(n, dtype=float) for n in nodes]

    if param_names is None:
        param_names = [f"param {i+1}" for i in range(len(params))]

    if perturb_at_grid_values:
        on_grid = np.array([p in n for p, n in zip(params, nodes)])
        params[on_grid] = np.nextafter(params[on_grid], np.inf)
        too_high = np.array([p > n[-1] for p, n in zip(params, nodes)])
        params[too_high] = np.nextafter(params[too_high], -np.inf)
        params[too_high] = np.nextafter(params[too_high], -np.inf)

    upper_vertex = []
    for p, pname, p_nodes in zip(params, param_names, nodes):
        if not (p_nodes[0] <= p <= p_nodes[-1]):
            msg = (f"Can't interpolate grid. {pname} is out of bounds. "
                   f"({p} not in [{p_nodes[0]}, {p_nodes[-1]}]).")
            raise LazyMultilinearInterpError(msg)
        upper_vertex.append(int(np.searchsorted(p_nodes, p, side="left")))

    upper_vertex = np.array(upper_vertex, dtype=int)
    isexact = np.array([params[i] == nodes[i][upper_vertex[i]] for i in range(len(params))])

    dims = tuple([2 for _ in upper_vertex])
    structure = np.empty(grid.shape[:2] + dims, dtype=float)

    for idx in np.ndindex(dims):
        local_inds = list(idx)
        atm_inds = local_inds.copy()
        for i, exact in enumerate(isexact):
            if exact:
                atm_inds[i] = 1
        atm_inds = np.array(atm_inds) + upper_vertex - 1
        structure[(slice(None), slice(None)) + idx] = grid[(slice(None), slice(None)) + tuple(atm_inds)]

    for i in range(len(params)):
        if isexact[i]:
            continue
        p1 = nodes[i][upper_vertex[i] - 1]
        p2 = nodes[i][upper_vertex[i]]
        x = (params[i] - p1) / (p2 - p1)

        inds1 = [0] * i + [0] + [slice(None)] * (len(params) - i - 1)
        inds2 = [0] * i + [1] + [slice(None)] * (len(params) - i - 1)

        s1 = (slice(None), slice(None)) + tuple(inds1)
        s2 = (slice(None), slice(None)) + tuple(inds2)
        structure[s1] = (1 - x) * structure[s1] + x * structure[s2]

    final_idx = (slice(None), slice(None)) + tuple([0] * len(params))
    return structure[final_idx]
