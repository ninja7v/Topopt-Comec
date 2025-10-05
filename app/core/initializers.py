# app/core/initializers.py
# MIT License - Copyright (c) 2025 Luc Prevost
# Material initializators.

import numpy as np
from typing import List

def rescale_densities(d: np.ndarray, volfrac: float) -> np.ndarray:
    """
    Smoothly rescale densities so their mean equals volfrac, 
    while keeping values in [0,1] and reducing movement near 0/1.
    """
    tol = 1e-4
    current = np.mean(d)
    if abs(current - volfrac) < tol:
        return np.clip(d, 0, 1)

    # Binary search for scaling parameter alpha
    lo, hi = -2.0, 2.0
    for _ in range(50):
        alpha = 0.5 * (lo + hi)
        d_new = d + (-4 * d**2 + 4 * d) * alpha # current + 2nd degree polynomial: p(0) = 0, p(1) = 0, p(0.5) = alpha
        mean_new = np.mean(d_new)
        if mean_new < volfrac:
            lo = alpha
        else:
            hi = alpha
        if abs(mean_new - volfrac) < tol:
            break

    return np.clip(d_new, 0, 1)

def initialize_material(init_type: int, volfrac: float, nelx: int, nely: int, nelz: int, all_x: List[int], all_y: List[int], all_z: List[int]) -> np.ndarray:
    """Initialize the material distribution based on the selected type."""
    is_3d = nelz > 0
    nel = nelx * nely * (nelz if is_3d else 1)

    if init_type == 0:  # Uniform
        return np.full(nel, volfrac)

    elif init_type == 1:  # Around activity points
        points = np.stack([all_x, all_y, all_z] if is_3d else [all_x, all_y], axis=1)
        if is_3d:
            zz, xx, yy = np.meshgrid(np.arange(nelz),
                                    np.arange(nelx),
                                    np.arange(nely),
                                    indexing="ij")
        else:
            xx, yy = np.meshgrid(np.arange(nelx),
                                 np.arange(nely),
                                 indexing="ij")
        coords = np.stack([xx.ravel(), yy.ravel(), zz.ravel()] if is_3d else [xx.ravel(), yy.ravel()], axis=1)
        diff = coords[:, None, :] - points[None, :, :]  # (nel, n_points, 2-3)
        dist = np.sqrt(np.sum(diff**2, axis=2))         # (nel, n_points)
        distance_max = np.sqrt(nelx**2 + nely**2 + (nelz**2 if is_3d else 0))
        d = (distance_max - np.min(dist, axis=1)) / distance_max
        d = rescale_densities(d, volfrac)
        return d

    elif init_type == 2:  # Random
        np.random.seed(42)
        d = np.random.rand(nel)
        d = rescale_densities(d, volfrac)
        return d

    else:
        raise ValueError("Invalid initialization type. Must be 0 (Uniform), 1 (Around activity points), or 2 (Random).")