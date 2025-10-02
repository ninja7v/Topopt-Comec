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

def initialize_material_2d(init_type: int, volfrac: float, nelx: int, nely: int, all_x: List[int], all_y: List[int]) -> np.ndarray:
    """Initialize the material distribution based on the selected type."""
    nel = nelx * nely

    if init_type == 0:  # Uniform
        return np.full(nel, volfrac)

    elif init_type == 1:  # Around activity points
        # Combine fixed and support points
        points = np.stack([all_x, all_y], axis=1)  # shape (n_points, 2)

        # Create full grid of element coordinates
        xx, yy = np.meshgrid(np.arange(nelx), np.arange(nely), indexing="ij")
        coords = np.stack([xx.ravel(), yy.ravel()], axis=1)  # shape (nel, 2)

        # Compute all distances (broadcasted)
        diff = coords[:, None, :] - points[None, :, :]  # shape (nel, n_points, 2)
        dist = np.sqrt(np.sum(diff**2, axis=2))         # shape (nel, n_points)

        # Get minimum distance per element
        distance_max = np.sqrt(nelx**2 + nely**2)
        d = ((distance_max - np.min(dist, axis=1)) / distance_max)**0.2

        # Rescale
        d = rescale_densities(d, volfrac)
        return d

    elif init_type == 2:  # Random
        np.random.seed(42)
        d = np.random.rand(nel)

        # Rescale
        d = rescale_densities(d, volfrac)
        return d

    else:
        raise ValueError("Invalid initialization type. Must be 0 (Uniform), 1 (Critical points), or 2 (Random).")

def initialize_material_3d(init_type: int, volfrac: float, nelx: int, nely: int, nelz: int, all_x: List[int], all_y: List[int], all_z: List[int]) -> np.ndarray:
    """Initialize the material distribution based on the selected type."""
    nel = nelx * nely * nelz

    if init_type == 0:  # Uniform
        return np.full(nel, volfrac)

    elif init_type == 1:  # Around activity points
        # Combine fixed and support points
        points = np.stack([all_x, all_y, all_z], axis=1)

        # Create full grid of element coordinates
        xx, yy, zz = np.meshgrid(np.arange(nelx),
                                 np.arange(nely),
                                 np.arange(nelz),
                                 indexing="ij")
        coords = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

        # Compute distances (broadcasted)
        diff = coords[:, None, :] - points[None, :, :]  # (nel, n_points, 3)
        dist = np.sqrt(np.sum(diff**2, axis=2))         # (nel, n_points)

        # Min distance per element
        distance_max = np.sqrt(nelx**2 + nely**2 + nelz**2)
        d = (distance_max - np.min(dist, axis=1)) / distance_max

        # Rescale
        d = rescale_densities(d, volfrac)
        return d

    elif init_type == 2:  # Random
        np.random.seed(42)
        d = np.random.rand(nel)

        # Rescale
        d = rescale_densities(d, volfrac)
        return d

    else:
        raise ValueError("Invalid initialization type. Must be 0 (Uniform), 1 (Critical points), or 2 (Random).")