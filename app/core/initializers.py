# app/core/initializers.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Material initializators.

import numpy as np
from scipy.spatial.distance import cdist


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
        d_new = (
            d + (-4 * d**2 + 4 * d) * alpha
        )  # current + 2nd degree polynomial: p(0) = 0, p(1) = 0, p(0.5) = alpha
        mean_new = np.mean(d_new)
        if mean_new < volfrac:
            lo = alpha
        else:
            hi = alpha
        if abs(mean_new - volfrac) < tol:
            break

    return np.clip(d_new, 0, 1)


def initialize_material(
    init_type: int,
    volfrac: float,
    nelx: int,
    nely: int,
    nelz: int,
    all_x: np.ndarray,
    all_y: np.ndarray,
    all_z: np.ndarray,
) -> np.ndarray:
    """Initialize the material distribution based on the selected type."""
    is_3d = nelz > 0
    nel = nelx * nely * (nelz if is_3d else 1)

    # 0. Uniform Distribution
    if init_type == 0:
        return np.full(nel, volfrac)

    # 1. Distance Field (Seeded at active points)
    elif init_type == 1:
        points = np.column_stack([all_x, all_y, all_z] if is_3d else [all_x, all_y])
        if len(points) == 0:
            return np.full(nel, volfrac)

        # Generate element center coordinates matching FEM loop order:
        # Loop order is: for ez... for ex... for ey...
        if is_3d:
            Z = np.repeat(np.arange(nelz), nelx * nely)
            X = np.tile(np.repeat(np.arange(nelx), nely), nelz)
            Y = np.tile(np.arange(nely), nelx * nelz)
            coords = np.column_stack((X, Y, Z))
        else:
            X = np.repeat(np.arange(nelx), nely)
            Y = np.tile(np.arange(nely), nelx)
            coords = np.column_stack((X, Y))

        # Vectorized distance calculation
        dists = cdist(coords, points, metric="euclidean")  # Shape: (nel, n_points)
        min_dist = dists.min(axis=1)

        # Invert distance: Near = 1.0, Far = 0.0
        distance_max = np.sqrt(nelx**2 + nely**2 + (nelz**2 if is_3d else 0))
        raw = (distance_max - min_dist) / distance_max

        return rescale_densities(raw, volfrac)

    # 2. Random Distribution
    elif init_type == 2:
        np.random.seed(42)
        raw = np.random.rand(nel)
        return rescale_densities(raw, volfrac)

    else:
        raise ValueError(f"Invalid init_type: {init_type}")


def initialize_materials(
    init_type: int,
    materials_percentage: list,
    volfrac: float,
    nelx: int,
    nely: int,
    nelz: int,
    all_x: np.ndarray,
    all_y: np.ndarray,
    all_z: np.ndarray,
) -> np.ndarray:
    """Initialize multi-material density fields.

    Args:
        init_type: Initialization strategy (0=Uniform, 1=Distance, 2=Random).
        materials_percentage: List of percentage of each material (sum to 100).
        volfrac: Total target volume fractions.
        nelx, nely, nelz: Grid dimensions.
        all_x, all_y, all_z: Active coordinate arrays for distance-based init.

    Returns:
        Array of shape (n_mat, nel) with per-material densities.
        Columns sum to 1 (partition of unity) and each row's mean
        approximates the corresponding volume fraction.
    """
    if sum(materials_percentage) != 100:
        return

    n_mat = len(materials_percentage)
    materials_frac = volfrac * np.array(materials_percentage) / 100
    nel = nelx * nely * (nelz if nelz > 0 else 1)

    # Start from the single-material spatial pattern for material 0
    base = initialize_material(
        init_type, materials_frac[0], nelx, nely, nelz, all_x, all_y, all_z
    )

    rho = np.zeros((n_mat, nel))
    rho[0] = base

    # Material 1 gets the complement
    if n_mat > 1:
        rho[1] = rescale_densities(volfrac - base, materials_frac[1])

    # Normalize columns so sum = volfrac (partition of unity)
    col_sums = rho.sum(axis=0)
    col_sums[col_sums == 0] = volfrac  # avoid division by zero
    rho *= volfrac / col_sums

    # Re-scale rows to hit target volume fractions
    for i in range(n_mat):
        rho[i] = rescale_densities(rho[i], materials_frac[i])

    # Final normalization pass
    col_sums = rho.sum(axis=0)
    col_sums[col_sums == 0] = volfrac
    rho *= volfrac / col_sums

    return np.clip(rho, 1e-6, 1.0)
