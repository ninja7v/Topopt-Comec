# app/core/optimizers.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Topology Optimizers.

from typing import Dict, Callable, Optional, Tuple
import numpy as np

from app.core import initializers
from app.core.fem import FEM


def oc(
    nel: int,
    x: np.ndarray,
    eta: float,
    max_change: float,
    dc: np.ndarray,
    dv: np.ndarray,
    g: float,
) -> Tuple[np.ndarray, float]:
    """
    Optimality Criterion (OC) update scheme.

    Args:
        nel: Total number of elements.
        x: Current design variables (densities).
        max_change: Maximum allowed change in design variables per iteration.
        dc: Sensitivities of the objective function.
        dv: Sensitivities of the volume constraint.
        g: Lagrangian multiplier for the volume constraint.

    Returns:
        A tuple containing the new design variables (xnew) and the updated gt value.
    """
    l1, l2 = 0.0, 1e9
    rhomin = 1e-6
    xnew = np.zeros(nel)

    while (l2 - l1) / (l1 + l2) > 1e-4 and l2 > 1e-40:
        lmid = 0.5 * (l2 + l1)
        # Bisection method to find the Lagrange multiplier
        # This is the OC update rule with move limits
        x_update = x * np.maximum(0.1, -dc / dv / lmid) ** eta
        xnew[:] = np.maximum(
            rhomin,
            np.maximum(
                x - max_change, np.minimum(1.0, np.minimum(x + max_change, x_update))
            ),
        )

        gt = g + np.sum(
            dv * (xnew - x)
        )  # Should be near zero for the volume constraint
        if gt > 0:
            l1 = lmid
        else:
            l2 = lmid
    return xnew, gt


def optimize(
    Dimensions: Dict,
    Forces: Dict,
    Materials: Dict,
    Optimizer: Dict,
    Supports: Optional[Dict] = None,
    Regions: Optional[Dict] = None,
    progress_callback: Optional[Callable] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Topology optimization

    Args:
        all parameters splited per section.
        progress_callback: A function to call with (iteration, objective, change) for UI updates.

    Returns:
        xPhys: Final physical densities after optimization.
        ui: Associated displacement vector.
    """
    print("Optimizer starting...")
    Supports, Regions = Supports or {}, Regions or {}

    # Initialize FEM Environment
    fem = FEM(Dimensions, Materials, Optimizer)
    fem.setup_boundary_conditions(Forces, Supports)

    # Initialize Material
    x = initializers.initialize_material(
        Materials.get("init_type", 0),
        Dimensions.get("volfrac", 0.5),
        fem.nelx,
        fem.nely,
        fem.nelz,
        # Helper to get active coordinate arrays for initialization
        *_get_active_coords(Supports, Forces, fem.is_3d),
    )
    xPhys = x.copy()
    g = 0.0

    # Optimization Params
    eta, max_change = Optimizer.get("eta", 1.0), Optimizer.get("max_change", 0.1)
    n_it = Optimizer.get("n_it", 30)

    print("   Preparation done -> Optimization loop starting...")
    loop, change = 0, 1.0
    while change > 0.01 and loop < n_it:
        loop += 1
        xold = x.copy()

        # Apply Constraints & Analyze
        xPhys = fem.apply_regions(xPhys, Regions)
        ui, uo = fem.solve(xPhys)

        # Compute Sensitivities & Filter
        obj_val, (dc, dv) = fem.compute_sensitivities(xPhys, ui, uo)

        # Update Design Variables
        x, g = oc(fem.nel, x, eta, max_change, dc, dv, g)
        xPhys = fem.update_xPhys(x)

        # Check Convergence
        change = np.linalg.norm(x - xold, np.inf)
        print(
            f"It.: {loop:3d}, Obj.: {obj_val:.4f}, Vol.: {xPhys.mean():.3f}, Ch.: {change:.3f}"
        )

        if progress_callback and progress_callback(loop, obj_val, change, xPhys):
            print("Optimization stopped by user.")
            break

    print("Optimizer finished.")
    return xPhys, ui


def _get_active_coords(Supports, Forces, is_3d):
    """Helper to extract active coordinates for material initialization."""

    # This extracts the logic previously doing np.concatenate on active indices
    def get_act(d, k_dim, k_flag):
        return np.array(d.get(k_dim, []))[
            [i for i, v in enumerate(d.get(k_flag, [])) if v != "-"]
        ]

    sx, sy = get_act(Supports, "sx", "sdim"), get_act(Supports, "sy", "sdim")
    fix, fiy = get_act(Forces, "fix", "fidir"), get_act(Forces, "fiy", "fidir")
    fox, foy = get_act(Forces, "fox", "fodir"), get_act(Forces, "foy", "fodir")

    all_x = np.concatenate([fix, fox, sx])
    all_y = np.concatenate([fiy, foy, sy])

    if is_3d:
        sz = get_act(Supports, "sz", "sdim")
        fiz, foz = get_act(Forces, "fiz", "fidir"), get_act(Forces, "foz", "fodir")
        return all_x, all_y, np.concatenate([fiz, foz, sz])
    return all_x, all_y, np.array([0] * len(all_x))
