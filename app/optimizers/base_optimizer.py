# app/optimizers/base_optimizer.py
# MIT License - Copyright (c) 2025 Luc Prevost
# Contains shared functions for the topology optimizers.

import numpy as np
from typing import Tuple

def oc(nel: int, x: np.ndarray, volfrac: float, dc: np.ndarray, dv: np.ndarray, g: float) -> Tuple[np.ndarray, float]:
    """
    Optimality Criterion (OC) update scheme.
    
    Args:
        nel: Total number of elements.
        x: Current design variables (densities).
        volfrac: Target volume fraction.
        dc: Sensitivities of the objective function.
        dv: Sensitivities of the volume constraint.
        g: Lagrangian multiplier for the volume constraint.

    Returns:
        A tuple containing the new design variables (xnew) and the updated gt value.
    """
    l1, l2 = 0., 1e9
    move = 0.05
    rhomin = 1e-6
    xnew = np.zeros(nel)
    
    while (l2 - l1) / (l1 + l2) > 1e-4 and l2 > 1e-40:
        lmid = 0.5 * (l2 + l1)
        # Bisection method to find the Lagrange multiplier
        # This is the OC update rule with move limits
        x_update = x * np.maximum(1e-10, -dc / dv / lmid) ** 0.3
        xnew[:] = np.maximum(rhomin, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x_update))))
        
        gt = g + np.sum(dv * (xnew - x))
        
        if gt > 0:
            l1 = lmid
        else:
            l2 = lmid
    
    return xnew, gt