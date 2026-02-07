# tests/test_optimizers.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Tests for the optimizers.

import json
from pathlib import Path

import numpy as np
import pytest

from app.core import optimizers


def test_lk_properties():
    """Unit Test: Checks the 2D element stiffness matrix (lk) function."""
    # compute 2D stiffness matrix
    KE_2D = optimizers.lk(E=1.0, nu=0.3, is_3d=False)
    assert isinstance(KE_2D, np.ndarray), "lk should return a NumPy array"
    assert KE_2D.shape == (8, 8), "The 2D stiffness matrix must be 8x8"
    assert np.allclose(KE_2D, KE_2D.T), "The stiffness matrix must be symmetric"

    # compute 3D stiffness matrix
    KE_3D = optimizers.lk(E=1.0, nu=0.3, is_3d=True)
    assert isinstance(KE_3D, np.ndarray), "lk should return a NumPy array"
    assert KE_3D.shape == (24, 24), "The 3D stiffness matrix must be 24x24"
    assert np.allclose(KE_3D, KE_3D.T), "The stiffness matrix must be symmetric"


def test_oc_update_rule():
    """Unit Test: Checks the Optimality Criteria function."""
    nel = 4
    x = np.array([0.5, 0.5, 0.5, 0.5])
    eta = 0.3
    max_change = 0.05
    dc = np.array([-1.0, -0.5, 0.0, 0.5])
    dv = np.array([1.0, 1.0, 1.0, 1.0])
    g = 0.0

    # Run OC update
    xnew, gt = optimizers.oc(nel, x, eta, max_change, dc, dv, g)

    # Check shape
    assert isinstance(xnew, np.ndarray), "oc should return a NumPy array"
    assert xnew.shape == (nel,), "Returned array should have shape (nel,)"

    # Check bounds
    assert np.all(xnew >= 1e-6), "All values should be >= 1e-6"
    assert np.all(xnew <= 1.0), "All values should be <= 1.0"

    # Check different
    assert not np.allclose(
        xnew, x
    ), "Updated values should differ from original if sensitivities are nonzero"

    # gt should be close to zero (KKT condition)
    assert abs(gt) < 1e-3, "KKT condition should be satisfied (gt close to zero)"


# Helper function to load the presets file for the test
def load_presets():
    """Finds and loads the presets.json file."""
    # Go up two directories from this test file to find the project root
    presets_path = Path(__file__).parent / "presets_test.json"
    with open(presets_path, "r") as f:
        presets_data = json.load(f)

    # Return the presets as a list of tuples for pytest
    return presets_data.items()


@pytest.mark.parametrize("preset_name, preset_params", load_presets())
def test_optimizers_with_presets(preset_name, preset_params):
    """Unit Test: Runs the optimizer with a given preset."""
    is_3d = preset_params["nelxyz"][2] > 0

    # Prepare the parameters for the optimizer function
    optimizer_params = preset_params.copy()
    # Remove all keys that are not part of the optimizer's function signature
    keys_to_remove = ["disp_factor", "disp_iterations", "percent", "color"]
    for key in keys_to_remove:
        optimizer_params.pop(key, None)  # Use .pop() to safely remove

    # Run the entire optimization
    result, u_vec = optimizers.optimize(**optimizer_params)

    # Check if not empty
    assert result is not None, "Optimizer returned None"
    assert u_vec is not None, "Displacement vector is None"

    # Check shape
    nel = (
        optimizer_params["nelxyz"][0]
        * optimizer_params["nelxyz"][1]
        * (optimizer_params["nelxyz"][2] if is_3d else 1)
    )
    assert result.shape == (nel,), f"Result shape is wrong, expected ({nel},)"
    ndof = (
        (3 if is_3d else 2)
        * (optimizer_params["nelxyz"][0] + 1)
        * (optimizer_params["nelxyz"][1] + 1)
        * ((optimizer_params["nelxyz"][2] + 1) if is_3d else 1)
    )
    assert u_vec.size == ndof * sum(
        1 for x in optimizer_params["fidir"] if x != "-"
    ), "Displacement vector should be (ndof x nb_active_iforces)"

    # Check displacement direction at the input forces
    j = 0
    active_iforces_indices = [
        i
        for i in range(len(optimizer_params["fidir"]))
        if optimizer_params["fidir"][i] != "-"
    ]
    for i in active_iforces_indices:
        idx = (
            (
                optimizer_params["fiz"][i]
                * optimizer_params["nelxyz"][0]
                * optimizer_params["nelxyz"][1]
                if is_3d
                else 0
            )
            + optimizer_params["fix"][i] * optimizer_params["nelxyz"][1]
            + optimizer_params["fiy"][i]
        )
        idx = idx * (3 if is_3d else 2) + (
            0
            if optimizer_params["fidir"][i] == "X:\u2192"
            or optimizer_params["fidir"][i] == "X:\u2190"
            else (
                1
                if optimizer_params["fidir"][i] == "Y:\u2193"
                or optimizer_params["fidir"][i] == "Y:\u2191"
                else 2
            )
        )
        direction_sign = (
            1
            if optimizer_params["fidir"][i] == "X:\u2192"
            or optimizer_params["fidir"][i] == "Y:\u2193"
            or optimizer_params["fidir"][i] == "Z:<"
            else -1
        )
        assert u_vec[idx, j] * direction_sign > 0
        j += 1

    # Check volume fraction
    volfrac = preset_params["volfrac"]
    assert np.isclose(
        result.mean(), volfrac, atol=0.05
    ), f"Final volume ({result.mean():.3f}) is far to target ({volfrac:.3f})"
