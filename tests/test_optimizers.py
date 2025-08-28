# tests/test_optimizers.py
# MIT License - Copyright (c) 2025 Luc Prevost
# Fixture for the optimizers.

import numpy as np
from app.optimizers import optimizer_2d, optimizer_3d
from app.optimizers.base_optimizer import oc
import json
from pathlib import Path
import pytest

def test_lk_2d_properties():
    """
    Unit Test: Checks the 2D element stiffness matrix (lk) function.
    It should always return an 8x8 symmetric matrix.
    """
    # Call the function with some standard material properties
    KE = optimizer_2d.lk(E=1.0, nu=0.3)

    # Assertions: These are the actual tests. If any assert fails, the test fails.
    assert isinstance(KE, np.ndarray), "lk should return a NumPy array"
    assert KE.shape == (8, 8), "The 2D stiffness matrix must be 8x8"
    assert np.allclose(KE, KE.T), "The stiffness matrix must be symmetric"

def test_lk_3d_properties():
    """
    Unit Test: Checks the 3D element stiffness matrix (lk) function.
    It should always return a 24x24 symmetric matrix.
    """
    KE = optimizer_3d.lk(E=1.0, nu=0.3)
    assert isinstance(KE, np.ndarray), "lk should return a NumPy array"
    assert KE.shape == (24, 24), "The 3D stiffness matrix must be 24x24"
    assert np.allclose(KE, KE.T), "The stiffness matrix must be symmetric"

def test_oc_update_rule():
    """Unit Test: Checks the Optimality Criteria function."""
    nel = 4
    x = np.array([0.5, 0.5, 0.5, 0.5])
    volfrac = 0.5
    dc = np.array([-1.0, -0.5, 0.0, 0.5])
    dv = np.array([1.0, 1.0, 1.0, 1.0])
    g = 0.0

    xnew, gt = oc(nel, x, volfrac, dc, dv, g)

    # Shape check
    assert isinstance(xnew, np.ndarray), "oc should return a NumPy array"
    assert xnew.shape == (nel,), "Returned array should have shape (nel,)"

    # Bounds check
    assert np.all(xnew >= 1e-6), "All values should be >= 1e-6"
    assert np.all(xnew <= 1.0), "All values should be <= 1.0"

    # Should be different from original unless sensitivities are zero
    assert not np.allclose(xnew, x), "Updated values should differ from original if sensitivities are nonzero"

    # gt should be close to zero (KKT condition)
    assert abs(gt) < 1e-3, "KKT condition should be satisfied (gt close to zero)"

# This is a helper function to load your presets file for the test
def load_presets():
    """Finds and loads the presets.json file."""
    # Go up two directories from this test file to find the project root
    presets_path = Path(__file__).parent.parent / "presets.json"
    with open(presets_path, 'r') as f:
        presets_data = json.load(f)
    
    # Return the presets as a list of tuples for pytest
    return presets_data.items()

# Pytest will run the test below once for every preset returned by the load_presets() function.
@pytest.mark.parametrize("preset_name, preset_params", load_presets())
def test_2d_optimizer_with_presets(preset_name, preset_params):
    """
    Integration Test: Runs the 2D optimizer with a given preset.
    Checks if it completes without error and respects the volume fraction.
    """
    # We only want to test the 2D presets in this test
    is_3d = preset_params['nelxyz'][2] > 0
    if is_3d:
        pytest.skip(f"Skipping 3D preset '{preset_name}' for 2D optimizer test.")

    # Prepare the parameters for the optimizer function
    optimizer_params = preset_params.copy()
    # To run the tests faster, we reduce the number of iterations
    optimizer_params['n_it'] = 2
    
    # Remove all keys that are not part of the optimizer's function signature
    keys_to_remove = ['fz', 'sz', 'disp_factor', 'disp_iterations']
    for key in keys_to_remove:
        optimizer_params.pop(key, None) # Use .pop() to safely remove
    
    # Run the entire optimization
    result, u_vec = optimizer_2d.optimize(**optimizer_params)
    
    # --- Assertions: Did it work? ---
    assert result is not None, "Optimizer returned None"
    
    nel = optimizer_params['nelxyz'][0] * optimizer_params['nelxyz'][1]
    assert result.shape == (nel,), f"Result shape is wrong, expected ({nel},)"

    # This is a great check: Is the final volume fraction close to the target?
    # We allow a small tolerance (atol) because the OC method is an approximation.
    volfrac = preset_params['volfrac']
    assert np.isclose(result.mean(), volfrac, atol=0.05), \
        f"Final volume ({result.mean():.3f}) is not close to target ({volfrac:.3f})"

@pytest.mark.parametrize("preset_name, preset_params", load_presets())
def test_3d_optimizer_with_presets(preset_name, preset_params):
    """
    Integration Test: Runs the 2D optimizer with a given preset.
    Checks if it completes without error and respects the volume fraction.
    """
    # We only want to test the 3D presets in this test
    is_3d = preset_params['nelxyz'][2] > 0
    if not is_3d:
        pytest.skip(f"Skipping 2D preset '{preset_name}' for 3D optimizer test.")

    # Prepare the parameters for the optimizer function
    optimizer_params = preset_params.copy()
    # To run the tests faster, we reduce the number of iterations
    optimizer_params['n_it'] = 2
    
    # Remove all keys that are not part of the optimizer's function signature
    keys_to_remove = ['disp_factor', 'disp_iterations']
    for key in keys_to_remove:
        optimizer_params.pop(key, None) # Use .pop() to safely remove
    
    # Run the entire optimization
    result, u_vec = optimizer_3d.optimize(**optimizer_params)
    
    # --- Assertions: Did it work? ---
    assert result is not None, "Optimizer returned None"
    
    nel = optimizer_params['nelxyz'][0] * optimizer_params['nelxyz'][1] * optimizer_params['nelxyz'][2]
    assert result.shape == (nel,), f"Result shape is wrong, expected ({nel},)"

    # This is a great check: Is the final volume fraction close to the target?
    # We allow a small tolerance (atol) because the OC method is an approximation.
    volfrac = preset_params['volfrac']
    assert np.isclose(result.mean(), volfrac, atol=0.05), \
        f"Final volume ({result.mean():.3f}) is not close to target ({volfrac:.3f})"
