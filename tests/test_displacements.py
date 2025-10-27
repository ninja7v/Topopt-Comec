# tests/test_displacements.py
# MIT License - Copyright (c) 2025 Luc Prevost
# Tests for the displacements.

import numpy as np
from app.core import displacements
import json
from pathlib import Path
import pytest

# Helper function to load the presets file for the test
def load_presets():
    """Finds and loads the presets.json file."""
    # Go up two directories from this test file to find the project root
    presets_path = Path(__file__).parent.parent / "presets.json"
    with open(presets_path, 'r') as f:
        presets_data = json.load(f)
    
    # Return the presets as a list of tuples for pytest
    return presets_data.items()

@pytest.mark.parametrize("preset_name, preset_params", load_presets())
def test_displacement_with_presets(preset_name, preset_params):
    """Unit Test: Runs the 2D/3D optimizer with a given preset."""
    # Prepare the parameters for the optimizer function
    disp_params = preset_params.copy()
    nelx, nely, nelz = disp_params['nelxyz']
    is_3d = nelz > 0
    # To run the tests faster, we reduce the number of iterations
    disp_params['disp_iterations'] = 1
    # Remove all keys that are not part of the optimizer's function signature
    keys_to_remove = ['filter_type', 'filter_radius_min', 'max_change', 'n_it']
    if not is_3d: keys_to_remove = keys_to_remove + ['rz', 'fz', 'sz']
    for key in keys_to_remove:
        disp_params.pop(key, None)
    
    # Generate a mock result and displacement vector
    nel = nelx * nely * (disp_params['nelxyz'][2] if is_3d else 1)
    ndof = (3 if is_3d else 2) * (nelx + 1) * (nely + 1) * ((nelz + 1) if is_3d else 1)
    p = 1/preset_params['volfrac'] - 1 # f(x) = (x/volfrac)^p -> integral(f(x)) from 0 to nel = volfrac * nel
    x = np.linspace(0, 1, nel)
    densities = x**p
    np.random.shuffle(densities)
    result = densities
    u_vec = np.random.rand(ndof, sum(1 for fdir in disp_params['fidir'] if fdir != "-"))
    
    # Check if not empty
    assert result is not None, "Optimizer returned None"
    assert u_vec is not None, "Displacement vector is None"
    
    # Test linear displacement function
    if is_3d:
        X, Y, Z = displacements.single_linear_displacement(u_vec, nelx, nely, nelz, 1.0)
        assert not(X is None or Y is None or Z is None), "Displacement function returned None arrays"
    else:
        X, Y = displacements.single_linear_displacement(u_vec, nelx, nely, nelz, 1.0)
        assert not(X is None or Y is None), "Displacement function returned None arrays"
    
    # Test iterative displacement function
    for frame in displacements.run_iterative_displacement(disp_params, result):
        last_result_displaced = frame
    assert last_result_displaced is not None, "Iterative displacement function returned None"
    assert last_result_displaced.shape == np.array(result).shape, "Iterative displacement function returned different shapes"
    vals = last_result_displaced
    assert np.max(vals) <= 1.0 and np.min(vals) >= 0.0, "Displaced densities should remain within [0, 1]"

    # Check volume fraction
    assert np.isclose(result.mean(), preset_params['volfrac'], atol=0.05), \
        f"Final volume ({result.mean():.3f}) is far to target ({preset_params['volfrac']:.3f})"