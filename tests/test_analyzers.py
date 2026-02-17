# tests/test_analyzers.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Tests for the analyzers.

import json
from pathlib import Path

import numpy as np
import pytest

from app.core import analyzers


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
def test_displacement_with_presets(preset_name, preset_params):
    """Unit Test: Runs the 2D/3D optimizer with a given preset."""
    # Prepare the parameters for the optimizer function
    disp_params = preset_params.copy()
    nelx, nely, nelz = disp_params["Dimensions"]["nelxyz"]
    is_3d = nelz > 0
    # Remove all keys that are not part of the optimizer's function signature
    keys_to_remove = ["filter_type", "filter_radius_min", "max_change", "n_it"]
    if not is_3d:
        keys_to_remove = keys_to_remove + ["rz", "fz", "sz"]
    for key in keys_to_remove:
        disp_params.pop(key, None)

    # Generate a mock result and displacement vector
    nel = nelx * nely * (nelz if is_3d else 1)
    ndof = (3 if is_3d else 2) * (nelx + 1) * (nely + 1) * ((nelz + 1) if is_3d else 1)
    p = (
        1 / preset_params["Dimensions"]["volfrac"] - 1
    )  # f(x) = (x/volfrac)^p -> integral(f(x)) from 0 to nel = volfrac * nel
    x = np.linspace(0, 1, nel)
    densities = x**p
    np.random.shuffle(densities)
    result = densities
    u = np.random.rand(
        ndof, sum(1 for fdir in disp_params["Forces"]["fidir"] if fdir != "-")
    )

    # Check if not empty
    assert result is not None, "Optimizer returned None"
    assert u is not None, "Displacement vector is None"

    # Test analyze function
    contains_checkerboard, is_watertight, is_thresholded, is_efficient = (
        analyzers.analyze(
            result, u, preset_params["Dimensions"], preset_params["Forces"]
        )
    )
    assert all(
        isinstance(v, bool)
        for v in (
            contains_checkerboard,
            is_watertight,
            is_thresholded,
            is_efficient,
        )
    )
