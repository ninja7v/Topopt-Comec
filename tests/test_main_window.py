# tests/test_main_window.py
# MIT License - Copyright (c) 2025 Luc Prevost
# Tests for the main window.

import pytest
from PySide6.QtWidgets import QApplication
from app.ui.main_window import MainWindow

# --- Test Cases for the Intelligent Comparison ---
# A base 2D preset
p_base_2d = {"nelxyz": [60, 40, 0],
             "dim": ["Y", "X"], "sx": [0, 60], "sy": [20, 20],
             "fx": [0, 100], "fy": [30, 40], "fz": [0, 0], "fdir": ["X:\u2192", "Y:\u2193"], "fnorm": [0.01, 0.01]}
# A 2D preset with an inactive support
p_inactive_support = {"nelxyz": [60, 40, 0],
                      "dim": ["Y", "X", "-"], "sx": [0, 60, 0], "sy": [20, 20, 0],
                      "fx": [0, 100], "fy": [30, 40], "fz": [0, 0], "fdir": ["X:\u2192", "Y:\u2193"], "fnorm": [0.01, 0.01]}
# A 2D preset with an inactive force
p_inactive_force = {"nelxyz": [60, 40, 0],
                    "dim": ["Y", "X", "-"], "sx": [0, 60], "sy": [20, 20],
                    "fx": [0, 100, 0], "fy": [30, 40, 0], "fz": [0, 0, 0], "fdir": ["X:\u2192", "Y:\u2193", "-"], "fnorm": [0.01, 0.01, 0.0]}
# A 2D preset with an inactive void
p_inactive_void = {"nelxyz": [60, 40, 0],
                    "dim": ["Y", "X", "-"], "sx": [0, 60], "sy": [20, 20],
                    "fx": [0, 100], "fy": [30, 40], "fz": [0, 0], "fdir": ["X:\u2192", "Y:\u2193"], "fnorm": [0.01, 0.01],
                    "vshape": ["-"], "vradius": [5], "vx": [30], "vy": [20], "vz": [0]}
# A preset that is truly different
p_different = {"nelxyz": [80, 50, 0],
               "dim": ["X", "Y"], "sx": [0, 60], "sy": [20, 20],
               "fx": [0, 100, 0], "fy": [30, 40, 0], "fz": [0, 0, 0], "fdir": ["X:\u2192", "Y:\u2193", "-"], "fnorm": [0.01, 0.01, 0.0],
               "vshape": ["□"], "vradius": [7], "vx": [20], "vy": [30], "vy": [0]}

@pytest.mark.parametrize("p1, p2, expected", [
    (p_base_2d, p_base_2d, True),          # Should equal itself
    (p_base_2d, p_inactive_support, True), # Should be equivalent despite extra inactive support
    (p_base_2d, p_inactive_force, True),   # Should be equivalent despite extra inactive force
    (p_base_2d, p_inactive_void, True),    # Should be equivalent despite extra inactive void
    (p_base_2d, p_different, False),       # Should be different
])
def test_are_parameters_equivalent(qt_app, p1, p2, expected):
    """Unit Test: Tests the intelligent parameter comparison function."""
    # We need a MainWindow instance to get access to the method
    window = MainWindow()
    assert window.are_parameters_equivalent(p1, p2) == expected

def test_gather_and_apply_parameters(qt_app):
    """
    Integration Test: Checks if gathering and applying parameters works correctly.
    """
    window = MainWindow()
    
    # 1. Get the initial parameters from the UI
    initial_params = window.gather_parameters()
    
    # 2. Modify a known value in the dictionary
    modified_params = initial_params.copy()
    modified_params["nelxyz"] = [100, 80, 10]
    modified_params["sx"][0] = 50

    # 3. Apply these modified parameters back to the UI
    window.apply_parameters(modified_params)
    
    # 4. Gather the parameters from the UI again
    new_params_from_ui = window.gather_parameters()
    
    # 5. Assert that the UI state now matches the modified parameters
    assert new_params_from_ui["nelxyz"] == [100, 80, 10]
    assert new_params_from_ui["sx"][0] == 50