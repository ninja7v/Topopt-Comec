# tests/test_main_window.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Tests for the main window.

import pytest

from app.ui.main_window import MainWindow

from unittest.mock import patch

# --- Test Cases for the Intelligent Comparison ---
# A base 2D preset
p_base_2d = {
    "nelxyz": [60, 40, 0],
    "sdim": ["Y", "X"],
    "sx": [0, 60],
    "sy": [20, 20],
    "fix": [0, 100],
    "fiy": [30, 40],
    "fiz": [0, 0],
    "fidir": ["X:\u2192", "Y:\u2193"],
    "finorm": [0.01, 0.01],
    "fox": [20],
    "foy": [20],
    "foz": [0],
    "fodir": ["X:\u2192"],
    "fonorm": [0.01],
}
# A 2D preset with an inactive support
p_inactive_support = {
    "nelxyz": [60, 40, 0],
    "sdim": ["Y", "X", "-"],
    "sx": [0, 60, 0],
    "sy": [20, 20, 0],
    "fix": [0, 100],
    "fiy": [30, 40],
    "fiz": [0, 0],
    "fidir": ["X:\u2192", "Y:\u2193"],
    "finorm": [0.01, 0.01],
    "fox": [20],
    "foy": [20],
    "foz": [0],
    "fodir": ["X:\u2192"],
    "fonorm": [0.01],
}
# A 2D preset with an different support
p_different_support = {
    "nelxyz": [60, 40, 0],
    "sdim": ["Y", "X", "X"],
    "sx": [0, 60, 0],
    "sy": [20, 20, 0],
    "fix": [0, 100],
    "fiy": [30, 40],
    "fiz": [0, 0],
    "fidir": ["X:\u2192", "Y:\u2193"],
    "finorm": [0.01, 0.01],
    "fox": [20],
    "foy": [20],
    "foz": [0],
    "fodir": ["X:\u2192"],
    "fonorm": [0.01],
}
# A 2D preset with an inactive force
p_inactive_force = {
    "nelxyz": [60, 40, 0],
    "sdim": ["Y", "X", "-"],
    "sx": [0, 60],
    "sy": [20, 20],
    "fix": [0, 100, 0],
    "fiy": [30, 40, 0],
    "fiz": [0, 0, 0],
    "fidir": ["X:\u2192", "Y:\u2193", "-"],
    "finorm": [0.01, 0.01, 0.0],
    "fox": [20],
    "foy": [20],
    "foz": [0],
    "fodir": ["X:\u2192"],
    "fonorm": [0.01],
}
# A 2D preset with an different force
p_different_force = {
    "nelxyz": [60, 40, 0],
    "sdim": ["Y", "X", "-"],
    "sx": [0, 60],
    "sy": [20, 20],
    "fix": [0, 100, 0],
    "fiy": [30, 40, 0],
    "fiz": [0, 0, 0],
    "fidir": ["X:\u2192", "Y:\u2193", "Y:\u2193"],
    "finorm": [0.01, 0.01, 0.0],
    "fox": [20],
    "foy": [20],
    "foz": [0],
    "fodir": ["X:\u2192"],
    "fonorm": [0.01],
}
# A 2D preset with an inactive region
p_inactive_region = {
    "nelxyz": [60, 40, 0],
    "sdim": ["Y", "X", "-"],
    "sx": [0, 60],
    "sy": [20, 20],
    "fix": [0, 100],
    "fiy": [30, 40],
    "fiz": [0, 0],
    "fidir": ["X:\u2192", "Y:\u2193"],
    "finorm": [0.01, 0.01],
    "fox": [20],
    "foy": [20],
    "foz": [0],
    "fodir": ["X:\u2192"],
    "fonorm": [0.01],
    "rshape": ["-"],
    "rstate": ["Void"],
    "rradius": [5],
    "rx": [30],
    "ry": [20],
    "rz": [0],
}
# A 2D preset with an different region
p_different_region = {
    "nelxyz": [60, 40, 0],
    "sdim": ["Y", "X", "-"],
    "sx": [0, 60],
    "sy": [20, 20],
    "fix": [0, 100],
    "fiy": [30, 40],
    "fiz": [0, 0],
    "fidir": ["X:\u2192", "Y:\u2193"],
    "finorm": [0.01, 0.01],
    "fox": [20],
    "foy": [20],
    "foz": [0],
    "fodir": ["X:\u2192"],
    "fonorm": [0.01],
    "rshape": ["□"],
    "rstate": ["Filled"],
    "rradius": [5],
    "rx": [25],
    "ry": [20],
    "rz": [0],
}
# A preset that is truly different
p_different = {
    "nelxyz": [80, 50, 0],
    "sdim": ["X", "Y"],
    "sx": [0, 60],
    "sy": [20, 20],
    "fix": [0, 100, 0],
    "fiy": [30, 40, 0],
    "fiz": [0, 0, 0],
    "fidir": ["X:\u2192", "Y:\u2193", "-"],
    "finorm": [0.01, 0.01, 0.0],
    "fox": [20],
    "foy": [20],
    "foz": [0],
    "fodir": ["X:\u2192"],
    "fonorm": [0.01],
    "rshape": ["□"],
    "rstate": ["Filled"],
    "rradius": [7],
    "rx": [20],
    "ry": [30],
    "rz": [0],
}


@pytest.mark.parametrize(
    "p1, p2, expected",
    [
        (p_base_2d, p_base_2d, True),  # Should equal itself
        (
            p_base_2d,
            p_inactive_support,
            True,
        ),  # Should be equivalent despite extra inactive support
        (p_base_2d, p_different_support, False),  # Should be different
        (
            p_base_2d,
            p_inactive_force,
            True,
        ),  # Should be equivalent despite extra inactive force
        (p_base_2d, p_different_force, False),  # Should be different
        (
            p_base_2d,
            p_inactive_region,
            True,
        ),  # Should be equivalent despite extra inactive region
        (p_base_2d, p_different_region, False),  # Should be different
        (p_base_2d, p_different, False),  # Should be different
    ],
)
def test_are_parameters_equivalent(qt_app, p1, p2, expected):
    """Unit Test: Tests the intelligent parameter comparison function."""
    # We need a MainWindow instance to get access to the method
    window = MainWindow()
    assert window.are_parameters_equivalent(p1, p2) == expected


def test_gather_and_apply_parameters(qt_app):
    """Unit Test: Checks if gathering and applying parameters works correctly."""
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


def test_save_result(qt_app):
    """Unit Test: Checks if the save result function works without error."""
    window = MainWindow()

    # Mock result data
    window.xPhys = [0.5] * 100
    window.last_params = {"nelxyz": (10, 10, 0)}
    window.figure = type("Fig", (), {"savefig": lambda *a, **k: None})()

    with patch("PySide6.QtWidgets.QFileDialog.getSaveFileName") as mock_dialog:
        mock_dialog.return_value = ("results/test.png", "PNG")

        # Should not raise
        window.save_result_as("png")
