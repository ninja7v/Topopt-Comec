# tests/test_main_window.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Tests for the main window.

import pytest

from app.ui.main_window import MainWindow

from unittest.mock import patch

# --- Test Cases for the Intelligent Comparison ---
p_base_2d = {
    "Dimensions": {"nelxyz": [60, 40, 0]},
    "Supports": {"sdim": ["Y", "X"], "sx": [0, 60], "sy": [20, 20], "sr": [0, 0]},
    "Forces": {
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
    },
    "Materials": {
        "E": [1.0],
        "nu": [0.25],
        "percent": [100],
        "color": ["#000000"],
        "init_type": 0,
    },
    "Regions": {},
    "Displacement": {"disp_factor": 1.0, "disp_iterations": 1},
}

p_inactive_support = {
    "Dimensions": {"nelxyz": [60, 40, 0]},
    "Supports": {
        "sdim": ["Y", "X", "-"],
        "sx": [0, 60, 0],
        "sy": [20, 20, 0],
        "sr": [0, 0, 0],
    },
    "Forces": {
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
    },
    "Materials": {
        "E": [1.0],
        "nu": [0.25],
        "percent": [100],
        "color": ["#000000"],
        "init_type": 0,
    },
    "Regions": {},
    "Displacement": {"disp_factor": 1.0, "disp_iterations": 1},
}

p_different_support = {
    "Dimensions": {"nelxyz": [60, 40, 0]},
    "Supports": {
        "sdim": ["Y", "X", "X"],
        "sx": [0, 60, 0],
        "sy": [20, 20, 0],
        "sr": [0, 0, 0],
    },
    "Forces": {
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
    },
    "Materials": {
        "E": [1.0],
        "nu": [0.25],
        "percent": [100],
        "color": ["#000000"],
        "init_type": 0,
    },
    "Regions": {},
    "Displacement": {"disp_factor": 1.0, "disp_iterations": 1},
}

p_inactive_force = {
    "Dimensions": {"nelxyz": [60, 40, 0]},
    "Supports": {"sdim": ["Y", "X"], "sx": [0, 60], "sy": [20, 20], "sr": [0, 0]},
    "Forces": {
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
    },
    "Materials": {
        "E": [1.0],
        "nu": [0.25],
        "percent": [100],
        "color": ["#000000"],
        "init_type": 0,
    },
    "Regions": {},
    "Displacement": {"disp_factor": 1.0, "disp_iterations": 1},
}

p_different_force = {
    "Dimensions": {"nelxyz": [60, 40, 0]},
    "Supports": {
        "sdim": ["Y", "X", "-"],
        "sx": [0, 60, 0],
        "sy": [20, 20, 0],
        "sr": [0, 0, 0],
    },
    "Forces": {
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
    },
    "Materials": {
        "E": [1.0],
        "nu": [0.25],
        "percent": [100],
        "color": ["#000000"],
        "init_type": 0,
    },
    "Regions": {},
    "Displacement": {"disp_factor": 1.0, "disp_iterations": 1},
}

p_inactive_region = {
    "Dimensions": {"nelxyz": [60, 40, 0]},
    "Supports": {"sdim": ["Y", "X"], "sx": [0, 60], "sy": [20, 20], "sr": [0, 0]},
    "Forces": {
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
    },
    "Regions": {
        "rshape": ["-"],
        "rstate": ["Void"],
        "rradius": [5],
        "rx": [30],
        "ry": [20],
        "rz": [0],
    },
    "Materials": {
        "E": [1.0],
        "nu": [0.25],
        "percent": [100],
        "color": ["#000000"],
        "init_type": 0,
    },
    "Displacement": {"disp_factor": 1.0, "disp_iterations": 1},
}

p_different_region = {
    "Dimensions": {"nelxyz": [60, 40, 0]},
    "Supports": {"sdim": ["Y", "X", "-"], "sx": [0, 60], "sy": [20, 20], "sr": [0, 0]},
    "Forces": {
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
    },
    "Regions": {
        "rshape": ["□"],
        "rstate": ["Filled"],
        "rradius": [5],
        "rx": [25],
        "ry": [20],
        "rz": [0],
    },
    "Materials": {
        "E": [1.0],
        "nu": [0.25],
        "percent": [100],
        "color": ["#000000"],
        "init_type": 0,
    },
    "Displacement": {"disp_factor": 1.0, "disp_iterations": 1},
}

p_different_material = {
    "Dimensions": {"nelxyz": [60, 40, 0]},
    "Supports": {"sdim": ["Y", "X"], "sx": [0, 60], "sy": [20, 20], "sr": [0, 0]},
    "Forces": {
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
    },
    "Materials": {
        "E": [3.0],
        "nu": [0.4],
        "percent": [100],
        "color": ["#000000"],
        "init_type": 0,
    },
    "Regions": {},
    "Displacement": {"disp_factor": 1.0, "disp_iterations": 1},
}
# A preset that is truly different
p_different = {
    "Dimensions": {"nelxyz": [80, 50, 0]},
    "Supports": {"sdim": ["X", "Y"], "sx": [0, 60], "sy": [20, 20], "sr": [0, 0]},
    "Forces": {
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
    },
    "Materials": {
        "E": [2.0],
        "nu": [0.25],
        "percent": [100],
        "color": ["#000500"],
        "init_type": 0,
    },
    "Regions": {
        "rshape": ["□"],
        "rstate": ["Filled"],
        "rradius": [7],
        "rx": [20],
        "ry": [30],
        "rz": [0],
    },
    "Displacement": {"disp_factor": 1.0, "disp_iterations": 1},
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
        (p_base_2d, p_different_material, False),  # Should be different
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
    modified_params["Dimensions"]["nelxyz"] = [100, 80, 10]
    modified_params["Supports"]["sx"][0] = 50

    # 3. Add regions (simulate a preset with multiple regions)
    if "Regions" not in modified_params:
        modified_params["Regions"] = {}
    modified_params["Regions"]["rshape"] = ["□", "◯"]
    modified_params["Regions"]["rstate"] = ["Void", "Filled"]
    modified_params["Regions"]["rradius"] = [5, 10]
    modified_params["Regions"]["rx"] = [10, 20]
    modified_params["Regions"]["ry"] = [10, 20]
    modified_params["Regions"]["rz"] = [0, 0]

    # 4. Apply these modified parameters back to the UI
    window.apply_parameters(modified_params)

    # 5. Gather the parameters from the UI again
    new_params_from_ui = window.gather_parameters()

    # 6. Assert that the UI state now matches the modified parameters
    assert new_params_from_ui["Dimensions"]["nelxyz"] == [100, 80, 10]
    assert new_params_from_ui["Supports"]["sx"][0] == 50
    assert len(new_params_from_ui["Regions"]["rshape"]) == 2
    assert new_params_from_ui["Regions"]["rshape"][1] == "◯"
    assert new_params_from_ui["Regions"]["rradius"][1] == 10


def test_save_result(qt_app):
    """Unit Test: Checks if the save result function works without error."""
    window = MainWindow()

    # Mock result data
    window.xPhys = [0.5] * 100
    window.last_params["Dimensions"] = {"nelxyz": (10, 10, 0)}
    window.figure = type("Fig", (), {"savefig": lambda *a, **k: None})()

    with patch("PySide6.QtWidgets.QFileDialog.getSaveFileName") as mock_dialog:
        mock_dialog.return_value = ("results/test.png", "PNG")

        # Should not raise
        window.save_result_as("png")
