# tests/test_cli.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Tests for the CLI.

import sys
import json
from unittest.mock import patch
import pytest
import numpy as np
from pathlib import Path
from app.cli import run_cli


@pytest.fixture
def mock_presets_data():
    """Load presets from tests/presets_test.json."""
    presets_path = Path(__file__).parent / "presets_test.json"
    with open(presets_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_cli_help():
    """Test that -h/--help exits with code 0."""
    with patch.object(sys, "argv", ["main.py", "-h"]):
        with pytest.raises(SystemExit) as cm:
            run_cli()
        assert cm.value.code == 0


@patch("app.cli.optimizers.optimize")
@patch("app.cli.exporters")
@patch("builtins.open")
@patch("json.load")
@patch.object(Path, "exists")
def test_run_cli_valid_png(
    mock_exists,
    mock_json_load,
    mock_open,
    mock_exporters,
    mock_optimize,
    mock_presets_data,
):
    """Test running CLI with a valid preset and png output."""
    # Setup mocks
    mock_exists.return_value = True
    mock_json_load.return_value = mock_presets_data

    # Mock result from optimize
    # ForceInverter_2Sup_2D has nelxyz = [15, 10, 0] -> nel = 150
    mock_xPhys = np.zeros(150)
    mock_optimize.return_value = (mock_xPhys, None)

    # Mock exporters success
    mock_exporters.save_as_png.return_value = (True, None)

    # Run
    preset_name = "ForceInverter_2Sup_2D"
    with patch.object(
        sys, "argv", ["main.py", "-preset", preset_name, "-format", "png"]
    ):
        run_cli()

    # Verify optimize called with correct parameters
    mock_optimize.assert_called_once()
    call_kwargs = mock_optimize.call_args.kwargs
    assert call_kwargs["nelxyz"] == [15, 10, 0]
    assert "disp_factor" not in call_kwargs  # Should be removed

    # Verify export called
    mock_exporters.save_as_png.assert_called_once()
    args, _ = mock_exporters.save_as_png.call_args
    # Check filename ends with .png and contains preset name
    assert str(args[2]).endswith(f"{preset_name}.png")


@patch("app.cli.optimizers.optimize")
@patch("app.cli.exporters")
@patch("builtins.open")
@patch("json.load")
@patch.object(Path, "exists")
def test_run_cli_all_formats(
    mock_exists,
    mock_json_load,
    mock_open,
    mock_exporters,
    mock_optimize,
    mock_presets_data,
):
    """Test running CLI with default format (all)."""
    mock_exists.return_value = True
    mock_json_load.return_value = mock_presets_data
    mock_optimize.return_value = (np.zeros(150), None)
    mock_exporters.save_as_png.return_value = (True, None)
    mock_exporters.save_as_vti.return_value = (True, None)
    mock_exporters.save_as_stl.return_value = (True, None)

    preset_name = "ForceInverter_2Sup_2D"
    with patch.object(sys, "argv", ["main.py", "-preset", preset_name]):
        run_cli()

    mock_exporters.save_as_png.assert_called_once()
    mock_exporters.save_as_vti.assert_called_once()
    mock_exporters.save_as_stl.assert_called_once()


@patch("app.cli.optimizers.optimize")
@patch("app.cli.exporters")
@patch("builtins.open")
@patch("json.load")
@patch.object(Path, "exists")
def test_run_cli_threshold(
    mock_exists,
    mock_json_load,
    mock_open,
    mock_exporters,
    mock_optimize,
    mock_presets_data,
):
    """Test running CLI with threshold option."""
    mock_exists.return_value = True
    mock_json_load.return_value = mock_presets_data

    # Return gray values: 0.2 and 0.8
    mock_xPhys = np.array([0.2, 0.8])
    mock_optimize.return_value = (mock_xPhys, None)

    mock_exporters.save_as_png.return_value = (True, None)

    preset_name = "ForceInverter_2Sup_2D"
    with patch.object(
        sys, "argv", ["main.py", "-preset", preset_name, "-format", "png", "-threshold"]
    ):
        run_cli()

    # Verify the exporter received binary values: 0.0 and 1.0
    args, _ = mock_exporters.save_as_png.call_args
    exported_xPhys = args[0]
    expected = np.array([0.0, 1.0])
    np.testing.assert_array_equal(exported_xPhys, expected)


@patch("builtins.open")
@patch("json.load")
@patch.object(Path, "exists")
def test_run_cli_invalid_preset(
    mock_exists, mock_json_load, mock_open, mock_presets_data
):
    """Test behavior when preset name is invalid."""
    mock_exists.return_value = True
    mock_json_load.return_value = mock_presets_data

    with patch.object(sys, "argv", ["main.py", "-preset", "NonExistentPreset"]):
        with pytest.raises(SystemExit) as cm:
            run_cli()
        assert cm.value.code == 1


@patch.object(Path, "exists")
def test_run_cli_presets_file_not_found(mock_exists):
    """Test behavior when presets.json is missing."""
    mock_exists.return_value = False

    with patch.object(sys, "argv", ["main.py", "-preset", "TestPreset"]):
        with pytest.raises(SystemExit) as cm:
            run_cli()
        assert cm.value.code == 1
