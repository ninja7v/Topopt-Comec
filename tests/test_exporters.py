# tests/test_exporters.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Tests for the exporters.

from pathlib import Path

import numpy as np

from app.ui import exporters


def test_exporters():
    """Unit Test: Runs the exporters."""
    filepath = Path(__file__).parent / "test_output_file"

    # Generate a 2D mock result
    (nelx_2d, nely_2d, nelz_2d) = 30, 20, 0
    nel_2d = nelx_2d * nely_2d
    p = (
        1 / 0.3 - 1
    )  # f(x) = (x/volfrac)^p -> integral(f(x)) from 0 to nel = volfrac * nel
    x = np.linspace(0, 1, nel_2d)
    densities_2d = x**p
    np.random.shuffle(densities_2d)
    result_2d = densities_2d

    # Generate a 3D mock result
    nelx_3d, nely_3d, nelz_3d = 30, 20, 10
    nel_3d = nelx_3d * nely_3d * nelz_3d
    p = (
        1 / 0.3 - 1
    )  # f(x) = (x/volfrac)^p -> integral(f(x)) from 0 to nel = volfrac * nel
    x = np.linspace(0, 1, nel_3d)
    densities_3d = x**p
    np.random.shuffle(densities_3d)
    result_3d = densities_3d

    # Save as PNG
    png_path = str(Path(__file__).parent / "test_output_file.png")
    success, error_msg = exporters.save_as_png(
        result_2d, (nelx_2d, nely_2d, nelz_2d), png_path
    )
    assert success, f"PNG export for 2D mechanism failed: {error_msg}"
    success, error_msg = exporters.save_as_png(
        result_3d, (nelx_3d, nely_3d, nelz_3d), png_path
    )
    assert success, f"PNG export for 3D mechanism failed: {error_msg}"

    # Save as VTI
    success, error_msg = exporters.save_as_vti(
        result_2d, (nelx_2d, nely_2d, nelz_2d), filepath
    )
    assert success, f"VTI export for 2D mechanism failed: {error_msg}"
    success, error_msg = exporters.save_as_vti(
        result_3d, (nelx_3d, nely_3d, nelz_3d), filepath
    )
    assert success, f"VTI export for 3D mechanism failed: {error_msg}"

    # Save as STL
    success, error_msg = exporters.save_as_stl(
        result_2d, (nelx_2d, nely_2d, nelz_2d), filepath
    )
    assert success, f"STL export for 2D mechanism failed: {error_msg}"
    success, error_msg = exporters.save_as_stl(
        result_3d, (nelx_3d, nely_3d, nelz_3d), filepath
    )
    assert success, f"STL export for 3D mechanism failed: {error_msg}"

    # Save as 3MF
    threemf_path = str(Path(__file__).parent / "test_output_file.3mf")
    success, error_msg = exporters.save_as_3mf(
        result_2d, (nelx_2d, nely_2d, nelz_2d), threemf_path
    )
    assert success, f"3MF export for 2D mechanism failed: {error_msg}"
    success, error_msg = exporters.save_as_3mf(
        result_3d, (nelx_3d, nely_3d, nelz_3d), threemf_path
    )
    assert success, f"3MF export for 3D mechanism failed: {error_msg}"
