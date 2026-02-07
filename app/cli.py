# app/cli.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# CLI entry point of Topopt-Comec.

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from app.core import optimizers
from app.ui import exporters


def run_cli():
    """Parses arguments and runs the optimization from the CLI."""
    parser = argparse.ArgumentParser(
        description="TopOpt-Comec CLI - Topology Optimization for Compliant Mechanisms"
    )
    parser.add_argument(
        "-preset",
        type=str,
        required=True,
        help="Name of the preset to use from presets.json",
    )
    parser.add_argument(
        "-format",
        type=str,
        default="all",
        choices=["png", "stl", "vti", "all"],
        help="Output format (png, stl, vti). Default: all",
    )
    parser.add_argument(
        "-threshold",
        action="store_true",
        help="Binarize the result (black and white)",
    )

    args = parser.parse_args()

    # Load presets
    presets_path = Path("presets.json")
    if not presets_path.exists():
        print(f"Error: presets.json not found at {presets_path.absolute()}")
        sys.exit(1)

    try:
        with open(presets_path, "r") as f:
            presets = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error reading presets.json: {e}")
        sys.exit(1)

    if args.preset not in presets:
        print(f"Error: Preset '{args.preset}' not found in presets.json")
        print("Available presets:", ", ".join(presets.keys()))
        sys.exit(1)

    params = presets[args.preset]
    print(f"Running optimization for preset: {args.preset}")

    # clean params for optimizer
    optimizer_params = params.copy()
    keys_to_remove = ["disp_factor", "disp_iterations"]
    for key in keys_to_remove:
        optimizer_params.pop(key, None)

    # Run optimization
    try:
        xPhys, _ = optimizers.optimize(**optimizer_params)
    except Exception as e:
        print(f"Optimization failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Apply threshold if requested
    if args.threshold:
        print("Applying threshold (0.5)...")
        xPhys = np.where(xPhys > 0.5, 1.0, 0.0)

    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    base_filename = results_dir / args.preset

    # Handle formats
    formats = []
    if args.format == "all":
        formats = ["png", "stl", "vti"]
    else:
        formats = [args.format]

    # Export
    nelxyz = params["nelxyz"]

    for fmt in formats:
        filename = str(base_filename.with_suffix(f".{fmt if fmt != 'stl' else 'stl'}"))
        print(f"Saving {fmt.upper()} to {filename}...")

        success = False
        error_msg = None

        if fmt == "png":
            success, error_msg = exporters.save_as_png(xPhys, nelxyz, filename)
        elif fmt == "vti":
            success, error_msg = exporters.save_as_vti(xPhys, nelxyz, filename)
        elif fmt == "stl":
            success, error_msg = exporters.save_as_stl(xPhys, nelxyz, filename)

        if not success:
            print(f"Error saving {fmt}: {error_msg}")
        else:
            print(f"Saved {filename}")

    print("Done.")
