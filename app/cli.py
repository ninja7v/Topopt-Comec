# app/cli.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# CLI entry point of TopoptComec.

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from app.core import optimizers
from app.ui import exporters


def _run_single_preset(preset_name, params, fmt, threshold, verbose=False):
    """Run optimization and export for a single preset. Returns (preset_name, error)."""
    if verbose:
        print(f"Running optimization for preset: {preset_name}")

    # Clean params for optimizer
    optimizer_params = params.copy()
    if "Displacement" in optimizer_params:
        optimizer_params.pop("Displacement")

    # Run optimization
    try:
        xPhys, _ = optimizers.optimize(**optimizer_params, verbose=verbose)
    except Exception as e:
        import traceback

        traceback.print_exc()
        return preset_name, f"Optimization failed: {e}"

    # Apply threshold if requested
    if threshold:
        if verbose:
            print(f"[{preset_name}] Applying threshold (0.5)...")
        xPhys = np.where(xPhys > 0.5, 1.0, 0.0)

    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    base_filename = results_dir / preset_name

    # Export
    nelxyz = params["Dimensions"]["nelxyz"]
    formats = ["png", "stl", "vti", "3mf"] if fmt == "all" else [fmt]
    for f in formats:
        filename = str(base_filename.with_suffix(f".{f}"))
        if verbose:
            print(f"[{preset_name}] Saving {f.upper()} to {filename}...")

        success, error_msg = _export(xPhys, nelxyz, filename, f)

        if not success:
            print(f"[{preset_name}] Error saving {f}: {error_msg}")
        else:
            print(f"[{preset_name}] Saved {filename}")

    if not verbose:
        print(f"Preset '{preset_name}' completed.")
    return preset_name, None


def _export(xPhys, nelxyz, filename, fmt):
    """Dispatch export to the correct exporter."""
    if fmt == "png":
        return exporters.save_as_png(xPhys, nelxyz, filename)
    elif fmt == "vti":
        return exporters.save_as_vti(xPhys, nelxyz, filename)
    elif fmt == "stl":
        return exporters.save_as_stl(xPhys, nelxyz, filename)
    elif fmt == "3mf":
        return exporters.save_as_3mf(xPhys, nelxyz, filename)
    return False, f"Unknown format: {fmt}"


def run_cli():
    """Parses arguments and runs the optimization from the CLI."""
    parser = argparse.ArgumentParser(
        description="TopoptComec CLI - Topology Optimization for Compliant Mechanisms"
    )
    parser.add_argument(
        "-p",
        "--preset",
        type=str,
        required=True,
        help="Preset name(s) from presets.json (comma-separated for parallel runs)",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default="all",
        choices=["png", "stl", "vti", "3mf", "all"],
        help="Output format (png, stl, vti, 3mf). Default: all",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        action="store_true",
        help="Binarize the result (black and white)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Suppress intermediate optimizer output (useful for parallel runs)",
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

    # Parse and validate preset names
    preset_names = [name.strip() for name in args.preset.split(",")]
    for name in preset_names:
        if name not in presets:
            print(f"Error: Preset '{name}' not found in presets.json")
            print("Available presets:", ", ".join(presets.keys()))
            sys.exit(1)

    # Run
    if len(preset_names) == 1:
        _, error = _run_single_preset(
            preset_names[0],
            presets[preset_names[0]],
            args.format,
            args.threshold,
            args.verbose,
        )
        if error:
            print(error)
            sys.exit(1)
    else:
        max_workers = min(len(preset_names), os.cpu_count() or 1)
        print(
            f"Running {len(preset_names)} presets in parallel ({max_workers} workers)"
        )
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _run_single_preset,
                    name,
                    presets[name],
                    args.format,
                    args.threshold,
                    args.verbose,
                ): name
                for name in preset_names
            }
            errors = []
            for future in futures:
                preset_name, error = future.result()
                if error:
                    errors.append(f"  {preset_name}: {error}")

        if errors:
            print("Errors occurred:\n" + "\n".join(errors))
            sys.exit(1)

    print("Done.")
