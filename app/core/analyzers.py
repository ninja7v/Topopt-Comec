# app/core/analyzers.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Analyze a mechanism.

from typing import Dict, Callable, Optional, Tuple
import numpy as np


def checkerboard(x: np.ndarray) -> bool:
    """check if the mechanism contains a checkerboard pattern"""
    # Apply a mask [[0, 1], [1, 0]] to the xPhys array with a tolerance to detect checkerboard patterns
    xbin = (x > 0.5).astype(int)
    if xbin.ndim == 2:
        mask1 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
        mask2 = ~mask1
        h, w = xbin.shape
        for i in range(h - 2):
            for j in range(w - 2):
                block = xbin[i : i + 3, j : j + 3]
                if np.array_equal(block, mask1) or np.array_equal(block, mask2):
                    return True
    else:
        mask1 = np.array(
            [
                [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
                [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            ],
            dtype=bool,
        )
        mask2 = ~mask1
        h, w, d = xbin.shape
        for i in range(h - 2):
            for j in range(w - 2):
                for k in range(d - 2):
                    block = xbin[i : i + 3, j : j + 3, k : d + 3]
                    if np.array_equal(block, mask1) or np.array_equal(block, mask2):
                        return True
    return False


def watertight(x: np.ndarray) -> bool:
    """check if the mechanism is watertight"""
    # Binarize xPhys with a threshold of 0.5 to get a binary image
    xbin = (x > 0.5).astype(int)
    from scipy.ndimage import label, generate_binary_structure

    # Define a structure to consider diagonal connections as touching
    structure = generate_binary_structure(rank=xbin.ndim, connectivity=xbin.ndim)
    res = label(xbin, structure)
    if isinstance(res, tuple):
        _, n = res
    else:
        n = res
    return n == 1  # If there is only one connected component (connex), it is watertight


def threholded(xPhys: np.ndarray) -> bool:
    """check if the mechanism is threholded"""
    # Check if np.mean(np.minimum(x, 1 - x)) is close to 0 (worst case is 0.5 where all elements are at 0.5)
    mean = np.mean(np.minimum(xPhys, 1 - xPhys))
    return bool(mean < 0.1)


def efficient(u: np.ndarray, Dimensions: Dict, Forces: Dict) -> bool:
    """check if the mechanism is efficient"""
    effectiveness = 0.0
    nelx, nely, nelz = Dimensions["nelxyz"]
    is_3d = nelz > 0
    active_iforces_indices = [
        i for i in range(len(Forces["fidir"])) if Forces["fidir"][i] != "-"
    ]
    nbForces = len(active_iforces_indices)
    j = 0
    output_forces = Forces.get("fodir", [])
    if output_forces:
        # Compliant mechanism: output displacement must be big relative to the input displacement (at respective positions)
        for i in active_iforces_indices:
            idx = (
                (Forces["fiz"][i] * nelx * nely if is_3d else 0)
                + Forces["fix"][i] * nely
                + Forces["fiy"][i]
            )
            idx = idx * (3 if is_3d else 2) + (
                0
                if Forces["fidir"][i] == "X:\u2192" or Forces["fidir"][i] == "X:\u2190"
                else (
                    1
                    if Forces["fidir"][i] == "Y:\u2193"
                    or Forces["fidir"][i] == "Y:\u2191"
                    else 2
                )
            )
            effectiveness += u[idx, j] - (Forces["finorm"][i] * 10)
            j += 1
        return bool(
            effectiveness < 0.5 * nbForces
        )  # The smaller effectiveness, the better
    else:
        # Rigid mechanism: displacement at input location must be small small relative to the applied force
        for i in active_iforces_indices:
            idx = (
                (Forces["fiz"][i] * nelx * nelz if is_3d else 0)
                + Forces["fix"][i] * nely
                + Forces["fiy"][i]
            )
            idx = idx * (3 if is_3d else 2) + (
                0
                if Forces["fidir"][i] == "X:\u2192" or Forces["fidir"][i] == "X:\u2190"
                else (
                    1
                    if Forces["fidir"][i] == "Y:\u2193"
                    or Forces["fidir"][i] == "Y:\u2191"
                    else 2
                )
            )
            effectiveness += u[idx, j] / (Forces["finorm"][i] * 10)
            j += 1
        return bool(
            effectiveness < 0.5 * nbForces
        )  # The smaller effectiveness, the better


def analyze(
    xPhys: np.ndarray,
    u: np.ndarray,
    Dimensions: Dict,
    Forces: Dict,
    progress_callback: Optional[Callable] = None,
) -> Tuple[bool, bool, bool, bool]:
    """Analyze the mechanism"""
    xPhys_copy = xPhys.copy()
    if xPhys.ndim == 2 and xPhys.shape[0] == 2:
        xPhys_copy = xPhys_copy.mean(axis=0, keepdims=True)
    x = (
        xPhys_copy.reshape(
            Dimensions["nelxyz"][2], Dimensions["nelxyz"][0], Dimensions["nelxyz"][1]
        )
        if Dimensions["nelxyz"][2] > 0
        else xPhys_copy.reshape(Dimensions["nelxyz"][0], Dimensions["nelxyz"][1])
    )

    contains_checkerboard = checkerboard(x)
    if progress_callback and progress_callback(1):
        print("Optimization stopped by user.")
        return contains_checkerboard, False, False, False

    is_watertight = watertight(x)
    if progress_callback and progress_callback(2):
        print("Optimization stopped by user.")
        return contains_checkerboard, is_watertight, False, False

    is_thresholded = threholded(xPhys)
    if progress_callback and progress_callback(3):
        print("Optimization stopped by user.")
        return contains_checkerboard, is_watertight, is_thresholded, False

    is_efficient = efficient(u, Dimensions, Forces)
    if progress_callback and progress_callback(4):
        print("Optimization stopped by user.")

    return contains_checkerboard, is_watertight, is_thresholded, is_efficient
