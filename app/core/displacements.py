# app/core/displacements.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Linear displacement computation.

import copy
import numpy as np
from scipy.interpolate import griddata
from app.core.fem import FEM


def single_linear_displacement(
    u: np.ndarray, nelx: int, nely: int, nelz: int, disp_factor: float
):
    """
    Computes the deformed mesh grid for a single-frame plot.
    Returns X, Y, Z meshgrid arrays for plotting.
    """
    is_3d = nelz > 0
    nodes_flat = np.arange((nelx + 1) * (nely + 1) * (nelz + 1 if is_3d else 1))

    # Generate grid coordinates
    if is_3d:
        slice_size = (nelx + 1) * (nely + 1)
        z_coords = nodes_flat // slice_size
        rem = nodes_flat % slice_size
        x_coords = rem // (nely + 1)
        y_coords = rem % (nely + 1)
    else:
        x_coords = nodes_flat // (nely + 1)
        y_coords = nodes_flat % (nely + 1)

    # Calculate average displacement
    # u shape is (ndof, n_forces) or (ndof,)
    if u.ndim > 1:
        u_vec = np.mean(u, axis=1)
    else:
        u_vec = u

    # Map DOF indices to Node indices
    elemndof = 3 if is_3d else 2

    ux_avg = u_vec[elemndof * nodes_flat]
    uy_avg = u_vec[elemndof * nodes_flat + 1]

    X_flat = x_coords + ux_avg * disp_factor
    Y_flat = y_coords - uy_avg * disp_factor  # Minus for Y flip in plotting usually

    if is_3d:
        uz_avg = u_vec[elemndof * nodes_flat + 2]
        Z_flat = z_coords + uz_avg * disp_factor

        # Reshape to (Z, X, Y) then transpose to (X, Y, Z) for standard plotting tools
        shape = (nelz + 1, nelx + 1, nely + 1)
        X = X_flat.reshape(shape).transpose(1, 2, 0)
        Y = Y_flat.reshape(shape).transpose(1, 2, 0)
        Z = Z_flat.reshape(shape).transpose(1, 2, 0)
        return X, Y, Z
    else:
        shape = (nelx + 1, nely + 1)
        X = X_flat.reshape(shape)
        Y = Y_flat.reshape(shape)
        return X, Y


def run_iterative_displacement(params, xPhys_initial, progress_callback=None):
    """
    Performs iterative FE analysis to simulate displacement.
    Yields the cropped density field for each iteration.
    """
    # Expand Domain
    dims = params["Dimensions"]
    nelx, nely, nelz = dims["nelxyz"]
    is_3d = nelz > 0

    # Enlarge domain (approx 20% total padding)
    mx, my, mz = nelx // 5, nely // 5, (nelz // 5 if is_3d else 0)
    sim_params = copy.deepcopy(params)
    sim_params["Dimensions"]["nelxyz"] = [nelx + 2 * mx, nely + 2 * my, nelz + 2 * mz]

    # Initialize FEM
    fem = FEM(
        sim_params["Dimensions"], sim_params["Materials"], sim_params["Optimizer"]
    )
    fem.setup_boundary_conditions(sim_params["Forces"], sim_params.get("Supports"))

    # Offset Supports and Forces coordinates to center the part in the new domain
    def offset_coords(container, keys):
        for k, o in zip(keys, [mx, my, mz]):
            if k in container:
                container[k] = [val + o for val in container[k]]

    if "Supports" in sim_params:
        offset_coords(sim_params["Supports"], ["sx", "sy", "sz"])
    offset_coords(sim_params["Forces"], ["fix", "fiy", "fiz"])
    # Note: Output forces aren't usually relevant for the displacement visualization loop,
    # but we offset them for consistency.
    offset_coords(sim_params["Forces"], ["fox", "foy", "foz"])

    # Embed Material into Expanded Domain
    full_shape = (fem.nelx, fem.nely, fem.nelz) if is_3d else (fem.nelx, fem.nely)
    xPhys_large = np.zeros(full_shape)
    if is_3d:
        xPhys_large[mx : mx + nelx, my : my + nely, mz : mz + nelz] = (
            xPhys_initial.reshape(nelx, nely, nelz)
        )
        xPhys = xPhys_large.flatten(order="F")
    else:
        xPhys_large[mx : mx + nelx, my : my + nely] = xPhys_initial.reshape(nelx, nely)
        xPhys = xPhys_large.flatten()

    volfrac = np.mean(xPhys)  # Keep track to normalize later

    # Simulation Parameters
    pd = params["Displacement"]
    delta_disp = pd["disp_factor"] / max(1, pd["disp_iterations"])

    # Points for interpolation (Eulerian grid)
    # Pre-calculate original centers
    shape = (fem.nelz, fem.nelx, fem.nely) if is_3d else (fem.nelx, fem.nely)
    unrvld = np.unravel_index(np.arange(fem.nel), shape)
    points_interp = (
        np.column_stack((unrvld[1], unrvld[2], unrvld[0]) if is_3d else unrvld) + 0.5
    )

    # Initial Yield
    yield xPhys_initial
    if progress_callback:
        progress_callback(1)

    node_ids = [2, 1, 6, 5, 3, 0, 7, 4] if is_3d else [2, 1, 3, 0]

    # Sigmoid consts (Calculated outside loop for efficiency)
    k = 4  # Steepness
    nominator = (
        (1 + np.exp(-k / 2)) * (1 + np.exp(k / 2)) / (np.exp(k / 2) - np.exp(-k / 2))
    )
    c_val = -nominator / (1 + np.exp(k / 2))

    # Iterative Loop
    for it in range(pd["disp_iterations"]):
        # Solve FEM to get the deformation
        ui, _ = fem.solve(xPhys)

        # Collapse multiple load cases to average if necessary (usually 1 input force in this context)
        u_curr = np.mean(ui, axis=1) if ui.shape[1] > 0 else np.zeros(fem.ndof)

        # Follow the Forces
        # Update force coordinates based on current integer displacement
        # Calculate DOF index for this force, Get integer displacement at force application point
        # Check direction to pick specific component, Update coordinate in params
        for i, f_idx in enumerate(fem.fi_indices):
            if (
                int(u_curr[fem.di_indices[i]]) != 0
                and "X" in sim_params["Forces"]["fidir"][f_idx]
            ):
                sim_params["Forces"]["fix"][f_idx] += 0

        # Interpolate / Warp Mesh
        # Get nodal displacements for every element
        # fem.edofMat: (nel, 8*dim)
        # We need to gather (nel, nodes_per_elem) -> average -> (nel, dim)
        u_elem = u_curr[fem.edofMat].reshape(fem.nel, len(node_ids), fem.elemndof)

        # Displace points
        u_avg = np.mean(u_elem, axis=1)  # (nel, dim)
        points = points_interp.copy()
        points[:, 0] += u_avg[:, 0] * delta_disp
        points[:, 1] -= u_avg[:, 1] * delta_disp
        if is_3d:
            points[:, 2] += u_avg[:, 2] * delta_disp

        # Interpolate density from old points to new (warped) points
        xPhys = np.nan_to_num(
            griddata(points, xPhys, points_interp, method="linear"), nan=0.0
        )

        # Threshold & Normalize (Sigmoid filter)
        xPhys = nominator / (1 + np.exp(-k * (xPhys - 0.5))) + c_val
        curr_sum = np.sum(xPhys)
        if curr_sum > 0:
            xPhys = volfrac * xPhys / (curr_sum / fem.nel)
        xPhys = np.clip(xPhys, 0.0, 1.0)

        # Crop and Yield
        if is_3d:
            cropped = xPhys.reshape(fem.nelz, fem.nelx, fem.nely)[
                mz : mz + nelz, mx : mx + nelx, my : my + nely
            ]
        else:
            cropped = xPhys.reshape(fem.nelx, fem.nely)[mx : mx + nelx, my : my + nely]
        yield cropped.flatten()
        if progress_callback:
            progress_callback(it + 2)
