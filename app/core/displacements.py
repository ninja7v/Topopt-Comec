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
    # 1. Setup Expanded Domain (Margins)
    dims = params["Dimensions"]
    nelx, nely, nelz = dims["nelxyz"]
    is_3d = nelz > 0

    # Calculate margins (approx 20% total padding)
    mx, my = nelx // 5, nely // 5
    mz = nelz // 5 if is_3d else 0

    # Create a deep copy of params to modify for the "Super FEM" environment
    sim_params = copy.deepcopy(params)
    sim_params["Dimensions"]["nelxyz"] = [nelx + 2 * mx, nely + 2 * my, nelz + 2 * mz]

    # Offset Supports and Forces coordinates to center the part in the new domain
    def offset_coords(container, keys):
        for key, offset in zip(keys, [mx, my, mz]):
            if key in container:
                container[key] = [val + offset for val in container[key]]

    if "Supports" in sim_params:
        offset_coords(sim_params["Supports"], ["sx", "sy", "sz"])
    offset_coords(sim_params["Forces"], ["fix", "fiy", "fiz"])
    # Note: Output forces aren't usually relevant for the displacement visualization loop,
    # but we offset them for consistency.
    offset_coords(sim_params["Forces"], ["fox", "foy", "foz"])

    # 2. Initialize FEM
    # We use the FEM class to handle mesh, stiffness, and solving
    fem = FEM(
        sim_params["Dimensions"], sim_params["Materials"], sim_params["Optimizer"]
    )
    fem.setup_boundary_conditions(
        sim_params["Forces"],
        sim_params["Supports"] if "Supports" in sim_params else None,
    )

    # 3. Embed Material into Expanded Domain
    # Create blank canvas
    full_shape = (fem.nelx, fem.nely, fem.nelz) if is_3d else (fem.nelx, fem.nely)
    xPhys_large = np.zeros(full_shape)

    # Reshape input to grid
    input_shape = (nelx, nely, nelz) if is_3d else (nelx, nely)
    x_input_grid = (
        xPhys_initial.reshape(input_shape[0], input_shape[1], input_shape[2])
        if is_3d
        else xPhys_initial.reshape(input_shape)
    )

    # Place input in center
    if is_3d:
        xPhys_large[mx : mx + nelx, my : my + nely, mz : mz + nelz] = x_input_grid
    else:
        xPhys_large[mx : mx + nelx, my : my + nely] = x_input_grid

    xPhys = xPhys_large.flatten(order="F") if is_3d else xPhys_large.flatten()
    volfrac = np.mean(xPhys)  # Keep track to normalize later

    # 4. Simulation Parameters
    pd = params["Displacement"]
    delta_disp = pd["disp_factor"] / max(1, pd["disp_iterations"])

    # Points for interpolation (Eulerian grid)
    # Pre-calculate original centers
    if is_3d:
        el_indices = np.arange(fem.nel)
        orig_z = el_indices // (fem.nelx * fem.nely) + 0.5
        rem = el_indices % (fem.nelx * fem.nely)
        orig_x = rem // fem.nely + 0.5
        orig_y = rem % fem.nely + 0.5
        points_interp = np.column_stack((orig_x, orig_y, orig_z))
    else:
        el_indices = np.arange(fem.nel)
        orig_x = el_indices // fem.nely + 0.5  # Center is +0.5
        orig_y = el_indices % fem.nely + 0.5
        points_interp = np.column_stack((orig_x, orig_y))

    # Initial Yield
    yield xPhys_initial
    if progress_callback:
        progress_callback(1)

    # 5. Iterative Loop
    node_ids = [2, 1, 6, 5, 3, 0, 7, 4] if is_3d else [2, 1, 3, 0]

    for it in range(pd["disp_iterations"]):
        # A. Solve FEM
        # Note: We only care about Input Forces (ui) for the deformation
        ui, _ = fem.solve(xPhys)

        # Collapse multiple load cases to average if necessary (usually 1 input force in this context)
        if ui.shape[1] > 0:
            u_curr = np.mean(ui, axis=1)
        else:
            u_curr = np.zeros(fem.ndof)

        # B. Moving Loads Logic (Follower Forces)
        # Update force coordinates based on current integer displacement
        updated_forces = False
        active_f_indices = fem.fi_indices  # Indices in the Forces dict

        for i, f_idx in enumerate(active_f_indices):
            # Calculate DOF index for this force
            dof_idx = fem.di_indices[i]

            # Get integer displacement at force application point
            # Check direction to pick specific component
            f_dir = sim_params["Forces"]["fidir"][f_idx]
            u_val = 0
            if "X" in f_dir:
                u_val = u_curr[dof_idx]
            elif "Y" in f_dir:
                u_val = u_curr[dof_idx]
            elif is_3d and "Z" in f_dir:
                u_val = u_curr[dof_idx]

            shift = int(u_val)
            if shift != 0:
                # Update coordinate in params
                if "X" in f_dir:
                    sim_params["Forces"]["fix"][f_idx] += 0
                pass

        if updated_forces:
            fem.setup_boundary_conditions(
                sim_params["Forces"],
                sim_params["Supports"] if "Supports" in sim_params else None,
            )

        # C. Interpolate / Warp Mesh
        # Get nodal displacements for every element
        # fem.edofMat: (nel, 8*dim)
        # We need to gather (nel, nodes_per_elem) -> average -> (nel, dim)

        u_elem = u_curr[fem.edofMat].reshape(fem.nel, len(node_ids), fem.elemndof)

        # Average displacement per element
        u_avg = np.mean(u_elem, axis=1)  # (nel, dim)
        ux, uy = u_avg[:, 0], u_avg[:, 1]
        uz = u_avg[:, 2] if is_3d else None

        # Displace points
        points = points_interp.copy()
        points[:, 0] += ux * delta_disp
        points[:, 1] += -uy * delta_disp
        if is_3d:
            points[:, 2] += uz * delta_disp

        # Interpolate density from old points to new (warped) points
        xPhys = griddata(points, xPhys, points_interp, method="linear")
        xPhys = np.nan_to_num(xPhys, nan=0.0)

        # D. Threshold & Normalize (Sigmoid filter)
        k, c_val = 4, 0.0  # Steepness
        # Sigmoid consts
        nominator = (
            (1 + np.exp(-k / 2))
            * (1 + np.exp(k / 2))
            / (np.exp(k / 2) - np.exp(-k / 2))
        )
        c_val = -nominator / (1 + np.exp(k / 2))

        xPhys = nominator / (1 + np.exp(-k * (xPhys - 0.5))) + c_val

        # Renormalize volume
        curr_sum = np.sum(xPhys)
        if curr_sum > 0:
            xPhys = volfrac * xPhys / (curr_sum / fem.nel)
        xPhys = np.clip(xPhys, 0.0, 1.0)

        # E. Crop and Yield
        if is_3d:
            grid = xPhys.reshape(fem.nelz, fem.nelx, fem.nely)
            cropped = grid[mz : mz + nelz, mx : mx + nelx, my : my + nely]
            # Transpose to match visualization expectation if needed,
            # but usually flat yield is expected in standard order
        else:
            grid = xPhys.reshape(fem.nelx, fem.nely)
            cropped = grid[mx : mx + nelx, my : my + nely]

        yield cropped.flatten()

        if progress_callback:
            progress_callback(it + 2)
