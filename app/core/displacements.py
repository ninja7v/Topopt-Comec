# app/core/displacements.py
# MIT License - Copyright (c) 2025 Luc Prevost
# Linear displacement computation.

import numpy as np
from cvxopt import cholmod, matrix, spmatrix  # Convex optimization
from scipy.interpolate import griddata
from scipy.sparse import coo_matrix  # Provides good N-dimensional array manipulation
from scipy.sparse.linalg import spsolve


def lk(E: float, nu: float, is_3d: bool) -> np.ndarray:
    """Get element stiffness matrix."""
    # Get K
    if is_3d:
        A = np.array(
            [
                [32, 6, -8, 6, -6, 4, 3, -6, -10, 3, -3, -3, -4, -8],
                [-48, 0, 0, -24, 24, 0, 0, 0, 12, -12, 0, 12, 12, 12],
            ]
        )
        k = 1 / 72 * (A.T @ np.array([1, nu]))
    else:
        k = np.array(
            [
                1 / 2 - nu / 6,
                1 / 8 + nu / 8,
                -1 / 4 - nu / 12,
                -1 / 8 + 3 * nu / 8,
                -1 / 4 + nu / 12,
                -1 / 8 - nu / 8,
                nu / 6,
                1 / 8 - 3 * nu / 8,
            ]
        )

    # Build KE
    if is_3d:
        K1 = np.array(
            [
                [k[0], k[1], k[1], k[2], k[4], k[4]],
                [k[1], k[0], k[1], k[3], k[5], k[6]],
                [k[1], k[1], k[0], k[3], k[6], k[5]],
                [k[2], k[3], k[3], k[0], k[7], k[7]],
                [k[4], k[5], k[6], k[7], k[0], k[1]],
                [k[4], k[6], k[5], k[7], k[1], k[0]],
            ]
        )
        K2 = np.array(
            [
                [k[8], k[7], k[11], k[5], k[3], k[6]],
                [k[7], k[8], k[11], k[4], k[2], k[4]],
                [k[9], k[9], k[12], k[6], k[3], k[5]],
                [k[5], k[4], k[10], k[8], k[1], k[9]],
                [k[3], k[2], k[4], k[1], k[8], k[11]],
                [k[10], k[3], k[5], k[11], k[9], k[12]],
            ]
        )
        K3 = np.array(
            [
                [k[5], k[6], k[3], k[8], k[11], k[7]],
                [k[6], k[5], k[3], k[9], k[12], k[9]],
                [k[4], k[4], k[2], k[7], k[11], k[8]],
                [k[8], k[9], k[1], k[5], k[10], k[4]],
                [k[11], k[12], k[9], k[10], k[5], k[3]],
                [k[1], k[11], k[8], k[3], k[4], k[2]],
            ]
        )
        K4 = np.array(
            [
                [k[13], k[10], k[10], k[12], k[9], k[9]],
                [k[10], k[13], k[10], k[11], k[8], k[7]],
                [k[10], k[10], k[13], k[11], k[7], k[8]],
                [k[12], k[11], k[11], k[13], k[6], k[6]],
                [k[9], k[8], k[7], k[6], k[13], k[10]],
                [k[9], k[7], k[8], k[6], k[10], k[13]],
            ]
        )
        K5 = np.array(
            [
                [k[0], k[1], k[7], k[2], k[4], k[3]],
                [k[1], k[0], k[7], k[3], k[5], k[10]],
                [k[7], k[7], k[0], k[4], k[10], k[5]],
                [k[2], k[3], k[4], k[0], k[7], k[1]],
                [k[4], k[5], k[10], k[7], k[0], k[7]],
                [k[3], k[10], k[5], k[1], k[7], k[0]],
            ]
        )
        K6 = np.array(
            [
                [k[13], k[10], k[6], k[12], k[9], k[11]],
                [k[10], k[13], k[6], k[11], k[8], k[1]],
                [k[6], k[6], k[13], k[9], k[1], k[8]],
                [k[12], k[11], k[9], k[13], k[6], k[10]],
                [k[9], k[8], k[1], k[6], k[13], k[6]],
                [k[11], k[1], k[8], k[10], k[6], k[13]],
            ]
        )
        KE = (
            E
            / ((nu + 1) * (1 - 2 * nu))
            * np.block(
                [
                    [K1, K2, K3, K4],
                    [K2.T, K5, K6, K3.T],
                    [K3.T, K6, K5.T, K2.T],
                    [K4, K3, K2, K1.T],
                ]
            )
        )
    else:
        KE = (
            E
            / (1 - nu**2)
            * np.array(
                [
                    [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                    [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                    [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                    [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                    [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                    [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                    [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                    [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
                ]
            )
        )
    return KE


def single_linear_displacement(u, nelx, nely, nelz, disp_factor):
    """
    Computes the deformed mesh grid and a grid pattern for a single-frame plot.
    Returns X, Y, Z meshgrid arrays for plotting.
    """
    is_3d = nelz > 0
    nodes_flat = np.arange((nelx + 1) * (nely + 1) * (nelz + 1 if is_3d else 1))

    if is_3d:
        x_coords = (nodes_flat % ((nely + 1) * (nelx + 1))) // (nely + 1)
        y_coords = (nodes_flat % ((nely + 1) * (nelx + 1))) % (nely + 1)
        z_coords = nodes_flat // ((nely + 1) * (nelx + 1))
    else:
        x_coords = nodes_flat // (nely + 1)
        y_coords = nodes_flat % (nely + 1)

    # Average displacement over all input forces
    elemndof = 3 if is_3d else 2  # Degrees of freedom per element
    n = elemndof * nodes_flat
    ux_avg = np.mean(u[n, :], axis=1)
    uy_avg = np.mean(u[n + 1, :], axis=1)
    if is_3d:
        uz_avg = np.mean(u[n + 2, :], axis=1)

    X_flat = x_coords + ux_avg * disp_factor
    Y_flat = y_coords - uy_avg * disp_factor
    if is_3d:
        Z_flat = z_coords + uz_avg * disp_factor

    if is_3d:
        # reshape according to the original flat ordering (z, x, y) then permute to (x, y, z)
        X = X_flat.reshape((nelz + 1, nelx + 1, nely + 1)).transpose(1, 2, 0)
        Y = Y_flat.reshape((nelz + 1, nelx + 1, nely + 1)).transpose(1, 2, 0)
        Z = Z_flat.reshape((nelz + 1, nelx + 1, nely + 1)).transpose(1, 2, 0)
        return X, Y, Z
    else:
        X = X_flat.reshape((nelx + 1, nely + 1))
        Y = Y_flat.reshape((nelx + 1, nely + 1))
        return X, Y


def run_iterative_displacement(params, xPhys_initial, progress_callback=None):
    """
    Performs iterative FE analysis to simulate displacement.
    Yields the cropped density field for each iteration.
    """
    # Initialization
    nelx, nely, nelz = params["nelxyz"][:3]
    is_3d = nelz > 0
    # Extend domain with margins (5%)
    margin_X = nelx // 5
    margin_Y = nely // 5
    if is_3d:
        margin_Z = nelz // 5
    nely += 2 * margin_Y
    nelx += 2 * margin_X
    if is_3d:
        nelz += 2 * margin_Z
    elemndof = 3 if is_3d else 2
    ndof = elemndof * (nelx + 1) * (nely + 1) * ((nelz + 1) if is_3d else 1)
    nel = nelx * nely * (nelz if is_3d else 1)
    Emin, Emax = 1e-9, 100.0
    KE = lk(params["E"], params["nu"], is_3d)
    size = 8 * (elemndof if is_3d else 1)
    edofMat = np.zeros((nel, size), dtype=int)
    for ex in range(nelx):
        for ey in range(nely):
            for ez in range(nelz if is_3d else 1):
                n1 = (
                    (ez * (nelx + 1) * (nely + 1) if is_3d else 0)
                    + ex * (nely + 1)
                    + ey
                )
                n2 = n1 + nely + 1
                n3 = n2 + 1
                n4 = n1 + 1
                nodes = [n4, n3, n2, n1]  # first square (XY plane, bottom face)
                if is_3d:
                    n5 = n1 + (nelx + 1) * (nely + 1)
                    n6 = n2 + (nelx + 1) * (nely + 1)
                    n7 = n3 + (nelx + 1) * (nely + 1)
                    n8 = n4 + (nelx + 1) * (nely + 1)
                    nodes += [n8, n7, n6, n5]  # second square (top face)
                dofs = []
                for n in nodes:
                    for d in range(elemndof):
                        dofs.append(elemndof * n + d)

                el = (
                    (ez * (nelx * nely) if is_3d else 0) + ex * nely + ey
                )  # element index
                edofMat[el, :] = dofs
    iK = np.kron(edofMat, np.ones((size, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, size))).flatten()

    # Supports
    active_supports_indices = [
        i for i in range(len(params["sdim"])) if params["sdim"][i] != "-"
    ]
    dofs = np.arange(ndof)
    fixed = []
    neighbors = (
        [
            (-1, -1, -1),
            (-1, -1, 1),
            (1, -1, -1),
            (-1, 1, -1),
            (-1, 1, 1),
            (1, -1, 1),
            (1, 1, -1),
            (1, 1, 1),
        ]
        if is_3d
        else [(-1, -1, 0), (-1, 1, 0), (1, -1, 0), (1, 1, 0)]
    )  # diagonals positions
    for i in active_supports_indices:
        x, y = params["sx"][i], params["sy"][i]
        z = params["sz"][i] if is_3d else 0
        voxels = [(x, y, z)] + [
            (x + dx, y + dy, z + dz) for dx, dy, dz in neighbors
        ]  # add fixed supports in the neighborhood to make sure that the mechanism doesn't detach
        for vx, vy, vz in voxels:
            if (
                0 <= vx <= nelx and 0 <= vy <= nely and (not is_3d or 0 <= vz <= nelz)
            ):  # stay inside domain
                node_idx = (
                    ((vz + margin_Z) * (nelx + 1) * (nely + 1) if is_3d else 0)
                    + (vx + margin_X) * (nely + 1)
                    + (vy + margin_Y)
                )
                if "X" in params["sdim"][i]:
                    fixed.append(elemndof * node_idx)
                if "Y" in params["sdim"][i]:
                    fixed.append(elemndof * node_idx + 1)
                if is_3d and "Z" in params["sdim"][i]:
                    fixed.append(elemndof * node_idx + 2)
    free = np.setdiff1d(dofs, fixed)

    # Forces
    din = []
    dinVal = []
    active_iforces_indices = [i for i, d in enumerate(params["fidir"]) if d != "-"]
    nf = len(active_iforces_indices)
    for i in active_iforces_indices:
        din_i = elemndof * (
            ((params["fiz"][i] + margin_Z) * (nelx + 1) * (nely + 1) if is_3d else 0)
            + (params["fix"][i] + margin_X) * (nely + 1)
            + (params["fiy"][i] + margin_Y)
        )
        dinVal_i = params["finorm"][i]
        if "X" in params["fidir"][i]:
            if "←" in params["fidir"][i]:
                dinVal_i = -dinVal_i
        elif "Y" in params["fidir"][i]:
            din_i += 1
            if "↑" in params["fidir"][i]:
                dinVal_i = -dinVal_i
        elif is_3d and "Z" in params["fidir"][i]:
            din_i += 2
            if "<" in params["fidir"][i]:
                dinVal_i = -dinVal_i
        din.append(din_i)
        dinVal.append(dinVal_i)

    # Material
    xPhys_large = np.zeros((nelx, nely, nelz)) if is_3d else np.zeros((nelx, nely))
    if is_3d:
        xPhys_large[
            margin_X : nelx - margin_X,
            margin_Y : nely - margin_Y,
            margin_Z : nelz - margin_Z,
        ] = np.transpose(
            xPhys_initial.reshape(
                nelz - 2 * margin_Z, (nelx - 2 * margin_X) * (nely - 2 * margin_Y)
            ).reshape(nelz - 2 * margin_Z, nelx - 2 * margin_X, nely - 2 * margin_Y),
            (1, 2, 0),
        )
    else:
        xPhys_large[margin_X:-margin_X, margin_Y:-margin_Y] = xPhys_initial.reshape(
            (nelx - 2 * margin_X, nely - 2 * margin_Y)
        )
    xPhys = np.zeros((nel))
    xPhys = xPhys_large.flatten(order="F") if is_3d else xPhys_large.flatten()
    volfrac = np.sum(xPhys) / nel
    u = np.zeros((ndof, nf))
    points = np.zeros((nel, elemndof))
    points_interp = np.zeros((nel, elemndof))
    k = 4  # Steepness
    nominator = (
        (1 + np.exp(-k / 2)) * (1 + np.exp(k / 2)) / (np.exp(k / 2) - np.exp(-k / 2))
    )
    c = -nominator / (1 + np.exp(k / 2))
    delta_disp = (
        params["disp_factor"] / params["disp_iterations"]
        if params["disp_iterations"] > 0
        else 0
    )

    # Yield the initial state as Frame 0
    yield xPhys_initial
    if progress_callback:
        progress_callback(1)

    # Displacement loop
    for it in range(params["disp_iterations"]):
        # Move input force according to previous displacement
        j = 0
        for i in active_iforces_indices:
            din[j] += (
                elemndof * (nelx + 1) * (nely + 1) * int(u[din[j]][j])
                if "Z" in params["fidir"][i]
                else (
                    elemndof * (nely + 1) * int(u[din[j]][j])
                    if "Y" in params["fidir"][i]
                    else elemndof * int(u[din[j]][j])
                )
            )
            j += 1
        # Finite Element Analysis
        f = np.zeros((ndof, nf))
        for i in range(nf):
            f[din[i], i] = dinVal[i]
        sK = (
            (KE.flatten()[np.newaxis]).T
            * (Emin + (xPhys) ** params["penal"] * (Emax - Emin))
        ).flatten(order="F")
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof))
        K = K.tolil() if is_3d else K.tocsc()
        for i in range(nf):
            K[din[i], din[i]] += 0.1  # Artificial spring
        if is_3d:
            keep = np.delete(np.arange(K.shape[0]), fixed)
            K = K[keep, :][:, keep]
            K = K.tocoo()
            K_solve = spmatrix(
                K.data.tolist(), K.row.astype(int).tolist(), K.col.astype(int)
            )
        else:
            K_solve = K[free, :][:, free]
        for i in range(nf):
            if is_3d:
                B = matrix(f[free, i])
                cholmod.linsolve(K_solve, B)
                u[free, i] = np.array(B)[:, 0]
            else:
                u[free, i] = spsolve(K_solve, f[free, i])

        # Move the density via interpolation (the core of the method)
        node_ids = [2, 1, 6, 5, 3, 0, 7, 4] if is_3d else [2, 1, 3, 0]
        ux_nodes = np.sum(
            u[edofMat][:, [elemndof * i + 0 for i in node_ids], :], axis=1
        ) / len(node_ids)
        uy_nodes = -np.sum(
            u[edofMat][:, [elemndof * i + 1 for i in node_ids], :], axis=1
        ) / len(node_ids)
        if is_3d:
            uz_nodes = np.sum(
                u[edofMat][:, [elemndof * i + 2 for i in node_ids], :], axis=1
            ) / len(node_ids)
        # Average over all input forces
        ux = np.mean(ux_nodes, axis=1)
        uy = np.mean(uy_nodes, axis=1)
        if is_3d:
            uz = np.mean(uz_nodes, axis=1)
        # Move the points
        for el in range(nel):
            if is_3d:
                (points_interp[el, 0], points_interp[el, 1], points_interp[el, 2]) = (
                    get3DCoordinates(el, True, nelx, nely, nelz)
                )
            else:
                points_interp[el, 0], points_interp[el, 1] = el // nely, el % nely
            points[el, 0] = points_interp[el, 0] + ux[el] * delta_disp
            points[el, 1] = points_interp[el, 1] + uy[el] * delta_disp
            if is_3d:
                points[el, 2] = points_interp[el, 2] + uz[el] * delta_disp
        xPhys = griddata(points, xPhys, points_interp, method="linear")
        xPhys = np.nan_to_num(xPhys, copy=True, nan=0.0, posinf=None, neginf=None)
        # Threshold density
        xPhys = nominator / (1 + np.exp(-k * (xPhys - 0.5))) + c
        # Normalize density
        xPhys = volfrac * xPhys / (np.sum(xPhys) / nel)
        xPhys = np.clip(xPhys, 0.0, 1.0)  # Ensure values remain within [0, 1]

        # Yield the visible part of the structure for plotting
        xPhys_cropped = (
            xPhys.reshape(nelz, nelx, nely)[
                margin_Z:-margin_Z, margin_X:-margin_X, margin_Y:-margin_Y
            ]
            if is_3d
            else xPhys.reshape(nelx, nely)[margin_X:-margin_X, margin_Y:-margin_Y]
        )
        yield xPhys_cropped.flatten()

        if progress_callback:
            progress_callback(it + 2)


def get3DCoordinates(c, center, nelx, nely, nelz):
    if center:  # center of the element
        x = (c % (nely * nelx)) // nely + 0.5
        y = (c % (nely * nelx)) % nely + 0.5
        z = c // (nely * nelx) + 0.5
    else:  # element index
        x = int((c % (nely * nelx)) // nely)
        y = int((c % (nely * nelx)) % nely)
        z = int(c // (nely * nelx))
    return (x, y, z)
