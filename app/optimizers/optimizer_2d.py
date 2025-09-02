# app/optimizers/optimizer_2d.py
# MIT License - Copyright (c) 2025 Luc Prevost
# A 2D Topology Optimizer

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from typing import List, Callable, Optional
from .base_optimizer import oc

def lk(E: float, nu: float) -> np.ndarray:
    """Element stiffness matrix for 2D plane stress."""
    k = np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
    KE = E/(1-nu**2)*np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
    ])
    return KE

def optimize(
    nelxyz: List[int], volfrac: float, vx: List[int], vy: List[int], vradius: List[int], vshape: List[str],
    fx: List[int], fy: List[int], fdir: List[str], fnorm: List[float],
    sx: List[int], sy: List[int], sdim: List[str],
    E: float, nu: float, filter_type: str, filter_radius_min: float, penal: float, max_change: float, n_it: int,
    progress_callback: Optional[Callable[[int, float, float], None]] = None
) -> np.ndarray:
    """
    Performs 2D topology optimization for a compliant mechanism.

    Args:
        progress_callback: A function to call with (iteration, objective, change) for UI updates.
    """
    # Initializations
    nelx, nely = nelxyz[0], nelxyz[1] # Number of elements in x and y directions
    nel = nelx * nely # Total number of elements
    ndof = 2 * (nelx + 1) * (nely + 1) # Total number of degrees of freedom
    Emin, Emax = 1e-9, E # Minimum and maximum Young's modulus
    x = volfrac * np.ones(nel, dtype=float) # Initial design variables (densities)
    xold = x.copy()
    xPhys = x.copy()
    g = 0. # Lagrangian multiplier for volume constraint
    
    # Element stiffness matrix
    KE = lk(E, nu)
    edofMat = np.zeros((nel, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx * nely
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edofMat[el, :] = [2*n1+2, 2*n1+3, # top left
                              2*n2+2, 2*n2+3, # top right
                              2*n2, 2*n2+1, # bottom right
                              2*n1, 2*n1+1] # bottom left
    iK = np.kron(edofMat, np.ones((8, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 8))).flatten()
    
    # Filter
    nfilter = int(nel * (2 * (np.ceil(filter_radius_min) - 1) + 1)**2)
    iH, jH, sH = np.zeros(nfilter), np.zeros(nfilter), np.zeros(nfilter)
    cc = 0
    for i in range(nelx):
        for j in range(nely):
            row = i * nely + j
            k_start, k_end = int(max(i - (np.ceil(filter_radius_min) - 1), 0)), int(min(i + np.ceil(filter_radius_min), nelx))
            l_start, l_end = int(max(j - (np.ceil(filter_radius_min) - 1), 0)), int(min(j + np.ceil(filter_radius_min), nely))
            for k in range(k_start, k_end):
                for l in range(l_start, l_end):
                    col = k * nely + l
                    fac = filter_radius_min - np.sqrt((i-k)**2 + (j-l)**2)
                    if fac > 0:
                        iH[cc], jH[cc], sH[cc] = row, col, fac
                        cc += 1
    H = coo_matrix((sH[:cc], (iH[:cc], jH[:cc])), shape=(nel, nel)).tocsc()
    Hs = H.sum(1)
    
    # Supports
    active_supports_indices = [i for i in range(len(sdim)) if sdim[i] != '-']
    dofs = np.arange(ndof)
    fixed = []
    for i in active_supports_indices:
        node_idx = sy[i] + sx[i] * (nely + 1)
        if 'X' in sdim[i]: fixed.append(2 * node_idx)
        if 'Y' in sdim[i]: fixed.append(2 * node_idx + 1)
    free = np.setdiff1d(dofs, fixed)
    
    # Forces
    active_forces_indices = [i for i in range(len(fdir)) if fdir[i] != '-']
    nb_forces = len(active_forces_indices)
    f = np.zeros((ndof, nb_forces))
    d, d_vals = [], []
    for i in active_forces_indices:
        node_idx = fy[i] + fx[i] * (nely + 1)
        d_val = 4.0 / 100.0
        if 'X' in fdir[i]:
            dof = 2 * node_idx
            if '←' in fdir[i]: d_val = -d_val
        elif 'Y' in fdir[i]:
            dof = 2 * node_idx + 1
            if '↑' in fdir[i]: d_val = -d_val
        d.append(dof)
        d_vals.append(d_val)
        f[dof, i] = d_val
    
    # Void regions
    active_voids_indices = [i for i in range(len(vshape)) if vshape[i] != '-']
    
    # Optimization loop
    loop, change = 0, 1.0
    u = np.zeros((ndof, nb_forces))
    print("2D Optimizer starting...")
    while change > 0.01 and loop < n_it:
        loop += 1
        xold[:] = x
        
        # Void regions
        for i in active_voids_indices:
            nelx, nely = nelxyz[0], nelxyz[1]
            if vshape[i] == '□':  # Square
                x_min, x_max = max(0, int(vx[i] - vradius[i])), min(nelx, int(vx[i] + vradius[i]))
                y_min, y_max = max(0, int(vy[i] - vradius[i])), min(nely, int(vy[i] + vradius[i]))
                
                idx_x = np.arange(x_min, x_max)
                idx_y = np.arange(y_min, y_max)
                if len(idx_x) > 0 and len(idx_y) > 0:
                    xx, yy = np.meshgrid(idx_x, idx_y, indexing='ij')
                    indices = (yy + xx * nely).flatten()
                    xPhys[indices] = 1e-6
            elif vshape[i] == '○':  # Circle
                x_min, x_max = max(0, int(vx[i] - vradius[i])), min(nelx, int(vx[i] + vradius[i]) + 1)
                y_min, y_max = max(0, int(vy[i] - vradius[i])), min(nely, int(vy[i] + vradius[i]) + 1)
                
                idx_x = np.arange(x_min, x_max)
                idx_y = np.arange(y_min, y_max)
                if len(idx_x) > 0 and len(idx_y) > 0:
                    i_grid, j_grid = np.meshgrid(idx_x, idx_y, indexing='ij')
                    mask = (i_grid - vx[i])**2 + (j_grid - vy[i])**2 <= vradius[i]**2
                    ii, jj = i_grid[mask], j_grid[mask]
                    indices = jj + ii * nely
                    xPhys[indices] = 1e-6

        # FE analysis
        sK = (KE.flatten()[np.newaxis]).T * (Emin + xPhys**penal * (Emax - Emin))
        K = coo_matrix((sK.flatten(order='F'), (iK, jK)), shape=(ndof, ndof)).tocsc()
        
        for i in range(len(d)):
            if fnorm[i] > 0: K[d[i], d[i]] += fnorm[i]
        
        K_free = K[free, :][:, free]
        for i in active_forces_indices:
            if np.any(f[free, i]):
                u[free, i] = spsolve(K_free, f[free, i])

        Ue_in = u[edofMat, 0]  # Shape: (nel, 8)
        ce_total = np.zeros(nel)
        
        # Calculate sensitivities for each output force using einsum for speed
        for i in active_forces_indices[1:]:  # Start from index 1 to skip a[0]
            Ue_out = u[edofMat, i]
            ce_total += np.einsum('ij,jk,ik->i', Ue_in, KE, Ue_out)
        
        # Objective
        obj_val = sum(abs(u[d[i], 0]) for i in active_forces_indices) / nb_forces
        
        # Filtering
        dc = penal * (xPhys ** (penal - 1)) * ce_total
        dv = np.ones(nel)
        if filter_type == 'Sensitivity':
            dc = np.asarray((H @ (x * dc)) / Hs.flatten()) / np.maximum(0.001, x)
        elif filter_type == 'Density':
            dc[:] = np.asarray(H*(dc[np.newaxis].T/Hs))[:, 0]
            dv[:] = np.asarray(H*(dv[np.newaxis].T/Hs))[:, 0]
        
        # OC update
        x, g = oc(nel, x, max_change, dc, dv, g)
        
        # Filter design variables
        if filter_type == 'Sensitivity':
            xPhys = x
        elif filter_type == 'Density':
            xPhys = np.asarray(H @ x / Hs.flatten())
        
        change = np.linalg.norm(x - xold, np.inf)

        print(f"It.: {loop:3d}, Obj.: {obj_val:.4f}, Vol.: {xPhys.mean():.3f}, Ch.: {change:.3f}")
        if progress_callback:
            should_stop = progress_callback(loop, obj_val, change, xPhys)
            if should_stop:
                print("Optimization stopped by user.")
                break

    print("2D Optimizer finished.")
    return xPhys, u