# app/core/optimizers.py
# MIT License - Copyright (c) 2025 Luc Prevost
# Topology Optimizers.

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from typing import List, Tuple, Callable, Optional

from app.core import initializers

def get_element_coordinates(el_idx: int, nelx: int, nely: int) -> Tuple[int, int, int]:
    """Get 3D integer coordinates of an element from its 1D index."""
    elx = int((el_idx % (nely * nelx)) // nely)
    ely = int((el_idx % (nely * nelx)) % nely)
    elz = int(el_idx // (nely * nelx))
    return elx, ely, elz

def lk(E: float, nu: float, is_3d: bool) -> np.ndarray:
    """Get element stiffness matrix."""
    # Get K
    if is_3d:
        A = np.array([[ 32, 6, -8,  6, -6,  4,  3, -6,-10,  3, -3, -3, -4, -8],
                      [-48, 0,  0,-24, 24,  0,  0,  0, 12,-12,  0, 12, 12, 12]])
        k = 1/72 * (A.T @ np.array([1, nu]))
    else:
        k = np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
    
    # Build KE
    if is_3d:
        K1 = np.array([[k[0], k[1], k[1], k[2], k[4], k[4]],
                       [k[1], k[0], k[1], k[3], k[5], k[6]],
                       [k[1], k[1], k[0], k[3], k[6], k[5]],
                       [k[2], k[3], k[3], k[0], k[7], k[7]],
                       [k[4], k[5], k[6], k[7], k[0], k[1]],
                       [k[4], k[6], k[5], k[7], k[1], k[0]]])
        K2 = np.array([[k[8], k[7], k[11],k[5], k[3], k[6]],
                       [k[7], k[8], k[11],k[4], k[2], k[4]],
                       [k[9], k[9], k[12],k[6], k[3], k[5]],
                       [k[5], k[4], k[10],k[8], k[1], k[9]],
                       [k[3], k[2], k[4], k[1], k[8], k[11]],
                       [k[10],k[3], k[5], k[11],k[9], k[12]]])
        K3 = np.array([[k[5], k[6], k[3], k[8], k[11],k[7]],
                       [k[6], k[5], k[3], k[9], k[12],k[9]],
                       [k[4], k[4], k[2], k[7], k[11],k[8]],
                       [k[8], k[9], k[1], k[5], k[10],k[4]],
                       [k[11],k[12],k[9], k[10],k[5], k[3]],
                       [k[1], k[11],k[8], k[3], k[4], k[2]]])
        K4 = np.array([[k[13],k[10],k[10],k[12],k[9], k[9]],
                       [k[10],k[13],k[10],k[11],k[8], k[7]],
                       [k[10],k[10],k[13],k[11],k[7], k[8]],
                       [k[12],k[11],k[11],k[13],k[6], k[6]],
                       [k[9], k[8], k[7], k[6], k[13],k[10]],
                       [k[9], k[7], k[8], k[6], k[10],k[13]]])
        K5 = np.array([[k[0], k[1], k[7], k[2], k[4], k[3]],
                       [k[1], k[0], k[7], k[3], k[5], k[10]],
                       [k[7], k[7], k[0], k[4], k[10],k[5]],
                       [k[2], k[3], k[4], k[0], k[7], k[1]],
                       [k[4], k[5], k[10],k[7], k[0], k[7]],
                       [k[3], k[10],k[5], k[1], k[7], k[0]]])
        K6 = np.array([[k[13],k[10],k[6], k[12],k[9], k[11]],
                       [k[10],k[13],k[6], k[11],k[8], k[1]],
                       [k[6], k[6], k[13],k[9], k[1], k[8]],
                       [k[12],k[11],k[9], k[13],k[6], k[10]],
                       [k[9], k[8], k[1], k[6], k[13],k[6]],
                       [k[11],k[1], k[8], k[10],k[6], k[13]]])
        KE = E/((nu+1)*(1-2*nu))*np.block([[K1,  K2,  K3,  K4],
                                           [K2.T,K5,  K6,  K3.T],
                                           [K3.T,K6,  K5.T,K2.T],
                                           [K4,  K3,  K2,  K1.T]])
    else:
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

def oc(nel: int, x: np.ndarray, max_change: float, dc: np.ndarray, dv: np.ndarray, g: float) -> Tuple[np.ndarray, float]:
    """
    Optimality Criterion (OC) update scheme.
    
    Args:
        nel: Total number of elements.
        x: Current design variables (densities).
        volfrac: Target volume fraction.
        dc: Sensitivities of the objective function.
        dv: Sensitivities of the volume constraint.
        g: Lagrangian multiplier for the volume constraint.

    Returns:
        A tuple containing the new design variables (xnew) and the updated gt value.
    """
    l1, l2 = 0., 1e9
    rhomin = 1e-6
    xnew = np.zeros(nel)
    
    while (l2 - l1) / (l1 + l2) > 1e-4 and l2 > 1e-40:
        lmid = 0.5 * (l2 + l1)
        # Bisection method to find the Lagrange multiplier
        # This is the OC update rule with move limits
        x_update = x * np.maximum(1e-10, -dc / dv / lmid) ** 0.3
        xnew[:] = np.maximum(rhomin, np.maximum(x - max_change, np.minimum(1.0, np.minimum(x + max_change, x_update))))
        
        gt = g + np.sum(dv * (xnew - x))
        
        if gt > 0:
            l1 = lmid
        else:
            l2 = lmid
    
    return xnew, gt

def optimize(
    nelxyz: List[int], volfrac: float, vx: List[int], vy: List[int], vz: List[int], vradius: float, vshape: str,
    fx: List[int], fy: List[int], fz: List[int], fdir: List[str], fnorm: List[float],
    sx: List[int], sy: List[int], sz: List[int], sdim: List[str],
    E: float, nu: float, init_type: int, filter_type: int, filter_radius_min: float, penal: float, max_change: float, n_it: int,
    progress_callback: Optional[Callable[[int, float, float], None]] = None
) -> np.ndarray:
    '''
    Topology optimization for a compliant mechanism.
    
    Args:
        progress_callback: A function to call with (iteration, objective, change) for UI updates.
    '''
    # Initializations
    nelx, nely, nelz = nelxyz # Number of elements in x, y and z directions
    is_3d = nelz > 0
    nel = nelx * nely * (nelz if is_3d else 1) # Total number of elements
    elemndof = 3 if is_3d else 2 # Degrees of freedom per element
    ndof = elemndof * (nelx + 1) * (nely + 1) * ((nelz + 1) if is_3d else 1) # Total number of degrees of freedom
    Emin, Emax = 1e-9, E # Minimum and maximum Young's modulus
    g = 0. # Lagrangian multiplier for volume constraint
    
    # Element stiffness matrix and DOF mapping
    KE = lk(E, nu, is_3d)
    size = 8 * (elemndof if is_3d else 1)
    edofMat = np.zeros((nel, size), dtype=int)
    for ex in range(nelx):
        for ey in range(nely):
            for ez in range(nelz if is_3d else 1):
                n1 = (ez * (nelx + 1) * (nely + 1) if is_3d else 0) + ex * (nely + 1) + ey
                n2 = n1 + nely + 1
                n3 = n2 + 1
                n4 = n1 + 1
                nodes = [n4, n3, n2, n1] # first square (XY plane, bottom face)
                if is_3d:
                    n5 = n1 + (nelx + 1) * (nely + 1)
                    n6 = n2 + (nelx + 1) * (nely + 1)
                    n7 = n3 + (nelx + 1) * (nely + 1)
                    n8 = n4 + (nelx + 1) * (nely + 1)
                    nodes += [n8, n7, n6, n5] # second square (top face)
                dofs = []
                for n in nodes:
                    for d in range(elemndof):
                        dofs.append(elemndof * n + d)
                
                el = (ez * (nelx * nely) if is_3d else 0) + ex * nely + ey # element index
                edofMat[el, :] = dofs
    iK = np.kron(edofMat, np.ones((size, 1))).flatten() # Kronecker product
    jK = np.kron(edofMat, np.ones((1, size))).flatten() # Kronecker product

    # Filter
    nfilter = int(nel * (2 * (np.ceil(filter_radius_min) - 1) + 1)**elemndof)
    iH, jH, sH = np.zeros(nfilter), np.zeros(nfilter), np.zeros(nfilter)
    cc = 0
    for ez in range(nelz if is_3d else 1):
        for ex in range(nelx):
            for ey in range(nely):
                el1 = (ez * nelx * nely if is_3d else 0) + ex * nely + ey

                k_start = int(max(ez - (np.ceil(filter_radius_min) - 1), 0))
                k_end   = int(min(ez + np.ceil(filter_radius_min), nelz if is_3d else 1))
                i_start = int(max(ex - (np.ceil(filter_radius_min) - 1), 0))
                i_end   = int(min(ex + np.ceil(filter_radius_min), nelx))
                j_start = int(max(ey - (np.ceil(filter_radius_min) - 1), 0))
                j_end   = int(min(ey + np.ceil(filter_radius_min), nely))

                for k in range(k_start, k_end):
                    for i in range(i_start, i_end):
                        for j in range(j_start, j_end):
                            el2 = (k * nelx * nely if is_3d else 0) + i * nely + j
                            dist = np.sqrt((ex - i)**2 + (ey - j)**2 + (ez - k)**2)
                            fac = filter_radius_min - dist
                            if fac > 0:
                                iH[cc], jH[cc], sH[cc] = el1, el2, fac
                                cc += 1
    H = coo_matrix((sH[:cc], (iH[:cc], jH[:cc])), shape=(nel, nel)).tocsc()
    Hs = H.sum(1)
    
    # Supports
    active_supports_indices = [i for i in range(len(sdim)) if sdim[i] != '-']
    dofs = np.arange(ndof)
    fixed = []
    for i in active_supports_indices:
        node_idx = (sz[i]*(nelx+1)*(nely+1) if is_3d else 0) + sx[i]*(nely+1) + sy[i]
        if 'X' in sdim[i]: fixed.append(elemndof * node_idx)
        if 'Y' in sdim[i]: fixed.append(elemndof * node_idx + 1)
        if is_3d and 'Z' in sdim[i]: fixed.append(elemndof * node_idx + 2)
    free = np.setdiff1d(dofs, np.unique(fixed))
    
    # Forces
    active_forces_indices = [i for i in range(len(fdir)) if fdir[i] != '-']
    nb_forces = len(active_forces_indices)
    d, d_vals, f = [], [], np.zeros((ndof, len(fx)))
    for i in active_forces_indices:
        node_idx = (fz[i]*(nelx+1)*(nely+1) if is_3d else 0) + fx[i]*(nely+1) + fy[i]
        d_val = 4 / 100.0
        if 'X' in fdir[i]:
            dof = elemndof * node_idx
            if '←' in fdir[i]: d_val = -d_val
        elif 'Y' in fdir[i]:
            dof = elemndof * node_idx + 1
            if '↑' in fdir[i]: d_val = -d_val
        elif is_3d and'Z' in fdir[i]:
            dof = elemndof * node_idx + 2
            if '>' in fdir[i]: d_val = -d_val
        d.append(dof)
        d_vals.append(d_val)
        f[dof, i] = d_val
    
    # Void regions
    active_voids_indices = [i for i in range(len(vshape)) if vshape[i] != '-']
    
    # Material
    sx_active = np.array(sx)[active_supports_indices]
    sy_active = np.array(sy)[active_supports_indices]
    if is_3d: sz_active = np.array(sz)[active_supports_indices]
    fx_active = np.array(fx)[active_forces_indices]
    fy_active = np.array(fy)[active_forces_indices]
    if is_3d: fz_active = np.array(fz)[active_forces_indices]
    all_x = np.concatenate([fx_active, sx_active])
    all_y = np.concatenate([fy_active, sy_active])
    all_z = np.concatenate([fz_active, sz_active]) if is_3d else np.array([0]*len(all_x))
    x = initializers.initialize_material(init_type, volfrac, nelx, nely, nelz, all_x, all_y, all_z)
    xold = x.copy()
    xPhys = x.copy()
    
    # Optimization loop
    loop, change = 0, 1.0
    u = np.zeros((ndof, nb_forces))
    print("3D Optimizer starting...")
    while change > 0.01 and loop < n_it:
        loop += 1
        xold[:] = x

        # Void regions
        for i in active_voids_indices:
            r = vradius[i]
            if vshape[i] == '□':  # cube/square
                for ez in range(max(0, int(vz[i] - r)), min(nelz, int(vz[i] + r)) if is_3d else 1):
                    for ex in range(max(0, int(vx[i] - r)), min(nelx, int(vx[i] + r))):
                        for ey in range(max(0, int(vy[i] - r)), min(nely, int(vy[i] + r))):
                            if vshape[i] == '□':  # cube/square
                                idx = (ez * nelx * nely if is_3d else 0) + ex * nely + ey
                                xPhys[idx] = 1e-6
            elif vshape[i] == '○':  # sphere/circle
                for ez in range(max(0, int(vz[i] - r)), min(nelz, int(vz[i] + r)) if is_3d else 1):
                    for ex in range(max(0, int(vx[i] - r)), min(nelx, int(vx[i] + r))):
                        for ey in range(max(0, int(vy[i] - r)), min(nely, int(vy[i] + r))):
                            dist2 = (ex - vx[i]) ** 2 + (ey - vy[i]) ** 2 + ((ez - vz[i]) ** 2 if is_3d else 0)
                            if dist2 <= r ** 2:
                                idx = (ez * nelx * nely if is_3d else 0) + ex * nely + ey
                                xPhys[idx] = 1e-6

        # FE analysis
        sK = (KE.flatten()[np.newaxis]).T * (Emin + xPhys**penal * (Emax - Emin))
        K = coo_matrix((sK.flatten(order='F'), (iK, jK)), shape=(ndof, ndof)).tocsc()
        
        for i in range(len(d)):
            if fnorm[i] > 0: K[d[i], d[i]] += fnorm[i]
        
        K_free = K[free, :][:, free]
        for i in active_forces_indices:
            if np.any(f[free, i]):
                u[free, i] = spsolve(K_free, f[free, i])

        # Objective
        obj_val = sum(abs(u[d[i], 0]) for i in active_forces_indices) / nb_forces

        # Filtering
        if is_3d:
            dc = np.zeros(nel)
            for el in range(nel):
                Ue_in = u[edofMat[el, :], [0]]
                ce_total = 0
                for i in active_forces_indices[1:]:  # Start from index 1 to skip a[0]
                    Ue_out = u[edofMat[el, :], [i]]
                    ce_total += (Ue_in.T @ KE @ Ue_out).item()
                dc[el] = penal * (xPhys[el] ** (penal - 1)) * ce_total
        else:
            Ue_in = u[edofMat, 0]  # Shape: (nel, 8)
            ce_total = np.zeros(nel)
            for i in active_forces_indices[1:]:  # Start from index 1 to skip a[0]
                Ue_out = u[edofMat, i]
                ce_total += np.einsum('ij,jk,ik->i', Ue_in, KE, Ue_out)
            dc = penal * (xPhys ** (penal - 1)) * ce_total
        dv = np.ones(nel)
        if filter_type == 'Sensitivity':
            if is_3d:
                dc[:] = np.asarray((H*(x*dc))[np.newaxis].T/Hs)[:,0] / np.maximum(0.001,x)
            else:
                dc = np.asarray((H @ (x * dc)) / Hs.flatten()) / np.maximum(0.001, x)
        elif filter_type == 'Density':
            dc[:] = np.asarray(H*(dc[np.newaxis].T/Hs))[:,0]
            dv[:] = np.asarray(H*(dv[np.newaxis].T/Hs))[:,0]

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
            
    print("Optimizer finished.")
    return xPhys, u