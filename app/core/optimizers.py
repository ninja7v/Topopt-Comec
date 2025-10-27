# app/core/optimizers.py
# MIT License - Copyright (c) 2025 Luc Prevost
# Topology Optimizers.

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from typing import List, Tuple, Callable, Optional

from app.core import initializers

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
        max_change: Maximum allowed change in design variables per iteration.
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
        
        gt = g + np.sum(dv * (xnew - x)) # Should be near zero for the volume constraint
        
        if gt > 0:
            l1 = lmid
        else:
            l2 = lmid
    
    return xnew, gt

def optimize(
    nelxyz: List[int], volfrac: float, rx: List[int], ry: List[int], rz: List[int], rradius: float, rshape: str, rstate: str,
    fix: List[int], fiy: List[int], fiz: List[int], fidir: List[str], finorm: List[float],
    fox: List[int], foy: List[int], foz: List[int], fodir: List[str], fonorm: List[float],
    sx: List[int], sy: List[int], sz: List[int], sdim: List[str],
    E: float, nu: float, init_type: int, filter_type: int, filter_radius_min: float, penal: float, max_change: float, n_it: int,
    progress_callback: Optional[Callable[[int, float, float], None]] = None
) -> tuple[np.ndarray, np.ndarray]:
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
    r = np.ceil(filter_radius_min)
    iH, jH, sH = [], [], []
    for ez in range(nelz if is_3d else 1):
        for ex in range(nelx):
            for ey in range(nely):
                el1 = (ez * nelx * nely if is_3d else 0) + ex * nely + ey

                k_start = int(max(ez - (r - 1), 0))
                k_end   = int(min(ez + r, nelz if is_3d else 1))
                i_start = int(max(ex - (r - 1), 0))
                i_end   = int(min(ex + r, nelx))
                j_start = int(max(ey - (r - 1), 0))
                j_end   = int(min(ey + r, nely))

                for k in range(k_start, k_end):
                    for i in range(i_start, i_end):
                        for j in range(j_start, j_end):
                            el2 = (k * nelx * nely if is_3d else 0) + i * nely + j
                            dist = np.sqrt((ex - i)**2 + (ey - j)**2 + (ez - k)**2)
                            fac = filter_radius_min - dist
                            if fac > 0:
                                iH.append(el1)
                                jH.append(el2)
                                sH.append(fac)
    H = coo_matrix((np.array(sH), (np.array(iH), np.array(jH))), shape=(nel, nel)).tocsc()
    Hs = H.sum(1)
    
    # Supports
    active_supports_indices = [i for i in range(len(sdim)) if sdim[i] != '-']
    dofs = np.arange(ndof)  
    fixed = []
    for i in active_supports_indices:
        node_idx = (sz[i]*(nelx+1)*(nely+1) if is_3d else 0) + sx[i]*(nely+1) + sy[i]
        if 'X' in sdim[i]:
            fixed.append(elemndof * node_idx)
        if 'Y' in sdim[i]:
            fixed.append(elemndof * node_idx + 1)
        if is_3d and 'Z' in sdim[i]:
            fixed.append(elemndof * node_idx + 2)
    free = np.setdiff1d(dofs, np.unique(fixed))
    
    # Forces
    active_iforces_indices = [i for i in range(len(fidir)) if fidir[i] != '-']
    active_oforces_indices = [i for i in range(len(fodir)) if fodir[i] != '-']
    nb_act_if = len(active_iforces_indices)
    nb_act_of = len(active_oforces_indices)
    fi = np.zeros((ndof, nb_act_if))
    fo = np.zeros((ndof, nb_act_of))
    di = []
    do = []
    for i in active_iforces_indices:
        node_idx = (fiz[i]*(nelx+1)*(nely+1) if is_3d else 0) + fix[i]*(nely+1) + fiy[i]
        d_val = 4 / 100.0 # Arfirtificial spring stiffness
        if 'X' in fidir[i]:
            dof = elemndof * node_idx
            if '←' in fidir[i]: d_val = -d_val
        elif 'Y' in fidir[i]:
            dof = elemndof * node_idx + 1
            if '↑' in fidir[i]: d_val = -d_val
        elif is_3d and'Z' in fidir[i]:
            dof = elemndof * node_idx + 2
            if '>' in fidir[i]: d_val = -d_val
        di.append(dof)
        fi[dof, i] = d_val
    for i in active_oforces_indices:
        node_idx = (foz[i]*(nelx+1)*(nely+1) if is_3d else 0) + fox[i]*(nely+1) + foy[i]
        d_val = 4 / 100.0 # Arfirtificial spring stiffness
        if 'X' in fodir[i]:
            dof = elemndof * node_idx
            if '←' in fodir[i]: d_val = -d_val
        elif 'Y' in fodir[i]:
            dof = elemndof * node_idx + 1
            if '↑' in fodir[i]: d_val = -d_val
        elif is_3d and'Z' in fodir[i]:
            dof = elemndof * node_idx + 2
            if '>' in fodir[i]: d_val = -d_val
        do.append(dof)
        fo[dof, i] = d_val
    
    # Regions
    active_regions_indices = [i for i in range(len(rshape)) if rshape[i] != '-']
    
    # Material
    sx_active = np.array(sx)[active_supports_indices]
    sy_active = np.array(sy)[active_supports_indices]
    if is_3d: sz_active = np.array(sz)[active_supports_indices]
    fix_active = np.array(fix)[active_iforces_indices]
    fiy_active = np.array(fiy)[active_iforces_indices]
    if is_3d: fiz_active = np.array(fiz)[active_iforces_indices]
    fox_active = np.array(fox)[active_oforces_indices]
    foy_active = np.array(foy)[active_oforces_indices]
    if is_3d: foz_active = np.array(foz)[active_oforces_indices]
    all_x = np.concatenate([fix_active, fox_active, sx_active])
    all_y = np.concatenate([fiy_active, foy_active, sy_active])
    all_z = np.concatenate([fiz_active, foz_active, sz_active]) if is_3d else np.array([0]*len(all_x))
    x = initializers.initialize_material(init_type, volfrac, nelx, nely, nelz, all_x, all_y, all_z)
    xold = x.copy()
    xPhys = x.copy()
    
    # Optimization loop
    KE = lk(E, nu, is_3d)
    ui = np.zeros((ndof, nb_act_if))
    uo = np.zeros((ndof, nb_act_of))
    print("3D Optimizer starting...")
    loop, change = 0, 1.0
    while change > 0.01 and loop < n_it:
        loop += 1
        xold[:] = x

        # Regions
        for i in active_regions_indices:
            r = rradius[i]
            if rshape[i] == '□':  # Cube/Square
                for ez in range(max(0, int(rz[i] - r)), min(nelz, int(rz[i] + r)) if is_3d else 1):
                    for ex in range(max(0, int(rx[i] - r)), min(nelx, int(rx[i] + r))):
                        for ey in range(max(0, int(ry[i] - r)), min(nely, int(ry[i] + r))):
                            if rshape[i] == '□':  # Cube/Square
                                idx = (ez * nelx * nely if is_3d else 0) + ex * nely + ey
                                xPhys[idx] = 1e-6 if rstate[i] == 'Void' else 1.0
            elif rshape[i] == '◯':  # Sphere/Circle
                for ez in range(max(0, int(rz[i] - r)), min(nelz, int(rz[i] + r)) if is_3d else 1):
                    for ex in range(max(0, int(rx[i] - r)), min(nelx, int(rx[i] + r))):
                        for ey in range(max(0, int(ry[i] - r)), min(nely, int(ry[i] + r))):
                            dist2 = (ex - rx[i]) ** 2 + (ey - ry[i]) ** 2 + ((ez - rz[i]) ** 2 if is_3d else 0)
                            if dist2 <= r ** 2:
                                idx = (ez * nelx * nely if is_3d else 0) + ex * nely + ey
                                xPhys[idx] = 1e-6 if rstate[i] == 'Void' else 1.0

        # FE analysis
        sK = (KE.flatten()[np.newaxis]).T * (Emin + xPhys**penal * (Emax - Emin))
        K = coo_matrix((sK.flatten(order='F'), (iK, jK)), shape=(ndof, ndof)).tocsc()
        
        for i in range(len(di)):
            if finorm[i] > 0:
                K[di[i], di[i]] += finorm[active_iforces_indices[i]]
        for i in range(len(do)):
            if fonorm[i] > 0:
                K[do[i], do[i]] += fonorm[active_oforces_indices[i]]
        
        K_free = K[free, :][:, free]
        for i in active_iforces_indices:
            if np.any(fi[free, i]):
                ui[free, i] = spsolve(K_free, fi[free, i])
        for i in active_oforces_indices:
            if np.any(fo[free, i]):
                uo[free, i] = spsolve(K_free, fo[free, i])

        # Objective (average displacement in the output forces directions)
        obj_val = 0
        for i in range(len(di)):
            obj_val += (sum(abs(uo[do[i], i]) for i in range(len(do)))) / (nb_act_of)

        # Filtering
        if is_3d:
            dc = np.zeros(nel)
            for el in range(nel):
                ce_total = 0
                for i_in in active_iforces_indices:
                    Ue_in = ui[edofMat[el, :], [i_in]]
                    for i_out in active_oforces_indices:
                        Ue_out = uo[edofMat[el, :], [i_out]]
                        ce_total += (Ue_in.T @ KE @ Ue_out).item()
                dc[el] = penal * (xPhys[el] ** (penal - 1)) * ce_total
        else:
            ce_total = np.zeros(nel)
            for i_in in active_iforces_indices:
                Ue_in = ui[edofMat, i_in]  # Shape: (nel, 8)
                for i_out in active_oforces_indices:
                    Ue_out = uo[edofMat, i_out]  # Shape: (nel, 8)
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
        
        # xPhys update
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
    return xPhys, ui