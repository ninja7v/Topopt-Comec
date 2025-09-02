# app/optimizers/optimizer_3d.py
# MIT License - Copyright (c) 2025 Luc Prevost
# A 3D Topology Optimizer

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from typing import List, Tuple, Callable, Optional
from .base_optimizer import oc

def get_element_coordinates(el_idx: int, nelx: int, nely: int) -> Tuple[int, int, int]:
    """Get 3D integer coordinates of an element from its 1D index."""
    elx = int((el_idx % (nely * nelx)) // nely)
    ely = int((el_idx % (nely * nelx)) % nely)
    elz = int(el_idx // (nely * nelx))
    return elx, ely, elz

def lk(E: float, nu: float) -> np.ndarray:
    """Element stiffness matrix for a 3D 8-node brick element."""
    A = np.array([
        [ 32, 6, -8,  6, -6, 4, 3, -6, -10,  3, -3, -3, -4, -8],
        [-48, 0,  0,-24, 24, 0, 0,  0,  12,-12,  0, 12, 12, 12]
    ])
    k = 1/72 * (A.T @ np.array([1, nu]))
    
    K1 = np.array([[k[0],k[1],k[1],k[2],k[4],k[4]],[k[1],k[0],k[1],k[3],k[5],k[6]],
                   [k[1],k[1],k[0],k[3],k[6],k[5]],[k[2],k[3],k[3],k[0],k[7],k[7]],
                   [k[4],k[5],k[6],k[7],k[0],k[1]],[k[4],k[6],k[5],k[7],k[1],k[0]]])
    K2 = np.array([[k[8],k[7],k[11],k[5],k[3],k[6]],[k[7],k[8],k[11],k[4],k[2],k[4]],
                   [k[9],k[9],k[12],k[6],k[3],k[5]],[k[5],k[4],k[10],k[8],k[1],k[9]],
                   [k[3],k[2],k[4],k[1],k[8],k[11]],[k[10],k[3],k[5],k[11],k[9],k[12]]])
    K3 = np.array([[k[5],k[6],k[3],k[8],k[11],k[7]],[k[6],k[5],k[3],k[9],k[12],k[9]],
                   [k[4],k[4],k[2],k[7],k[11],k[8]],[k[8],k[9],k[1],k[5],k[10],k[4]],
                   [k[11],k[12],k[9],k[10],k[5],k[3]],[k[1],k[11],k[8],k[3],k[4],k[2]]])
    K4 = np.array([[k[13],k[10],k[10],k[12],k[9],k[9]],[k[10],k[13],k[10],k[11],k[8],k[7]],
                   [k[10],k[10],k[13],k[11],k[7],k[8]],[k[12],k[11],k[11],k[13],k[6],k[6]],
                   [k[9],k[8],k[7],k[6],k[13],k[10]],[k[9],k[7],k[8],k[6],k[10],k[13]]])
    K5 = np.array([[k[0],k[1],k[7],k[2],k[4],k[3]],[k[1],k[0],k[7],k[3],k[5],k[10]],
                   [k[7],k[7],k[0],k[4],k[10],k[5]],[k[2],k[3],k[4],k[0],k[7],k[1]],
                   [k[4],k[5],k[10],k[7],k[0],k[7]],[k[3],k[10],k[5],k[1],k[7],k[0]]])
    K6 = np.array([[k[13],k[10],k[6],k[12],k[9],k[11]],[k[10],k[13],k[6],k[11],k[8],k[1]],
                   [k[6],k[6],k[13],k[9],k[1],k[8]],[k[12],k[11],k[9],k[13],k[6],k[10]],
                   [k[9],k[8],k[1],k[6],k[13],k[6]],[k[11],k[1],k[8],k[10],k[6],k[13]]])
    KE = E/((nu+1)*(1-2*nu))*np.block([[K1,K2,K3,K4],[K2.T,K5,K6,K3.T],[K3.T,K6,K5.T,K2.T],[K4,K3,K2,K1.T]])
    return KE

def optimize(
    nelxyz: List[int], volfrac: float, vx: List[int], vy: List[int], vz: List[int], vradius: float, vshape: str,
    fx: List[int], fy: List[int], fz: List[int], fdir: List[str], fnorm: List[float],
    sx: List[int], sy: List[int], sz: List[int], sdim: List[str],
    E: float, nu: float, filter_type: int, filter_radius_min: float, penal: float, n_it: int,
    progress_callback: Optional[Callable[[int, float, float], None]] = None
) -> np.ndarray:
    """Performs 3D topology optimization."""
    # Initializations
    nelx, nely, nelz = nelxyz
    nel = nelx * nely * nelz
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
    Emin, Emax = 1e-9, E
    x = volfrac * np.ones(nel, dtype=float)
    xold, xPhys = x.copy(), x.copy()
    g = 0.
    
    # Element stiffness matrix and DOF mapping
    KE = lk(E, nu)
    n_el_nodes = 8
    edofMat = np.zeros((nel, 3 * n_el_nodes), dtype=int)
    for el in range (nel):
        (elx, ely, elz) = getCoordinates(el, 2, nelx, nely, nelz)
        n1 = elz*(nelx+1)*(nely+1) + elx*(nely+1) + ely
        n2 = n1 + nely
        n3 = n1 + (nelx+1)*(nely+1)
        n4 = n3 + nely
        edofMat[el, :] = np.array([3*n1+3,3*n1+4,3*n1+5,3*n2+6,3*n2+7,3*n2+8,
                                   3*n2+3,3*n2+4,3*n2+5,3*n1,  3*n1+1,3*n1+2,
                                   3*n3+3,3*n3+4,3*n3+5,3*n4+6,3*n4+7,3*n4+8,
                                   3*n4+3,3*n4+4,3*n4+5,3*n3,  3*n3+1,3*n3+2])
    iK = np.kron(edofMat, np.ones((24, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 24))).flatten()

    # Filter
    nfilter = int(nel * (2 * (np.ceil(filter_radius_min) - 1) + 1)**3)
    iH, jH, sH = np.zeros(nfilter), np.zeros(nfilter), np.zeros(nfilter)
    cc = 0
    for elz in range(nelz):
        for elx in range(nelx):
            for ely in range(nely):
                el1 = elz * nelx*nely + elx*nely + ely
                k_start, k_end = int(max(elz-(np.ceil(filter_radius_min)-1),0)), int(min(elz+np.ceil(filter_radius_min),nelz))
                i_start, i_end = int(max(elx-(np.ceil(filter_radius_min)-1),0)), int(min(elx+np.ceil(filter_radius_min),nelx))
                j_start, j_end = int(max(ely-(np.ceil(filter_radius_min)-1),0)), int(min(ely+np.ceil(filter_radius_min),nely))
                for k in range(k_start,k_end):
                    for i in range(i_start,i_end):
                        for j in range(j_start,j_end):
                            el2 = k * nelx*nely + i*nely + j
                            dist = np.sqrt((elx-i)**2 + (ely-j)**2 + (elz-k)**2)
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
        node_idx = sz[i]*(nelx+1)*(nely+1) + sx[i]*(nely+1) + sy[i]
        if 'X' in sdim[i]: fixed.append(3 * node_idx)
        if 'Y' in sdim[i]: fixed.append(3 * node_idx + 1)
        if 'Z' in sdim[i]: fixed.append(3 * node_idx + 2)
    free = np.setdiff1d(dofs, np.unique(fixed))
    
    # Forces
    active_forces_indices = [i for i in range(len(fdir)) if fdir[i] != '-']
    nb_forces = len(active_forces_indices)
    d, d_vals, f = [], [], np.zeros((ndof, len(fx)))
    for i in active_forces_indices:
        node_idx = fz[i]*(nelx+1)*(nely+1) + fx[i]*(nely+1) + fy[i]
        d_val = 4 / 100.0
        if 'X' in fdir[i]:
            dof = 3 * node_idx
            if '←' in fdir[i]: d_val = -d_val
        elif 'Y' in fdir[i]:
            dof = 3 * node_idx + 1
            if '↓' in fdir[i]: d_val = -d_val
        elif 'Z' in fdir[i]:
            dof = 3 * node_idx + 2
            if '>' in fdir[i]: d_val = -d_val
        d.append(dof)
        d_vals.append(d_val)
        f[dof, i] = d_val
    
    # Void regions
    active_voids_indices = [i for i in range(1, len(vshape)) if vshape[i] != '-']
    
    # Optimization loop
    loop, change = 0, 1.0
    u = np.zeros((ndof, nb_forces))
    print("3D Optimizer starting...")
    while change > 0.01 and loop < n_it:
        loop += 1
        xold[:] = x

        # Void regions
        for i in active_voids_indices:
            if vshape[i] == '□': # Cube
                for i in range(max(0, vx[i]-vradius[i]), min(nelx, vx[i]+vradius[i])):
                    for j in range(max(0, vy[i]-vradius[i]), min(nely, vy[i]+vradius[i])):
                        for k in range(max(0, vz[i]-vradius[i]), min(nelz, vz[i]+vradius[i])):
                            xPhys[k*(nelx*nely) + i*nely + j] = 1e-6
            elif vshape[i] == '○': # Sphere
                for k in range(nelz):
                    for i in range(nelx):
                        for j in range(nely):
                            if (i-vx[i])**2 + (j-vy[i])**2 + (k-vz[i])**2 <= vradius[i]**2:
                                xPhys[k*(nelx*nely) + i*nely + j] = 1e-6

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
        dc = np.zeros(nel)
        for el in range(nel):
            Ue_in = u[edofMat[el, :], [0]]
            ce_total = 0
            for i in active_forces_indices[1:]:
                Ue_out = u[edofMat[el, :], [i]]
                ce_total += (Ue_in.T @ KE @ Ue_out).item()
            dc[el] = penal * (xPhys[el]**(penal-1)) * ce_total
        
        dv = np.ones(nel)
        if filter_type == 'Sensitivity':
            dc[:] = np.asarray((H*(x*dc))[np.newaxis].T/Hs)[:,0] / np.maximum(0.001,x)
        elif filter_type == 'Density':
            dc[:] = np.asarray(H*(dc[np.newaxis].T/Hs))[:,0]
            dv[:] = np.asarray(H*(dv[np.newaxis].T/Hs))[:,0]

        x, g = oc(nel, x, volfrac, dc, dv, g)
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
            
    print("3D Optimizer finished.")
    return xPhys, u

def getCoordinates(c, isElement, nelx, nely, nelz):
    if isElement==0:
        x = ((c//3)%((nely+1)*(nelx+1)))//(nely+1)
        y = ((c//3)%((nely+1)*(nelx+1)))%(nely+1)
        z = (c//3)//((nely+1)*(nelx+1))
    elif isElement==1:
        x = (c%(nely*nelx))//nely+0.5
        y = (c%(nely*nelx))%nely+0.5
        z = c//(nely*nelx)+0.5
    elif isElement==2:
        x = int((c%(nely*nelx))//nely)
        y = int((c%(nely*nelx))%nely)
        z = int(c//(nely*nelx))
    return(x, y, z)