# app/core/displacements.py
# MIT License - Copyright (c) 2025 Luc Prevost
# Linear displacement computation.

import numpy as np
from scipy.sparse import coo_matrix # Provides good N-dimensional array manipulation
from scipy.sparse.linalg import spsolve
from scipy.interpolate import griddata
from cvxopt import matrix, spmatrix, cholmod # Convex optimization
import mcubes # Generate isosurface

def lk(E: float, nu: float, is_3d: bool) -> np.ndarray:
    """Element stiffness matrix for a 3D 8-node brick element."""
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

def single_linear_displacement_2d(u, nelx, nely, disp_factor):
    """
    Computes the deformed mesh grid and a grid pattern for a single-frame plot.
    Returns X, Y meshgrid arrays for plotting.
    """
    nodes_flat = np.arange((nelx + 1) * (nely + 1))
    i_coords = nodes_flat // (nely + 1)
    j_coords = nodes_flat % (nely + 1)

    X_flat = i_coords + u[2 * nodes_flat, 0] * disp_factor
    Y_flat = j_coords - u[2 * nodes_flat + 1, 0] * disp_factor # Use '+' for origin='lower' and '-' for 'upper'

    X = X_flat.reshape((nelx + 1, nely + 1))
    Y = Y_flat.reshape((nelx + 1, nely + 1))

    return X, Y

def single_linear_displacement_3d(xPhys, u, nelx, nely, nelz, disp_factor):
    """
    Computes the deformed 3D mesh using PyMCubes.
    Returns the vertices and triangles of the deformed mesh.
    """
    # 1. Get the element-to-dof mapping
    edofMat = get_edofMat(nelx, nely, nelz)
    
    # 2. Calculate the average displacement for each element
    ux, uy, uz = np.mean(u[edofMat].reshape(-1, 8, 3), axis=1).T

    # 3. Generate the initial mesh (vertices and triangles) from the density field
    density = xPhys.reshape((nelz, nelx, nely))
    # Marching cubes finds the 0.5 isosurface
    vertices, triangles = mcubes.marching_cubes(density, 0.5)

    # 4. Deform the vertices
    # We find the closest element for each vertex and apply its displacement vector
    vertices_moved = vertices.copy()
    
    # Get the integer coordinates of each vertex to find its host element
    elx = vertices[:, 1].astype(int)
    ely = vertices[:, 2].astype(int)
    elz = vertices[:, 0].astype(int)
    
    # Clamp coordinates to be within the valid domain
    np.clip(elx, 0, nelx - 1, out=elx)
    np.clip(ely, 0, nely - 1, out=ely)
    np.clip(elz, 0, nelz - 1, out=elz)
    
    # Calculate the 1D element index for each vertex
    el_indices = elz * (nelx * nely) + elx * nely + ely
    
    # Apply the displacement in a single vectorized operation
    vertices_moved[:, 1] += ux[el_indices] * disp_factor
    vertices_moved[:, 2] += uy[el_indices] * disp_factor
    vertices_moved[:, 0] += uz[el_indices] * disp_factor

    return vertices_moved, triangles

def run_iterative_displacement_2d(params, xPhys_initial, progress_callback=None):
    """
    Performs an iterative FE analysis to simulate displacement.
    Yields the cropped density field for each iteration.
    """
    # Initialization
    nelx_orig, nely_orig = params['nelxyz'][:2]
    delta_disp = params['disp_factor'] / params['disp_iterations'] if params['disp_iterations'] > 0 else 0
    nf = 1 # 1 input force so far
    Emin, Emax = 1e-9, params['E'] # Minimum and maximum Young's modulus

    # Extend domain with margins
    margin_X, margin_Y = nelx_orig // 10, nely_orig // 10 # Margin size=10%
    nelx, nely = nelx_orig + 2 * margin_X, nely_orig + 2 * margin_Y
    ndof, nel = 2 * (nelx + 1) * (nely + 1), nelx * nely

    # Create FE model elements (stiffness matrix, dof mapping)
    KE = lk(params['E'], params['nu'], False)
    edofMat = np.zeros((nel, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx * nely
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edofMat[el, :] = np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3, 2*n2, 2*n2+1, 2*n1, 2*n1+1])
    iK = np.kron(edofMat, np.ones((8, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 8))).flatten()
    
    # Define supports
    active_supports_indices = [i for i in range(len(params['sdim'])) if params['sdim'][i] != '-']
    dofs = np.arange(ndof)
    fixed = []
    neighbors = [(-1, -1), (-1, 1), (1, -1), (1, 1)] # diagonals positions
    for i in active_supports_indices:
        x, y = params['sx'][i], params['sy'][i]
        voxels = [(x, y)] + [(x+dx, y+dy) for dx, dy in neighbors] # add fixed supports in the neighborhood to make sure that the mechanism doesn't detach
        for vx, vy in voxels:
            if 0 <= vx <= nelx and 0 <= vy <= nely: # stay inside domain
                node_idx = (vx + margin_X) * (nely + 1) + (vy + margin_Y)
                if 'X' in params['sdim'][i]:
                    fixed.append(2 * node_idx)
                if 'Y' in params['sdim'][i]:
                    fixed.append(2 * node_idx + 1)
    free = np.setdiff1d(dofs, fixed)
    
    # Define forces
    din = []
    dinVal = []
    for i in range(nf):
        din_i = 2 * ((params['fx'][i] + margin_X) * (nely + 1) + (params['fy'][i] + margin_Y))
        dinVal_i = params['fnorm'][i]
        if 'X' in params['fdir'][i]:
            if '←' in params['fdir'][i]: dinVal_i = -dinVal_i
        elif 'Y' in params['fdir'][i]:
            din_i += 1
            if '↑' in params['fdir'][i]: dinVal_i = -dinVal_i
        din.append(din_i)
        dinVal.append(dinVal_i)

    # Embed the optimized structure into the larger domain
    xPhys2d = np.zeros((nelx, nely))
    xPhys2d[margin_X:-margin_X, margin_Y:-margin_Y] = xPhys_initial.reshape((nelx_orig, nely_orig))
    xPhys = xPhys2d.flatten()
    volfrac = np.sum(xPhys) / nel
    
    u = np.zeros((ndof, 2))
    points, points_interp = np.zeros((nel, 2)), np.zeros((nel, 2))
    k = 4 # steepness
    l = (1+np.exp(-k/2))*(1+np.exp(k/2))/(np.exp(k/2)-np.exp(-k/2))
    c = -l/(1+np.exp(k/2))
    
    # Yield the initial state as Frame 0
    yield xPhys_initial
    if progress_callback:
        progress_callback(1)

    for it in range(params['disp_iterations']):
        # Finite Element Analysis
        for i in range(nf):
            Fi = coo_matrix((np.array([dinVal[i]]), (np.array([din[i]]), np.array([0]))), shape=(ndof, 1)).toarray()
            f = Fi if i == 0 else np.concatenate((f, Fi), axis=1)
        sK = ((KE.flatten()[np.newaxis]).T * (Emin + xPhys**params['penal'] * (Emax - Emin))).flatten(order='F')
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        for i in range(nf):
            K[din[i], din[i]] += 0.1 # Artificial spring
        K_free = K[free, :][:, free]
        for i in range(nf):
            u[free, 0] = spsolve(K_free, f[free, i])
        # Move the density via interpolation (the core of the method)
        ux = (u[edofMat][:, 4, 0] + u[edofMat][:, 2, 0] + u[edofMat][:, 6, 0] + u[edofMat][:, 0, 0]) / 4
        uy = -(u[edofMat][:, 5, 0] + u[edofMat][:, 3, 0] + u[edofMat][:, 7, 0] + u[edofMat][:, 1, 0]) / 4
        for el in range(nel):
            points_interp[el, 0], points_interp[el, 1] = el // nely, el % nely
            points[el, 0] = points_interp[el, 0] + ux[el] * delta_disp
            points[el, 1] = points_interp[el, 1] + uy[el] * delta_disp
        xPhys = griddata(points, xPhys, points_interp, method='linear', fill_value=0.0)
        xPhys = np.nan_to_num(xPhys) # Handle any floating point issues
        # Threshold density
        xPhys = l/(1+np.exp(-k*(xPhys-0.5)))+c
        # Normalize density
        xPhys = volfrac * xPhys / (np.sum(xPhys) / nel)
        xPhys = np.clip(xPhys, 0.0, 1.0) # Ensure values remain within [0, 1]
        
        # Yield the visible part of the structure for plotting
        xPhys_cropped = xPhys.reshape(nelx, nely)[margin_X:-margin_X, margin_Y:-margin_Y]
        yield xPhys_cropped.flatten()
        
        if progress_callback:
            progress_callback(it + 2)

def run_iterative_displacement_3d(params, xPhys_initial, progress_callback=None):
    """
    Performs iterative 3D FE analysis to simulate displacement.
    Yields the cropped density field for each iteration.
    """
    # Initialization
    xPhyst = xPhys_initial
    nelx, nely, nelz = params['nelxyz'][:3]
    # Extend domain with margins
    margin_X = nelx//5
    margin_Y = nely//5
    margin_Z = 1
    nely += 2*margin_Y
    nelx += 2*margin_X
    nelz += 2*margin_Z
    ndof = 3*(nelx+1)*(nely+1)*(nelz+1)
    nel = nelx*nely*nelz
    Emin, Emax = 1e-9, 100.0
    k = 4 # steepness
    nf = 1 # 1 input force so far
    ns = sum(1 for sdim in params['fdir'] if sdim != "-")
    KE = lk(params['E'], params['nu'], True)
    edofMat=np.zeros((nel,24),dtype=int)
    for el in range (nel):
        (elx, ely, elz) = get3DCoordinates(el, 2, nelx, nely, nelz)
        n1 = elz*(nelx+1)*(nely+1) + elx*(nely+1) + ely
        n2 = n1 + nely
        n3 = n1 + (nelx+1)*(nely+1)
        n4 = n3 + nely
        edofMat[el, :] = np.array([3*n1+3,3*n1+4,3*n1+5,3*n2+6,3*n2+7,3*n2+8,
                                   3*n2+3,3*n2+4,3*n2+5,3*n1,  3*n1+1,3*n1+2,
                                   3*n3+3,3*n3+4,3*n3+5,3*n4+6,3*n4+7,3*n4+8,
                                   3*n4+3,3*n4+4,3*n4+5,3*n3,  3*n3+1,3*n3+2])
    iK = np.kron(edofMat,np.ones((24, 1))).flatten()
    jK = np.kron(edofMat,np.ones((1, 24))).flatten()
    
    # Define supports
    active_supports_indices = [i for i in range(len(params['sdim'])) if params['sdim'][i] != '-']
    dofs = np.arange(ndof)
    fixed = []
    neighbors = [(-1, -1, -1), (-1, -1, 1), (1, -1, -1), (-1, 1, -1), (-1, 1, 1), (1, -1, 1), (1, 1, -1), (1, 1, 1)] # diagonals positions
    for i in active_supports_indices:
        x, y, z = params['sx'][i], params['sy'][i], params['sz'][i]
        voxels = [(x, y, z)] + [(x+dx, y+dy, z+dz) for dx, dy, dz in neighbors] # add fixed supports in the neighborhood to make sure that the mechanism doesn't detach
        for vx, vy, vz in voxels:
            if 0 <= vx <= nelx and 0 <= vy <= nely and 0 <= vz <= nelz: # stay inside domain
                node_idx = 3 * ((vz + margin_Z) * (nelx + 1) * (nely + 1) + (vx + margin_X) * (nely+1) + (vy + margin_Y))
                if 'X' in params['sdim'][i]:
                    fixed.append(2 * node_idx)
                if 'Y' in params['sdim'][i]:
                    fixed.append(2 * node_idx + 1)
                if 'Z' in params['sdim'][i]:
                    fixed.append(2 * node_idx + 2)
    free = np.setdiff1d(dofs,fixed)
    
    # Define forces
    din = []
    dinVal = []
    for i in range(nf):
        x, y, z = params['fx'][i], params['fy'][i], params['fz'][i]
        din_i = 3 * ((vz + margin_Z) * (nelx + 1) * (nely + 1) + (vx + margin_X) * (nely+1) + (vy + margin_Y))
        dinVal_i = params['fnorm'][i]
        if 'X' in params['fdir'][i]:
            if '←' in params['fdir'][i]: dinVal_i = -dinVal_i
        elif 'Y' in params['fdir'][i]:
            din_i += 1
            if '↑' in params['fdir'][i]: dinVal_i = -dinVal_i
        elif 'Z' in params['fdir'][i]:
            din_i += 2
            if '<' in params['fdir'][i]: dinVal_i = -dinVal_i
        din.append(din_i)
        dinVal.append(dinVal_i)
    
    xPhys3d = np.zeros((nelx, nely, nelz))
    xPhys3d[margin_X:nelx-margin_X, margin_Y:nely-margin_Y, margin_Z:nelz-margin_Z]=np.transpose(xPhyst.reshape(nelz-2*margin_Z,(nelx-2*margin_X)*(nely-2*margin_Y)).reshape(nelz-2*margin_Z,nelx-2*margin_X,nely-2*margin_Y),(1,2,0))#reshape((nelx-2*margin_X, nely-2*margin_Y, nelz-2*margin_Z))
    xPhys = np.zeros((nel))
    xPhys = xPhys3d.flatten(order='F')
    volfrac = np.sum(xPhys)/nel
    u = np.zeros((ndof, 2))
    points = np.zeros((nel, 3))
    points_interp = np.zeros((nel, 3))
    l = (1+np.exp(-k/2))*(1+np.exp(k/2))/(np.exp(k/2)-np.exp(-k/2))
    c = -l/(1+np.exp(k/2))
    delta_disp = params['disp_factor'] / params['disp_iterations'] if params['disp_iterations'] > 0 else 0
    
    # Yield the initial state as Frame 0
    yield xPhys_initial
    if progress_callback:
        progress_callback(1)
    
    for it in range(params['disp_iterations']):
        # Finite Element Analysis
        for i in range(nf):
            din[i] += 3*(nelx+1)*(nely+1)*int(u[din[i]][0])
        for i in range(nf):
            Fi = coo_matrix((np.array([dinVal[i]]), (np.array([din[i]]), np.array([0]))), shape=(ndof, 1)).toarray()
            f = Fi if i == 0 else np.concatenate((f, Fi), axis=1)
        sK = ((KE.flatten()[np.newaxis]).T*(Emin+(xPhys)**params['penal']*(Emax-Emin))).flatten(order='F')
        K = coo_matrix((sK,(iK,jK)), shape=(ndof, ndof))
        K = K.tolil()
        for i in range(nf):
            K[din[i], din[i]] += 0.1 # Artificial spring
        K = K.tocsr()
        m = K.shape[0]
        keep = np.delete(np.arange(m), fixed)
        K = K[keep, :][:, keep]
        K = K.tocoo()
        K_cvx = spmatrix(K.data.tolist(),
                        K.row.astype(int).tolist(),
                        K.col.astype(int))
        for i in range(nf):
            B = matrix(f[free,i])
            cholmod.linsolve(K_cvx,B)
            u[free,i]=np.array(B)[:,0]
        # Move the density via interpolation (the core of the method)
        ux = (u[edofMat][:,3*2+0,0] + u[edofMat][:,3*1+0,0]+ u[edofMat][:,3*6+0,0]+\
              u[edofMat][:,3*5+0,0] + u[edofMat][:,3*3+0,0] + u[edofMat][:,3*0+0,0]+\
              u[edofMat][:,3*7+0,0]+ u[edofMat][:,3*4+0,0])/8
        uy = -(u[edofMat][:,3*2+1,0] + u[edofMat][:,3*1+1,0]+ u[edofMat][:,3*6+1,0]+\
              u[edofMat][:,3*5+1,0] + u[edofMat][:,3*3+1,0] + u[edofMat][:,3*0+1,0]+\
              u[edofMat][:,3*7+1,0]+ u[edofMat][:,3*4+1,0])/8
        uz = (u[edofMat][:,3*2+2,0] + u[edofMat][:,3*1+2,0]+ u[edofMat][:,3*6+2,0]+\
              u[edofMat][:,3*5+2,0] + u[edofMat][:,3*3+2,0] + u[edofMat][:,3*0+2,0]+\
              u[edofMat][:,3*7+2,0]+ u[edofMat][:,3*4+2,0])/8
        for el in range(nel):
            (points_interp[el, 0], points_interp[el, 1], points_interp[el, 2]) = get3DCoordinates(el, 1, nelx, nely, nelz)
            points[el, 0] = points_interp[el, 0]+ux[el]*delta_disp
            points[el, 1] = points_interp[el, 1]+uy[el]*delta_disp
            points[el, 2] = points_interp[el, 2]+uz[el]*delta_disp
        xPhys = griddata(points,xPhys,points_interp,method='linear')
        xPhys = np.nan_to_num(xPhys, copy=True, nan=0.0, posinf=None, neginf=None)
        # Threshold density
        xPhys = l/(1+np.exp(-k*(xPhys-0.5)))+c
        # Normalize density
        xPhys = volfrac/(np.sum(xPhys)/nel)*xPhys
        xPhys = np.clip(xPhys, 0.0, 1.0) # Ensure values remain within [0, 1]
        
        # Yield the visible part of the structure for plotting
        xPhys_cropped = xPhys.reshape(nelz, nelx, nely)[margin_Z:-margin_Z, margin_X:-margin_X, margin_Y:-margin_Y]
        yield xPhys_cropped.flatten()
        
        if progress_callback:
            progress_callback(it + 2)

def get3DCoordinates(c, isElement, nelx, nely, nelz):
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

def get_edofMat(nelx, nely, nelz):
    """
    Generates the element-to-dof mapping matrix for a 3D case.
    Returns the shuffled DOF indices for all elements.
    """
    nel = nelx * nely * nelz
    
    # Create coordinate grids for element origins
    elz, elx, ely = np.meshgrid(np.arange(nelz), np.arange(nelx), np.arange(nely), indexing='ij')

    # Calculate the 8 node numbers for each element
    n1 = (elz * (nelx + 1) * (nely + 1)) + (elx * (nely + 1)) + ely
    n2 = n1 + (nely + 1)
    n3 = n1 + 1
    n4 = n2 + 1
    n5 = n1 + (nelx + 1) * (nely + 1)
    n6 = n5 + (nely + 1)
    n7 = n5 + 1
    n8 = n6 + 1

    # Assemble the DOFs for all elements
    node_ids = np.stack([n1, n2, n3, n4, n5, n6, n7, n8], axis=-1)
    dof_indices = 3 * np.repeat(node_ids, 3, axis=-1) + np.tile([0, 1, 2], 8)
    
    # Reorder to match the stiffness matrix convention
    # This is a bit tricky, but it maps the nodes correctly
    # The order is: n1, n2, n4, n3, n5, n6, n8, n7
    shuffled_dof_indices = dof_indices.reshape(nel, 8, 3)[:, [0, 1, 3, 2, 4, 5, 7, 6], :].reshape(nel, 24)
    
    return shuffled_dof_indices