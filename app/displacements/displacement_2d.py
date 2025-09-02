# app/analysis/displacement_2d.py
# MIT License - Copyright (c) 2025 Luc Prevost
# 2D linear displacement

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import griddata

def lk(E=1.0, nu=0.25):
    """
    Element stiffness matrix for 2D plane stress.
    Returns the 8x8 stiffness matrix for a 4-node square element.
    """
    # Element stiffness matrix coefficients for 2D 4-node square elements 
    k = np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
    KE = E/(1-nu**2)*np.array([[k[0],k[1],k[2],k[3],k[4],k[5],k[6],k[7]],[k[1],k[0],k[7],k[6],k[5],k[4],k[3],k[2]],
                               [k[2],k[7],k[0],k[5],k[6],k[3],k[4],k[1]],[k[3],k[6],k[5],k[0],k[7],k[2],k[1],k[4]],
                               [k[4],k[5],k[6],k[7],k[0],k[1],k[2],k[3]],[k[5],k[4],k[3],k[2],k[1],k[0],k[7],k[6]],
                               [k[6],k[3],k[4],k[1],k[2],k[7],k[0],k[5]],[k[7],k[2],k[1],k[4],k[3],k[6],k[5],k[0]]])
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

def run_iterative_displacement_2d(params, xPhys_initial, progress_callback=None):
    """
    Performs an iterative FE analysis to simulate displacement.
    Yields the cropped density field for each iteration.
    """
    # Initialization from GUI parameters
    nelx_orig, nely_orig = params['nelxyz'][:2]
    penal = params['penal']
    E = params['E']
    nu = params['nu']
    total_disp = params['disp_factor']
    iterations = params['disp_iterations']
    delta_disp = total_disp / iterations if iterations > 0 else 0
    nf = 1 # 1 input force so far

    # Extend domain with margins
    margin_X, margin_Y = nelx_orig // 10, nely_orig // 10 # Margin size=10%
    nelx, nely = nelx_orig + 2 * margin_X, nely_orig + 2 * margin_Y
    ndof, nel = 2 * (nelx + 1) * (nely + 1), nelx * nely

    # Create FE model elements (stiffness matrix, dof mapping)
    KE = lk(E=E, nu=nu)
    edofMat = np.zeros((nel, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx * nely
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edofMat[el, :] = np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3, 2*n2, 2*n2+1, 2*n1, 2*n1+1])
    iK = np.kron(edofMat, np.ones((8, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 8))).flatten()
    
    # Define Boundary Conditions and Forces (from original script)
    dofs = np.arange(ndof)
    fixed_nodes_y_pos = (nely + 1) * margin_X + margin_Y
    fixed = np.union1d(dofs[2*fixed_nodes_y_pos : 2*fixed_nodes_y_pos+2],
                       dofs[2*((nely+1)*(nelx-margin_X)-margin_Y-1) : 2*((nely+1)*(nelx-margin_X)-margin_Y-1)+2])
    free = np.setdiff1d(dofs, fixed)
    
    # The point of force application
    din = [(params['fx'][0] + margin_X) * (nely + 1) + (params['fy'][0] + margin_Y)]
    dinVal = [params['fnorm'][0]]
    for i in range(nf):
        if 'X' in params['fdir'][0]:
            din[i] = 2 * din[i]
            if '←' in params['fdir'][0]: dinVal[i] = -dinVal[i]
        elif 'Y' in params['fdir'][0]:
            din[i] = 2 * din[i] + 1
            if '↑' in params['fdir'][0]: dinVal[i] = -dinVal[i]

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

    for it in range(iterations):
        # Finite Element Analysis
        for i in range(nf):
            Fi = coo_matrix((np.array([dinVal[i]]), (np.array([din[i]]), np.array([0]))), shape=(ndof, 1)).toarray()
            f = Fi if i == 0 else np.concatenate((f, Fi), axis=1)
        sK = ((KE.flatten()[np.newaxis]).T * (1e-9 + xPhys**penal * (E - 1e-9))).flatten(order='F')
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K[din, din] += 0.1 # Artificial spring
        K_free = K[free, :][:, free]
        u[free, 0] = spsolve(K_free, f[free, 0])
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