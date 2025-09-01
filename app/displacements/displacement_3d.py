# app/analysis/displacement_3d.py
# MIT License - Copyright (c) 2025 Luc Prevost
# Refactored 3D linear and non-linear displacement analysis.

import numpy as np
from scipy.sparse import coo_matrix # Provides good N-dimensional array manipulation
from scipy.interpolate import griddata
from cvxopt import matrix, spmatrix, cholmod # Convex optimization
import mcubes # Generate isosurface

def get_edofMat(nelx, nely, nelz):
    """
    Generates the element-to-dof mapping matrix for a 3D problem.
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

def get_element_displacements(u, edofMat):
    """
    Averages the nodal displacements to get a single displacement vector per element.
    Returns ux, uy, uz arrays for each element.
    """
    # Average the displacements of the 8 nodes for each element
    # This is a vectorized version of the original script's logic
    u_elements = u[edofMat].reshape(-1, 8, 3) # Shape: (nel, 8_nodes, 3_dims)
    
    # Average along the nodes axis (axis=1)
    ux, uy, uz = np.mean(u_elements, axis=1).T
    
    return ux, uy, uz

def single_linear_displacement_3d(xPhys, u, nelx, nely, nelz, disp_factor):
    """
    Computes the deformed 3D mesh using PyMCubes.
    Returns the vertices and triangles of the deformed mesh.
    """
    nel = nelx * nely * nelz
    
    # 1. Get the element-to-dof mapping
    edofMat = get_edofMat(nelx, nely, nelz)
    
    # 2. Calculate the average displacement for each element
    ux, uy, uz = get_element_displacements(u, edofMat)

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

def run_iterative_displacement_3d(params, xPhys_initial, progress_callback=None):
    """
    Generator function that performs iterative 3D FE analysis to simulate displacement.
    Yields the cropped density field for each iteration.
    """
    xPhyst = xPhys_initial
    # Initialization
    nelx, nely, nelz = params['nelxyz'][:3]
    margin_X = nelx//5
    margin_Y = nely//5
    margin_Z = 1
    nely += 2*margin_Y
    nelx += 2*margin_X
    nelz += 2*margin_Z
    ndof = 3*(nelx+1)*(nely+1)*(nelz+1)
    nel = nelx*nely*nelz
    penal = params['penal']
    Emin = 0.1
    Emax = 100.0
    k = 4 # steepness
    nf = 1 # 1 input force so far
    ns = len(params['sx'])
    KE = lk()
    edofMat=np.zeros((nel,24),dtype=int)
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
    iK = np.kron(edofMat,np.ones((24, 1))).flatten()
    jK = np.kron(edofMat,np.ones((1, 24))).flatten()
    dofs = np.arange(ndof)
    s = [3*(margin_Z*(nelx+1)*(nely+1)+margin_X*(nely+1)+margin_Y), 3*(margin_Z*(nelx+1)*(nely+1)+(margin_X+1)*(nely+1)-margin_Y-1),\
         3*(margin_Z*(nelx+1)*(nely+1)+(nely+1-margin_X-1)*(nely+1)+margin_Y), 3*(margin_Z*(nelx+1)*(nely+1)+(nely-margin_X)*(nely+1)-margin_Y-1)]
    for i in range (ns):
        fixed = np.arange(s[0], s[0]+3) if i == 0 else np.union1d(fixed, np.arange(s[i], s[i]+3))
    free = np.setdiff1d(dofs,fixed)
    d = [3*(margin_Z*(nely+1)*(nely+1)+((nely+1)*(nelx+1))//2)+2, ndof-3*(margin_Z*(nely+1)*(nely+1)+((nely+1)*(nelx+1))//2+1)+2]
    dVal = [4/100, -4/100]
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
    n_it = params['disp_iterations']
    delta_disp = params['disp_factor'] / n_it if n_it > 0 else 0
    
    # Yield the initial state as Frame 0
    yield xPhys_initial
    if progress_callback:
        progress_callback(1)
    
    for it in range(n_it):
        # Finite element analysis
        for i in range(nf):
            d[i] += 3*(nelx+1)*(nely+1)*int(u[d[i]][0]*delta_disp)
        for i in range(nf):
            Fi = coo_matrix((np.array([dVal[i]]), (np.array([d[i]]), np.array([0]))), shape=(ndof, 1)).toarray()
            f = Fi if i == 0 else np.concatenate((f, Fi), axis=1)
        sK = ((KE.flatten()[np.newaxis]).T*(Emin+(xPhys)**penal*(Emax-Emin))).flatten(order='F')
        K = coo_matrix((sK,(iK,jK)),shape=(ndof,ndof)).tocsc()
        for i in range(nf):
            K[d[i], d[i]] += 0.1
        K = deleterowcol(K,fixed,fixed).tocoo()
        K = cvxopt.spmatrix(K.data,K.row.astype(int),K.col.astype(int))
        for i in range(nf):
            B = cvxopt.matrix(f[free,i])
            cvxopt.cholmod.linsolve(K,B)
            u[free,i]=np.array(B)[:,0]
        # Move density
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
            (points_interp[el, 0], points_interp[el, 1], points_interp[el, 2]) = getCoordinates(el, 1, nelx, nely, nelz)
            points[el, 0] = points_interp[el, 0]+ux[el]*delta_disp
            points[el, 1] = points_interp[el, 1]+uy[el]*delta_disp
            points[el, 2] = points_interp[el, 2]+uz[el]*delta_disp
        xPhys = griddata(points,xPhys,points_interp,method='linear')
        xPhys = np.nan_to_num(xPhys, copy=True, nan=0.0, posinf=None, neginf=None)
        # Threshold density
        xPhys = l/(1+np.exp(-k*(xPhys-0.5)))+c
        # Normalize density
        xPhys = volfrac/(np.sum(xPhys)/nel)*xPhys
        
        # Yield the visible part of the structure for plotting
        xPhys_cropped = xPhys.reshape(nelx, nely, nelz)[margin_X:-margin_X, margin_Y:-margin_Y, margin_Z:-margin_Z]
        yield xPhys_cropped.flatten()
        
        if progress_callback:
            progress_callback(it + 2)
            
def lk():
    E=1
    nu=0.25
    A = np.array([[ 32, 6, -8,  6, -6, 4, 3, -6, -10,  3, -3, -3, -4, -8],
                  [-48, 0,  0,-24, 24, 0, 0,  0,  12,-12,  0, 12, 12, 12]])
    k = 1/72 * np.matmul(A.T,np.array([1,nu]).T)
    K1 = np.array([[k[0], k[1], k[1], k[2], k[4], k[4]],
                   [k[1], k[0], k[1], k[3], k[5], k[6]],
                   [k[1], k[1], k[0], k[3], k[6], k[5]],
                   [k[2], k[3], k[3], k[0], k[7], k[7]],
                   [k[4], k[5], k[6], k[7], k[0], k[1]],
                   [k[4], k[6], k[5], k[7], k[1], k[0]]])
    K2 = np.array([[k[8], k[7], k[11], k[5], k[3], k[6]],
                   [k[7], k[8], k[11], k[4], k[2], k[4]],
                   [k[9], k[9], k[12], k[6], k[3], k[5]],
                   [k[5], k[4], k[10], k[8], k[1], k[9]],
                   [k[3], k[2], k[4], k[1], k[8], k[11]],
                   [k[10], k[3], k[5], k[11], k[9], k[12]]])
    K3 = np.array([[k[5], k[6], k[3], k[8], k[11], k[7]],
                   [k[6], k[5], k[3], k[9], k[12], k[9]],
                   [k[4], k[4], k[2], k[7], k[11], k[8]],
                   [k[8], k[9], k[1], k[5], k[10], k[4]],
                   [k[11], k[12], k[9], k[10], k[5], k[3]],
                   [k[1], k[11], k[8], k[3], k[4], k[2]]])
    K4 = np.array([[k[13], k[10], k[10], k[12], k[9], k[9]],
                   [k[10], k[13], k[10], k[11], k[8], k[7]],
                   [k[10], k[10], k[13], k[11], k[7], k[8]],
                   [k[12], k[11], k[11], k[13], k[6], k[6]],
                   [k[9], k[8], k[7], k[6], k[13], k[10]],
                   [k[9], k[7], k[8], k[6], k[10], k[13]]])
    K5 = np.array([[k[0], k[1], k[7], k[2], k[4], k[3]],
                   [k[1], k[0], k[7], k[3], k[5], k[10]],
                   [k[7], k[7], k[0], k[4], k[10], k[5]],
                   [k[2], k[3], k[4], k[0], k[7], k[1]],
                   [k[4], k[5], k[10], k[7], k[0], k[7]],
                   [k[3], k[10], k[5], k[1], k[7], k[0]]])
    K6 = np.array([[k[13], k[10], k[6], k[12], k[9], k[11]],
                   [k[10], k[13], k[6], k[11], k[8], k[1]],
                   [k[6], k[6], k[13], k[9], k[1], k[8]],
                   [k[12], k[11], k[9], k[13], k[6], k[10]],
                   [k[9], k[8], k[1], k[6], k[13], k[6]],
                   [k[11], k[1], k[8], k[10], k[6], k[13]]])
    KE = E/((nu+1)*(1-2*nu))*np.block([[K1,   K2,  K3,   K4],
                                       [K2.T, K5,  K6,   K3.T],
                                       [K3.T, K6,  K5.T, K2.T],
                                       [K4,   K3,  K2,   K1.T]])
    return (KE)

def deleterowcol(A, delrow, delcol):
    m = A.shape[0]
    keep = np.delete(np.arange(0, m), delrow)
    A = A[keep, :]
    keep = np.delete(np.arange(0, m), delcol)
    A = A[:, keep]
    return A

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