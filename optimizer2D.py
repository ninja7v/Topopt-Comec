# A 157 LINES CODE FOR A 2D TOPOLOGY OPTIMIZATION BY LUC PREVOST, 2021
import numpy as np                      # Math
from scipy.sparse import coo_matrix     # Sparse N-dimensional array manipulation
from scipy.sparse.linalg import spsolve # Linear solver

def optimize(nelxyz, volfrac, c, r, v, fx, fy, a, fv, sx, sy, dim, E, nu, ft, rmin, penal, n_it):
    # Initializations
    (nelx, nely) = (nelxyz[0], nelxyz[1]) # Dimensions
    nel = nelx*nely                       # Total number of element
    ndof = 2*(nelx+1)*(nely+1)            # Total number of degree of freedom
    (Emin, Emax) = (1e-9, E)              # Min/Max stifness
    x = volfrac*np.ones(nel, dtype=float) # Density field
    (xold, xPhys) = (x.copy(), x.copy())  # Some other density field
    g = 0                                 # To use the NGuyen/Paulino OC approach
    # Element stifness matrix
    KE = lk(E, nu)
    edofMat = np.zeros((nel, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely+elx*nely
            (n1, n2) = ((nely+1)*elx+ely, (nely+1)*(elx+1)+ely)
            edofMat[el,:] = np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1])
    # Index pointers for the coo format
    iK = np.kron(edofMat, np.ones((8, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 8))).flatten()
    # Filter: Build (and assemble) the index+data vectors for the coo matrix format
    nfilter = int(nel*((2*(np.ceil(rmin)-1)+1)**2))
    (iH, jH, sH) = (np.zeros(nfilter), np.zeros(nfilter), np.zeros(nfilter))
    cc = 0
    for i in range(nelx):
        for j in range(nely):
            row = i*nely+j
            kk1 = int(np.maximum(i-(np.ceil(rmin)-1), 0))
            kk2 = int(np.minimum(i+np.ceil(rmin), nelx))
            ll1 = int(np.maximum(j-(np.ceil(rmin)-1), 0))
            ll2 = int(np.minimum(j+np.ceil(rmin), nely))
            for k in range(kk1, kk2):
                for l in range(ll1, ll2):
                    col = k*nely+l
                    fac = rmin-np.sqrt(((i-k)*(i-k)+(j-l)*(j-l)))
                    (iH[cc], jH[cc], sH[cc]) = (row, col, np.maximum(0.0, fac))
                    cc = cc+1
    # Finalize assembly and convert to csc format
    H = coo_matrix((sH, (iH, jH)), shape=(nel, nel)).tocsc()
    Hs = H.sum(1)
    # Supports
    dofs = np.arange(ndof)
    fixed = []
    for i in range(len(sx)):
        if dim[i] != '-':
            if dim[i] == 'X' or dim[i] == 'XYZ':
                fixed.append(2*(sy[i] + sx[i]*(nely+1)))
            if dim[i] == 'Y' or dim[i] == 'XYZ':
                fixed.append(2*(sy[i] + sx[i]*(nely+1))+1)
    free = np.setdiff1d(dofs, fixed)
    # Forces
    (xyz, dVal, d) = (0, 0, [])
    for i in range(len(fx)):
        if  a[i] != '-':
            if a[i] == 'X:???' or a[i] == 'X:???': xyz = 0
            elif a[i] == 'Y:???' or a[i] == 'Y:???': xyz = 1
            d.append(2*(fy[i] + fx[i]*(nely+1)) + xyz)
            if a[i] == 'X:???' or a[i] == 'Y:???': dVal = +4/100
            if a[i] == 'X:???' or a[i] == 'Y:???': dVal = -4/100
            Fi = coo_matrix((np.array([dVal]), (np.array([d[i]]), np.array([0]))), shape=(ndof, 1)).toarray()
            f = Fi if i == 0 else np.concatenate((f, Fi), axis=1)
    # Initializations before loop
    loop = 0                     # Loop counter
    change = 1                   # Change from an iteration to another
    obj = np.zeros(n_it)         # Loss
    u = np.zeros((ndof,len(fx))) # Solution displacement
    (dv, dc, ce) = (np.ones(nel), np.ones(nel), np.ones(nel))
    # Print text
    print("Minimum compliance problem with OC")
    print("Filter method: " + ["Sensitivity based", "Density based"][ft])
    # Opmimization loop
    while change > 0.04 and loop < n_it:
        loop += 1
        # Set void
        if v != '-':
            for j in range(c[1]-int(r), c[1]+int(r)):
                for i in range(c[0]-int(r), c[0]+int(r)):
                    if 0 <= i < nelx and 0 <= j < nely:
                        if v == '???' and (c[0]-i)**2+(c[1]-j)**2<=r**2:
                            xPhys[j+i*nely] = 0
                        elif v == '???':
                            xPhys[j+i*nely] = 0
        # Setup and solve FE problem
        sK = ((KE.flatten()[np.newaxis]).T*(Emin+(xPhys)**penal*(Emax-Emin))).flatten(order='F')
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        # Add artificial spring resistance to force the mechanism to make a stiff structure
        K[d[0], d[0]] += fv[0]
        if a[1] != '-': K[d[1], d[1]] += fv[1]
        if a[2] != '-': K[d[2], d[2]] += fv[2]
        # Remove constrained dofs from matrix
        K = K[free, :][:, free]
        # Solve system
        u[free, 0]=spsolve(K, f[free, 0])
        if a[1] != '-': u[free, 1] = spsolve(K, f[free, 1])
        if a[2] != '-': u[free, 2] = spsolve(K, f[free, 2])
        # Objective and sensitivity
        obj[loop-1] = abs(u[d[1]][0])
        dv[:] = np.ones(nel)
        for ely in range(nely):
            for elx in range(nelx):
                (n1, n2) = ((nely+1)*(elx)+ely+1, (nely+1)*(elx+1)+ely+1)
                Ue1 = u[[[2*n1], [2*n1+1], [2*n2], [2*n2+1], [2*n2-2], [2*n2-1], [2*n1-2], [2*n1-1]], [0]]
                if a[1] != '-':
                    Ue2 = u[[[2*n1], [2*n1+1], [2*n2], [2*n2+1], [2*n2-2], [2*n2-1], [2*n1-2], [2*n1-1]], [1]]
                    ce = np.squeeze(np.dot(np.dot(Ue1.transpose(), KE).reshape(8), Ue2))
                if a[2] != '-':
                    Ue3 = u[[[2*n1], [2*n1+1], [2*n2], [2*n2+1], [2*n2-2], [2*n2-1], [2*n1-2], [2*n1-1]], [2]]
                    ce = np.squeeze(np.dot(np.dot(Ue1.transpose(), KE).reshape(8), Ue3))
                if a[1] != '-' and a[2] != '-':
                    ce = np.squeeze(np.dot(np.dot(Ue1.transpose(), KE).reshape(8), Ue2))\
                        + np.squeeze(np.dot(np.dot(Ue1.transpose(), KE).reshape(8), Ue3))
                dc[nely*elx+ely] = penal*xPhys[nely*elx+ely]**(penal-1)*ce
        # Sensitivity filtering:
        if ft == 0: # Sensitivity
            dc[:] = np.asarray((H*(x*dc))[np.newaxis].T/Hs)[:, 0] / np.maximum(0.001,x)
        elif ft == 1: # Density
            dc[:] = np.asarray(H*(dc[np.newaxis].T/Hs))[:, 0]
            dv[:] = np.asarray(H*(dv[np.newaxis].T/Hs))[:, 0]
        # Optimality criteria
        xold[:] = x
        (x[:],g) = oc(nelx, nely, x, volfrac,dc, dv, g)
        if ft == 0: xPhys[:] = x
        elif ft == 1: xPhys[:] = np.asarray(H*x[np.newaxis].T/Hs)[:, 0]
        # Compute the change by the inf. norm
        change = np.linalg.norm(x.reshape(nel, 1)-xold.reshape(nel, 1),np.inf)
        print("it.: {0}, obj.: {1:.3f}, Vol.: {2:.3f}, ch.: {3:.3f}".format(loop, obj[loop-1], (g+volfrac*nel)/(nel), change))
    return xPhys
# Element stiffness matrix
def lk(E, nu):
    k = np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
    KE = E/(1-nu**2)*np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                               [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                               [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                               [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                               [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                               [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                               [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                               [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]]);
    return (KE)
# Optimality criterion
def oc(nelx, nely, x, volfrac, dc, dv, g):
    (l1, l2) = (0., 1e9)
    move = 0.05
    Rhomin = 1e-6
    xnew = np.zeros(nelx*nely)
    while (l2-l1)/(l1+l2) > 1e-4 and l2 > 1e-40:
        lmid = 0.5*(l2+l1)
        xnew[:]= np.maximum(Rhomin, np.maximum(x-move, np.minimum(1.0, np.minimum(x+move, x*np.maximum(1e-10, -dc/dv/lmid)**0.3))))
        gt = g+np.sum((dv*(xnew-x)))
        if gt > 0: l1 = lmid
        else: l2 = lmid
    return (xnew, gt)
