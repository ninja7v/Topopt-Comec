# A 241 LINE CODE FOR A 3D TOPOLOGY OPTIMIZATION BY LUC PREVOST, 2021
import numpy as np                      # Math
from scipy.sparse import coo_matrix     # Sparse N-dimensional array manipulation
from scipy.sparse.linalg import spsolve # Linear solver

def optimize(nelxyz, volfrac, c, r, v, fx, fy, fz, a, fv, sx, sy, sz, dim, E, nu, ft, rmin, penal, n_it):
    # Initialization
    (nelx, nely, nelz) = (nelxyz[0], nelxyz[1], nelxyz[2]) # Dimensions
    nel = nelx*nely*nelz                  # Total number of element (<230000)
    ndof = 3*(nelx+1)*(nely+1)*(nelz+1)   # Total number of degree of freedom
    (Emin, Emax) = (1e-9, E)              # Min/Max stifness
    x = volfrac*np.ones(nel, dtype=float) # Density field
    (xold, xPhys) = (x.copy(), x.copy())  # Some other density field
    g = 0                                 # To use the NGuyen/Paulino OC approach
    # Element stifness matrix
    KE = lk(E, nu)
    edofMat = np.zeros((nel,24),dtype=int)
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
    # Construct the index pointers for the coo format
    iK = np.kron(edofMat,np.ones((24, 1))).flatten()
    jK = np.kron(edofMat,np.ones((1, 24))).flatten()
    # Filter: Build (and assemble) the index+data vectors for the coo matrix format
    nfilter = int(nel*(2*(np.ceil(rmin)-1)+1)**3)
    (iH, jH, sH) = (np.zeros(nfilter), np.zeros(nfilter), np.zeros(nfilter))
    cc = 0
    for elx in range(nelx):
        for ely in range(nely):
            for elz in range(nelz):
                el1 = elz * nelx*nely + elx*nely + ely
                ii1=int(np.maximum(elx-(np.ceil(rmin)-1),0))
                ii2=int(np.minimum(elx+np.ceil(rmin),nelx))
                jj1=int(np.maximum(ely-(np.ceil(rmin)-1),0))
                jj2=int(np.minimum(ely+np.ceil(rmin),nely))
                kk1=int(np.maximum(elz-(np.ceil(rmin)-1),0))
                kk2=int(np.minimum(elz+np.ceil(rmin),nelz))
                for k in range(kk1,kk2):
                    for i in range(ii1,ii2):
                        for j in range(jj1,jj2):
                            el2 = k * nelx*nely + i*nely + j
                            fac = rmin-np.sqrt(((elx-i)*(elx-i)+(ely-j)*(ely-j)+(elz-k)*(elz-k)))
                            (iH[cc], jH[cc], sH[cc]) = (el1, el2, np.maximum(0.0,fac))
                            cc += 1
    # Finalize assembly and convert to csc format
    H=coo_matrix((sH,(iH,jH)),shape=(nel,nel)).tocsc()
    Hs=H.sum(1)
    # Supports
    dofs = np.arange(ndof)
    fixed = []
    for i in range(len(sx)):
        if dim[i] != '-':
            if dim[i] == 'X' or dim[i] == 'XYZ':
                fixed.append(3*(sy[i] + sx[i]*(nely+1) + sz[i]*(nelx+1)*(nely+1)))
            if dim[i] == 'Y' or dim[i] == 'XYZ':
                fixed.append(3*(sy[i] + sx[i]*(nely+1) + sz[i]*(nelx+1)*(nely+1))+1)
            if dim[i] == 'Z' or dim[i] == 'XYZ':
                fixed.append(3*(sy[i] + sx[i]*(nely+1) + sz[i]*(nelx+1)*(nely+1))+2)
    free = np.setdiff1d(dofs, fixed)
    # Forces
    (xyz, dVal, d) = (0, 0, [])
    for i in range(len(fx)):
        if  a[i] != '-':
            if a[i] == 'X:←' or a[i] == 'X:→': xyz = 1
            elif a[i] == 'Y:↑' or a[i] == 'Y:↓': xyz = 0
            elif a[i] == 'Z:<' or a[i] == 'Z:>': xyz = 2
            d.append(3*(fy[i] + fx[i]*(nely+1) + fz[i]*(nelx+1)*(nely+1)) + xyz)
            if a[i] == 'X:→' or a[i] == 'Y:↑' or a[i] == 'Z:<': dVal = +4/100
            if a[i] == 'X:←' or a[i] == 'Y:↓' or a[i] == 'Z:>': dVal = -4/100
            Fi = coo_matrix((np.array([dVal]), (np.array([d[i]]), np.array([0]))), shape=(ndof, 1)).toarray()
            f = Fi if i == 0 else np.concatenate((f, Fi), axis=1)
    # Set loop counter, gradient and solution vectors
    loop = 0
    change = 1
    obj = np.zeros(n_it) # Loss
    u = np.zeros((ndof,len(fx))) # Solution displacement
    (dv, dc, ce) = (np.ones(nel), np.ones(nel), np.ones(nel))
    # Print text
    print("Minimum compliance problem with OC")
    print("Filter method: " + ["Sensitivity based","Density based"][ft])
    # Optimization loop
    while change > 0.04 and loop < n_it:
        # Set void
        if v != '-':
            for k in range(c[2]-int(r), c[2]+int(r)+1):
                for j in range(c[1]-int(r), c[1]+int(r)+1):
                    for i in range(c[0]-int(r), c[0]+int(r)+1):
                        if 0 <= i < nelx and 0 <= j < nely and 0 <= k < nelz:
                            if v == '○' and (c[0]-i)**2+(c[1]-j)**2+(c[2]-k)**2<=r**2:
                                xPhys[j+i*nely+k*(nelx*nely)] = 0
                            elif v == '□':
                                xPhys[j+i*nely] = 0
        loop += 1
        # Setup and solve FE problem
        sK = ((KE.flatten()[np.newaxis]).T*(Emin+(xPhys)**penal*(Emax-Emin))).flatten(order='F')
        K = coo_matrix((sK,(iK,jK)),shape=(ndof,ndof)).tocsc()
        # Add artificial spring resistance to force the mechanism to make a stiff structure
        K[d[0], d[0]] += fv[0]
        if a[1] != '-':
            K[d[1], d[1]] += fv[1]
        if a[2] != '-':
            K[d[2], d[2]] += fv[2]
        # Remove constrained dofs from matrix
        K = K[free,:][:,free]
        # Solve system
        u[free, 0]=spsolve(K, f[free, 0])
        if a[1] != '-': u[free, 1] = spsolve(K, f[free, 1])
        if a[2] != '-': u[free, 2] = spsolve(K, f[free, 2])
        # Objective and sensitivity
        obj[loop-1] = u[1][0]
        (dc, Ct) = (np.zeros(nel), np.zeros(nel))
        for el in range (nel):
            (elx, ely, elz) = getCoordinates(el, 2, nelx, nely, nelz)
            n1 = elz*(nelx+1)*(nely+1) + elx*(nely+1) + ely
            n2 = n1 + nely+1
            n3 = n1 + (nelx+1)*(nely+1)
            n4 = n3 + nely+1
            coef = [[3*n1+3], [3*n1+4], [3*n1+5], [3*n2+3], [3*n2+4], [3*n2+5],
                    [3*n2],   [3*n2+1], [3*n2+2], [3*n1],   [3*n1+1], [3*n1+2],
                    [3*n3+3], [3*n3+4], [3*n3+5], [3*n4+3], [3*n4+4], [3*n4+5],
                    [3*n4],   [3*n4+1], [3*n4+2], [3*n3],   [3*n3+1], [3*n3+2]]
            Ue1 = u[coef, [0]]
            if a[1] != '-':
                Ue2 = u[coef, [1]]
                ce = np.squeeze(np.dot(np.dot(Ue1.transpose(), KE).reshape(24), Ue2))
            if a[2] != '-':
                Ue3 = u[coef, [2]]
                ce = np.squeeze(np.dot(np.dot(Ue1.transpose(), KE).reshape(24), Ue3))
            if a[1] != '-' and a[2] != '-':
                ce = np.squeeze(np.dot(np.dot(Ue1.transpose(), KE).reshape(24), Ue2))\
                    + np.squeeze(np.dot(np.dot(Ue1.transpose(), KE).reshape(24), Ue3))
            dc[el] = penal*xPhys[el]**(penal-1)*ce
        # Sensitivity filtering:
        if ft==0: # Sensitivity
            dc[:] = np.asarray((H*(x*dc))[np.newaxis].T/Hs)[:,0] / np.maximum(0.001,x)
        elif ft==1: # Density
            dc[:] = np.asarray(H*(dc[np.newaxis].T/Hs))[:,0]
            dv[:] = np.asarray(H*(dv[np.newaxis].T/Hs))[:,0]
        # Optimality criteria
        xold[:]=x
        (x[:],g)=oc(nel,x,volfrac,dc,dv,g)
        # Filter design variables
        if ft==0: xPhys[:]=x
        elif ft==1: xPhys[:]=np.asarray(H*x[np.newaxis].T/Hs)[:,0]
        # Compute the change by the inf. norm
        change=np.linalg.norm(x.reshape(nel,1)-xold.reshape(nel,1),np.inf)
        print("it.: {0} , obj.: {1:.3f}, Vol.: {2:.3f}, ch.: {3:.3f}".format( loop,obj[loop-1],(g+volfrac*nel)/nel,change))
    return xPhys
# Element stiffness matrix
def lk(E, nu):
    E=1
    nu=0.25
    A = np.array([[ 32, 6, -8,  6, -6, 4, 3, -6, -10,  3, -3, -3, -4, -8],
                  [-48, 0,  0,-24, 24, 0, 0,  0,  12,-12,  0, 12, 12, 12]])
    k = 1/72 * np.matmul(A.T,np.array([1,nu]).T)
    # Generate six sub-matrices and then get KE matrix
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
# Optimality criterion
def oc(nel,x,volfrac,dc,dv,g):
    (l1, l2) = (0, 1e9)
    move = 0.05
    Rhomin = 1e-6
    xnew=np.zeros(nel)
    while (l2-l1)/(l1+l2) > 1e-4 and l2 > 1e-40:
        lmid = (l2+l1)/2
        xnew[:] = np.maximum(Rhomin, np.maximum(x-move, np.minimum(1.0, np.minimum(x+move, x*np.maximum(1e-10, -dc/dv/lmid)**0.3))))
        gt = g+np.sum((dv*(xnew-x)))
        if gt > 0 : l1 = lmid
        else: l2 = lmid
    return (xnew,gt)

def deleterowcol(A, delrow, delcol):
    # Assumes that matrix is in symmetric csc form
    m = A.shape[0]
    keep = np.delete(np.arange(0, m), delrow)
    A = A[keep, :]
    keep = np.delete(np.arange(0, m), delcol)
    A = A[:, keep]
    return A
# 1D to 3D coordinates
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