# A 375 LINE CODE FOR TOPOLOGY OPTIMIZATION OF A FORCE INVERTER BY LUC PREVOST, 2021
from __future__ import division      # To support py2 & py3
import numpy as np
from scipy.sparse import coo_matrix  # Provides good N-dimensional array manipulation
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt      # Plot
# import cvxopt                        # Convex Optimization
# import cvxopt.cholmod
from mpl_toolkits.mplot3d import Axes3D
import time

def optimize(nelx, nely, nelz, volfrac, fix, fiy, fiz, a1, fox, foy, foz, a2, s1x, s1y, s1z, d1, s2x, s2y, s2z, d2, E, nu):
    # Initialization
    rmin = 1.3
    penal = 3.0
    ft = 0
    nel = nelx*nely
    print("Minimum compliance problem with OC")
    print("ndes: " + str(nelx) + " x " + str(nely) + " x " + str(nelz))
    print("volfrac: " + str(volfrac) + ", rmin: " + str(rmin) + ", penal: " + str(penal))
    print("Filter method: " + ["Sensitivity based","Density based"][ft])
    penal = 3
    Emin = 1e-9 # min stifness
    Emax = 1.0 # max stifness
    ndof = 3*(nelx+1)*(nely+1)*(nelz+1) # dofs
    nel = nelx*nely*nelz #number of element<230000
    # Allocate design variables (as array), initialize and allocate sens.
    x = volfrac*np.ones(nel,dtype=float)
    xold = x.copy()
    xPhys = x.copy()
    g = 0 # must be initialized to use the NGuyen/Paulino OC approach
    dc=np.zeros((nelz,nely,nelx), dtype=float)
    KE=lk(E, nu)
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
    # Construct the index pointers for the coo format
    iK = np.kron(edofMat,np.ones((24, 1))).flatten()
    jK = np.kron(edofMat,np.ones((1, 24))).flatten()
    # Filter: Build (and assemble) the index+data vectors for the coo matrix format
    nfilter = int(nel*(2*(np.ceil(rmin)-1)+1)**3)
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
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
                            iH[cc] = el1
                            jH[cc] = el2
                            sH[cc] = np.maximum(0.0,fac)
                            cc += 1
    # Finalize assembly and convert to csc format
    H=coo_matrix((sH,(iH,jH)),shape=(nel,nel)).tocsc()
    Hs=H.sum(1)
    # BC's and support
    dofs = np.arange(ndof)
    s = [0, 3*nely, 3*(nelx+1)*(nely+1)-3, 3*nelx*(nely+1)]
    #s = [0, 3*nely]
    ns = len(s)
    for i in range (ns):
        fixed = np.arange(s[0], s[0]+3) if i == 0 else np.union1d(fixed, np.arange(s[i], s[i]+3))
    free = np.setdiff1d(dofs,fixed)
    # Set load for a FORCE INVERTER (or a keybord key for din=dout)
    shift = nelx//20
    # din = 2*(self.fiy + self.fix*(self.nely+1) + self.fiz*(self.nelx+1)*(self.nely+1)) + (self.a1 == '↑' or self.a1 == '↓')
    # dout = 2*(self.foy + self.fox*(self.nely+1) + self.foz*(self.nelx+1)*(self.nely+1)) + (self.a2 == '↑' or self.a2 == '↓')
    d = [3*(((nely+1)*(nelx+1))//2+shift*(nely+1)*(nelx+1))+2,\
          ndof-3*(((nely+1)*(nelx+1))//2+shift*(nely+1)*(nelx+1)+1)+2] # [din, dout]
    dVal = [4/100, -4/100] # [dinVal, doutVal]
    # Set load for a SWICH
    # d = [3*((nely+1)//2+nelz*(nely+1)*(nelx+1))+2,\
    #      din+3*((nely+1)*(nelx)) # [din, dout]
    # dVal = [-4/10, 4/10] # [dinVal, doutVal]
    # Set load for a GRIPPER
    # d = [3*((nely+1)//2+nelz*(nely+1)*(nelx+1))+2,\
    #       3*((nely+1)//2+(nely+1)*(nelx))+2,\
    #       3*((nelz//2)*(nely+1)*(nelx+1)-(nely+1)//2)-1]
    # dVal = [-4/10, 4/10, -4/10]
    # Set load for a GRIPPER 2D
    # d = [3*(nelz*(nely+1)*(nelx+1))+2,\
    #      3*(nelz//2+1)*(nely+1)-1]
    # dVal = [-4/10, +4/10]
    nf = len(d) # number of force: 1 in & 1 out
    for i in range(nf):
        Fi = coo_matrix((np.array([dVal[i]]), (np.array([d[i]]), np.array([0]))), shape=(ndof, 1)).toarray()
        f = Fi if i == 0 else np.concatenate((f, Fi), axis=1)
    # Solution and RHS vectors
    u = np.zeros((ndof,nf))
    # Conditions
    passive = np.zeros(nel)
    # (xc, yc, zc) = [nelx/2, nely/2, 2*nelz/3]
    # for el in range(nel):
    #     (xe, ye, ze) = getCoordinates(el, 1, nelx, nely, nelz)
    #     if np.sqrt((xe-xc)**2+(ye-yc)**2+(ze-zc)**2) < 5: # circle
    #         passive[el] = 2 # 0:free 1:void 2:material
    # Set loop counter and gradient vectors
    loop = 0
    loop_max = 40
    change = 1
    obj = np.ones(loop_max)
    dv = np.ones(nel)
    dc = np.ones(nel)
    ce = np.ones(nel)
    while change > 0.01 and loop < loop_max:
        t0 = time.time()
        loop += 1
        # Setup and solve FE problem
        sK = ((KE.flatten()[np.newaxis]).T*(Emin+(xPhys)**penal*(Emax-Emin))).flatten(order='F')
        K = coo_matrix((sK,(iK,jK)),shape=(ndof,ndof)).tocsc()
        # Add artificial spring resistance to force the mechanism to make a stiff structure
        for i in range(nf):
            K[d[i], d[i]] += 0.01
        # Solve without cvxopt
        K = K[free,:][:,free]
        u[free,0]=spsolve(K,f[free,0])
        # Remove constrained dofs from matrix and convert to coo
        # K = deleterowcol(K,fixed,fixed).tocoo()
        # # Solve system
        # K = cvxopt.spmatrix(K.data,K.row.astype(np.int),K.col.astype(np.int))
        # for i in range(nf):
        #     B = cvxopt.matrix(f[free,i])
        #     cvxopt.cholmod.linsolve(K,B) # KX=B, on exit B contains the solution
        #     u[free,i]=np.array(B)[:,0]
        # Objective and sensitivity
        obj[loop-1] = u[d[nf-1]][0]
        dc = np.zeros((nel))
        Ct = np.zeros((nel))
        for el in range (nel):
            (elx, ely, elz) = getCoordinates(el, 2, nelx, nely, nelz)
            n1 = elz*(nelx+1)*(nely+1) + elx*(nely+1) + ely #(nely - ely)-1
            n2 = n1 + nely+1
            n3 = n1 + (nelx+1)*(nely+1)
            n4 = n3 + nely+1
            coef = [[3*n1+3], [3*n1+4], [3*n1+5], [3*n2+3], [3*n2+4], [3*n2+5],
                    [3*n2],   [3*n2+1], [3*n2+2], [3*n1],   [3*n1+1], [3*n1+2],
                    [3*n3+3], [3*n3+4], [3*n3+5], [3*n4+3], [3*n4+4], [3*n4+5],
                    [3*n4],   [3*n4+1], [3*n4+2], [3*n3],   [3*n3+1], [3*n3+2]]
            # Ue1 = u[coef, [0]]
            # Ue2 = u[coef, [1]]
            # ce = np.squeeze(np.dot(np.dot(Ue1.transpose(), KE).reshape(24), Ue2))
            # dc[el] = penal*xPhys[el]**(penal-1)*ce
            for i in range(nf-1):
                ce = np.squeeze(np.dot(np.dot(u[coef, [0]].transpose(), KE).reshape(24), u[coef, [i+1]]))
                if loop == loop_max:
                    Ct[el] = ce
                dc[el] += penal*xPhys[el]**(penal-1)*ce
        # Sensitivity filtering:
        if ft==0:
            dc[:] = np.asarray((H*(x*dc))[np.newaxis].T/Hs)[:,0] / np.maximum(0.001,x)
        elif ft==1:
            dc[:] = np.asarray(H*(dc[np.newaxis].T/Hs))[:,0]
            dv[:] = np.asarray(H*(dv[np.newaxis].T/Hs))[:,0]
        # Optimality criteria
        xold[:]=x
        (x[:],g)=oc(nel,x,volfrac,dc,dv,g, passive)
        # Filter design variables
        if ft==0:
            xPhys[:]=x
        elif ft==1:
            xPhys[:]=np.asarray(H*x[np.newaxis].T/Hs)[:,0]
        change=np.linalg.norm(x.reshape(nel,1)-xold.reshape(nel,1),np.inf)
        # Write iteration history to screen (req. Python 2.6 or newer)
        t1 = time.time()
        print("it.: {0} , obj.: {1:.3f}, Vol.: {2:.3f}, ch.: {3:.3f}, time_it.: {4:.3f}".format(\
            loop,obj[loop-1],(g+volfrac*nel)/nel,change, t1-t0))
    # Plot the mechanism
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = []
    y = []
    z = []
    for el in range(nel):
        if xPhys[el] > 0.6:
            (xe, ye, ze) = getCoordinates(el, 1, nelx, nely, nelz)
            x.append(xe)
            y.append(ye)
            z.append(ze)
    ax.scatter(x, y, z, s=6000/min(nelx, nely, nelz), marker='s', c='black')
    # Plot forces
    # for i in range(nf):
    #     (xf, yf, zf) = getCoordinates(d[i], 0, nelx, nely, nelz)
    #     if d[i]%3==0:
    #         ax.quiver(xf, yf, zf, np.sign(dVal[i])*nelx//2, 0, 0, linewidths=10, color = ['m'])
    #     elif d[i]%3==1:
    #         ax.quiver(xf, yf, zf, 0, np.sign(dVal[i])*nely//2, 0, linewidths=10, color = ['m'])
    #     elif d[i]%3==2:
    #         ax.quiver(xf, yf, zf, 0, 0, np.sign(dVal[i])*nelz//2, linewidths=10, color = ['m'])
    # Plot supports
    # for i in range(ns):
    #     (xs, ys, zs) = getCoordinates(s[i], 0, nelx, nely, nelz)
    #     ax.scatter([xs], [ys], [zs], s=1000/max(nelx, nely, nelz)+10, marker='s', c='blue')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title('3D compliant force inverter\
        \n it.: {0} , Vol.: {1:.3f}, penal.:{2}, rmin:{3}'.format(loop, (g+volfrac*nel)/nel, penal, rmin), pad = 10)
    plt.show()
    # Save figure
    fig.savefig('force_inverter_3D.png', bbox_inches='tight', dpi=200)
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
def oc(nel,x,volfrac,dc,dv,g, passive):
    l1 = 0
    l2 = 1e9
    move = 0.05
    Rhomin = 1e-6
    # reshape to perform vector operations
    xnew=np.zeros(nel)
    while (l2-l1)/(l1+l2) > 1e-4 and l2 > 1e-40:
        lmid = (l2+l1)/2
        xnew[:] = np.maximum(Rhomin, np.maximum(x-move, np.minimum(1.0, np.minimum(x+move, x*np.maximum(1e-10, -dc/dv/lmid)**0.3))))
        # for el in range(nel):
        #     if passive[el] == 1: x[el] = 0
        #     elif passive[el] == 2: x[el] = 1
        gt = g+np.sum((dv*(xnew-x)))
        if gt > 0 :
            l1 = lmid
        else:
            l2 = lmid
    return (xnew,gt)

def deleterowcol(A, delrow, delcol):
    # Assumes that matrix is in symmetric csc form !
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