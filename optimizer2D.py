# A 167 LINE TOPOLOGY OPTIMIZATION CODE BY LUC PREVOST, 2021
from __future__ import division
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors
import matplotlib.pyplot as plt
import time

def optimize(nelx, nely, volfrac, fix, fiy, a1, fox, foy, a2, fo2x, fo2y, a3, s1x, s1y, d1, s2x, s2y, d2, E, nu):
	# Initialization
	rmin = 1.3
	penal = 3.0
	ft = 0
	nel = nelx*nely
	# Print text
	print("Minimum compliance problem with OC")
	print("volfrac: " + str(volfrac) + ", rmin: " + str(rmin) + ", penal: " + str(penal))
	print("Filter method: " + ["Sensitivity based", "Density based"][ft])
	# Max and min stiffness
	Emin = 1e-9
	Emax = 1.0
	# dofs:
	ndof = 2*(nelx+1)*(nely+1)
        # Allocate design variables (as array), initialize and allocate sens.
	x = volfrac*np.ones(nel, dtype=float)
	xold = x.copy()
	xPhys = x.copy()
	g = 0 # must be initialized to use the NGuyen/Paulino OC approach
	dc = np.zeros((nely, nelx), dtype=float)
        # FE: Build the index vectors for the for coo matrix format.
	KE = lk(E, nu)
	edofMat = np.zeros((nel, 8), dtype=int)
	for elx in range(nelx):
		for ely in range(nely):
			el = ely+elx*nely
			n1 = (nely+1)*elx+ely
			n2 = (nely+1)*(elx+1)+ely
			edofMat[el,:] = np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1])
	# Construct the index pointers for the coo format
	iK = np.kron(edofMat, np.ones((8, 1))).flatten()
	jK = np.kron(edofMat, np.ones((1, 8))).flatten()
	# Filter: Build (and assemble) the index+data vectors for the coo matrix format
	nfilter = int(nel*((2*(np.ceil(rmin)-1)+1)**2))
	iH = np.zeros(nfilter)
	jH = np.zeros(nfilter)
	sH = np.zeros(nfilter)
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
					iH[cc] = row
					jH[cc] = col
					sH[cc] = np.maximum(0.0, fac)
					cc = cc+1
	# Finalize assembly and convert to csc format
	H = coo_matrix((sH, (iH, jH)), shape=(nel, nel)).tocsc()
	Hs = H.sum(1)
	# BC's and support
	dofs = np.arange(ndof)
	s1 = 2*(s1y + s1x*(nely+1))
	s2 = 2*(s2y + s2x*(nely+1))
	fixed = np.union1d(np.arange(s1, s1+2), np.arange(s2, s2+2))
	
	free = np.setdiff1d(dofs, fixed)
	# Set load for a FORCE INVERTER
	din = 2*(fiy + fix*(nely+1)) + (a1 == '↑' or a1 == '↓')
	dout = 2*(foy + fox*(nely+1)) + (a2 == '↑' or a2 == '↓')
	if a1 == '↑': dinVal = 4/100
	if a1 == '↓': dinVal = -4/100
	if a1 == '→': dinVal = -4/100
	if a1 == '←': dinVal = 4/100
	if a2 == '↑': doutVal = 4/100
	if a2 == '↓': doutVal = -4/100
	if a2 == '→': doutVal = -4/100
	if a2 == '←': doutVal = 4/100
	Fin = coo_matrix((np.array([dinVal]), (np.array([din]), np.array([0]))), shape=(ndof, 1)).toarray()
	Fout = coo_matrix((np.array([doutVal]), (np.array([dout]), np.array([0]))), shape=(ndof, 1)).toarray()
	f = np.concatenate((Fin, Fout), axis=1)
	# Solution and RHS vectors
	u = np.zeros((ndof, 2))
	# Set loop counter and gradient vectors
	loop = 0
	change = 1
	obj = np.ones(40)
	dv = np.ones(nel)
	dc = np.ones(nel)
	ce = np.ones(nel)
	while change > 0.01 and loop < 5:
		loop = loop+1
		# Setup and solve FE problem
		sK = ((KE.flatten()[np.newaxis]).T*(Emin+(xPhys)**penal*(Emax-Emin))).flatten(order='F')
		K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
		# Add artificial spring resistance to force the mechanism to make a stiff structure
		K[din, din] = K[din, din]+0.1
		K[dout, dout] = K[dout, dout]+0.1
		# Remove constrained dofs from matrix
		K = K[free, :][:, free]
		# Solve system
		u[free, 0]=spsolve(K, f[free, 0])
		u[free, 1]=spsolve(K, f[free, 1])
		# Objective and sensitivity
		obj[loop-1] = u[dout][0]
		dv[:] = np.ones(nel)
		for ely in range(nely):
			for elx in range(nelx):
				n1 = (nely+1)*(elx)+ely+1
				n2 = (nely+1)*(elx+1)+ely+1
				Ue1 = u[[[2*n1], [2*n1+1], [2*n2], [2*n2+1], [2*n2-2], [2*n2-1], [2*n1-2], [2*n1-1]], [0]]
				Ue2 = u[[[2*n1], [2*n1+1], [2*n2], [2*n2+1], [2*n2-2], [2*n2-1], [2*n1-2], [2*n1-1]], [1]]
				ce = np.squeeze(np.dot(np.dot(Ue1.transpose(), KE).reshape(8), Ue2))
				dc[nely*elx+ely] = penal*xPhys[nely*elx+ely]**(penal-1)*ce
		# Sensitivity filtering:
		if ft == 0:
			dc[:] = np.asarray((H*(x*dc))[np.newaxis].T/Hs)[:, 0] / np.maximum(0.001,x)
		elif ft == 1:
			dc[:] = np.asarray(H*(dc[np.newaxis].T/Hs))[:, 0]
			dv[:] = np.asarray(H*(dv[np.newaxis].T/Hs))[:, 0]
		# Optimality criteria
		xold[:] = x
		(x[:],g) = oc(nelx, nely, x, volfrac,dc, dv, g)
		if ft == 0:
			xPhys[:] = x
		elif ft == 1:
			xPhys[:] = np.asarray(H*x[np.newaxis].T/Hs)[:, 0]
		# Compute the change by the inf. norm
		change = np.linalg.norm(x.reshape(nel, 1)-xold.reshape(nel, 1),np.inf)
		print("it.: {0}, obj.: {1:.3f}, Vol.: {2:.3f}, ch.: {3:.3f}".format(\
		loop, obj[loop-1], (g+volfrac*nel)/(nel), change))
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
    l1 = 0.
    l2 = 1e9
    move = 0.05
    Rhomin = 1e-6
    # reshape to perform vector operations
    xnew = np.zeros(nelx*nely)
    while (l2-l1)/(l1+l2) > 1e-4 and l2 > 1e-40:
        lmid = 0.5*(l2+l1)
        xnew[:]= np.maximum(Rhomin, np.maximum(x-move, np.minimum(1.0, np.minimum(x+move, x*np.maximum(1e-10, -dc/dv/lmid)**0.3))))
        gt = g+np.sum((dv*(xnew-x)))
        if gt > 0:
            l1 = lmid
        else:
            l2 = lmid
    return (xnew, gt)
