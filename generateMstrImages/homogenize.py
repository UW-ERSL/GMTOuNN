import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import  csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors

class Homogenization:
  def __init__(self, lx, ly, nelx, nely, phiInDeg, matProp, penal):
    self.lx = lx
    self.ly = ly
    self.cellVolume = self.lx*self.ly
    self.nelx = nelx
    self.nely = nely
    self.dx = lx/nelx
    self.dy = ly/nely
    self.elemSize = np.array([self.dx, self.dy])
    self.numElems = nelx*nely
    self.penal = penal
    self.keLambda, self.keMu, self.feLambda, self.feMu = self.elementMatVec(self.dx/2, self.dy/2, phiInDeg)
    nodenrs = np.reshape(np.arange(0,(1+self.nely)*(1+self.nelx)),(1+self.nelx,1+self.nely))
    edofVec = np.reshape(2*nodenrs[0:-1,0:-1]+2,(self.numElems,1))

    edofMat = np.repeat(edofVec,8 , axis = 1)
    edofMat = edofMat + np.repeat(np.array([0, 1, 2*self.nely+2, 2*self.nely+3,\
                                            2*self.nely, 2*self.nely+1, -2, -1])\
                                  [:,np.newaxis], self.numElems, axis = 1).T

    nn = (self.nelx+1)*(self.nely+1) # Total number of nodes
    nnP = (self.nelx)*(self.nely)   # Total number of unique nodes
    nnPArray = np.reshape(np.arange(0,nnP), (self.nelx, self.nely))
    nnPArray = np.vstack((nnPArray, nnPArray[0,:]))
    nnPArray = np.hstack((nnPArray, nnPArray[:,0][:,np.newaxis]))

    dofVector = np.zeros((2*nn))

    dofVector[0::2] = 2*nnPArray.flatten()
    dofVector[1::2] = 2*nnPArray.flatten()+1

    self.edofMat = dofVector[edofMat]
    self.ndof = 2*nnP

    self.iK = np.kron(self.edofMat,np.ones((8,1))).T.flatten(order='F')
    self.jK = np.kron(self.edofMat,np.ones((1,8))).T.flatten(order='F')

    self.iF = np.tile(self.edofMat,3).T.flatten(order = 'F')
    self.jF = np.tile(np.hstack((np.zeros(8),1*np.ones(8),2*np.ones(8))),self.numElems)

    self.computeChi0()

    if(matProp['type'] == 'lame'):
      self.lam = matProp['lam']
      self.mu = matProp['mu']
    else:
      E = matProp['E']
      nu = matProp['nu']
      lam = E*nu/((1+nu)*(1-2*nu))
      mu = E/(2*(1+nu))
      self.lam = 2*mu*lam/(lam+2*mu)
      self.mu = mu


  def homogenize(self, x):
    x = 1e-3 + x # add eps to ensure non singularity

    self.netLam = self.lam * np.power(x,self.penal)
    self.netMu = self.mu * np.power(x,self.penal)

    CH = np.zeros((3,3))

    objElemAll = np.zeros((self.nely,self.nelx,3,3))

    chi = self.computeDisplacements(x)

    for i in range(3):
      for j in range(3):

        vi = self.chi0[:,:,i] - chi[(self.edofMat+(i)*self.ndof)%self.ndof,\
                                    ((self.edofMat+(i)*self.ndof)//self.ndof)]
        vj = self.chi0[:,:,j] - chi[(self.edofMat+(j)*self.ndof)%self.ndof,\
                                    ((self.edofMat+(j)*self.ndof)//self.ndof)]

        sumLambda = np.multiply(np.dot(vi,self.keLambda),vj)
        sumMu = np.multiply(np.dot(vi,self.keMu),vj)
        sumLambda = np.reshape(np.sum(sumLambda,1), (self.nelx, self.nely)).T
        sumMu = np.reshape(np.sum(sumMu,1), (self.nelx, self.nely)).T



        objElemAll[:,:,i,j] = 1/self.cellVolume*(np.multiply(self.netLam,sumLambda)\
                                                   + np.multiply(self.netMu,sumMu))

        CH[i,j] = np.sum(np.sum(objElemAll[:,:,i,j]))

    return CH

  def computeDisplacements(self, x): 
    # TODO : make this computation faster by using cvxopt package and solver 
    lam = self.netLam
    mu = self.netMu


    tp1 = self.keLambda.flatten(order='F')
    tp2 = lam.flatten(order='F').T
    tp3 = self.keMu.flatten(order='F')
    tp4 = mu.flatten(order='F').T
    tp5 = (np.outer(tp1,tp2) + np.outer(tp3,tp4)).flatten(order='F')
    sK = tp5
    K = csr_matrix((sK,(self.iK,self.jK)),shape=(self.ndof,self.ndof)).tocsc()


    tp1 = self.feLambda.flatten(order='F')
    tp2 = lam.flatten(order='F').T
    tp3 = self.feMu.flatten(order='F')
    tp4 = mu.flatten(order='F').T
    tp5 = (np.outer(tp1,tp2) + np.outer(tp3,tp4)).flatten(order='F')
    sF = tp5
    F = csr_matrix((sF,(self.iF,self.jF)),shape=(self.ndof,3)).tocsc()

    chi = lil_matrix((2*self.numElems,3))
    a = spsolve(K[2:self.ndof,2:self.ndof], F[2:self.ndof,:])

    chi[2:,:] =  a
    chi = chi.tocsr()

    return chi

  def computeChi0(self):
    chi0 = np.zeros((self.numElems, 8, 3))
    chi0_e = np.zeros((8, 3))

    ke = self.keMu + self.keLambda
    fe = self.feMu + self.feLambda

    idx = np.array([2,4,5,6,7])
    chi0_e[idx,:] = np.linalg.solve(ke[np.ix_(idx,idx)], fe[idx,:])
    chi0[:,:,0] = np.kron(chi0_e[:,0], np.ones((self.numElems,1)))
    chi0[:,:,1] = np.kron(chi0_e[:,1], np.ones((self.numElems,1)))
    chi0[:,:,2] = np.kron(chi0_e[:,2], np.ones((self.numElems,1)))

    self.chi0 = chi0
    return chi0

  def elementMatVec(self, a, b, phi):

    CMu = np.diag([2, 2, 1])
    CLambda = np.zeros((3,3))
    CLambda[0:2,0:2] = 1
    xx=np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    yy = xx
    ww = np.array([1,1])

    keLambda = np.zeros((8,8))
    keMu = np.zeros((8,8))
    feLambda = np.zeros((8,3))
    feMu = np.zeros((8,3))
    L = np.zeros((3,4))
    L[0,0] = 1
    L[1,3] = 1
    L[2,1:3] = 1
    for ii in range(2):
      for jj in range(2):
        x = xx[ii]
        y = yy[jj]
        dNx = 0.25*np.array([-(1-y), (1-y), (1+y), -(1+y)])
        dNy = 0.25*np.array([-(1-x), -(1+x), (1+x), (1-x)])
        Nvec = np.hstack((dNx,dNy)).T.reshape((2,4))
        Mtr = np.array([-a, -b, a, -b, a+2*b/np.tan(phi*np.pi/180), b, \
                        2*b/np.tan(phi*np.pi/180)-a, b]).reshape((4,2))
        J = np.dot(Nvec, Mtr)
        detJ = J[0,0]*J[1,1] - J[0,1]*J[1,0]
        invJ = 1/detJ*np.array([J[1,1], -J[0,1], -J[1,0], J[0,0]]).reshape((2,2))
        weight = ww[ii]*ww[jj]*detJ
        G = np.zeros((4,4))
        G[0:2,0:2] = invJ
        G[2:4, 2:4] = invJ
        dN = np.zeros((4,8))
        dN[0,0:8:2] = dNx
        dN[1,0:8:2] = dNy
        dN[2,1:8:2] = dNx
        dN[3,1:8:2] = dNy
        B = np.dot(G, dN)
        B = np.dot(L, B)
        keLambda = keLambda + weight*(np.dot(B.T, np.dot(CLambda, B)))
        keMu = keMu + weight*(np.dot(B.T, np.dot(CMu, B)))
        feLambda = feLambda + weight*(np.dot(B.T, np.dot(CLambda, np.diag([1, 1, 1]))))
        feMu = feMu + weight*(np.dot(B.T, np.dot(CMu, np.diag([1, 1, 1]))))

    return keLambda, keMu, feLambda, feMu

  def plotMicroStructure(self, x):
    fig, ax = plt.subplots()
    ax.imshow(-np.flipud(x.T), cmap='gray',\
                interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
    fig.tight_layout()
    fig.show()
