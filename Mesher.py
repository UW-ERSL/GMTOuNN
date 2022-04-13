import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import matplotlib

class RectangularGridMesher:
  #--------------------------#
  def __init__(self, ndim, nelx, nely, elemSize, bcSettings):
    self.meshType = 'gridMesh'
    self.ndim = ndim
    self.nelx = nelx
    self.nely = nely
    self.elemSize = elemSize
    self.bcSettings = bcSettings
    self.numElems = self.nelx*self.nely
    self.elemArea = self.elemSize[0]*self.elemSize[1]*\
                    jnp.ones((self.numElems)) # all same areas for grid
    self.totalMeshArea = jnp.sum(self.elemArea)
    self.numNodes = (self.nelx+1)*(self.nely+1)
    self.nodesPerElem = 4 # grid quad mesh
    self.ndof = self.bcSettings['dofsPerNode']*self.numNodes
    self.edofMat, self.nodeIdx, self.elemNodes, self.nodeXY, self.bb = \
                                    self.getMeshStructure()
    self.elemCenters = self.generatePoints()
    self.processBoundaryCondition()
    self.BMatrix = self.getBMatrix(0., 0.)
    self.fig, self.ax = plt.subplots()
    self.bb = {'xmax':self.nelx*self.elemSize[0], 'xmin':0.,\
               'ymax':self.nely*self.elemSize[1], 'ymin':0.}
  #--------------------------#
  def getBMatrix(self, xi, eta):
    dx, dy = self.elemSize[0], self.elemSize[1]
    B = np.zeros((3,8))
    r1 = np.array([(2.*(eta/4. - 1./4.))/dx, -(2.*(eta/4. - 1./4))/dx,\
                    (2.*(eta/4. + 1./4))/dx,\
                    -(2.*(eta/4. + 1./4))/dx]).reshape(-1)
    r2 = np.array([(2.*(xi/4. - 1./4))/dy, -(2.*(xi/4. + 1./4))/dy,\
                    (2.*(xi/4. + 1./4))/dy, -(2.*(xi/4. - 1./4))/dy])
    
    B = [[r1[0], 0., r1[1], 0., r1[2], 0., r1[3], 0.],\
          [0., r2[0], 0., r2[1], 0., r2[2], 0., r2[3]],\
          [r2[0], r1[0], r2[1], r1[1], r2[2], r1[2], r2[3], r1[3]]]

    return jnp.array(B)
  #--------------------------#
  def getMeshStructure(self):
    # returns edofMat: array of size (numElemsX8) with
    # the global dof of each elem
    # idx: A tuple informing the position for assembly of computed entries
    n = self.bcSettings['dofsPerNode']*self.nodesPerElem;
    edofMat=np.zeros((self.nelx*self.nely,n),dtype=int)
    if(self.bcSettings['dofsPerNode'] == 2): # as in structural
      for elx in range(self.nelx):
        for ely in range(self.nely):
          el = ely+elx*self.nely
          n1=(self.nely+1)*elx+ely
          n2=(self.nely+1)*(elx+1)+ely
          edofMat[el,:]=np.array([2*n1+2, 2*n1+3, 2*n2+2,\
                          2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1])
              
    elif(self.bcSettings['dofsPerNode'] == 1): # as in thermal
      nodenrs = np.reshape(np.arange(0, self.ndof), \
                           (1+self.nelx, 1+self.nely)).T
      edofVec = np.reshape(nodenrs[0:-1,0:-1]+1, \
                           self.numElems,'F')
      edofMat = np.matlib.repmat(edofVec,4,1).T + \
            np.matlib.repmat(np.array([0, self.nely+1, self.nely, -1]),\
                             self.numElems,1)
    
    iK = tuple(np.kron(edofMat,np.ones((n,1))).flatten().astype(int))
    jK = tuple(np.kron(edofMat,np.ones((1,n))).flatten().astype(int))
    nodeIdx = jax.ops.index[iK,jK]


    elemNodes = np.zeros((self.numElems, self.nodesPerElem))
    for elx in range(self.nelx):
      for ely in range(self.nely):
        el = ely+elx*self.nely
        n1=(self.nely+1)*elx+ely
        n2=(self.nely+1)*(elx+1)+ely
        elemNodes[el,:] = np.array([n1+1, n2+1, n2, n1])
    bb = {}
    bb['xmin'],bb['xmax'],bb['ymin'],bb['ymax'] = \
        0., self.nelx*self.elemSize[0],\
        0., self.nely*self.elemSize[1]
        
    nodeXY = np.zeros((self.numNodes, 2))
    ctr = 0
    for i in range(self.nelx+1):
      for j in range(self.nely+1):
        nodeXY[ctr,0] = self.elemSize[0]*i
        nodeXY[ctr,1] = self.elemSize[1]*j
        ctr += 1
            
    return edofMat, nodeIdx, elemNodes, nodeXY, bb
  #--------------------------#

  def generatePoints(self, res=1):
    # args: Mesh is dictionary containing nelx, nely, elemSize...
    # res is the number of points per elem
    # returns an array of size (numpts X 2)
    xy = np.zeros((res**2*self.numElems, 2))
    ctr = 0
    for i in range(res*self.nelx):
      for j in range(res*self.nely):
        xy[ctr, 0] = self.elemSize[0]*(i + 0.5)/res
        xy[ctr, 1] = self.elemSize[1]*(j + 0.5)/res
        ctr += 1
    return xy
  #--------------------------#
  def processBoundaryCondition(self):
    force = np.zeros((self.ndof,1))
    dofs=np.arange(self.ndof)
    fixed = dofs[self.bcSettings['fixedNodes']]
    free = np.setdiff1d(np.arange(self.ndof), fixed)
    force[self.bcSettings['forceNodes']] = self.bcSettings['forceMagnitude']
    self.bc = {'force':force, 'fixed':fixed,'free':free}
  #--------------------------#
  def plotFieldOnMesh(self, field, titleStr):
    plt.ion(); plt.clf()
    plt.imshow(-np.flipud(field.reshape((self.nelx,self.nely)).T), \
               cmap='gray', interpolation='none')
    plt.axis('Equal')
    plt.grid(False)
    plt.title(titleStr)
    plt.pause(0.01)
    self.fig.canvas.draw()
#--------------------------#
class UnstructuredMesher:
  def __init__(self, bcFiles):
    self.bcFiles = bcFiles
    self.meshProp = {}
    self.meshType = 'unstructuredMesh'
    self.ndim = 2 # 2D structures
    self.dofsPerNode = 2 # structural

    self.readMeshData()
    self.fig, self.ax = plt.subplots()
  #-----------------------#
  def readMeshData(self):
    # Only structural mesh 
    self.bc = {}
    self.nodesPerElem = 4 # grid quad mesh
    # Force
    with open(self.bcFiles['forceFile']) as f:
      self.bc['force'] = np.array([float(line.rstrip()) for line in f]).reshape(-1,1)
    self.ndof  = self.bc['force'].shape[0]
    self.numNodes = int(0.5*self.ndof) # structural
    # Fixed
    with open(self.bcFiles['fixedFile']) as f:
      self.bc['fixed'] = np.array([int(line.rstrip()) for line in f]).reshape(-1)
    self.bc['free'] = np.setdiff1d(np.arange(self.ndof),self.bc['fixed'])
        
    # Node XY
    self.nodeXY = np.zeros((self.numNodes,self.ndim))
    ctr = 0
    f = open(self.bcFiles['nodeXYFile'])
    for line in f:
      self.nodeXY[ctr,:] = line.rstrip().split('\t')
      ctr += 1
    
    # edofMat
    ctr = 0
    f = open(self.bcFiles['elemNodesFile'])
    self.numElems = int(f.readline().rstrip())
    self.elemSize = np.zeros((2))
    self.elemSize[0], self.elemSize[1] = \
      f.readline().rstrip().split('\t')
    self.elemArea = self.elemSize[0]*self.elemSize[1]*\
                    jnp.ones((self.numElems)) # all same areas for grid
    self.totalMeshArea = jnp.sum(self.elemArea)

    self.elemNodes = np.zeros((self.numElems, self.nodesPerElem))
    self.edofMat = np.zeros((self.numElems,\
                                 self.nodesPerElem*self.dofsPerNode))
    for line in f:
      self.elemNodes[ctr,:] = line.rstrip().split('\t')
      # if(self.physics == self.PHYSICS_OPTIONS['Structural']):
      self.edofMat[ctr,:] = np.array([[2*self.elemNodes[ctr,i],\
                                self.dofsPerNode*self.elemNodes[ctr,i]+1] \
                               for i in range(self.nodesPerElem) ]).reshape(-1)
      # else:
      #   self.meshProp['edofMat'][ctr,:] = self.meshProp['elemNodes'][ctr,:];
      ctr += 1
    self.edofMat = self.edofMat.astype(int)
    self.elemNodes = self.elemNodes.astype(int)

 
    # compute elemCenters
    self.elemCenters = np.zeros((self.numElems, self.ndim))
    for elem in range(self.numElems):
      nodes = ((self.edofMat[elem,0::2]+2)/2).astype(int)-1
      for i in range(4):
        self.elemCenters[elem,0] += 0.25*self.nodeXY[nodes[i],0]
        self.elemCenters[elem,1] += 0.25*self.nodeXY[nodes[i],1]

    self.elemVertices = np.zeros((self.numElems,\
                            self.nodesPerElem,self.ndim))
    for elem in range(self.numElems):
      nodes = ((self.edofMat[elem,0::2]+2)/2).astype(int)-1
      self.elemVertices[elem,:,0] = self.nodeXY[nodes,0]
      self.elemVertices[elem,:,1] = self.nodeXY[nodes,1]

    iK = np.kron(self.edofMat,np.ones((8,1))).flatten().astype(int)
    jK = np.kron(self.edofMat,np.ones((1,8))).flatten().astype(int)

    self.nodeIdx = jax.ops.index[iK,jK]
    self.bb = {}
    self.bb['xmin'],self.bb['xmax'],self.bb['ymin'],self.bb['ymax'] =  \
      np.min(self.nodeXY[:,0]), np.max(self.nodeXY[:,0]),\
      np.min(self.nodeXY[:,1]), np.max(self.nodeXY[:,1])
  #-----------------------#
  def generatePoints(self, res = 1, includeEndPts = False):
    if(includeEndPts):
      endPts = 2
      resMin, resMax = 0, res+2
    else:
      endPts = 0
      resMin, resMax = 1, res+1
    points = np.zeros((self.numElems*(res+endPts)**2,2))
    ctr = 0
    for elm in range(self.numElems):
      nodes = self.elemNodes[elm,:]
      xmin, xmax = np.min(self.nodeXY[nodes,0]), np.max(self.nodeXY[nodes,0])
      ymin, ymax = np.min(self.nodeXY[nodes,1]), np.max(self.nodeXY[nodes,1])
      delX = (xmax-xmin)/(res+1.)
      delY = (ymax-ymin)/(res+1.)
      for rx in range(resMin,resMax):
        xv = xmin + rx*(delX)
        for ry in range(resMin, resMax):
          points[ctr,0] = xv
          points[ctr,1] = ymin + ry*(delY)
          ctr+= 1
    return points

#-----------------------#
  def plotFieldOnMesh(self, field, titleStr, res = 1):

    y = self.nodeXY[:,0]
    z = self.nodeXY[:,1]

    def quatplot(y,z, quatrangles, values, ax=None, **kwargs):

      if not ax: ax=plt.gca()
      yz = np.c_[y,z]
      verts= yz[quatrangles]
      pc = matplotlib.collections.PolyCollection(verts, **kwargs)
      pc.set_array(values)
      ax.add_collection(pc)
      ax.autoscale()
      ax.set_aspect('equal')
      return pc

    plt.ion(); plt.clf()
    
    pc = quatplot(y,z, np.asarray(self.elemNodes), -field, ax=None, 
             edgecolor="crimson", cmap="gray")

    plt.title(titleStr)
    plt.pause(0.001)
    plt.show()
    self.fig.canvas.draw()
    #--------------------------#