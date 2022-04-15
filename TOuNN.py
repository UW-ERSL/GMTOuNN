import numpy as np
import jax.numpy as jnp
import jax
from jax import jit
import matplotlib.pyplot as plt
from matplotlib import colors
from FE_Solver import JAXSolver
from network import TopNet
from projections import applyFourierMap, \
      applySymmetry, applyRotationalSymmetry, applyExtrusion
from jax.experimental import optimizers
from materialCoeffs import microStrs
import pickle

class TOuNN:
  def __init__(self, exampleName, mesh, material, nnSettings, symMap, \
                        fourierMap, rotationalSymmetry, extrusion):
    self.exampleName = exampleName
    self.FE = JAXSolver(mesh, material)
    self.xy = self.FE.mesh.elemCenters
    self.fourierMap = fourierMap
    if(fourierMap['isOn']):
      nnSettings['inputDim'] = 2*fourierMap['numTerms']
    else:
      nnSettings['inputDim'] = self.FE.mesh.ndim
    self.topNet = TopNet(nnSettings)
    
    self.symMap = symMap
    self.mstrData = microStrs
    self.rotationalSymmetry = rotationalSymmetry
    self.extrusion = extrusion
    #-----------------------#
  
  def optimizeDesign(self, optParams, savedNet):
    convgHistory = {'epoch':[], 'vf':[], 'J':[]}
    xyS = applySymmetry(self.xy, self.symMap)
    xyE = applyExtrusion(xyS, self.extrusion)
    xyR = applyRotationalSymmetry(xyE, self.rotationalSymmetry)
    if(self.fourierMap['isOn']):
      xyF = applyFourierMap(xyR, self.fourierMap)
    else:
      xyF = xyR
    penal = 1
    
    # C Matrix
    components = ['00', '11', '22', '01', '02', '12']
    def getCfromCPolynomial(vfracPow, mstr):
      C = {}
      for c in components:
        C[c] = jnp.zeros((self.FE.mesh.numElems))
      for c in components:
        for pw in range(mstr['order']+1):
            C[c] = C[c].at[:].add(mstr[c][str(pw)]*vfracPow[str(pw)])
      return C #{dict with 6 keys each of size numElems}
    
    def getCfromEigenPolynomial(vfracPow, mstr):

      lmda = jnp.zeros((self.FE.mesh.numElems, 3 , 3))
      for lmdIdx in range(3): # 3 eigen values
        lamStr = 'lambda'+str(lmdIdx+1)
        for pw in range(mstr['order']+1):
          lmda = lmda.at[:,lmdIdx,lmdIdx].add(mstr[lamStr][str(pw)]*vfracPow[str(pw)])
      VL = jnp.einsum('ij,ejk->eik', mstr['eVec'], lmda)
      VLVt = jnp.einsum('eij,jk->eik', VL, mstr['eVec'].T)
      
      C = {'00':VLVt[:,0,0], '11':VLVt[:,1,1], '22':VLVt[:,2,2], \
           '01':VLVt[:,0,1], '02':VLVt[:,0,2], '12':VLVt[:,1,2]}
      return C
    @jit
    def getCMatrix(mstrType, nn_rho):
      vfracPow = {} # compute the powers once to avoid repeated calc
      for pw in range(self.mstrData['square']['order']+1):# TODO: use the max order of all mstrs
        vfracPow[str(pw)] = nn_rho**pw
      C = {}
      for c in components:
        C[c] = jnp.zeros((self.FE.mesh.numElems))

      for mstrCtr, mstr in enumerate(self.mstrData): # mstrsEig # mstrs
        if(self.mstrData[mstr]['type'] == 'eig'):
          Cmstr = getCfromEigenPolynomial(vfracPow, self.mstrData[mstr])
        else:
          Cmstr = getCfromCPolynomial(vfracPow, self.mstrData[mstr])
        mstrPenal =  mstrType[:,mstrCtr]**penal
        for c in components:
          C[c] = C[c].at[:].add(jnp.einsum('i,i->i', mstrPenal, Cmstr[c]))
      
      return C

    # optimizer
    opt_init, opt_update, get_params = optimizers.adam(optParams['learningRate'])
    opt_state = opt_init(self.topNet.wts)
    opt_update = jit(opt_update)
    
    if(savedNet['isAvailable']):
      saved_params = pickle.load(open(savedNet['file'], "rb"))
      opt_state = optimizers.pack_optimizer_state(saved_params)

    # fwd once to get J0-scaling param
    mstrType, density0  = self.topNet.forward(get_params(opt_state), xyF)
    C = getCMatrix(mstrType, 0.01+density0)
    J0 = self.FE.objectiveHandle(C)

    # - jitting this causes undefined behavior
    def computeLoss(objective, constraints):
      if(optParams['lossMethod']['type'] == 'penalty'):
        alpha = optParams['lossMethod']['alpha0'] + \
                self.epoch*optParams['lossMethod']['delAlpha'] # penalty method
        loss = objective
        for c in constraints:
          loss += alpha*c**2
      if(optParams['lossMethod']['type'] == 'logBarrier'):
        t = optParams['lossMethod']['t0']* \
                          optParams['lossMethod']['mu']**self.epoch
        loss = objective
        for c in constraints:
          if(c < (-1/t**2)):
            psi = -jnp.log(-c)/t
          else:
            psi = t*c - jnp.log(1/t**2)/t + 1/t
          loss += psi
      return loss
    
    # closure function - jitting this causes undefined behavior
    def closure(nnwts):
      mstrType, density  = self.topNet.forward(nnwts, xyF)
      volCons = (jnp.mean(density)/optParams['desiredVolumeFraction'])- 1.
      C = getCMatrix(mstrType, 0.01+density)
      J = self.FE.objectiveHandle(C)
      return computeLoss(J/J0, [volCons])
    
    # optimization loop
    for self.epoch in range(optParams['maxEpochs']):
      penal = min(8.0, 1. + self.epoch*0.02)
      opt_state = opt_update(self.epoch, \
                  optimizers.clip_grads(jax.grad(closure)(get_params(opt_state)), 1.),\
                  opt_state)
  
      if(self.epoch%10 == 0):
        convgHistory['epoch'].append(self.epoch)
        mstrType, density = self.topNet.forward(get_params(opt_state), xyF)
        C = getCMatrix(mstrType, 0.01+density) # getCfromCPolynomial
        J = self.FE.objectiveHandle(C)
        convgHistory['J'].append(J)
        volf= jnp.mean(density)
        convgHistory['vf'].append(volf)
        if(self.epoch == 10):
          J0 = J
        status = 'epoch \t {:d} \t J \t {:.2E} \t vf \t {:.2F}'.format(self.epoch, J, volf)
        print(status)
        if(self.epoch%30 == 0):
          self.FE.mesh.plotFieldOnMesh(density, status)
    if(savedNet['isDump']):
      trained_params = optimizers.unpack_optimizer_state(opt_state)
      pickle.dump(trained_params, open(savedNet['file'], "wb"))
    return convgHistory
  #-----------------------#
  def plotCompositeTopology(self, res):
    xy = self.FE.mesh.generatePoints(res)
    xyS = applySymmetry(xy, self.symMap)
    xyE = applyExtrusion(xyS, self.extrusion)
    xyR = applyRotationalSymmetry(xyE, self.rotationalSymmetry)
    if(self.fourierMap['isOn']):
      xyF = applyFourierMap(xyR, self.fourierMap)
    else:
      xyF = xyR
    mstrType, density = self.topNet.forward(self.topNet.wts, xyF)
    
    fillColors = ['white', (1,0,0), (0,1,0), (0,0,1), (0,0,0), (0,1,1),\
       (1,0,1), (0.5,0,0.5), (1,0.55,0), (0,0.5,0.5), (0,0,0.5), (0,0.5,0),\
         (0.5,0,0), (0.5,0.5,0)]
    microstrImages = np.load('./microStrImages.npy')
    
    NX = res*int(np.ceil((self.FE.mesh.bb['xmax']-\
            self.FE.mesh.bb['xmin'])/self.FE.mesh.elemSize[0]))
    NY = res*int(np.ceil((self.FE.mesh.bb['ymax']-\
            self.FE.mesh.bb['ymin'])/self.FE.mesh.elemSize[1]))
    nx, ny = microstrImages.shape[2], microstrImages.shape[3]
    compositeImg = np.zeros((NX*nx, NY*ny))
    colorImg = np.zeros((NX, NY))
    densityImg = np.zeros((NX, NY))
    maxC = 0
    step = 0.01 # step used when gen mstr images!
    cutOff = 0.98 # val above which its a dark square
    for elem in range(xy.shape[0]):
      cx = int((res*xy[elem,0])/self.FE.mesh.elemSize[0])
      cy = int((res*xy[elem,1])/self.FE.mesh.elemSize[1])
      densityImg[cx, cy] = int(100.*density[elem])
      if(density[elem] > cutOff):
        compositeImg[nx*cx:(cx+1)*nx, ny*cy:(cy+1)*ny] = np.ones((nx, ny))
        colorImg[cx, cy] = 1
      else:
        mstrIdx = min(microstrImages.shape[1]-1, int(density[elem]//step))
        mstrTypeIdx = np.argmax(mstrType[elem,:])
        mstrimg = microstrImages[mstrTypeIdx, mstrIdx,:,:].T
        c = np.argmax(mstrType[elem,:])+1
        if(c > maxC):
          maxC = c
        compositeImg[nx*cx:(cx+1)*nx, ny*cy:(cy+1)*ny] = mstrimg*c
        colorImg[cx, cy] = c
    plt.figure()
    plt.imshow(compositeImg.T, cmap = colors.ListedColormap(fillColors[:maxC+1]),\
               interpolation='none',vmin=0, vmax=maxC, origin = 'lower') 
    plt.show()

    plt.savefig(f'./frames/top_{self.epoch:d}.pdf', dpi=300)

    plt.figure()
    plt.imshow(colorImg.T, cmap = colors.ListedColormap(fillColors[:maxC+1]),\
               interpolation='none',vmin=0, vmax=maxC, origin = 'lower') 
    plt.show()

    plt.figure()
    plt.imshow(-densityImg.T, cmap = 'gray',\
               interpolation='none', origin = 'lower') 
    plt.show()
    #-----------------------#
