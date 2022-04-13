import numpy as np
import time
import matplotlib.pyplot as plt

    #  ~~~~~~~~~~~~ Examples ~~~~~~~~~~~~~#
def getExampleBC(example, nelx, nely, elemSize):
  if(example == 1): # tip cantilever
    exampleName = 'TipCantilever'
    bcSettings = {'fixedNodes': np.arange(0,2*(nely+1),1),\
                  'forceMagnitude': -1.,\
                  'forceNodes': 2*(nelx+1)*(nely+1)-2*nely+1, \
                  'dofsPerNode':2}
    symMap = {'XAxis':{'isOn':False, 'midPt':0.5*nely*elemSize[1]},\
      'YAxis':{'isOn':False, 'midPt':0.5*nelx*elemSize[0]}}
  
  elif(example == 2): # mid cantilever
    exampleName = 'MidCantilever'
    bcSettings = {'fixedNodes': np.arange(0,2*(nely+1),1),\
                  'forceMagnitude': -1.,\
                  'forceNodes': 2*(nelx+1)*(nely+1)- (nely+1),\
                  'dofsPerNode':2}
    symMap = {'XAxis':{'isOn':True, 'midPt':0.5*nely*elemSize[1]},\
      'YAxis':{'isOn':False, 'midPt':0.5*nelx*elemSize[0]}}
  
  elif(example == 3): #  MBBBeam
    exampleName = 'MBBBeam'
    fn= np.union1d(np.arange(0,2*(nely+1),2), 2*(nelx+1)*(nely+1)-2*(nely+1)+1);
    bcSettings = {'fixedNodes': fn,\
                  'forceMagnitude': -1.,\
                  'forceNodes': 2*(nely+1)+1,\
                  'dofsPerNode':2}
    symMap = {'XAxis':{'isOn':False, 'midPt':0.5*nely*elemSize[1]},\
      'YAxis':{'isOn':False, 'midPt':0.5*nelx*elemSize[0]}}
  
  elif(example == 4): #  Michell
    exampleName = 'Michell'
    fn = np.array([ 0,1,2*(nelx+1)*(nely+1)-2*nely] )
    bcSettings = {'fixedNodes': fn,\
                  'forceMagnitude': -1.,\
                  'forceNodes': nelx*(nely+1)+1,\
                  'dofsPerNode':2}
    symMap = {'XAxis':{'isOn':False, 'midPt':0.5*nely*elemSize[1]},\
      'YAxis':{'isOn':True, 'midPt':0.5*nelx*elemSize[0]}}
  
  elif(example == 5): #  DistributedMBB
    exampleName = 'Bridge'
    fixn = np.array([ 0,1,2*(nelx+1)*(nely+1)-2*nely+1,2*(nelx+1)*(nely+1)-2*nely] );
    frcn = np.arange(2*nely+1, 2*(nelx+1)*(nely+1), 8*(nely+1))
    bcSettings = {'fixedNodes': fixn,\
                  'forceMagnitude': -1./(nelx+1.),\
                  'forceNodes': frcn ,\
                  'dofsPerNode':2} 
    symMap = {'XAxis':{'isOn':False, 'midPt':0.5*nely*elemSize[1]},\
      'YAxis':{'isOn':True, 'midPt':0.5*nelx*elemSize[0]}}

  elif(example == 6): # Tensile bar
    exampleName = 'TensileBar'
    fn =np.union1d(np.arange(0,2*(nely+1),2), 1); 
    midDofX= 2*(nelx+1)*(nely+1)- (nely)
    bcSettings = {'fixedNodes': fn,\
                  'forceMagnitude': 1.,\
                  'forceNodes': np.arange(midDofX-6, midDofX+6, 2),\
                  'dofsPerNode':2}; 
    symMap = {'XAxis':{'isOn':True, 'midPt':0.5*nely*elemSize[1]},\
      'YAxis':{'isOn':False, 'midPt':0.5*nelx*elemSize[0]}}
    
  elif(example == 7): # full right cantilever
    exampleName = 'fullRightCantilever'
    forceNodes = np.arange(2*(nelx+1)*(nely+1)-2*nely+1,\
                           2*(nelx+1)*(nely+1), 2)
    bcSettings = {'fixedNodes': np.arange(0,2*(nely+1),1),\
                  'forceMagnitude': -4.76,\
                  'forceNodes': forceNodes,\
                  'dofsPerNode':2}
    symMap = {'XAxis':{'isOn':True, 'midPt':0.5*nely*elemSize[1]},\
      'YAxis':{'isOn':False, 'midPt':0.5*nelx*elemSize[0]}}

  elif(example == 8): # Torsion
    exampleName = 'Torsion'
    forceFile = './Mesh/Torsion/force.txt'
    fixedFile = './Mesh/Torsion/fixed.txt'
    nodeXYFile = './Mesh/Torsion/nodeXY.txt'
    elemNodesFile = './Mesh/Torsion/elemNodes.txt'
    bcSettings = {'forceFile': forceFile,\
                  'fixedFile': fixedFile,\
                  'elemNodesFile': elemNodesFile,\
                  'nodeXYFile': nodeXYFile,\
                  'dofsPerNode':2}; 
    symMap = {'XAxis':{'isOn':True, 'midPt':50},\
      'YAxis':{'isOn':True, 'midPt':50}}

  elif(example == 9):
    exampleName = 'LBracket'
    forceFile = './Mesh/LBracket/force.txt'
    fixedFile = './Mesh/LBracket/fixed.txt'
    nodeXYFile = './Mesh/LBracket/nodeXY.txt'
    elemNodesFile = './Mesh/LBracket/elemNodes.txt'
    bcSettings = {'forceFile': forceFile,\
                  'fixedFile': fixedFile,\
                  'elemNodesFile': elemNodesFile,\
                  'nodeXYFile': nodeXYFile,\
                  'dofsPerNode':2}
    symMap = {'XAxis':{'isOn':False, 'midPt':5},\
      'YAxis':{'isOn':False, 'midPt':5}}
  elif(example == 10):
    exampleName = 'midLoadMBB'
    forceFile = './Mesh/midLoadMBB/force.txt'
    fixedFile = './Mesh/midLoadMBB/fixed.txt'
    nodeXYFile = './Mesh/midLoadMBB/nodeXY.txt'
    elemNodesFile = './Mesh/midLoadMBB/elemNodes.txt'
    bcSettings = {'forceFile': forceFile,\
                  'fixedFile': fixedFile,\
                  'elemNodesFile': elemNodesFile,\
                  'nodeXYFile': nodeXYFile,\
                  'dofsPerNode':2}
    symMap = {'XAxis':{'isOn':False, 'midPt':5},\
      'YAxis':{'isOn':True, 'midPt':50}}
  return exampleName, bcSettings, symMap

    
