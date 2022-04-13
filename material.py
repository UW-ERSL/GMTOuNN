import numpy as np

class Material:
  def __init__(self, matProp):
    self.matProp = matProp
    E, nu = matProp['Emax'], matProp['nu']
    self.C = E/(1-nu**2)* \
            np.array([[1, nu, 0],\
                      [nu, 1, 0],\
                      [0, 0, (1-nu)/2]])
  #--------------------------#
  
  def computeSIMP_Interpolation(self, rho, penal):
    E = 0.001*self.matProp['Emax'] + \
            (0.999*self.matProp['Emax'])*\
            (rho+0.01)**penal
    return E
  #--------------------------#
  
  def computeRAMP_Interpolation(self, rho, penal):
    E = 0.001*self.matProp['Emax']  +\
        (0.999*self.matProp['Emax'])*\
            (rho/(1.+penal*(1.-rho)))
    return E
  #--------------------------#
  def getD0elemMatrix(self, mesh):
    if(mesh.meshType == 'gridMesh'):
       E = 1
       nu = self.matProp['nu']
       k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8,\
                     -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
       D0 = E/(1-nu**2)*np.array\
    ([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
       [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
       [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
       [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
       [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
       [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
       [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
       [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ])
       # all the elems have same base stiffness
       return D0

    #--------------------------#