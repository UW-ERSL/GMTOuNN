import jax.numpy as jnp
import numpy as np
from jax.ops import index, index_update

#-------FOURIER LENGTH SCALE-----------#
def computeFourierMap(mesh, fourierMap):
  # compute the map
  coordnMapSize = (mesh.ndim, fourierMap['numTerms'])
  freqSign = np.random.choice([-1.,1.], coordnMapSize)
  stdUniform = np.random.uniform(0.,1., coordnMapSize) 
  wmin = 1./(2*fourierMap['maxRadius'])
  wmax = 1./(2*fourierMap['minRadius']) # w~1/R
  wu = wmin +  (wmax - wmin)*stdUniform
  coordnMap = np.einsum('ij,ij->ij', freqSign, wu)
  return coordnMap
#-----------------#
def applyFourierMap(xy, fourierMap):
  if(fourierMap['isOn']):
    c = jnp.cos(2*np.pi*jnp.einsum('ij,jk->ik', xy, fourierMap['map']))
    s = jnp.sin(2*np.pi*jnp.einsum('ij,jk->ik', xy, fourierMap['map']))
    xy = jnp.concatenate((c, s), axis = 1)
  return xy

#-------DENSITY PROJECTION-----------#

def applyDensityProjection(x, densityProj):
  if(densityProj['isOn']):
    b = densityProj['sharpness']
    nmr = np.tanh(0.5*b) + jnp.tanh(b*(x-0.5))
    x = 0.5*nmr/np.tanh(0.5*b)
  return x

#-------SYMMETRY-----------#
def applySymmetry(x, symMap):
  if(symMap['YAxis']['isOn']):
    xv = index_update( x[:,0], index[:], symMap['YAxis']['midPt'] \
                          + jnp.abs(x[:,0] - symMap['YAxis']['midPt']) )
  else:
    xv = x[:, 0]
  if(symMap['XAxis']['isOn']):
    yv = index_update( x[:,1], index[:], symMap['XAxis']['midPt'] \
                          + jnp.abs(x[:,1] - symMap['XAxis']['midPt']) )
  else:
    yv = x[:, 1]
  x = jnp.stack((xv, yv)).T
  return x
#--------------------------#
def applyRotationalSymmetry(xyCoordn, rotationalSymmetry):
  if(rotationalSymmetry['isOn']):
    dx = xyCoordn[:,0] - rotationalSymmetry['centerCoordn'][0]
    dy = xyCoordn[:,1] - rotationalSymmetry['centerCoordn'][1]
    radius = jnp.sqrt( (dx)**2 + (dy)**2 )
    angle = jnp.arctan2(dy, dx)
    correctedAngle = jnp.remainder(angle, np.pi*rotationalSymmetry['sectorAngleDeg']/180.)
    x, y = radius*jnp.cos(correctedAngle), radius*jnp.sin(correctedAngle)
    xyCoordn = jnp.stack((x,y)).T
  return xyCoordn
#--------------------------#
def applyExtrusion(xy, extrusion):
  if(extrusion['X']['isOn']):
    xv = xy[:,0]%extrusion['X']['delta']
  else:
    xv = xy[:,0]
  if(extrusion['Y']['isOn']):
    yv = xy[:,1]%extrusion['Y']['delta']
  else:
    yv = xy[:,1]
  x = jnp.stack((xv, yv)).T
  return x
#--------------------------#

  