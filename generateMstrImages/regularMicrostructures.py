import numpy as np
import scipy.io
import math
import matplotlib.pyplot as plt
from PIL import Image
import random
from scipy import ndimage, misc
from scipy.ndimage import rotate
from matplotlib import colors
from homogenize import Homogenization
import time
def to_np(x):
  return x.detach().cpu().numpy()

class SingleParamMicrostructures:
  #------------------------------------------------------------------------#
  def __init__(self, nelx, nely, eps = 0):
    self.nelx, self.nely, self.eps = nelx, nely, eps
  #------------------------------------------------------------------------#
  def square(self, t):
    '''
    t between 0 and 1
    '''
    microStr = np.zeros((self.nelx,self.nely))
    t = 0.5*t*self.nelx # scale t

    for rw in range(self.nelx):
      for col in range(self.nely):
        if((rw < t) or (self.nelx-rw < t) or (col < t) or (self.nely-col < t)):
          microStr[rw,col] = 1
        else:
          microStr[rw,col] = self.eps
    return microStr
  #------------------------------------------------------------------------#
  def vbox(self, t):
    '''
    t between 0 and 1
    '''
    microStr = self.eps*np.ones((self.nelx,self.nely))
    t = 0.25*t*self.nelx # scale t

    for rw in range(self.nelx):
      for col in range(self.nely):
        if(col>0.5*self.nely-0.5*t and col < 0.5*self.nely+0.5*t): # bar
          microStr[rw,col] = 1
        if((rw < t) or (self.nelx-rw < t) or (col < t) or (self.nely-col < t)): #box
          microStr[rw,col] = 1
    return microStr
  #------------------------------------------------------------------------#
  def hbox(self, t):
    '''
    t between 0 and 1
    '''
    microStr = self.eps*np.ones((self.nelx,self.nely))
    t = 0.31*t*self.nelx # scale t

    for rw in range(self.nelx):
      for col in range(self.nely):
        if(rw>0.5*self.nelx-0.5*t and rw < 0.5*self.nelx+0.5*t): #bar
          microStr[rw,col] = 1
        if((rw < t) or (self.nelx-rw < t) or (col < t) or (self.nely-col < t)): #box
          microStr[rw,col] = 1
    return microStr
  #------------------------------------------------------------------------#
  def plusbox(self, t):
    '''
    t between 0 and 1
    '''
    microStr = self.eps*np.ones((self.nelx,self.nely))
    t = 0.23*t*self.nelx # scale t

    for rw in range(self.nelx):
      for col in range(self.nely):
        if(col>0.5*self.nely-0.5*t and col < 0.5*self.nely+0.5*t): # vbar
          microStr[rw,col] = 1
        if(rw>0.5*self.nelx-0.5*t and rw < 0.5*self.nelx+0.5*t): #hbar
          microStr[rw,col] = 1
        if((rw < t) or (self.nelx-rw < t) or (col < t) or (self.nely-col < t)): #box
          microStr[rw,col] = 1
    return microStr
  #------------------------------------------------------------------------#
  def X(self, t, theta = np.pi/4.):
    microStr = self.eps*np.ones((self.nelx, self.nely))
    t = 0.35*t*self.nelx
    cx = 0.5*self.nelx
    cy = 0.5*self.nely
    tTheta = math.tan(theta)
    for rw in range(self.nelx):
      x = abs(rw-cx)
      for col in range(self.nely):
        y = abs(col-cy)
        XIntercept = x - (1.0*y)/tTheta
        if(abs(XIntercept) < t):
          microStr[rw,col] = 1
    return microStr
  #------------------------------------------------------------------------#
  def Xbox(self, t, theta = np.pi/4.):
    microStr = self.eps*np.ones((self.nelx, self.nely));
    t = 0.2*t*self.nelx
    cx = 0.5*self.nelx
    cy = 0.5*self.nely
    tTheta = math.tan(theta)
    for rw in range(self.nelx):
      x = abs(rw-cx)
      for col in range(self.nely):
        y = abs(col-cy)
        XIntercept = x - (1.0*y)/tTheta
        if(abs(XIntercept) < 0.75*t):
          microStr[rw,col] = 1
        if((rw < t) or (self.nelx-rw < t) or (col < t) or (self.nely-col < t)):
          microStr[rw,col] = 1
    return microStr
  #------------------------------------------------------------------------#
  def xpbox(self, t, theta = np.pi/4.):
    microStr = self.eps*np.ones((self.nelx, self.nely))
    t = 0.15*t*self.nelx
    cx = 0.5*self.nelx
    cy = 0.5*self.nely
    tTheta = math.tan(theta)
    for rw in range(self.nelx):
      x = abs(rw-cx)
      for col in range(self.nely):
        y = abs(col-cy)
        XIntercept = x - (1.0*y)/tTheta
        if(abs(XIntercept) < 0.75*t): # X
          microStr[rw,col] = 1
        if(col>0.5*self.nely-0.5*t and col < 0.5*self.nely+0.5*t): # vbar
          microStr[rw,col] = 1
        if(rw>0.5*self.nelx-0.5*t and rw < 0.5*self.nelx+0.5*t): #hbar
          microStr[rw,col] = 1
        if((rw < t) or (self.nelx-rw < t) or (col < t) or (self.nely-col < t)): # box
          microStr[rw,col] = 1

    return microStr
  #------------------------------------------------------------------------#
  def Zbox(self, t):
    # t - thickness between 0 and 1
    microStr = self.eps*np.ones((self.nelx, self.nely))
    t = 0.25*t*self.nelx
    for rw in range(self.nelx):
      for col in range(self.nely):
        if((rw < t) or (self.nelx-rw < t) or (col < t) or (self.nely-col < t) \
                                 or (abs(rw-col) < 0.7*t)):
          microStr[rw,col] = 1
    return microStr
  #------------------------------------------------------------------------#
  def Nbox(self, t):
    # t - thickness between 0 and 1
    microStr = self.eps*np.ones((self.nelx, self.nely))
    t = 0.25*t*self.nelx
    for rw in range(self.nelx):
      for col in range(self.nely):
        if((rw < t) or (self.nelx-rw < t) or (col < t) or (self.nely-col < t) \
                                 or (abs(rw-col) < 0.7*t)):
          microStr[rw,self.nely-col-1] = 1
    return microStr
  #------------------------------------------------------------------------#
  def diamBox(self, t):
    microStr = self.eps*np.ones((self.nelx, self.nely))
    delt = 0.4*t
    boxt = 0.35*delt*self.nelx
    for rw in range(self.nelx):
      xc = -1. + (2.*rw)/(self.nelx-1)
      for col in range(self.nely):
        yc = -1. + (2.*col)/(self.nely-1)
        val = np.abs(xc) + np.abs(yc)
        
        if( val > 1.-delt and val < 1.+delt):
          microStr[rw, col] = 1.
        if((rw < boxt) or (self.nelx-rw < boxt) or (col < boxt) or (self.nely-col < boxt)):
          microStr[rw,col] = 1
    return microStr
  #------------------------------------------------------------------------#
  def diamXBox(self, t):
    microStr = self.eps*np.ones((self.nelx, self.nely))
    delt = 0.3*t
    boxt = 0.4*delt*self.nelx
    cx = 0.5*self.nelx
    cy = 0.5*self.nely
    theta = np.pi/4.
    tTheta = math.tan(theta)
    for rw in range(self.nelx):
      xc = -1. + (2.*rw)/(self.nelx-1)
      xPlus = abs(rw-cx)
      for col in range(self.nely):
        yPlus = abs(col-cy)
        XIntercept = xPlus - (1.0*yPlus)/tTheta
        if(abs(XIntercept) < 0.75*boxt):
          microStr[rw,col] = 1
        if((rw < boxt) or (self.nelx-rw < boxt) or (col < boxt) or (self.nely-col < boxt)):
          microStr[rw,col] = 1
        
        yc = -1. + (2.*col)/(self.nely-1)
        val = np.abs(xc) + np.abs(yc)
        
        if( val > 1.-delt and val < 1.+delt):
          microStr[rw, col] = 1.
        if((rw < boxt) or (self.nelx-rw < boxt) or (col < boxt) or (self.nely-col < boxt)):
          microStr[rw,col] = 1

    return microStr
  #------------------------------------------------------------------------#
  def diamHBox(self, t):
    microStr = self.eps*np.ones((self.nelx, self.nely))
    delt = 0.3*t
    boxt = 0.4*delt*self.nelx
    for rw in range(self.nelx):
      xc = -1. + (2.*rw)/(self.nelx-1)
      for col in range(self.nely):
        yc = -1. + (2.*col)/(self.nely-1)
        val = np.abs(xc) + np.abs(yc)
        
        if( val > 1.-delt and val < 1.+delt):
          microStr[rw, col] = 1.
        if((rw < boxt) or (self.nelx-rw < boxt) or (col < boxt) or (self.nely-col < boxt)):
          microStr[rw,col] = 1
        if(rw>0.5*self.nelx-0.5*boxt and rw < 0.5*self.nelx+0.5*boxt): #bar
          microStr[rw,col] = 1
        
    return microStr
  #------------------------------------------------------------------------#
  def diamVBox(self, t):
    microStr = self.eps*np.ones((self.nelx, self.nely))
    delt = 0.3*t
    boxt = 0.4*delt*self.nelx
    for rw in range(self.nelx):
      xc = -1. + (2.*rw)/(self.nelx-1)
      for col in range(self.nely):
        yc = -1. + (2.*col)/(self.nely-1)
        val = np.abs(xc) + np.abs(yc)
        
        if( val > 1.-delt and val < 1.+delt):
          microStr[rw, col] = 1.
        if((rw < boxt) or (self.nelx-rw < boxt) or (col < boxt) or (self.nely-col < boxt)):
          microStr[rw,col] = 1
        if(col>0.5*self.nely-0.5*boxt and col < 0.5*self.nely+0.5*boxt): #bar
          microStr[rw,col] = 1
    return microStr
  #------------------------------------------------------------------------#
  def diamPlusBox(self, t):
    microStr = self.eps*np.ones((self.nelx, self.nely))
    delt = 0.3*t
    boxt = 0.4*delt*self.nelx
    for rw in range(self.nelx):
      xc = -1. + (2.*rw)/(self.nelx-1)
      for col in range(self.nely):
        yc = -1. + (2.*col)/(self.nely-1)
        val = np.abs(xc) + np.abs(yc)
        
        if( val > 1.-delt and val < 1.+delt):
          microStr[rw, col] = 1.
        if((rw < boxt) or (self.nelx-rw < boxt) or (col < boxt) or (self.nely-col < boxt)):
          microStr[rw,col] = 1
        if(col>0.5*self.nely-0.5*boxt and col < 0.5*self.nely+0.5*boxt): #vbar
          microStr[rw,col] = 1
        if(rw>0.5*self.nelx-0.5*boxt and rw < 0.5*self.nelx+0.5*boxt): #hbar
          microStr[rw,col] = 1
    return microStr
  #------------------------------------------------------------------------#
  def void0(self, t):
    # default microstr for void
    microStr = self.eps*np.ones((self.nelx, self.nely))
    noise = np.random.uniform(0, 0.01, microStr.shape )
    return microStr + noise
  #------------------------------------------------------------------------#
  def plotMicrostructure(self, mc):
    mc = mc.reshape(self.nelx, self.nely)
    fig, ax = plt.subplots()

    ax.imshow(-np.flipud(mc), cmap='gray',\
                interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
    plt.title(f' vf: {np.mean(mc):.2F}' )
    fig.tight_layout()
    fig.show()
  #------------------------------------------------------------------------#
  
class ThreeParamMicrostructures:
  
  #------------------------------------------------------------------------#
  def __init__(self, nelx, nely, eps = 0):
    self.nelx, self.nely, self.eps = nelx, nely, eps
  #------------------------------------------------------------------------#
  def ellipticalHole(self, a, b, thDeg):
    microStr = np.ones((self.nelx,self.nely))
    a = 0.5*a*self.nelx
    b = 0.5*b*self.nely
    thRad = np.pi*thDeg/180.
    center = np.array([0.5*self.nelx, 0.5*self.nely])
    for rw in range(self.nelx):
      x = rw-center[1]
      for col in range(self.nely):
        y = col-center[0]
        xrot = x*np.cos(thRad) - y*np.sin(thRad)
        yrot = x*np.sin(thRad) + y*np.cos(thRad)
        if ( ((xrot/a)**2 + (yrot/b)**2) <= 1. ):
          microStr[rw,col] = self.eps
    return microStr
  #------------------------------------------------------------------------#
  def victorGauss(self, a, b, n):
    # generate higher order  hole mus with semi-axis len a and b
    # a and b are floats between 0 and 1
    # center - array with 2 elems between 0 and 1 as well
    microStr = np.ones((self.nelx,self.nely))
    a = 0.5*a*self.nelx
    b = 0.5*b*self.nely
    center = np.array([0.5*self.nelx, 0.5*self.nely])
    for rw in range(self.nelx):
      x = np.abs(rw-center[1])
      for col in range(self.nely):
        y = np.abs(col-center[0])
        if ( ((x/a)**n + (y/b)**n) <= 1. ):
          microStr[rw,col] = self.eps
    return microStr
  #------------------------------------------------------------------------#
  def plotMicrostructure(self, mc):
    mc = mc.reshape(self.nelx, self.nely)
    fig, ax = plt.subplots()

    ax.imshow(-np.flipud(mc), cmap='gray',\
                interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
    fig.tight_layout()
    fig.show()
  #------------------------------------------------------------------------#
class Microstructures:
  def __init__(self, nelx = 10, nely = 10, eps = 0):
    self.nelx = nelx
    self.nely = nely
    self.eps = eps
  #------------------------------------------------------------------------#
  def setMicroStructureSize(self, nelx, nely):
    self.nelx = nelx
    self.nely = nely
  #------------------------------------------------------------------------#
  def generateRandomMicrostructure(self):
    mcr = np.random.choice([2,3,5,6])#1+ np.random.randint(8)

    if(mcr == 1):
      t = np.random.random()
      theta = 5. + 85*np.random.random()
      microStr = self.EMicroStructure(t, theta)
      return 'E', t, theta, microStr
    elif(mcr == 2):
      a = np.random.random()
      b = np.random.random()
      microStr = self.rectangleVoidMicrostructure(a, b)
      return 'rect', a, b, microStr
    elif(mcr == 3):
      a = 0.1 + 0.9*np.random.random()
      b = 0.1 + 0.9*np.random.random()
      microStr = self.circularVoidMicrostructure(a, b)
      return 'circ', a, b, microStr
    elif(mcr == 4):
      t = np.random.random()
      theta = 5. + 85*np.random.random()
      microStr = self.BarMicrostructure(t, theta)
      return 'bar', t, theta, microStr
    elif(mcr == 5):
      t = np.random.random()
      flipped = random.choice([True, False])
      microStr = self.ZMicroStructure(t, flipped)
      return 'Z', t, int(flipped), microStr
    elif(mcr == 6):
      t = np.random.random()
      flipped = random.choice([True, False])
      microStr = self.NMicroStructure(t, flipped)
      return 'N', t, int(flipped), microStr
    elif(mcr == 7):
      t = np.random.random()
      theta = 0.5*np.pi*np.random.random()
      microStr = self.XMicrostructure(t, theta)
      return 'X', t, theta, microStr
    elif(mcr == 8):
      t = np.random.random()
      theta = 5+ 85*np.random.random()
      microStr = self.tictacMicrostructure(t, theta)
      return 'tictac', t, theta, microStr

  #------------------------------------------------------------------------#
  # Rectangular void
  def rectangleVoidMicrostructure(self, a, b):
    # generate a rect mus with void of size a along X and b along Y
    # a and b are float between 1 and 0
    microStr = np.zeros((self.nelx,self.nely))
    a = 0.5*a*self.nelx
    b = 0.5*b*self.nely
    for rw in range(self.nelx):
      for col in range(self.nely):
        if((rw < a) or (self.nelx-rw < a) or (col < b) or (self.nely-col < b) ):
          microStr[rw,col] = 1
        else:
          microStr[rw,col] = self.eps
    return microStr
  #------------------------------------------------------------------------#
  def circularVoidMicrostructure(self, a, b):
    # generate circular hole mus with semi-axis len a and b
    # a and b are floats between 0 and 1
    # center - array with 2 elems between 0 and 1 as well
    microStr = np.ones((self.nelx,self.nely))
    a = 0.5*a*self.nelx
    b = 0.5*b*self.nely
    center = np.array([0.5*self.nelx, 0.5*self.nely])
    for rw in range(self.nelx):
      x = rw-center[1]
      for col in range(self.nely):
        y = col-center[0]
        if ( ((x/a)**2 + (y/b)**2) <= 1. ):
          microStr[rw,col] = self.eps
    return microStr
  #------------------------------------------------------------------------#
  def doubleLobeCircularVoid(self, a, b, yHeight = 0.5):
    microStr = np.ones((self.nelx,self.nely))
    a = 0.5*a*self.nelx
    b = 0.5*b*self.nely
    center = [np.array([0., yHeight*self.nely]), np.array([self.nelx, yHeight*self.nely])]
    for c in center:
      for rw in range(self.nelx):
        x = rw-c[1]
        for col in range(self.nely):
          y = col-c[0]
          if ( ((x/a)**2 + (y/b)**2) <= 1. ):
            microStr[rw,col] = self.eps
    return microStr
  #------------------------------------------------------------------------#
  def BarMicrostructure(self, t, theta):
    # t varies between 0 and 1
    microStr = self.eps*np.ones((self.nelx,self.nely))
    t = 0.5*t*self.nely
    center = self.nely*0.5
    for col in range(self.nely):
      delY = col-center
      if(abs(delY) <= t):
        microStr[:,col] = 1
    microStr = self.rotateMicrostructure(microStr, theta)
    return microStr
  #------------------------------------------------------------------------#
  def ZMicroStructure(self,  t, flipped):
    # t - thickness between 0 and 1
    microStr = self.eps*np.ones((self.nelx, self.nely))
    t = 0.5*t*self.nelx
    for rw in range(self.nelx):
      for col in range(self.nely):
        if((rw < t) or (self.nelx-rw < t) or (abs(rw-col) < t)):
          if(flipped):
            microStr[rw,col] = 1
          else:
            microStr[self.nelx-rw-1,col] = 1
    return microStr
  #------------------------------------------------------------------------#
  def NMicroStructure(self, t, flipped):
    # t - thickness between 0 and 1
    microStr = self.eps*np.ones((self.nelx, self.nely))
    t = 0.5*t*self.nelx
    for rw in range(self.nelx):
      for col in range(self.nely):
        if((col < t) or (self.nely-col < t) or (abs(rw-col) < t)):
          if(not flipped):
            microStr[rw,col] = 1
          else:
            microStr[rw,self.nely-col-1] = 1
    return microStr
  #------------------------------------------------------------------------#

  def XMicrostructure(self, t, theta):
    microStr = self.eps*np.ones((self.nelx, self.nely))
    t = 0.5*t*self.nelx
    cx = 0.5*self.nelx
    cy = 0.5*self.nely
    tTheta = math.tan(theta)
    for rw in range(self.nelx):
      x = abs(rw-cx)
      for col in range(self.nely):
        y = abs(col-cy)
        XIntercept = x - (1.0*y)/tTheta
        if(abs(XIntercept) < t):
          microStr[rw,col] = 1
    return microStr
  #------------------------------------------------------------------------#
  def tictacMicrostructure(self, t, theta):
    microStr = self.eps*np.ones((self.nelx, self.nely))
    t = 0.125*t*self.nelx
    microStr[0:self.nelx,int(0.333*self.nely-t):int(0.333*self.nely+t)] = 1.
    microStr[0:self.nelx,int(0.667*self.nely-t):int(0.669*self.nely+t)] = 1.
    microStr[int(0.333*self.nelx-t):int(0.333*self.nelx+t),0:self.nely] = 1.
    microStr[int(0.667*self.nelx-t):int(0.667*self.nelx+t),0:self.nely] = 1.
    microStr = self.rotateMicrostructure(microStr, theta)
    return microStr
  #------------------------------------------------------------------------#
  def EMicroStructure(self, t, theta):
      microStr = self.eps*np.ones((self.nelx, self.nely))
      t = 0.2*t*self.nelx 
      microStr[0:self.nelx,0:int(t)] = 1.
      microStr[0:self.nelx,int(0.5*self.nely-t):int(0.5*self.nely+t)] = 1.
      microStr[0:self.nelx,int(self.nely-t):self.nely] = 1.
      microStr[0:int(t),0:self.nely] = 1.
      microStr = self.rotateMicrostructure(microStr, theta)
      return microStr
  #------------------------------------------------------------------------#
  def rotateMicrostructure(self, microStr, th):
    microStr = rotate(microStr,th,reshape = False,order = 0)
    for i in range(self.nelx):
      for j in range(self.nely):
        microStr[i,j] = max(self.eps,min(1,microStr[i,j]))
#    # give a border of a few pixel thick to ensure continuity
    t = 1 # t pixel thick border
    microStr[0:self.nelx,0:t] = 1
    microStr[0:self.nelx,self.nely-t:self.nely] = 1
    microStr[0:t,0:self.nely] = 1
    microStr[self.nelx-t:self.nelx,0:self.nely] = 1
    return microStr
  #------------------------------------------------------------------------#
  def plotMicrostructure(self, mc):
    mc = mc.reshape(self.nelx, self.nely)
    fig, ax = plt.subplots()

    ax.imshow(-np.flipud(mc), cmap='gray',\
                interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0)); # self.topAx
    fig.tight_layout() # self.topFig
    fig.show()

def generateMstrImages(step = 0.01):
  t = np.arange(0, 1., step)
  nelx, nely = 50, 50
  M = SingleParamMicrostructures(nelx, nely)
  mstrFn = {'0':lambda t: M.square(t),\
    '1':lambda t: M.Xbox(t),\
    '2':lambda t: M.X(t),\
    '3':lambda t: M.diamBox(t),\
    '4':lambda t: M.xpbox(t),\
    '5':lambda t: M.diamPlusBox(t),\
    '6':lambda t: M.diamXBox(t),\
    '7':lambda t: M.Zbox(t),\
    '8':lambda t: M.Nbox(t),\
    '9':lambda t: M.diamVBox(t),\
    '10':lambda t: M.diamHBox(t) }
  microstructures = np.zeros((len(mstrFn), t.shape[0], nelx, nely))
  for m in range(len(mstrFn)):
    for i in range(t.shape[0]):
      print(f'{m*t.shape[0] + i:d}/{(t.shape[0]*len(mstrFn)):d}')
      mstr = mstrFn[str(m)](t[i])
      microstructures[m,i,:,:] = mstr
    
  np.save('./microStrImages.npy', microstructures)


generateMstrImages(step = 0.01)





def test():
  plt.close('all')
  nelx, nely = 200, 200
  lx, ly = 1., 1.
  M = SingleParamMicrostructures(nelx, nely)
  t = 0.95
  mstr = M.diamXBox(t)

  M.plotMicrostructure(mstr)

  matProp = {'type':'Hooke', 'E':1.0, 'nu':0.3}
  H = Homogenization(lx, ly, nelx, nely, 90, matProp, 1)
  start = time.perf_counter()
  # ch =H.homogenize(mstr)
  print('tiomesds ', time.perf_counter() - start)
  vf = np.mean(mstr)
  # print('ch :', ch)
  print('vf :', vf)
  plt.show(block=True)
# test()
  