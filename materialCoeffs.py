import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
font = {'family' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)

microStrs = {\
'square':{'type':'C', 'order':5,\
'00':{'5':0,'4':1.765988e+00,'3':-2.196562e+00,'2':1.148814e+00,'1':3.776242e-01,'0':0.000000e+00},\
'11':{'5':0,'4':1.765988e+00,'3':-2.196562e+00,'2':1.148814e+00,'1':3.776242e-01,'0':0.000000e+00},\
'22':{'5':0,'4':5.809716e-01,'3':-2.169071e-01,'2':1.395702e-02,'1':7.943468e-03,'0':0.000000e+00},\
'01':{'5':0,'4':6.314688e-01,'3':-6.267269e-01,'2':3.538324e-01,'1':-2.900825e-02,'0':0.000000e+00},\
'02':{'5':0,'4':-1.451078e-15,'3':2.789744e-15,'2':-1.639482e-15,'1':3.255219e-16,'0':0.000000e+00},\
'12':{'5':0,'4':1.633770e-15,'3':-2.661224e-15,'2':1.012881e-15,'1':-4.538574e-17,'0':0.000000e+00}},\
'xBox':{'type':'C', 'order':5,\
'00':{'5':0,'4':3.049580e+00,'3':-4.219083e+00,'2':2.192740e+00,'1':7.002101e-02,'0':0.000000e+00},\
'11':{'5':0,'4':3.049580e+00,'3':-4.219083e+00,'2':2.192740e+00,'1':7.002101e-02,'0':0.000000e+00},\
'22':{'5':0,'4':8.960803e-01,'3':-1.179418e+00,'2':5.964568e-01,'1':7.016843e-02,'0':0.000000e+00},\
'01':{'5':0,'4':1.312877e+00,'3':-2.157037e+00,'2':1.182032e+00,'1':-1.226736e-02,'0':0.000000e+00},\
'02':{'5':0,'4':-6.536823e-16,'3':1.286999e-15,'2':-8.235490e-16,'1':2.029872e-16,'0':0.000000e+00},\
'12':{'5':0,'4':-3.500502e-16,'3':6.876995e-16,'2':-4.920418e-16,'1':9.474385e-17,'0':0.000000e+00}},\
'x':{'type':'C', 'order':5,\
'00':{'5':0,'4':1.742170e+00,'3':-1.565819e+00,'2':7.350411e-01,'1':1.876824e-01,'0':0.000000e+00},\
'11':{'5':0,'4':1.742170e+00,'3':-1.565819e+00,'2':7.350411e-01,'1':1.876824e-01,'0':0.000000e+00},\
'22':{'5':0,'4':5.725205e-01,'3':-7.964643e-01,'2':4.054276e-01,'1':2.016516e-01,'0':0.000000e+00},\
'01':{'5':0,'4':6.330989e-01,'3':-1.224588e+00,'2':7.535134e-01,'1':1.646271e-01,'0':0.000000e+00},\
'02':{'5':0,'4':-6.840533e-16,'3':1.232766e-15,'2':-6.840170e-16,'1':1.483102e-16,'0':0.000000e+00},\
'12':{'5':0,'4':5.005868e-16,'3':-8.298357e-16,'2':3.870303e-16,'1':-1.120707e-16,'0':0.000000e+00}},\
'diam':{'type':'C', 'order':5,\
'00':{'5':0,'4':1.476007e+00,'3':-1.352767e+00,'2':6.749168e-01,'1':2.995396e-01,'0':0.000000e+00},\
'11':{'5':0,'4':1.476007e+00,'3':-1.352767e+00,'2':6.749168e-01,'1':2.995396e-01,'0':0.000000e+00},\
'22':{'5':0,'4':8.066563e-01,'3':-9.676015e-01,'2':4.491515e-01,'1':9.809805e-02,'0':0.000000e+00},\
'01':{'5':0,'4':1.162081e+00,'3':-1.698600e+00,'2':8.179405e-01,'1':5.018857e-02,'0':0.000000e+00},\
'02':{'5':0,'4':-1.019370e-15,'3':7.461251e-16,'2':2.800116e-16,'1':-8.475976e-18,'0':0.000000e+00},\
'12':{'5':0,'4':2.931580e-15,'3':-5.600952e-15,'2':3.161665e-15,'1':-5.518624e-16,'0':0.000000e+00}},\
'star':{'type':'C', 'order':5,\
'00':{'5':5.099309e+00,'4':-1.033117e+01,'3':8.014080e+00,'2':-2.375104e+00,'1':6.918191e-01,'0':0.000000e+00},\
'11':{'5':5.099309e+00,'4':-1.033117e+01,'3':8.014080e+00,'2':-2.375104e+00,'1':6.918191e-01,'0':0.000000e+00},\
'22':{'5':4.946136e-01,'4':-5.104447e-01,'3':2.952280e-01,'2':1.349745e-03,'1':1.041980e-01,'0':0.000000e+00},\
'01':{'5':2.955840e+00,'4':-6.497245e+00,'3':5.198164e+00,'2':-1.635140e+00,'1':3.079763e-01,'0':0.000000e+00},\
'02':{'5':-2.452104e-15,'4':6.702301e-15,'3':-6.628741e-15,'2':2.747229e-15,'1':-3.581529e-16,'0':0.000000e+00},\
'12':{'5':-3.499996e-15,'4':9.657540e-15,'3':-9.491044e-15,'2':3.779495e-15,'1':-5.027157e-16,'0':0.000000e+00}},\
'diamPlus':{'type':'C', 'order':5,\
'00':{'5':5.099309e+00,'4':-1.033117e+01,'3':8.014080e+00,'2':-2.375104e+00,'1':6.918191e-01,'0':0.000000e+00},\
'11':{'5':5.099309e+00,'4':-1.033117e+01,'3':8.014080e+00,'2':-2.375104e+00,'1':6.918191e-01,'0':0.000000e+00},\
'22':{'5':4.946136e-01,'4':-5.104447e-01,'3':2.952280e-01,'2':1.349745e-03,'1':1.041980e-01,'0':0.000000e+00},\
'01':{'5':2.955840e+00,'4':-6.497245e+00,'3':5.198164e+00,'2':-1.635140e+00,'1':3.079763e-01,'0':0.000000e+00},\
'02':{'5':3.753027e-16,'4':-1.395174e-15,'3':1.508601e-15,'2':-5.468863e-16,'1':6.559061e-17,'0':0.000000e+00},\
'12':{'5':-6.866285e-16,'4':1.930563e-15,'3':-2.052985e-15,'2':9.185990e-16,'1':-1.660393e-16,'0':0.000000e+00}},\
'xDiam':{'type':'C', 'order':5,\
'00':{'5':-6.082419e-01,'4':3.464680e+00,'3':-3.528930e+00,'2':1.654915e+00,'1':1.162933e-01,'0':0.000000e+00},\
'11':{'5':-6.082419e-01,'4':3.464680e+00,'3':-3.528930e+00,'2':1.654915e+00,'1':1.162933e-01,'0':0.000000e+00},\
'22':{'5':-2.244705e-03,'4':4.589152e-01,'3':-4.982538e-01,'2':2.709899e-01,'1':1.551385e-01,'0':0.000000e+00},\
'01':{'5':-1.993904e-01,'4':1.165671e+00,'3':-1.492903e+00,'2':7.644715e-01,'1':9.206605e-02,'0':0.000000e+00},\
'02':{'5':-4.242421e-15,'4':9.669757e-15,'3':-7.895403e-15,'2':2.801855e-15,'1':-3.282246e-16,'0':0.000000e+00},\
'12':{'5':-6.520351e-16,'4':1.413946e-15,'3':-1.050480e-15,'2':2.954339e-16,'1':-6.352434e-17,'0':0.000000e+00}},\
'zBox':{'type':'C', 'order':5,\
'00':{'5':4.891562e+00,'4':-8.673194e+00,'3':5.748766e+00,'2':-1.404713e+00,'1':5.359446e-01,'0':0.000000e+00},\
'11':{'5':4.891562e+00,'4':-8.673194e+00,'3':5.748766e+00,'2':-1.404713e+00,'1':5.359446e-01,'0':0.000000e+00},\
'22':{'5':5.232406e-01,'4':-7.502414e-01,'3':6.265462e-01,'2':-1.382494e-01,'1':1.230142e-01,'0':0.000000e+00},\
'01':{'5':1.555223e+00,'4':-2.717795e+00,'3':1.765744e+00,'2':-4.209085e-01,'1':1.471981e-01,'0':0.000000e+00},\
'02':{'5':5.066595e-01,'4':-6.335077e-01,'3':3.257317e-01,'2':-9.977765e-02,'1':-9.913910e-02,'0':0.000000e+00},\
'12':{'5':5.066595e-01,'4':-6.335077e-01,'3':3.257317e-01,'2':-9.977765e-02,'1':-9.913910e-02,'0':0.000000e+00}},\
'nBox':{'type':'C', 'order':5,\
'00':{'5':4.891562e+00,'4':-8.673194e+00,'3':5.748766e+00,'2':-1.404713e+00,'1':5.359446e-01,'0':0.000000e+00},\
'11':{'5':4.891562e+00,'4':-8.673194e+00,'3':5.748766e+00,'2':-1.404713e+00,'1':5.359446e-01,'0':0.000000e+00},\
'22':{'5':5.232406e-01,'4':-7.502414e-01,'3':6.265462e-01,'2':-1.382494e-01,'1':1.230142e-01,'0':0.000000e+00},\
'01':{'5':1.555223e+00,'4':-2.717795e+00,'3':1.765744e+00,'2':-4.209085e-01,'1':1.471981e-01,'0':0.000000e+00},\
'02':{'5':-5.066595e-01,'4':6.335077e-01,'3':-3.257317e-01,'2':9.977765e-02,'1':9.913910e-02,'0':0.000000e+00},\
'12':{'5':-5.066595e-01,'4':6.335077e-01,'3':-3.257317e-01,'2':9.977765e-02,'1':9.913910e-02,'0':0.000000e+00}},\
'vDiam':{'type':'C', 'order':5,\
'00':{'5':5.241266e+00,'4':-1.001222e+01,'3':7.512779e+00,'2':-2.198696e+00,'1':5.546819e-01,'0':0.000000e+00},\
'11':{'5':1.881136e+00,'4':-3.156149e+00,'3':2.301058e+00,'2':-4.652193e-01,'1':5.381293e-01,'0':0.000000e+00},\
'22':{'5':2.492084e+00,'4':-5.112902e+00,'3':3.969181e+00,'2':-1.228707e+00,'1':2.643465e-01,'0':0.000000e+00},\
'01':{'5':3.721014e+00,'4':-8.067803e+00,'3':6.285683e+00,'2':-1.963678e+00,'1':3.534585e-01,'0':0.000000e+00},\
'02':{'5':5.109984e-15,'4':-1.501810e-14,'3':1.470763e-14,'2':-5.492796e-15,'1':7.002128e-16,'0':0.000000e+00},\
'12':{'5':2.532153e-15,'4':-5.006160e-15,'3':2.969901e-15,'2':-5.611872e-16,'1':3.499140e-18,'0':0.000000e+00}},\
'hDiam':{'type':'C', 'order':5,\
'00':{'5':1.881136e+00,'4':-3.156149e+00,'3':2.301058e+00,'2':-4.652193e-01,'1':5.381293e-01,'0':0.000000e+00},\
'11':{'5':5.241266e+00,'4':-1.001222e+01,'3':7.512779e+00,'2':-2.198696e+00,'1':5.546819e-01,'0':0.000000e+00},\
'22':{'5':2.492084e+00,'4':-5.112902e+00,'3':3.969181e+00,'2':-1.228707e+00,'1':2.643465e-01,'0':0.000000e+00},\
'01':{'5':3.721014e+00,'4':-8.067803e+00,'3':6.285683e+00,'2':-1.963678e+00,'1':3.534585e-01,'0':0.000000e+00},\
'02':{'5':-9.080534e-15,'4':2.016523e-14,'3':-1.552808e-14,'2':4.972437e-15,'1':-5.249843e-16,'0':0.000000e+00},\
'12':{'5':4.922409e-15,'4':-1.172860e-14,'3':9.229934e-15,'2':-2.678052e-15,'1':1.994746e-16,'0':0.000000e+00}}}

def plotInterpolateCoeffs():
  numPts = 100
  v = np.linspace(0, 1, numPts)
  # Cmatrix
  components = ['00', '22', '01']
  C = {}
  for c in components:
    C[c] = np.zeros((numPts))
    for pw in range(5):
      C[c] += microStrs['x'][c][str(pw)]*(v**pw)
  plt.figure()
  clrs = ['red', 'black', 'blue', 'green', 'pink']
  mrkrs = ['o', 's', 'd', 'h', '*']
  stls = [':', '--', '-', '-.', '-']
  for ctr, c in enumerate(components):
    plt.plot(v, C[c], color = clrs[ctr], linestyle = stls[ctr], marker = mrkrs[ctr], markevery=20, label = f'$C{c}$')
  plt.xlabel('v')
  plt.ylabel('C')
  plt.legend()
  plt.show()
  
# plotInterpolateCoeffs()