import numpy as np
import jax.numpy as jnp
import jax.nn as nn
from jax import jit, random
np.random.seed(0)

rand_key = random.PRNGKey(0) # reproducibility
from jax.experimental import stax
#-----------------------#
def elementwise(fun, **fun_kwargs):
    """Layer that applies a scalar function elementwise on its inputs."""
    init_fun = lambda rng, input_shape: (input_shape, ())
    apply_fun = lambda params, inputs, **kwargs: fun(inputs, **fun_kwargs)
    return init_fun, apply_fun
Swish = elementwise(nn.swish)
#-----------------------#
class TopNet:
  def __init__(self, nnSettings):
    self.nnSettings = nnSettings
    init_fn, applyNN = self.makeNetwork(nnSettings)
    self.fwdNN = jit(lambda nnwts, x: applyNN(nnwts, x))
    _, self.wts = init_fn(rand_key, (-1, nnSettings['inputDim']))
  #-----------------------#
  def makeNetwork(self, nnSettings):
    # JAX network definition
    layers = []
    for i in range(nnSettings['numLayers']-1):
      layers.append(stax.Dense(nnSettings['numNeuronsPerLayer']))
      layers.append(Swish)#(stax.LeakyRelu)
    layers.append(stax.Dense(nnSettings['outputDim']))
    return stax.serial(*layers)
  #-----------------------#
  def forward(self, wts, x):
    self.wts = wts
    nnOut = self.fwdNN(wts, x)
    mstrType = nn.softmax(nnOut[:,0:-1])
    rho = nn.sigmoid(nnOut[:,-1]).reshape(-1)
    return mstrType, rho
  #-----------------------#
