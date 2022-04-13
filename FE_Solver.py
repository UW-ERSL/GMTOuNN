import jax.numpy as jnp
import numpy as np
from jax import jit
import jax
from FE_templates import getKMatrixGridMeshTemplates

class JAXSolver:
  def __init__(self, mesh, material):
    self.mesh = mesh
    self.material = material
    self.Ktemplates = getKMatrixGridMeshTemplates(mesh.elemSize, 'structural')
    self.objectiveHandle = jit(self.objective)
    self.D0 = self.material.getD0elemMatrix(self.mesh)
  #-----------------------# 
  def objective(self, C):
    @jit
    def assembleK(C):
      sK = jnp.zeros((self.mesh.numElems, 8, 8)) # 8 is ndof perelem
      for k in C:
        sK += jnp.einsum('e,jk->ejk', C[k], self.Ktemplates[k])
      K = jnp.zeros((self.mesh.ndof, self.mesh.ndof))
      K = jax.ops.index_add(K, self.mesh.nodeIdx, sK.flatten())
      return K
    #-----------------------#
    @jit
    def solve(K):
      # eliminate fixed dofs for solving sys of eqns
      u_free = jax.scipy.linalg.solve(K[self.mesh.bc['free'],:][:,self.mesh.bc['free']], \
              self.mesh.bc['force'][self.mesh.bc['free']], sym_pos = True, check_finite=False)
      u = jnp.zeros((self.mesh.ndof))
      u = jax.ops.index_add(u, self.mesh.bc['free'], u_free.reshape(-1)) # homog bc wherev fixed
      return u
    #-----------------------#
    @jit
    def computeCompliance(K, u):
      J = jnp.dot(self.mesh.bc['force'].reshape(-1).T, u)
      return J
    #-----------------------#
    K = assembleK(C)
    u = solve(K)
    J = computeCompliance(K, u)
    return J
    #-----------------------#
