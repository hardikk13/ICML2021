from typing import Callable, Tuple

import meshio
import numpy as np
from skimage import measure
from stl import mesh

# The unit reference hexahedron in [0, 1]^3
def UnitHexahedron() -> np.ndarray:
  return np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
  ])

def UnitGyroid(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
  """Evaluates x, y, z using gyroid function. If any point is outside [0, 1]^3,
  this returns kSmallPositiveValue.

  Args:
    x, y, z: numpy arrays of the same shape
  Returns:
    numpy array of same shape as inputs
  """
  kSmallPositiveValue = 1e-6
  kIsovalue = 0.2
  result = (np.cos(2*np.pi*x)*np.sin(2*np.pi*y) + \
            np.cos(2*np.pi*y)*np.sin(2*np.pi*z) + \
            np.cos(2*np.pi*z)*np.sin(2*np.pi*x))**2 - kIsovalue**2
  result[x < 0] = kSmallPositiveValue
  result[y < 0] = kSmallPositiveValue
  result[z < 0] = kSmallPositiveValue
  result[x > 1] = kSmallPositiveValue
  result[y > 1] = kSmallPositiveValue
  result[z > 1] = kSmallPositiveValue
  return result

def TetToTetMap(x: np.ndarray, y: np.ndarray, z: np.ndarray, tet0: np.ndarray,
                tet1: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Maps tet0 to tet1 using affine transformation.
  See https://people.sc.fsu.edu/~jburkardt/presentations/cg_lab_mapping_tetrahedrons.pdf
  
  Args:
    x, y, z: each is an (N,) numpy array
    tet0, tet1: each is a (4, 3) numpy array of tet vertices in VTK ordering
  Returns:
    xt, yt, zt: transformed points of same shape as x, y, z
  """
  m0 = np.concatenate(((tet0[0] - tet0[3]).reshape((-1, 1)),
                       (tet0[1] - tet0[3]).reshape((-1, 1)),
                       (tet0[2] - tet0[3]).reshape((-1, 1))), axis=1)
  m1 = np.concatenate(((tet1[0] - tet1[3]).reshape((-1, 1)),
                       (tet1[1] - tet1[3]).reshape((-1, 1)),
                       (tet1[2] - tet1[3]).reshape((-1, 1))), axis=1)
  m0_inv = np.linalg.inv(m0)
  m = m1 @ m0_inv
  b = (tet1[3] - m @ tet0[3]).reshape((3, 1))  # (3, 1)
  xyz = np.concatenate((x.reshape((1, -1)),
                        y.reshape((1, -1)),
                        z.reshape((1, -1))), axis=0)  # (3, N)
  uvw = m @ xyz + b # (3, N)
  return uvw[0, :], uvw[1, :], uvw[2, :]

def Tetrahedron(x: np.ndarray, y: np.ndarray, z: np.ndarray, tet: np.ndarray) \
                -> np.ndarray:
  """Returns intersection of four halfspaces to form a tetrahedron"""
  n0 = np.cross(tet[2] - tet[0], tet[1] - tet[0])
  n1 = np.cross(tet[1] - tet[0], tet[3] - tet[0])
  n2 = np.cross(tet[2] - tet[1], tet[3] - tet[1])
  n3 = np.cross(tet[0] - tet[2], tet[3] - tet[2])
  xyz = np.concatenate((x.reshape((-1, 1)),
                        y.reshape((-1, 1)),
                        z.reshape((-1, 1))), axis=1) # (N, 3)
  f0 = (xyz - tet[0].reshape((1, 3))) @ n0  # (N,)
  f1 = (xyz - tet[0].reshape((1, 3))) @ n1  # (N,)
  f2 = (xyz - tet[1].reshape((1, 3))) @ n2  # (N,)
  f3 = (xyz - tet[0].reshape((1, 3))) @ n3  # (N,)
  return np.maximum(np.maximum(np.maximum(f0, f1), f2), f3)

def CreateTetGyroidFunctor(tet0: np.ndarray, tet1: np.ndarray) -> Callable:
  def f(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    # Map xyz from tet0 to tet1
    xm, ym, zm = TetToTetMap(x, y, z, tet0, tet1)
    # Evaluate xm, ym, zm on unit gyroid function
    val = UnitGyroid(xm, ym, zm)
    # Chop result with implicit function for tet1
    tet1_val = Tetrahedron(xm, ym, zm, tet1)
    val = np.maximum(val, tet1_val)
    return val
  return f

# For each tet, construct an implicit surface of the tet with gyroid inside
# We do this using the following steps:
# - Map coordinates in global space to [0, 1]^3 using affine map between tets
# - Evaluate point in [0, 1]^3 using unit gyroid function

# A single arbitrary hex
# Change this if you want. It just needs to not be degenerate.
hex_pts = np.array([
  [0, 0, 0],
  [1.25, 0, -0.25],
  [1.5, 1.5, 0],
  [-0.25, 0.75, 0.25],
  [-0.25, -0.5, 2],
  [2, 0.25, 2],
  [2, 1.5, 1.75],
  [0, 1.75, 1.5]
])
# hex_pts = np.array([
#   [-1, -1, -1],
#   [1, -1, -1],
#   [1, 1, -1],
#   [-1, 1, -1],
#   [-1, -1, 1],
#   [1, -1, 1],
#   [1, 1, 1],
#   [-1, 1, 1],
# ])

# Break hex into tet
# NOTE: There's no *right* way to do this so we just choose *a* method.
# See https://arxiv.org/pdf/1801.01288.pdf for possibilities
tet_cells = np.array([
  [0, 5, 1, 3],
  [4, 5, 0, 3],
  [1, 2, 3, 5],
  [3, 2, 6, 5],
  [3, 6, 7, 5],
  [4, 3, 7, 5],
])

def GetTetVertices(indices: np.ndarray, pts: np.ndarray) -> np.ndarray:
  """Returns pts[indices]"""
  result = np.zeros((4, 3))
  for i in range(4):
    result[i] = pts[indices[i]]
  return result

# List of tet gyroid implicit funcs
funcs = [CreateTetGyroidFunctor(GetTetVertices(tet_cells[i], hex_pts),
                                GetTetVertices(tet_cells[i], UnitHexahedron())) \
         for i in range(6)]


# Take union of all the funcs evaluated on grid
n = 200 # num voxels
domain = np.linspace(-1, 3, n)
voxel_size = (domain[-1] - domain[0]) / (n - 1)
xgrid, ygrid, zgrid = np.meshgrid(domain, domain, domain, indexing='ij')
xvec = xgrid.reshape((-1, 1))
yvec = ygrid.reshape((-1, 1))
zvec = zgrid.reshape((-1, 1))
vals = funcs[0](xvec, yvec, zvec)
for i in range(1, len(funcs)):
  vals = np.minimum(vals, funcs[i](xvec, yvec, zvec))

# Perform marching cubes so you can actually visualize the result

def MeshFromVertsFaces(vertices, faces):
  m = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
  for i, f in enumerate(faces):
    for j in range(3):
        m.vectors[i][j] = vertices[f[j],:]
  return m

verts, faces, _, _ = measure.marching_cubes_lewiner(vals.reshape(xgrid.shape), level=0, spacing=(voxel_size, voxel_size, voxel_size))
verts += domain[0]
m = MeshFromVertsFaces(verts, faces)
# out_path = '/content/drive/My Drive/Shared/HexToTet/tet_gyroid.stl'
# m.save(out_path)