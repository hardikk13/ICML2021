# Create a tetrahedral-based helmet pad using both struts and implicit surfaces.

import meshio  # For reading/writing VTU and STL
import numpy as np

# Some boolean operations

def Intersection(f1: np.ndarray, f2: np.ndarray):
    return np.maximum(f1, f2)

def Union(f1: np.ndarray, f2: np.ndarray):
    return np.minimum(f1, f2)

# Subtracts f2 from f1
def Difference(f1: np.ndarray, f2: np.ndarray):
    return np.maximum(f1, -f2)

# Mesh I/O

def ReadTetrahedronMesh(path: str) -> Tuple[]:
    """Reads a VTU tetrahedron mesh. Mesh contains N points and M tets.

    Args:
        path: string path to VTU file.
    Returns:
        (points, cells) where points is (N, 3) array and cells is (M, 4)
    """
    mesh = meshio.read(path)
    assert mesh.cells[0][0] == 'tetra', 'VTU file is not a tetrahedron mesh!'
    return mesh.points, mesh.cells[0][1]

# Transformations

def TetToTetMap(xyz: np.ndarray, tet0: np.ndarray, tet1: np.ndarray) \
                -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Maps tet0 to tet1 using affine transformation.
  See https://people.sc.fsu.edu/~jburkardt/presentations/cg_lab_mapping_tetrahedrons.pdf
  
  Args:
    xyz: (N, 3) numpy array in global coordinates
    tet0, tet1: each is a (4, 3) numpy array of tet vertices in VTK ordering
  Returns:
    uvw: (N, 3) numpy array in tetrahedron local coordinates
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
  uvw = m @ xyz.reshape((3, -1)) + b # (3, N)
  return uvw.reshape((-1, 3))

# Primitive shape functions

def Tetrahedron(xyz: np.ndarray, tet: np.ndarray) -> np.ndarray:
  """Returns intersection of four halfspaces to form a tetrahedron

  Args:
    xyz: (N, 3) numpy array in global coordinates
    tet: (4, 3) numpy array of tet vertices in VTK ordering
  Returns:
    (N,) array of implicit surface values
  """
  n0 = np.cross(tet[2] - tet[0], tet[1] - tet[0])
  n1 = np.cross(tet[1] - tet[0], tet[3] - tet[0])
  n2 = np.cross(tet[2] - tet[1], tet[3] - tet[1])
  n3 = np.cross(tet[0] - tet[2], tet[3] - tet[2])
  f0 = (xyz - tet[0].reshape((1, 3))) @ n0  # (N,)
  f1 = (xyz - tet[0].reshape((1, 3))) @ n1  # (N,)
  f2 = (xyz - tet[1].reshape((1, 3))) @ n2  # (N,)
  f3 = (xyz - tet[0].reshape((1, 3))) @ n3  # (N,)
  return Intersection(Intersection(Intersection(f0, f1), f2), f3)

# Capsule to represent single strut. Original code from Inigo Quirez.
# See https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
# Capsule / Line - exact
# float sdCapsule( vec3 p, vec3 a, vec3 b, float r )
# {
#   vec3 pa = p - a, ba = b - a;
#   float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
#   return length( pa - ba*h ) - r;
# }
def Capsule(xyz: np.ndarray, p0: np.ndarray, p1: np.ndarray, r: float) -> np.ndarray:
    """Create capsule with endpoints p0, p1 and radius r.

    Args:
        xyz: (N, 3) numpy array in global coordinates
        p0: (3,) starting point
        p1: (3,) ending point
        r: radius
    Returns:
        (N,) array of implicit surface values
    """
    