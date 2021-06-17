import os

import meshio
import numpy as np

def GatherNodesEdges(vtu_path: str, pts_out_file: str, edges_out_file: str):
    mesh = meshio.read(vtu_path)
    assert mesh.cells[0][0] == 'tetra', 'VTU file must be for a tetrahedron mesh'
    pts = mesh.points
    cells = mesh.cells[0][1].astype(int)
    edges = set()
    num_cells = cells.shape[0]
    for i in range(num_cells):
        v0, v1, v2, v3 = np.sort(cells[i])
        edges.add((v0, v1))
        edges.add((v0, v2))
        edges.add((v0, v3))
        edges.add((v1, v2))
        edges.add((v1, v3))
        edges.add((v2, v3))
    with open(edges_out_file, 'w') as f:
        for edge in edges:
            f.write('{} {}\n'.format(edge[0], edge[1]))
    with open(pts_out_file, 'w') as f:
        for i in range(pts.shape[0]):
            f.write('{} {} {}\n'.format(pts[i, 0], pts[i, 1], pts[i, 2]))

if __name__ == '__main__':
    kDir = 'TetMeshesForHardik'
    tet_file = os.path.join(kDir, 'tetmesh_1000k.vtu')
    pts_file = os.path.join(kDir, 'tetmesh_1000k_pts.txt')
    edges_file = os.path.join(kDir, 'tetmesh_1000k_edges.txt')
    GatherNodesEdges(tet_file, pts_file, edges_file)