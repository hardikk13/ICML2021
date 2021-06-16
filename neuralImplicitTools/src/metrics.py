# simple utility script for turning models into meshes!

import geometry as gm
import argparse
from skimage import measure
import tensorflow as tf
import pyigl as igl
from decimal import *
import iglhelpers
import os
import numpy as np
import csv
import time

from typing import Union

import numpy as np
from skimage import measure
# from stl import mesh

def gyroidFunc2(uniformGrid: np.ndarray) -> np.ndarray:
    """Evaluates uniform grid (N, 3) using gyroid implicit equation. Returns (N,) result."""
    x = uniformGrid[:, 0]
    y = uniformGrid[:, 1]
    z = uniformGrid[:, 2]
    # kCellSize = 0.014408772790049425*3
    kCellSize = 0.15/2.
    # 0.125/2.  # you can change this if you want
    t = 0.5  # the isovalue, change if you want
    result = (np.cos(2*np.pi*x/kCellSize) * np.sin(2*np.pi*y/kCellSize) + \
             np.cos(2*np.pi*y/kCellSize) * np.sin(2*np.pi*z/kCellSize) + \
             np.cos(2*np.pi*z/kCellSize) * np.sin(2*np.pi*x/kCellSize)) - t**2
    return result


def gyroidFunc(uniformGrid: np.ndarray) -> np.ndarray:
    """Evaluates uniform grid (N, 3) using gyroid implicit equation. Returns (N,) result."""
    x = uniformGrid[:, 0]
    y = uniformGrid[:, 1]
    z = uniformGrid[:, 2]
    kCellSize = 0.014408772790049425*3
    # 0.125/2.  # you can change this if you want
    t = 0.1  # the isovalue, change if you want
    result = (np.cos(2*np.pi*x/kCellSize) * np.sin(2*np.pi*y/kCellSize) + \
             np.cos(2*np.pi*y/kCellSize) * np.sin(2*np.pi*z/kCellSize) + \
             np.cos(2*np.pi*z/kCellSize) * np.sin(2*np.pi*x/kCellSize)) - t**2
    return result

def intersectionFunc(voxels1: np.ndarray, voxels2: np.ndarray) -> np.ndarray:
    """Takes boolean intersection of voxels1 and voxels2. Typically this is just max or smoothmax."""
    assert voxels1.shape == voxels2.shape, 'expected same shape, but got {} and {}'.format(voxels1.shape, voxels2.shape)
    return np.maximum(voxels1, voxels2)

def loadModel(modelPath, archPath = None):
    # LOAD THE MODEL
    #load serialized model
    if archPath is None:
        jsonFile = open(modelPath+'.json', 'r')
    else:
        jsonFile = open(archPath, 'r')

    sdfModel = tf.keras.models.model_from_json(jsonFile.read())
    sdfModel.summary()
    jsonFile.close()
    #load weights
    sdfModel.load_weights(modelPath + '.h5')
    #sdfModel.summary()
    return sdfModel

# Define an arbitrary hexahedron
# hex_pts = np.array([
#     [0, 0, 0],
#     [1.25, 0, -0.25],
#     [1.5, 1.5, 0],
#     [-0.25, 0.75, 0.25],
#     [-0.25, -0.5, 2],
#     [2, 0.25, 2],
#     [2, 1.5, 1.75],
#     [0, 1.75, 1.5]
# ])

hex_pts = np.array([
    [0., 0., 0.],
    [0.9, 0, 0.],
    [0.9, 0.9, 0.],
    [0, 0.9, 0.],
    [0, 0, 0.5],
    [0.9, 0., 0.5],
    [0.9, 0.9, 0.5],
    [0., 0.9, 0.5]
])

def TrilinearMap(uvw: np.ndarray, hex_pts: np.ndarray) -> np.ndarray:
    """Trilinear map between points in [0, 1]^3 to arbitrary hex defined by hex_pts.
    
    Args:
        uvw: (N, 3) array of points in [0, 1]^3 (reference configuration) to map.
        hex_pts: (8, 3) array of hex points in deformed configuration.
    Returns:
        (N, 3) array of mapped points
    """
    u = uvw[:, 0].reshape((-1, 1))  # (N, 1)
    v = uvw[:, 1].reshape((-1, 1))  # (N, 1)
    w = uvw[:, 2].reshape((-1, 1))  # (N, 1)
    p0 = hex_pts[0].reshape((1, 3))  # (1, 3)
    p1 = hex_pts[1].reshape((1, 3))  # (1, 3)
    p2 = hex_pts[2].reshape((1, 3))  # (1, 3)
    p3 = hex_pts[3].reshape((1, 3))  # (1, 3)
    p4 = hex_pts[4].reshape((1, 3))  # (1, 3)
    p5 = hex_pts[5].reshape((1, 3))  # (1, 3)
    p6 = hex_pts[6].reshape((1, 3))  # (1, 3)
    p7 = hex_pts[7].reshape((1, 3))  # (1, 3)
    return ((1 - u)*(1 - v)*(1 - w)) @ p0 + (u*(1 - v)*(1 - w)) @ p1 + \
           (u*v*(1 - w)) @ p2 + ((1 - u)*v*(1 - w)) @ p3 + \
           ((1 - u)*(1 - v)*w) @ p4 + (u*(1 - v)*w) @ p5 + \
           (u*v*w) @ p6 + ((1 - u)*v*w) @ p7

def VectorizedThreeByThreeDeterminant(m: np.ndarray) -> np.ndarray:
    """Returns the determinant of N x (3 x 3) matrices. The matrices should be represented as a single
    (N, 9) array where each row holds the following entries: [m11 m21 m31 m12 m22 m32 m13 m23 m33].
    See https://en.wikipedia.org/wiki/Determinant

    Args:
        m: (N, 9) array of N x (3 x 3) matrices to take inverse of.
    Returns:
        (N,) array of determinants
    """
    # Given a matrix:
    # [a b c]
    # [d e f]
    # [g h i]
    # The determinant is given by aei + bfg + cdh - ceg - bdi - afh
    a = m[:, 0]
    b = m[:, 3]
    c = m[:, 6]
    d = m[:, 1]
    e = m[:, 4]
    f = m[:, 7]
    g = m[:, 2]
    h = m[:, 5]
    i = m[:, 8]
    return a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h

def VectorizedThreeByThreeInverse(m: np.ndarray) -> np.ndarray:
    """Returns the inverse of N x (3 x 3) matrices. The matrices should be represented as a single
    (N, 9) array where each row holds the following entries: [m11 m21 m31 m12 m22 m32 m13 m23 m33].
    See https://en.wikipedia.org/wiki/Invertible_matrix#Inversion_of_3_%C3%97_3_matrices

    Args:
        m: (N, 9) array of N x (3 x 3) matrices to take inverse of.
    Returns:
        (N, 9) array of inverse matrices
    """
    det = VectorizedThreeByThreeDeterminant(m).reshape((-1, 1))  # (N, 1)
    # Given a matrix:
    # [a b c]
    # [d e f]
    # [g h i]
    a = m[:, 0].reshape((-1, 1))  # (N, 1)
    b = m[:, 3].reshape((-1, 1))  # (N, 1)
    c = m[:, 6].reshape((-1, 1))  # (N, 1)
    d = m[:, 1].reshape((-1, 1))  # (N, 1)
    e = m[:, 4].reshape((-1, 1))  # (N, 1)
    f = m[:, 7].reshape((-1, 1))  # (N, 1)
    g = m[:, 2].reshape((-1, 1))  # (N, 1)
    h = m[:, 5].reshape((-1, 1))  # (N, 1)
    i = m[:, 8].reshape((-1, 1))  # (N, 1)
    aa = e*i - f*h  # (N, 1)
    bb = f*g - d*i  # (N, 1)
    cc = d*h - e*g  # (N, 1)
    dd = c*h - b*i  # (N, 1)
    ee = a*i - c*g  # (N, 1)
    ff = b*g - a*h  # (N, 1)
    gg = b*f - c*e  # (N, 1)
    hh = c*d - a*f  # (N, 1)
    ii = a*e - b*d  # (N, 1)
    result = np.concatenate((aa, bb, cc, dd, ee, ff, gg, hh, ii), axis=1)  # (N, 9)
    result /= det
    return result

def DeformationGradient(uvw: np.ndarray, hex_pts: np.ndarray) -> np.ndarray:
    """Computes the deformation gradient at query points uvw for the deformation from the unit
    hexahedron to the deformed configuration given by hex_pts.
    NOTE: To keep everything vectorized, we use vectors instead of matrices.

    Args:
        uvw: (N, 3) array of points in [0, 1]^3 (reference configuration) to query.
        hex_pts: (8, 3) array of hex points in deformed configuration.
    Returns:
        (N, 9) array where the j-th column is the j-th component of the flattened deformation
        gradient matrix: [f11 f21 f31 f12 f22 f32 f13 f23 f33]
    """
    n = uvw.shape[0]
    u = uvw[:, 0].reshape((-1, 1))  # (N, 1)
    v = uvw[:, 1].reshape((-1, 1))  # (N, 1)
    w = uvw[:, 2].reshape((-1, 1))  # (N, 1)
    p0 = hex_pts[0].reshape((1, 3))  # (1, 3)
    p1 = hex_pts[1].reshape((1, 3))  # (1, 3)
    p2 = hex_pts[2].reshape((1, 3))  # (1, 3)
    p3 = hex_pts[3].reshape((1, 3))  # (1, 3)
    p4 = hex_pts[4].reshape((1, 3))  # (1, 3)
    p5 = hex_pts[5].reshape((1, 3))  # (1, 3)
    p6 = hex_pts[6].reshape((1, 3))  # (1, 3)
    p7 = hex_pts[7].reshape((1, 3))  # (1, 3)
    dmdu = ((1 - v)*(1 - w)) @ (p1 - p0) + (v*(1 - w)) @ (p2 - p3) + \
           ((1 - v)*w) @ (p5 - p4) + (v*w) @ (p6 - p7)  # (N, 3)
    dmdv = ((1 - u)*(1 - w)) @ (p3 - p0) + (u*(1 - w)) @ (p2 - p1) + \
           ((1 - u)*w) @ (p7 - p4) + (u*w) @ (p6 - p5)  # (N, 3)
    dmdw = ((1 - u)*(1 - v)) @ (p4 - p0) + (u*(1 - v)) @ (p5 - p1) + \
           (u*v) @ (p6 - p2) + ((1 - u)*v) @ (p7 - p3)  # (N, 3)
    return np.concatenate((dmdu, dmdv, dmdw), axis=1)

def EstimateInverseTrilinearMap(hex_pts: np.ndarray, xyz: np.ndarray) \
                                -> np.ndarray:
    """Given a point xyz and hex cell defined by hex_pts, this estimates
    what the element coordinates uvw would be. This should be used as
    a first guess to Newton Raphson iterations.
    NOTE: There is no "unique" or "correct" implementation to this. However,
    better estimates will reduce the number of Newton Raphson iterations needed.

    Args:
        hex_pts: (8, 3) array of hex points in deformed configuration.
        xyz: (N, 3) array of points in deformed configuration
    Returns:
        (N, 3) array of uvw estimates
    """
    bbox_min = np.min(hex_pts, axis=0)  # (3,)
    bbox_max = np.max(hex_pts, axis=0)  # (3,)
    bbox_size = bbox_max - bbox_min  # (3,)
    return (xyz - bbox_min) / bbox_size  # (N, 3)

def InverseTrilinearMap(hex_pts: np.ndarray, xyz: np.ndarray,
                        tol: Union[float, None]=1e-6, max_iter: int=100) \
                        -> np.ndarray:
    """Maps a point (x, y, z) into element coordinates (u, v, w) based on provided
    hex_pts using Newton Raphson iterations. Iterations stop when either
    convergence is reached (based on tol) or max_iter is reached.

    Args:
        hex_pts: (8, 3) array of hex points in deformed configuration.
        xyz: (N, 3) array of points in deformed configuration
        tol: None or float, used for determining convergence
        max_iter: int, max Newton iterations to perform
    Returns:
        (N, 3) array of uvw points
    """
    # Provide an estimate to (u, v, w)
    uvw = EstimateInverseTrilinearMap(hex_pts, xyz)  # (N, 3)
    for i in range(max_iter):
        jac = DeformationGradient(uvw, hex_pts)  # (N, 9)
        fn = TrilinearMap(uvw, hex_pts) - xyz  # (N, 3)
        # We need to solve the system jac * res = -fn for res.
        # We'll do it explicitly by using res = - jac_inv * fn.
        jac_inv = VectorizedThreeByThreeInverse(jac)  # (N, 9)
        res = np.zeros_like(uvw)  # (N, 3)
        res[:, 0] = -jac_inv[:, 0]*fn[:, 0] - jac_inv[:, 3]*fn[:, 1] - jac_inv[:, 6]*fn[:, 2]
        res[:, 1] = -jac_inv[:, 1]*fn[:, 0] - jac_inv[:, 4]*fn[:, 1] - jac_inv[:, 7]*fn[:, 2]
        res[:, 2] = -jac_inv[:, 2]*fn[:, 0] - jac_inv[:, 5]*fn[:, 1] - jac_inv[:, 8]*fn[:, 2]
        uvw += res  # (N, 3)
        # Maybe check every row of res to see if the row's norm is below tol
        if tol != None:
            res_norm = np.linalg.norm(res, axis=1)  # (N,)
            if np.all(res_norm < tol): break
    return uvw

if __name__ == "__main__":
    # this should handle folders of meshes, parallelizing the meshing to avail cores
    parser = argparse.ArgumentParser(description='given a sdf weight set, generate mesh')
    parser.add_argument('weightPath', help='path to weight sets!')
    parser.add_argument('meshPath', help='path to corresponding mesh geometries.') 
    parser.add_argument('--archPath', default=None)
    parser.add_argument('--res', default=128, type=int)
    args = parser.parse_args()

    trainedModels = list([f.split('.h5')[0] for f in os.listdir(args.weightPath) if '.h5' in f])
    print("TrainedModels:")
    print(trainedModels)
    cubeMarcher = gm.CubeMarcher()
    uniformGrid = cubeMarcher.createGrid(args.res)
    # parametricGrid = (uniformGrid)
    # uniformGrid = uniformGrid
    ### DEBUGGING
    print('uniformGrid type:', type(uniformGrid))
    print('uniformGrid shape:', uniformGrid.shape[0])
    print('uniformGrid shape:', uniformGrid.shape[1])
    # np.savetxt("org.csv", uniformGrid, delimiter=",")
    # uniformGrid = (uniformGrid + 1) / 2.
    assert uniformGrid.shape[1] == 3
    transformed = uniformGrid
    # transformed = InverseTrilinearMap(hex_pts, uniformGrid)
    # forward = TrilinearMap(uniformGrid, hex_pts)
    # new_forward = TrilinearMap(transformed, hex_pts)
    #     jobs.append((hex_pts, xyz))
    # with multiprocessing.Pool() as p:
    #     result = p.map(MultiprocessFunc, jobs)
    # print("Shape of the result:", result.shape)
    # transformed = transformed * 2 - 1    
    # np.savetxt("transformed.csv", transformed, delimiter=",")    
    # np.savetxt("forward.csv", forward, delimiter=",")
    # # print('uniformGrid size:', uniformGrid.size)
    # exit(0)
    # don't need this anymore # intersection_test = intersectionFunc(gyroid_voxels, gyroid_voxels)  # just making sure no runtime errors in intersectionFunc
    ### END DEBUGGING
    # csvFile = open('results.csv', 'w')

    # csvWriter = csv.writer(csvFile, delimiter=',')
    # csvWriter.writerow(['Name', 'Grid Error', 'Surface Error', 'Importance Error'])

    for m in trainedModels:
        modelPath = os.path.join(args.weightPath, m)
        meshPath = os.path.join(args.meshPath, m)
        # print(meshPath)
        try:
            print("[INFO] Loading model: ", m)
            print("[INFO] Loading with path: ", modelPath)
            start = time.time()
            sdfModel = loadModel(modelPath, archPath=args.archPath)
            end = time.time()
            print("Time to read in", end - start)

            meshPath += ".stl"
            print("[INFO] Loading mesh: ", meshPath)
            mesh = gm.Mesh(meshPath, doNormalize=True)
            file_name = m + ".obj"
            mesh.save(file_name)

            gyMarcher = gm.CubeMarcher()
            start = time.time()
            gyroid_voxels = gyroidFunc(transformed)
            gyroid_voxels2 = gyroidFunc2(transformed)
            # gyMarcher.march(uniformGrid, gyroid_voxels)
            # gymarchedMesh = gyMarcher.getMesh()
            end = time.time()
            # # gymarchedMesh.save("gyroid.obj")

            print('Gyroid eval in time: ', end - start)

            print("[INFO] Inferring Grid")
            x = np.append([[-0.032438, 0.37253597053168713, 0.45408282803863353]], 
                          [[-0.560611, -0.063418, 0.701209]], axis=0)
            print('x shape:', x.shape[0])
            print('x shape:', x.shape[1])
            print("[INFO], sdf at x: ", sdfModel.predict(x))
            exit(1)
            start = time.time()
            gridPred = sdfModel.predict(transformed)
            end = time.time()
            print("Time to eval implicit neural net", end - start)

            cubeMarcher = gm.CubeMarcher()
            cubeMarcher.march(transformed, gridPred)
            marchedMesh = cubeMarcher.getMesh()
            file_path = m + "_recon.obj"
            marchedMesh.save(file_path)
            exit(1)
            # print('Saved file', file_path)
            start = time.time()
            intersection_gridPred_gyroid = intersectionFunc(gyroid_voxels, gridPred.squeeze())
            # intersection_gridPred_gyroid = intersectionFunc(gyroid_voxels2, intersection_gridPred_gyroid)
            end = time.time()
            print("Time to perform intersection", end - start)

            cubeMarcher = gm.CubeMarcher()
            cubeMarcher.march(transformed, intersection_gridPred_gyroid)
            marchedMesh = cubeMarcher.getMesh()
            file_path = m + "_gyroid.obj"
            marchedMesh.save(file_path)
            # print('Saved file', file_path)
            exit(1)  # exit here for now

            # voxel_size = 2. / (args.res - 1)
            # num_x = args.res
            # num_y = args.res
            # num_z = args.res
            # print("[INFO] Inferring Grid")

            # # Find the minimum point in uniformGrid
            # min_pt = np.min(uniformGrid, axis=0)
            # print("[INFO] min_pt", min_pt)
            # # Convert nx3 uniformGrid and nx1 gridPred into an num_x*num_y*num_z array
            # voxelgrid = np.ones((num_x, num_y, num_z)) # initialize the voxel grid
            # num_pts_total = uniformGrid.shape[0]
            # print("[INFO] # of points in the voxel_grid ", num_pts_total)
            # for i in range(num_pts_total):
            #     # Convert x, y, z location to u, v, w integer index
            #     # Make sure to shift by min_pt so we get non negative indices
            #     x, y, z = uniformGrid[i] - min_pt
            #     u = int(np.round(x / voxel_size))
            #     v = int(np.round(y / voxel_size))
            #     w = int(np.round(z / voxel_size))
            #     # if (gridPred[i] <= 0):
            #     #     print("u", u, ",v ", v, ",w ", w, ", grid_pt: ", uniformGrid[i], ", value:", gridPred[i])
            #     voxelgrid[u, v, w] = gridPred[i]
            # print("[INFO] Inferring Grid")
            # # Feed voxelgrid into scikit learn marching cubes
            # # Maybe don't worry about padding for now
            # verts, faces, _, _ = measure.marching_cubes_lewiner(voxelgrid, level=0, spacing=(voxel_size, voxel_size, voxel_size))
            # print("[INFO] generated recond")
            # faces = faces + 1
            # reconst_mesh = gm.Mesh(meshPath=None, V=igl.eigen.MatrixXd(verts), F=igl.eigen.MatrixXi(faces))
            # print("[INFO] recond mesh initiated")
            # file_path = m + "_recon.obj"
            # reconst_mesh.save(file_path)
        
            print("[INFO] Inferring Surface Points")
            surfaceSampler = gm.PointSampler(mesh, ratio=0.0, std=0.0)
            surfacePts = surfaceSampler.sample(100000)
            surface_transformed = surfacePts
            # surface_transformed = InverseTrilinearMap(hex_pts, surfacePts)
            surfacePred = sdfModel.predict(surface_transformed)
            print("[INFO] Computing normal by taking gradient of the network")
            x_tensor = tf.convert_to_tensor(surface_transformed, dtype=tf.float32)
            with tf.GradientTape() as tape:
                print(sdfModel(x_tensor).shape)

            with tf.GradientTape() as tape:
              tape.watch(x_tensor)
              output = sdfModel(x_tensor)

            surfacePred = output
            normals = tape.gradient(surfacePred, x_tensor)
            print("[INFO] creating a single np.ndarray for output")
            r = np.hstack((surface_transformed, surfacePred, normals))            
            np.savetxt("results.csv", r, delimiter=",")
            exit(1)
            

            # print("[INFO] Inferring Importance Points")
            # impSampler = gm.PointSampler(mesh, ratio=0.1, std=0.01)
            # impPts = impSampler.sample(100000)
            # impPred = sdfModel.predict(impPts)

            # print("[INFO] Calculating true sdf")
            # sdf = gm.SDF(mesh)
            # gridTrue = sdf.query(uniformGrid)
            # impTrue = sdf.query(impPts)
            # cubeMarcher2 = gm.CubeMarcher()
            # cubeMarcher2.march(uniformGrid, gridTrue)
            # gt_marchedMesh = cubeMarcher2.getMesh()
            # file_path = m + "_org.obj"
            # gt_marchedMesh.save(file_path)

            # print("[INFO] Calculating Error")
            
            # gridError = np.mean(np.abs(gridTrue - gridPred))
            # surfaceError = np.max(np.abs(surfacePred))
            # impError = np.mean(np.abs(impTrue - impPred))

            # print("[INFO] Grid Error: ", gridError)
            # print("[INFO] Surface Error: ", surfaceError)   
            # print("[INFO] Imp Error (loss): ", impError)

            # csvWriter.writerow([m, gridError, surfaceError, impError])
            
        except Exception as e:
            print (e)
    
    csvFile.close()
