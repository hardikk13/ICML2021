# simple utility script for turning models into meshes!

import geometry as gm
import argparse
from skimage import measure
import tensorflow as tf
import pyigl as igl
import os
import numpy as np
import csv

def loadModel(modelPath, archPath = None):
    # LOAD THE MODEL
    #load serialized model
    if archPath is None:
        jsonFile = open(modelPath+'.json', 'r')
    else:
        jsonFile = open(archPath, 'r')

    sdfModel = tf.keras.models.model_from_json(jsonFile.read())
    jsonFile.close()
    #load weights
    sdfModel.load_weights(modelPath + '.h5')
    #sdfModel.summary()
    return sdfModel

if __name__ == "__main__":
    # this should handle folders of meshes, parallelizing the meshing to avail cores
    parser = argparse.ArgumentParser(description='given a sdf weight set, generate mesh')
    parser.add_argument('weightPath', help='path to weight sets!')
    parser.add_argument('meshPath', help='path to corresponding mesh geometries.') 
    parser.add_argument('--archPath', default=None)
    parser.add_argument('--res', default=128, type=int)
    args = parser.parse_args()

    trainedModels = list([f.split('.h5')[0] for f in os.listdir(args.weightPath) if '.h5' in f])

    cubeMarcher = gm.CubeMarcher()
    
    uniformGrid = cubeMarcher.createGrid(args.res)

    csvFile = open('results.csv', 'w')

    csvWriter = csv.writer(csvFile, delimiter=',')
    csvWriter.writerow(['Name', 'Grid Error', 'Surface Error', 'Importance Error'])

    for m in trainedModels:
        modelPath = os.path.join(args.weightPath, m)
        meshPath = os.path.join(args.meshPath, m)
        # print(meshPath)
        try:
            print("[INFO] Loading model: ", m)
            print("[INFO] Loading with path: ", modelPath)
            sdfModel = loadModel(modelPath, archPath=args.archPath)

            meshPath += ".obj"
            print("[INFO] Loading mesh: ", meshPath)
            mesh = gm.Mesh(meshPath, doNormalize = True)
            file_name = m + ".obj"
            mesh.save(file_name)

            print("[INFO] Inferring Grid")
            gridPred = sdfModel.predict(uniformGrid)
            # pred = np.hstack((uniformGrid, gridPred))
            # file_name = m + "predicted.csv"
            # np.savetxt(file_name, pred, delimiter=",")

            voxel_size = 2. / (args.res - 1)
            num_x = args.res
            num_y = args.res
            num_z = args.res
            print("[INFO] Inferring Grid")

            # Find the minimum point in uniformGrid
            min_pt = np.min(uniformGrid, axis=0)
            print("[INFO] min_pt", min_pt)
            # Convert nx3 uniformGrid and nx1 gridPred into an num_x*num_y*num_z array
            voxelgrid = np.ones((num_x, num_y, num_z)) # initialize the voxel grid
            num_pts_total = uniformGrid.shape[0]
            print("[INFO] # of points in the voxel_grid ", num_pts_total)
            for i in range(num_pts_total):
                # Convert x, y, z location to u, v, w integer index
                # Make sure to shift by min_pt so we get non negative indices
                x, y, z = uniformGrid[i] - min_pt
                u = int(np.round(x / voxel_size))
                v = int(np.round(y / voxel_size))
                w = int(np.round(z / voxel_size))
                # if (gridPred[i] <= 0):
                #     print("u", u, ",v ", v, ",w ", w, ", grid_pt: ", uniformGrid[i], ", value:", gridPred[i])
                voxelgrid[u, v, w] = gridPred[i]
            print("[INFO] Inferring Grid")
            # Feed voxelgrid into scikit learn marching cubes
            # Maybe don't worry about padding for now
            verts, faces, _, _ = measure.marching_cubes_lewiner(voxelgrid, level=0, spacing=(voxel_size, voxel_size, voxel_size))
            print("[INFO] generated recond")
            reconst_mesh = gm.Mesh(meshPath=None, V=igl.eigen.MatrixXd(verts), F=igl.eigen.MatrixXi(faces), doNormalize = True)
            file_path = m + "_recon.obj"
            reconst_mesh.save(file_path)
        
            # print("[INFO] Inferring Surface Points")
            # surfaceSampler = gm.PointSampler(mesh, ratio=0.0, std=0.0)
            # surfacePts = surfaceSampler.sample(100000)
            # surfacePred = sdfModel.predict(surfacePts)
            # surf_pred = np.hstack((surfacePts, surfacePred))
            # file_name = m + "_surface.csv"
            # np.savetxt(file_name, surf_pred, delimiter=",")

            # print("[INFO] Inferring Importance Points")
            # impSampler = gm.PointSampler(mesh, ratio=0.1, std=0.01)
            # impPts = impSampler.sample(100000)
            # impPred = sdfModel.predict(impPts)

            # print("[INFO] Calculating true sdf")
            # sdf = gm.SDF(mesh)
            # gridTrue = sdf.query(uniformGrid)
            # impTrue = sdf.query(impPts)
            # gt = np.hstack((uniformGrid, gridTrue))
            # file_name = m + "ground_truth.csv"
            # np.savetxt(file_name, gt, delimiter=",")

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

    


