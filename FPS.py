import sys
import os
from RGBDPose.utils import ply_loader
import numpy as np
import json
import transforms3d as tf3d
import OpenEXR, Imath


def load_pcd(cat):
    # load meshes
    model_vsd = ply_loader.load_ply(cat)

    return model_vsd['pts']


def main(argv):
    root = argv[0]
    samples = argv[1]
    meshes = os.listdir(argv[0])
    meshes = [k for k in meshes if k.endswith('.ply')]

    mesh_dict = dict()

    for mesh_name in meshes:
        if mesh_name[-3:] == 'ply':
            path = root + '/' + mesh_name
            pts = load_pcd(path)
            print(mesh_name)

            control_points = np.zeros((int(samples), 3), dtype=np.float32)

            # choose starting point
            norms = np.linalg.norm(pts, 2, 1)
            first_k = np.argmax(norms)
            control_points[0, :] = pts[first_k, :]

            for k in range(int(samples)-1):

                distances = []
                for q_p in pts:
                    dist_poi = 0.0
                    for p_p in control_points:
                        dist_poi += np.linalg.norm((q_p - p_p), 2)
                    distances.append(dist_poi)

                point_k = np.argmax(distances)
                control_points[k+1, :] = pts[point_k, :]
            mesh_dict.update({mesh_name[-6:-4]: control_points.tolist()})

    with open(root + '/features.json', 'w') as fp:
        json.dump(mesh_dict, fp)






if __name__ == "__main__":
    main(sys.argv[1:])