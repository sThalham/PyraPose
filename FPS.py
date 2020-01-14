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

            control_points = np.zeros((1, 3), dtype=np.float32)

            # choose starting point
            norms = np.linalg.norm(pts, 2, 1)
            first_k = np.argmax(norms)
            control_points[0, :] = pts[first_k, :]

            max_x = np.max(pts[:, 0])
            min_x = np.min(pts[:, 0])
            max_y = np.max(pts[:, 1])
            min_y = np.min(pts[:, 1])
            max_z = np.max(pts[:, 2])
            min_z = np.min(pts[:, 2])
            x_dim = (max_x - min_x) * 0.5
            y_dim = (max_y - min_y) * 0.5
            z_dim = (max_z - min_z) * 0.5
            min_side_length = min([x_dim], [y_dim], [z_dim])

            for k in range(int(samples)-1):

                distances = []
                for ind, q_p in enumerate(pts):
                    dist_poi = 0.0
                    for p_p in range(control_points.shape[0]):
                        this_mother_fucking_points_distance = np.linalg.norm((q_p - control_points[p_p, :]), 2)
                        #if this_mother_fucking_points_distance > min_side_length:
                        #    distances.append(0.0)
                        #    skipped = True
                        #    print('skip that point')
                        #    continue

                        dist_poi += np.linalg.norm((q_p - control_points[p_p, :]), 2)
                    #if skipped == False: # hell of a bad workaround
                    #    print('point not skipped')
                    distances.append(dist_poi)

                point_k = np.argmax(distances)
                cp_now = np.zeros((1, 3), dtype=np.float32)
                cp_now[0, :] = pts[point_k, :]
                print(control_points.shape)
                print(cp_now.shape)
                control_points = np.concatenate([control_points, cp_now], axis=0)
            mesh_dict.update({mesh_name[-6:-4]: control_points.tolist()})

    with open(root + '/features.json', 'w') as fp:
        json.dump(mesh_dict, fp)


if __name__ == "__main__":
    main(sys.argv[1:])