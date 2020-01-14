import sys
import os
from RGBDPose.utils import ply_loader
import numpy as np
import json
import open3d
import copy


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
            colors = np.zeros(pts.shape)
            pcd_model = open3d.PointCloud()
            pcd_model.points = open3d.Vector3dVector(pts)
            pcd_model.colors = open3d.Vector3dVector(colors)
            draw_models = []
            draw_models.append(pcd_model)
            print(mesh_name)

            control_points = np.zeros((1, 3), dtype=np.float32)

            # choose starting point
            norms = np.linalg.norm(pts, 2, 1)
            first_k = np.argmax(norms)
            control_points[0, :] = pts[first_k, :]

            mesh_sphere = open3d.create_mesh_coordinate_frame(size=0.01,
                                                              origin=pts[first_k, :])  # geometry.TriangleMesh.
            mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
            draw_models.append(mesh_sphere)

            max_x = np.max(pts[:, 0])
            min_x = np.min(pts[:, 0])
            max_y = np.max(pts[:, 1])
            min_y = np.min(pts[:, 1])
            max_z = np.max(pts[:, 2])
            min_z = np.min(pts[:, 2])
            x_dim = (max_x - min_x) #* 0.5
            y_dim = (max_y - min_y) #* 0.5
            z_dim = (max_z - min_z) #* 0.5
            #min_side_length = min([x_dim], [y_dim], [z_dim])
            min_side_length = (x_dim + y_dim + z_dim)/7
            print(min_side_length)

            for k in range(int(samples)-1):

                distances = []

                for ind, q_p in enumerate(pts):
                    dist_sum = 0.0
                    skipped = False

                    for p_p in range(control_points.shape[0]):
                        dist_poi = np.linalg.norm((q_p - control_points[p_p, :]), 2)
                        dist_sum += dist_poi
                        if dist_poi < min_side_length:
                            skipped = True

                    if skipped == True: # hell of a bad workaround
                        distances.append(0.0)
                    else:
                        distances.append(dist_sum)

                point_k = np.argmax(distances)
                print(distances[point_k])
                cp_now = np.zeros((1, 3), dtype=np.float32)
                cp_now[0, :] = pts[point_k, :]
                mesh_sphere = open3d.create_mesh_coordinate_frame(size=0.01, origin=pts[point_k, :]) # geometry.TriangleMesh.
                mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
                #mesh_transform = np.ones((4, 4))
                #mesh_transform[0, 3] = pts[point_k, 0]
                #mesh_transform[1, 3] = pts[point_k, 1]
                #mesh_transform[2, 3] = pts[point_k, 2]
                #mesh_sphere.transform(mesh_transform)
                #open3d.draw_geometries([mesh_sphere])
                draw_models.append(mesh_sphere)
                control_points = np.concatenate([control_points, cp_now], axis=0)
            open3d.draw_geometries(draw_models)
            mesh_dict.update({mesh_name[-6:-4]: control_points.tolist()})

    with open(root + '/features.json', 'w') as fp:
        json.dump(mesh_dict, fp)


if __name__ == "__main__":
    main(sys.argv[1:])