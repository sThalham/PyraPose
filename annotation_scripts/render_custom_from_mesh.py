import os
import sys
import yaml
import cv2
import numpy as np
import datetime
import copy
import transforms3d as tf3d
import time
import random
import json
import open3d
import math

from pathlib import Path

from misc import manipulate_RGB, toPix_array, toPix, calculate_feature_visibility

# Import bop_renderer and bop_toolkit.
# ------------------------------------------------------------------------------
bop_renderer_path = '/home/stefan/bop_renderer/build'
sys.path.append(bop_renderer_path)

import bop_renderer


def lookAt(eye, target, up):
    # eye is from
    # target is to
    # expects numpy arrays
    f = eye - target
    f = f/np.linalg.norm(f)

    s = np.cross(up, f)
    s = s/np.linalg.norm(s)
    u = np.cross(f, s)
    u = u/np.linalg.norm(u)

    tx = np.dot(s, eye.T)
    ty = np.dot(u, eye.T)
    tz = np.dot(f, eye.T)

    m = np.zeros((4, 4), dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = f
    m[:, 3] = [tx, ty, tz, 1]

    #m[0, :-1] = s
    #m[1, :-1] = u
    #m[2, :-1] = -f
    #m[-1, -1] = 1.0

    return m

def m3dLookAt(eye, target, up):
    mz = normalize(eye - target) # inverse line of sight
    mx = normalize( cross( up, mz ) )
    my = normalize( cross( mz, mx ) )
    tx =  dot( mx, eye )
    ty =  dot( my, eye )
    tz = -dot( mz, eye )
    return np.array([mx[0], my[0], mz[0], 0, mx[1], my[1], mz[1], 0, mx[2], my[2], mz[2], 0, tx, ty, tz, 1])


if __name__ == "__main__":

    #mesh_path = sys.argv[1]
    #mesh_path = '/home/stefan/data/Meshes/linemod_13' # linemod
    #mesh_path = '/media/stefan/CBED-050F/MMAssist/models_reconstructed/ply' #fronius

    # Fronius
    #mesh_path = '/home/stefan/data/Meshes/Meshes_color_invert/Fronius_enum/'
    #background = '/home/stefan/data/datasets/cocoval2017/'
    #target = '/home/stefan/data/train_data/sanity_check/'

    # InDex
    mesh_path = '/home/stefan/data/Meshes/CIT_color/'
    background = '/home/stefan/data/datasets/cocoval2017/'
    target = '/home/stefan/data/train_data/CIT/'

    # metal Markus

    objsperimg = 9

    #print(open3d.__version__)
    #pcd = open3d.io.read_point_cloud("/media/stefan/CBED-050F/MMAssist/models_reconstructed/pcd/sidepanel_left/3D_model.pcd")
    #open3d.visualization.draw_geometries([pcd])
    #distances = pcd.compute_nearest_neighbor_distance()
    #avg_dist = np.mean(distances)
    #radius = 4 * avg_dist
    #bpa_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, open3d.utility.DoubleVector(
    #    [radius, radius * 2]))
    #print('stuck here')
    #dec_mesh = bpa_mesh.simplify_quadric_decimation(100000)
    #dec_mesh = bpa_mesh
    #dec_mesh.remove_degenerate_triangles()
    #dec_mesh.remove_duplicated_triangles()
    #dec_mesh.remove_duplicated_vertices()
    #dec_mesh.remove_non_manifold_edges()
    #open3d.io.write_triangle_mesh(filename="/media/stefan/CBED-050F/MMAssist/models_reconstructed/ply/seite_rechts.ply", mesh=dec_mesh, write_ascii=True)


    visu = False
    # kinect V1
    #resX = 640
    #resY = 480
    #fx = 572.41140
    #fy = 573.57043
    #cx = 325.26110
    #cy = 242.04899
    # Realsense d415
    resX = 640
    resY = 480
    fx = 615.40063
    fy = 615.04529
    cx = 312.87567
    cy = 250.85875
    #a_x = 57°
    #a_y = 43°
    K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]

    ren = bop_renderer.Renderer()
    ren.init(resX, resY)
    mesh_id = 1
    categories = []

    for mesh_now in os.listdir(mesh_path):
        mesh_path_now = os.path.join(mesh_path, mesh_now)
        if mesh_now[-4:] != '.ply':
            continue
        mesh_id = int(mesh_now[:-4])
        ren.add_object(mesh_id, mesh_path_now)
        categories.append(mesh_id)
        mesh_id += 1

    #mesh_id = 3
    #ren.add_object(3, '/home/stefan/data/Meshes/CIT_color/03.ply')
    #categories.append(mesh_id)
    #mesh_id = 6
    #ren.add_object(6, '/home/stefan/data/Meshes/CIT_color/06.ply')
    #categories.append(mesh_id)

    '''
    # InDex cube
    threeD_boxes = np.ndarray((2, 8, 3), dtype=np.float32)
    threeD_boxes[1, :, :] = np.array([[0.025, 0.025, 0.025],  # Metal [30, 210, 70] links
                                      [0.025, 0.025, -0.025],
                                      [0.025, -0.025, -0.025],
                                      [0.025, -0.025, 0.025],
                                      [-0.025, 0.025, 0.025],
                                      [-0.025, 0.025, -0.025],
                                      [-0.025, -0.025, -0.025],
                                      [-0.025, -0.025, 0.025]])

    # Metal Markus

    threeD_boxes = np.ndarray((2, 8, 3), dtype=np.float32)
    threeD_boxes[1, :, :] = np.array([[0.015, 0.105, 0.035],  # Metal [30, 210, 70] links
                                      [0.015, 0.105, -0.035],
                                      [0.015, -0.105, -0.035],
                                      [0.015, -0.105, 0.035],
                                      [-0.015, 0.105, 0.035],
                                      [-0.015, 0.105, -0.035],
                                      [-0.015, -0.105, -0.035],
                                      [-0.015, -0.105, 0.035]])
    '''

    # interlude for debugging
    mesh_info = os.path.join(mesh_path, 'models_info.yml')
    threeD_boxes = np.ndarray((34, 8, 3), dtype=np.float32)
    #sym_cont = np.ndarray((34, 3), dtype=np.float32)
    #sym_disc = np.ndarray((34, 9), dtype=np.float32)

    #for key, value in yaml.load(open(mesh_info)).items():
    for key, value in json.load(open(mesh_info)).items():
        fac = 0.001
        x_minus = value['min_x']
        y_minus = value['min_y']
        z_minus = value['min_z']
        x_plus = value['size_x'] + x_minus
        y_plus = value['size_y'] + y_minus
        z_plus = value['size_z'] + z_minus
        three_box_solo = np.array([[x_plus, y_plus, z_plus],
                                   [x_plus, y_plus, z_minus],
                                   [x_plus, y_minus, z_minus],
                                   [x_plus, y_minus, z_plus],
                                   [x_minus, y_plus, z_plus],
                                   [x_minus, y_plus, z_minus],
                                   [x_minus, y_minus, z_minus],
                                   [x_minus, y_minus, z_plus]])

        threeD_boxes[int(key), :, :] = three_box_solo * fac

        #if "symmetries_continuous" in value:
        #    sym_cont[int(key), :] = np.asarray(value['symmetries_continuous'][0]['axis'], dtype=np.float32)
        #elif "symmetries_discrete" in value:
        #    sym_disc[int(key), :] = np.array(value['symmetries_discrete'])
        #else:
        #    pass


    '''
    threeD_boxes = np.ndarray((4, 8, 3), dtype=np.float32)
    threeD_boxes[2, :, :] = np.array([[0.060, 0.1, 0.03],  # Seite-AC [120, 198, 45] links
                                      [0.060, 0.1, -0.03],
                                      [0.060, -0.1, -0.03],
                                      [0.060, -0.1, 0.03],
                                      [-0.060, 0.1, 0.03],
                                      [-0.060, 0.1, -0.03],
                                      [-0.060, -0.1, -0.03],
                                      [-0.060, -0.1, 0.03]])
    threeD_boxes[0, :, :] = np.array([[0.05, 0.04, 0.03],  # AC-Abdeckung [81, 68, 25] expand last dim
                                      [0.05, 0.04, -0.03],
                                      [0.05, -0.04, -0.03],
                                      [0.05, -0.04, 0.03],
                                      [-0.05, 0.04, 0.03],
                                      [-0.05, 0.04, -0.03],
                                      [-0.05, -0.04, -0.03],
                                      [-0.05, -0.04, 0.03]])
    threeD_boxes[1, :, :] = np.array([[0.05, 0.04, 0.03],  # DC [81, 72, 38]
                                      [0.05, 0.04, -0.03],
                                      [0.05, -0.04, -0.03],
                                      [0.05, -0.04, 0.03],
                                      [-0.05, 0.04, 0.03],
                                      [-0.05, 0.04, -0.03],
                                      [-0.05, -0.04, -0.03],
                                      [-0.05, -0.04, 0.03]])
    threeD_boxes[3, :, :] = np.array([[0.060, 0.1, 0.03],  # Seite-DC [120, 206, 56] rechts
                                      [0.060, 0.1, -0.03],
                                      [0.060, -0.1, -0.03],
                                      [0.060, -0.1, 0.03],
                                      [-0.060, 0.1, 0.03],
                                      [-0.060, 0.1, -0.03],
                                      [-0.060, -0.1, -0.03],
                                      [-0.060, -0.1, 0.03]])
    '''

    now = datetime.datetime.now()
    dateT = str(now)

    dict = {"info": {
        "description": "tless",
        "url": "cmp.felk.cvut.cz/t-less/",
        "version": "1.0",
        "year": 2020,
        "contributor": "Stefan Thalhammer",
        "date_created": dateT
    },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    dictVal = copy.deepcopy(dict)

    annoID = 0
    gloCo = 1
    times = 0
    loops = 6

    syns = os.listdir(background)
    all_data = (len(syns) * loops) + 1

    for o_idx in range(1,loops):
        for bg_img_path in syns[:10]:
            start_t = time.time()

            bg_img_path_j = os.path.join(background, bg_img_path)

            bg_img = cv2.imread(bg_img_path_j)
            bg_x, bg_y, _ = bg_img.shape

            if bg_y > bg_x:
                bg_img = np.swapaxes(bg_img, 0, 1)

            bg_img = cv2.resize(bg_img, (resX, resY))

            samp = int(bg_img_path[:-4])

            template_samp = '00000'
            imgNum = str(o_idx) + template_samp[:-len(str(samp))] + str(samp)
            img_id = int(imgNum)
            imgNam = imgNum + '.png'
            iname = str(imgNam)

            fileName = target + 'images/train/' + imgNam[:-4] + '_rgb.png'
            myFile = Path(fileName)

            cnt = 0
            mask_ind = 0
            mask_img = np.zeros((480, 640), dtype=np.uint8)
            visib_img = np.zeros((480, 640, 3), dtype=np.uint8)

            boxes3D = []
            calib_K = []
            zeds = []
            renderings = []
            rotations = []
            translations = []
            visibilities = []
            bboxes = []
            full_visib = []
            areas = []
            mask_idxs = []
            poses = []
            
            obj_ids = np.random.choice(categories, size=objsperimg, replace=True)

            right, top = False, False
            seq_obj = 0
            for objID in obj_ids:
                # sample rotation and append
                R_ren = tf3d.euler.euler2mat((np.random.rand() * 2 * math.pi) - math.pi, (np.random.rand() * 2 * math.pi) - math.pi, (np.random.rand() * 2 * math.pi) - math.pi)
                # CIT
                z = 0.4 + np.random.rand() * 0.6
                if objID == 2 or objID == 5:
                    x = (2 * (0.35 * z)) * np.random.rand() - (0.35 * z)  # 0.55 each side kinect
                    y = (2 * (0.2 * z)) * np.random.rand() - (0.2 * z)  # 0.40 each side kinect
                else:
                    x = (2 * (0.45 * z)) * np.random.rand() - (0.45 * z)  # 0.55 each side kinect
                    y = (2 * (0.3 * z)) * np.random.rand() - (0.3 * z)  # 0.40 each side kinect
                # fronius
                #z = 0.6 + np.random.rand() * 1.0
                # InDex
                #z = 0.3 + np.random.rand() * 1.0
                #x = (2 * (0.45 * z)) * np.random.rand() - (0.45 * z) # 0.55 each side kinect
                #y = (2 * (0.3 * z)) * np.random.rand() - (0.3 * z) # 0.40 each side kinect
                #x = (0.45 * 2 * z) * np.random.rand() - (0.45 * z)
                #y = (0.3 * 2 * z) * np.random.rand() - (0.3 * z)

                # metal_Markus
                '''
                if right == False and top == False:
                    x = (-0.45 * z) * np.random.rand()
                    y = (-0.3 * z) * np.random.rand()
                elif right == False and top == True:
                    x = (-0.45 * z) * np.random.rand()
                    y = (0.3 * z) * np.random.rand()
                elif right == True and top == False:
                    x = (0.45 * z) * np.random.rand()
                    y = (-0.3 * z) * np.random.rand()
                elif right == True and top == True:
                    x = (0.45 * z) * np.random.rand()
                    y = (0.3 * z) * np.random.rand()
                '''
                '''
                # InDex cube
                if right == False and top == False:
                    x = ((-0.4 * z) * np.random.rand()) - 0.05
                    y = ((-0.25 * z) * np.random.rand()) - 0.05
                elif right == False and top == True:
                    x = ((-0.4 * z) * np.random.rand()) - 0.05
                    y = ((0.25 * z) * np.random.rand()) + 0.05
                elif right == True and top == False:
                    x = ((0.4 * z) * np.random.rand()) + 0.05
                    y = ((-0.25 * z) * np.random.rand()) - 0.05
                elif right == True and top == True:
                    x = ((0.4 * z) * np.random.rand()) + 0.05
                    y = ((0.25 * z) * np.random.rand()) + 0.05

                if seq_obj == 0 or seq_obj == 2:
                    top = True
                else:
                    top = False
                if seq_obj > 0:
                    right = True
                seq_obj += 1
                '''

                t = np.array([[x, y, z]]).T
                rotations.append(R_ren)
                translations.append(t)
                zeds.append(z)

                R_list = R_ren.flatten().tolist()
                t_list = t.flatten().tolist()

                # pose [x, y, z, roll, pitch, yaw] for anno
                R = np.asarray(R_ren, dtype=np.float32)
                rot = tf3d.quaternions.mat2quat(R.reshape(3, 3))
                rot = np.asarray(rot, dtype=np.float32)
                tra = np.asarray(t, dtype=np.float32)
                pose = [tra[0], tra[1], tra[2], rot[0], rot[1], rot[2], rot[3]]
                pose = [np.asscalar(pose[0]), np.asscalar(pose[1]), np.asscalar(pose[2]),
                        np.asscalar(pose[3]), np.asscalar(pose[4]), np.asscalar(pose[5]), np.asscalar(pose[6])]
                trans = np.asarray([pose[0], pose[1], pose[2]], dtype=np.float32)
                #R = tf3d.quaternions.quat2mat(np.asarray([pose[3], pose[4], pose[5], pose[6]], dtype=np.float32))

                #3Dbox for visualization
                tDbox = R.reshape(3, 3).dot(threeD_boxes[objID, :, :].T).T
                tDbox = tDbox + np.repeat(trans[:, np.newaxis].T, 8, axis=0)
                box3D = toPix_array(tDbox, fx=fx, fy=fy, cx=cx, cy=cy)
                box3D = np.reshape(box3D, (16))
                box3D = box3D.tolist()

                poses.append(pose)
                calib_K.append(K)
                boxes3D.append(box3D)

                # light, render and append
                light_pose = [np.random.rand() * 3 - 1.0, np.random.rand() * 2 - 1.0, 0.0]
                #light_color = [np.random.rand() * 0.1 + 0.9, np.random.rand() * 0.1 + 0.9, np.random.rand() * 0.1 + 0.9]
                # standard
                light_color = [1.0, 1.0, 1.0]
                light_ambient_weight = 0.2 + np.random.rand() * 0.5 # pretty good
                #light_ambient_weight = 0.2 + np.random.rand() * 0.8
                # Fronius
                #if objID != 1:
                #    light_diffuse_weight = 0.75 + np.random.rand() * 0.25
                #    light_spec_weight = 0.2 + np.random.rand() * 0.3
                #    light_spec_shine = np.random.rand() * 2.0
                #else:
                #    light_diffuse_weight = 0.15 + np.random.rand() * 0.25
                #    light_spec_weight = 0.5 + np.random.rand() * 0.4
                #    light_spec_shine = np.random.rand() * 6.0
                # CIT
                if objID == 3 or objID == 6:
                    light_diffuse_weight = 0.3 + np.random.rand() * 0.2
                    light_spec_weight = 0.6 + np.random.rand() * 0.4
                    light_spec_shine = 0.5 + np.random.rand() * 1.0
                elif objID == 1: # fine with that
                    light_diffuse_weight = 0.6 + np.random.rand() * 0.3
                    light_spec_weight = 0.1 + np.random.rand() * 0.3
                    light_spec_shine = 0.9 + np.random.rand() * 0.2
                #elif objID == 8 or objID == 9:
                #    light_diffuse_weight = 0.25 + np.random.rand() * 0.2
                #    light_spec_weight = 0.45 + np.random.rand() * 0.3
                #    light_spec_shine = 0.5 + np.random.rand() * 0.75
                else: # fine with that
                    light_diffuse_weight = 0.4 + np.random.rand() * 0.3
                    light_spec_weight = 0.4 + np.random.rand() * 0.6
                    light_spec_shine = 0.5 + np.random.rand() * 0.25


                ren.set_light(light_pose, light_color, light_ambient_weight, light_diffuse_weight, light_spec_weight, light_spec_shine)
                ren.render_object(objID, R_list, t_list, fx, fy, cx, cy)
                rgb_img = ren.get_color_image(objID)
                renderings.append(rgb_img)

                # render for visibility mask
                z_straight = np.linalg.norm
                t_list = np.array([[0.0, 0.0, t[2]]]).T
                t_list = t_list.flatten().tolist()
                T_2obj = lookAt(t.T[0], np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
                R_2obj = T_2obj[:3, :3]
                t_2obj = T_2obj[:3, 3]
                R_fv = np.dot(R_2obj, np.linalg.inv(R).T)
                R_list = R_fv.flatten().tolist()
                ren.render_object(objID, R_list, t_list, fx, fy, cx, cy)
                full_visib_img = ren.get_color_image(objID)
                full_visib.append(full_visib_img)

            zeds = np.asarray(zeds, dtype=np.float32)
            low2high = np.argsort(zeds)
            high2low = low2high[::-1]
            full_seg = []

            for v_idx in low2high:

                obj_id = int(obj_ids[v_idx])
                ren_img = renderings[v_idx]

                # partial visibility mask
                partial_visib_img = np.where(visib_img > 0, 0, ren_img)
                partial_visib_mask = np.nan_to_num(partial_visib_img, copy=True, nan=0, posinf=0, neginf=0)
                partial_visib_mask = np.where(np.any(partial_visib_mask, axis=2) > 0, 1, 0)
                partial_mask_surf = np.sum(partial_visib_mask)
                #partvisibName = target + 'images/train/' + imgNam[:-4] + str(v_idx) + '_pv.png'
                #cv2.imwrite(partvisibName, partial_visib_mask*255)
                full_visib_img = full_visib[v_idx]
                full_visib_mask = np.nan_to_num(full_visib_img, copy=True, nan=0, posinf=0, neginf=0)
                full_visib_mask = np.where(np.any(full_visib_mask, axis=2) > 0, 1, 0)
                surf_visib = np.sum(full_visib_mask)
                #fullvisibName = target + 'images/train/' + imgNam[:-4] + str(v_idx) + '_fv.png'
                #cv2.imwrite(fullvisibName, full_visib_mask*255)
                # some systematic error in visibility calculation, yet I can't point the finger at it
                visib_fract = float(partial_mask_surf / surf_visib)
                if visib_fract > 1.0:
                    visib_fract = float(1.0)
                visibilities.append(visib_fract)
                visib_img = np.where(visib_img > 0, visib_img, ren_img)

                # compute bounding box and append
                non_zero = np.nonzero(partial_visib_mask)
                surf_ren = np.sum(non_zero[0])
                if non_zero[0].size != 0:
                    bb_xmin = np.nanmin(non_zero[1])
                    bb_xmax = np.nanmax(non_zero[1])
                    bb_ymin = np.nanmin(non_zero[0])
                    bb_ymax = np.nanmax(non_zero[0])
                    obj_bb = [int(bb_xmin), int(bb_ymin), int(bb_xmax - bb_xmin), int(bb_ymax - bb_ymin)]
                    # out of order with other lists
                    bboxes.append(obj_bb)
                    area = int(obj_bb[2] * obj_bb[3])
                    areas.append(area)
                else:
                    area = int(0)
                    obj_bb = [int(0), int(0), int(0), int(0)]
                    bboxes.append(obj_bb)
                    areas.append(area)

                bg_img = np.where(partial_visib_img > 0, partial_visib_img, bg_img)

                # mask calculation
                mask_id = mask_ind + 1
                mask_img = np.where(partial_visib_img.any(axis=2) > 0, mask_id, mask_img)
                mask_ind = mask_ind + 1

                annoID = annoID + 1
                pose = poses[v_idx]
                box3D = boxes3D[v_idx]

                #sym_cont_anno = []
                #sym_disc_anno = []
                #if obj_id == 3:
                #    sym_disc_anno = sym_disc[obj_id, :].tolist()

                #print(sym_disc_anno)

                tempTA = {
                    "id": annoID,
                    "image_id": img_id,
                    "category_id": obj_id,
                    "bbox": obj_bb,
                    "pose": pose,
                    "segmentation": box3D,
                    "mask_id": mask_id,
                    "area": area,
                    "iscrowd": 0,
                    "feature_visibility": visib_fract,
                }

                dict["annotations"].append(tempTA)
                cnt = cnt + 1

            tempTL = {
                "url": "https://bop.felk.cvut.cz/home/",
                "id": img_id,
                "name": iname,
            }
            dict["licenses"].append(tempTL)

            if myFile.exists():
                print('File exists, skip encoding, ', fileName)
            else:
                cv2.imwrite(fileName, bg_img)
                print("storing image in : ", fileName)
                mask_safe_path = fileName[:-8] + '_mask.png'
                cv2.imwrite(mask_safe_path, mask_img.astype(np.int8))

                tempTV = {
                    "license": 2,
                    "url": "https://bop.felk.cvut.cz/home/",
                    "file_name": iname,
                    "height": resY,
                    "width": resX,
                    "fx": fx,
                    "fy": fy,
                    "cx": cx,
                    "cy": cy,
                    "date_captured": dateT,
                    "id": img_id,
                }
                dict["images"].append(tempTV)

                if visu is True:

                    boxes3D = [boxes3D[x] for x in low2high]
                    obj_ids = [obj_ids[x] for x in low2high]
                    #boxes3D = boxes3D[low2high]
                    #obj_ids = obj_ids[low2high]
                    img = bg_img
                    for i, bb in enumerate(bboxes):

                        bb = np.array(bb)
                        cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[0] + bb[2]), int(bb[1] + bb[3])),
                                      (255, 255, 255), 2)

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        bottomLeftCornerOfText = (int(bb[0]), int(bb[1]))
                        fontScale = 1
                        fontColor = (0, 0, 0)
                        fontthickness = 1
                        lineType = 2
                        gtText = str(obj_ids[i])

                        fontColor2 = (255, 255, 255)
                        fontthickness2 = 3
                        cv2.putText(img, gtText,
                                    bottomLeftCornerOfText,
                                    font,
                                    fontScale,
                                    fontColor2,
                                    fontthickness2,
                                    lineType)

                        pose = np.asarray(boxes3D[i], dtype=np.float32)

                        colR = 250
                        colG = 25
                        colB = 175

                        img = cv2.line(img, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), (130, 245, 13), 2)
                        img = cv2.line(img, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), (50, 112, 220), 2)
                        img = cv2.line(img, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), (50, 112, 220), 2)
                        img = cv2.line(img, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), (50, 112, 220), 2)
                        img = cv2.line(img, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), (colR, colG, colB),
                                       2)
                        img = cv2.line(img, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()),
                                       (colR, colG, colB), 2)
                        img = cv2.line(img, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()),
                                       (colR, colG, colB), 2)
                        img = cv2.line(img, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()),
                                       (colR, colG, colB), 2)
                        img = cv2.line(img, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()),
                                       (colR, colG, colB), 2)
                        img = cv2.line(img, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()),
                                       (colR, colG, colB), 2)
                        img = cv2.line(img, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()),
                                       (colR, colG, colB), 2)
                        img = cv2.line(img, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()),
                                       (colR, colG, colB), 2)

                        '''
                        img = cv2.line(img, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), (130, 245, 13), 2)
                        img = cv2.line(img, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), (50, 112, 220), 2)
                        img = cv2.line(img, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), (50, 112, 220), 2)
                        img = cv2.line(img, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), (50, 112, 220), 2)
                        img = cv2.line(img, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), (colR, colG, colB),
                                           2)
                        img = cv2.line(img, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()),
                                           (colR, colG, colB), 2)
                        img = cv2.line(img, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()),
                                           (colR, colG, colB), 2)
                        img = cv2.line(img, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()),
                                           (colR, colG, colB), 2)
                        img = cv2.line(img, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()),
                                           (colR, colG, colB), 2)
                        img = cv2.line(img, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()),
                                           (colR, colG, colB), 2)
                        img = cv2.line(img, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()),
                                           (colR, colG, colB), 2)
                        img = cv2.line(img, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()),
                                           (colR, colG, colB), 2)
                        '''

                    # print(camR_vis[i], camT_vis[i])
                    # draw_axis(img, camR_vis[i], camT_vis[i], K)
                    cv2.imwrite(fileName, img)

                    #print('STOP')

            end_t = time.time()
            times += end_t - start_t
            avg_time = times / gloCo
            rem_time = ((all_data - gloCo) * avg_time) / 60
            print('time remaining: ', rem_time, ' min')
            gloCo += 1

    for s in categories:
        objName = str(s)
        tempC = {
            "id": s,
            "name": objName,
            "supercategory": "object"
        }
        dict["categories"].append(tempC)

    valAnno = target + 'annotations/instances_train.json'

    with open(valAnno, 'w') as fpT:
        json.dump(dict, fpT)

    print('everythings done')
