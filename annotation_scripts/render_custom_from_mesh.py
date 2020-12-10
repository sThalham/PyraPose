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

from pathlib import Path

from misc import manipulate_RGB, toPix_array, toPix, calculate_feature_visibility

# Import bop_renderer and bop_toolkit.
# ------------------------------------------------------------------------------
# Import bop_renderer and bop_toolkit.
# ------------------------------------------------------------------------------
bop_renderer_path = '/home/stefan/bop_renderer/build'
sys.path.append(bop_renderer_path)

import bop_renderer


if __name__ == "__main__":

    mesh_path = sys.argv[1]
    background = '/home/stefan/data/datasets/cocoval2017/'
    target = '/home/stefan/data/train_data/fronius_train/'

    visu = False
    resX = 640
    resY = 480
    fx = 572.41140
    fy = 573.57043
    cx = 325.26110
    cy = 242.04899

    ren = bop_renderer.Renderer()
    ren.init(resX, resY)
    mesh_id = 0
    obj_ids = []

    for mesh_now in os.listdir(mesh_path):
        mesh_path_now = os.path.join(mesh_path, mesh_now)
        if mesh_now[-4:] != '.ply':
            continue
        ren.add_object(mesh_id, mesh_path_now)
        obj_ids.append(mesh_id)
        mesh_id += 1

    # interlude for debugging
    mesh_info = '/home/stefan/data/Meshes/linemod_13/models_info.yml'
    threeD_boxes = np.ndarray((34, 8, 3), dtype=np.float32)
    K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]

    for key, value in yaml.load(open(mesh_info)).items():
        # for key, value in json.load(open(mesh_info)).items():
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
        threeD_boxes[int(key), :, :] = three_box_solo

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
    gloCo = 0
    times = []

    syns = os.listdir(background)
    for o_idx in range(1,10):
        for bg_img_path in syns:

            bg_img_path_j = os.path.join(background, bg_img_path)

            bg_img = cv2.imread(bg_img_path_j)
            print(bg_img.shape)
            bg_x, bg_y, _ = bg_img.shape

            if bg_y > bg_x:
                bg_img = np.swapaxes(bg_img, 0, 1)

            bg_img = cv2.resize(bg_img, (resX, resY))

            print(bg_img.shape)

            samp = int(bg_img_path[:-4])

            template_samp = '00000'
            imgNum = str(o_idx) + template_samp[:-len(str(samp))] + str(samp)
            img_id = int(imgNum)
            imgNam = imgNum + '.png'
            iname = str(imgNam)

            fileName = target + 'images/train/' + imgNam[:-4] + '_rgb.png'
            myFile = Path(fileName)

            bbox_vis = []
            cat_vis = []
            camR_vis = []
            camT_vis = []
            calib_K = []
            mask_ind = 0
            mask_img = np.zeros((480, 640), dtype=np.uint8)
            visib_img = np.zeros((480, 640, 3), dtype=np.uint8)
            bbvis = []
            cnt = 0

            zeds = []
            renderings = []
            rotations = []
            translations = []
            obj_ids = np.random.choice(obj_ids, size=1, replace=False)

            for objID in obj_ids:
                R = tf3d.euler.euler2mat(np.random.rand(), np.random.rand(), np.random.rand())
                z = 0.3 + np.random.rand() * 1.2
                x = (2 * (0.6 * z)) * np.random.rand() - (0.6 * z)
                y = (2 * (0.4 * z)) * np.random.rand() - (0.4 * z)
                t = np.array([[x, y, z]]).T
                rotations.append(R)
                translations.append(t)
                zeds.append(z)

                R_list = R.flatten().tolist()
                t_list = t.flatten().tolist()

                light_pose = [np.random.rand() * 2 - 1.0, np.random.rand() * 2 - 1.0, np.random.rand() * 2 - 1.0]
                light_color = [np.random.rand() * 0.2 + 0.8, np.random.rand() * 0.2 + 0.8, np.random.rand() * 0.2 + 0.8]
                light_ambient_weight = np.random.rand()
                light_diffuse_weight = 0.5 + np.random.rand() * 0.5
                light_spec_weight = 0.5 + np.random.rand() * 0.5
                light_spec_shine = np.random.rand() * 10.0
                ren.set_light(light_pose, light_color, light_ambient_weight, light_diffuse_weight, light_spec_weight, light_spec_shine)
                ren.render_object(objID, R_list, t_list, fx, fy, cx, cy)
                rgb_img = ren.get_color_image(objID)
                renderings.append(rgb_img)

            zeds = np.asarray(zeds, dtype=np.float32)
            low2high = np.argsort(zeds)
            high2low = low2high[::-1]

            for i_idx in high2low:
                ren_img = renderings[i_idx]
                R = rotations[i_idx]
                t = translations[i_idx]
                obj_id = obj_ids[i_idx]
                print(obj_id)

                # full object surface
                R_list = R.flatten()
                t_list = np.array([[0.0, 0.0, t[2]]]).T
                R_list = R_list.flatten().tolist()
                t_list = t_list.flatten().tolist()
                ren.render_object(obj_id, R_list, t_list, fx, fy, cx, cy)
                full_visib_img = ren.get_color_image(obj_id)
                visib_non_zero = np.nonzero(ren_img)
                surf_visib = np.sum(visib_non_zero[0])
                fullvisibName = target + 'images/train/' + imgNam[:-4] + str(obj_id) + '_fv.png'
                cv2.imwrite(fullvisibName, full_visib_img)

                # partial visibility mask
                partial_visib_img = np.where(visib_img > 0, 0.0, ren_img)
                partial_non_zero = np.nonzero(partial_visib_img)
                partial_surf_visib = np.sum(partial_non_zero[0])
                partvisibName = target + 'images/train/' + imgNam[:-4] + str(obj_id) + '_pv.png'
                cv2.imwrite(partvisibName, partial_visib_img)

                visib_fract = partial_surf_visib / surf_visib
                print(visib_fract)
                visib_img = np.where(ren_img > 0, ren_img, visib_img)
                visibName = target + 'images/train/' + imgNam[:-4] + str(obj_id) + '_vi.png'
                cv2.imwrite(visibName, visib_img)

                # da final image
                bg_img = np.where(ren_img > 0, ren_img, bg_img)

                mask_id = mask_ind + 1
                mask_img = np.where(ren_img.any(axis=2) > 0, mask_id, mask_img)
                mask_ind = mask_ind + 1

                non_zero = np.nonzero(ren_img)
                surf_ren = np.sum(non_zero[0])
                bb_xmin = np.nanmin(non_zero[1])
                bb_xmax = np.nanmax(non_zero[1])
                bb_ymin = np.nanmin(non_zero[0])
                bb_ymax = np.nanmax(non_zero[0])

                obj_bb = [bb_xmin, bb_ymin, bb_xmax-bb_xmin, bb_ymax-bb_ymin]
                bbox_vis.append(obj_bb)

                cat_vis.append(obj_id)

                # pose [x, y, z, roll, pitch, yaw]
                R = np.asarray(R, dtype=np.float32)
                rot = tf3d.quaternions.mat2quat(R.reshape(3, 3))
                rot = np.asarray(rot, dtype=np.float32)
                tra = np.asarray(t, dtype=np.float32)
                pose = [tra[0], tra[1], tra[2], rot[0], rot[1], rot[2], rot[3]]


                area = obj_bb[2] * obj_bb[3]

                trans = np.asarray([pose[0], pose[1], pose[2]], dtype=np.float32)
                R = tf3d.quaternions.quat2mat(np.asarray([pose[3], pose[4], pose[5], pose[6]], dtype=np.float32))
                tDbox = R.reshape(3, 3).dot(threeD_boxes[obj_id, :, :].T).T
                tDbox = tDbox + np.repeat(trans.T, 8, axis=0)
                box3D = toPix_array(tDbox, fx=fx, fy=fy, cx=cx, cy=cy)
                box3D = np.reshape(box3D, (16))
                box3D = box3D.tolist()

                pose = [np.asscalar(pose[0]), np.asscalar(pose[1]), np.asscalar(pose[2]),
                            np.asscalar(pose[3]), np.asscalar(pose[4]), np.asscalar(pose[5]), np.asscalar(pose[6])]

                # if obj_id in [10, 11, 14]:
                bbox_vis.append(obj_bb)
                bbvis.append(box3D)
                camR_vis.append(np.asarray([pose[3], pose[4], pose[5], pose[6]], dtype=np.float32))
                camT_vis.append(np.asarray([pose[0], pose[1], pose[2]], dtype=np.float32))
                calib_K.append(K)

                nx1 = obj_bb[0]
                ny1 = obj_bb[1]
                nx2 = nx1 + obj_bb[2]
                ny2 = ny1 + obj_bb[3]
                npseg = np.array([nx1, ny1, nx2, ny1, nx2, ny2, nx1, ny2])
                cont = npseg.tolist()

                annoID = annoID + 1
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
                    "feature_visibility": visib_fract
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
                cv2.imwrite(mask_safe_path, mask_img)

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
                    img = rgbImg
                    for i, bb in enumerate(bbvis):

                        bb = np.array(bb)

                        phler = True
                        if phler:
                            pose = np.asarray(bbvis[i], dtype=np.float32)

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

                    # print(camR_vis[i], camT_vis[i])
                    # draw_axis(img, camR_vis[i], camT_vis[i], K)
                    cv2.imwrite(rgb_name, img)

                    print('STOP')

            if dataset == 'linemod':
                catsInt = range(1, 16)
            elif dataset == 'occlusion':
                catsInt = range(1, 9)
            elif dataset == 'ycbv':
                catsInt = range(1, 22)
            elif dataset == 'tless':
                catsInt = range(1, 31)
            elif dataset == 'homebrewed':
                catsInt = range(1, 34)

            if specific_object_set == True:
                catsInt = range(1, (len(spec_objs) + 1))

            for s in catsInt:
                objName = str(s)
                tempC = {
                    "id": s,
                    "name": objName,
                    "supercategory": "object"
                }
                dict["categories"].append(tempC)

            valAnno = target + 'annotations/instances_' + traintestval + '.json'

            with open(valAnno, 'w') as fpT:
                json.dump(dict, fpT)

            print('everythings done')
