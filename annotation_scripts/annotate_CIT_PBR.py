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


def load_rendering(fn_gt, fn_part, fn_rgb):

    with open(fn_gt, 'r') as stream:
        query = yaml.load(stream)
        if query is None:
            print('Whatever is wrong there.... ¯\_(ツ)_/¯')
            return None, None, None, None, None, None, None

        bboxes = np.zeros((len(query), 5), np.int)
        poses = np.zeros((len(query), 7), np.float32)
        mask_ids = np.zeros((len(query)), np.int)
        for j in range(len(query)-1): # skip cam pose
            qr = query[j]
            class_id = qr['class_id']
            bbox = qr['bbox']
            mask_ids[j] = int(qr['mask_id'])
            pose = np.array(qr['pose']).reshape(4, 4)
            bboxes[j, 0] = class_id
            bboxes[j, 1:5] = np.array(bbox)
            q_pose = tf3d.quaternions.mat2quat(pose[:3, :3])
            poses[j, 3:7] = np.array(q_pose)
            poses[j, 0:3] = np.array([pose[0, 3], pose[1, 3], pose[2, 3]])

    if bboxes.shape[0] < 2:
        print('invalid train image, no bboxes in fov')
        return None, None, None, None, None, None, None

    #partmask = cv2.imread(fn_part, 0)
    partmask = np.load(fn_part)
    rgb_img = cv2.imread(fn_rgb, 1)

    return rgb_img, partmask, bboxes, poses, mask_ids


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

    data_path = '/home/stefan/data/renderings/CIT_render/patches'
    mesh_path = '/home/stefan/data/Meshes/CIT_color/'
    target = '/home/stefan/data/train_data/CIT_PBR/'

    visu = True
    resX = 640
    resY = 480
    fx = 623.1298104626079 # blender calc
    fy = 617.1590544390115 # blender calc
    cx = 320.0
    cy = 240.0
    K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]

    ren = bop_renderer.Renderer()
    ren.init(resX, resY)
    mesh_id = 1
    light_pose = [0.0, 0.0, 0.0]
    light_color = [1.0, 0.0, 0.0]
    light_ambient_weight = 1.0
    light_diffuse_weight = 1.0
    light_spec_weight = 0.0
    light_spec_shine = 1.0
    ren.set_light(light_pose, light_color, light_ambient_weight, light_diffuse_weight, light_spec_weight,
                  light_spec_shine)
    categories = []

    for mesh_now in os.listdir(mesh_path):
        mesh_path_now = os.path.join(mesh_path, mesh_now)
        if mesh_now[-4:] != '.ply':
            continue
        mesh_id = int(mesh_now[-6:-4])
        ren.add_object(mesh_id, mesh_path_now)
        categories.append(mesh_id)
        #mesh_id += 1

    mesh_info = os.path.join(mesh_path, 'models_info.yml')
    threeD_boxes = np.ndarray((34, 8, 3), dtype=np.float32)
    # sym_cont = np.ndarray((34, 3), dtype=np.float32)
    # sym_disc = np.ndarray((34, 9), dtype=np.float32)

    max_box = [0, 0, 0, 0]
    max_box_area = 0
    min_box = [0, 0, 0, 0]
    min_box_area = 300 * 300

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

        threeD_boxes[int(key), :, :] = three_box_solo * fac

    now = datetime.datetime.now()
    dateT = str(now)

    dict = {"info": {
        "description": "cit",
        "version": "1.0",
        "year": 2021,
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

    syns = os.listdir(data_path)

    for samp in syns:

        print(samp)

        if not samp.endswith('.yaml'):
            continue

        start_t = time.time()

        anno_path = os.path.join(data_path, samp)
        img_path = os.path.join(data_path, 'rgb', samp[:-8] + '_rgb.png')
        mask_path = os.path.join(data_path, 'mask', samp[:-8] + '_mask.npy')

        img_PBR, mask_PBR, boxes_PBR, poses_PBR, mask_ids_PBR = load_rendering(anno_path, mask_path, img_path)

        img_id = int(samp[:-8])
        imgNam = samp[:-8] + '.png'
        iname = str(imgNam)

        fileName = target + 'images/train/' + samp[:-8] + '_rgb.png'
        myFile = Path(fileName)

        cnt = 0
        mask_ind = 0

        boxes3D = []
        obj_ids = []
        calib_K = []
        bboxes = []

        seq_obj = 0
        for idx, box in enumerate(boxes_PBR[:-1]):

            # ID + box
            objID = int(box[0])
            print('objID: ', objID)
            bbox = box[1:]
            obj_bb = [int(bbox[1]), int(bbox[0]), int(bbox[3] - bbox[1]), int(bbox[2] - bbox[0])]
            area = int(obj_bb[2] * obj_bb[3])
            obj_ids.append(objID)
            bboxes.append(obj_bb)

            # pose + 3Dbox
            pose = [np.asscalar(poses_PBR[idx][0]), np.asscalar(poses_PBR[idx][1]), np.asscalar(poses_PBR[idx][2]),
                    np.asscalar(poses_PBR[idx][3]), np.asscalar(poses_PBR[idx][4]), np.asscalar(poses_PBR[idx][5]), np.asscalar(poses_PBR[idx][6])]
            trans = np.asarray([pose[0], pose[1], pose[2]], dtype=np.float32)
            R = tf3d.quaternions.quat2mat(np.asarray([pose[3], pose[4], pose[5], pose[6]], dtype=np.float32))
            #3Dbox for visualization
            tDbox = R.reshape(3, 3).dot(threeD_boxes[objID, :, :].T).T
            tDbox = tDbox + np.repeat(trans[:, np.newaxis].T, 8, axis=0)
            box3D = toPix_array(tDbox, fx=fx, fy=fy, cx=cx, cy=cy)
            box3D = np.reshape(box3D, (16))
            box3D = box3D.tolist()
            calib_K.append(K)
            boxes3D.append(box3D)

            # obj_mask
            mask_id_PBR = int(mask_ids_PBR[idx])
            print('mask id: ', mask_id_PBR)

            t_list = np.array([[0.0, 0.0, trans[2]]]).T
            t_list = t_list.flatten().tolist()
            T_2obj = lookAt(trans.T, np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
            R_2obj = T_2obj[:3, :3]
            t_2obj = T_2obj[:3, 3]
            R_fv = R_2obj @ np.linalg.inv(R).T
            R_list = R_fv.flatten().tolist()
            ren.render_object(objID, R_list, t_list, fx, fy, cx, cy)
            full_visib_img = ren.get_color_image(objID)
            #full_visib_mask = np.where(np.any(full_visib_img, axis=2) > 0, 255, 0)

            partial_visib_mask = np.where(mask_PBR==mask_id_PBR, 1, 0)
            partial_mask_surf = np.sum(partial_visib_mask)
            full_visib_mask = np.nan_to_num(full_visib_img, copy=True, nan=0, posinf=0, neginf=0)
            full_visib_mask = np.where(np.any(full_visib_mask, axis=2) > 0, 1, 0)
            surf_visib = np.sum(full_visib_mask)
            visib_fract = float(partial_mask_surf / surf_visib)
            if visib_fract > 1.0:
                visib_fract = float(1.0)

            print('visib_fract: ', visib_fract)

            #bg_img = np.where(partial_visib_img > 0, partial_visib_img, bg_img)
            # mask calculation
            #mask_id = mask_ind + 1
            #mask_img = np.where(mask_PBR==mask_id_PBR, mask_id, mask_img)
            #mask_ind = mask_ind + 1
            annoID = annoID + 1
            tempTA = {
                "id": annoID,
                "image_id": img_id,
                "category_id": objID,
                "bbox": obj_bb,
                "pose": pose,
                "segmentation": box3D,
                "mask_id": mask_id_PBR,
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

        # safe images
        if myFile.exists():
            print('File exists, skip encoding, ', fileName)
        else:
            cv2.imwrite(fileName, img_PBR)
            print("storing image in : ", fileName)
            mask_safe_path = fileName[:-8] + '_mask.png'
            cv2.imwrite(mask_safe_path, mask_PBR.astype(np.int8))
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
                img = img_PBR
                for i, bb in enumerate(bboxes):
                    bb = np.array(bb)
                    #cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[0] + bb[2]), int(bb[1] + bb[3])),
                    #              (255, 255, 255), 2)
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
            rem_time = ((len(syns) - gloCo) * avg_time) / 60
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
