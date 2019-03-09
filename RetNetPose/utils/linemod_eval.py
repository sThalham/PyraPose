"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#from pycocotools.cocoeval import COCOeval

import keras
import numpy as np
import json
import pyquaternion
import math
import transforms3d as tf3d
import geometry
import os
import cv2
from .pose_error import add
from .pose_error import adi
from .pose_error import reproj
from .pose_error import re
from .pose_error import te

import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."

# LineMOD
fxkin = 572.41140
fykin = 573.57043
cxkin = 325.26110
cykin = 242.04899


def create_point_cloud(depth, ds):

    rows, cols = depth.shape

    depRe = depth.reshape(rows * cols)
    zP = np.multiply(depRe, ds)

    x, y = np.meshgrid(np.arange(0, cols, 1), np.arange(0, rows, 1), indexing='xy')
    yP = y.reshape(rows * cols) - cykin
    xP = x.reshape(rows * cols) - cxkin
    yP = np.multiply(yP, zP)
    xP = np.multiply(xP, zP)
    yP = np.divide(yP, fykin)
    xP = np.divide(xP, fxkin)

    cloud_final = np.transpose(np.array((xP, yP, zP)))

    return cloud_final


def boxoverlap(a, b):
    a = np.array([a[0], a[1], a[0] + a[2], a[1] + a[3]])
    b = np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]])

    x1 = np.amax(np.array([a[0], b[0]]))
    y1 = np.amax(np.array([a[1], b[1]]))
    x2 = np.amin(np.array([a[2], b[2]]))
    y2 = np.amin(np.array([a[3], b[3]]))

    wid = x2-x1+1
    hei = y2-y1+1
    inter = wid * hei
    aarea = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    barea = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    # intersection over union overlap
    ovlap = inter / (aarea + barea - inter)
    # set invalid entries to 0 overlap
    maskwid = wid <= 0
    maskhei = hei <= 0
    np.where(ovlap, maskwid, 0)
    np.where(ovlap, maskhei, 0)

    return ovlap


def evaluate_linemod(generator, model, threshold=0.05):
    threshold = 0.5
    """ Use the pycocotools to evaluate a COCO model on a dataset.

    Args
        generator : The generator for generating the evaluation data.
        model     : The model to evaluate.
        threshold : The score threshold to use.
    """
    # start collecting results
    results = []
    image_ids = []
    image_indices = []
    idx = 0

    tp = np.zeros((16), dtype=np.uint32)
    fp = np.zeros((16), dtype=np.uint32)
    fn = np.zeros((16), dtype=np.uint32)
    xyD = []
    xyzD = []
    zD = []
    less5cm_imgplane = []
    less5cm = []
    less10cm = []
    less15cm = []
    less20cm = []
    less25cm = []
    rotD = []
    less5deg = []
    less10deg = []
    less15deg = []
    less20deg = []
    less25deg = []

    # load meshes
    mesh_path = "/home/sthalham/data/LINEMOD/models/"
    sub = os.listdir(mesh_path)
    mesh_dict = {}
    for m in sub:
        if m.endswith('.ply'):
            name = m[:-4]
            key = str(int(name[-2:]))
            mesh = np.genfromtxt(mesh_path+m, skip_header=16, usecols=(0, 1, 2))
            mask = np.where(mesh[:, 0] != 3)
            mesh = mesh[mask]
            mesh_dict[key] = mesh

    for index in progressbar.progressbar(range(generator.size()), prefix='LineMOD evaluation: '):

        image_raw = generator.load_image(index)
        image = generator.preprocess_image(image_raw)
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network
        #boxes, trans, deps, rots, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes, trans, deps, rol, pit, ya, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

        # correct boxes for image scale
        boxes /= scale

        # change to (x, y, w, h) (MS COCO standard)
        boxes[:, :, 2] -= boxes[:, :, 0]
        boxes[:, :, 3] -= boxes[:, :, 1]

        # target annotation
        anno = generator.load_annotations(index)
        if len(anno['labels']) > 1:
            continue
        else:
            t_cat = int(anno['labels']) + 1
        t_bbox = np.asarray(anno['bboxes'], dtype=np.float32)[0]
        t_tra = anno['poses'][0][:2]
        t_dep = anno['poses'][0][2]
        t_rot = anno['poses'][0][3:]
        fn[t_cat] += 1
        fnit = True
        # compute predicted labels and scores
        #for box, trans, deps, quat, score, label in zip(boxes[0], trans[0], deps[0], rots[0], scores[0], labels[0]):
        for box, trans, deps, rolls, pitchs, yaws, score, label in zip(boxes[0], trans[0], deps[0], rol[0], pit[0], ya[0], scores[0], labels[0]):
            # scores are sorted, so we can break
            if score < threshold:
                continue

            if label < 0:
                continue
            cls = generator.label_to_inv_label(label)
            tra = trans[(cls-1), :]
            #dep = np.array([(np.argmax(deps[(cls-1), :]) * 0.03 ) - 0.015])
            dep = deps[(cls - 1), :]
            #rot = quat[(cls-1), :]
            #pose = tra.tolist() + dep.tolist() + rot.tolist()
            #roll = np.array([(np.argmax(rolls[(cls - 1), :]) * 0.69813170079) - np.pi])
            #pitch = np.array([(np.argmax(pitchs[(cls - 1), :]) * 0.69813170079) - np.pi])
            #yaw = np.array([(np.argmax(yaws[(cls - 1), :]) * 0.69813170079) - np.pi])
            roll = np.array([(np.argmax(rolls[(cls - 1), :]) * 0.21666156231) - np.pi]) + (0.21666156231*0.5)
            pitch = np.array([(np.argmax(pitchs[(cls - 1), :]) * 0.21666156231) - np.pi]) + (0.21666156231*0.5)
            yaw = np.array([(np.argmax(yaws[(cls - 1), :]) * 0.21666156231) - np.pi]) + (0.21666156231*0.5)
            pose = tra.tolist() + dep.tolist() + roll.tolist() + pitch.tolist() + yaw.tolist()

            # append detection for each positively labeled class
            image_result = {
                'image_id'    : generator.image_ids[index],
                'category_id' : generator.label_to_inv_label(label),
                'score'       : float(score),
                'bbox'        : box.tolist(),
                'pose'        : pose
            }

            # append detection to results
            results.append(image_result)

            if cls == t_cat:
                b1 = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]])
                b2 = np.array([t_bbox[0], t_bbox[1], t_bbox[2], t_bbox[3]])
                IoU = boxoverlap(b1, b2)
                # occurences of 2 or more instances not possible in LINEMOD
                if IoU > 0.5:
                    if fnit is True:
                        tp[t_cat] += 1
                        fn[t_cat] -= 1
                        fnit = False
                        #q1 = pyquaternion.Quaternion(t_pose)
                        #q2 = pyquaternion.Quaternion(rot)
                        #print('translation target: ', t_tra, '       estimation: ', tra)
                        #print('depth target: ', t_dep, '             estimation: ', dep)
                        x = (((box[0] + box[2]*0.5) - cxkin) * dep) / fxkin * 0.001
                        y = (((box[1] + box[3]*0.5) - cykin) * dep) / fykin * 0.001

                        x_t = ((t_tra[0] - cxkin) * t_dep) / fxkin * 0.001
                        y_t = ((t_tra[1] - cykin) * t_dep) / fykin * 0.001

                        #only image plane
                        x_o = ((tra[0] - cxkin) * t_dep) / fxkin * 0.001
                        y_o = ((tra[1] - cykin) * t_dep) / fykin * 0.001
                        x_o_d = np.abs(np.abs(x_o) - np.abs(x_t))
                        y_o_d = np.abs(np.abs(y_o) - np.abs(y_t))
                        xy = np.linalg.norm(np.asarray([x_o_d, y_o_d], dtype=np.float32))
                        if not math.isnan(xy):
                            xyD.append(xy)
                            if xy < 0.05:
                                less5cm_imgplane.append(xy)

                        xd = np.abs(np.abs(x) - np.abs(x_t))
                        yd = np.abs(np.abs(y) - np.abs(y_t))

                        # naive depth estimate
                        #path = generator.load_image_dep(index)
                        #image_dep =cv2.imread(path, cv2.IMREAD_UNCHANGED)

                        '''
                        #dep_val = image_dep[int(box[1] + box[3] * 0.5), int(box[0] + box[2] * 0.5)]*0.001
                        dep_val = image_dep[int(y_o), int(x_o)] * 0.001
                        if cls == 1:
                            dep = dep_val + 0.041
                        elif cls == 2:
                            dep = dep_val + 0.0928
                        elif cls == 3:
                            dep = dep_val + 0.0675
                        elif cls == 4:
                            dep = dep_val + 0.0633
                        elif cls == 5:
                            dep = dep_val + 0.0795
                        elif cls == 6:
                            dep = dep_val + 0.052
                        elif cls == 7:
                            dep = dep_val + 0.0508
                        elif cls == 8:
                            dep = dep_val + 0.0853
                        elif cls == 9:
                            dep = dep_val + 0.0445
                        elif cls == 10:
                            dep = dep_val + 0.0543
                        elif cls == 11:
                            dep = dep_val + 0.048
                        elif cls == 12:
                            dep = dep_val + 0.05
                        elif cls == 13:
                            dep = dep_val + 0.0862
                        elif cls == 14:
                            dep = dep_val + 0.0888
                        elif cls == 15:
                            dep = dep_val + 0.071
                        '''

                        zd = np.abs(np.abs(dep) - np.abs(t_dep * 0.001))
                        xyz = np.linalg.norm(np.asarray([xd, yd, zd], dtype=np.float32))
                        if not math.isnan(xyz):
                            xyzD.append(xyz)
                            if xyz < 0.05:
                                less5cm.append(xyz)
                            if xyz < 0.1:
                                less10cm.append(xyz)
                            if xyz < 0.15:
                                less15cm.append(xyz)
                            if xyz < 0.2:
                                less20cm.append(xyz)
                            if xyz < 0.25:
                                less25cm.append(xyz)

                        if not math.isnan(zd):
                            zD.append(zd)

                        '''
                        if len(rot) < 4 and len(t_rot) < 4:
                            lie = [[0.0, np.asscalar(rot[0]), np.asscalar(rot[1])],
                                    [np.asscalar(-rot[0]), 0.0, np.asscalar(rot[2])],
                                    [np.asscalar(-rot[1]), np.asscalar(-rot[2]), 0.0]]
                            lie = np.asarray(lie, dtype=np.float32)
                            eul = geometry.rotations.map_hat(lie)
                            rot = tf3d.euler.euler2quat(eul[0], eul[1], eul[2])
                            rot = np.asarray(rot, dtype=np.float32)

                            t_lie = [[0.0, np.asscalar(t_rot[0]), np.asscalar(t_rot[1])],
                                   [np.asscalar(-t_rot[0]), 0.0, np.asscalar(t_rot[2])],
                                   [np.asscalar(-t_rot[1]), np.asscalar(-t_rot[2]), 0.0]]
                            t_lie = np.asarray(t_lie, dtype=np.float32)
                            t_eul = geometry.rotations.map_hat(t_lie)
                            t_rot = tf3d.euler.euler2quat(t_eul[0], t_eul[1], t_eul[2])
                            t_rot = np.asarray(t_rot, dtype=np.float32)

                        q1 = pyquaternion.Quaternion(t_rot).unit
                        q2 = pyquaternion.Quaternion(rot).unit
                        rd = pyquaternion.Quaternion.absolute_distance(q1, q2)
                        t_rot = t_rot / np.linalg.norm(t_rot)
                        rot = rot / np.linalg.norm(rot)
                        rot_t_mat = geometry.rotations.rotation_from_quaternion(t_rot)
                        rot_mat = geometry.rotations.rotation_from_quaternion(rot)
                        matd = geometry.rotations.geodesic_distance_for_rotations(rot_t_mat, rot_mat)
                        rd = matd
                        '''
                        print(float(roll), float(pitch), float(yaw))
                        print(t_rot[0], t_rot[1], t_rot[2])

                        #rot = np.array((roll, pitch, yaw), dtype=np.float32)
                        #quat = tf3d.euler.euler2quat(float(roll), float(pitch), float(yaw))
                        #quat_t = tf3d.euler.euler2quat(t_rot[0], t_rot[1], t_rot[2])
                        rot_mat = tf3d.euler.euler2mat(float(roll), float(pitch), float(yaw))
                        rot_t_mat = tf3d.euler.euler2mat(t_rot[0], t_rot[1], t_rot[2])
                        #quat = pyquaternion.Quaternion(quat).unit
                        #quat_t = pyquaternion.Quaternion(quat_t).unit
                        #rot_t_mat = geometry.rotations.rotation_from_quaternion(quat_t)
                        #rot_mat = geometry.rotations.rotation_from_quaternion(quat)
                        rd = re(rot_mat, rot_t_mat)
                        # rd = pyquaternion.Quaternion.distance(quat_t, quat)
                        #rot = np.array((roll, pitch, yaw), dtype=np.float32)
                        #rot = np.linalg.norm(rot)
                        #t_rot = np.linalg.norm(t_rot)
                        #rd = np.arccos(np.dot(rot, t_rot))


                        print(' ')
                        print(rd)

                        #print(' ')
                        #print(np.arccos(2 * np.power(np.dot(t_rot, rot), 2) - 1) * 180/math.pi, ' / ', (rd * 180/math.pi), (matd * 180/math.pi))
                        #print(1 - np.power(np.dot(t_rot, rot), 2), ' / ', (rd * 180 / math.pi))

                        if not math.isnan(rd):
                            rotD.append(rd)
                            if (rd * 180/math.pi) < 5.0:
                                less5deg.append(rd)
                            if (rd * 180/math.pi) < 10.0:
                                less10deg.append(xyz)
                            if (rd * 180/math.pi) < 15.0:
                                less15deg.append(xyz)
                            if (rd * 180/math.pi) < 20.0:
                                less20deg.append(xyz)
                            if (rd * 180/math.pi) < 25.0:
                                less25deg.append(xyz)

                        #ADDS
                        #mesh = mesh_dict[str(int(t_cat))]
                        #mesh = np.dot(mesh, tf3d.quaternions.quat2mat(q2))
                        #mesh = np.add(mesh, np.asarray([t_tra[0]*0.001, t_tra[1]*0.001, t_dep*0.001], dtype=np.float32))
                        #kd_true = sci_spa.KDTree(mesh)

                        #crop = image_raw[int(box[1]):int(box[1] + box[3]), int(box[0]):int(box[0] + box[2]), :]
                        #create_point_cloud(crop, 0.01)

                else:
                    fp[t_cat] += 1

        # append image to list of processed images
        image_ids.append(generator.image_ids[index])
        image_indices.append(index)
        idx += 1

    if not len(results):
        return

    # write output
    json.dump(results, open('{}_bbox_results.json'.format(generator.set_name), 'w'), indent=4)
    #json.dump(image_ids, open('{}_processed_image_ids.json'.format(generator.set_name), 'w'), indent=4)

    detPre = [0] * 16
    detRec = [0] * 16

    np.set_printoptions(precision=2)
    for ind in range(1, 16):
        if ind == 0:
            continue

        if tp[ind] == 0:
            detPre[ind] = 0.0
            detRec[ind] = 0.0
        else:
            detRec[ind] = tp[ind] / (tp[ind] + fn[ind])
            detPre[ind] = tp[ind] / (tp[ind] + fp[ind])

        #print('precision category ', ind, ': ', detPre[ind])
        #print('recall category ', ind, ': ', detRec[ind])

    dataset_recall = sum(tp) / (sum(tp) + sum(fp))
    dataset_precision = sum(tp) / (sum(tp) + sum(fn))
    dataset_xyz_diff = (sum(xyzD) / len(xyzD))
    dataset_xy_diff = (sum(xyD) / len(xyD))
    print(' ')
    print('center difference in image plane', dataset_xy_diff)
    print('percent < 5cm in image plane', len(less5cm_imgplane)/len(xyD))
    dataset_depth_diff = (sum(zD) / len(zD))
    dataset_rot_diff = (sum(rotD) / len(rotD)) * 180.0 / math.pi
    less5cm = len(less5cm)/len(xyzD)
    less10cm = len(less10cm) / len(xyzD)
    less15cm = len(less15cm) / len(xyzD)
    less20cm = len(less20cm) / len(xyzD)
    less25cm = len(less25cm) / len(xyzD)
    less5deg = len(less5deg) / len(rotD)
    less10deg = len(less10deg) / len(rotD)
    less15deg = len(less15deg) / len(rotD)
    less20deg = len(less20deg) / len(rotD)
    less25deg = len(less25deg) / len(rotD)
    print('linemod::percent below 5 cm: ', less5cm, '%')
    print('linemod::percent below 10 cm: ', less10cm, '%')
    print('linemod::percent below 15 cm: ', less15cm, '%')
    print('linemod::percent below 20 cm: ', less20cm, '%')
    print('linemod::percent below 25 cm: ', less25cm, '%')
    print('linemod::percent below 5 deg: ', less5deg, '%')
    print('linemod::percent below 10 deg: ', less10deg, '%')
    print('linemod::percent below 15 deg: ', less15deg, '%')
    print('linemod::percent below 20 deg: ', less20deg, '%')
    print('linemod::percent below 25 deg: ', less25deg, '%')

    return dataset_recall, dataset_precision, dataset_xyz_diff, dataset_depth_diff, dataset_rot_diff, less5cm, less5deg
