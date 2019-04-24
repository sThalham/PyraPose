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
import copy
import cv2
import open3d
from ..utils import ply_loader
from .pose_error import vsd, reproj, add, adi, re, te

import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."

# LineMOD
fxkin = 572.41140
fykin = 573.57043
cxkin = 325.26110
cykin = 242.04899


threeD_boxes = np.ndarray((15, 8, 3), dtype=np.float32)
threeD_boxes[0, :, :] = np.array([[0.038, 0.039, 0.046],  # ape [76, 78, 92]
                                     [0.038, 0.039, -0.046],
                                     [0.038, -0.039, -0.046],
                                     [0.038, -0.039, 0.046],
                                     [-0.038, 0.039, 0.046],
                                     [-0.038, 0.039, -0.046],
                                     [-0.038, -0.039, -0.046],
                                     [-0.038, -0.039, 0.046]])
threeD_boxes[1, :, :] = np.array([[0.108, 0.061, 0.1095],  # benchvise [216, 122, 219]
                                     [0.108, 0.061, -0.1095],
                                     [0.108, -0.061, -0.1095],
                                     [0.108, -0.061, 0.1095],
                                     [-0.108, 0.061, 0.1095],
                                     [-0.108, 0.061, -0.1095],
                                     [-0.108, -0.061, -0.1095],
                                     [-0.108, -0.061, 0.1095]])
threeD_boxes[2, :, :] = np.array([[0.083, 0.0825, 0.037],  # bowl [166, 165, 74]
                                     [0.083, 0.0825, -0.037],
                                     [0.083, -0.0825, -0.037],
                                     [0.083, -0.0825, 0.037],
                                     [-0.083, 0.0825, 0.037],
                                     [-0.083, 0.0825, -0.037],
                                     [-0.083, -0.0825, -0.037],
                                     [-0.083, -0.0825, 0.037]])
threeD_boxes[3, :, :] = np.array([[0.0685, 0.0715, 0.05],  # camera [137, 143, 100]
                                     [0.0685, 0.0715, -0.05],
                                     [0.0685, -0.0715, -0.05],
                                     [0.0685, -0.0715, 0.05],
                                     [-0.0685, 0.0715, 0.05],
                                     [-0.0685, 0.0715, -0.05],
                                     [-0.0685, -0.0715, -0.05],
                                     [-0.0685, -0.0715, 0.05]])
threeD_boxes[4, :, :] = np.array([[0.0505, 0.091, 0.097],  # can [101, 182, 194]
                                     [0.0505, 0.091, -0.097],
                                     [0.0505, -0.091, -0.097],
                                     [0.0505, -0.091, 0.097],
                                     [-0.0505, 0.091, 0.097],
                                     [-0.0505, 0.091, -0.097],
                                     [-0.0505, -0.091, -0.097],
                                     [-0.0505, -0.091, 0.097]])
threeD_boxes[5, :, :] = np.array([[0.0335, 0.064, 0.0585],  # cat [67, 128, 117]
                                     [0.0335, 0.064, -0.0585],
                                     [0.0335, -0.064, -0.0585],
                                     [0.0335, -0.064, 0.0585],
                                     [-0.0335, 0.064, 0.0585],
                                     [-0.0335, 0.064, -0.0585],
                                     [-0.0335, -0.064, -0.0585],
                                     [-0.0335, -0.064, 0.0585]])
threeD_boxes[6, :, :] = np.array([[0.059, 0.046, 0.0475],  # mug [118, 92, 95]
                                     [0.059, 0.046, -0.0475],
                                     [0.059, -0.046, -0.0475],
                                     [0.059, -0.046, 0.0475],
                                     [-0.059, 0.046, 0.0475],
                                     [-0.059, 0.046, -0.0475],
                                     [-0.059, -0.046, -0.0475],
                                     [-0.059, -0.046, 0.0475]])
threeD_boxes[7, :, :] = np.array([[0.115, 0.038, 0.104],  # drill [230, 76, 208]
                                     [0.115, 0.038, -0.104],
                                     [0.115, -0.038, -0.104],
                                     [0.115, -0.038, 0.104],
                                     [-0.115, 0.038, 0.104],
                                     [-0.115, 0.038, -0.104],
                                     [-0.115, -0.038, -0.104],
                                     [-0.115, -0.038, 0.104]])
threeD_boxes[8, :, :] = np.array([[0.052, 0.0385, 0.043],  # duck [104, 77, 86]
                                     [0.052, 0.0385, -0.043],
                                     [0.052, -0.0385, -0.043],
                                     [0.052, -0.0385, 0.043],
                                     [-0.052, 0.0385, 0.043],
                                     [-0.052, 0.0385, -0.043],
                                     [-0.052, -0.0385, -0.043],
                                     [-0.052, -0.0385, 0.043]])
threeD_boxes[9, :, :] = np.array([[0.075, 0.0535, 0.0345],  # eggbox [150, 107, 69]
                                     [0.075, 0.0535, -0.0345],
                                     [0.075, -0.0535, -0.0345],
                                     [0.075, -0.0535, 0.0345],
                                     [-0.075, 0.0535, 0.0345],
                                     [-0.075, 0.0535, -0.0345],
                                     [-0.075, -0.0535, -0.0345],
                                     [-0.075, -0.0535, 0.0345]])
threeD_boxes[10, :, :] = np.array([[0.0185, 0.039, 0.0865],  # glue [37, 78, 173]
                                     [0.0185, 0.039, -0.0865],
                                     [0.0185, -0.039, -0.0865],
                                     [0.0185, -0.039, 0.0865],
                                     [-0.0185, 0.039, 0.0865],
                                     [-0.0185, 0.039, -0.0865],
                                     [-0.0185, -0.039, -0.0865],
                                     [-0.0185, -0.039, 0.0865]])
threeD_boxes[11, :, :] = np.array([[0.0505, 0.054, 0.04505],  # holepuncher [101, 108, 91]
                                     [0.0505, 0.054, -0.04505],
                                     [0.0505, -0.054, -0.04505],
                                     [0.0505, -0.054, 0.04505],
                                     [-0.0505, 0.054, 0.04505],
                                     [-0.0505, 0.054, -0.04505],
                                     [-0.0505, -0.054, -0.04505],
                                     [-0.0505, -0.054, 0.04505]])
threeD_boxes[12, :, :] = np.array([[0.115, 0.038, 0.104],  # drill [230, 76, 208]
                                     [0.115, 0.038, -0.104],
                                     [0.115, -0.038, -0.104],
                                     [0.115, -0.038, 0.104],
                                     [-0.115, 0.038, 0.104],
                                     [-0.115, 0.038, -0.104],
                                     [-0.115, -0.038, -0.104],
                                     [-0.115, -0.038, 0.104]])
threeD_boxes[13, :, :] = np.array([[0.129, 0.059, 0.0705],  # iron [258, 118, 141]
                                     [0.129, 0.059, -0.0705],
                                     [0.129, -0.059, -0.0705],
                                     [0.129, -0.059, 0.0705],
                                     [-0.129, 0.059, 0.0705],
                                     [-0.129, 0.059, -0.0705],
                                     [-0.129, -0.059, -0.0705],
                                     [-0.129, -0.059, 0.0705]])
threeD_boxes[14, :, :] = np.array([[0.047, 0.0735, 0.0925],  # phone [94, 147, 185]
                                     [0.047, 0.0735, -0.0925],
                                     [0.047, -0.0735, -0.0925],
                                     [0.047, -0.0735, 0.0925],
                                     [-0.047, 0.0735, 0.0925],
                                     [-0.047, 0.0735, -0.0925],
                                     [-0.047, -0.0735, -0.0925],
                                     [-0.047, -0.0735, 0.0925]])

model_radii = np.array([0.041, 0.0928, 0.0675, 0.0633, 0.0795, 0.052, 0.0508, 0.0853, 0.0445, 0.0543, 0.048, 0.05, 0.0862, 0.0888, 0.071])


def load_pcd(cat):
    # load meshes
    mesh_path = "/home/sthalham/data/LINEMOD/models/"
    #mesh_path = "/home/stefan/data/val_linemod_cc_rgb/models_ply/"
    ply_path = mesh_path + 'obj_' + cat + '.ply'
    model_vsd = ply_loader.load_ply(ply_path)
    pcd_model = open3d.PointCloud()
    pcd_model.points = open3d.Vector3dVector(model_vsd['pts'])
    open3d.estimate_normals(pcd_model, search_param=open3d.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    # open3d.draw_geometries([pcd_model])
    model_vsd_mm = copy.deepcopy(model_vsd)
    model_vsd_mm['pts'] = model_vsd_mm['pts'] * 1000.0

    return model_vsd, model_vsd_mm


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

    tp_add = np.zeros((16), dtype=np.uint32)
    fp_add = np.zeros((16), dtype=np.uint32)
    fn_add = np.zeros((16), dtype=np.uint32)

    rotD = np.zeros((16), dtype=np.uint32)
    less5 = np.zeros((16), dtype=np.uint32)
    rep_e = np.zeros((16), dtype=np.uint32)
    rep_less5 = np.zeros((16), dtype=np.uint32)
    add_e = np.zeros((16), dtype=np.uint32)
    add_less_d = np.zeros((16), dtype=np.uint32)
    vsd_e = np.zeros((16), dtype=np.uint32)
    vsd_less_t = np.zeros((16), dtype=np.uint32)

    model_pre = []

    for index in progressbar.progressbar(range(generator.size()), prefix='LineMOD evaluation: '):

        image_raw = generator.load_image(index)
        image = generator.preprocess_image(image_raw)
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network
        #boxes, trans, deps, rots, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        #boxes, trans, deps, rol, pit, ya, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes, boxes3D, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

        # correct boxes for image scale
        boxes /= scale

        # change to (x, y, w, h) (MS COCO standard)
        boxes[:, :, 2] -= boxes[:, :, 0]
        boxes[:, :, 3] -= boxes[:, :, 1]

        # target annotation
        anno = generator.load_annotations(index)
        if len(anno['labels']) > 1:
            t_cat = 2
            obj_name = '02'
            ent = np.where(anno['labels'] == 1.0)
            t_bbox = np.asarray(anno['bboxes'], dtype=np.float32)[ent][0]
            t_tra = anno['poses'][ent][0][:3]
            t_rot = anno['poses'][ent][0][3:]

        else:
            t_cat = int(anno['labels']) + 1
            obj_name = str(t_cat)
            if len(obj_name) < 2:
                obj_name = '0' + obj_name
            t_bbox = np.asarray(anno['bboxes'], dtype=np.float32)[0]
            t_tra = anno['poses'][0][:3]
            t_rot = anno['poses'][0][3:]

        if obj_name is '03' or obj_name is '07':
            print(t_cat, ' ====> skip')
            continue

        if obj_name is not model_pre:
            model_vsd, model_vsd_mm = load_pcd(obj_name)
            model_pre = obj_name

        rotD[t_cat] += 1
        rep_e[t_cat] += 1
        add_e[t_cat] += 1
        vsd_e[t_cat] += 1
        #t_bbox = np.asarray(anno['bboxes'], dtype=np.float32)[0]
        #t_tra = anno['poses'][0][:3]
        #t_rot = anno['poses'][0][3:]
        fn[t_cat] += 1
        fnit = True

        # compute predicted labels and scores
        for box, box3D, score, label in zip(boxes[0], boxes3D[0], scores[0], labels[0]):
            # scores are sorted, so we can break
            if score < threshold:
                continue

            if label < 0:
                continue
            cls = generator.label_to_inv_label(label)
            control_points = box3D[(cls - 1), :]

            # append detection for each positively labeled class
            image_result = {
                'image_id'    : generator.image_ids[index],
                'category_id' : generator.label_to_inv_label(label),
                'score'       : float(score),
                'bbox'        : box.tolist(),
                'pose'        : control_points.tolist()
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

                        obj_points = np.ascontiguousarray(threeD_boxes[cls-1, :, :], dtype=np.float32) #.reshape((8, 1, 3))
                        est_points = np.ascontiguousarray(control_points.T, dtype=np.float32).reshape((8, 1, 2))

                        K = np.float32([fxkin, 0., cxkin, 0., fykin, cykin, 0., 0., 1.]).reshape(3, 3)

                        #retval, orvec, otvec = cv2.solvePnP(obj_points, est_points, K, None, None, None, False, cv2.SOLVEPNP_ITERATIVE)
                        retval, orvec, otvec, inliers = cv2.solvePnPRansac(objectPoints=obj_points,
                                                                           imagePoints=est_points, cameraMatrix=K,
                                                                           distCoeffs=None, rvec=None, tvec=None,
                                                                           useExtrinsicGuess=False, iterationsCount=100,
                                                                           reprojectionError=5.0, confidence=0.99,
                                                                           flags=cv2.SOLVEPNP_ITERATIVE)

                        R_est, _ = cv2.Rodrigues(orvec)
                        t_est = otvec

                        t_rot = tf3d.euler.euler2mat(t_rot[0], t_rot[1], t_rot[2])
                        R_gt = np.array(t_rot, dtype=np.float32).reshape(3, 3)
                        t_gt = np.array(t_tra, dtype=np.float32) * 0.001

                        rd = re(R_gt, R_est)
                        xyz = te(t_gt, t_est.T)

                        if not math.isnan(rd):
                            if rd < 5.0 and xyz < 0.05:
                                less5[t_cat] += 1

                        image_dep_path = generator.load_image_dep(index)
                        image_dep = cv2.imread(image_dep_path, cv2.IMREAD_UNCHANGED)
                        err_vsd = vsd(R_est, t_est * 1000.0, R_gt, t_gt * 1000.0, model_vsd_mm, image_dep, K, 0.3, 20.0)
                        # print('vsd: ', err_vsd)
                        if not math.isnan(err_vsd):
                            if err_vsd < 0.3:
                                vsd_less_t[t_cat] += 1

                        err_repr = reproj(K, R_est, t_est, R_gt, t_gt, model_vsd["pts"])

                        if not math.isnan(err_repr):
                            if err_repr < 5.0:
                                rep_less5[t_cat] += 1

                        if cls == 3 or cls == 7 or cls == 10 or cls == 11:
                            err_add = adi(R_est, t_est, R_gt, t_gt, model_vsd["pts"])

                        else:
                            err_add = add(R_est, t_est, R_gt, t_gt, model_vsd["pts"])

                        if not math.isnan(err_add):
                            if err_add < (model_radii[cls - 1] * 2 * 0.1):
                                add_less_d[t_cat] += 1

                        if not math.isnan(err_add):
                            if err_add < (model_radii[cls - 1] * 2 * 0.15):
                                tp_add[t_cat] += 1
                            else:
                                fn_add[t_cat] -= 1

                else:
                    fp[t_cat] += 1
                    fp_add[t_cat] += 1

        # append image to list of processed images
        image_ids.append(generator.image_ids[index])
        image_indices.append(index)
        idx += 1

    if not len(results):
        return

    # write output
    json.dump(results, open('{}_bbox_results.json'.format(generator.set_name), 'w'), indent=4)
    #json.dump(image_ids, open('{}_processed_image_ids.json'.format(generator.set_name), 'w'), indent=4)

    detPre = [0.0] * 16
    detRec = [0.0] * 16
    detPre_add = [0.0] * 16
    detRec_add = [0.0] * 16
    F1_add = [0.0] * 16
    less_55 = [0.0] * 16
    less_repr_5 = [0.0] * 16
    less_add_d = [0.0] * 16
    less_vsd_t = [0.0] * 16

    np.set_printoptions(precision=2)
    print('')
    for ind in range(1, 16):
        if ind == 0:
            continue

        if tp[ind] == 0:
            detPre[ind] = 0.0
            detRec[ind] = 0.0
            detPre_add[ind] = 0.0
            detRec_add[ind] = 0.0
            less_55[ind] = 0.0
            less_repr_5[ind] = 0.0
            less_add_d[ind] = 0.0
            less_vsd_t[ind] = 0.0
        else:
            detRec[ind] = tp[ind] / (tp[ind] + fn[ind]) * 100.0
            detPre[ind] = tp[ind] / (tp[ind] + fp[ind]) * 100.0
            detRec_add[ind] = tp_add[ind] / (tp_add[ind] + fn_add[ind]) * 100.0
            detPre_add[ind] = tp_add[ind] / (tp_add[ind] + fp_add[ind]) * 100.0
            F1_add[ind] = 2 * ((detPre_add[ind] * detRec_add[ind])/(detPre_add[ind] + detRec_add[ind]))
            less_55[ind] = sum(less5[ind]) / sum(rotD[ind]) * 100.0
            less_repr_5[ind] = sum(rep_less5[ind]) / sum(rep_e[ind]) * 100.0
            less_add_d[ind] = sum(add_less_d[ind]) / sum(add_e[ind]) * 100.0
            less_vsd_t[ind] = sum(vsd_less_t[ind]) / sum(vsd_e[ind]) * 100.0

        print('cat ', ind, ' rec ', detPre[ind], ' pre ', detRec[ind], ' less5 ', less_55[ind], ' repr ',
                  less_repr_5[ind], ' add ', less_add_d[ind], ' vsd ', less_vsd_t[ind], ' F1 add 0.15d ', F1_add[ind])

    dataset_recall = sum(tp) / (sum(tp) + sum(fp)) * 100.0
    dataset_precision = sum(tp) / (sum(tp) + sum(fn)) * 100.0
    dataset_recall_add = sum(tp_add) / (sum(tp_add) + sum(fp_add)) * 100.0
    dataset_precision_add = sum(tp_add) / (sum(tp_add) + sum(fn_add)) * 100.0
    F1_add_all = 2 * ((dataset_precision_add * dataset_recall_add)/(dataset_precision_add + dataset_recall_add))
    print('F1 add 0.15d: ', F1_add_all)
    less_55 = sum(less5) / sum(rotD) * 100.0
    less_repr_5 = sum(rep_less5) / sum(rep_e) * 100.0
    less_add_d = sum(add_less_d) / sum(add_e) * 100.0
    less_vsd_t = sum(vsd_less_t) / sum(vsd_e) * 100.0

    return dataset_recall, dataset_precision, less_55, less_vsd_t, less_repr_5, less_add_d
