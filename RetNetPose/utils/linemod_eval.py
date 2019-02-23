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

import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."

# LineMOD
fxkin = 572.41140
fykin = 573.57043
cxkin = 325.26110
cykin = 242.04899


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
    zD = []
    less5cm = []
    rotD = []
    less5deg = []
    #val_size = generator.size()
    for index in progressbar.progressbar(range(generator.size()), prefix='LineMOD evaluation: '):

        image = generator.load_image(index)
        image = generator.preprocess_image(image)
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network
        boxes, trans, deps, rots, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

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
        for box, trans, deps, quat, score, label in zip(boxes[0], trans[0], deps[0], rots[0], scores[0], labels[0]):
            # scores are sorted, so we can break
            if score < threshold:
                continue

            if label < 0:
                continue
            cls = generator.label_to_inv_label(label)
            tra = trans[(cls-1), :]
            dep = np.argmax(deps[(cls-1), :]) * 0.035
            rot = quat[(cls-1), :]
            pose = tra.tolist() + [dep] + rot.tolist()

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
                        q1 = pyquaternion.Quaternion(t_rot).unit
                        q2 = pyquaternion.Quaternion(rot).unit
                        #q1 = pyquaternion.Quaternion(t_pose)
                        #q2 = pyquaternion.Quaternion(rot)
                        #print('translation target: ', t_tra, '       estimation: ', tra)
                        #print('depth target: ', t_dep, '             estimation: ', dep)

                        x = ((tra[0] - cxkin) * t_dep) / fxkin * 0.001
                        x_t = ((t_tra[0] - cxkin) * t_dep) / fxkin * 0.001
                        y = ((tra[1] - cykin) * t_dep) / fykin * 0.001
                        y_t = ((t_tra[1] - cykin) * t_dep) / fykin * 0.001
                        xd = np.abs(np.abs(x) - np.abs(x_t))
                        yd = np.abs(np.abs(y) - np.abs(y_t))
                        xyd = np.linalg.norm(np.asarray([xd, yd], dtype=np.float32))
                        if not math.isnan(xyd):
                            xyD.append(xyd)
                            if xyd < 0.05:
                                less5cm.append(xyd)
                        zd = np.abs(np.abs(dep) - np.abs(t_dep*0.001))
                        if not math.isnan(zd):
                            zD.append(zd)
                        rd = pyquaternion.Quaternion.distance(q1, q2)
                        if not math.isnan(rd):
                            rotD.append(rd)
                            if (rd * 180/math.pi) < 5.0:
                                less5deg.append(rd)
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
    dataset_xy_diff = (sum(xyD) / len(xyD))
    dataset_depth_diff = (sum(zD) / len(zD))
    dataset_rot_diff = (sum(rotD) / len(rotD)) * 180.0 / math.pi
    less5cm = len(less5cm)/len(xyD)
    less5deg = len(less5deg) / len(rotD)

    return dataset_recall, dataset_precision, dataset_xy_diff, dataset_depth_diff, dataset_rot_diff, less5cm, less5deg
