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

import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."


def load_true_anno(generator, images):
    #annotations = generator.load_annotations(id)
    val_annos = []
    for id in images:
        val_annos.append(generator.load_annotations(int(id)))

    return val_annos


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
    """ Use the pycocotools to evaluate a COCO model on a dataset.

    Args
        generator : The generator for generating the evaluation data.
        model     : The model to evaluate.
        threshold : The score threshold to use.
    """
    # start collecting results
    results = []
    image_ids = []
    for index in progressbar.progressbar(range(generator.size()), prefix='LineMOD evaluation: '):

        image = generator.load_image(index)
        image = generator.preprocess_image(image)
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network
        boxes, quats, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        #print('boxes: ', boxes.shape)
        #print('boxes[0]', boxes[0])
        #print('quats: ', quats.shape)
        #print('labels: ', scores.shape)
        #print('labels: ', labels[0])

        # correct boxes for image scale
        boxes /= scale

        # change to (x, y, w, h) (MS COCO standard)
        boxes[:, :, 2] -= boxes[:, :, 0]
        boxes[:, :, 3] -= boxes[:, :, 1]

        # compute predicted labels and scores
        for box, quat, score, label in zip(boxes[0], quats[0], scores[0], labels[0]):
            # scores are sorted, so we can break
            #if score < threshold:
            if score < 0.5:
                break

            cls = generator.label_to_inv_label(label)

            # append detection for each positively labeled class
            image_result = {
                'image_id'    : generator.image_ids[index],
                'category_id' : generator.label_to_inv_label(label),
                'score'       : float(score),
                'bbox'        : box.tolist(),
                'pose'        : quat[(cls-1), :].tolist()
            }
            #print('bbox: ', box)
            #print('pose: ', quat[(cls-1), :])

            # append detection to results
            results.append(image_result)

        #print(generator.image_ids[index])
        # append image to list of processed images
        image_ids.append(generator.image_ids[index])

    if not len(results):
        return

    # write output
    json.dump(results, open('{}_bbox_results.json'.format(generator.set_name), 'w'), indent=4)
    json.dump(image_ids, open('{}_processed_image_ids.json'.format(generator.set_name), 'w'), indent=4)

    # load results in COCO evaluation tool
    ann_true = load_true_anno(generator, image_idx)

    tp = []
    tn = []
    fn = []
    for i, gt in enumerate(ann_true):

        print('gt: ', gt['labels'])

        pred = results['image_id'== image_ids[i]]
        cat_ind = pred['category_id'] == gt['labels']
        if not bool(cat_ind):
            fn.append(int(gt['labels']))
        else:
            print('else')

    coco_true = generator.coco
    coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(generator.set_name))

    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats
