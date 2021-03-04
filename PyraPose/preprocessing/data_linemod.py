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

from ..utils.image import read_image_bgr
from collections import defaultdict

import numpy as np
import os
import json
import yaml
import itertools
import random
import warnings
import copy
import cv2
import time

import tensorflow.keras as keras
import tensorflow as tf

from ..utils.anchors import (
    anchor_targets_bbox,
    anchors_for_shape,
    guess_shapes
)
from ..utils.config import parse_anchor_parameters
from ..utils.image import (
    TransformParameters,
    adjust_transform_for_image,
    adjust_transform_for_mask,
    apply_transform2mask,
    adjust_pose_annotation,
    apply_transform,
    preprocess_image,
    resize_image,
    read_image_bgr,
)
from ..utils.transform import transform_aabb


def LinemodGenerator(data_dir='/home/stefan/data/train_data/linemod_PBR_BOP', set_name='train', transform_generator=None, self_dir='val', batch_size=1, image_min_side=480, image_max_side=640):

    def _isArrayLike(obj):
        return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

    def load_classes(categories):
        """ Loads the class to label mapping (and inverse) for COCO.
        """
        if _isArrayLike(categories):
            categories = [categories[id] for id in categories]
        elif type(categories) == int:
            categories = [categories[categories]]
        categories.sort(key=lambda x: x['id'])
        classes = {}
        labels = {}
        labels_inverse = {}
        for c in categories:
            labels[len(classes)] = c['id']
            labels_inverse[c['id']] = len(classes)
            classes[c['name']] = len(classes)
        # also load the reverse (label -> name)
        labels_rev = {}
        for key, value in classes.items():
            labels_rev[value] = key

        return classes, labels, labels_inverse, labels_rev

    # Parameters
    data_dir = data_dir
    set_name = set_name
    domain = self_dir
    path = os.path.join(data_dir, 'annotations', 'instances_' + set_name + '.json')
    mesh_info = os.path.join(data_dir, 'annotations', 'models_info' + '.yml')

    batch_size             = int(batch_size)
    image_min_side         = image_min_side
    image_max_side         = image_max_side
    transform_parameters   = TransformParameters()
    compute_anchor_targets = anchor_targets_bbox
    compute_shapes         = guess_shapes

    with open(path, 'r') as js:
        data = json.load(js)

    # load source w/ annotations
    image_ann = data["images"]
    anno_ann = data["annotations"]
    cat_ann = data["categories"]
    cats = {}
    image_ids = []
    image_paths = []
    imgToAnns, catToImgs = defaultdict(list), defaultdict(list)

    for img in image_ann:
        if "fx" in img:
            fx = img["fx"]
            fy = img["fy"]
            cx = img["cx"]
            cy = img["cy"]
        image_ids.append(img['id'])  # to correlate indexing to self.image_ann
        image_paths.append(os.path.join(data_dir, 'images', set_name, img['file_name']))

    for cat in cat_ann:
        cats[cat['id']] = cat

    for ann in anno_ann:
        imgToAnns[ann['image_id']].append(ann)
        catToImgs[ann['category_id']].append(ann['image_id'])

    classes, labels, labels_inverse, labels_rev = load_classes(cats)

    #if self.domain is not None:
        # load target w/o annotations
    self_path = os.path.join(data_dir, 'annotations', 'instances_' + self_dir + '.json')
    with open(self_path, 'r') as js:
        data_pseudo = json.load(js)

    image_ann_ss = data_pseudo["images"]
    image_ids_ss = []
    image_paths_ss = []
    for img in image_ann_ss:
        image_ids_ss.append(img['id'])  # to correlate indexing to self.image_ann
        image_paths_ss.append(os.path.join(data_dir, 'images', self_dir, img['file_name']))

    # load 3D boxes
    TDboxes = np.ndarray((16, 8, 3), dtype=np.float32)

    for key, value in yaml.load(open(mesh_info)).items():
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
        TDboxes[int(key), :, :] = three_box_solo


    def load_image(image_index):
        """ Load an image at the image_index.
        """
        path = image_paths[image_index]
        path = path[:-4] + '_rgb' + path[-4:]

        return read_image_bgr(path)

    def load_annotations(image_index):
        """ Load annotations for an image_index.
            CHECK DONE HERE: Annotations + images correct
        """
        #ids = image_ids[image_index]

        #lists = [imgToAnns[imgId] for imgId in ids if imgId in imgToAnns]
        #anns = list(itertools.chain.from_iterable(lists))
        anns = imgToAnns[image_ids[image_index]]

        path = image_paths[image_index]
        mask_path = path[:-4] + '_mask.png'  # + path[-4:]
        mask = cv2.imread(mask_path, -1)

        target_domain = np.full((1), False)
        if str(domain) in path:
            target_domain = True

        annotations     = {'mask': mask, 'target_domain': target_domain, 'labels': np.empty((0,)), 'bboxes': np.empty((0, 4)), 'poses': np.empty((0, 7)), 'segmentations': np.empty((0, 8, 3)), 'cam_params': np.empty((0, 4)), 'mask_ids': np.empty((0,))}

        for idx, a in enumerate(anns):
            if set_name == 'train':
                if a['feature_visibility'] < 0.5:
                    continue
            annotations['labels'] = np.concatenate([annotations['labels'], [labels_inverse[a['category_id']]]], axis=0)
            annotations['bboxes'] = np.concatenate([annotations['bboxes'], [[
                a['bbox'][0],
                a['bbox'][1],
                a['bbox'][0] + a['bbox'][2],
                a['bbox'][1] + a['bbox'][3],
            ]]], axis=0)
            if a['pose'][2] < 10.0:  # needed for adjusting pose annotations
                a['pose'][0] = a['pose'][0] * 1000.0
                a['pose'][1] = a['pose'][1] * 1000.0
                a['pose'][2] = a['pose'][2] * 1000.0
            annotations['poses'] = np.concatenate([annotations['poses'], [[
                a['pose'][0],
                a['pose'][1],
                a['pose'][2],
                a['pose'][3],
                a['pose'][4],
                a['pose'][5],
                a['pose'][6],
            ]]], axis=0)
            annotations['mask_ids'] = np.concatenate([annotations['mask_ids'], [
                a['mask_id'],
            ]], axis=0)
            objID = a['category_id']
            threeDbox = TDboxes[objID, :, :]
            annotations['segmentations'] = np.concatenate([annotations['segmentations'], [threeDbox]], axis=0)
            annotations['cam_params'] = np.concatenate([annotations['cam_params'], [[
                fx,
                fy,
                cx,
                cy,
            ]]], axis=0)

        return annotations

    def random_transform_group_entry(image, annotations, transform=None):
        """ Randomly transforms image and annotation.
        """
        # randomly transform both image and annotations
        if transform is not None or transform_generator:
            if transform is None:
                next_transform = next(transform_generator)
                transform = adjust_transform_for_image(next_transform, image, transform_parameters.relative_translation)
                transform_mask = adjust_transform_for_mask(next_transform, annotations['mask'],transform_parameters.relative_translation)

            image = apply_transform(transform, image, transform_parameters)
            annotations['mask'] = apply_transform2mask(transform_mask, annotations['mask'], transform_parameters)

            # Transform the bounding boxes in the annotations.
            annotations['bboxes'] = annotations['bboxes'].copy()
            annotations['segmentations'] = annotations['segmentations'].copy()
            for index in range(annotations['bboxes'].shape[0]):
                annotations['bboxes'][index, :] = transform_aabb(transform, annotations['bboxes'][index, :])
                annotations['poses'][index, :] = adjust_pose_annotation(transform, annotations['poses'][index, :], annotations['cam_params'][index, :])

        return image, annotations

    def random_transform_target_entry(image, transform=None):
        """ Randomly transforms image and annotation.
        """

        # randomly transform both image and annotations
        if transform is not None or transform_generator:
            if transform is None:
                #transform = adjust_transform_for_image(next(self.transform_generator), image, self.transform_parameters.relative_translation)
                next_transform = next(transform_generator)
                transform = adjust_transform_for_image(next_transform, image,
                                                       transform_parameters.relative_translation)

            image = apply_transform(transform, image, transform_parameters)

        return image

    max_shape = (image_min_side, image_max_side, 3)
    anchors = anchors_for_shape(max_shape, anchor_params=None, shapes_callback=compute_shapes)


    while True:
        order_syn = list(range(len(image_ids)))
        order_real = list(range(len(image_ids_ss)))
        np.random.shuffle(order_syn)
        np.random.shuffle(order_real)
        groups_syn = [[order_syn[x % len(order_syn)] for x in range(i, i + batch_size)] for i in
                  range(0, len(order_syn), batch_size)]
        groups_real = [[order_real[x % len(order_real)] for x in range(i, i + batch_size)] for i in
                      range(0, len(order_real), batch_size)]

        batches_syn = np.arange(len(groups_syn))
        batches_real = np.arange(len(groups_real))

        for btx in range(len(batches_syn)):
            x_s, y_s, x_t = [], [], []

            x_s = [load_image(image_index) for image_index in groups_syn[btx]]
            x_t = [load_image(image_index) for image_index in groups_real[btx]]
            y_s = [load_annotations(image_index) for image_index in groups_syn[btx]]

            assert (len(x_s) == len(y_s) == len(x_t))

            # filter annotations
            for index, (image, annotations) in enumerate(zip(x_s, y_s)):
                # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
                invalid_indices = np.where(
                    (annotations['bboxes'][:, 2] <= annotations['bboxes'][:, 0]) |
                    (annotations['bboxes'][:, 3] <= annotations['bboxes'][:, 1]) |
                    (annotations['bboxes'][:, 0] < 0) |
                    (annotations['bboxes'][:, 1] < 0) |
                    (annotations['bboxes'][:, 2] > image.shape[1]) |
                    (annotations['bboxes'][:, 3] > image.shape[0])
                )[0]

                # delete invalid indices
                if len(invalid_indices):
                    for k in y_s[index].keys():
                        if k == 'target_domain' or k == 'mask' or k == 'depth':
                            continue
                        y_s[index][k] = np.delete(annotations[k], invalid_indices, axis=0)

                # transform a single group entry
                x_s[index], y_s[index] = random_transform_group_entry(x_s[index],y_s[index])

                # transform a single group entry
                x_t[index] = random_transform_target_entry(x_t[index])

                # preprocess
                x_s[index] = preprocess_image(x_s[index])
                x_t[index] = preprocess_image(x_t[index])
                x_s[index] = keras.backend.cast_to_floatx(x_s[index])
                x_t[index] = keras.backend.cast_to_floatx(x_t[index])

            # x_s to image_batch
            image_source_batch = np.zeros((batch_size,) + max_shape, dtype=keras.backend.floatx())
            for image_index, image in enumerate(x_s):
                image_source_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

            image_target_batch = np.zeros((batch_size,) + max_shape, dtype=keras.backend.floatx())
            for image_index, image in enumerate(x_s):
                image_target_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

            target_batch = compute_anchor_targets(anchors,x_s,y_s,len(classes))

            image_source_batch = tf.convert_to_tensor(image_source_batch, dtype=tf.float32)
            target_batch = tf.tuple(target_batch)
            image_target_batch = tf.convert_to_tensor(image_target_batch, dtype=tf.float32)

            #print(type(image_source_batch))
            #print(type(target_batch))
            #print(type(image_target_batch))

            yield image_source_batch, target_batch, image_target_batch