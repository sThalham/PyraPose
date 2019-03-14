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

import keras.backend
from .dynamic import meshgrid


def bbox_transform_inv(boxes, deltas, mean=None, std=None):
    """ Applies deltas (usually regression results) to boxes (usually anchors).

    Before applying the deltas to the boxes, the normalization that was previously applied (in the generator) has to be removed.
    The mean and std are the mean and std as applied in the generator. They are unnormalized in this function and then applied to the boxes.

    Args
        boxes : np.array of shape (B, N, 4), where B is the batch size, N the number of boxes and 4 values for (x1, y1, x2, y2).
        deltas: np.array of same shape as boxes. These deltas (d_x1, d_y1, d_x2, d_y2) are a factor of the width/height.
        mean  : The mean value used when computing deltas (defaults to [0, 0, 0, 0]).
        std   : The standard deviation used when computing deltas (defaults to [0.2, 0.2, 0.2, 0.2]).

    Returns
        A np.array of the same shape as boxes, but with deltas applied to each box.
        The mean and std are used during training to normalize the regression values (networks love normalization).
    """
    if mean is None:
        mean = [0, 0, 0, 0]
    if std is None:
        std = [0.2, 0.2, 0.2, 0.2]

    width  = boxes[:, :, 2] - boxes[:, :, 0]
    height = boxes[:, :, 3] - boxes[:, :, 1]

    x1 = boxes[:, :, 0] + (deltas[:, :, 0] * std[0] + mean[0]) * width
    y1 = boxes[:, :, 1] + (deltas[:, :, 1] * std[1] + mean[1]) * height
    x2 = boxes[:, :, 2] + (deltas[:, :, 2] * std[2] + mean[2]) * width
    y2 = boxes[:, :, 3] + (deltas[:, :, 3] * std[3] + mean[3]) * height

    pred_boxes = keras.backend.stack([x1, y1, x2, y2], axis=2)

    return pred_boxes


def box3D_transform_inv(boxes, deltas, mean=None, std=None):

    if mean is None:
        mean = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if std is None:
        std = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

    num_classes = keras.backend.int_shape(deltas)[2]

    boxes_exp = keras.backend.expand_dims(boxes, axis=2)
    boxes_exp = keras.backend.repeat_elements(boxes_exp, num_classes, axis=2)

    width  = boxes_exp[:, :, :, 2] - boxes_exp[:, :, :, 0]
    height = boxes_exp[:, :, :, 3] - boxes_exp[:, :, :, 1]

    x1 = boxes_exp[:, :, :, 0] + (deltas[:, :, :, 0] * std[0] + mean[0]) * width
    y1 = boxes_exp[:, :, :, 1] + (deltas[:, :, :, 1] * std[1] + mean[1]) * height
    x2 = boxes_exp[:, :, :, 2] + (deltas[:, :, :, 2] * std[2] + mean[2]) * width
    y2 = boxes_exp[:, :, :, 3] + (deltas[:, :, :, 3] * std[3] + mean[3]) * height
    x3 = boxes_exp[:, :, :, 0] + (deltas[:, :, :, 4] * std[4] + mean[4]) * width
    y3 = boxes_exp[:, :, :, 1] + (deltas[:, :, :, 5] * std[5] + mean[5]) * height
    x4 = boxes_exp[:, :, :, 2] + (deltas[:, :, :, 6] * std[6] + mean[6]) * width
    y4 = boxes_exp[:, :, :, 3] + (deltas[:, :, :, 7] * std[7] + mean[7]) * height
    x5 = boxes_exp[:, :, :, 0] + (deltas[:, :, :, 8] * std[8] + mean[8]) * width
    y5 = boxes_exp[:, :, :, 1] + (deltas[:, :, :, 9] * std[9] + mean[9]) * height
    x6 = boxes_exp[:, :, :, 2] + (deltas[:, :, :, 10] * std[10] + mean[10]) * width
    y6 = boxes_exp[:, :, :, 3] + (deltas[:, :, :, 11] * std[11] + mean[11]) * height
    x7 = boxes_exp[:, :, :, 0] + (deltas[:, :, :, 12] * std[12] + mean[12]) * width
    y7 = boxes_exp[:, :, :, 1] + (deltas[:, :, :, 13] * std[13] + mean[13]) * height
    x8 = boxes_exp[:, :, :, 2] + (deltas[:, :, :, 14] * std[14] + mean[14]) * width
    y8 = boxes_exp[:, :, :, 3] + (deltas[:, :, :, 15] * std[15] + mean[15]) * height

    pred_boxes = keras.backend.stack([x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8], axis=3)

    return pred_boxes


def shift(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size.

    Args
        shape  : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """
    shift_x = (keras.backend.arange(0, shape[1], dtype=keras.backend.floatx()) + keras.backend.constant(0.5, dtype=keras.backend.floatx())) * stride
    shift_y = (keras.backend.arange(0, shape[0], dtype=keras.backend.floatx()) + keras.backend.constant(0.5, dtype=keras.backend.floatx())) * stride

    shift_x, shift_y = meshgrid(shift_x, shift_y)
    shift_x = keras.backend.reshape(shift_x, [-1])
    shift_y = keras.backend.reshape(shift_y, [-1])

    shifts = keras.backend.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    shifts            = keras.backend.transpose(shifts)
    number_of_anchors = keras.backend.shape(anchors)[0]

    k = keras.backend.shape(shifts)[0]  # number of base points = feat_h * feat_w

    shifted_anchors = keras.backend.reshape(anchors, [1, number_of_anchors, 4]) + keras.backend.cast(keras.backend.reshape(shifts, [k, 1, 4]), keras.backend.floatx())
    shifted_anchors = keras.backend.reshape(shifted_anchors, [k * number_of_anchors, 4])

    return shifted_anchors


def translation_transform_inv(anchors, deltas, deltas_pose, mean=None, std=None):
    if mean is None:
        mean = [0.0, 0.0]
    if std is None:
        std = [0.4, 0.4]

    num_classes = keras.backend.int_shape(deltas_pose)[2]

    deltas_exp = keras.backend.expand_dims(deltas, axis=2)
    #deltas_exp = keras.backend.expand_dims(anchors, axis=2)
    deltas_exp = keras.backend.repeat_elements(deltas_exp, num_classes, axis=2)

    width = deltas_exp[:, :, :, 2] - deltas_exp[:, :, :, 0]
    height = deltas_exp[:, :, :, 3] - deltas_exp[:, :, :, 1]

    x = deltas_exp[:, :, :, 0] + (deltas_pose[:, :, :, 0] * std[0] + mean[0]) * width
    y = deltas_exp[:, :, :, 1] + (deltas_pose[:, :, :, 1] * std[1] + mean[1]) * height

    pred_pose = keras.backend.stack([x, y], axis=3)

    return pred_pose


def depth_transform_inv(boxes, deltas, deltas_pose, mean=None, std=None):
    if mean is None:
        mean = [0.0]
    if std is None:
        std = [0.4]

    #widths = deltas[:, :, 2] - deltas[:, :, 0]
    #heights = deltas[:, :, 3] - deltas[:, :, 1]
    #box_diag = keras.backend.sqrt((keras.backend.square(widths) + keras.backend.square(heights)))

    #num_classes = keras.backend.int_shape(deltas_pose)[2]
    #deltas_exp = keras.backend.expand_dims(box_diag, axis=2)
    #deltas_exp = keras.backend.repeat_elements(deltas_exp, num_classes, axis=2)
    #deltas_exp = keras.backend.expand_dims(deltas_exp, axis=3)
    #meters = (deltas_pose[:, :, :, 0] * deltas_exp[:, :, :, 0]) * 0.01 * std[0] + mean[0]
    meters = deltas_pose[:, :, :, 0] * std[0] + mean[0]

    dep_cls = keras.backend.stack([meters], axis=3)

    return dep_cls


def rotation_transform_inv(poses, deltas, mean=None, std=None):
    if mean is None:
        mean = [0.0, 0.0, 0.0, 0.0]
    if std is None:
        std = [1.0, 1.0, 1.0, 1.0]

    rx = deltas[:, :, :, 0] * std[0] + mean[0]
    ry = deltas[:, :, :, 1] * std[1] + mean[1]
    rz = deltas[:, :, :, 2] * std[2] + mean[2]
    rw = deltas[:, :, :, 3] * std[3] + mean[3]

    pred_pose = keras.backend.stack([rx, ry, rz, rw], axis=3)
    #pred_pose = keras.backend.stack([rx, ry, rz], axis=3)

    return pred_pose

