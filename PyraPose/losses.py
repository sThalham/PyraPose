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

from tensorflow import keras
import tensorflow as tf
from . import backend


def focal(alpha=0.25, gamma=2.0):
    """ Create a functor for computing the focal loss.

    Args
        alpha: Scale the focal weight with alpha. vanilla 0.25 2.0
        gamma: Take the power of the focal weight with gamma.

    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """
    def _focal(y_true, y_pred):
        """ Compute the focal loss given the target tensor and the predicted tensor.

        As defined in https://arxiv.org/abs/1708.02002

        Args
            y_true: Tensor of target data from the generator with shape (B, N, num_classes).
            y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).

        Returns
            The focal loss of y_pred w.r.t. y_true.
        """
        labels         = y_true[:, :, :-1]
        anchor_state   = y_true[:, :, -1]  # -1 for ignore, 0 for background, 1 for object
        classification = y_pred

        # filter out "ignore" anchors
        indices        = backend.where(keras.backend.not_equal(anchor_state, -1))
        labels         = backend.gather_nd(labels, indices)
        classification = backend.gather_nd(classification, indices)

        # compute the focal loss
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = backend.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = backend.where(keras.backend.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)

        # compute the normalizer: the number of positive anchors
        normalizer = backend.where(keras.backend.equal(anchor_state, 1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)

        loss = keras.backend.sum(cls_loss) / normalizer

        return loss

    return _focal


def focal_mask(alpha=0.25, gamma=2.0):
    """ Create a functor for computing the focal loss.

    Args
        alpha: Scale the focal weight with alpha. vanilla 0.25 2.0
        gamma: Take the power of the focal weight with gamma.

    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """
    def _focal_mask(y_true, y_pred):
        """ Compute the focal loss given the target tensor and the predicted tensor.

        As defined in https://arxiv.org/abs/1708.02002

        Args
            y_true: Tensor of target data from the generator with shape (B, N, num_classes).
            y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).

        Returns
            The focal loss of y_pred w.r.t. y_true.
        """
        labels         = y_true[:, :, :-1]
        anchor_state   = y_true[:, :, -1]  # -1 for ignore, 0 for background, 1 for object
        classification = y_pred

        # filter out "ignore" anchors
        indices        = backend.where(keras.backend.not_equal(anchor_state, -1))
        labels         = backend.gather_nd(labels, indices)
        classification = backend.gather_nd(classification, indices)

        # compute the focal loss
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = backend.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = backend.where(keras.backend.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = 0.1 * focal_weight * keras.backend.binary_crossentropy(labels, classification)

        # compute the normalizer: the number of positive anchors
        normalizer = backend.where(keras.backend.equal(anchor_state, 1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)

        return keras.backend.sum(cls_loss) / normalizer

    return _focal_mask


def smooth_l1(sigma=3.0):
    """ Create a smooth L1 loss functor.

    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.

    Returns
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.

        Args
            y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4).

        Returns
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        regression        = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state      = y_true[:, :, -1]

        # filter out "ignore" anchors
        indices           = backend.where(keras.backend.equal(anchor_state, 1))
        regression        = backend.gather_nd(regression, indices)
        regression_target = backend.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = backend.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        loss = keras.backend.sum(regression_loss) / normalizer
        return loss

    return _smooth_l1


def smooth_l1_pose(sigma=3.0):
    """ Create a smooth L1 loss functor.

    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.

    Returns
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    def _smooth_l1_pose(y_true, y_pred):
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.

        Args
            y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4).

        Returns
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        regression        = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state      = y_true[:, :, -1]

        # filter out "ignore" anchors
        indices           = backend.where(keras.backend.equal(anchor_state, 1))
        regression        = backend.gather_nd(regression, indices)
        regression_target = backend.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = backend.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        loss = keras.backend.sum(regression_loss) / normalizer
        return 0.1 * loss

    return _smooth_l1_pose


def orthogonal_l1(weight=0.125, sigma=3.0):

    weight_xy = 0.8
    weight_orth = 0.2
    sigma_squared = sigma ** 2

    def _orth_l1(y_true, y_pred):

        regression        = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state      = y_true[:, :, -1]

        #### filter out "ignore" anchors
        indices           = backend.where(keras.backend.equal(anchor_state, 1))
        regression        = backend.gather_nd(regression, indices)
        regression_target = backend.gather_nd(regression_target, indices)

        x1 = (regression[:, 0] - regression[:, 6]) - (regression[:, 2] - regression[:, 4])
        y1 = (regression[:, 1] - regression[:, 7]) - (regression[:, 3] - regression[:, 5])
        x2 = (regression[:, 0] - regression[:, 6]) - (regression[:, 8] - regression[:, 14])
        y2 = (regression[:, 1] - regression[:, 7]) - (regression[:, 9] - regression[:, 15])
        x3 = (regression[:, 0] - regression[:, 2]) - (regression[:, 6] - regression[:, 4])
        y3 = (regression[:, 1] - regression[:, 3]) - (regression[:, 7] - regression[:, 5])
        x4 = (regression[:, 0] - regression[:, 2]) - (regression[:, 8] - regression[:, 10])
        y4 = (regression[:, 1] - regression[:, 3]) - (regression[:, 9] - regression[:, 11])   # up to here ok
        x5 = (regression[:, 0] - regression[:, 8]) - (regression[:, 2] - regression[:, 10])
        y5 = (regression[:, 1] - regression[:, 9]) - (regression[:, 3] - regression[:, 11])
        x6 = (regression[:, 0] - regression[:, 8]) - (regression[:, 6] - regression[:, 14])
        y6 = (regression[:, 1] - regression[:, 9]) - (regression[:, 7] - regression[:, 15])   # half way done
        x7 = (regression[:, 12] - regression[:, 10]) - (regression[:, 14] - regression[:, 8])
        y7 = (regression[:, 13] - regression[:, 11]) - (regression[:, 15] - regression[:, 9])
        x8 = (regression[:, 12] - regression[:, 10]) - (regression[:, 4] - regression[:, 2])
        y8 = (regression[:, 13] - regression[:, 11]) - (regression[:, 5] - regression[:, 3])
        x9 = (regression[:, 12] - regression[:, 4]) - (regression[:, 10] - regression[:, 2])
        y9 = (regression[:, 13] - regression[:, 5]) - (regression[:, 11] - regression[:, 3])
        x10 = (regression[:, 12] - regression[:, 4]) - (regression[:, 14] - regression[:, 6])
        y10 = (regression[:, 13] - regression[:, 5]) - (regression[:, 15] - regression[:, 7])
        x11 = (regression[:, 12] - regression[:, 14]) - (regression[:, 4] - regression[:, 6])
        y11 = (regression[:, 13] - regression[:, 15]) - (regression[:, 5] - regression[:, 7])
        x12 = (regression[:, 12] - regression[:, 14]) - (regression[:, 10] - regression[:, 8])
        y12 = (regression[:, 13] - regression[:, 15]) - (regression[:, 11] - regression[:, 9])
        orths = keras.backend.stack([x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9, x10, y10, x11, y11, x12, y12], axis=1)

        xt1 = (regression_target[:, 0] - regression_target[:, 6]) - (regression_target[:, 2] - regression_target[:, 4])
        yt1 = (regression_target[:, 1] - regression_target[:, 7]) - (regression_target[:, 3] - regression_target[:, 5])
        xt2 = (regression_target[:, 0] - regression_target[:, 6]) - (regression_target[:, 8] - regression_target[:, 14])
        yt2 = (regression_target[:, 1] - regression_target[:, 7]) - (regression_target[:, 9] - regression_target[:, 15])
        xt3 = (regression_target[:, 0] - regression_target[:, 2]) - (regression_target[:, 6] - regression_target[:, 4])
        yt3 = (regression_target[:, 1] - regression_target[:, 3]) - (regression_target[:, 7] - regression_target[:, 5])
        xt4 = (regression_target[:, 0] - regression_target[:, 2]) - (regression_target[:, 8] - regression_target[:, 10])
        yt4 = (regression_target[:, 1] - regression_target[:, 3]) - (regression_target[:, 9] - regression_target[:, 11])  # up to here ok
        xt5 = (regression_target[:, 0] - regression_target[:, 8]) - (regression_target[:, 2] - regression_target[:, 10])
        yt5 = (regression_target[:, 1] - regression_target[:, 9]) - (regression_target[:, 3] - regression_target[:, 11])
        xt6 = (regression_target[:, 0] - regression_target[:, 8]) - (regression_target[:, 6] - regression_target[:, 14])
        yt6 = (regression_target[:, 1] - regression_target[:, 9]) - (regression_target[:, 7] - regression_target[:, 15])  # half way done
        xt7 = (regression_target[:, 12] - regression_target[:, 10]) - (regression_target[:, 14] - regression_target[:, 8])
        yt7 = (regression_target[:, 13] - regression_target[:, 11]) - (regression_target[:, 15] - regression_target[:, 9])
        xt8 = (regression_target[:, 12] - regression_target[:, 10]) - (regression_target[:, 4] - regression_target[:, 2])
        yt8 = (regression_target[:, 13] - regression_target[:, 11]) - (regression_target[:, 5] - regression_target[:, 3])
        xt9 = (regression_target[:, 12] - regression_target[:, 4]) - (regression_target[:, 10] - regression_target[:, 2])
        yt9 = (regression_target[:, 13] - regression_target[:, 5]) - (regression_target[:, 11] - regression_target[:, 3])
        xt10 = (regression_target[:, 12] - regression_target[:, 4]) - (regression_target[:, 14] - regression_target[:, 6])
        yt10 = (regression_target[:, 13] - regression_target[:, 5]) - (regression_target[:, 15] - regression_target[:, 7])
        xt11 = (regression_target[:, 12] - regression_target[:, 14]) - (regression_target[:, 4] - regression_target[:, 6])
        yt11 = (regression_target[:, 13] - regression_target[:, 15]) - (regression_target[:, 5] - regression_target[:, 7])
        xt12 = (regression_target[:, 12] - regression_target[:, 14]) - (regression_target[:, 10] - regression_target[:, 8])
        yt12 = (regression_target[:, 13] - regression_target[:, 15]) - (regression_target[:, 11] - regression_target[:, 9])
        orths_target = keras.backend.stack(
            [xt1, yt1, xt2, yt2, xt3, yt3, xt4, yt4, xt5, yt5, xt6, yt6, xt7, yt7, xt8, yt8, xt9, yt9, xt10, yt10, xt11, yt11, xt12, yt12],
            axis=1)

        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_xy = backend.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )
        regression_orth = keras.losses.mean_absolute_error(orths, orths_target)

        #### compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        regression_loss_xy = keras.backend.sum(regression_xy) / normalizer
        regression_loss_orth = keras.backend.sum(regression_orth) / normalizer
        return weight * (weight_xy * regression_loss_xy + weight_orth * regression_loss_orth)

    return _orth_l1


def orthogonality_loss_hyps(inputs):
    weight_xy = 0.8
    sigma_squared = 9.0

    regression, regression_target = inputs

    #print('orthogonal_loss_hyps::regression: ', regression)
    #print('orthogonal_loss_hyps::regression_target: ', regression_target)

    x1 = (regression[0] - regression[6]) - (regression[2] - regression[4])
    y1 = (regression[1] - regression[7]) - (regression[3] - regression[5])
    x2 = (regression[0] - regression[6]) - (regression[8] - regression[14])
    y2 = (regression[1] - regression[7]) - (regression[9] - regression[15])
    x3 = (regression[0] - regression[2]) - (regression[6] - regression[4])
    y3 = (regression[1] - regression[3]) - (regression[7] - regression[5])
    x4 = (regression[0] - regression[2]) - (regression[8] - regression[10])
    y4 = (regression[1] - regression[3]) - (regression[9] - regression[11])  # up to here ok
    x5 = (regression[0] - regression[8]) - (regression[2] - regression[10])
    y5 = (regression[1] - regression[9]) - (regression[3] - regression[11])
    x6 = (regression[0] - regression[8]) - (regression[6] - regression[14])
    y6 = (regression[1] - regression[9]) - (regression[7] - regression[15])  # half way done
    x7 = (regression[12] - regression[10]) - (regression[14] - regression[8])
    y7 = (regression[13] - regression[11]) - (regression[15] - regression[9])
    x8 = (regression[12] - regression[10]) - (regression[4] - regression[2])
    y8 = (regression[13] - regression[11]) - (regression[5] - regression[3])
    x9 = (regression[12] - regression[4]) - (regression[10] - regression[2])
    y9 = (regression[13] - regression[5]) - (regression[11] - regression[3])
    x10 = (regression[12] - regression[4]) - (regression[14] - regression[6])
    y10 = (regression[13] - regression[5]) - (regression[15] - regression[7])
    x11 = (regression[12] - regression[14]) - (regression[4] - regression[6])
    y11 = (regression[13] - regression[15]) - (regression[5] - regression[7])
    x12 = (regression[12] - regression[14]) - (regression[10] - regression[8])
    y12 = (regression[3] - regression[15]) - (regression[11] - regression[9])
    orths = keras.backend.stack(
        [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9, x10, y10, x11, y11, x12, y12], axis=0)

    xt1 = (regression_target[0] - regression_target[6]) - (regression_target[2] - regression_target[4])
    yt1 = (regression_target[1] - regression_target[7]) - (regression_target[3] - regression_target[5])
    xt2 = (regression_target[0] - regression_target[6]) - (regression_target[8] - regression_target[14])
    yt2 = (regression_target[1] - regression_target[7]) - (regression_target[9] - regression_target[15])
    xt3 = (regression_target[0] - regression_target[2]) - (regression_target[6] - regression_target[4])
    yt3 = (regression_target[1] - regression_target[3]) - (regression_target[7] - regression_target[5])
    xt4 = (regression_target[0] - regression_target[2]) - (regression_target[8] - regression_target[10])
    yt4 = (regression_target[1] - regression_target[3]) - (regression_target[9] - regression_target[11])  # up to here ok
    xt5 = (regression_target[0] - regression_target[8]) - (regression_target[2] - regression_target[10])
    yt5 = (regression_target[1] - regression_target[9]) - (regression_target[3] - regression_target[11])
    xt6 = (regression_target[0] - regression_target[8]) - (regression_target[6] - regression_target[14])
    yt6 = (regression_target[1] - regression_target[9]) - (regression_target[7] - regression_target[15])  # half way done
    xt7 = (regression_target[12] - regression_target[10]) - (regression_target[14] - regression_target[8])
    yt7 = (regression_target[13] - regression_target[11]) - (regression_target[15] - regression_target[9])
    xt8 = (regression_target[12] - regression_target[10]) - (regression_target[4] - regression_target[2])
    yt8 = (regression_target[13] - regression_target[11]) - (regression_target[5] - regression_target[3])
    xt9 = (regression_target[12] - regression_target[4]) - (regression_target[10] - regression_target[2])
    yt9 = (regression_target[13] - regression_target[5]) - (regression_target[11] - regression_target[3])
    xt10 = (regression_target[12] - regression_target[4]) - (regression_target[14] - regression_target[6])
    yt10 = (regression_target[13] - regression_target[5]) - (regression_target[15] - regression_target[7])
    xt11 = (regression_target[12] - regression_target[14]) - (regression_target[4] - regression_target[6])
    yt11 = (regression_target[13] - regression_target[15]) - (regression_target[5] - regression_target[7])
    xt12 = (regression_target[12] - regression_target[14]) - (regression_target[10] - regression_target[8])
    yt12 = (regression_target[13] - regression_target[15]) - (regression_target[11] - regression_target[9])
    orths_target = keras.backend.stack(
        [xt1, yt1, xt2, yt2, xt3, yt3, xt4, yt4, xt5, yt5, xt6, yt6, xt7, yt7, xt8, yt8, xt9, yt9, xt10, yt10, xt11,
         yt11, xt12, yt12],
        axis=0)

    regression_orth = keras.losses.mean_absolute_error(orths, orths_target)

    regression_diff = regression - regression_target
    regression_diff = keras.backend.abs(regression_diff)
    regression_xy = backend.where(
        keras.backend.less(regression_diff, 1.0 / sigma_squared),
        0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
        regression_diff - 0.5 / sigma_squared
    )
    regression_xy = keras.backend.sum(regression_xy)
    #print('orthogonal_loss_hyps::regression_xy: ', regression_xy)
    #print('orthogonal_loss_hyps::regression_orth: ', regression_orth)
    #regression_orth = keras.backend.print_tensor(regression_orth, message='regression_orth: ')
    #regression_xy = keras.backend.print_tensor(regression_xy, message='regression_xy: ')

    return weight_xy * regression_xy + (1 - weight_xy) * regression_orth
    #return 0.85 * regression_xy


def orthogonal_l1_local(inputs):
    regression, regression_target = inputs
    #print('orthogonal_l1_local::regression: ', regression)
    #print('orthogonal_l1_local::regression_target: ', regression_target)

    loss_hypotheses = backend.vectorized_map(orthogonality_loss_hyps, (regression, regression_target))
    #print('orthogonal_l1_local::loss_hypotheses: ', loss_hypotheses)
    min_loss_idx = keras.backend.argmin(loss_hypotheses, axis=0)
    #print('orthogonal_l1_local::min_loss_idx: ', min_loss_idx)
    #loss_hypotheses = keras.backend.print_tensor(loss_hypotheses, message='loss_hypotheses: ')
    #min_loss_idx = keras.backend.print_tensor(min_loss_idx, message='loss_min: ')
    loss_min = backend.gather(loss_hypotheses, min_loss_idx)
    #print('orthogonal_l1_local::loss_min: ', loss_min)

    return loss_min


'''
def sym_orthogonal_l1(weight=0.125, sigma=3.0):

    weight_xy = 0.8
    sigma_squared = sigma ** 2

    def _sym_orth_l1(y_true, y_pred):

        #y_pred = keras.backend.expand_dims(y_pred, axis=2) # hackiest !
        y_pred = keras.backend.repeat_elements(x=y_pred, rep=4, axis=2)
        #print('y_pred: ', y_pred)
        regression        = y_pred
        regression_target = y_true[:, :, :, :-1]
        anchor_state      = y_true[:, :, 0, -1]

        #### filter out "ignore" anchors
        indices           = backend.where(keras.backend.equal(anchor_state, 1))
        regression        = backend.gather_nd(regression, indices)
        regression_target = backend.gather_nd(regression_target, indices)

        #print('sym_orthogonal_l1::regression: ', regression)
        #print('sym_orthogonal_l1::regression_target: ', regression_target)

        regression_loss = backend.vectorized_map(orthogonal_l1_local, (regression, regression_target))

        #### compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        regression_loss = keras.backend.sum(regression_loss) / normalizer

        return weight * regression_loss

    return _sym_orth_l1
'''


def sym_orthogonal_l1(weight=0.125, sigma=3.0):

    sigma_squared = sigma ** 2

    def _sym_orthogonal_l1(y_true, y_pred):
        #regression = y_pred
        regression_target = y_true[:, :, :, :-1]
        anchor_state = y_true[:, :, 0, -1]

        in_shape = tf.shape(regression_target)
        tf.print('in_shape: ', in_shape)
        anchor_state = tf.reshape(anchor_state, [in_shape[0] * in_shape[1]])
        #indices = tf.math.reduce_max(anchor_state, axis=1)
        indices = tf.where(tf.math.equal(anchor_state, 1))[:, 0]

        y_pred_res = tf.reshape(y_pred, [in_shape[0] * in_shape[1], in_shape[3]])
        regression = tf.gather(y_pred_res, indices, axis=0)
        y_true_res = tf.reshape(regression_target, [in_shape[0] * in_shape[1], in_shape[2], in_shape[3]])
        regression_target = tf.gather(y_true_res, indices, axis=0)

        regression = tf.transpose(regression, perm=[1, 0])
        regression_target = tf.transpose(regression_target, perm=[1, 2, 0])

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression_target - regression
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = backend.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )
        regression_loss = tf.math.reduce_min(regression_loss, axis=0) # reduce regression loss to min hypothesis
        regression_loss = tf.math.reduce_sum(regression_loss, axis=1)

        #normalizer = tf.math.reduce_sum(anchor_state)
        normalizer = tf.cast(tf.shape(indices)[0], dtype=tf.float32)
        tf.print('normalizer: ', normalizer)

        return weight * tf.math.divide_no_nan(regression_loss, normalizer)

    return _sym_orthogonal_l1


def project_points(inputs):

    #xpix = (translation[:, 0] * fx) / translation[:, 2]
    #ypix = (translation[:, 1] * fy) / translation[:, 2]

    regression, target_intrinsics, target_pose = inputs
    x_values = regression[::2]
    y_values = regression[1::2]
    #print('regression: ', regression)
    #print('target_intrinsics: ', target_intrinsics)
    #print('target_pose: ', target_pose)

    xVal = (x_values * target_pose[11]) / target_intrinsics[0]
    yVal = (y_values * target_pose[11]) / target_intrinsics[1]
    #print('xVal: ', xVal)
    #print('yVal: ', yVal)
    point_array = [xVal[0], yVal[0], xVal[1], yVal[1], xVal[2], yVal[2], xVal[3], yVal[3], xVal[4], yVal[4], xVal[5], yVal[5], xVal[6], yVal[6], xVal[7], yVal[7]]
    point_array = keras.backend.stack(point_array, axis=0)
    #print('point_array: ', point_array)

    return point_array


def transformer_smooth_l1(weight=0.125, sigma=3.0):

    weight_xy = 0.8
    sigma_squared = sigma ** 2

    def _transformer_smooth_l1(y_true, y_pred):

        regression        = y_pred
        regression_target = y_true[:, :, :16]
        target_intrinsics = y_true[:, :, 16:20]
        target_pose = y_true[:, :, 20:36]
        anchor_state      = y_true[:, :, -1]

        #print('regression_target: ', regression_target)
        #print('target_intrinsics: ', target_intrinsics)
        #print('target_pose: ', target_pose)
        #print('anchor_state: ', anchor_state)

        #### filter out "ignore" anchors
        indices           = backend.where(keras.backend.equal(anchor_state, 1))
        regression        = backend.gather_nd(regression, indices)
        regression_target = backend.gather_nd(regression_target, indices)
        target_intrinsics = backend.gather_nd(target_intrinsics, indices)
        target_pose = backend.gather_nd(target_pose, indices)

        #print('indices: ', indices)
        #print('regression: ', regression)
        print('regression_target: ', regression_target)

        regression_projected = backend.vectorized_map(project_points, (regression, target_intrinsics, target_pose))

        print('regression_projected: ', regression_projected)

        regression_diff = regression_projected - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = backend.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        #### compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        regression_loss = keras.backend.sum(regression_loss) / normalizer

        return weight * regression_loss

    return _transformer_smooth_l1



'''    
def sym_orthogonal_l1(weight=0.125, sigma=3.0):

    weight_xy = 0.8
    sigma_squared = sigma ** 2

    def _sym_orth_l1(y_true, y_pred):

        #y_true = keras.backend.expand_dims(y_true, axis=2) # hackiest !
        #y_true = keras.backend.repeat_elements(x=y_true, rep=8, axis=2)
        regression        = y_pred
        regression_target = y_true[:, :, :, :-1]
        anchor_state      = y_true[:, :, 0, -1]

        #### filter out "ignore" anchors
        indices           = backend.where(keras.backend.equal(anchor_state, 1))
        regression        = backend.gather_nd(regression, indices)[:, 0, :]
        regression_target = backend.gather_nd(regression_target, indices)

        oct0 = orthogonal_l1_local(regression, regression_target[:, 0, :], indices, weight=weight, sigma_squared=sigma_squared, weight_xy=weight_xy)
        oct1 = orthogonal_l1_local(regression, regression_target[:, 1, :], indices,
                                   weight=weight, sigma_squared=sigma_squared, weight_xy=weight_xy)
        oct2 = orthogonal_l1_local(regression, regression_target[:, 2, :], indices,
                                   weight=weight, sigma_squared=sigma_squared, weight_xy=weight_xy)
        oct3 = orthogonal_l1_local(regression, regression_target[:, 3, :], indices,
                                   weight=weight, sigma_squared=sigma_squared, weight_xy=weight_xy)
        oct4 = orthogonal_l1_local(regression, regression_target[:, 4, :], indices,
                                   weight=weight, sigma_squared=sigma_squared, weight_xy=weight_xy)
        oct5 = orthogonal_l1_local(regression, regression_target[:, 5, :], indices,
                                   weight=weight, sigma_squared=sigma_squared, weight_xy=weight_xy)
        oct6 = orthogonal_l1_local(regression, regression_target[:, 6, :], indices,
                                   weight=weight, sigma_squared=sigma_squared, weight_xy=weight_xy)
        oct7 = orthogonal_l1_local(regression, regression_target[:, 7, :], indices,
                                   weight=weight, sigma_squared=sigma_squared, weight_xy=weight_xy)

        oct0 = keras.backend.print_tensor(oct0, message='Value of 0')
        oct1 = keras.backend.print_tensor(oct1, message='Value of 1')
        oct2 = keras.backend.print_tensor(oct2, message='Value of 2')
        oct3 = keras.backend.print_tensor(oct3, message='Value of 3')
        oct4 = keras.backend.print_tensor(oct4, message='Value of 4')
        oct5 = keras.backend.print_tensor(oct5, message='Value of 5')
        oct6 = keras.backend.print_tensor(oct6, message='Value of 6')
        oct7 = keras.backend.print_tensor(oct7, message='Value of 7')

        hypotheses = keras.backend.stack([oct0, oct1, oct2, oct3, oct4, oct5, oct6, oct7], axis=0)
        #print('hypotheses: ', hypotheses)
        #print('keras.backend.argmin(hypotheses): ', keras.backend.argmin(hypotheses))
        #print('keras.backend.gather_nd(hypotheses, keras.backend.argmin(hypotheses)): ', backend.gather_nd(hypotheses, keras.backend.argmin(hypotheses)))
        #lowest_loss = backend.gather_nd(hypotheses, keras.backend.argmin(hypotheses))

        return hypotheses

    return _sym_orth_l1
    '''

