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

import numpy as np
import random
import warnings
import copy
import cv2
import time

import tensorflow.keras as keras
#import keras

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
    apply_transform2depth,
    adjust_pose_annotation,
    apply_transform,
    preprocess_image,
    resize_image,
)
from ..utils.transform import transform_aabb


def gen(sr=8000, seconds=3, batch_size=16, shuffle=True):
    dim = sr * seconds

    def loadFile(file):
        wav, _ = librosa.load(file, sr=sr, mono=True)
        if len(wav) > dim:
            return wav[:dim]
        return np.pad(wav, (0, dim - len(wav)), 'constant', constant_values=0)

    while True:
        indexs = np.arange(len(df))
        if shuffle:
            np.random.shuffle(indexs)

        for x in range(len(df) // batch_size):
            X, y = [], []
            for i in indexs[np.arange(x * batch_size, (x + 1) * batch_size)]:
                X.append(librosa.feature.mfcc(loadFile(df.filepath[i]), sr).T)
                y.append(df.label[i])

            yield tf.convert_to_tensor(X), to_categorical(y, num_classes=2)