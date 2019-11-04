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

import keras
from keras.utils import get_file
import keras_resnet
import keras_resnet.models

from . import retinanet
from . import Backbone
from ..utils.image import preprocess_image


class ResNetBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def __init__(self, backbone):
        super(ResNetBackbone, self).__init__(backbone)
        self.custom_objects.update(keras_resnet.custom_objects)

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return resnet_retinanet(*args, **kwargs)

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['resnet50', 'resnet101', 'resnet152']
        backbone = self.backbone.split('_')[0]

        if backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='caffe')


def resnet_retinanet(num_classes, inputs=None, modifier=None, **kwargs):

    if inputs is None:
        if keras.backend.image_data_format() == 'channels_first':
            inputs = keras.layers.Input(shape=(3, None, None))
        else:
            inputs = keras.layers.Input(shape=(None, None, 3))

    resnet = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_tensor=inputs, pooling=None, classes=num_classes)
    #resnet = keras_resnet.models.ResNet50(inputs, include_top=False, freeze_bn=True)

    # invoke modifier if given
    if modifier:
        resnet = modifier(resnet)

    return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=resnet.outputs[1:], **kwargs)


def resnet50_retinanet(num_classes, inputs=None, **kwargs):
    return resnet_retinanet(num_classes=num_classes, backbone='resnet50', inputs=inputs, **kwargs)

