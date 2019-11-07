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
import tensorflow as tf
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
            #inputs = keras.layers.Input(shape=(3, None, None))
            inputs_0 = keras.layers.Input(shape=(3, None, None))
            inputs_1 = keras.layers.Input(shape=(3, None, None))
        else:
            #inputs = keras.layers.Input(shape=(None, None, 3))
            inputs_0 = keras.layers.Input(shape=(None, None, 3))
            inputs_1 = keras.layers.Input(shape=(None, None, 3))
        #inputs = keras.layers.Concatenate()([inputs_0, inputs_1])

    resnet_rgb = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_tensor=inputs_0, pooling=None, classes=num_classes)
    resnet_dep = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_tensor=inputs_1, pooling=None, classes=num_classes)

    for i, layer in enumerate(resnet_dep.layers):
        layer.name = 'layer_' + str(i)

    #[<tf.Tensor 'res3d_relu/Relu:0' shape=(?, ?, ?, 512) dtype=float32>, <tf.Tensor 'res4f_relu/Relu:0' shape=(?, ?, ?, 1024) dtype=float32>, <tf.Tensor 'res5c_relu/Relu:0' shape=(?, ?, ?, 2048) dtype=float32>]

    # invoke modifier if given
    if modifier:
        resnet_rgb = modifier(resnet_rgb)
        resnet_dep = modifier(resnet_dep)

    #outputs_rgb = [resnet_rgb.layers[80].output, resnet_rgb.layers[142].output, resnet_rgb.layers[174].output]
    #outputs_dep = [resnet_dep.layers[80].output, resnet_dep.layers[142].output, resnet_dep.layers[174].output]
    #outputs_rgb = tf.convert_to_tensor(outputs_rgb)
    #outputs_dep = tf.convert_to_tensor(outputs_dep)
    #print(outputs_rgb)

    #return retinanet.retinanet(inputs=[inputs_0, inputs_1], num_classes=num_classes, backbone_layers_1=outputs_rgb, backbone_layers_2=outputs_dep, **kwargs)
    return retinanet.retinanet(inputs=[inputs_0, inputs_1], num_classes=num_classes, b1=resnet_rgb.layers[80].output, b2=resnet_rgb.layers[142].output, b3=resnet_rgb.layers[174].output,
                               b4=resnet_dep.layers[80].output, b5=resnet_dep.layers[142].output, b6=resnet_dep.layers[174].output, **kwargs)


def resnet50_retinanet(num_classes, inputs=None, **kwargs):
    return resnet_retinanet(num_classes=num_classes, backbone='resnet50', inputs=inputs, **kwargs)

