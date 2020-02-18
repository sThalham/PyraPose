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
from keras_efficientnets import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4

from . import retinanet
from . import Backbone
from ..utils.image import preprocess_image


class ResNetBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def __init__(self, backbone):
        super(ResNetBackbone, self).__init__(backbone)
        #self.custom_objects.update(keras.models)

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
            inputs_0 = keras.layers.Input(shape=(480, 640, 3))
            inputs_1 = keras.layers.Input(shape=(480, 640, 3))
            #inputs_0 = keras.layers.Input(shape=(None, None, 3))
            #inputs_1 = keras.layers.Input(shape=(None, None, 3))
        #inputs = keras.layers.Concatenate()([inputs_0, inputs_1])

    #resnet_rgb = keras_resnet.models.ResNet34(inputs_0, include_top=False, freeze_bn=True)
    #resnet_dep = keras_resnet.models.ResNet34(inputs_1, include_top=False, freeze_bn=True)

    resnet_rgb = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_tensor=inputs_0, pooling=None, classes=num_classes)
    resnet_dep = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_tensor=inputs_1, pooling=None, classes=num_classes)

    for i, layer in enumerate(resnet_rgb.layers):
        print(i, layer.name)
        if i < 39:
            layer.trainable = False
        layer.name = 'layer_' + str(i)


    # <39 is first 2 layers... 11 layer, freeze either <22 or <25
    # resnet18

    #print(effnet_rgb.summary())
    #[<tf.Tensor 'res3d_relu/Relu:0' shape=(?, ?, ?, 512) dtype=float32>, <tf.Tensor 'res4f_relu/Relu:0' shape=(?, ?, ?, 1024) dtype=float32>, <tf.Tensor 'res5c_relu/Relu:0' shape=(?, ?, ?, 2048) dtype=float32>]
    # [B1:236(swish_98) B1/B2:338(swish_138), B3:383(swish_156)] separate
    # B3: swish 138:294
    # B3: 338, 264/235, 190/146

    # invoke modifier if given
    if modifier:
        resnet_rgb = modifier(resnet_rgb)
        resnet_dep = modifier(resnet_dep)

    layer_names = [80, 142, 174]  # resnet50
    #layer_names = [72, 128, 157]  # resnet34
    #layer_names = [45, 65, 85]  # resnet18
    layer_outputs_rgb = [resnet_rgb.layers[idx].output for idx in layer_names]
    layer_outputs_dep = [resnet_dep.layers[idx].output for idx in layer_names]

    return retinanet.retinanet(inputs=[inputs_0, inputs_1], num_classes=num_classes,
                               backbone_layers_rgb=layer_outputs_rgb, backbone_layers_dep=layer_outputs_dep, **kwargs)

def resnet50_retinanet(num_classes, inputs=None, **kwargs):
    return resnet_retinanet(num_classes=num_classes, backbone='resnet50', inputs=inputs, **kwargs)

