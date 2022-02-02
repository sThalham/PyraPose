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

from . import retinanet
from . import Backbone
from ..utils.image import preprocess_image


class ResNetBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def __init__(self, backbone):
        super(ResNetBackbone, self).__init__(backbone)
        from .. import layers
        #self.custom_objects.update({'DenormRegression': layers.DenormRegression()})

    def model(self, *args, **kwargs):
        """ Returns PyraPose using the correct backbone.
        """
        return resnet_model(*args, **kwargs)

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='caffe')


def resnet_model(num_classes, inputs=None, modifier=None, **kwargs):

    if inputs is None:
        if keras.backend.image_data_format() == 'channels_first':
            inputs = keras.layers.Input(shape=(3, None, None))
        else:
            inputs = keras.layers.Input(shape=(540, 960, 3))
            #inputs = keras.layers.Input(shape=(1080, 1920, 3))

    resnet = tf.keras.applications.ResNet50(
        include_top=False, weights='imagenet', input_tensor=inputs, classes=num_classes)

    for i, layer in enumerate(resnet.layers):
        if i < 39 or 'bn' in layer.name:  # freezing first 2 stages
        #if i < 81 or 'bn' in layer.name:  # freezing first 2 stages
            layer.trainable = False
        #layer.trainable = False
        #print(i, layer.name, layer)

    #resnet = replace_relu_with_swish(resnet)

        # invoke modifier if given
    if modifier:
        resnet = modifier(resnet)

    resnet_outputs = [resnet.layers[80].output, resnet.layers[142].output, resnet.layers[174].output]

    return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=resnet_outputs, **kwargs)

