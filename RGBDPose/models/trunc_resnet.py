

import keras
from itertools import chain

from . import retinanet
from . import Backbone
from ..utils.image import preprocess_image


class TruNetBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def __init__(self, backbone):
        super(TruNetBackbone, self).__init__(backbone)

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return resnet_retinanet(*args, **kwargs)

    def download_imagenet(self):

        pass

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['trunet']
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

    resnet_rgb = keras.applications.resnet.ResNet50(include_top=False, weights="imagenet", input_tensor=inputs_0, classes=num_classes)
    resnet_dep = keras.applications.resnet.ResNet50(include_top=False, weights="imagenet", input_tensor=inputs_1, classes=num_classes)

    '''
    keep_layers = list(range(1,29)) + list(range(39, 61)) + list(range(81, 103)) + list(range(143, 165))
    # layer17 list,
    #keep_inputs = list(range(0, 12)) + [6, 12, 13, 14, [15, 16]] + list(range(17, 26)) + [[18, 26], 27,
    #               28] + list(range(39, 44)) + [28, 44, 45, 46, [47, 48]] + list(range(49, 58)) + [[50, 58], 59,
    #               60] + list(range(81, 86)) + [60, 86, 87, 88, [89, 90]] + list(range(91, 100)) + [[92, 100], 101,
    #               102] + list(range(143, 148)) + [102, 148, 149, 150, [151, 152]] + list(range(153, 162)) + [[154, 162], 163]
    # 0-26 stage 1+2, 39-60 stage 3,
    keep_inputs = list(range(0, 11)) + [5, 11, 12, 13, [14, 15]] + list(range(16, 25)) + [[17, 25], 26,
                    27] + list(range(28, 33)) + [27, 33, 34, 35, [36, 37]] + list(range(38, 47)) + [[39, 47], 48,
                    49] + list(range(50, 55)) + [49, 55, 56, 57, [58, 59]] + list(range(60, 69)) + [[61, 69], 70,
                    71] + list(range(72, 77)) + [71, 77, 78, 79, [80, 81]] + list(range(82, 91)) + [[83, 91], 92]
    '''

    rgb_outs = []
    input_rgb = resnet_rgb.input
    # stage 1
    rgb1_pad = resnet_rgb.layers[1](input_rgb)
    rgb1_conv = resnet_rgb.layers[2](rgb1_pad) # conv 1
    rgb1_bn = resnet_rgb.layers[3](rgb1_conv)
    rgb1_relu = resnet_rgb.layers[4](rgb1_bn)
    rgb1_pool1_pad = resnet_rgb.layers[5](rgb1_relu)
    rgb1_pool1_pool = resnet_rgb.layers[6](rgb1_pool1_pad)
    # stage 2
    rgb2_b1_1_conv = resnet_rgb.layers[7](rgb1_pool1_pool)
    rgb2_b1_1_bn = resnet_rgb.layers[8](rgb2_b1_1_conv)
    rgb2_b1_1_relu = resnet_rgb.layers[9](rgb2_b1_1_bn)
    rgb2_b1_2_conv = resnet_rgb.layers[10](rgb2_b1_1_relu)
    rgb2_b1_2_bn = resnet_rgb.layers[11](rgb2_b1_2_conv)
    rgb2_b1_2_relu = resnet_rgb.layers[12](rgb2_b1_2_bn)
    rgb2_b1_0_conv = resnet_rgb.layers[13](rgb1_pool1_pool)
    rgb2_b1_3_conv = resnet_rgb.layers[14](rgb2_b1_2_relu)
    rgb2_b1_0_bn = resnet_rgb.layers[15](rgb2_b1_0_conv)
    rgb2_b1_3_bn = resnet_rgb.layers[16](rgb2_b1_3_conv)
    rgb2_b1_add = resnet_rgb.layers[17]([rgb2_b1_0_bn, rgb2_b1_3_bn])
    rgb2_b1_out = resnet_rgb.layers[18](rgb2_b1_add)
    rgb2_b2_1_conv = resnet_rgb.layers[19](rgb2_b1_out)
    rgb2_b2_1_bn = resnet_rgb.layers[20](rgb2_b2_1_conv)
    rgb2_b2_1_relu = resnet_rgb.layers[21](rgb2_b2_1_bn)
    rgb2_b2_2_conv = resnet_rgb.layers[22](rgb2_b2_1_relu)
    rgb2_b2_2_bn = resnet_rgb.layers[23](rgb2_b2_2_conv)
    rgb2_b2_2_relu = resnet_rgb.layers[24](rgb2_b2_2_bn)
    rgb2_b2_3_conv = resnet_rgb.layers[25](rgb2_b2_2_relu)
    rgb2_b2_3_bn = resnet_rgb.layers[26](rgb2_b2_3_conv)
    rgb2_b2_add = resnet_rgb.layers[27]([rgb2_b1_out, rgb2_b2_3_bn])
    rgb2_b2_out = resnet_rgb.layers[28](rgb2_b2_add)
    # stage 3
    rgb3_b1_1_conv = resnet_rgb.layers[39](rgb2_b2_out)
    rgb3_b1_1_bn = resnet_rgb.layers[40](rgb3_b1_1_conv)
    rgb3_b1_1_relu = resnet_rgb.layers[41](rgb3_b1_1_bn)
    rgb3_b1_2_conv = resnet_rgb.layers[42](rgb3_b1_1_relu)
    rgb3_b1_2_bn = resnet_rgb.layers[43](rgb3_b1_2_conv)
    rgb3_b1_2_relu = resnet_rgb.layers[44](rgb3_b1_2_bn)
    rgb3_b1_0_conv = resnet_rgb.layers[45](rgb2_b2_out)
    rgb3_b1_3_conv = resnet_rgb.layers[46](rgb3_b1_2_relu)
    rgb3_b1_0_bn = resnet_rgb.layers[47](rgb3_b1_0_conv)
    rgb3_b1_3_bn = resnet_rgb.layers[48](rgb3_b1_3_conv)
    rgb3_b1_add = resnet_rgb.layers[49]([rgb3_b1_0_bn, rgb3_b1_3_bn])
    rgb3_b1_out = resnet_rgb.layers[50](rgb3_b1_add)
    rgb_outs.append(rgb3_b1_out)
    # stage 4
    rgb4_b1_1_conv = resnet_rgb.layers[81](rgb3_b1_out)
    rgb4_b1_1_bn = resnet_rgb.layers[82](rgb4_b1_1_conv)
    rgb4_b1_1_relu = resnet_rgb.layers[83](rgb4_b1_1_bn)
    rgb4_b1_2_conv = resnet_rgb.layers[84](rgb4_b1_1_relu)
    rgb4_b1_2_bn = resnet_rgb.layers[85](rgb4_b1_2_conv)
    rgb4_b1_2_relu = resnet_rgb.layers[86](rgb4_b1_2_bn)
    rgb4_b1_0_conv = resnet_rgb.layers[87](rgb3_b1_out)
    rgb4_b1_3_conv = resnet_rgb.layers[88](rgb4_b1_2_relu)
    rgb4_b1_0_bn = resnet_rgb.layers[89](rgb4_b1_0_conv)
    rgb4_b1_3_bn = resnet_rgb.layers[90](rgb4_b1_3_conv)
    rgb4_b1_add = resnet_rgb.layers[91]([rgb4_b1_0_bn, rgb4_b1_3_bn])
    rgb4_b1_out = resnet_rgb.layers[92](rgb4_b1_add)
    rgb_outs.append(rgb4_b1_out)
    # stage 5
    rgb5_b1_1_conv = resnet_rgb.layers[143](rgb4_b1_out)
    rgb5_b1_1_bn = resnet_rgb.layers[144](rgb5_b1_1_conv)
    rgb5_b1_1_relu = resnet_rgb.layers[145](rgb5_b1_1_bn)
    rgb5_b1_2_conv = resnet_rgb.layers[146](rgb5_b1_1_relu)
    rgb5_b1_2_bn = resnet_rgb.layers[147](rgb5_b1_2_conv)
    rgb5_b1_2_relu = resnet_rgb.layers[148](rgb5_b1_2_bn)
    rgb5_b1_0_conv = resnet_rgb.layers[149](rgb4_b1_out)
    rgb5_b1_3_conv = resnet_rgb.layers[150](rgb5_b1_2_relu)
    rgb5_b1_0_bn = resnet_rgb.layers[151](rgb5_b1_0_conv)
    rgb5_b1_3_bn = resnet_rgb.layers[152](rgb5_b1_3_conv)
    rgb5_b1_add = resnet_rgb.layers[153]([rgb5_b1_0_bn, rgb5_b1_3_bn])
    rgb5_out = resnet_rgb.layers[154](rgb5_b1_add)
    rgb_outs.append(rgb5_out)
    resnet_rgb = keras.models.Model(inputs=input_rgb, outputs=rgb5_out)

    dep_outs = []
    input_dep = resnet_dep.input
    # stage 1
    dep1_pad = resnet_dep.layers[1](input_dep)
    dep1_conv = resnet_dep.layers[2](dep1_pad)  # conv 1
    dep1_bn = resnet_dep.layers[3](dep1_conv)
    dep1_relu = resnet_dep.layers[4](dep1_bn)
    dep1_pool1_pad = resnet_dep.layers[5](dep1_relu)
    dep1_pool1_pool = resnet_dep.layers[6](dep1_pool1_pad)
    # stage 2
    dep2_b1_1_conv = resnet_dep.layers[7](dep1_pool1_pool)
    dep2_b1_1_bn = resnet_dep.layers[8](dep2_b1_1_conv)
    dep2_b1_1_relu = resnet_dep.layers[9](dep2_b1_1_bn)
    dep2_b1_2_conv = resnet_dep.layers[10](dep2_b1_1_relu)
    dep2_b1_2_bn = resnet_dep.layers[11](dep2_b1_2_conv)
    dep2_b1_2_relu = resnet_dep.layers[12](dep2_b1_2_bn)
    dep2_b1_0_conv = resnet_dep.layers[13](dep1_pool1_pool)
    dep2_b1_3_conv = resnet_dep.layers[14](dep2_b1_2_relu)
    dep2_b1_0_bn = resnet_dep.layers[15](dep2_b1_0_conv)
    dep2_b1_3_bn = resnet_dep.layers[16](dep2_b1_3_conv)
    dep2_b1_add = resnet_dep.layers[17]([dep2_b1_0_bn, dep2_b1_3_bn])
    dep2_b1_out = resnet_dep.layers[18](dep2_b1_add)
    dep2_b2_1_conv = resnet_dep.layers[19](dep2_b1_out)
    dep2_b2_1_bn = resnet_dep.layers[20](dep2_b2_1_conv)
    dep2_b2_1_relu = resnet_dep.layers[21](dep2_b2_1_bn)
    dep2_b2_2_conv = resnet_dep.layers[22](dep2_b2_1_relu)
    dep2_b2_2_bn = resnet_dep.layers[23](dep2_b2_2_conv)
    dep2_b2_2_relu = resnet_dep.layers[24](dep2_b2_2_bn)
    dep2_b2_3_conv = resnet_dep.layers[25](dep2_b2_2_relu)
    dep2_b2_3_bn = resnet_dep.layers[26](dep2_b2_3_conv)
    dep2_b2_add = resnet_dep.layers[27]([dep2_b1_out, dep2_b2_3_bn])
    dep2_b2_out = resnet_dep.layers[28](dep2_b2_add)
    # stage 3
    dep3_b1_1_conv = resnet_dep.layers[39](dep2_b2_out)
    dep3_b1_1_bn = resnet_dep.layers[40](dep3_b1_1_conv)
    dep3_b1_1_relu = resnet_dep.layers[41](dep3_b1_1_bn)
    dep3_b1_2_conv = resnet_dep.layers[42](dep3_b1_1_relu)
    dep3_b1_2_bn = resnet_dep.layers[43](dep3_b1_2_conv)
    dep3_b1_2_relu = resnet_dep.layers[44](dep3_b1_2_bn)
    dep3_b1_0_conv = resnet_dep.layers[45](dep2_b2_out)
    dep3_b1_3_conv = resnet_dep.layers[46](dep3_b1_2_relu)
    dep3_b1_0_bn = resnet_dep.layers[47](dep3_b1_0_conv)
    dep3_b1_3_bn = resnet_dep.layers[48](dep3_b1_3_conv)
    dep3_b1_add = resnet_dep.layers[49]([dep3_b1_0_bn, dep3_b1_3_bn])
    dep3_b1_out = resnet_dep.layers[50](dep3_b1_add)
    dep_outs.append(dep3_b1_out)
    # stage 4
    dep4_b1_1_conv = resnet_dep.layers[81](dep3_b1_out)
    dep4_b1_1_bn = resnet_dep.layers[82](dep4_b1_1_conv)
    dep4_b1_1_relu = resnet_dep.layers[83](dep4_b1_1_bn)
    dep4_b1_2_conv = resnet_dep.layers[84](dep4_b1_1_relu)
    dep4_b1_2_bn = resnet_dep.layers[85](dep4_b1_2_conv)
    dep4_b1_2_relu = resnet_dep.layers[86](dep4_b1_2_bn)
    dep4_b1_0_conv = resnet_dep.layers[87](dep3_b1_out)
    dep4_b1_3_conv = resnet_dep.layers[88](dep4_b1_2_relu)
    dep4_b1_0_bn = resnet_dep.layers[89](dep4_b1_0_conv)
    dep4_b1_3_bn = resnet_dep.layers[90](dep4_b1_3_conv)
    dep4_b1_add = resnet_dep.layers[91]([dep4_b1_0_bn, dep4_b1_3_bn])
    dep4_b1_out = resnet_dep.layers[92](dep4_b1_add)
    dep_outs.append(dep4_b1_out)
    # stage 5
    dep5_b1_1_conv = resnet_dep.layers[143](dep4_b1_out)
    dep5_b1_1_bn = resnet_dep.layers[144](dep5_b1_1_conv)
    dep5_b1_1_relu = resnet_dep.layers[145](dep5_b1_1_bn)
    dep5_b1_2_conv = resnet_dep.layers[146](dep5_b1_1_relu)
    dep5_b1_2_bn = resnet_dep.layers[147](dep5_b1_2_conv)
    dep5_b1_2_relu = resnet_dep.layers[148](dep5_b1_2_bn)
    dep5_b1_0_conv = resnet_dep.layers[149](dep4_b1_out)
    dep5_b1_3_conv = resnet_dep.layers[150](dep5_b1_2_relu)
    dep5_b1_0_bn = resnet_dep.layers[151](dep5_b1_0_conv)
    dep5_b1_3_bn = resnet_dep.layers[152](dep5_b1_3_conv)
    dep5_b1_add = resnet_dep.layers[153]([dep5_b1_0_bn, dep5_b1_3_bn])
    dep5_out = resnet_dep.layers[154](dep5_b1_add)
    dep_outs.append(dep5_out)
    resnet_dep = keras.models.Model(inputs=input_dep, outputs=dep5_out)

    for i, layer in enumerate(resnet_rgb.layers):
        #if i < 19 or 'bn' in layer.name: # 19 = 5 conv layers, 29 = 8 conv layers
        if i < 19 and 'bn' not in layer.name:
            layer.trainable=False

    for i, layer in enumerate(resnet_dep.layers):
        #if 'bn' in layer.name:
        #    layer.trainable = False
        layer.name = 'layer_' + str(i)

        # invoke modifier if given
    if modifier:
        resnet_rgb = modifier(resnet_rgb)
        resnet_dep = modifier(resnet_dep)

    return retinanet.retinanet(inputs=[inputs_0, inputs_1], num_classes=num_classes, backbone_layers_rgb=rgb_outs, backbone_layers_dep=dep_outs, **kwargs)


