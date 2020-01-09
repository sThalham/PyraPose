import keras
import tensorflow as tf
from .. import initializers
from .. import layers
from ..utils.anchors import AnchorParameters
from . import assert_training_model


def swish(x, beta=1.0):
    return x * keras.activations.sigmoid(beta*x)


#class batch_norm(keras.layers.layers):

#    def call(self, inputs, **kwargs):

#        return keras.layers.BatchNormalization(axis=-1)(inputs)

#    def compute_output_shape(self, input_shape):
#        return input_shape


def max_norm(w):
    norms = K.sqrt(K.sum(K.square(w), keepdims=True))
    desired = K.clip(norms, 0, self.max_value)
    w *= (desired / (K.epsilon() + norms))
    return w


def default_classification_model(
    num_classes,
    num_anchors,
    pyramid_feature_size=256,
    prior_probability=0.01,
    classification_feature_size=256,
    name='classification_submodel'
):
    options = {
        'kernel_size' : 3,
        'strides'     : 1,
        'padding'     : 'same',
    }

    if keras.backend.image_data_format() == 'channels_first':
        inputs  = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))

    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
        #outputs = keras.layers.SeparableConv2D(
            filters=classification_feature_size,
            activation='relu',
            #name='pyramid_classification_{}'.format(i),
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(
        filters=num_classes * num_anchors,
        kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer=initializers.PriorProbability(probability=prior_probability),
        #name='pyramid_classification',
        **options
    )(outputs)

    # reshape output and apply sigmoid
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1))(outputs) #, name='pyramid_classification_permute'
    outputs = keras.layers.Reshape((-1, num_classes))(outputs) # , name='pyramid_classification_reshape'
    outputs = keras.layers.Activation('sigmoid')(outputs) # , name='pyramid_classification_sigmoid'

    return keras.models.Model(inputs=inputs, outputs=outputs) #, name=name)


def default_regression_model(num_values, num_anchors, pyramid_feature_size=256, regression_feature_size=256, name='regression_submodel'):
    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_initializer' : keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer'   : 'zeros'
    }

    if keras.backend.image_data_format() == 'channels_first':
        inputs  = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))

    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
        #outputs = keras.layers.SeparableConv2D(
            filters=regression_feature_size,
            activation='relu',
            #name='pyramid_regression_{}'.format(i),
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(num_anchors * num_values, **options)(outputs) #, name='pyramid_regression'
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1))(outputs) # , name='pyramid_regression_permute'
    outputs = keras.layers.Reshape((-1, num_values))(outputs) # , name='pyramid_regression_reshape'

    return keras.models.Model(inputs=inputs, outputs=outputs) #, name=name)


def default_3Dregression_model(num_values, num_anchors, pyramid_feature_size=256, regression_feature_size=256 , name='3Dregression_submodel'):
    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_initializer' : keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer'   : 'zeros',
        'kernel_regularizer' : keras.regularizers.l2(0.001),
    }

    if keras.backend.image_data_format() == 'channels_first':
        inputs  = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))

    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
        #outputs = keras.layers.SeparableConv2D(
            filters=regression_feature_size,
            activation='relu',
            #name='pyramid_regression3D_{}'.format(i),
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(num_anchors * num_values, **options)(outputs) #, name='pyramid_regression3D'
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1))(outputs) # , name='pyramid_regression3D_permute'
    outputs = keras.layers.Reshape((-1, num_values))(outputs) # , name='pyramid_regression3D_reshape'

    return keras.models.Model(inputs=inputs, outputs=outputs) #, name=name)


def __create_pyramid_features(C3, C4, C5, feature_size=256):
    P5 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C5)
    P5_upsampled = layers.UpsampleLike()([P5, C4])
    P5 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5_con')(P5)

    # add P5 elementwise to C4
    P4 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C4)
    P4 = keras.layers.Add()([P5_upsampled, P4])
    P4_upsampled = layers.UpsampleLike()([P4, C3])
    P4 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4_con')(P4)

    # add P4 elementwise to C3
    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C3)
    P3 = keras.layers.Add()([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3_con')(P3)

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6_con')(C5)

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    P7 = keras.layers.Activation('relu')(P6)
    P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7_con')(P7)

    return [P3, P4, P5, P6, P7]


def __create_FPN(C3, C4, C5, feature_size=256):
    P5 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C5)
    P5_upsampled = layers.UpsampleLike()([P5, C4])
    P5 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5_con')(P5)

    # add P5 elementwise to C4
    P4 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C4)
    P4 = keras.layers.Add()([P5_upsampled, P4])
    P4_upsampled = layers.UpsampleLike()([P4, C3])
    P4 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4_con')(P4)

    # add P4 elementwise to C3
    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C3)
    P3 = keras.layers.Add()([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3_con')(P3)

    return [P3, P4, P5]


def __create_PANet(C3, C4, C5, feature_size=256):
    P5 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C5)
    P5_upsampled = layers.UpsampleLike()([P5, C4])
    P5 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5_con')(P5)

    # add P5 elementwise to C4
    P4 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C4)
    P4 = keras.layers.Add()([P5_upsampled, P4])
    P4_upsampled = layers.UpsampleLike()([P4, C3])
    P4 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4_con')(P4)

    # add P4 elementwise to C3
    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C3)
    P3 = keras.layers.Add()([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3_con')(P3)

    # Top-down here (PANet-style)
    P3_down = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same')(P3)
    P4 = keras.layers.Add()([P3_down, P4])
    P4 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same')(P4)

    # Top-down here (PANet-style)
    P4_down = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same')(P4)
    P5 = keras.layers.Add()([P4_down, P5])
    P5 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same')(P5)

    return [P3, P4, P5]


def __create_BiFPN(C3_R, C4_R, C5_R, C3_D, C4_D, C5_D, feature_size=256):
    P3_r = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C3_R)
    P4_r = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C4_R)
    P5_r = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C5_R)

    P3_d = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C3_D)
    P4_d = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C4_D)
    P5_d = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C5_D)

    P3 = keras.layers.Add()([P3_r, P3_d])
    P4 = keras.layers.Add()([P4_r, P4_d])
    P5 = keras.layers.Add()([P5_r, P5_d])

    P5_upsampled = layers.UpsampleLike()([P5, C4_R])
    P4_mid = keras.layers.Add()([P5_upsampled, P4])
    P4_mid = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same')(P4_mid)
    P4_upsampled = layers.UpsampleLike()([P4_mid, C3_R])
    P3 = keras.layers.Add()([P4_upsampled, P3])

    P3_down = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same')(P3)
    P4 = keras.layers.Add()([P3_down, P4_mid, P4])
    P4 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same')(P4)

    P4_down = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same')(P4)
    P5 = keras.layers.Add()([P4_down, P5])
    P5 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same')(P5)

    return [P3, P4, P5]


def __create_sparceFPN(C3_R, C4_R, C5_R, C3_D, C4_D, C5_D, feature_size=256):
    # FPN-Fusion test 1
    P3_r = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C3_R)
    P4_r = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C4_R)
    P5_r = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C5_R)

    P3_d = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C3_D)
    P4_d = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C4_D)
    P5_d = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C5_D)

    P3 = keras.layers.Add()([P3_r, P3_d])
    P4 = keras.layers.Add()([P4_r, P4_d])
    P5 = keras.layers.Add()([P5_r, P5_d])

    # uncomment for FPN-Fusion test 2
    #P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same')(P3)
    #P4 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same')(P4)
    #P5 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same')(P5)

    # only from here for FPN-fusion test 3
    #C3 = keras.layers.Add()([C3_R, C3_D])
    #C4 = keras.layers.Add()([C4_R, C4_D])
    #C5 = keras.layers.Add()([C5_R, C5_D])

    #P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C3)
    #P4 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C4)
    #P5 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C5)

    # 3x3 conv for test 4
    #P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same')(C3)
    #P4 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same')(C4)
    #P5 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same')(C5)

    P5_upsampled = layers.UpsampleLike()([P5, C4_R])
    P4_upsampled = layers.UpsampleLike()([P4, C3_R])
    P4_mid = keras.layers.Add()([P5_upsampled, P4])
    P4_mid = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same')(P4_mid)    # replace with depthwise and 3x1+1x3
    P3_mid = keras.layers.Add()([P4_upsampled, P3])
    P3_mid = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same')(P3_mid)    # replace with depthwise and 3x1+1x3
    P3_down = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same')(P3_mid)
    P3_fin = keras.layers.Add()([P3_mid, P3])  # skip connection
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3_fin) # replace with depthwise and 3x1+1x3

    P4_fin = keras.layers.Add()([P3_down, P4_mid])
    P4_down = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same')(P4_mid)
    P4_fin = keras.layers.Add()([P4_fin, P4])  # skip connection
    P4 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4_fin) # replace with depthwise and 3x1+1x3

    P5_fin = keras.layers.Add()([P4_down, P5])
    P5 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5_fin) # replace with depthwise and 3x1+1x3

    return [P3, P4, P5]


def default_submodels(num_classes, num_anchors):
    return [
        ('bbox', default_regression_model(4, num_anchors)),
        ('3Dbox', default_3Dregression_model(16, num_anchors)),
        ('cls', default_classification_model(num_classes, num_anchors))
    ]


def default_submodels_2(num_classes, num_anchors):
    return [
        ('bbox', default_regression_model(4, num_anchors)),
        ('3Dbox', default_3Dregression_model(16, num_anchors)),
        ('cls', default_classification_model(num_classes, num_anchors)),
        ('bbox_dep', default_regression_model(4, num_anchors)),
        ('3Dbox_dep', default_3Dregression_model(16, num_anchors)),
        ('cls_dep', default_classification_model(num_classes, num_anchors))
    ]


def __build_fusion_pyramid(name_rgb, model_rgb, name_dep, model_dep, features_rgb, features_dep, num_anchors):
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer': 'zeros',
    }

    outputs = []

    for idx, feat1 in enumerate(features_rgb):
        feat2 = features_dep[idx]
        out_rgb = model_rgb(feat1)
        out_dep = model_dep(feat2)
        if name_rgb == 'bbox':
            num_values = 4
        if name_rgb == '3Dbox':
            num_values = 16
        if name_rgb == 'cls':
            num_values = 13

        # fuse before output calculation
        #features = keras.layers.Concatenate()([feat1, feat2])
        #num_features = 256
        #features = keras.layers.Conv2D(num_features, **options)(features)
        #outputs_head = keras.layers.Concatenate(axis=-1)([out_rgb, out_dep, features])
        #outputs_head = keras.layers.Conv2D(num_anchors * num_values, **options)(outputs_head)

        # fuse outputs directly
        features = keras.layers.Concatenate()([feat1, feat2])
        num_features = keras.backend.int_shape(features)[-1]
        num_all = keras.backend.int_shape(out_rgb)[-1]#
        features = keras.layers.Reshape((-1, num_features))(features)
        out_rgb = keras.layers.Reshape((-1, num_all))(out_rgb)
        out_dep = keras.layers.Reshape((-1, num_all))(out_dep)
        outputs_head = keras.layers.Concatenate(axis=-1)([out_rgb, out_dep, features])
        outputs_head = keras.layers.Dense(256)(outputs_head)
        outputs_head = keras.layers.Dense(num_anchors * num_values)(outputs_head)

        if keras.backend.image_data_format() == 'channels_first':
            outputs_head = keras.layers.Permute((2, 3, 1))(outputs_head)
        outputs_head = keras.layers.Reshape((-1, num_values), name='submodel' + name_rgb + str(idx))(outputs_head)
        if name_rgb == 'cls':
            outputs_head = keras.layers.Activation('sigmoid')(outputs_head)  # , name='pyramid_classification_sigmoid'
            #outputs_head = keras.layers.Activation('softmax')(outputs_head)

        outputs.append(outputs_head)

    return keras.layers.Concatenate(axis=1, name=name_rgb)(outputs)


def __fuse_pyramid(models, features1, features2, num_anchors):
    outputs = []

    for idx, m_rgb in enumerate(models[:3]):
        m_dep = models[3:][idx]
        fused_models = __build_fusion_pyramid(m_rgb[0], m_rgb[1], m_dep[0], m_dep[1], features1, features2, num_anchors)
        outputs.append(fused_models)

    return outputs


def __build_model_pyramid(name, model, features):
    return keras.layers.Concatenate(axis=1, name=name)([model(f) for f in features])


def __build_pyramid(models, features):
    return [__build_model_pyramid(n, m, features) for n, m in models]


def __build_pyramid_duo(models, features1, features2):
    outs = []
    for n, m in models[:3]:
        outs.append(__build_model_pyramid(n, m, features1))
    for n, m in models[3:]:
        outs.append(__build_model_pyramid(n, m, features2))

    return outs


def __build_anchors(anchor_parameters, features):
    anchors = [
        layers.Anchors(
            size=anchor_parameters.sizes[i],
            stride=anchor_parameters.strides[i],
            ratios=anchor_parameters.ratios,
            scales=anchor_parameters.scales,
            name='anchors_{}'.format(i)
        )(f) for i, f in enumerate(features)
    ]

    return keras.layers.Concatenate(axis=1, name='anchors')(anchors)


def output_fusion_model(pyramids, num_anchors):

    pyramids1 = pyramids[:3]
    pyramids2 = pyramids[3:]
    output_list = []

    for idx, pyra1 in enumerate(pyramids1):
        pyra2 = pyramids2[idx]
        num_values = keras.backend.int_shape(pyra2)[-1]

        outputs_head = keras.layers.Concatenate()([pyra1, pyra2])
        outputs_head = keras.layers.Dense(num_anchors * num_values)(outputs_head)

        if keras.backend.image_data_format() == 'channels_first':
            outputs_head = keras.layers.Permute((2, 3, 1))(outputs_head)
        if idx == 0:
            re_name = 'bbox'
        elif idx == 1:
            re_name = '3Dbox'
        elif idx == 2:
            re_name = 'cls'
        outputs_head = keras.layers.Reshape((-1, num_values), name=re_name)(outputs_head)

        output_list.append(outputs_head)

    return output_list


def default_fusion_model(pyramids, features, num_anchors, intermediate_feature_size=512):

    pyramid_feature_size = keras.backend.int_shape(features)[-1]

    outputs = features
    pyramids1 = pyramids[:3]
    pyramids2 = pyramids[3:]

    output_list = []

    for idx, pyra1 in enumerate(pyramids1):
        pyra2 = pyramids2[idx]
        num_values = keras.backend.int_shape(pyra2)[-1]

        outputs_head = keras.layers.Reshape((-1, pyramid_feature_size))(outputs)
        outputs_head = keras.layers.Concatenate()([outputs_head, pyra1, pyra2])
        outputs_head = keras.layers.Dense(num_anchors * num_values)(outputs_head)

        if keras.backend.image_data_format() == 'channels_first':
            outputs_head = keras.layers.Permute((2, 3, 1))(outputs_head)
        if idx == 0:
            re_name = 'bbox'
        elif idx == 1:
            re_name = '3Dbox'
        elif idx == 2:
            re_name = 'cls'
        outputs_head = keras.layers.Reshape((-1, num_values), name=re_name)(outputs_head)

        output_list.append(outputs_head)

    return output_list


def retinanet(
    inputs,
    backbone_layers_rgb,
    backbone_layers_dep,
    num_classes,
    num_anchors             = None,
    create_pyramid_features = __create_sparceFPN,
    submodels               = None,
    name                    = 'retinanet'
):

    options = {
        'kernel_size': 1,
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer': 'zeros',
        'activation'   : swish,
        #'kernel_regularizer': keras.regularizers.l2(0.001),
    }

    options2 = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer': 'zeros',
        #'activation': swish,
        # 'kernel_regularizer': keras.regularizers.l2(0.001),
    }

    if num_anchors is None:
        num_anchors = AnchorParameters.default.num_anchors()

    if submodels is None:

        submodels = default_submodels(num_classes, num_anchors)
        #submodels_2 = default_submodels_2(num_classes, num_anchors)

    b1, b2, b3 = backbone_layers_rgb
    b4, b5, b6 = backbone_layers_dep

    ############################
    ####   pre FPN fusion    ###
    ############################
    #C3 = keras.layers.Concatenate()([b1, b4])
    #C3 = keras.layers.Conv2D(256, **options)(C3)
    #C4 = keras.layers.Concatenate()([b2, b5])
    #C4 = keras.layers.Conv2D(256, **options)(C4)
    #C5 = keras.layers.Concatenate()([b3, b6])
    #C5 = keras.layers.Conv2D(256, **options)(C5)
    #features = create_pyramid_features(C3, C4, C5)
    #pyramids = __build_pyramid(submodels, features)

    ############################
    ####      FPN fusion     ###
    ############################
    # FPN fusion
    features = create_pyramid_features(b1, b2, b3, b4, b5, b6)
    pyramids = __build_pyramid(submodels, features)

    ############################
    ####   post FPN fusion   ###
    ############################
    # FPN, feature concatenation to 512 feature maps
    # features1 = create_pyramid_features_2(b1, b2, b3)
    # features2 = create_pyramid_features_2(b4, b5, b6)
    # P3_con = keras.layers.Concatenate(name='P3_con')([features1[0], features2[0]])
    # P4_con = keras.layers.Concatenate(name='P4_con')([features1[1], features2[1]])
    # P5_con = keras.layers.Concatenate(name='P5_con')([features1[2], features2[2]])
    # P6_con = keras.layers.Concatenate(name='P6_con')([features1[3], features2[3]])
    # P7_con = keras.layers.Concatenate(name='P7_con')([features1[4], features2[4]])
    # features = [P3_con, P4_con, P5_con, P6_con, P7_con]
    # pyramids = __build_pyramid(submodels, features)
    # print(pyramids)

    # FPN, feature convolution to 256 feature maps
    #features1 = reduced_pyramid_features(b1, b2, b3)
    #features2 = reduced_pyramid_features(b4, b5, b6)
    #P3_con = keras.layers.Concatenate()([features1[0], features2[0]])
    # P3_con = keras.layers.Conv2D(256, **options)(P3_con)
    #P3_con = keras.layers.Conv2D(256, name='P3_con', **options2)(P3_con)
    #P4_con = keras.layers.Concatenate()([features1[1], features2[1]])
    # P4_con = keras.layers.Conv2D(256, **options)(P4_con)
    #P4_con = keras.layers.Conv2D(256, name='P4_con', **options2)(P4_con)
    #P5_con = keras.layers.Concatenate()([features1[2], features2[2]])
    # P5_con = keras.layers.Conv2D(256, **options)(P5_con)
    #P5_con = keras.layers.Conv2D(256, name='P5_con', **options2)(P5_con)
    # P6_con = keras.layers.Concatenate()([features1[3], features2[3]])
    # P6_con = keras.layers.Conv2D(256, **options)(P6_con)
    # P6_con = keras.layers.Conv2D(256, name='P6_con', **options2)(P6_con)
    # P7_con = keras.layers.Concatenate()([features1[4], features2[4]])
    # P7_con = keras.layers.Conv2D(256, **options)(P7_con)
    # P7_con = keras.layers.Conv2D(256, name='P7_con', **options2)(P7_con)
    #features = [P3_con, P4_con, P5_con]#, P6_con, P7_con]
    #pyramids = __build_pyramid(submodels, features)

    # correlated features
    # features1 = create_pyramid_features(b1, b2, b3)
    # features2 = create_pyramid_features(b4, b5, b6)
    #b14_con = keras.layers.Concatenate()([b1, b4])
    #b25_con = keras.layers.Concatenate()([b2, b5])
    #b36_con = keras.layers.Concatenate()([b3, b6])
    #features_corr = create_pyramid_features(b14_con, b25_con, b36_con)
    # P3_con = keras.layers.Concatenate(name='P3_con')([features1[0], features2[0], features_corr[0]])
    # P4_con = keras.layers.Concatenate(name='P4_con')([features1[1], features2[1], features_corr[1]])
    # P5_con = keras.layers.Concatenate(name='P5_con')([features1[2], features2[2], features_corr[2]])
    # P6_con = keras.layers.Concatenate(name='P6_con')([features1[3], features2[3], features_corr[3]])
    # P7_con = keras.layers.Concatenate(name='P7_con')([features1[4], features2[4], features_corr[4]])

    # naive output and output with feature fusion
    #features1 = create_pyramid_features(b1, b2, b3)
    #features2 = create_pyramid_features_2(b4, b5, b6)
    #pyramids = __build_pyramid_duo(submodels_2, features1, features2)
    #fused_pyramids = output_fusion_model(pyramids, num_anchors)
    #pyramids = __fuse_pyramid(submodels_2, features1, features2, num_anchors)



    #fused_pyramids = default_fusion_model(pyramids, features, num_anchors)

    return keras.models.Model(inputs=inputs, outputs=pyramids, name=name)


def retinanet_bbox(
    model                 = None,
    nms                   = True,
    class_specific_filter = True,
    name                  = 'retinanet-bbox',
    anchor_params         = None,
    **kwargs
):

    # if no anchor parameters are passed, use default values
    if anchor_params is None:
        anchor_params = AnchorParameters.default

    # create RetinaNet model
    if model is None:
        model = retinanet(num_anchors=anchor_params.num_anchors(), **kwargs)
    else:
        assert_training_model(model)

    # compute the anchors
    #features = [model.get_layer(p_name).output for p_name in ['P3_con', 'P4_con', 'P5_con', 'P6_con', 'P7_con']]
    features = [model.get_layer(p_name).output for p_name in ['P3', 'P4', 'P5']]
    anchors = __build_anchors(anchor_params, features)

    # we expect the anchors, regression and classification values as first output
    #intermediate_tensor_function = ([model.inputs], [model.outputs[-1]])
    #pyramids = intermediate_tensor_function([model.outputs[-1]])[0]
    regression = model.outputs[0]
    regression3D = model.outputs[1]
    classification = model.outputs[2]
    other = model.outputs[3:]

    # apply predicted regression to anchors
    boxes = layers.RegressBoxes(name='boxes')([anchors, regression])
    boxes = layers.ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])

    boxes3D = layers.RegressBoxes3D(name='boxes3D')([anchors, regression3D])

    # filter detections (apply NMS / score threshold / select top-k)
    detections = layers.FilterDetections(
        nms                   = nms,
        class_specific_filter = class_specific_filter,
        name                  = 'filtered_detections'
    )([boxes, boxes3D, classification] + other)

    # construct the model
    return keras.models.Model(inputs=model.inputs, outputs=detections, name=name)
