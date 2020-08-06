import keras
import tensorflow as tf
from .. import initializers
from .. import layers
from ..utils.anchors import AnchorParameters
from . import assert_training_model


def swish(x, beta=1.0):
    return x * keras.activations.sigmoid(beta*x)


class wBiFPNAdd(keras.layers.Layer):
    def __init__(self, epsilon=1e-4, **kwargs):
        super(wBiFPNAdd, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        num_in = len(input_shape)
        self.w = self.add_weight(name=self.name,
                                 shape=(num_in,),
                                 initializer=keras.initializers.constant(1 / num_in),
                                 trainable=True,
                                 dtype=tf.float32)

    def call(self, inputs, **kwargs):
        w = keras.activations.relu(self.w)
        x = tf.reduce_sum([w[i] * inputs[i] for i in range(len(inputs))], axis=0)
        x = x / (tf.reduce_sum(w) + self.epsilon)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(wBiFPNAdd, self).get_config()
        config.update({
            'epsilon': self.epsilon
        })
        return config


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


def default_mask_model(
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
            filters=classification_feature_size,
            activation='relu',
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(
        filters=num_classes,
        kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer=initializers.PriorProbability(probability=prior_probability),
        **options
    )(outputs)

    # reshape output and apply sigmoid
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1))(outputs) #, name='pyramid_classification_permute'
    outputs = keras.layers.Reshape((-1, num_classes))(outputs) # , name='pyramid_classification_reshape'
    outputs = keras.layers.Activation('sigmoid')(outputs) # , name='pyramid_classification_sigmoid'

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


def __create_BiFPN_noW(C3_R, C4_R, C5_R, C3_D, C4_D, C5_D, feature_size=256):
    P3_r = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C3_R)
    P4_r = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C4_R)
    P5_r = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C5_R)
    P6_r = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same')(C5_R)
    P7_r = keras.layers.Activation('relu')(P6_r)
    P7_r = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same')(P7_r)

    P3_d = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C3_D)
    P4_d = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C4_D)
    P5_d = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C5_D)
    P6_d = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same')(C5_D)
    P7_d = keras.layers.Activation('relu')(P6_d)
    P7_d = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same')(P7_d)

    P3 = keras.layers.Add()([P3_r, P3_d])
    P4 = keras.layers.Add()([P4_r, P4_d])
    P5 = keras.layers.Add()([P5_r, P5_d])
    P6 = keras.layers.Add()([P6_r, P6_d])
    P7 = keras.layers.Add()([P7_r, P7_d])

    P7_upsampled = layers.UpsampleLike()([P7, P6])
    P6_td = wBiFPNAdd()([P7_upsampled, P6])
    #P6_td = keras.layers.Add()([P7_upsampled, P6])
    P6_td = keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(P6_td)
    P6_td = keras.layers.BatchNormalization(axis=-1)(P6_td)
    P6_td = keras.layers.Activation('relu')(P6_td)

    P6_upsampled = layers.UpsampleLike()([P6_td, P5])
    P5_td = wBiFPNAdd()([P6_upsampled, P5])
    #P5_td = keras.layers.Add()([P6_upsampled, P5])
    P5_td = keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(P5_td)
    P5_td = keras.layers.BatchNormalization(axis=-1)(P5_td)
    P5_td = keras.layers.Activation('relu')(P5_td)

    P5_upsampled = layers.UpsampleLike()([P5_td, P4])
    P4_td = wBiFPNAdd()([P5_upsampled, P4])
    #P4_td = keras.layers.Add()([P5_upsampled, P4])
    P4_td = keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(P4_td)
    P4_td = keras.layers.BatchNormalization(axis=-1)(P4_td)
    P4_td = keras.layers.Activation('relu')(P4_td)

    P4_upsampled = layers.UpsampleLike()([P4_td, P3])
    P3_out = wBiFPNAdd()([P4_upsampled, P3])
    #P3_out = keras.layers.Add()([P4_upsampled, P3])
    P3_out = keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(P3_out)
    P3_out = keras.layers.BatchNormalization(axis=-1)(P3_out)
    P3_out = keras.layers.Activation('relu', name='P3_con')(P3_out)

    P3_down = keras.layers.MaxPooling2D(strides=2)(P3_out)
    P4_out = wBiFPNAdd()([P3_down, P4_td, P4])
    #P4_out = keras.layers.Add()([P3_down, P4_td, P4])
    P4_out = keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(P4_out)
    P4_out = keras.layers.BatchNormalization(axis=-1)(P4_out)
    P4_out = keras.layers.Activation('relu', name='P4_con')(P4_out)

    P4_down = keras.layers.MaxPooling2D(strides=2)(P4_out)
    P5_out = wBiFPNAdd()([P4_down, P5_td, P5])
    #P5_out = keras.layers.Add()([P4_down, P5_td, P5])
    P5_out = keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(P5_out)
    P5_out = keras.layers.BatchNormalization(axis=-1)(P5_out)
    P5_out = keras.layers.Activation('relu', name='P5_con')(P5_out)

    P5_down = keras.layers.MaxPooling2D(strides=2, padding='same')(P5_out)
    P6_out = wBiFPNAdd()([P5_down, P6_td, P6])
    #P6_out = keras.layers.Add()([P5_down, P6_td, P6])
    P6_out = keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(P6_out)
    P6_out = keras.layers.BatchNormalization(axis=-1)(P6_out)
    P6_out = keras.layers.Activation('relu', name='P6_con')(P6_out)

    P6_down = keras.layers.MaxPooling2D(strides=2)(P6_out)
    P7_out = wBiFPNAdd()([P6_down, P7])
    #P7_out = keras.layers.Add()([P6_down, P7])
    P7_out = keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(P7_out)
    P7_out = keras.layers.BatchNormalization(axis=-1)(P7_out)
    P7_out = keras.layers.Activation('relu', name='P7_con')(P7_out)

    return [P3_out, P4_out, P5_out, P6_out, P7_out]


def __create_sparceFPN(C3_R, C4_R, C5_R, C3_D, C4_D, C5_D, feature_size=256):

    # only from here for FPN-fusion test 3
    C3 = keras.layers.Add()([C3_R, C3_D])
    C4 = keras.layers.Add()([C4_R, C4_D])
    C5 = keras.layers.Add()([C5_R, C5_D])

    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same')(C3)
    P4 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same')(C4)
    P5 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same')(C5)
    
    # 3x3 conv for test 4
    #P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C3_D)
    #P4 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C4_D)
    #P5 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C5_D)

    P5_upsampled = layers.UpsampleLike()([P5, C4_D])
    P4_upsampled = layers.UpsampleLike()([P4, C3_D])
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
        ('3Dbox', default_3Dregression_model(16, num_anchors)),
        ('cls', default_classification_model(num_classes, num_anchors)),
        ('mask', default_mask_model(num_classes, num_anchors))
    ]


def __build_model_pyramid(name, model, features):
    return keras.layers.Concatenate(axis=1, name=name)([model(f) for f in features])


def __build_pyramid(models, features):
    return [__build_model_pyramid(n, m, features) for n, m in models]


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

    if num_anchors is None:
        num_anchors = AnchorParameters.default.num_anchors()

    if submodels is None:

        submodels = default_submodels(num_classes, num_anchors)
        #submodels_2 = default_submodels_2(num_classes, num_anchors)

    b1, b2, b3 = backbone_layers_rgb
    b4, b5, b6 = backbone_layers_dep

    # FPN fusion
    features = create_pyramid_features(b1, b2, b3, b4, b5, b6)
    pyramids = __build_pyramid(submodels, features)

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

    #for i in range(len(model.layers)):
    #    layer = model.layers[i]
    #    print(i, layer.name, layer.output.shape)
    #print(model.layers[368].output)
    #print(model.layers[369].output)
    #print(model.layers[370].output)
    #print(model.layers[355].output)
    #print(model.layers[351].output)
    #print(model.layers[356].output)

    #print(model.layers[190].output.get_weights())
    #print(model.layers[191].output)
    #print(model.layers[192].output)

    #for layer in model.layers:
    #    if 'conv' not in layer.name:
    #        continue
    #    filters, biases = layer.get_weights()
    #    print(layer.name, filters.shape)

    # we expect the anchors, regression and classification values as first output
    #intermediate_tensor_function = ([model.inputs], [model.outputs[-1]])
    #pyramids = intermediate_tensor_function([model.outputs[-1]])[0]
    #regression = model.outputs[0]
    regression3D = model.outputs[0]
    classification = model.outputs[1]
    mask = model.outputs[2]

    # apply predicted regression to anchors
    #boxes = layers.RegressBoxes(name='boxes')([anchors, regression])
    #boxes = layers.ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])

    boxes3D = layers.RegressBoxes3D(name='boxes3D')([anchors, regression3D])

    # filter detections (apply NMS / score threshold / select top-k)
    #detections = layers.FilterDetections(
    #    nms                   = nms,
    #    class_specific_filter = class_specific_filter,
    #    name                  = 'filtered_detections'
    #)([boxes, boxes3D, classification] + other)

    # construct the model
    return keras.models.Model(inputs=model.inputs, outputs=[boxes3D, classification, mask], name=name)
