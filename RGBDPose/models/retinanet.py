import keras
import tensorflow as tf
from .. import initializers
from .. import layers
from ..utils.anchors import AnchorParameters
from . import assert_training_model


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


def default_3Dregression_model(num_values, num_anchors, pyramid_feature_size=256, regression_feature_size=256, name='3Dregression_submodel'):
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


def default_mask_decoder(
        num_classes,
        num_anchors):

    options3 = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer': 'zeros',
        'kernel_regularizer' : keras.regularizers.l2(0.001),
    }

    if keras.backend.image_data_format() == 'channels_first':
        inputs_P5 = keras.layers.Input(shape=(256, None, None))
        inputs_P4 = keras.layers.Input(shape=(256, None, None))
        inputs_P3 = keras.layers.Input(shape=(256, None, None))
        # inputs_P2 = keras.layers.Input(shape=(64, None, None))
    else:
        inputs_P5 = keras.layers.Input(shape=(15, 20, 256))
        inputs_P4 = keras.layers.Input(shape=(30, 40, 256))
        inputs_P3 = keras.layers.Input(shape=(60, 80, 256))
        # inputs_P2 = keras.layers.Input(shape=(None, None, 64))

    inputs = [inputs_P3, inputs_P4, inputs_P5]

    D5 = keras.layers.Conv2D(256, activation='relu', **options3)(inputs_P5)
    D5 = keras.layers.Conv2D(256, activation='relu', **options3)(D5)
    D5_up = keras.layers.Conv2DTranspose(256, activation='relu', kernel_size=2, strides=2, padding='valid')(D5)
    D4 = keras.layers.Add()([D5_up, inputs_P4])

    D4 = keras.layers.Conv2D(256, activation='relu', **options3)(D4)
    D4 = keras.layers.Conv2D(256, activation='relu', **options3)(D4)
    D4_up = keras.layers.Conv2DTranspose(256, activation='relu', kernel_size=2, strides=2, padding='valid')(D4)
    D3 = keras.layers.Add()([D4_up, inputs_P3])

    D3 = keras.layers.Conv2D(128, activation='relu', **options3)(D3)
    D3 = keras.layers.Conv2D(128, activation='relu', **options3)(D3)
    outputs = keras.layers.Conv2D(filters=num_classes, **options3)(D3)

    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1))(outputs)
    outputs = keras.layers.Reshape((-1, num_classes))(outputs)

    outputs = keras.layers.Activation('sigmoid')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name='mask')  # , name=name)


def default_attention_pnp(
        num_classes,
        num_values,
):
    if keras.backend.image_data_format() == 'channels_first':
        inputs = keras.layers.Input(shape=(num_values, None))
    else:
        inputs = keras.layers.Input(shape=(None, num_values))

    outputs = inputs

    outputs = keras.layers.Conv1D(filters=128, kernel_size=1, padding="same", activation='relu')(outputs)
    outputs = keras.layers.Conv1D(filters=128, kernel_size=1, padding="same", activation='relu')(outputs)
    outputs = keras.layers.Conv1D(filters=128, kernel_size=1, padding="same", activation='relu')(outputs)
    outputs = keras.layers.GlobalMaxPool1D()(outputs)

    outputs = keras.layers.Dense(512, activation='relu')(outputs)
    outputs = keras.layers.Dense(256, activation='relu')(outputs)
    outputs = keras.layers.Dense(num_classes * 7)(outputs)

    cls_outputs = keras.layers.Reshape((num_classes, 7))(outputs)

    return keras.models.Model(inputs=inputs, outputs=cls_outputs, name='poses')


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


def __create_sparceFPN(P3, P4, P5, feature_size=256):
#def __create_sparceFPN(C3_R, C4_R, C5_R, C3_D, C4_D, C5_D, feature_size=256):

    # only from here for FPN-fusion test 3
    #C3 = keras.layers.Add()([C3_R, C3_D])
    #C4 = keras.layers.Add()([C4_R, C4_D])
    #C5 = keras.layers.Add()([C5_R, C5_D])

    #P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same')(C3)
    #P4 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same')(C4)
    #P5 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same')(C5)
    
    # 3x3 conv for test 4
    #P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C3_D)
    #P4 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C4_D)
    #P5 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C5_D)

    P5_upsampled = layers.UpsampleLike()([P5, P4])
    P4_upsampled = layers.UpsampleLike()([P4, P3])
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
        #('bbox', default_regression_model(4, num_anchors)),
        ('3Dbox', default_3Dregression_model(16, num_anchors)),
        ('cls', default_classification_model(num_classes, num_anchors))
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


def __build_anchors_pnp(anchor_parameters, features):
    anchors = [
        layers.Anchors(
            size=anchor_parameters.sizes[i],
            stride=anchor_parameters.strides[i],
            ratios=anchor_parameters.ratios,
            scales=anchor_parameters.scales,
            name='anchors_pnp_{}'.format(i)
        )(f) for i, f in enumerate(features)
    ]

    return keras.layers.Concatenate(axis=1, name='anchors_pnp')(anchors)


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

    mask_head = default_mask_decoder(num_classes=num_classes, num_anchors=num_anchors)
    attention_pnp = default_attention_pnp(num_classes, 16)

    b1, b2, b3 = backbone_layers_rgb
    b4, b5, b6 = backbone_layers_dep

    # feature fusion
    C3 = keras.layers.Add()([b1, b4])
    C4 = keras.layers.Add()([b2, b5])
    C5 = keras.layers.Add()([b3, b6])

    P3 = keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(C3)
    P4 = keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(C4)
    P5 = keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(C5)

    features = create_pyramid_features(P3, P4, P5)
    # features = create_pyramid_features(b1, b2, b3, b4, b5, b6)
    pyramids = __build_pyramid(submodels, features)

    #anchors = __build_anchors_pnp(AnchorParameters.default, features)
    #boxes = pyramids[0]
    #boxes = layers.DenormBoxes3D()([anchors, boxes])

    masks = mask_head([P3, P4, P5])
    pyramids.append(masks)

    #poses = attention_pnp(boxes)
    #pyramids.append(poses)

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
    poses = model.outputs[2]
    other = model.outputs[3:]

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

    #detections.append(model.layers[368].output)
    #detections.append(model.layers[369].output)
    #detections.append(model.layers[370].output)

    #detections.append(model.layers[190].output)
    #detections.append(model.layers[191].output)
    #detections.append(model.layers[192].output)

    # construct the model
    #return keras.models.Model(inputs=model.inputs, outputs=detections, name=name)
    return keras.models.Model(inputs=model.inputs, outputs=[boxes3D, classification, poses], name=name)
