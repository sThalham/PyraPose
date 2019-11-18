import keras
import tensorflow
from .. import initializers
from .. import layers
from ..utils.anchors import AnchorParameters
from . import assert_training_model


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
            filters=regression_feature_size,
            activation='relu',
            #name='pyramid_regression3D_{}'.format(i),
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(num_anchors * num_values, **options)(outputs) #, name='pyramid_regression3D'
    #outputs = keras.layers.Flatten
    #outputs = keras.layers.Dense(num_anchors* num_values, name='pyramid_regression3D')(outputs)
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1))(outputs) # , name='pyramid_regression3D_permute'
    outputs = keras.layers.Reshape((-1, num_values))(outputs) # , name='pyramid_regression3D_reshape'

    return keras.models.Model(inputs=inputs, outputs=outputs) #, name=name)


def __create_pyramid_features(C3, C4, C5, feature_size=256):
    P5 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C5)
    P5_upsampled = layers.UpsampleLike()([P5, C4])
    P5 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)

    # add P5 elementwise to C4
    P4 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C4)
    P4 = keras.layers.Add()([P5_upsampled, P4])
    P4_upsampled = layers.UpsampleLike()([P4, C3])
    P4 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)

    # add P4 elementwise to C3
    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C3)
    P3 = keras.layers.Add()([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    P7 = keras.layers.Activation('relu')(P6)
    P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    return [P3, P4, P5, P6, P7]


def __create_pyramid_features_2(C3, C4, C5, feature_size=256):
    P5 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C5)
    P5_upsampled = layers.UpsampleLike()([P5, C4])
    P5 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5_2')(P5)

    # add P5 elementwise to C4
    P4 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C4)
    P4 = keras.layers.Add()([P5_upsampled, P4])
    P4_upsampled = layers.UpsampleLike()([P4, C3])
    P4 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4_2')(P4)

    # add P4 elementwise to C3
    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C3)
    P3 = keras.layers.Add()([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    P7 = keras.layers.Activation('relu')(P6)
    P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    return [P3, P4, P5, P6, P7]


def projection_block(inputs, feature_size=256, BN=True):
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer': 'zeros'
    }

    outputs = keras.layers.Conv2D(filters=feature_size, activation='relu', kernel_size=7, padding='valid', ** options)(inputs)
    #outputs = keras.layers.BatchNormalization(axis=3)(outputs)
    #outputs = keras.activations.relu(outputs)

    #outputs = keras.layers.Conv2D(filters=feature_size, activation='relu', kernel_size=3, padding='valid')(outputs)
    #outputs = keras.layers.BatchNormalization(axis=3)(outputs)
    #outputs = keras.activations.relu(outputs)

    #outputs = keras.layers.Conv2D(filters=feature_size, activation='relu', **options)(inputs)
    #outputs = keras.layers.GlobalMaxPooling2D()(inputs)

    return outputs


def __create_projection_features(C3, C4, C5, feature_size=256):

    F3 = projection_block(C5)
    F2 = projection_block(C4)
    F1 = projection_block(C3)

    return [F1, F2, F3]


def default_submodels(num_classes, num_anchors):
    return [
        ('bbox', default_regression_model(4, num_anchors)),
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


def fuse_rgb_and_d(anchors1, anchors2):

    anchors = []
    for i, an in enumerate(anchors):

        bb = keras.layers.Concatenate(axis=1)([anchors1[i][0], anchors2[i][0]])
        bb = keras.layers.Dense()



def retinanet(
    inputs,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    num_classes,
    num_anchors             = None,
    create_pyramid_features = __create_pyramid_features,
    submodels               = None,
    name                    = 'retinanet'
):

    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer': 'zeros',
        'kernel_regularizer': keras.regularizers.l2(0.001),
    }

    if num_anchors is None:
        num_anchors = AnchorParameters.default.num_anchors()

    if submodels is None:

        submodels = default_submodels(num_classes, num_anchors)

    # FPN, feature concatenation to 512 feature maps
    #features1 = create_pyramid_features(b1, b2, b3)
    #features2 = create_pyramid_features(b4, b5, b6)
    #P3_con = keras.layers.Concatenate(name='P3_con')([features1[0], features2[0]])
    #P4_con = keras.layers.Concatenate(name='P4_con')([features1[1], features2[1]])
    #P5_con = keras.layers.Concatenate(name='P5_con')([features1[2], features2[2]])
    #P6_con = keras.layers.Concatenate(name='P6_con')([features1[3], features2[3]])
    #P7_con = keras.layers.Concatenate(name='P7_con')([features1[4], features2[4]])
    #features = [P3_con, P4_con, P5_con, P6_con, P7_con]
    #pyramids = __build_pyramid(submodels, features)

    # FPN, feature convolution to 256 feature maps
    #features1 = create_pyramid_features(b1, b2, b3)
    #features2 = create_pyramid_features(b4, b5, b6)
    #P3_con = keras.layers.Concatenate()([features1[0], features2[0]])
    #P3_con = keras.layers.Conv2D(256, name='P3_con', **options)(P3_con)
    #P4_con = keras.layers.Concatenate()([features1[1], features2[1]])
    #P4_con = keras.layers.Conv2D(256, name='P4_con', **options)(P4_con)
    #P5_con = keras.layers.Concatenate()([features1[2], features2[2]])
    #P5_con = keras.layers.Conv2D(256, name='P5_con', **options)(P5_con)
    #P6_con = keras.layers.Concatenate()([features1[3], features2[3]])
    #P6_con = keras.layers.Conv2D(256, name='P6_con', **options)(P6_con)
    #P7_con = keras.layers.Concatenate()([features1[4], features2[4]])
    #P7_con = keras.layers.Conv2D(256, name='P7_con', **options)(P7_con)
    #features = [P3_con, P4_con, P5_con, P6_con, P7_con]
    #pyramids = __build_pyramid(submodels, features)

    # projection module
    #features1 = __create_projection_features(b1, b2, b3)
    #features2 = __create_projection_features(b4, b5, b6)
    #P3_con = keras.layers.Concatenate(axis=3, name='P3_con')([features1[0], features2[0]])
    #P4_con = keras.layers.Concatenate(axis=3, name='P4_con')([features1[1], features2[1]])
    #P5_con = keras.layers.Concatenate(axis=3, name='P5_con')([features1[2], features2[2]])
    #if keras.backend.image_data_format() == 'channels_first':
    #    P3_con = keras.layers.Permute((2, 3, 1))(P3_con)
    #    P4_con = keras.layers.Permute((2, 3, 1))(P4_con)
    #    P5_con = keras.layers.Permute((2, 3, 1))(P5_con)
    #print(P3_con)
    #print(P4_con)
    #print(P5_con)
    #P3_con = keras.backend.expand_dims(P3_con, axis=1)
    #P4_con = keras.backend.expand_dims(P4_con, axis=1)
    #P5_con = keras.backend.expand_dims(P5_con, axis=1)
    #P3_con = keras.layers.Reshape((-1, 512))(P3_con)
    #P4_con = keras.layers.Reshape((-1, 512))(P4_con)
    #P5_con = keras.layers.Reshape((-1, 512))(P5_con)
    #features = [P3_con, P4_con, P5_con]
    #pyramids = __build_pyramid(submodels, features)

    # learn fusion
    features1 = create_pyramid_features(b1, b2, b3)
    features2 = create_pyramid_features_2(b4, b5, b6)
    pyramids1 = __build_pyramid(submodels, features1)
    pyramids2 = __build_pyramid(submodels, features2)

    return keras.models.Model(inputs=inputs, outputs=[pyramids1, pyramids2], name=name)


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
    #features = [model.get_layer(p_name).output for p_name in ['P3_con', 'P4_con', 'P5_con']]
    features1 = [model.get_layer(p_name).output for p_name in ['P3', 'P4', 'P5', 'P6', 'P7']]
    features2 = [model.get_layer(p_name).output for p_name in ['P3_2', 'P4_2', 'P5_2', 'P6_2', 'P7_2']]
    anchors1  = __build_anchors(anchor_params, features1)
    anchors2 = __build_anchors(anchor_params, features2)

    fused_outputs = fuse_rgb_and_d(anchors1, anchors2)

    # we expect the anchors, regression and classification values as first output
    #regression = model.outputs[0]
    #regression3D = model.outputs[1]
    #classification = model.outputs[2]
    #other = model.outputs[3:]

    regression = fused_outputs.outputs[0]
    regression3D = fused_outputs.outputs[1]
    classification = fused_outputs.outputs[2]
    other = fused_outputs.outputs[3:]

    # apply predicted regression to anchors
    boxes = layers.RegressBoxes(name='boxes')([anchors1, regression])
    boxes = layers.ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])

    boxes3D = layers.RegressBoxes3D(name='boxes3D')([anchors1, regression3D])

    # filter detections (apply NMS / score threshold / select top-k)
    detections = layers.FilterDetections(
        nms                   = nms,
        class_specific_filter = class_specific_filter,
        name                  = 'filtered_detections'
    )([boxes, boxes3D, classification] + other)

    # construct the model
    return keras.models.Model(inputs=model.inputs, outputs=detections, name=name)
