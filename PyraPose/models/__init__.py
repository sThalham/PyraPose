from __future__ import print_function
import sys


class Backbone(object):
    """ This class stores additional information on backbones.
    """
    def __init__(self, backbone):
        # a dictionary mapping custom layer names to the correct classes
        from .. import layers
        from .. import losses
        from .. import initializers
        from . import retinanet
        self.custom_objects = {
            'UpsampleLike'     : layers.UpsampleLike,
            'PriorProbability' : initializers.PriorProbability,
            'RegressBoxes'     : layers.RegressBoxes,
            #'FilterDetections' : layers.FilterDetections,
            'Anchors'          : layers.Anchors,
            'ClipBoxes'        : layers.ClipBoxes,
            '_smooth_l1'       : losses.smooth_l1(),
            '_smooth_l1_pose'  : losses.smooth_l1_pose(),
            '_focal'           : losses.focal(),
            '_focal_mask'      : losses.focal_mask(),
            '_cross'           : losses.cross(),
            '_wMSE'            : losses.weighted_mse(),
            '_wl1'            : losses.weighted_l1(),
            '_msle'           : losses.weighted_msle(),
            '_orth_l1'         : losses.orthogonal_l1(),
            'RegressBoxes3D'   : layers.RegressBoxes3D(),
            'DenormBoxes3D'   : layers.DenormBoxes3D(),
            '_smooth_reconstruction_l1' : losses.smooth_reconstruction_l1(),
        }

        self.backbone = backbone
        self.validate()

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        raise NotImplementedError('retinanet method not implemented.')

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        raise NotImplementedError('validate method not implemented.')

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        Having this function in Backbone allows other backbones to define a specific preprocessing step.
        """
        raise NotImplementedError('preprocess_image method not implemented.')


def backbone(backbone_name):
    if 'resnet' in backbone_name:
        from .resnet import ResNetBackbone as b
    elif 'efficientnets' in backbone_name:
        from .efficientnet import EffNetBackbone as b
    elif 'densenet' in backbone_name:
        from .densenet import DenseNetBackbone as b
    else:
        raise NotImplementedError('Backbone class for  \'{}\' not implemented.'.format(backbone))

    return b(backbone_name)


def load_model(filepath, backbone_name='resnet50'):
    import tensorflow.keras as keras
    #import keras as keras
    from .. import layers
    from .. import losses
    from .. import initializers
    from . import retinanet
    custom_objects = {
        #'UpsampleLike': layers.UpsampleLike,
        #'PriorProbability': initializers.PriorProbability,
        #'RegressBoxes': layers.RegressBoxes,
        # 'FilterDetections' : layers.FilterDetections,
        #'Anchors': layers.Anchors,
        #'ClipBoxes': layers.ClipBoxes,
        #'_smooth_l1': losses.smooth_l1(),
        #'_smooth_l1_pose': losses.smooth_l1_pose(),
        #'_focal': losses.focal(),
        #'_focal_mask': losses.focal_mask(),
        #'_cross': losses.cross(),
        #'_wMSE': losses.weighted_mse(),
        #'_wl1': losses.weighted_l1(),
        #'_msle': losses.weighted_msle(),
        #'_orth_l1': losses.orthogonal_l1(),
        #'RegressBoxes3D': layers.RegressBoxes3D(),
        #'DenormBoxes3D': layers.DenormBoxes3D(),
        #'_smooth_reconstruction_l1': losses.smooth_reconstruction_l1(),
    }

    print(backbone(backbone_name).custom_objects)
    return keras.models.load_model(filepath, custom_objects=backbone(backbone_name).custom_objects)
    #return keras.models.load_model(filepath, custom_objects=custom_objects)


def convert_model(model, nms=True, class_specific_filter=True, anchor_params=None):
    from .retinanet import retinanet_bbox
    return retinanet_bbox(model=model, nms=nms, class_specific_filter=class_specific_filter, anchor_params=anchor_params)


def assert_training_model(model):
    #assert (all(output in model.output_names for output in ['3Dbox', 'cls', 'mask'])), "Input is not a training model. Outputs were found, outputs are: {}).".format(model.output_names)
    #assert (all(output in model.output_names for output in ['3Dbox', 'cls', 'mask', 'domain', 'features'])), "Input is not a training model. Outputs were found, outputs are: {}).".format(model.output_names)
    #assert (all(output in model.output_names for output in ['3Dbox', 'cls', 'mask', 'reconstruction', 'domain'])), "Input is not a training model. Outputs were found, outputs are: {}).".format(model.output_names)
    assert (all(output in model.output_names for output in ['3Dbox', 'cls', 'mask'])), "Input is not a training model. Outputs were found, outputs are: {}).".format(model.output_names)


def check_training_model(model):
    try:
        assert_training_model(model)
    except AssertionError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
