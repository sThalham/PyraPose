#!/usr/bin/env python

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

import argparse
import os
import sys
import warnings

import tensorflow.keras as keras
import tensorflow.keras.preprocessing.image
import tensorflow as tf
import numpy as np
import yaml
import json

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import PyraPose.bin  # noqa: F401
    __package__ = "PyraPose.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import layers  # noqa: F401
from .. import losses
from .. import models
from ..callbacks import RedirectModel
from ..callbacks.eval import Evaluate
from ..models.retinanet import retinanet_bbox
from ..utils.anchors import make_shapes_callback
from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.model import freeze as freeze_model
from ..utils.transform import random_transform_generator


def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


#def get_session():
#    config = tf.ConfigProto()
#    config.gpu_options.allow_growth = True
#    return tf.Session(config=config)


def model_with_weights(model, weights, skip_mismatch):
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def create_models(backbone_retinanet, num_classes, weights, multi_gpu=0,
                  freeze_backbone=False, lr=1e-5):

    modifier = freeze_model if freeze_backbone else None

    # load anchor parameters, or pass None (so that defaults will be used)
    anchor_params = None
    num_anchors   = None

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    if multi_gpu > 1:
        from keras.utils import multi_gpu_model
        with tf.device('/cpu:0'):
            model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        model          = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = model

    # compile model
    training_model.compile(
        loss={
            '3Dbox'        : losses.sym_orthogonal_l1(),
            'cls'          : losses.focal(),
            #'mask'          : losses.focal(),
        },
        optimizer=keras.optimizers.Adam(lr=lr, clipnorm=0.001)
    )

    return model, training_model


def create_callbacks(model, training_model, args):
    callbacks = []

    tensorboard_callback = None

    '''
    if args.evaluation and validation_generator:
        if args.dataset_type == 'coco':
            from ..callbacks.coco import CocoEval

            # use prediction model for evaluation
            evaluation = CocoEval(validation_generator, tensorboard=tensorboard_callback)
        elif args.dataset_type == 'linemod':
            from ..callbacks.linemod import LinemodEval
            evaluation = LinemodEval(validation_generator, tensorboard=tensorboard_callback)

        else:
            evaluation = Evaluate(validation_generator, tensorboard=tensorboard_callback, weighted_average=args.weighted_average)
        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)
    '''

        # save the model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(args.snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                '{backbone}_{dataset_type}_{{epoch:02d}}.h5'.format(backbone=args.backbone, dataset_type=args.dataset_type)
            ),
            #verbose=1,
            #save_best_only=True,
            #monitor="val_loss",
            #mode='auto'
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor    = 'loss',
        factor     = 0.1,
        patience   = 2,
        verbose    = 1,
        mode       = 'auto',
        min_delta  = 0.0001,
        cooldown   = 0,
        min_lr     = 0
    ))

    return callbacks


def create_generators(args, preprocess_image):
    """ Create generators for training and validation.

    Args
        args             : parseargs object containing configuration for generators.
        preprocess_image : Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size'       : args.batch_size,
        'image_min_side'   : args.image_min_side,
        'image_max_side'   : args.image_max_side,
        'preprocess_image' : preprocess_image,
    }

    transform_generator = random_transform_generator(
            min_translation=(0.0, 0.0),
            max_translation=(0.0, 0.0),
            min_scaling=(0.95, 0.95),
            max_scaling=(1.05, 1.05),
        )

    if args.dataset_type == 'ycbv':
        # import here to prevent unnecessary dependency on cocoapi
        from ..preprocessing.ycbv import YCBvGenerator

        train_generator = YCBvGenerator(
            args.ycbv_path,
            'train',
            transform_generator=transform_generator,
            **common_args
        )

        validation_generator = YCBvGenerator(
            args.ycbv_path,
            'val',
            **common_args
        )
    elif args.dataset_type == 'linemod':
        from ..preprocessing.linemod import LinemodGenerator

        train_generator = LinemodGenerator(
            args.linemod_path,
            'train',
            transform_generator=transform_generator,
            **common_args
        )

        validation_generator = LinemodGenerator(
            args.linemod_path,
            'val',
            transform_generator=transform_generator,
            **common_args
        )

    elif args.dataset_type == 'occlusion':
        from ..preprocessing.occlusion import OcclusionGenerator

        train_generator = OcclusionGenerator(
            args.occlusion_path,
            'train',
            transform_generator=transform_generator,
            **common_args
        )

        validation_generator = OcclusionGenerator(
            args.linemod_path,
            'val',
            transform_generator=transform_generator,
            **common_args
        )

    elif args.dataset_type == 'tless':
        from ..preprocessing.tless import TlessGenerator

        train_generator = TlessGenerator(
            args.tless_path,
            'train',
            transform_generator=transform_generator,
            **common_args
        )

        validation_generator = TlessGenerator(
            args.tless_path,
            'val',
            transform_generator=transform_generator,
            **common_args
        )
    elif args.dataset_type == 'homebrewed':
        from ..preprocessing.homebrewed import HomebrewedGenerator

        train_generator = HomebrewedGenerator(
            args.homebrewed_path,
            'train',
            transform_generator=transform_generator,
            **common_args
        )

        validation_generator = HomebrewedGenerator(
            args.homebrewed_path,
            'val',
            transform_generator=transform_generator,
            **common_args
        )
    elif args.dataset_type == 'custom':
        from ..preprocessing.data_custom import CustomDataset

        dataset = CustomDataset(args.custom_path, 'train', batch_size=args.batch_size)
        num_classes = 20
        train_samples = 10300
        dataset = tf.data.Dataset.range(args.workers).interleave(
            lambda _: dataset,
            # num_parallel_calls=tf.data.experimental.AUTOTUNE
            num_parallel_calls=args.workers
        )
        mesh_info = os.path.join(args.custom_path, 'annotations', 'models_info' + '.yml')
        correspondences = np.ndarray((num_classes, 8, 3), dtype=np.float32)
        sphere_diameters = np.ndarray((num_classes), dtype=np.float32)
        for key, value in yaml.load(open(mesh_info)).items():
            x_minus = value['min_x']
            y_minus = value['min_y']
            z_minus = value['min_z']
            x_plus = value['size_x'] + x_minus
            y_plus = value['size_y'] + y_minus
            z_plus = value['size_z'] + z_minus
            three_box_solo = np.array([[x_plus, y_plus, z_plus],
                                       [x_plus, y_plus, z_minus],
                                       [x_plus, y_minus, z_minus],
                                       [x_plus, y_minus, z_plus],
                                       [x_minus, y_plus, z_plus],
                                       [x_minus, y_plus, z_minus],
                                       [x_minus, y_minus, z_minus],
                                       [x_minus, y_minus, z_plus]])
            correspondences[int(key) - 1, :, :] = three_box_solo
            sphere_diameters[int(key) - 1] = value['diameter']
        path = os.path.join(args.custom_path, 'annotations', 'instances_train.json')
        with open(path, 'r') as js:
            data = json.load(js)
        image_ann = data["images"]
        intrinsics = np.ndarray((4), dtype=np.float32)
        for img in image_ann:
            if "fx" in img:
                intrinsics[0] = img["fx"]
                intrinsics[1] = img["fy"]
                intrinsics[2] = img["cx"]
                intrinsics[3] = img["cy"]
            break
        validation_generator = None

    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    #return train_generator, validation_generator
    return dataset, num_classes, correspondences, sphere_diameters, train_samples, intrinsics


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network with object pose estimation.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    ycbv_parser = subparsers.add_parser('ycbv')
    ycbv_parser.add_argument('ycbv_path', help='Path to dataset directory (ie. /tmp/ycbv).')

    linemod_parser = subparsers.add_parser('linemod')
    linemod_parser.add_argument('linemod_path', help='Path to dataset directory (ie. /tmp/linemod).')

    occlusion_parser = subparsers.add_parser('occlusion')
    occlusion_parser.add_argument('occlusion_path', help='Path to dataset directory (ie. /tmp/occlusion.')

    tless_parser = subparsers.add_parser('tless')
    tless_parser.add_argument('tless_path', help='Path to dataset directory (ie. /tmp/tless).')

    homebrewed_parser = subparsers.add_parser('homebrewed')
    homebrewed_parser.add_argument('homebrewed_path', help='Path to dataset directory (ie. /tmp/tless).')

    custom_parser = subparsers.add_parser('custom')
    custom_parser.add_argument('custom_path', help='Path to dataset directory (ie. /tmp/tless).')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot',          help='Resume training from a snapshot.')
    group.add_argument('--imagenet-weights',  help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=True, default=True)
    group.add_argument('--weights',           help='Initialize the model with weights from a file.')
    group.add_argument('--no-weights',        help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=False)

    parser.add_argument('--backbone', help='Backbone model used by retinanet.', default='resnet50', type=str)
    parser.add_argument('--batch-size',       help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--epochs',           help='Number of epochs to train.', type=int, default=20)
    parser.add_argument('--lr',               help='Learning rate.', type=float, default=1e-5)
    parser.add_argument('--snapshot-path',    help='Path to store snapshots of models during training (defaults to \'./models\')', default='./models')
    parser.add_argument('--tensorboard-dir',  help='Log directory for Tensorboard output', default='./logs')
    parser.add_argument('--no-snapshots',     help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation',    help='Disable per epoch evaluation.', dest='evaluation', action='store_true')
    parser.add_argument('--freeze-backbone',  help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=540)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=960)
    parser.add_argument('--weighted-average', help='Compute the mAP using the weighted average of precisions among classes.', action='store_true')

    # Fit generator arguments
    parser.add_argument('--workers', help='Number of multiprocessing workers. To disable multiprocessing, set workers to 0', type=int, default=2)
    parser.add_argument('--max-queue-size', help='Queue length for multiprocessing workers in fit generator.', type=int, default=10)

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    backbone = models.backbone(args.backbone)

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    #keras.backend.tensorflow_backend.set_session(get_session())

    # create the generators
    #train_generator, validation_generator = create_generators(args, backbone.preprocess_image)
    dataset, num_classes, correspondences, obj_diameters, train_samples, intrinsics = create_generators(args, backbone.preprocess_image)

    # create the model
    if args.snapshot is not None:
        print('Loading model, this may take a second...')
        model            = models.load_model(args.snapshot, backbone_name=args.backbone)
        training_model   = model
        anchor_params    = None
        if args.config and 'anchor_parameters' in args.config:
            anchor_params = parse_anchor_parameters(args.config)
        #prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)
    else:
        weights = args.weights
        # default to imagenet if nothing else is specified
        #if weights is None and args.imagenet_weights:
        #    weights = backbone.download_imagenet()

        print('Creating model, this may take a second...')
        model, training_model = create_models(
            backbone_retinanet=backbone.model,
            #num_classes=train_generator.num_classes(),
            num_classes=num_classes,
            weights=weights,
            multi_gpu=0,
            freeze_backbone=args.freeze_backbone,
            lr=args.lr,
        )

    # print model summary
    print(model.summary())

    # create the callbacks
    callbacks = create_callbacks(
        model,
        training_model,
        args,
    )

    # Use multiprocessing if workers > 0
    if args.workers > 0:
        use_multiprocessing = True
    else:
        use_multiprocessing = False

    training_model.fit(
        x=dataset,
        steps_per_epoch=train_samples / args.batch_size,
        # steps_per_epoch=10,
        epochs=args.epochs,
        # epochs=1,
        verbose=1,
        callbacks=callbacks,
        workers=args.workers,
        use_multiprocessing=use_multiprocessing,
        max_queue_size=args.max_queue_size
    )

    '''
    # debugging
    from ..preprocessing.data_custom import CustomDataset
    #benchmark(
    #    LinemodDataset().prefetch(tf.data.AUTOTUNE)
    #)

    #dataset = CustomDataset(args.custom_path, 'train', batch_size=args.batch_size)
    #dataset = tf.data.Dataset.range(args.workers).interleave(
    #    lambda _: dataset,
        #num_parallel_calls=tf.data.experimental.AUTOTUNE
    #    num_parallel_calls=args.workers
    #)

    training_model.fit(
        x=train_generator,
        steps_per_epoch=train_generator.size() / args.batch_size,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
        workers=args.workers,
        use_multiprocessing=use_multiprocessing,
        max_queue_size=args.max_queue_size
    )
    '''

if __name__ == '__main__':
    main()
