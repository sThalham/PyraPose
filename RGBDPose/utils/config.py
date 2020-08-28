

import configparser
import numpy as np
import keras
from ..utils.anchors import AnchorParameters


def read_config_file(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)

    return config


def parse_anchor_parameters(config):
    ratios  = np.array(list(map(float, config['anchor_parameters']['ratios'].split(' '))), keras.backend.floatx())
    scales  = np.array(list(map(float, config['anchor_parameters']['scales'].split(' '))), keras.backend.floatx())
    sizes   = list(map(int, config['anchor_parameters']['sizes'].split(' ')))
    strides = list(map(int, config['anchor_parameters']['strides'].split(' ')))

    return AnchorParameters(sizes, strides, ratios, scales)
