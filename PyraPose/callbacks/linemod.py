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
from ..utils.linemod_eval import reannotate_linemod


class LinemodEval(keras.callbacks.Callback):

    def __init__(self, generator, train_generator, tensorboard=None, threshold=0.5):

        self.generator = generator
        self.train_generator = train_generator
        self.threshold = threshold
        self.tensorboard = tensorboard

        super(LinemodEval, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        reannotate_linemod(self.generator, self.model, self.threshold)

