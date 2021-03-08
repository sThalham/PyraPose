import tensorflow.keras as keras
import tensorflow as tf
from .. import initializers
from .. import layers
from ..utils.anchors import AnchorParameters
from . import assert_training_model


class CustomModel(tf.keras.Model):
    def __init__(self, pyrapose, discriminator):
        super(CustomModel, self).__init__()
        self.discriminator = discriminator
        self.pyrapose = pyrapose

    def compile(self, gen_optimizer, dis_optimizer, gen_loss, dis_loss, **kwargs):
        super(CustomModel, self).compile(**kwargs)
        self.optimizer_generator = gen_optimizer
        self.optimizer_discriminator = dis_optimizer
        self.loss_generator = gen_loss
        self.loss_discriminator = dis_loss
        #self.loss_sum = keras.metrics.Sum()

    '''
    @tf.function
    def train_step(self, data):
        x_s = data[0]['x']
        y_s = data[0]['y']
        x_t = data[0]['domain']
        loss_names = []
        losses = []
        loss_sum = 0

        with tf.GradientTape() as tape:
            # with tape:
            predicts = self.pyrapose(x_s)
            for ldx, loss_func in enumerate(self.loss_generator):
                loss_names.append(loss_func)
                y_now = tf.convert_to_tensor(y_s[ldx], dtype=tf.float32)
                loss = self.loss_generator[loss_func](y_now, predicts[ldx])
                # loss = self.loss_generator[loss_func](y_now, filtered_predictions[ldx])
                losses.append(loss)
                loss_sum += loss

        grads_gen = tape.gradient(loss_sum, self.pyrapose.trainable_weights)
        self.optimizer_generator.apply_gradients(zip(grads_gen, self.pyrapose.trainable_weights))

        return_losses = {}
        return_losses["loss"] = loss_sum
        for name, loss in zip(loss_names, losses):
            return_losses[name] = loss

        return return_losses
    '''

    @tf.function
    def train_step(self, data):

        #x_s = data[0]['x']
        #y_s = data[0]['y']
        #x_t = data[0]['domain']

        x_s = data[0]
        y_s = data[1]
        x_t = data[2]

        loss_names = []
        losses = []
        loss_sum = 0
        d_loss_sum = 0

        '''
        #print(keras.backend.int_shape(x_s))
        #print(keras.backend.int_shape(y_s))
        #print(keras.backend.int_shape(x_t))

        # Sample random points in the latent space
        batch_size = tf.shape(x_s)[0]
        # UDA mask
        #valid = tf.ones((batch_size, 60, 80,  2))
        #fake1 = tf.zeros((batch_size, 60, 80, 1))
        #fake2 = tf.ones((batch_size, 60, 80, 1))
        # UDA pose
        validP3 = tf.ones((batch_size, 60, 80, 2))
        fake1P3 = tf.zeros((batch_size, 60, 80, 1))
        fake2P3 = tf.ones((batch_size, 60, 80, 1))
        fakeP3 = tf.concat([fake1P3, fake2P3], axis=3)
        labelsP3 = tf.concat([validP3, fakeP3], axis=0)

        validP4 = tf.ones((batch_size, 30, 40, 2))
        fake1P4 = tf.zeros((batch_size, 30, 40, 1))
        fake2P4 = tf.ones((batch_size, 30, 40, 1))
        fakeP4 = tf.concat([fake1P4, fake2P4], axis=3)
        labelsP4 = tf.concat([validP4, fakeP4], axis=0)

        validP5 = tf.ones((batch_size, 15, 20, 2))
        fake1P5 = tf.zeros((batch_size, 15, 20, 1))
        fake2P5 = tf.ones((batch_size, 15, 20, 1))
        fakeP5 = tf.concat([fake1P5, fake2P5], axis=3)
        labelsP5 = tf.concat([validP5, fakeP5], axis=0)

        disc_labels = [labelsP3, labelsP4, labelsP5]

        # from Chollet
        # Add random noise to the labels - important trick!
        # labels += 0.05 * tf.random.uniform(tf.shape(labels))
        x_st = tf.concat([x_s, x_t], axis=0)
        with tf.GradientTape() as tape:
            predicts_gen = self.pyrapose(x_st)

        points = predicts_gen[0]
        locations = predicts_gen[1]
        masks = predicts_gen[2]
        domain = predicts_gen[3]
        featuresP3 = predicts_gen[4]
        featuresP4 = predicts_gen[5]
        featuresP5 = predicts_gen[6]
        source_points, target_points = tf.split(points, num_or_size_splits=2, axis=0)
        source_locations, target_locations = tf.split(locations, num_or_size_splits=2, axis=0)
        source_mask, target_mask = tf.split(masks, num_or_size_splits=2, axis=0)
        source_domain, target_domain = tf.split(domain, num_or_size_splits=2, axis=0)
        source_featuresP3, target_featuresP3 = tf.split(featuresP3, num_or_size_splits=2, axis=0)
        source_featuresP4, target_featuresP4 = tf.split(featuresP4, num_or_size_splits=2, axis=0)
        source_featuresP5, target_featuresP5 = tf.split(featuresP5, num_or_size_splits=2, axis=0)
        cls_shape = tf.shape(source_mask)[2]

        # discriminator for feature map conditioned on predicted pose
        source_pointsP3, source_pointsP4, source_pointsP5 = tf.split(source_points, num_or_size_splits=[43200, 10800, 2700], axis=1)
        source_pointsP3_re = tf.reshape(source_pointsP3, (batch_size, 60, 80, 9 * 16))
        source_pointsP4_re = tf.reshape(source_pointsP4, (batch_size, 30, 40, 9 * 16))
        source_pointsP5_re = tf.reshape(source_pointsP5, (batch_size, 15, 20, 9 * 16))

        target_pointsP3, target_pointsP4, target_pointsP5 = tf.split(source_points,
                                                                     num_or_size_splits=[43200, 10800, 2700], axis=1)
        target_pointsP3_re = tf.reshape(target_pointsP3, (batch_size, 60, 80, 9 * 16))
        target_pointsP4_re = tf.reshape(target_pointsP4, (batch_size, 30, 40, 9 * 16))
        target_pointsP5_re = tf.reshape(target_pointsP5, (batch_size, 15, 20, 9 * 16))

        source_featuresP3_re = tf.reshape(source_featuresP3, (batch_size, 60, 80, 256))
        source_featuresP4_re = tf.reshape(source_featuresP4, (batch_size, 30, 40, 256))
        source_featuresP5_re = tf.reshape(source_featuresP5, (batch_size, 15, 20, 256))

        target_featuresP3_re = tf.reshape(target_featuresP3, (batch_size, 60, 80, 256))
        target_featuresP4_re = tf.reshape(target_featuresP4, (batch_size, 30, 40, 256))
        target_featuresP5_re = tf.reshape(target_featuresP5, (batch_size, 15, 20, 256))

        source_patchP3 = tf.concat([source_featuresP3_re, source_pointsP3_re], axis=3)
        target_patchP3 = tf.concat([target_featuresP3_re, target_pointsP3_re], axis=3)
        disc_patchP3 = tf.concat([target_patchP3, source_patchP3], axis=0)

        source_patchP4 = tf.concat([source_featuresP4_re, source_pointsP4_re], axis=3)
        target_patchP4 = tf.concat([target_featuresP4_re, target_pointsP4_re], axis=3)
        disc_patchP4 = tf.concat([target_patchP4, source_patchP4], axis=0)

        source_patchP5 = tf.concat([source_featuresP5_re, source_pointsP5_re], axis=3)
        target_patchP5 = tf.concat([target_featuresP5_re, target_pointsP5_re], axis=3)
        disc_patchP5 = tf.concat([target_patchP5, source_patchP5], axis=0)

        disc_patches = [disc_patchP3, disc_patchP4, disc_patchP5]

        ## discriminator conditioned on feature space alone
        #disc_patch = tf.concat([target_features_re, source_features_re], axis=0)

        # discriminator for feature map conditioned on predicted mask
        #source_mask_re = tf.reshape(source_mask, (batch_size, 60, 80, cls_shape))
        #target_mask_re = tf.reshape(target_mask, (batch_size, 60, 80, cls_shape))
        #source_features_re = tf.reshape(source_features, (batch_size, 60, 80, 256))
        #target_features_re = tf.reshape(target_features, (batch_size, 60, 80, 256))
        #source_patch = tf.concat([source_features_re, source_mask_re], axis=3)
        #target_patch = tf.concat([target_features_re, target_mask_re], axis=3)
        #disc_patch = tf.concat([target_patch, source_patch], axis=0)

        #with tape:
        #    domain = self.discriminator(disc_patch)
        #    d_loss = self.loss_discriminator(labels, domain)

        with tape:
            for ddx, disc_map in enumerate(disc_patches):
                domain = self.discriminator(disc_map)
                loss = self.loss_discriminator(disc_labels[ddx], domain)
                d_loss_sum += loss

        grads_dis = tape.gradient(d_loss_sum, self.discriminator.trainable_weights)
        self.optimizer_discriminator.apply_gradients(zip(grads_dis, self.discriminator.trainable_weights))
        '''

        #filtered_predictions = [source_points, source_locations, source_mask, source_domain]

        with tf.GradientTape() as tape:
        #with tape:
            predicts = self.pyrapose(x_s)
            for ldx, loss_func in enumerate(self.loss_generator):
                loss_names.append(loss_func)
                y_now = tf.convert_to_tensor(y_s[ldx], dtype=tf.float32)
                loss = self.loss_generator[loss_func](y_now, predicts[ldx])
                #loss = self.loss_generator[loss_func](y_now, filtered_predictions[ldx])
                losses.append(loss)
                loss_sum += loss

        grads_gen = tape.gradient(loss_sum, self.pyrapose.trainable_weights)
        self.optimizer_generator.apply_gradients(zip(grads_gen, self.pyrapose.trainable_weights))

        #del tape

        return_losses = {}
        return_losses["loss"] = loss_sum
        for name, loss in zip(loss_names, losses):
            return_losses[name] = loss

        return return_losses

    '''
    @tf.function
    def train_step(self, data):
        x_s = data[0]['x']
        y_s = data[0]['y']
        x_t = data[0]['domain']
        # Sample random points in the latent space
        batch_size = tf.shape(x_s)[0]
        valid = tf.ones((batch_size, 60, 80,  2))
        fake1 = tf.zeros((batch_size, 60, 80, 1))
        fake2 = tf.ones((batch_size, 60, 80, 1))
        fake = tf.concat([fake1, fake2], axis=3)
        labels = tf.concat([valid, fake], axis=0)

        # from Chollet
        # Add random noise to the labels - important trick!
        # labels += 0.05 * tf.random.uniform(tf.shape(labels))
        x_st = tf.concat([x_s, x_t], axis=0)
        with tf.GradientTape(persistent=True) as tape:
            predicts_gen = self.pyrapose(x_st)

        points = predicts_gen[0]
        locations = predicts_gen[1]
        masks = predicts_gen[2]
        domain = predicts_gen[3]
        features = predicts_gen[4]
        source_points, target_points = tf.split(points, num_or_size_splits=2, axis=0)
        source_locations, target_locations = tf.split(locations, num_or_size_splits=2, axis=0)
        source_mask, target_mask = tf.split(masks, num_or_size_splits=2, axis=0)
        source_domain, target_domain = tf.split(domain, num_or_size_splits=2, axis=0)
        source_features, target_features = tf.split(features, num_or_size_splits=2, axis=0)
        cls_shape = tf.shape(source_mask)[2]

        source_mask_re = tf.reshape(source_mask, (batch_size, 60, 80, cls_shape))
        target_mask_re = tf.reshape(target_mask, (batch_size, 60, 80, cls_shape))
        source_features_re = tf.reshape(source_features, (batch_size, 60, 80, 256))
        target_features_re = tf.reshape(target_features, (batch_size, 60, 80, 256))

        # discriminator conditioned on feature space
        disc_patch = tf.concat([target_features_re, source_features_re], axis=0)

        # discriminator for feature map conditioned on predicted mask
        #source_patch = tf.concat([source_features_re, source_mask_re], axis=3)
        #target_patch = tf.concat([target_features_re, target_mask_re], axis=3)
        #disc_patch = tf.concat([target_patch, source_patch], axis=0)

        with tape:
            domain = self.discriminator(disc_patch)
            d_loss = self.loss_discriminator(labels, domain)

        grads_dis = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.optimizer_discriminator.apply_gradients(zip(grads_dis, self.discriminator.trainable_weights))

        source_features = tf.reshape(source_features, (batch_size, 4800, 256))
        filtered_predictions = [source_points, source_locations, source_mask, source_domain, source_features]

        loss_names = []
        losses = []
        loss_sum = 0

        #with tf.GradientTape() as tape:
        with tape:
            #tape.watch(x_s)
            #predicts = self.pyrapose.predics(x_s, batch_size=None, steps=1)
            #predicts = self.pyrapose(x_s)
            for ldx, loss_func in enumerate(self.loss_generator):
                loss_names.append(loss_func)
                y_now = tf.convert_to_tensor(y_s[ldx], dtype=tf.float32)
                loss = self.loss_generator[loss_func](y_now, filtered_predictions[ldx])
                losses.append(loss)
                loss_sum += loss
                #grads_gen = tape.gradient(loss, self.pyrapose.trainable_weights)
                #accum_gradient = [(acum_grad+grad) for acum_grad, grad in zip(accum_gradient, grads_gen)]
        grads_gen = tape.gradient(loss_sum, self.pyrapose.trainable_weights)
        self.optimizer_generator.apply_gradients(zip(grads_gen, self.pyrapose.trainable_weights))

        #del tape

        return_losses = {}
        return_losses["loss"] = loss_sum
        for name, loss in zip(loss_names, losses):
            return_losses[name] = loss

        return return_losses
    '''

    def call(self, inputs, training=False):
        x = self.pyrapose(inputs[0])
        if training:
            x = self.pyrapose(inputs[0])
        return x

    #def call(self, inputs, training=False):
    #    x = self.pyrapose(inputs['x'])
    #    if training:
    #        x = self.pyrapose(inputs['x'])
    #    return x



