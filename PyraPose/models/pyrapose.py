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
        self.loss_sum = keras.metrics.Sum()

    def train_step(self, data):
        self.loss_sum.reset_states()

        print(data[0])

        #if isinstance(data, tuple):
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
        with tf.GradientTape() as tape:
            predicts_gen = self.pyrapose.predict(x_st, batch_size=8, steps=1)

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

        source_patch = tf.concat([source_features_re, source_mask_re], axis=3)
        target_patch = tf.concat([target_features_re, target_mask_re], axis=3)
        disc_patch = tf.concat([target_patch, source_patch], axis=0)

        with tape:
            domain = self.discriminator(disc_patch)
            d_loss = self.loss_discriminator(labels, domain)

        grads_dis = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.omni_optimizer.apply_gradients(zip(grads_dis, self.discriminator.trainable_weights))

        source_features = tf.reshape(source_features, (batch_size, 4800, 256))
        filtered_predictions = [source_points, source_locations, source_mask, source_domain, source_features]

        loss_names = []
        losses = []
        loss_sum = 0
        accum_gradient = []

        '''
        accum_gradient = [tf.zeros_like(this_var) for this_var in train_vars]

        for j in enumerate(self.loss_generator)):
            sample = samples[j]
            with tf.GradientTape as tape:
                prediction = self.model(sample)
                loss_value = self.loss_function(y_true=labels[j], y_pred=prediction)
            total_loss += loss_value

            # get gradients of this tape
            gradients = tape.gradient(loss_value, train_vars)
            # Accumulate the gradients
            accum_gradient = [(acum_grad + grad) for acum_grad, grad in zip(accum_gradient, gradients)]

        # Now, after executing all the tapes you needed, we apply the optimization step
        # (but first we take the average of the gradients)
        accum_gradient = [this_grad / num_samples for this_grad in accum_gradient]
        # apply optimization step
        self.optimizer.apply_gradients(zip(accum_gradient, train_vars))
        '''

        train_vars = self.pyrapose.trainable_weights
        # Create empty gradient list (not a tf.Variable list)
        accum_gradient = [tf.zeros_like(this_var) for this_var in train_vars]

        with tf.GradientTape() as tape:
            #tape.watch(x_s)
            #predicts = self.pyrapose.predics(x_s, batch_size=None, steps=1)
            predicts = self.pyrapose(x_s)
            for ldx, loss_func in enumerate(self.loss_generator):
                loss_names.append(loss_func)
                y_now = tf.convert_to_tensor(y_s[ldx], dtype=tf.float32)
                loss = self.loss_generator[loss_func](y_now, predicts[ldx])
                loss_sum += loss
                grads_gen = tape.gradient(loss, self.pyrapose.trainable_weights)
                accum_gradient = [(acum_grad+grad) for acum_grad, grad in zip(accum_gradient, grads_gen)]
        print([var.name for var in tape.watched_variables()])
        self.omni_optimizer.apply_gradients(zip(accum_gradient, self.pyrapose.trainable_weights))

        #del tape

        return_losses = {}
        return_losses["loss"] = self.loss_sum
        for name, loss in zip(loss_names, losses):
            return_losses["name"] = loss

        #'3Dbox', 'cls', 'mask', 'domain', 'P3'
        #return {"d_loss": d_loss, "g_loss": g_loss}

        return return_losses

    '''
    def train_step(self, data):
        self.loss_sum.reset_states()

        #if isinstance(data, tuple):
        x_s = data[0]
        y_s = data[1]
        x_t = data[2]
        # Sample random points in the latent space
        batch_size = tf.shape(x_s)[0]
        valid = tf.ones((batch_size, 60, 80,  2))
        fake1 = tf.zeros((batch_size, 60, 80, 1))
        fake2 = tf.ones((batch_size, 60, 80, 1 ))
        fake = tf.concat([fake1, fake2], axis=3)
        labels = tf.concat([valid, fake], axis=0)

        # from Chollet
        # Add random noise to the labels - important trick!
        # labels += 0.05 * tf.random.uniform(tf.shape(labels))
        x_st = tf.concat([x_s, x_t], axis=0)
        with tf.GradientTape(persistent=True) as tape:
            predicts_gen = self.pyrapose.predict(x_st, batch_size=None, steps=1)

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

        source_patch = tf.concat([source_features, source_mask_re], axis=3)
        target_patch = tf.concat([target_features, target_mask_re], axis=3)
        disc_patch = tf.concat([target_patch, source_patch], axis=0)

        source_features = tf.reshape(source_features, (batch_size, 4800, 256))
        filtered_predictions = [source_points, source_locations, source_mask, source_domain, source_features]

        loss_names = []
        losses = []
        for ldx, loss_func in enumerate(self.loss_generator):
            with tape:
                loss_names.append(loss_func)
                loss = self.loss_generator[loss_func](y_s[ldx], filtered_predictions[ldx])
                losses.append(loss)
                #loss_sum += loss
                #losses = [self.loss_generator[loss_func](y_s[ldx], filtered_predictions[ldx])] + losses
                # We sum all losses together. (And calculate their mean value.)
                # You might want to split this if you are interested in the separate losses.
            grads_gen = tape.gradient(loss, self.pyrapose.trainable_weights)
            self.omni_optimizer.apply_gradients(zip(grads_gen, self.pyrapose.trainable_weights))
            self.loss_sum.update_state(loss)

        #del tape

        with tf.GradientTape() as tape:
            domain = self.discriminator(disc_patch)
            d_loss = self.loss_discriminator(labels, domain)
        grads_dis = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.omni_optimizer.apply_gradients(zip(grads_dis, self.discriminator.trainable_weights))

        return_losses = {}
        return_losses["loss"] = self.loss_sum
        for name, loss in zip(loss_names, losses):
            return_losses["name"] = loss

        #'3Dbox', 'cls', 'mask', 'domain', 'P3'
        #return {"d_loss": d_loss, "g_loss": g_loss}

        return return_losses
    '''

    def call(self, inputs, training=False):
        x = self.pyrapose(inputs['x'])
        if training:
            x = self.pyrapose(inputs['x'])
        return x

    #def call(self, inputs, training=False):
    #    x = self.dense1(inputs)
    #    if training:
    #        x = self.dropout(x, training=training)
    #    return self.dense2(x)


