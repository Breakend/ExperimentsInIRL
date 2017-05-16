import tensorflow as tf
import numpy as np
from .nn_utils import *
import matplotlib.pyplot as plt
import sandbox.rocky.tf.core.layers as L
import math
import itertools

# This is mostly taken and modified from: https://github.com/Breakend/third_person_im/blob/master/sandbox/bradly/third_person/discriminators/discriminator.py
# TODO: redo this using the layers and networks from rllab https://github.com/Breakend/rllab/blob/master/sandbox/rocky/tf/core/network.py. that way we can match
#       the policy and discriminator networks

class Discriminator(object):
    def __init__(self, input_dim, output_dim_class=2, output_dim_dom=None, tf_sess=None, config=None):
        self.input_dim = input_dim
        self.output_dim_class = output_dim_class
        self.output_dim_dom = output_dim_dom
        self.learning_rate = 0.001
        self.loss = None
        self.discrimination_logits = None
        self.optimizer = None
        self.nn_input = None
        self.class_target = None
        self.sess = tf_sess
        self.config = config
        self.train_step = tf.placeholder(tf.float32, shape=(), name="train_step")
        self.actual_train_step = 0
        self.decaying_noise = tf.train.exponential_decay(.1, self.train_step, 5, 0.95, staircase=True)
        self.decaying_reward_bonus = tf.train.exponential_decay(0.05, self.train_step, 5, 0.95, staircase=True)
        self.decaying_dropout = tf.train.exponential_decay(0.4, self.train_step, 5, 0.9, staircase=True)

    def init_tf(self):
        # Hack to only initialize unitialized variables
        if self.sess is None:
            self.sess = tf.Session()
        initialize_uninitialized(self.sess)
        # init = tf.global_variables_initializer()
        # self.sess.run(init)

    def make_network(self, dim_input, output_dim_class, output_dim_dom):
        raise NotImplementedError

    def compute_pos_weight(self, targets_batch):
        summs = np.sum(targets_batch, axis=0)
        pos_samples = summs
        neg_samples = targets_batch.shape[0] - pos_samples
        if pos_samples <= 0:
            ratio = 0.0
        else:
            ratio = (np.float32(neg_samples)/np.float32(pos_samples))[0]

        return ratio

    def train(self, data_batch, targets_batch):
        cost = self.sess.run([self.optimizer, self.loss], feed_dict={
                                                                    self.train_step : self.actual_train_step,
                                                                    #  self.pos_weighting: self.compute_pos_weight(targets_batch),
                                                                     self.nn_input: data_batch,
                                                                     self.class_target: targets_batch})[1]
        return cost

    def eval(self, data, softmax=True):
        logits = self.discrimination_logits

        if softmax is True:
            if not self.config["short_run_is_bad"]:
                if self.config["add_decaying_reward_bonus"]:
                    raise Exception("Decaying bonus for short_run_is_bad=False is not implemented yet")
                logits = tf.nn.sigmoid(logits) - 1.0
            else:
                logits = tf.nn.sigmoid(logits)
                if self.config["add_decaying_reward_bonus"]:
                    logits += self.decaying_reward_bonus * (tf.exp(logits))

        if self.config["use_gaussian_noise_on_eval"]:
             logits = gaussian_noise_layer(logits, self.decaying_noise)

        log_prob = self.sess.run([logits], feed_dict={self.train_step : self.actual_train_step, self.nn_input: data})[0]
        return log_prob

    @staticmethod
    def init_weights(shape, name=None):
        return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)

    @staticmethod
    def init_bias(shape, name=None):
        return tf.Variable(tf.zeros(shape, dtype='float'), name=name)

    @staticmethod
    def conv2d(img, w, b):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'), b))

    @staticmethod
    def conv1d(vec, w, b):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv1d(vec, w, stride=1, padding='SAME'), b))

    @staticmethod
    def max_pool(img, k):
        return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def get_mlp_layers(self, mlp_input, number_layers, dimension_hidden, name_prefix='', dropout=None):
        """compute MLP with specified number of layers.
            math: sigma(Wx + b)
            for each layer, where sigma is by default relu"""
        cur_top = mlp_input
        for layer_step in range(0, number_layers):
            in_shape = cur_top.get_shape().dims[1].value
            cur_weight = self.init_weights([in_shape, dimension_hidden[layer_step]],
                                           name='w_' + name_prefix + str(layer_step))
            cur_bias = self.init_bias([dimension_hidden[layer_step]],
                                      name='b_' + name_prefix + str(layer_step))
            if layer_step != number_layers-1:  # final layer has no RELU
                cur_top = tf.nn.relu(tf.matmul(cur_top, cur_weight) + cur_bias)
                if self.config["use_decaying_dropout"]:
                    print("Using dropout")
                    pred = tf.nn.dropout(cur_top, (1.0 - self.decaying_dropout))
            else:
                cur_top = tf.matmul(cur_top, cur_weight) + cur_bias
        return cur_top

    @staticmethod
    def get_xavier_weights(filter_shape, poolsize=(2, 2)):
        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
               np.prod(poolsize))

        low = -4*np.sqrt(6.0/(fan_in + fan_out)) # use 4 for sigmoid, 1 for tanh activation
        high = 4*np.sqrt(6.0/(fan_in + fan_out))
        return tf.Variable(tf.random_uniform(filter_shape, minval=low, maxval=high, dtype=tf.float32))
        #return tf.Variable(tf.random_normal(filter_shape, mean=0.01, stddev=0.001, dtype=tf.float32))

    def get_loss_layer(self, pred, target_output):
        # http://stackoverflow.com/questions/40698709/tensorflow-interpretation-of-weight-in-weighted-cross-entropy
        # self.pos_weighting = tf.placeholder('float', [], name='pos_weighting')

        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=target_output)#, pos_weight=self.pos_weighting)

        if "entropy_penalty" in self.config and self.config["entropy_penalty"] > 0.0:
            cross_entropy -= float(self.config["entropy_penalty"])*logit_bernoulli_entropy(pred)
        cost = tf.reduce_sum(cross_entropy, axis=0)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        return cost, optimizer


class ConvStateBasedDiscriminator(Discriminator):
    """ A state based descriminator, assuming a state vector """

    def __init__(self, input_dim, dim_output=1, config=None, nn_input=None, target=None, tf_sess=None, scope="external"):
        with tf.variable_scope(scope):
            super(ConvStateBasedDiscriminator, self).__init__(input_dim, config=config, tf_sess=tf_sess)
            if config["img_input"]:
                print("Using image network for discriminator...")
                self.make_network_image(dim_input=input_dim, dim_output=dim_output, nn_input=nn_input, target=target)
            else:
                print("Using state network for discriminator...")
                self.make_network(dim_input=input_dim, dim_output=dim_output, nn_input=nn_input, target=target)

            self.init_tf()

    def get_lab_accuracy(self, data, class_labels):
        return self.sess.run([self.label_accuracy], feed_dict={self.nn_input: data,
                                                               self.class_target: class_labels})[0]

    def get_mse(self, data, class_labels):
        return self.sess.run([self.mse], feed_dict={self.nn_input: data,
                                                               self.class_target: class_labels})[0]

    def get_lab_precision(self, data, class_labels):
        return self.sess.run([self.label_precision], feed_dict={self.nn_input: data,
                                                               self.class_target: class_labels})[0]

    def get_lab_recall(self, data, class_labels):
        return self.sess.run([self.label_recall], feed_dict={self.nn_input: data,
                                                               self.class_target: class_labels})[0]


    def make_network_image(self, dim_input, dim_output, nn_input=None, target=None):
        """
        An example a network in tf that has both state and image inputs.
        Args:
            dim_input: Dimensionality of input. expecting 2d tuple (num_frames x num_batches)
            dim_output: Dimensionality of the output.
            batch_size: Batch size.
            network_config: dictionary of network structure parameters
        Returns:
            A tfMap object that stores inputs, outputs, and scalar loss.
        """
        if dim_input[0] != 1:
            raise Exception("Currently don't support concatenating timesteps for images")

        n_mlp_layers = 2
        layer_size = 128
        dim_hidden = (n_mlp_layers - 1) * [layer_size]
        dim_hidden.append(dim_output)
        pool_size = 2
        filter_size = 3

        num_filters = [5, 5]

        #TODO: don't do this grossness
        if nn_input is None:
            nn_input, _ = self.get_input_layer_image(dim_input[1], dim_output)

        if target is None:
            _, target = self.get_input_layer_image(dim_input[1], dim_output)

        conv_filters = [5, 5]
        conv_filter_sizes = [3, 3]
        conv_pads = ['SAME', 'SAME']
        max_pool_sizes = [2, 2]
        conv_strides = [1, 1]
        hidden_sizes = [100, 100]
        hidden_nonlinearity = tf.nn.relu
        output_nonlinearity = None

        l_in = L.InputLayer(shape=tuple(nn_input.get_shape().as_list()), input_var=nn_input)

        l_hid = L.reshape(l_in, ([0],) + dim_input[1], name="reshape_input")

        for idx, conv_filter, filter_size, stride, pad, max_pool_size in zip(
                range(len(conv_filters)),
                conv_filters,
                conv_filter_sizes,
                conv_strides,
                conv_pads,
                max_pool_sizes
        ):
            l_hid = L.Conv2DLayer(
                l_hid,
                num_filters=conv_filter,
                filter_size=filter_size,
                stride=(stride, stride),
                pad=pad,
                nonlinearity=hidden_nonlinearity,
                name="conv_hidden_%d" % idx,
            )
            if max_pool_size is not None:
                l_hid = L.Pool2DLayer(l_hid, max_pool_size, pad="SAME")

        l_hid = L.flatten(l_hid, name="conv_flatten")

        for idx, hidden_size in enumerate(hidden_sizes):
            l_hid = L.DenseLayer(
                l_hid,
                num_units=hidden_size,
                nonlinearity=hidden_nonlinearity,
                name="hidden_%d" % idx,
            )

        fc_output = L.get_output(L.DenseLayer(
            l_hid,
            num_units=dim_output,
            nonlinearity=output_nonlinearity,
            name="output",
        ))

        loss, optimizer = self.get_loss_layer(pred=fc_output, target_output=target)

        self.class_target = target
        self.nn_input = nn_input
        self.discrimination_logits = fc_output
        self.optimizer = optimizer
        self.loss = loss
        label_accuracy = tf.equal(tf.round(tf.nn.sigmoid(self.discrimination_logits)), tf.round(self.class_target))
        self.label_accuracy = tf.reduce_mean(tf.cast(label_accuracy, tf.float32))
        self.mse = tf.reduce_mean(tf.nn.l2_loss(tf.nn.sigmoid(self.discrimination_logits) - self.class_target))
        ones = tf.ones_like(self.class_target)

        true_positives = tf.round(tf.nn.sigmoid(self.discrimination_logits)) * tf.round(self.class_target)
        predicted_positives = tf.round(tf.nn.sigmoid(self.discrimination_logits))

        false_negatives = tf.logical_not(tf.logical_xor(tf.equal(tf.round(tf.nn.sigmoid(self.discrimination_logits)), ones), tf.equal(tf.round(self.class_target), ones)))

        self.label_precision = tf.reduce_sum(tf.cast(true_positives, tf.float32)) / tf.reduce_sum(tf.cast(predicted_positives, tf.float32))
        self.label_recall = tf.reduce_sum(tf.cast(true_positives, tf.float32)) / (tf.reduce_sum(tf.cast(true_positives, tf.float32)) + tf.reduce_sum(tf.cast(false_negatives, tf.float32)))

    def make_network(self, dim_input, dim_output, nn_input=None, target=None):
        """
        An example a network in tf that has both state and image inputs.
        Args:
            dim_input: Dimensionality of input. expecting 2d tuple (num_frames x num_batches)
            dim_output: Dimensionality of the output.
            batch_size: Batch size.
            network_config: dictionary of network structure parameters
        Returns:
            A tfMap object that stores inputs, outputs, and scalar loss.
        """
        n_mlp_layers = 2
        layer_size = 128
        dim_hidden = (n_mlp_layers - 1) * [layer_size]
        dim_hidden.append(dim_output)
        pool_size = 2
        filter_size = 3

        num_filters = [5, 5]

        if nn_input is None:
            #TODO: don't do this grossness, just create variables the normal way
            nn_input, _ = self.get_input_layer(dim_input[0], dim_input[1], dim_output)

        if target is None:
            _, target = self.get_input_layer(dim_input[0], dim_input[1], dim_output)

        conv_out_size = int(dim_input[0] * num_filters[1])

        conv_layer_0 = tf.layers.conv1d(nn_input, filters=5, kernel_size=3, strides=1, padding='same')

        conv_layer_1 = tf.layers.conv1d(conv_layer_0, filters=5, kernel_size=3, strides=1, padding='same')

        conv_out_flat = tf.reshape(conv_layer_1, [-1, conv_out_size])

        fc_output = self.get_mlp_layers(conv_out_flat, n_mlp_layers, dim_hidden, dropout=None)

        loss, optimizer = self.get_loss_layer(pred=fc_output, target_output=target)

        self.class_target = target
        self.nn_input = nn_input
        self.discrimination_logits = fc_output
        self.optimizer = optimizer
        self.loss = loss

        label_accuracy = tf.equal(tf.round(tf.nn.sigmoid(self.discrimination_logits)), tf.round(self.class_target))

        self.label_accuracy = tf.reduce_mean(tf.cast(label_accuracy, tf.float32))
        self.mse = tf.reduce_mean(tf.nn.l2_loss(tf.nn.sigmoid(self.discrimination_logits) - self.class_target))

        ones = tf.ones_like(self.class_target)

        true_positives = tf.round(tf.nn.sigmoid(self.discrimination_logits)) * tf.round(self.class_target)
        predicted_positives = tf.round(tf.nn.sigmoid(self.discrimination_logits))

        false_negatives = tf.logical_not(tf.logical_xor(tf.equal(tf.round(tf.nn.sigmoid(self.discrimination_logits)), ones), tf.equal(tf.round(self.class_target), ones)))

        self.label_precision = tf.reduce_sum(tf.cast(true_positives, tf.float32)) / tf.reduce_sum(tf.cast(predicted_positives, tf.float32))
        self.label_recall = tf.reduce_sum(tf.cast(true_positives, tf.float32)) / (tf.reduce_sum(tf.cast(true_positives, tf.float32)) + tf.reduce_sum(tf.cast(false_negatives, tf.float32)))

    @staticmethod
    def get_input_layer(num_frames, state_size, dim_output=1):
        """produce the placeholder inputs that are used to run ops forward and backwards.
        net_input: usually an observation.
        action: mu, the ground truth actions we're trying to learn.
        precision: precision matrix used to commpute loss."""
        net_input = tf.placeholder('float', [None, num_frames, state_size], name='nn_input')
        targets = tf.placeholder('float', [None, dim_output], name='targets')
        return net_input, targets

    @staticmethod
    def get_input_layer_image(state_size, dim_output=1):
        """produce the placeholder inputs that are used to run ops forward and backwards.
        net_input: usually an observation.
        action: mu, the ground truth actions we're trying to learn.
        precision: precision matrix used to commpute loss."""
        im_w, im_h, channels = state_size
        #TODO: right now hack cuz RLLAb flattens everything so we have to unflatten
        net_input = tf.placeholder('float', [None, 1, im_w * im_h * channels], name='nn_input')
        targets = tf.placeholder('float', [None, dim_output], name='targets')
        return net_input, targets

class ConvStateBasedDiscriminatorWithOptions(Discriminator):
    """ A state based descriminator, assuming a state vector """

    def __init__(self, input_dim, num_options=4, mixtures=True, config=None):
        super(ConvStateBasedDiscriminatorWithOptions, self).__init__(input_dim, config=config)
        self.num_options = num_options
        # TODO: move the other args into the config
        self.config = config
        self.use_l1_loss = False
        self.input_dim = input_dim
        self.dim_output = 1
        self.mixtures = mixtures
        self.make_network(dim_input=input_dim, dim_output=self.dim_output, mixtures=mixtures)
        self.init_tf()

    def get_lab_accuracy(self, data, class_labels):
        return self.sess.run([self.label_accuracy], feed_dict={self.nn_input: data,
                                                               self.class_target: class_labels})[0]

    def get_mse(self, data, class_labels):
        return self.sess.run([self.mse], feed_dict={self.nn_input: data,
                                                               self.class_target: class_labels})[0]

    def get_lab_precision(self, data, class_labels):
        return self.sess.run([self.label_precision], feed_dict={self.nn_input: data,
                                                               self.class_target: class_labels})[0]

    def get_lab_recall(self, data, class_labels):
        return self.sess.run([self.label_recall], feed_dict={self.nn_input: data,
                                                               self.class_target: class_labels})[0]

    def get_loss_layer(self, pred, target_output):
        # http://stackoverflow.com/questions/40698709/tensorflow-interpretation-of-weight-in-weighted-cross-entropy
        # self.pos_weighting = tf.placeholder('float', [], name='pos_weighting')

        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=target_output)#, pos_weight=self.pos_weighting)

        if "entropy_penalty" in self.config and self.config["entropy_penalty"] > 0.0:
            cross_entropy -= float(self.config["entropy_penalty"])*logit_bernoulli_entropy(pred)
        cost = tf.reduce_sum(cross_entropy, axis=0)

        if self.config["use_cv_penalty"]:
            print("Using CV penalty")
            mean, var = tf.nn.moments(self.termination_importance_values, axes=[0])
            cv = var/mean
            importance_weight = self.config["importance_weights"]
            cost += importance_weight*tf.nn.l2_loss(cv)

        if self.config["use_mutual_info_penalty_nn_paper"]:
            print("Using Mutual info penalty")
            combos = [item for idx, item in enumerate(itertools.combinations(range(len(self.discriminator_options)), 2))]
            mi = tf.Variable(0, dtype=tf.float32)
            for (i,j) in combos:

                # cond_ent = tf.reduce_mean(-tf.reduce_sum(tf.multiply(tf.log(tf.sigmoid(self.discriminator_options[i].discrimination_logits) + TINY), tf.sigmoid(self.discriminator_options[j].discrimination_logits)), 1))
                # ent = tf.reduce_mean(-tf.reduce_sum(tf.multiply(tf.log(tf.sigmoid(self.discriminator_options[i].discrimination_logits) + TINY), tf.sigmoid(self.discriminator_options[i].discrimination_logits)), 1))
                # # ent = tf.reduce_mean(-tf.reduce_sum(tf.multiply(discriminator_options[i].discrimination_logits, discriminator_options[i].discrimination_logits), 1))
                # mi += (cond_ent + ent)
                # As defined in equation (4) @ https://www.cs.bham.ac.uk/~xin/papers/IJHIS-03-009-yao-liu.pdf
                mean_i, var_i = tf.nn.moments(self.discriminator_options[i].discrimination_logits, axes=[0])
                mean_j, var_j = tf.nn.moments(self.discriminator_options[j].discrimination_logits, axes=[0])
                mean_ij, var_ij = tf.nn.moments(tf.multiply(self.discriminator_options[i].discrimination_logits,
                                                        self.discriminator_options[j].discrimination_logits), axes=[0])
                # TODO: ^ Does this make sense mathematically ??
                corr_numerator = mean_ij-mean_i*mean_j
                corr_denominator = tf.square(var_i)*tf.square(var_j) + TINY
                corr_coeff = corr_numerator/corr_denominator
                mutual_info = -(1/2.0) * log10(1-tf.square(corr_coeff))

                mi += mutual_info
            importance_weight = self.config["importance_weights"]
            cost += (importance_weight)*tf.nn.l2_loss(mi)
        elif self.config["use_mutual_info_penalty_infogan"]:
            print("Using Mutual info penalty")
            combos = [item for idx, item in enumerate(itertools.combinations(range(len(self.discriminator_options)), 2))]
            mi = tf.Variable(0, dtype=tf.float32)
            for (i,j) in combos:
                # http://wiseodd.github.io/techblog/2017/01/29/infogan/
                cond_ent = tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.log(tf.sigmoid(self.discriminator_options[i].discrimination_logits) + TINY), tf.sigmoid(self.discriminator_options[j].discrimination_logits)), 1))
                ent = tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.log(tf.sigmoid(self.discriminator_options[i].discrimination_logits) + TINY), tf.sigmoid(self.discriminator_options[i].discrimination_logits)), 1))
                mi += (cond_ent + ent)

            importance_weight = self.config["importance_weights"]
            cost += (importance_weight)*tf.nn.l2_loss(mi)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        return cost, optimizer

    def _remake_network_from_disc_options(self, discriminator_options, stop_gradients = False, num_extra_options = 0):
        """
        An example a network in tf that has both state and image inputs.
        Args:
            dim_input: Dimensionality of input. expecting 2d tuple (num_frames x num_batches)
            dim_output: Dimensionality of the output.
            batch_size: Batch size.
            network_config: dictionary of network structure parameters
        Returns:
            A tfMap object that stores inputs, outputs, and scalar loss.
        """
        #TODO: a better formulation is to have terminations be a latent variable that somehow sums to 1
        #TODO: can we combined these mixtures in interesting ways to train each other
        self.actual_train_step = 0

        if stop_gradients:
            for x in discriminator_options:
                x.discrimination_logits = tf.stop_gradient(x.discrimination_logits)

        if num_extra_options > 0:
            self.num_options += num_extra_options
            for i in range(num_extra_options):
                discriminator_options.append(ConvStateBasedDiscriminator(self.input_dim, nn_input=self.nn_input, dim_output=self.dim_output, scope="extraoption%d"%i, config=self.config))

        self.discriminator_options = discriminator_options

        # import pdb; pdb.set_trace()
        with tf.variable_scope("second"):
            termination_options = ConvStateBasedDiscriminator(self.input_dim, nn_input=self.nn_input, dim_output=self.num_options, config=self.config)

        #TODO: make this configurable
        self.termination_softmax_logits = termination_options.discrimination_logits

        if not self.mixtures:
            # TODO: then it's options, this flag is gross, change it
            # import pdb; pdb.set_trace()
            k = 1
            indices = tf.nn.top_k(self.termination_softmax_logits, k=k).indices
            vec = tf.zeros( tf.shape(self.termination_softmax_logits))
            for k in range(k):
                vec += tf.reshape(tf.one_hot(indices[:,k], tf.shape(self.termination_softmax_logits)[1]), tf.shape(self.termination_softmax_logits))
            v = tf.cast(vec == 0, vec.dtype) * -math.inf
            self.termination_softmax_logits = vec * v
            # self.termination_softmax_logits = self.termination_softmax_logits *  tf.reshape(tf.one_hot(tf.nn.top_k(self.termination_softmax_logits).indices, tf.shape(self.termination_softmax_logits)[1]), tf.shape(self.termination_softmax_logits))

            # tf.reshape(tf.one_hot(tf.nn.top_k(self.termination_softmax_logits).indices, tf.shape(self.termination_softmax_logits)[1]), tf.shape(self.termination_softmax_logits))

        self.termination_softmax_logits = tf.nn.softmax(termination_options.discrimination_logits)

        #TODO: add gaussian noise and top-k? https://arxiv.org/pdf/1701.06538.pdf
        # import pdb; pdb.set_trace()
        combined_options = tf.concat([x.discrimination_logits for x in discriminator_options], axis=1)
        #combined_options should have shape (batch_size x num_options )
        self.discrimination_logits = tf.reshape(tf.reduce_sum(combined_options * self.termination_softmax_logits, axis=1), [-1, 1])
        #self.discrimination_logits should have shape (batch_size x 1)

        # tf.add_n([tf.transpose(tf.multiply(tf.transpose(x.discrimination_logits), self.termination_softmax_logits[:,i])) for i, x in enumerate(discriminator_options)]) + regularization_penalty

        #TODO: what works better, this loss function or each individual loss function
        # add importance to loss
        self.termination_importance_values = tf.reduce_sum(self.termination_softmax_logits, axis=0)

        self.loss, self.optimizer = self.get_loss_layer(pred=self.discrimination_logits, target_output=self.class_target)

        label_accuracy = tf.equal(tf.round(tf.nn.sigmoid(self.discrimination_logits)), tf.round(self.class_target))

        self.label_accuracy = tf.reduce_mean(tf.cast(label_accuracy, tf.float32))

        self.mse = tf.reduce_mean(tf.nn.l2_loss(tf.nn.sigmoid(self.discrimination_logits) - self.class_target))

        ones = tf.ones_like(self.class_target)

        true_positives = tf.round(tf.nn.sigmoid(self.discrimination_logits)) * tf.round(self.class_target)
        predicted_positives = tf.round(tf.nn.sigmoid(self.discrimination_logits))

        false_negatives = tf.logical_not(tf.logical_xor(tf.equal(tf.round(tf.nn.sigmoid(self.discrimination_logits)), ones), tf.equal(tf.round(self.class_target), ones)))

        self.label_precision = tf.reduce_sum(tf.cast(true_positives, tf.float32)) / tf.reduce_sum(tf.cast(predicted_positives, tf.float32))
        self.label_recall = tf.reduce_sum(tf.cast(true_positives, tf.float32)) / (tf.reduce_sum(tf.cast(true_positives, tf.float32)) + tf.reduce_sum(tf.cast(false_negatives, tf.float32)))

        # Why do this? apply these termination functions as termination functions for policy options?
        # use TRPO to train N different policies with the termination function taking into account the states

        self.init_tf()

    def make_network(self, dim_input, dim_output, mixtures=False):
        """
        An example a network in tf that has both state and image inputs.
        Args:
            dim_input: Dimensionality of input. expecting 2d tuple (num_frames x num_batches)
            dim_output: Dimensionality of the output.
            batch_size: Batch size.
            network_config: dictionary of network structure parameters
        Returns:
            A tfMap object that stores inputs, outputs, and scalar loss.
        """
        discriminator_options = []

        if self.config["img_input"]:
            nn_input, target = self.get_input_layer_image(dim_input[1], dim_output)
        else:
            nn_input, target = self.get_input_layer(dim_input[0], dim_input[1], dim_output)


        for i in range(self.num_options):
            discriminator_options.append(ConvStateBasedDiscriminator(dim_input, nn_input=nn_input, dim_output=dim_output, scope="option%d"%i, config=self.config))

        #TODO: a better formulation is to have terminations be a latent variable that somehow sums to 1
        #TODO: can we combined these mixtures in interesting ways to train each other?

        # For example, can we have one net that takes into it pixels and another that takes in states, then we backprop
        # through the whole network using information from the states during training, but then drop that part of the network
        # and in that way keep information from the state and transfer it to the image inputs.
        # NOTE: if we don't do this and you see this in our code and decide to do it, please reach out to us first.

        termination_options = ConvStateBasedDiscriminator(dim_input, nn_input=nn_input, dim_output=self.num_options, config=self.config)

        #TODO: make this configurable
        self.termination_softmax_logits = termination_options.discrimination_logits

        if not mixtures:
            # TODO: then it's options, this flag is gross, change it
            # import pdb; pdb.set_trace()
            k = 1
            indices = tf.nn.top_k(self.termination_softmax_logits, k=k).indices
            vec = tf.zeros( tf.shape(self.termination_softmax_logits))
            for k in range(k):
                vec += tf.reshape(tf.one_hot(indices[:,k], tf.shape(self.termination_softmax_logits)[1]), tf.shape(self.termination_softmax_logits))
            v = tf.cast(vec == 0, vec.dtype) * -math.inf
            self.termination_softmax_logits = vec * v
            # self.termination_softmax_logits = self.termination_softmax_logits *  tf.reshape(tf.one_hot(tf.nn.top_k(self.termination_softmax_logits).indices, tf.shape(self.termination_softmax_logits)[1]), tf.shape(self.termination_softmax_logits))

            # tf.reshape(tf.one_hot(tf.nn.top_k(self.termination_softmax_logits).indices, tf.shape(self.termination_softmax_logits)[1]), tf.shape(self.termination_softmax_logits))

        self.termination_softmax_logits = tf.nn.softmax(termination_options.discrimination_logits)

        # try to make the weights sparse so we dropout features
        if self.use_l1_loss:
            l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.1, scope=None)
            regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, termination_options.weights)
        else:
            regularization_penalty = 0.0

        #TODO: add gaussian noise and top-k? https://arxiv.org/pdf/1701.06538.pdf
        self.class_target = target
        self.nn_input = nn_input
        self.discriminator_options = discriminator_options
        # import pdb; pdb.set_trace()
        combined_options = tf.concat([x.discrimination_logits for x in discriminator_options], axis=1)
        #combined_options should have shape (batch_size x num_options )
        self.discrimination_logits = tf.reshape(tf.reduce_sum(combined_options * self.termination_softmax_logits, axis=1), [-1, 1])
        #self.discrimination_logits should have shape (batch_size x 1)

        # tf.add_n([tf.transpose(tf.multiply(tf.transpose(x.discrimination_logits), self.termination_softmax_logits[:,i])) for i, x in enumerate(discriminator_options)]) + regularization_penalty

        #TODO: what works better, this loss function or each individual loss function
        # add importance to loss
        self.termination_importance_values = tf.reduce_sum(self.termination_softmax_logits, axis=0)

        self.loss, self.optimizer = self.get_loss_layer(pred=self.discrimination_logits, target_output=target)




        label_accuracy = tf.equal(tf.round(tf.nn.sigmoid(self.discrimination_logits)), tf.round(self.class_target))

        self.label_accuracy = tf.reduce_mean(tf.cast(label_accuracy, tf.float32))
        self.mse = tf.reduce_mean(tf.nn.l2_loss(tf.nn.sigmoid(self.discrimination_logits) - self.class_target))

        ones = tf.ones_like(self.class_target)

        true_positives = tf.round(tf.nn.sigmoid(self.discrimination_logits)) * tf.round(self.class_target)
        predicted_positives = tf.round(tf.nn.sigmoid(self.discrimination_logits))

        false_negatives = tf.logical_not(tf.logical_xor(tf.equal(tf.round(tf.nn.sigmoid(self.discrimination_logits)), ones), tf.equal(tf.round(self.class_target), ones)))

        self.label_precision = tf.reduce_sum(tf.cast(true_positives, tf.float32)) / tf.reduce_sum(tf.cast(predicted_positives, tf.float32))
        self.label_recall = tf.reduce_sum(tf.cast(true_positives, tf.float32)) / (tf.reduce_sum(tf.cast(true_positives, tf.float32)) + tf.reduce_sum(tf.cast(false_negatives, tf.float32)))


        # Why do this? apply these termination functions as termination functions for policy options?
        # use TRPO to train N different policies with the termination function taking into account the states

    def output_termination_activations(self, num_frames):
        number_data_points_per_dimension = 100
        # TODO: make these variable per dimension
        dim_min = -3
        dim_max = 3
        lins = []
        for i in range(self.input_dim[1]):
            lins.append(np.arange(dim_min, dim_max, 0.2))

        meshes = np.meshgrid(*lins, sparse=False)
        # TODO: finish this

        data = np.array(meshes).T.reshape(-1, num_frames, self.input_dim[1])
        # TODO: this is gross
        for dimension in range(self.input_dim[1]):
            mean_activations_for_values = []
            for i in np.arange(dim_min, dim_max, 0.2):
                input_arrays = []
                for x in data:
                    # TODO: the 0 below indexes into the frame, this is gross... this is all gross...
                    if x[0][dimension] == i:
                        input_arrays.append(x)
                activations = self.sess.run([self.termination_softmax_logits], feed_dict={self.nn_input: np.array(input_arrays)})[0]
                mean_activations = np.sum(activations, axis=0)
                mean_activations_for_values.append(mean_activations)
            dim_range = np.arange(dim_min, dim_max, 0.2)
            fig = plt.figure()
            for j in range(self.num_options):
                plt.plot(dim_range, [x[j] for x in mean_activations_for_values], label="Reward Option %d" % j)
            plt.xlabel('Inputs Along Dimension %d' % dimension, fontsize=18)
            plt.ylabel('Sum of Activations Along Uniform Grid', fontsize=16)
            plt.legend()
            fig.suptitle('Summed Activations for Dimension %d' % dimension)
            fig.savefig('activations_dim_%d.png' % dimension)
            plt.clf()

    @staticmethod
    def get_input_layer_image(state_size, dim_output=1):
        """produce the placeholder inputs that are used to run ops forward and backwards.
        net_input: usually an observation.
        action: mu, the ground truth actions we're trying to learn.
        precision: precision matrix used to commpute loss."""
        im_w, im_h, channels = state_size
        net_input = tf.placeholder('float', [None,1,  im_w * im_h * channels], name='nn_input')
        targets = tf.placeholder('float', [None, dim_output], name='targets')
        return net_input, targets

    @staticmethod
    def get_input_layer(num_frames, state_size, dim_output=1):
        """produce the placeholder inputs that are used to run ops forward and backwards.
        net_input: usually an observation.
        action: mu, the ground truth actions we're trying to learn.
        precision: precision matrix used to commpute loss."""
        net_input = tf.placeholder('float', [None, num_frames, state_size], name='nn_input')
        targets = tf.placeholder('float', [None, dim_output], name='targets')
        return net_input, targets
