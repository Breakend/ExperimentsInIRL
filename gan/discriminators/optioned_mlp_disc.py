import tensorflow as tf
import numpy as np
from gan.nn_utils import *
import matplotlib.pyplot as plt
import sandbox.rocky.tf.core.layers as L
import math
import itertools
from sandbox.rocky.tf.core.network import MLP
from gan.discriminators.base import Discriminator


class MLPMixingDiscriminator(Discriminator):
    """ A state based descriminator, assuming a state vector """

    def __init__(self, input_dim, num_options=4, mixtures=True, config=None):
        super(MLPMixingDiscriminator, self).__init__(input_dim, config=config)
        self.num_options = num_options
        # TODO: move the other args into the config
        self.config = config
        self.use_l1_loss = False
        self.input_dim = input_dim
        self.dim_output = 1
        self.mixtures = mixtures
        self.cv_penalty = None
        self.old_mi = None
        self.new_mi = None

        if "subnetwork_hidden_sizes" in config:
            self.subnetwork_hidden_sizes = config["subnetwork_hidden_sizes"]
        else:
            self.subnetwork_hidden_sizes = (32,32)

        self.make_network(dim_input=input_dim, dim_output=self.dim_output, subnetwork_hidden_sizes =self.subnetwork_hidden_sizes )

        self.init_tf()

        self.losses_printed = []
        losses = []
        if self.old_mi is not None:
            losses.append(self.old_mi)
            self.losses_printed.append("old_mi")
        if self.new_mi is not None:
            losses.append(self.new_mi)
            self.losses_printed.append("new_mi")
        if self.cv_penalty is not None:
            losses.append(self.cv_penalty)
            self.losses_printed.append("cv_penalty")
        self.extra_losses = losses

    def get_lab_accuracy(self, data, class_labels):
        return self.sess.run([self.label_accuracy], feed_dict={self.nn_input: data,
                                                               self.class_target: class_labels})[0]

    def get_mse(self, data, class_labels):
        return self.sess.run([self.mse], feed_dict={self.nn_input: data,
                                                               self.class_target: class_labels})[0]

    def get_separate_losses(self, data, class_labels):
        return self.sess.run(self.extra_losses, feed_dict={self.nn_input: data,
                                                               self.class_target: class_labels})

    def get_lab_precision(self, data, class_labels):
        return self.sess.run([self.label_precision], feed_dict={self.nn_input: data,
                                                               self.class_target: class_labels})[0]

    def get_lab_recall(self, data, class_labels):
        return self.sess.run([self.label_recall], feed_dict={self.nn_input: data,
                                                               self.class_target: class_labels})[0]

    def get_termination_activations(self, data, class_labels):
        return self.sess.run([self.termination_softmax_logits], feed_dict={self.nn_input: data,
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
            self.cv_penalty = importance_weight*10.0*tf.nn.l2_loss(cv)
            cost += self.cv_penalty

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
                mean_i, var_i = tf.nn.moments(self.discriminator_options[i], axes=[0])
                mean_j, var_j = tf.nn.moments(self.discriminator_options[j], axes=[0])
                mean_ij, var_ij = tf.nn.moments(tf.multiply(self.discriminator_options[i],
                                                        self.discriminator_options[j]), axes=[0])
                # TODO: ^ Does this make sense mathematically ??
                corr_numerator = mean_ij-mean_i*mean_j
                corr_denominator = tf.square(var_i)*tf.square(var_j) + TINY
                corr_coeff = corr_numerator/corr_denominator
                mutual_info = -(1/2.0) * log10(1-tf.square(corr_coeff))

                mi += mutual_info
            mi /= float(len(combos))
            importance_weight = self.config["importance_weights"]
            self.old_mi = (importance_weight)*tf.nn.l2_loss(mi)
            cost += self.old_mi
        elif self.config["use_mutual_info_penalty_infogan"]:
            print("Using Mutual info penalty")
            combos = [item for idx, item in enumerate(itertools.combinations(range(len(self.discriminator_options)), 2))]
            mi = tf.Variable(0, dtype=tf.float32)
            for (i,j) in combos:
                # http://wiseodd.github.io/techblog/2017/01/29/infogan/
                cond_ent = tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.log(self.discriminator_options[i].discrimination_logits + TINY), self.discriminator_options[j].discrimination_logits), 1))
                ent = tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.log(self.discriminator_options[i].discrimination_logits + TINY), self.discriminator_options[i].discrimination_logits), 1))
                mi += (cond_ent + ent)

            mi /= float(len(combos))
            importance_weight = self.config["importance_weights"]
            self.new_mi = (importance_weight)*tf.nn.l2_loss(mi)
            cost += self.new_mi

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        return cost, optimizer

    def _remake_network_from_disc_options(self, discriminator_options, stop_gradients = False, num_extra_options = 0):
        self.actual_train_step = 0

        if stop_gradients:
            self.discriminator_options = [tf.stop_gradient(x) for x in discriminator_options]

        if num_extra_options > 0:
            self.num_options += num_extra_options
            for i in range(num_extra_options):
                subnet = self._make_subnetwork(l_in, dim_output=1, hidden_sizes=self.subnetwork_hidden_sizes, output_nonlinearity=None, name="extraoption%d" % i)
                discriminator_options.append(subnet)

        with tf.variable_scope("second"):
            self.make_network(self.input_dim, self.dim_output, self.mixtures, self.nn_input, self.target, discriminator_options)
        self.init_tf()


    def _make_subnetwork(self, input_layer, dim_output, hidden_sizes, output_nonlinearity=tf.sigmoid, name="pred_network"):

        prob_network = MLP(
                # input_shape=(env_spec.observation_space.flat_dim,),
                output_dim=dim_output,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=lrelu,
                output_nonlinearity=output_nonlinearity,
                name=name,
                input_layer=input_layer
            )

        return L.get_output(prob_network.output_layer)

    def _make_gating_network(self, input_layer, hidden_sizes, apply_softmax=False):
        if apply_softmax:
            output_nonlinearity = tf.nn.softmax
        else:
            output_nonlinearity = None
        return self._make_subnetwork(input_layer, dim_output=self.num_options, hidden_sizes=hidden_sizes, output_nonlinearity=output_nonlinearity, name="gate")

    def make_network(self, dim_input, dim_output, subnetwork_hidden_sizes, nn_input=None, target=None, discriminator_options=[]):
        # create input layer
        if nn_input is None:
            nn_input = tf.placeholder('float', [None, dim_input[0], dim_input[1]], name='nn_input')

        if target is None:
            target = tf.placeholder('float', [None, dim_output], name='targets')

        l_in = L.InputLayer(shape=(None,) + tuple(dim_input), input_var=nn_input)

        if len(discriminator_options) < self.num_options:
            for i in range(len(discriminator_options), self.num_options):
                subnet = self._make_subnetwork(l_in, dim_output=1, hidden_sizes=subnetwork_hidden_sizes, output_nonlinearity=None, name="option%d" % i)
                discriminator_options.append(subnet)

        # only apply softmax if we're doing mixtures, if sparse mixtures or options, need to apply after sparsifying
        gating_network = self._make_gating_network(l_in, apply_softmax = True, hidden_sizes=subnetwork_hidden_sizes)

        #TODO: a better formulation is to have terminations be a latent variable that somehow sums to 1
        #TODO: can we combined these mixtures in interesting ways to train each other?

        # For example, can we have one net that takes into it pixels and another that takes in states, then we backprop
        # through the whole network using information from the states during training, but then drop that part of the network
        # and in that way keep information from the state and transfer it to the image inputs.
        # NOTE: if we don't do this and you see this in our code and decide to do it, please reach out to us first.

        # Get the top K options if using optiongan
        if not self.mixtures:
            print("Using options")
            k = 1
            indices = tf.nn.top_k(gating_network, k=k).indices
            vec = tf.zeros( tf.shape(gating_network))
            for k in range(k):
                vec += tf.reshape(tf.one_hot(indices[:,k], tf.shape(gating_network)[1]), tf.shape(gating_network))
            # v = tf.cast(vec == 0, vec.dtype) * -math.inf
            # gating_network = tf.nn.softmax(vec )

        self.class_target = target
        self.nn_input = nn_input
        self.discriminator_options = discriminator_options
        self.termination_softmax_logits = gating_network

        combined_options = tf.concat(discriminator_options, axis=1)
        self.discrimination_logits = tf.reshape(tf.reduce_sum(combined_options * gating_network, axis=1), [-1, 1])

        self.termination_importance_values = tf.reduce_sum(self.termination_softmax_logits, axis=0)

        self.loss, self.optimizer = self.get_loss_layer(pred=self.discrimination_logits, target_output=target)

        # #################
        # Metrics
        # #################

        # accuracy
        label_accuracy = tf.equal(tf.round(tf.nn.sigmoid(self.discrimination_logits)), tf.round(self.class_target))
        self.label_accuracy = tf.reduce_mean(tf.cast(label_accuracy, tf.float32))

        # error
        self.mse = tf.reduce_mean(tf.nn.l2_loss(tf.nn.sigmoid(self.discrimination_logits) - self.class_target))

        # precision and recall
        ones = tf.ones_like(self.class_target)
        true_positives = tf.round(tf.nn.sigmoid(self.discrimination_logits)) * tf.round(self.class_target)
        predicted_positives = tf.round(tf.nn.sigmoid(self.discrimination_logits))
        false_negatives = tf.logical_not(tf.logical_xor(tf.equal(tf.round(tf.nn.sigmoid(self.discrimination_logits)), ones), tf.equal(tf.round(self.class_target), ones)))
        self.label_precision = tf.reduce_sum(tf.cast(true_positives, tf.float32)) / tf.reduce_sum(tf.cast(predicted_positives, tf.float32))
        self.label_recall = tf.reduce_sum(tf.cast(true_positives, tf.float32)) / (tf.reduce_sum(tf.cast(true_positives, tf.float32)) + tf.reduce_sum(tf.cast(false_negatives, tf.float32)))

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
