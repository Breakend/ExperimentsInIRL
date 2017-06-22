import tensorflow as tf
import numpy as np
from gan.nn_utils import *
import matplotlib.pyplot as plt
import sandbox.rocky.tf.core.layers as L
import math
import itertools
from sandbox.rocky.tf.core.network import MLP
from gan.discriminators.base import Discriminator


class MLPDiscriminator(Discriminator):
    """ A state based descriminator, assuming a state vector """

    def __init__(self, input_dim, dim_output=1, config=None, nn_input=None, target=None, tf_sess=None, scope="external"):
        with tf.variable_scope(scope):
            super(MLPDiscriminator, self).__init__(input_dim, config=config, tf_sess=tf_sess)
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


    def make_network(self, dim_input, dim_output, nn_input=None, target=None, hidden_sizes = (50,)):
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

        if nn_input is None:
            nn_input = tf.placeholder('float', [None, dim_input[0], dim_input[1]], name='nn_input')

        if target is None:
            target = tf.placeholder('float', [None, dim_output], name='targets')

        l_in = L.InputLayer(shape=(None,) + tuple(dim_input), input_var=nn_input, name="input")

        prob_network = MLP(
                output_dim=dim_output,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=tf.nn.relu,
                output_nonlinearity=None,
                name="pred_network",
                input_layer=l_in
            )

        fc_output = L.get_output(prob_network.output_layer)

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

    def get_loss_layer(self, pred, target_output):
        # http://stackoverflow.com/questions/40698709/tensorflow-interpretation-of-weight-in-weighted-cross-entropy
        # self.pos_weighting = tf.placeholder('float', [], name='pos_weighting')

        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=target_output)#, pos_weight=self.pos_weighting)

        if "entropy_penalty" in self.config and self.config["entropy_penalty"] > 0.0:
            cross_entropy -= float(self.config["entropy_penalty"])*logit_bernoulli_entropy(pred)
        cost = tf.reduce_sum(cross_entropy, axis=0)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        return cost, optimizer
