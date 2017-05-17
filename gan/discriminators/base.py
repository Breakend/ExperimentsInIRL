import tensorflow as tf
import numpy as np
from gan.nn_utils import *
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
        self.learning_rate = config["learning_rate"]
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
        self.decaying_dropout = tf.train.exponential_decay(0.6, self.train_step, 5, 0.9, staircase=True)

    def init_tf(self):
        # Hack to only initialize unitialized variables
        if self.sess is None:
            self.sess = tf.Session()
        initialize_uninitialized(self.sess)

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
