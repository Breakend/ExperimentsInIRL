import tensorflow as tf
import numpy as np

# This is mostly taken and modified from: https://github.com/Breakend/third_person_im/blob/master/sandbox/bradly/third_person/discriminators/discriminator.py

class Discriminator(object):
    def __init__(self, input_dim, output_dim_class=2, output_dim_dom=None, tf_sess=None):
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

    def init_tf(self):
        if self.sess is None:
            self.sess = tf.Session()
        init = tf.initialize_all_variables()
        self.sess.run(init)

    def make_network(self, dim_input, output_dim_class, output_dim_dom):
        raise NotImplementedError

    def train(self, data_batch, targets_batch):
        cost = self.sess.run([self.optimizer, self.loss], feed_dict={self.nn_input: data_batch,
                                                                     self.class_target: targets_batch})[1]
        return cost

    def eval(self, data, softmax=True):
        if softmax is True:
            logits = tf.nn.softmax(self.discrimination_logits)
        else:
            logits = self.discrimination_logits
        log_prob = self.sess.run([logits], feed_dict={self.nn_input: data})[0]
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
                if dropout is not None:
                    cur_top = tf.nn.dropout(cur_top, dropout)
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
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, target_output))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        return cost, optimizer


class MLPDiscriminator(Discriminator):
    def __init__(self, input_dim):
        super(MLPDiscriminator, self).__init__(input_dim)
        self.make_network(dim_input=input_dim, dim_output=2)
        self.init_tf()

    def make_network(self, dim_input, dim_output):
        n_layers = 3
        dim_hidden = (n_layers - 1) * [40]
        dim_hidden.append(dim_output)
        nn_input, target = self.get_input_layer(dim_input, dim_output)
        mlp_applied = self.get_mlp_layers(nn_input, n_layers, dim_hidden)
        loss, optimizer = self.get_loss_layer(pred=mlp_applied, target_output=target)
        self.class_target = target
        self.nn_input = nn_input
        self.discrimination_logits = mlp_applied
        self.optimizer = optimizer
        self.loss = loss

    def get_input_layer(self, dim_input, dim_output):
        """produce the placeholder inputs that are used to run ops forward and backwards.
        net_input: usually an observation.
        action: mu, the ground truth actions we're trying to learn.
        precision: precision matrix used to commpute loss."""
        net_input = tf.placeholder('float', [None, dim_input], name='nn_input')
        targets = tf.placeholder('float', [None, dim_output], name='targets')
        return net_input, targets
