import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def make_network(self, dim_input, output_dim_class, output_dim_dom):
        raise NotImplementedError

    def compute_pos_weight(self, targets_batch):
        summs = np.sum(targets_batch, axis=0)
        pos_samples = summs[0]
        neg_samples = summs[1]
        if pos_samples <= 0:
            ratio = 0.0
        else:
            ratio = (np.float32(neg_samples)/np.float32(targets_batch.shape[0]))
        # print(ratio)
        # import pdb; pdb.set_trace()
        return ratio

    def train(self, data_batch, targets_batch):
        cost = self.sess.run([self.optimizer, self.loss], feed_dict={
                                                                     self.pos_weighting: self.compute_pos_weight(targets_batch),
                                                                     self.nn_input: data_batch,
                                                                     self.class_target: targets_batch})[1]
        return cost

    def eval(self, data, softmax=True):
        if softmax is True:
            logits = tf.nn.softmax(self.discrimination_logits)
        else:
            logits = self.discrimination_logits
        # import pdb; pdb.set_trace()
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
        # http://stackoverflow.com/questions/40698709/tensorflow-interpretation-of-weight-in-weighted-cross-entropy
        self.pos_weighting = tf.placeholder('float', [], name='pos_weighting')
        # import pdb; pdb.set_trace()

        class_weight = [[self.pos_weighting, 1.0 - self.pos_weighting]]
        weight_per_label = tf.transpose(tf.matmul(target_output,tf.transpose(class_weight) ))

        # cross_entropy = tf.nn.weighted_cross_entropy_with_logits(logits=pred, targets=target_output, pos_weight=self.pos_weighting)
        xent = tf.multiply(weight_per_label, tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=target_output, name="xent_raw")) #shape [1, batch_size]
        cost = tf.reduce_mean(xent)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        return cost, optimizer

class ConvStateBasedDiscriminator(Discriminator):
    """ A state based descriminator, assuming a state vector """

    def __init__(self, input_dim, dim_output=2):
        super(ConvStateBasedDiscriminator, self).__init__(input_dim)
        self.make_network(dim_input=input_dim, dim_output=dim_output)
        self.init_tf()

    def get_lab_accuracy(self, data, class_labels):
        return self.sess.run([self.label_accuracy], feed_dict={self.nn_input: data,
                                                               self.class_target: class_labels})[0]
    def make_network(self, dim_input, dim_output):
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
        # im_width = dim_input[0]
        # im_height = dim_input[1]
        # num_channels = dim_input[2]
        num_filters = [5, 5]

        nn_input, target = self.get_input_layer(dim_input[0], dim_input[1], dim_output)

        # we pool twice, each time reducing the image size by a factor of 2.
        #conv_out_size = int(im_width * im_height * num_filters[1] / ((2.0 * pool_size) * (2.0 * pool_size)))
        conv_out_size = int(dim_input[0] * num_filters[1])
        # first_dense_size = conv_out_size

        # Store layers weight & bias
        weights = {
            'wc1': self.get_xavier_weights([filter_size, dim_input[1], num_filters[0]], (pool_size, pool_size)),
        # 5x5 conv, 1 input, 32 outputs
            'wc2': self.get_xavier_weights([filter_size, num_filters[0], num_filters[1]], (pool_size, pool_size)),
        # 5x5 conv, 32 inputs, 64 outputs
        }

        biases = {
            'bc1': self.init_bias([num_filters[0]]),
            'bc2': self.init_bias([num_filters[1]]),
        }

        conv_layer_0 = self.conv1d(vec=nn_input, w=weights['wc1'], b=biases['bc1'])

        # conv_layer_0 = self.max_pool(conv_layer_0, k=pool_size)

        conv_layer_1 = self.conv1d(vec=conv_layer_0, w=weights['wc2'], b=biases['bc2'])

        conv_layer_0 = tf.layers.conv1d(nn_input, filters=5, kernel_size=3, strides=1, padding='same')
        # Don't need to max pool? state space too small?
        conv_layer_1 = tf.layers.conv1d(conv_layer_0, filters=5, kernel_size=3, strides=1, padding='same')

        # conv_layer_1 = self.max_pool(conv_layer_1, k=pool_size)

        conv_out_flat = tf.reshape(conv_layer_1, [-1, conv_out_size])

        fc_output = self.get_mlp_layers(conv_out_flat, n_mlp_layers, dim_hidden, dropout=None)

        loss, optimizer = self.get_loss_layer(pred=fc_output, target_output=target)

        self.class_target = target
        self.nn_input = nn_input
        self.discrimination_logits = fc_output
        self.optimizer = optimizer
        self.loss = loss
        # label_accuracy = tf.equal(tf.round(tf.nn.sigmoid(self.discrimination_logits)), tf.round(self.class_target))
        label_accuracy = tf.equal(tf.argmax(self.class_target, 1),
                          tf.argmax(tf.nn.softmax(self.discrimination_logits), 1))
        self.label_accuracy = tf.reduce_mean(tf.cast(label_accuracy, tf.float32))

    @staticmethod
    def get_input_layer(num_frames, state_size, dim_output=2):
        """produce the placeholder inputs that are used to run ops forward and backwards.
        net_input: usually an observation.
        action: mu, the ground truth actions we're trying to learn.
        precision: precision matrix used to commpute loss."""
        net_input = tf.placeholder('float', [None, num_frames, state_size], name='nn_input')
        targets = tf.placeholder('float', [None, dim_output], name='targets')
        return net_input, targets

class ConvStateBasedDiscriminatorWithExternalIO(Discriminator):
    """ A state based descriminator, assuming a state vector """

    def __init__(self, input_dim, nn_input, target, dim_output=2, scope="external", tf_sess=None):
        with tf.variable_scope(scope):
            super(ConvStateBasedDiscriminatorWithExternalIO, self).__init__(input_dim, tf_sess=tf_sess)
            self.make_network(input_dim, dim_output, nn_input, target)
            self.init_tf()

    def get_lab_accuracy(self, data, class_labels):
        return self.sess.run([self.label_accuracy], feed_dict={self.nn_input: data,
                                                               self.class_target: class_labels})[0]

    def make_network(self, dim_input, dim_output, nn_input, target):
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
        # im_width = dim_input[0]
        # im_height = dim_input[1]
        # num_channels = dim_input[2]
        num_filters = [5, 5]

        # we pool twice, each time reducing the image size by a factor of 2.
        #conv_out_size = int(im_width * im_height * num_filters[1] / ((2.0 * pool_size) * (2.0 * pool_size)))
        conv_out_size = int(dim_input[0] * num_filters[1])
        # first_dense_size = conv_out_size

        # Store layers weight & bias
        # weights = {
        #     'wc1': self.get_xavier_weights([filter_size, dim_input[1], num_filters[0]], (pool_size, pool_size)),
        # # 5x5 conv, 1 input, 32 outputs
        #     'wc2': self.get_xavier_weights([filter_size, num_filters[0], num_filters[1]], (pool_size, pool_size)),
        # # 5x5 conv, 32 inputs, 64 outputs
        # }
        #
        # biases = {
        #     'bc1': self.init_bias([num_filters[0]]),
        #     'bc2': self.init_bias([num_filters[1]]),
        # }

        conv_layer_0 = tf.layers.conv1d(nn_input, filters=5, kernel_size=3, strides=1, padding='same', name="conv0")
        # Don't need to max pool? state space too small?
        conv_layer_1 = tf.layers.conv1d(conv_layer_0, filters=5, kernel_size=3, strides=1, padding='same', name="conv1")
        # conv_layer_0 = self.conv1d(vec=nn_input, w=weights['wc1'], b=biases['bc1'])

        # conv_layer_0 = self.max_pool(conv_layer_0, k=pool_size)

        # conv_layer_1 = self.conv1d(vec=conv_layer_0, w=weights['wc2'], b=biases['bc2'])

        # conv_layer_1 = self.max_pool(conv_layer_1, k=pool_size)

        conv_out_flat = tf.reshape(conv_layer_1, [-1, conv_out_size])

        fc_output = self.get_mlp_layers(conv_out_flat, n_mlp_layers, dim_hidden, dropout=None)

        # loss, optimizer = self.get_loss_layer(pred=fc_output, target_output=target)

        self.class_target = target
        self.nn_input = nn_input
        self.discrimination_logits = fc_output
        # self.optimizer = optimizer
        # self.loss = loss
        label_accuracy = tf.equal(tf.argmax(self.class_target, 1),
                          tf.argmax(tf.nn.softmax(self.discrimination_logits), 1))
        self.label_accuracy = tf.reduce_mean(tf.cast(label_accuracy, tf.float32))

    @staticmethod
    def get_input_layer(num_frames, state_size, dim_output=2):
        """produce the placeholder inputs that are used to run ops forward and backwards.
        net_input: usually an observation.
        action: mu, the ground truth actions we're trying to learn.
        precision: precision matrix used to commpute loss."""
        net_input = tf.placeholder('float', [None, num_frames, state_size], name='nn_input')
        targets = tf.placeholder('float', [None, dim_output], name='targets')
        return net_input, targets

class ConvStateBasedDiscriminatorWithOptions(Discriminator):
    """ A state based descriminator, assuming a state vector """

    def __init__(self, input_dim, num_options=4, mixtures=True, config={}):
        super(ConvStateBasedDiscriminatorWithOptions, self).__init__(input_dim)
        self.num_options = num_options
        # TODO: move the other args into the config
        self.config = config
        self.use_l1_loss = False
        self.input_dim = input_dim
        self.make_network(dim_input=input_dim, dim_output=2, mixtures=mixtures)
        self.init_tf()

    def get_lab_accuracy(self, data, class_labels):
        return self.sess.run([self.label_accuracy], feed_dict={self.nn_input: data,
                                                               self.class_target: class_labels})[0]

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
        nn_input, target = self.get_input_layer(dim_input[0], dim_input[1], dim_output)

        #TODO: a better formulation is to have terminations be a latent variable that somehow sums to 1
        #TODO: can we combined these mixtures in interesting ways to train each other?

        # For example, can we have one net that takes into it pixels and another that takes in states, then we backprop
        # through the whole network using information from the states during training, but then drop that part of the network
        # and in that way keep information from the state and transfer it to the image inputs.
        # NOTE: if we don't do this and you see this in our code and decide to do it, please reach out to us first.

        termination_options = ConvStateBasedDiscriminatorWithExternalIO(dim_input, nn_input, target, self.num_options)

        #TODO: make this configurable
        k = 1

        self.termination_softmax_logits = tf.nn.softmax(termination_options.discrimination_logits)
        # import pdb; pdb.set_trace()

        if not mixtures:
            # TODO: then it's options, this flag is gross, change it
            self.termination_softmax_logits = tf.reshape(tf.one_hot(tf.nn.top_k(self.termination_softmax_logits).indices, tf.shape(self.termination_softmax_logits)[1]), tf.shape(self.termination_softmax_logits))
            # self.termination_softmax_logits = tf.one_hot(tf.nn.top_k(self.termination_softmax_logits).indices, tf.shape(self.termination_softmax_logits)[0])

        for i in range(self.num_options):
            discriminator_options.append(ConvStateBasedDiscriminatorWithExternalIO(dim_input, nn_input, target, scope="option%d"%i))

        # try to make the weights sparse so we dropout features
        if self.use_l1_loss:
            l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.1, scope=None)
            regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, termination_options.weights)
        else:
            regularization_penalty = 0.0

        #TODO: add gaussian noise and top-k? https://arxiv.org/pdf/1701.06538.pdf
        self.class_target = target
        # import pdb; pdb.set_trace()
        self.nn_input = nn_input
        #TODO: softmax
        self.discrimination_logits = tf.add_n([tf.transpose(tf.multiply(tf.transpose(x.discrimination_logits), self.termination_softmax_logits[:,i])) for i, x in enumerate(discriminator_options)]) + regularization_penalty
        # TODO: add importance or dropout to the loss function or something. I.e. have each expert be responsible for a state space somehow. and all the termiantion values should some to 1
        self.loss, self.optimizer = self.get_loss_layer(pred=self.discrimination_logits, target_output=target)

        # add importance to loss
        termination_importance_values = tf.reduce_sum(self.termination_softmax_logits, axis=0)
        # import pdb; pdb.set_trace()
        mean, var = tf.nn.moments(termination_importance_values, axes=[0])
        cv = var/mean
        if "importance_weight" in self.config:
            importance_weight = config["importance_weight"]
        else:
            importance_weight = 0.00
        self.loss += importance_weight*tf.nn.l2_loss(cv)

        label_accuracy = tf.equal(tf.argmax(self.class_target, 1),
                          tf.argmax(tf.nn.softmax(self.discrimination_logits), 1))
        # label_accuracy = tf.equal(tf.round(tf.nn.sigmoid(self.discrimination_logits)), tf.round(self.class_target))

        self.label_accuracy = tf.reduce_mean(tf.cast(label_accuracy, tf.float32))

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
            # lins.append(np.linspace(dim_min, dim_max, number_data_points_per_dimension))
        # import pdb; pdb.set_trace()
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
    def get_input_layer(num_frames, state_size, dim_output=2):
        """produce the placeholder inputs that are used to run ops forward and backwards.
        net_input: usually an observation.
        action: mu, the ground truth actions we're trying to learn.
        precision: precision matrix used to commpute loss."""
        net_input = tf.placeholder('float', [None, num_frames, state_size], name='nn_input')
        targets = tf.placeholder('float', [None, dim_output], name='targets')
        return net_input, targets
