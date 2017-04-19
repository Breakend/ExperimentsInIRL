from .standard_discriminators import ConvStateBasedDiscriminatorWithExternalIO
import tensorflow as tf
from .utils import *
import numpy as np

class WGANCostTrainer(object):
    # This is some attempt at trying to do Wasserstein stuffffff?

    def __init__(self, input_dims, config={}):
        # input_dims is the size of the feature vectors
        self.sess = sess = tf.Session()
        self.inputs_novice, self.targets_novice = ConvStateBasedDiscriminatorWithExternalIO.get_input_layer(input_dims[0], input_dims[1], 2)
        self.inputs_expert, self.targets_expert = ConvStateBasedDiscriminatorWithExternalIO.get_input_layer(input_dims[0], input_dims[1], 2)
        # tf.get_variable_scope().reuse_variables()
        self.disc_exp = ConvStateBasedDiscriminatorWithExternalIO(input_dims, self.inputs_novice, self.targets_novice, tf_sess=sess)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            self.disc_novice = ConvStateBasedDiscriminatorWithExternalIO(input_dims, self.inputs_expert, self.targets_expert, tf_sess = sess)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.disc_exp.discrimination_logits, labels=self.targets_expert)) + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.disc_novice.discrimination_logits, labels=self.targets_novice))
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def get_reward(self, samples):
        logits = tf.nn.softmax(self.disc_novice.discrimination_logits)
        log_prob = self.sess.run([logits], feed_dict={self.inputs_novice: samples, self.inputs_expert: samples})[0]
        return log_prob[:, 0]

    def dump_datapoints(self, num_frames=4):
        if num_frames != 1:
            print("only support graphing internal things with 1 frame concated for now")
            return
        return

    def train_cost(self, novice_rollouts_tensor, expert_rollouts_tensor, number_epochs=2, num_frames=4):
        # import pdb; pdb.set_trace()
        data_matrix_exp, class_matrix_exp = shuffle_to_training_data_single(expert_rollouts_tensor, novice=False, num_frames=num_frames)
        data_matrix_nov, class_matrix_nov = shuffle_to_training_data_single(novice_rollouts_tensor, novice=True, num_frames=num_frames)

        self._train_cost(data_matrix_nov,data_matrix_exp, class_matrix_nov, class_matrix_exp, epochs=number_epochs)

    def _train_cost(self, data_nov, data_exp, nov_classes, exp_classes, epochs=2, batch_size=20):
        for iter_step in range(0, epochs):
            batch_losses = []
            lab_nov_acc = []
            lab_exp_acc = []
            # import pdb; pdb.set_trace()
            for batch_step in range(0, data_exp.shape[0], batch_size):

                data_exp_batch = data_exp[batch_step: batch_step+batch_size]
                exp_classes_batch = exp_classes[batch_step: batch_step+batch_size]
                data_nov_batch = data_nov[batch_step: batch_step+batch_size]
                nov_classes_batch = nov_classes[batch_step: batch_step+batch_size]
                cost = self.sess.run([self.optimizer, self.loss],
                                     feed_dict={
                                        self.inputs_novice : data_nov_batch,
                                        self.inputs_expert : data_exp_batch,
                                        self.targets_novice: nov_classes_batch,
                                        self.targets_expert: exp_classes_batch})[1]
                batch_losses.append(cost)
                # lab_nov_acc.append(self.disc_novice.get_lab_accuracy(data_nov_batch, nov_classes_batch))
                # lab_exp_acc.append(self.disc_novice.get_lab_accuracy(data_exp_batch, exp_classes_batch))
            print('loss is ' + str(np.mean(np.array(batch_losses))))
            # print('novice acc is ' + str(np.mean(np.array(lab_nov_acc))))
            # print('expert acc is ' + str(np.mean(np.array(lab_exp_acc))))
