from .standard_discriminators import ConvStateBasedDiscriminator
import numpy as np

class ApprenticeshipCostLearningTrainer(object):

    def __init__(self, input_dims):
        # self.W = tf.Variable?? something like that some initialize
        # tf.Variable(tf.random_uniform(filter_shape, minval=low, maxval=high, dtype=tf.float32))
        # input_dims is the size of the feature vectors

    def get_reward(self, samples):
        # TODO: multiply W dot samples
        return

    def train_cost(self, novice_rollouts_tensor, expert_rollouts_tensor, number_epochs=2):
        #  run the svm optimization things?
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/estimators/svm.py
        # https://github.com/nfmcclure/tensorflow_cookbook/blob/master/04_Support_Vector_Machines/05_Implementing_Nonlinear_SVMs/05_nonlinear_svm.py
        import pdb; pdb.set_trace()

        # I guess you need to have the discounted sum of expert_rollouts_tensor states across a trajectory. I forget what the expert_rollouts_tensor shape looks like

        return
