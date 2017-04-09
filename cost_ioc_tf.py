""" This file defines neural network cost function. """
import copy
import logging
import numpy as np
import tempfile
import os

import tensorflow as tf
from tf_cost_utils import *
from sampling_utils import GpsBatchSampler

LOGGER = logging.getLogger(__name__)



class GuidedCostLearningTrainer(object):
    """ Set up weighted neural network norm loss with learned parameters. """
    def __init__(self, observation_dimensions, rollout_batch_size, trajectory_length, learning_rate, sess, tf_random_seed=123):
        self.observation_dimensions = observation_dimensions
        # T is the trajectory lengths
        self.trajectory_length = trajectory_length
        self.demo_batch_size = rollout_batch_size
        self.sample_batch_size = rollout_batch_size
        self.learning_rate = learning_rate
        tf.set_random_seed(tf_random_seed)
        # self.graph = tf.Graph()
        self.smooth_reg_weight = 0 #TODO
        self.mono_reg_weight = 0 #TODO
        self.gp_reg_weight = 0 #TODO
        self.learn_wu = False #TODO
        self.tf_random_seed = tf_random_seed
        self.session = sess
        # with self.graph.as_default():
        self._init_solver()


    def construct_nn_cost_net_tf(self, num_hidden=3, dim_hidden=42, dim_input=27, T=100,
                                 demo_batch_size=5, sample_batch_size=5, phase=None,
                                 Nq=1, smooth_reg_weight=0.0, mono_reg_weight=0.0, gp_reg_weight=0.0, learn_wu=False, x_idx=None, img_idx=None,
                                 num_filters=[15,15,15]):
        """ Construct cost net (with images and robot config).
        Args:
            ...
            if with images, x_idx is required, and should indicate the indices corresponding to the robot config
            if with images, img_idx is required, and should indicate the indices corresponding to the imagej
        """

        inputs = {}
        inputs['expert_observations'] = expert_observations = tf.placeholder(tf.float32, shape=(demo_batch_size, T, dim_input))
        inputs['novice_observations'] = novice_observations = tf.placeholder(tf.float32, shape=(sample_batch_size, T, dim_input))
        inputs['test_obs'] = test_obs = tf.placeholder(tf.float32, shape=(None, dim_input), name='test_obs')
        inputs['test_obs_single'] = test_obs_single = tf.placeholder(tf.float32, shape=(dim_input), name='test_obs_single')

        _, test_cost  = nn_forward(test_obs, num_hidden=num_hidden, dim_hidden=dim_hidden)
        expert_cost_preu, expert_costs = nn_forward(expert_observations, num_hidden=num_hidden, dim_hidden=dim_hidden)
        sample_cost_preu, sample_costs = nn_forward(novice_observations, num_hidden=num_hidden, dim_hidden=dim_hidden)

        # Build a differentiable test cost by feeding each timestep individually
        test_obs_single = tf.expand_dims(test_obs_single, 0)
        test_feat_single = compute_feats(test_obs_single, num_hidden=num_hidden, dim_hidden=dim_hidden)
        test_cost_single_preu, _ = nn_forward(test_obs_single, num_hidden=num_hidden, dim_hidden=dim_hidden)
        test_cost_single = tf.squeeze(test_cost_single_preu)

        expert_sample_preu = tf.concat(axis=0, values=[expert_cost_preu, sample_cost_preu])
        sample_demo_size = sample_batch_size+demo_batch_size
        assert_shape(expert_sample_preu, [sample_demo_size, T, 1])
        costs_prev = tf.slice(expert_sample_preu, begin=[0, 0,0], size=[sample_demo_size, T-2, -1])
        costs_next = tf.slice(expert_sample_preu, begin=[0, 2,0], size=[sample_demo_size, T-2, -1])
        costs_cur = tf.slice(expert_sample_preu, begin=[0, 1,0], size=[sample_demo_size, T-2, -1])
        # cur-prev
        slope_prev = costs_cur-costs_prev
        # next-cur
        slope_next = costs_next-costs_cur

        if smooth_reg_weight > 0:
            # regularization
            """
            """
            raise NotImplementedError("Smoothness reg not implemented")

        if mono_reg_weight > 0:
            demo_slope = tf.slice(slope_next, begin=[0,0,0], size=[demo_batch_size, -1, -1])
            slope_reshape = tf.reshape(demo_slope, shape=[-1,1])
            mono_reg = l2_mono_loss(slope_reshape)*mono_reg_weight
        else:
            mono_reg = 0

        ioc_loss = icml_loss(expert_costs, sample_costs)
        ioc_loss += mono_reg

        outputs = {
            # 'multiobj_loss': sup_loss+ioc_loss,
            'ioc_loss': ioc_loss,
            'test_loss': test_cost,
            'test_loss_single': test_cost_single,
            'test_feat_single': test_feat_single,
        }

        if x_idx is not None:
            outputs['test_imgfeat'] = test_imgfeat
            outputs['test_X_single'] = test_X_single

        return inputs, outputs

    def get_reward(self, sample_observations):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample:  A single sample
        """
        # T = sample.T
        # obs = sample.get_obs()
        # sample_u = sample.get_U()

        # dO = self.observation_dimensions
        # dU = sample.dU
        # dX = sample.dX
        # Initialize terms.
        # l = np.zeros(T)

        # tq_norm = np.sum(self._hyperparams['wu'] * (sample_u ** 2), axis=1, keepdims=True)
        import pdb; pdb.set_trace()


        l = np.squeeze(self.run([self.outputs['test_loss']], test_obs=sample_observations)[0])

        return l

    def train_cost(self, demoO, sampleO, epochs):
        """
        Learn cost function with generic function representation.
        Args:
            demoU: the actions of demonstrations.
            demoO: the observations of demonstrations.
            d_log_iw: log importance weights for demos.
            sampleU: the actions of samples.
            sampleO: the observations of samples.
            s_log_iw: log importance weights for samples.
        """
        # demo_torque_norm = np.sum(self._hyperparams['wu']* (demoU **2), axis=2, keepdims=True)
        # sample_torque_norm = np.sum(self._hyperparams['wu'] * (sampleU **2), axis=2, keepdims=True)

        # num_samp = sampleU.shape[0]
        # s_log_iw = s_log_iw[-num_samp:,:]
        for epoch in range(epochs):
            d_sampler = GpsBatchSampler([demoO])
            s_sampler = GpsBatchSampler([sampleO])

            for i, (d_batch, s_batch) in enumerate(
                    zip(d_sampler.with_replacement(batch_size=self.demo_batch_size), \
                        s_sampler.with_replacement(batch_size=self.sample_batch_size))):
                ioc_loss, grad = self.run([self.ioc_loss, self.ioc_optimizer],
                                          expert_observations=d_batch[0],
                                          novice_observations = s_batch[0])
                if i%200 == 0:
                    LOGGER.debug("Iteration %d loss: %f", i, ioc_loss)

    def _init_solver(self, sample_batch_size=None):
        """ Helper method to initialize the solver. """

        # Pass in net parameter by protostring (could add option to input prototxt file).
        network_arch_params = {}

        network_arch_params['dim_input'] = self.observation_dimensions
        network_arch_params['demo_batch_size'] = self.demo_batch_size
        if sample_batch_size is None:
            network_arch_params['sample_batch_size'] = self.demo_batch_size
        else:
            network_arch_params['sample_batch_size'] = sample_batch_size
        network_arch_params['T'] = self.trajectory_length
        # network_arch_params['ioc_loss'] = self.ioc_loss
        network_arch_params['smooth_reg_weight'] = self.smooth_reg_weight
        network_arch_params['mono_reg_weight'] = self.mono_reg_weight
        network_arch_params['gp_reg_weight'] = self.gp_reg_weight
        network_arch_params['learn_wu'] = self.learn_wu
        inputs, outputs = self.construct_nn_cost_net_tf(**network_arch_params)
        self.outputs = outputs

        self.input_dict = inputs
        self.ioc_loss = outputs['ioc_loss']

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.ioc_optimizer = optimizer.minimize(self.ioc_loss)

        # Set up gradients
        l_single, obs_single = outputs['test_loss_single'], inputs['test_obs_single']
        # self.dldx =  tf.gradients(l_single, obs_single)[0]
        # self.dldxx = jacobian(self.dldx, obs_single)
        # self.dfdx = jacobian(outputs['test_feat_single'][0], obs_single)

        self.saver = tf.train.Saver()

    def run(self, targets, **feeds):
        tf.set_random_seed(self.tf_random_seed)
        feed_dict = {self.input_dict[k]:v for (k,v) in feeds.items()}
        result = self.session.run(targets, feed_dict=feed_dict)
        return result

    def save_model(self, fname):
        LOGGER.debug('Saving model to: %s', fname)
        self.saver.save(self.session, fname)

    def restore_model(self, fname):
        self.saver.restore(self.session, fname)
        LOGGER.debug('Restoring model from: %s', fname)

    # For pickling.
    def __getstate__(self):
        with tempfile.NamedTemporaryFile('w+b', delete=True) as f:
            self.save_model(f.name)
            f.seek(0)
            with open(f.name, 'r') as f2:
                wts = f2.read()
            os.remove(f.name+'.meta')
        return {
            'wts': wts,
        }

    # For unpickling.
    def __setstate__(self, state):
        self.__init__(state['hyperparams'])
        with tempfile.NamedTemporaryFile('w+b', delete=True) as f:
            f.write(state['wts'])
            f.seek(0)
            self.restore_model(f.name)
