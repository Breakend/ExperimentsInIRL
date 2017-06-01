import numpy as np

from sandbox.rocky.tf.core.layers_powered import LayersPowered
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.core.network import MLP, ConvNetwork
from sandbox.rocky.tf.spaces.box import Box

from rllab.core.serializable import Serializable
from sandbox.rocky.tf.policies.base import StochasticPolicy
from sandbox.rocky.tf.distributions.diagonal_gaussian import DiagonalGaussian
from rllab.misc.overrides import overrides
from rllab.misc import logger
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf

class ElemwiseMultiplyReduceSoftmaxReshapeLayer(L.MergeLayer):
    def __init__(self, incomings, **kwargs):
        super(ElemwiseMultiplyReduceSoftmaxReshapeLayer, self).__init__(incomings, **kwargs)

    def get_output_for(self, inputs, **kwargs):
        # assert len(inputs) == 2
        # return tf.nn.softmax(tf.reshape(tf.reduce_sum(combined_options * gating_network, axis=1), [-1, 1]))
        # import pdb; pdb.set_trace()
        gate = inputs[-1]
        others = inputs[:-1]

        stuff =[]
        for i,x in enumerate(others):
            stuff.append(x * tf.reshape(gate[:,i], (-1, 1)))

        return tf.reduce_sum(stuff, axis=0)

    def get_output_shape_for(self, input_shapes):
        assert len(set(input_shapes)) == 1
        return input_shapes[0]

class GaussianDecomposedPolicy(StochasticPolicy, LayersPowered, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            hidden_sizes=(32, 32),
            learn_std=True,
            init_std=1.0,
            adaptive_std=False,
            std_share_network=False,
            std_hidden_sizes=(32, 32),
            min_std=1e-6,
            std_hidden_nonlinearity=tf.nn.tanh,
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
            mean_network=None,
            std_network=None,
            std_parametrization='exp',
            num_options = 4,
            gating_network = None,
            input_layer = None,
            conv_filters = None, conv_filter_sizes = None, conv_strides = None, conv_pads = None, input_shape=None
    ):
        """
        :param env_spec:
        :param hidden_sizes: list of sizes for the fully-connected hidden layers
        :param learn_std: Is std trainable
        :param init_std: Initial std
        :param adaptive_std:
        :param std_share_network:
        :param std_hidden_sizes: list of sizes for the fully-connected layers for std
        :param min_std: whether to make sure that the std is at least some threshold value, to avoid numerical issues
        :param std_hidden_nonlinearity:
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param output_nonlinearity: nonlinearity for the output layer
        :param mean_network: custom network for the output mean
        :param std_network: custom network for the output log std
        :param std_parametrization: how the std should be parametrized. There are a few options:
            - exp: the logarithm of the std will be stored, and applied a exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        :return:
        """
        Serializable.quick_init(self, locals())
        assert isinstance(env_spec.action_space, Box)

        with tf.variable_scope(name):

            obs_dim = env_spec.observation_space.flat_dim
            action_dim = env_spec.action_space.flat_dim

            self.num_options = num_options

            input_layer, output_layer = self.make_network((obs_dim,),
                                                          action_dim,
                                                          hidden_sizes,
                                                          hidden_nonlinearity=hidden_nonlinearity,
                                                          gating_network = gating_network,
                                                          l_in = input_layer,
                                                          conv_filters=conv_filters,
                                                          conv_filter_sizes=conv_filter_sizes,
                                                          conv_strides=conv_strides,
                                                          conv_pads=conv_pads,
                                                          input_shape=input_shape)

            self._mean_network_output_layer = output_layer

            l_mean = output_layer
            obs_var = input_layer.input_var

            if std_network is not None:
                l_std_param = std_network.output_layer
            else:
                if adaptive_std:
                    raise NotImplementedError
                else:
                    if std_parametrization == 'exp':
                        init_std_param = np.log(init_std)
                    elif std_parametrization == 'softplus':
                        init_std_param = np.log(np.exp(init_std) - 1)
                    else:
                        raise NotImplementedError
                    l_std_param = L.ParamLayer(
                        input_layer,
                        num_units=action_dim,
                        param=tf.constant_initializer(init_std_param),
                        name="output_std_param",
                        trainable=learn_std,
                    )

            self.std_parametrization = std_parametrization

            if std_parametrization == 'exp':
                min_std_param = np.log(min_std)
            elif std_parametrization == 'softplus':
                min_std_param = np.log(np.exp(min_std) - 1)
            else:
                raise NotImplementedError

            self.min_std_param = min_std_param

            self._l_mean = l_mean
            self._l_std_param = l_std_param

            self._dist = DiagonalGaussian(action_dim)

            LayersPowered.__init__(self, [l_mean, l_std_param])
            super(GaussianDecomposedPolicy, self).__init__(env_spec)

            dist_info_sym = self.dist_info_sym(input_layer.input_var, dict())
            mean_var = dist_info_sym["mean"]
            log_std_var = dist_info_sym["log_std"]

            self._f_dist = tensor_utils.compile_function(
                inputs=[obs_var],
                outputs=[mean_var, log_std_var],
            )

    def _make_subnetwork(self, input_layer, dim_output, hidden_sizes, output_nonlinearity=tf.sigmoid, hidden_nonlinearity=tf.nn.tanh, name="pred_network",
                             conv_filters = None, conv_filter_sizes = None, conv_strides = None, conv_pads = None, input_shape = None):

        if conv_filters is not None:
            input_layer = L.reshape(input_layer, ([0],) + input_shape, name="reshape_input")
            prob_network = ConvNetwork(
                    input_shape = input_shape,
                    output_dim=dim_output,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=output_nonlinearity,
                    name=name,
                    input_layer=input_layer,
                    conv_filters=conv_filters,
                    conv_filter_sizes=conv_filter_sizes,
                    conv_strides=conv_strides,
                    conv_pads=conv_pads)
        else:
            prob_network = MLP(
                    output_dim=dim_output,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=output_nonlinearity,
                    name=name,
                    input_layer=input_layer
                )

        return prob_network.output_layer

    def _make_gating_network(self, input_layer, hidden_sizes, apply_softmax=False,
                         conv_filters = None, conv_filter_sizes = None, conv_strides = None, conv_pads = None, input_shape=None):
        if apply_softmax:
            output_nonlinearity = tf.nn.softmax
        else:
            output_nonlinearity = None
        return self._make_subnetwork(input_layer,
                                     dim_output=self.num_options,
                                     hidden_sizes=hidden_sizes,
                                     output_nonlinearity=output_nonlinearity,
                                     name="gate",
                                     conv_filters=conv_filters,
                                     conv_filter_sizes=conv_filter_sizes,
                                     conv_strides=conv_strides,
                                     conv_pads=conv_pads,
                                     input_shape=input_shape)

    def make_network(self, dim_input, dim_output, subnetwork_hidden_sizes, discriminator_options=[], hidden_nonlinearity=tf.nn.tanh, gating_network=None, l_in = None,
                     conv_filters = None, conv_filter_sizes = None, conv_strides = None, conv_pads = None, input_shape=None):
        if l_in is None:
            l_in = L.InputLayer(shape=(None,) + tuple(dim_input))

        if len(discriminator_options) < self.num_options:
            for i in range(len(discriminator_options), self.num_options):
                subnet = self._make_subnetwork(l_in,
                                               dim_output=dim_output,
                                               hidden_sizes=subnetwork_hidden_sizes,
                                               output_nonlinearity=None,
                                               hidden_nonlinearity=hidden_nonlinearity,
                                               name="option%d" % i,
                                               conv_filters=conv_filters,
                                               conv_filter_sizes=conv_filter_sizes,
                                               conv_strides=conv_strides,
                                               conv_pads=conv_pads,
                                               input_shape=input_shape)
                discriminator_options.append(subnet)

        # only apply softmax if we're doing mixtures, if sparse mixtures or options, need to apply after sparsifying
        if gating_network is None:
            gating_network = self._make_gating_network(l_in,
                                                       apply_softmax = True,
                                                       hidden_sizes=subnetwork_hidden_sizes,
                                                       conv_filters=conv_filters,
                                                       conv_filter_sizes=conv_filter_sizes,
                                                       conv_strides=conv_strides,
                                                       conv_pads=conv_pads,
                                                       input_shape=input_shape)

        # combined_options = L.ConcatLayer(discriminator_options, axis=1)
        # combined_options = tf.concat(discriminator_options, axis=1)
        output = ElemwiseMultiplyReduceSoftmaxReshapeLayer(discriminator_options + [gating_network])

        # self.termination_importance_values = tf.reduce_sum(self.termination_softmax_logits, axis=0)

        return l_in, output

    @property
    def vectorized(self):
        return True

    def dist_info_sym(self, obs_var, state_info_vars=None):
        mean_var, std_param_var = L.get_output([self._l_mean, self._l_std_param], obs_var)
        if self.min_std_param is not None:
            std_param_var = tf.maximum(std_param_var, self.min_std_param)
        if self.std_parametrization == 'exp':
            log_std_var = std_param_var
        elif self.std_parametrization == 'softplus':
            log_std_var = tf.log(tf.log(1. + tf.exp(std_param_var)))
        else:
            raise NotImplementedError
        return dict(mean=mean_var, log_std=log_std_var)

    @overrides
    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        mean, log_std = [x[0] for x in self._f_dist([flat_obs])]
        rnd = np.random.normal(size=mean.shape)
        action = rnd * np.exp(log_std) + mean
        return action, dict(mean=mean, log_std=log_std)

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        means, log_stds = self._f_dist(flat_obs)
        rnd = np.random.normal(size=means.shape)
        actions = rnd * np.exp(log_stds) + means
        return actions, dict(mean=means, log_std=log_stds)

    def get_reparam_action_sym(self, obs_var, action_var, old_dist_info_vars):
        """
        Given observations, old actions, and distribution of old actions, return a symbolically reparameterized
        representation of the actions in terms of the policy parameters
        :param obs_var:
        :param action_var:
        :param old_dist_info_vars:
        :return:
        """
        new_dist_info_vars = self.dist_info_sym(obs_var, action_var)
        new_mean_var, new_log_std_var = new_dist_info_vars["mean"], new_dist_info_vars["log_std"]
        old_mean_var, old_log_std_var = old_dist_info_vars["mean"], old_dist_info_vars["log_std"]
        epsilon_var = (action_var - old_mean_var) / (tf.exp(old_log_std_var) + 1e-8)
        new_action_var = new_mean_var + epsilon_var * tf.exp(new_log_std_var)
        return new_action_var

    def log_diagnostics(self, paths):
        log_stds = np.vstack([path["agent_infos"]["log_std"] for path in paths])
        logger.record_tabular('AveragePolicyStd', np.mean(np.exp(log_stds)))

    @property
    def distribution(self):
        return self._dist
