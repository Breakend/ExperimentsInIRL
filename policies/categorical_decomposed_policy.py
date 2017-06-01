from sandbox.rocky.tf.core.layers_powered import LayersPowered
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.core.network import MLP, ConvNetwork
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.distributions.categorical import Categorical
from sandbox.rocky.tf.policies.base import StochasticPolicy
from rllab.misc import ext
from sandbox.rocky.tf.misc import tensor_utils
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.spaces.discrete import Discrete
import tensorflow as tf
from gan.nn_utils import *


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

        return tf.nn.softmax(tf.reduce_sum(stuff, axis=0))

    def get_output_shape_for(self, input_shapes):
        assert len(set(input_shapes)) == 1
        return input_shapes[0]


class CategoricalDecomposedPolicy(StochasticPolicy, LayersPowered, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.tanh,
            gating_network=None,
            input_layer=None,
            num_options=4,
            conv_filters = None, conv_filter_sizes = None, conv_strides = None, conv_pads = None, input_shape=None
    ):
        """
        :param env_spec: A spec for the mdp.
        :param hidden_sizes: list of sizes for the fully connected hidden layers
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param prob_network: manually specified network for this policy, other network params
        are ignored
        :return:
        """
        Serializable.quick_init(self, locals())

        self.num_options = num_options

        assert isinstance(env_spec.action_space, Discrete)

        with tf.variable_scope(name):
            input_layer, output_layer = self.make_network((env_spec.observation_space.flat_dim,),
                                                          env_spec.action_space.n,
                                                          hidden_sizes,
                                                          hidden_nonlinearity=hidden_nonlinearity,
                                                          gating_network = gating_network,
                                                          l_in = input_layer,
                                                          conv_filters=conv_filters,
                                                          conv_filter_sizes=conv_filter_sizes,
                                                          conv_strides=conv_strides,
                                                          conv_pads=conv_pads,
                                                          input_shape=input_shape)
            self._l_prob = output_layer
            self._l_obs = input_layer

            self._f_prob = tensor_utils.compile_function(
                [input_layer.input_var],
                L.get_output(output_layer)
            )

            self._dist = Categorical(env_spec.action_space.n)

            super(CategoricalDecomposedPolicy, self).__init__(env_spec)
            LayersPowered.__init__(self, [output_layer])


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

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars=None):
        # TODO: fix this
        return dict(prob=L.get_output(self._l_prob, {self._l_obs: tf.cast(obs_var, tf.float32)}))

    @overrides
    def dist_info(self, obs, state_infos=None):
        return dict(prob=self._f_prob(obs))

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        prob = self._f_prob([flat_obs])[0]
        action = self.action_space.weighted_sample(prob)
        return action, dict(prob=prob)

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        probs = self._f_prob(flat_obs)
        actions = list(map(self.action_space.weighted_sample, probs))
        return actions, dict(prob=probs)

    @property
    def distribution(self):
        return self._dist
