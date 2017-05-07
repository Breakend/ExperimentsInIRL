from .transformers import BaseTransformer
from pre_trained_inception.pretrained_layers import pre_trained_net
import tensorflow as tf
from rllab.misc.overrides import overrides
from rllab.spaces.box import Box
from cached_property import cached_property
from rllab.core.serializable import Serializable
4
class InceptionTransformer(BaseTransformer, Serializable):

    def __init__(self, input_dims, inception_checkpoint_path, sess = None):
        Serializable.quick_init(self, locals())
        if sess is None:
            sess = tf.Session()
        init_fn, self.inputs, self.net_out = pre_trained_net(input_dims, inception_checkpoint_path)
        self.net_out = tf.reshape(tf.tanh(self.net_out), [-1])
        # import pdb; pdb.set_trace()
        self.out_shape = self.net_out.get_shape().as_list()
        init_fn(sess)
        self.sess = sess

    @overrides
    def transform(self, observation):
        return self.sess.run([self.net_out], feed_dict={self.inputs : observation})[0]

    @overrides
    def transformed_observation_space(self, wrapped_observation_space):
        if type(wrapped_observation_space) is Box:
            return Box(0.0, 1.0, self.out_shape)
        else:
            raise NotImplementedError("Currently only support Box observation spaces for InceptionTransformer")
