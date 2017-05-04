
from rllab.misc.overrides import overrides
from scipy.misc import imresize
from rllab.spaces.box import Box
from cached_property import cached_property
import scipy.misc
from rllab.misc.overrides import overrides

# TODO: move this to folder with different files

class BaseTransformer(object):

    def transform(self, observation):
        raise NotImplementedError

    def transformed_observation_space(self, prev_observation_space):
        raise NotImplementedError


class ResizeImageTransformer(BaseTransformer):

    def __init__(self, fraction_of_current_size):
        self.fraction_of_current_size = fraction_of_current_size

    @overrides
    def transform(self, observation):
        return scipy.misc.imresize(observation, self.fraction_of_current_size)

    @overrides
    def transformed_observation_space(self, wrapped_observation_space):
        if type(wrapped_observation_space) is Box:
            return Box(scipy.misc.imresize(wrapped_observation_space.low, self.fraction_of_current_size), scipy.misc.imresize(wrapped_observation_space.high, self.fraction_of_current_size))
        else:
            raise NotImplementedError("Currently only support Box observation spaces for ResizeImageTransformer")
