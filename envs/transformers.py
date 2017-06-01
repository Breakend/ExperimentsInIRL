
from rllab.misc.overrides import overrides
from scipy.misc import imresize
from rllab.spaces.box import Box
from cached_property import cached_property
import scipy.misc
import numpy as np
from rllab.misc.overrides import overrides

# TODO: move this to folder with different files

class BaseTransformer(object):

    def transform(self, observation):
        raise NotImplementedError

    def transformed_observation_space(self, prev_observation_space):
        # import pdb; pdb.set_trace()
        raise prev_observation_space

class SimpleNormalizePixelIntensitiesTransformer(BaseTransformer):
    """
    Normalizes pixel intensities simply by dividing by 255.
    """
    @overrides
    def transform(self, observation):
        return np.array(observation).astype(np.float32) / 255.0

    @overrides
    def transformed_observation_space(self, wrapped_observation_space):
        return wrapped_observation_space

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

class RandomSensorMaskTransformer(BaseTransformer):

    def __init__(self, env, percent_of_sensors_to_occlude=.15):
        """
        Knock out random sensors
        """
        self.percent_of_sensors_to_occlude = percent_of_sensors_to_occlude
        self.obs_dims = env.observation_space.flat_dim
        # self._set_sensor_mask(env, sensor_idx)

    def occlude(self, obs):
        sensor_idx = np.random.randint(0, self.obs_dims-1, size=int(self.obs_dims * self.percent_of_sensors_to_occlude))
        obs[sensor_idx] = 0
        return obs

    @overrides
    def transform(self, observation):
        # import pdb; pdb.set_trace()
        return self.occlude(observation)

    # def _set_sensor_mask(self, env, sensor_idx):
    #     # from https://raw.githubusercontent.com/openai/rllab/7f632c97c936b7658bc04b7957e46ccda262b8fa/rllab/envs/occlusion_env.py
    #     obsdim = env.observation_space.flat_dim
    #     if len(sensor_idx) > obsdim:
    #         raise ValueError("Length of sensor mask ({0}) cannot be greater than observation dim ({1})".format(len(sensor_idx), obsdim))
    #     if len(sensor_idx) == obsdim and not np.any(np.array(sensor_idx) > 1):
    #         sensor_mask = np.array(sensor_idx, dtype=np.bool)
    #     elif np.any( np.unique(sensor_idx, return_counts=True)[1] > 1):
    #         raise ValueError("Double entries or boolean mask with dim ({0}) < observation dim ({1})".format(len(sensor_idx), obsdim))
    #     else:
    #         sensor_mask = np.zeros((obsdim,), dtype=np.bool)
    #         sensor_mask[sensor_idx] = 1
    #     self._sensor_mask = sensor_mask
