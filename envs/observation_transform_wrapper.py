import numpy as np
from cached_property import cached_property

from rllab import spaces
from rllab.core.serializable import Serializable
from rllab.envs.proxy_env import ProxyEnv
from rllab.misc.overrides import overrides
from rllab.envs.base import Step

BIG = 1e6

class ObservationTransformWrapper(ProxyEnv, Serializable):
    ''' Occludes part of the observation.'''

    def __init__(self, env, transformer):
        '''
        :param sensor_idx: list or ndarray of indices to be shown. Other indices will be occluded. Can be either list of
            integer indices or boolean mask.
        '''
        Serializable.quick_init(self, locals())
        self.transformer = transformer
        super(ObservationTransformWrapper, self).__init__(env)

    @cached_property
    @overrides
    def observation_space(self):
        obs_space = self.transformer.transformed_observation_space(self._wrapped_env.observation_space)
        print("Transformed observation space : {}".format(obs_space))
        return obs_space

    @overrides
    def reset(self):
        obs = self._wrapped_env.reset()
        return self.transformer.transform(obs)

    @overrides
    def step(self, action):
        next_obs, reward, done, info = self._wrapped_env.step(action)
        return Step(self.transformer.transform(next_obs), reward, done, **info)

    @overrides
    def log_diagnostics(self, paths):
        pass # the wrapped env will be expecting its own observations in paths, but they're not
