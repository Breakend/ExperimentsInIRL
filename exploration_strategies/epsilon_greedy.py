from rllab.core.serializable import Serializable
from sandbox.rocky.tf.spaces.discrete import Discrete
from rllab.exploration_strategies.base import ExplorationStrategy
import numpy as np


class GaussianEpsilonGreedyStrategy(ExplorationStrategy, Serializable):
    """
    This strategy adds Gaussian noise to the action taken by the deterministic policy.
    """

    def __init__(self, env_spec, policy, max_sigma=1.0, min_sigma=0.1, decay_period=1000000):
        assert isinstance(env_spec.action_space, Discrete)
        # assert len(env_spec.action_space.shape) == 1
        Serializable.quick_init(self, locals())
        self._max_sigma = max_sigma
        self._min_sigma = min_sigma
        self._decay_period = decay_period
        self._action_space = env_spec.action_space
        self.t = 0
        self.policy = policy

    # @overrides
    def get_action(self, observation, **kwargs):
        action, agent_info = self.policy.get_action(observation)
        #TOD
        # sigma = self._max_sigma - (self._max_sigma - self._min_sigma) * min(1.0, self.t * 1.0) / self._decay_period
        # self.t += 1
        # import pdb; pdb.set_trace()

        if np.random.random() > .1:
            return action, agent_info
        else:
            # Explore (test all arms)
            return np.random.randint(self._action_space.n), agent_info
