import gym
from gym import Wrapper
from gym import error, version
import os, json, logging, numpy as np, six, time
from gym.monitoring import stats_recorder, video_recorder
from gym.utils import atomic_write, closer
from gym.utils.json_utils import json_encode_np
import pickle

logger = logging.getLogger(__name__)

FILE_PREFIX = 'openaigym'
MANIFEST_PREFIX = FILE_PREFIX + '.manifest'

class NumpyStateRewardRecordingMonitor(Wrapper):
    def __init__(self, env, dump_dir, name="a3c"):
        super(NumpyStateRewardRecordingMonitor, self).__init__(env)
        self.rewards = []
        self.actions = []
        self.observations = []
        self.episode_num = 1
        self.name= name
        self.dump_dir = dump_dir


    def _step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        self.actions.append(action)
        self.observations.append(observation)

        return observation, reward, done, info


    def _reset(self):
        reward = np.sum(self.rewards)
        timestr = time.strftime("%Y%m%d-%H%M%S")

        if len(self.rewards) > 0:
            with open("%s/%s-%s-episode%d-numpy-wrapper-demo-Seaquest-v0-reward-%s.pickle" % (self.dump_dir, self.name,timestr, self.episode_num, reward), "wb") as output_file:
                pickle.dump(dict(observations=self.observations, rewards=self.rewards, actions=self.actions), output_file)

        observation = self.env.reset()
        self.rewards = []
        self.actions = []
        self.observations = []
        self.episode_num += 1
        return observation

    def _close(self):
        super(NumpyStateRewardRecordingMonitor, self)._close()
