import pickle
from rllab.misc import tensor_utils
import numpy as np
import time
from tqdm import tqdm
import copy

def batchify_dict(samples, batch_size, total_len):
    for i in range(0, total_len, batch_size):
        yield select_from_tensor_dict(samples, i, min(total_len, i+batch_size))

def batchify_list(samples, batch_size):
    total_len = len(samples)
    for i in range(0, total_len, batch_size):
        yield select_from_tensor(samples, i, min(total_len, i+batch_size))

def select_from_tensor_dict(tensor_dict, start, end):
    keys = list(tensor_dict.keys())
    ret = dict()
    for k in keys:
        if isinstance(tensor_dict[k], dict):
            ret[k] = select_from_tensor_dict(tensor_dict[k], start, end)
        else:
            ret[k] = select_from_tensor(tensor_dict[k], start, end)
    return ret

def select_from_tensor(x, start, end):
    return x[start:end]


def shorten_tensor_dict(tensor_dict, max_len):
    keys = list(tensor_dict.keys())
    ret = dict()
    for k in keys:
        if isinstance(tensor_dict[k], dict):
            ret[k] = shorten_tensor_dict(tensor_dict[k], max_len)
        else:
            ret[k] = shorten_tensor(tensor_dict[k], max_len)
    return ret

def shorten_tensor(x, max_len):
    return x[:max_len]

def sample_policy_trajectories(policy, number_of_trajectories, env, horizon=200, reward_extractor=None, num_frames=4, concat_timesteps=True):
    """
    Mostly taken from https://github.com/bstadie/third_person_im/blob/master/sandbox/bradly/third_person/algos/cyberpunk_trainer.py#L164
    Generate a sampling dataset for a given number of rollouts
    """
    paths = []

    for iter_step in range(0, number_of_trajectories):
        paths.append(rollout_policy(agent=policy, env=env, max_path_length=horizon, reward_extractor=reward_extractor, num_frames=num_frames, concat_timesteps=concat_timesteps))

    return paths

def load_expert_rollouts(filepath):
    # why encoding? http://stackoverflow.com/questions/11305790/pickle-incompatability-of-numpy-arrays-between-python-2-and-3
    return pickle.load(open(filepath, "rb"), encoding='latin1')

def process_samples_with_reward_extractor_no_batch(samples, reward_extractor, concat_timesteps, num_frames):
    t0 = time.time()
    for sample in tqdm(samples):
        if reward_extractor:
            true_rewards = sample['rewards']
            observations = sample['observations']
            feature_space = len(observations[0])
            all_datas = []
            for time_key in range(len(observations)):
                data_matrix = np.zeros(shape=(1, num_frames, feature_space))
                # we want the first thing in the sequence to be repeated until we have enough to form a sequence
                # TODO: replicate this in the extract rewards function
                for t in range(0, num_frames):
                    time_key_plus_one = max(time_key - t, 0)
                    data_matrix[0, num_frames-t-1, :] = observations[time_key_plus_one, :]
                all_datas.append(data_matrix)
            rewards = reward_extractor.get_reward(np.vstack(all_datas))
            sample['rewards'] = rewards
            sample['true_rewards'] = true_rewards
        else:
            sample['true_rewards'] = sample['rewards']

    t1 = time.time()
    print("Time to process samples: %d" % (t1-t0))
    return samples

def process_samples_with_reward_extractor(samples, reward_extractor, concat_timesteps, num_frames, batch_size=2000):
    t0 = time.time()
    super_all_datas = []
    #TODO: this is gross for now, but should be good enough?
    samples = copy.deepcopy(samples)
    # splits = []
    if reward_extractor:
        # convert all the data to the proper format, concat frames if needed
        for sample in samples:
            true_rewards = sample['rewards']
            observations = sample['observations']
            feature_space = len(observations[0])
            all_datas = []
            for time_key in range(len(observations)):
                data_matrix = np.zeros(shape=(1, num_frames, feature_space))
                # we want the first thing in the sequence to be repeated until we have enough to form a sequence
                # TODO: replicate this in the extract rewards function
                for t in range(0, num_frames):
                    time_key_plus_one = max(time_key - t, 0)
                    data_matrix[0, num_frames-t-1, :] = observations[time_key_plus_one, :]
                # all_datas.append(data_matrix)
                super_all_datas.append(data_matrix)

        extracted_rewards = []
        for batch in batchify_list(np.vstack(super_all_datas), batch_size): #TODO: make batch_size configurable
            extracted_rewards.extend(np.split(reward_extractor.get_reward(batch), len(batch))) #TODO: unnecessary computation here

        index = 0
        extracted_rewards = np.vstack(extracted_rewards)
        for sample in samples:#len(extracted_rewards):
            sample['true_rewards'] = sample['rewards']
            num_obs = len(sample['observations'])
            sample['rewards'] = select_from_tensor(extracted_rewards, index, index+num_obs).reshape(-1)
            if len(sample['true_rewards']) != len(sample['rewards']):
                import pdb; pdb.set_trace()
                raise Exception("Problem, extracted rewards not equal in length to old rewards!")
            index += num_obs
    else:
        for sample in tqdm(samples):
            sample['true_rewards'] = sample['rewards']

    t1 = time.time()
    print("Time to process samples: %d" % (t1-t0))
    return samples

def rollout_policy(agent, env, max_path_length=200, reward_extractor=None, speedup=1, get_image_observations=False, num_frames=4, concat_timesteps=True):
    """
    Mostly taken from https://github.com/bstadie/third_person_im/blob/master/sandbox/bradly/third_person/algos/cyberpunk_trainer.py#L164
    Generate a rollout for a given policy
    """
    observations = []
    im_observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    path_length = 0

    while path_length <= max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))

        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        o = next_o
        if get_image_observations:
            pixel_array = env.render(mode="rgb_array")
            if pixel_array is None:
                # Not convinced that behaviour works for all environments, so until
                # such a time as I'm convinced of this, drop into a debug shell
                print("Problem! Couldn't get pixels! Dropping into debug shell.")
                import pdb; pdb.set_trace()
            im_observations.append(pixel_array)
        if d:
            rewards.append(r)
            break
        else:
            rewards.append(r)
    # if animated:
    # env.render(close=True)

    im_observations = tensor_utils.stack_tensor_list(im_observations)

    observations = tensor_utils.stack_tensor_list(observations)

    if reward_extractor is not None:
        #TODO: remove/replace this
        if concat_timesteps:
            true_rewards = tensor_utils.stack_tensor_list(rewards)
            obs_pls_three = np.zeros((observations.shape[0], num_frames, observations.shape[1]))
            # import pdb; pdb.set_trace()
            for iter_step in range(0, obs_pls_three.shape[0]):
                for i in range(num_frames):
                    idx_plus_three = min(iter_step+num_frames, obs_pls_three.shape[0]-1)
                    obs_pls_three[iter_step, i, :] = observations[idx_plus_three, :]
            rewards = reward_extractor.get_reward(obs_pls_three)
        else:
            true_rewards = tensor_utils.stack_tensor_list(rewards)
            rewards = reward_extractor.get_reward(observations)
    else:
        rewards = tensor_utils.stack_tensor_list(rewards)
        true_rewards = rewards

    return dict(
        observations=observations,
        im_observations=im_observations,
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=rewards,
        true_rewards=true_rewards,
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )

import numpy as np
import rllab.misc.logger as logger

class SimpleReplayPool(object):
    """
    Used from https://raw.githubusercontent.com/shaneshixiang/rllabplusplus/master/rllab/pool/simple_pool.py
    """
    def __init__(
            self, max_pool_size, observation_dim, action_dim,
            replacement_policy='stochastic', replacement_prob=1.0,
            max_skip_episode=10):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_pool_size = max_pool_size
        self._replacement_policy = replacement_policy
        self._replacement_prob = replacement_prob
        self._max_skip_episode = max_skip_episode
        self._observations = np.zeros(
            (max_pool_size, observation_dim),
        )
        self._actions = np.zeros(
            (max_pool_size, action_dim),
        )
        self._rewards = np.zeros(max_pool_size)
        self._terminals = np.zeros(max_pool_size, dtype='uint8')
        self._initials = np.zeros(max_pool_size, dtype='uint8')
        self._bottom = 0
        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, reward, terminal, initial):
        self.check_replacement()
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._initials[self._top] = initial
        self.advance()

    def add_rollout(self, rollout):
        for i, stuff in enumerate(zip(rollout["observations"], rollout['actions'], rollout['rewards'])):
            observation, action, reward = stuff
            terminal = (i==len(rollout["observations"])-1)
            initial = (i == 0)
            self.add_sample(observation, action, reward, terminal, inital)

    def check_replacement(self):
        if self._replacement_prob < 1.0:
            if self._size < self._max_pool_size or \
                not self._initials[self._top]: return
            self.advance_until_terminate()

    def get_skip_flag(self):
        if self._replacement_policy == 'full': skip = False
        elif self._replacement_policy == 'stochastic':
            skip = np.random.uniform() > self._replacement_prob
        else: raise NotImplementedError
        return skip

    def advance_until_terminate(self):
        skip = self.get_skip_flag()
        n_skips = 0
        old_top = self._top
        new_top = (old_top + 1) % self._max_pool_size
        while skip and old_top != new_top and n_skips < self._max_skip_episode:
            n_skips += 1
            self.advance()
            while not self._initials[self._top]:
                self.advance()
            skip = self.get_skip_flag()
            new_top = self._top
        logger.log("add_sample, skipped %d episodes, top=%d->%d"%(
            n_skips, old_top, new_top))

    def advance(self):
        self._top = (self._top + 1) % self._max_pool_size
        if self._size >= self._max_pool_size:
            self._bottom = (self._bottom + 1) % self._max_pool_size
        else:
            self._size += 1

    def random_batch(self, batch_size):
        assert self._size > batch_size
        indices = np.zeros(batch_size, dtype='uint64')
        transition_indices = np.zeros(batch_size, dtype='uint64')
        count = 0
        while count < batch_size:
            index = np.random.randint(self._bottom, self._bottom + self._size) % self._max_pool_size
            # make sure that the transition is valid: if we are at the end of the pool, we need to discard
            # this sample
            if index == self._size - 1 and self._size <= self._max_pool_size:
                continue
            # if self._terminals[index]:
            #     continue
            transition_index = (index + 1) % self._max_pool_size
            # make sure that the transition is valid: discard the transition if it crosses horizon-triggered resets
            if not self._terminals[index] and self._initials[transition_index]:
                continue
            indices[count] = index
            transition_indices[count] = transition_index
            count += 1
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            initials=self._initials[indices],
            next_observations=self._observations[transition_indices]
        )

    @property
    def size(self):
        return self._size

class RolloutReplayPool(object):
    """
    Stores complete rollouts
    """
    def __init__(
            self, max_pool_size,
            replacement_policy='stochastic', replacement_prob=1.0,
            max_skip_episode=10):
        self._max_pool_size = max_pool_size
        self._replacement_policy = replacement_policy
        self._replacement_prob = replacement_prob
        self._max_skip_episode = max_skip_episode
        self._rollouts = []
        self._replacement_index = 0

    def add_rollout(self, rollout):
        if len(self._rollouts) < self._max_pool_size:
            self._rollouts.append(rollout)
        else:
            skip = np.random.uniform() > self._replacement_prob
            if skip:
                return
            else:
                self._rollouts[self._replacement_index] = rollout
                self._replacement_index += 1

    def random_batch(self, batch_size):
        return np.random.choice(self._rollouts, size=batch_size)
