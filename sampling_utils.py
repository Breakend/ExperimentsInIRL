import pickle
from rllab.misc import tensor_utils
import numpy as np



def batchify_dict(samples, batch_size, total_len):
    for i in range(0, total_len, batch_size):
        yield select_from_tensor_dict(samples, i, min(total_len, i+batch_size))

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

def process_samples_with_reward_extractor(samples, reward_extractor, concat_timesteps, num_frames):
    for sample in samples:
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

    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        if d:
            rewards.append(0.0)
            # break
        else:
            rewards.append(r)
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
