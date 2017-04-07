

def sample_policy_trajectories(policy, number_of_trajectories, env, horizon=200):
    """
    Mostly taken from https://github.com/bstadie/third_person_im/blob/master/sandbox/bradly/third_person/algos/cyberpunk_trainer.py#L164
    Generate a sampling dataset for a given number of rollouts
    """
    paths = []

    for iter_step in range(0, n_trajs):
        paths.append(rollout_policy(agent=pol, env=env, max_path_length=horizon, reward_extractor=None))

    return paths


def rollout_policy(agent, env, max_path_length=200, reward_extractor=None, animated=True, speedup=1):
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
    o = env.reset_trial()
    path_length = 0

    if animated:
        env.render()
    else:
        env.render(mode='robot')

    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            im = env.render()
            im_observations.append(im)
        else:
            im = env.render(mode='robot')
            im_observations.append(im)
    if animated:
        env.render(close=True)

    im_observations = tensor_utils.stack_tensor_list(im_observations)

    observations = tensor_utils.stack_tensor_list(observations)

    if reward_extractor is not None:
        true_rewards = tensor_utils.stack_tensor_list(rewards)
        obs_pls_three = np.copy(im_observations)
        for iter_step in range(0, obs_pls_three.shape[0]):  # cant figure out how to do this with indexing.
            idx_plus_three = min(iter_step+3, obs_pls_three.shape[0]-1)
            obs_pls_three[iter_step, :, :, :] = im_observations[idx_plus_three, :, :, :]
        rewards = reward_extractor.get_reward(data=[im_observations, obs_pls_three], softmax=True)[:, 0]  # this is the prob of being an expert.
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
