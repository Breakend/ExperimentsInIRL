#!/usr/bin/env python
"""
Load a snapshotted agent from an hdf5 file and animate it's behavior. Save state action pairs to pickle file with image snapshots.
NOTE: taken and modified heavily from https://raw.githubusercontent.com/joschu/modular_rl/master/sim_agent.py
"""

import argparse
import pickle, h5py, numpy as np, time
from collections import defaultdict
import gym
from rllab.misc import tensor_utils


def animate_rollout_seq(env, agent, max_path_length=200, reward_extractor = None, speedup = 1, get_image_observations=True, num_frames=4, concat_timesteps=True):
    """
    Mostly taken from https://github.com/bstadie/third_person_im/blob/master/sandbox/bradly/third_person/algos/cyberpunk_trainer.py#L164
    Generate a rollout for a given policy
    """
    if hasattr(agent,"reset"): agent.reset()

    observations = []
    im_observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    path_length = 0

    while path_length < max_path_length:
        o = agent.obfilt(o)
        a, agent_info = agent.act(o)
        next_o, r, d, env_info = env.step(a)
        pixel_array = env.render(mode="rgb_array")

        # observations.append(env.observation_space.flatten(o))
        observations.append(o)
        if d:
            rewards.append(0.0)
        else:
            rewards.append(r)
        # actions.append(env.action_space.flatten(a))
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        o = next_o
        if get_image_observations:
            pixel_array = env.render(mode="rgb_array")
            if len(pixel_array) == 0:
                # Not convinced that behaviour works for all environments, so until
                # such a time as I'm convinced of this, drop into a debug shell
                print("Problem! Couldn't get pixels! Dropping into debug shell.")
                import pdb; pdb.set_trace()
            im_observations.append(pixel_array)

    # if animated:
    # env.render(close=True)

    im_observations = tensor_utils.stack_tensor_list(im_observations)

    observations = tensor_utils.stack_tensor_list(observations)

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

def animate_rollout(env, agent, n_timesteps, collect_images=False, delay=.01):
    infos = defaultdict(list)
    ob = env.reset()
    if hasattr(agent,"reset"): agent.reset()
    env.render()
    for i in xrange(n_timesteps):
        ob = agent.obfilt(ob)
        a, _info = agent.act(ob)
        (ob, rew, done, info) = env.step(a)
        pixel_array = env.render(mode="rgb_array")
        if pixel_array is None:
            # Not convinced that behaviour works for all environments, so until
            # such a time as I'm convinced of this, drop into a debug shell
            print("Problem! Couldn't get pixels! Dropping into debug shell.")
            import pdb; pdb.set_trace()
        if done:
            print("terminated after %s timesteps"%i)
            rew = 0
            # break
        # TODO: not sure why this was here in the first place.
        # for (k,v) in info.items():
        #     infos[k].append(v)
        infos['observations'].append(ob)
        infos['rewards'].append(rew)
        infos['true_rewards'].append(rew)
        infos['action'].append(a)
        infos['env_infos'].append(info)
        infos['agent_infos'].append(_info)
        if collect_images:
            infos['im_observations'].append(pixel_array)
        # time.sleep(delay)
    return infos

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf")
    parser.add_argument("--timestep_limit",type=int)
    parser.add_argument("--snapname")
    parser.add_argument("--collect_images", action='store_true')
    parser.add_argument("num_rollouts", type=int)
    args = parser.parse_args()

    hdf = h5py.File(args.hdf,'r')

    snapnames = hdf['agent_snapshots'].keys()
    print("snapshots:\n",snapnames)
    if args.snapname is None:
        snapname = snapnames[-1]
    elif args.snapname not in snapnames:
        raise ValueError("Invalid snapshot name %s"%args.snapname)
    else:
        snapname = args.snapname

    env = gym.make(hdf["env_id"].value)

    agent = pickle.loads(hdf['agent_snapshots'][snapname].value)
    agent.stochastic=False

    timestep_limit = args.timestep_limit or env.spec.timestep_limit

    # while True:
    rollouts = []

    for iter_step in range(0, args.num_rollouts):
        rollouts.append(animate_rollout_seq(agent=agent, env=env, max_path_length=timestep_limit, get_image_observations=args.collect_images,  num_frames=4, concat_timesteps=True))
        # rollouts.append(animate_rollout(agent=agent, env=env, n_timesteps=timestep_limit))

    pickle.dump(rollouts, open("expert_rollouts_%s.o"%args.hdf.split("/")[-1], "wb"))
    print "Dumped rollouts !"

if __name__ == "__main__":
    main()
