#!/usr/bin/env python
"""
Load a snapshotted agent from an hdf5 file and animate it's behavior. Save state action pairs to pickle file with image snapshots.
NOTE: taken and modified heavily from https://raw.githubusercontent.com/joschu/modular_rl/master/sim_agent.py
"""

import argparse
import pickle, h5py, numpy as np, time
from collections import defaultdict
import gym

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
    for i in range(args.num_rollouts):
        infos = animate_rollout(env,agent,n_timesteps=timestep_limit,
            delay=1.0/env.metadata.get('video.frames_per_second', 30), collect_images=args.collect_images)
        rollouts.append(infos)
        # raw_input("press enter to continue")

    pickle.dump(rollouts, open("expert_rollouts_%s.o"%args.hdf.split("/")[-1], "w"))

if __name__ == "__main__":
    main()
