#!/usr/bin/env python
from __future__ import print_function

import sys, gym
import time
import numpy as np
from gym.envs.classic_control import rendering

import pickle
#
# Test yourself as a learning agent! Pass environment name as a command-line argument.
#

env = gym.make('Seaquest-v0' if len(sys.argv)<2 else sys.argv[1])

# 3 - left
# 4 - right
# 5  - down
# 2 - up
# 1 - shoot

if not hasattr(env.action_space, 'n'):
    raise Exception('Keyboard agent only supports discrete action spaces')
ACTIONS = env.action_space.n
print("Actions %d" % ACTIONS)
ROLLOUT_TIME = 100000
SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                    # can test what skip is still usable.

human_agent_action = 0
human_wants_restart = False
human_sets_pause = False

def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key==0xff0d: human_wants_restart = True
    if key==65506: human_sets_pause = not human_sets_pause
    # print(key)

    key_map = {'a' : 4, 's':5, 'w':2, 'd':3, ' ': 1}
    if chr(key) in key_map:
        a = key_map[chr(key)]
    else:
        return

    # a =
    if a <= 0 or a >= ACTIONS: return
    human_agent_action = a

def key_release(key, mod):
    global human_agent_action
    key_map = {'a' : 4, 's':5, 'w':2, 'd':3, ' ': 1}
    if chr(key) in key_map:
        a = key_map[chr(key)]
    else:
        return
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0



observations = []
rewards = []


def repeat_upsample(rgb_array, k=1, l=1, err=[]):
    # repeat kinda crashes if k/l are zero
    if k <= 0 or l <= 0:
        if not err:
            print("Number of repeats must be larger than 0, k: {}, l: {}, returning default array!".format(k, l))
            err.append('logged')
        return rgb_array

    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)

viewer = rendering.SimpleImageViewer()
rgb = env.render('rgb_array')
upscaled = repeat_upsample(rgb, 2, 2)
viewer.imshow(upscaled)
viewer.window.on_key_press = key_press
viewer.window.on_key_release = key_release

def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    for t in range(ROLLOUT_TIME):
        if not skip:
            #print("taking action {}".format(human_agent_action))
            a = human_agent_action
            skip = SKIP_CONTROL
        else:
            skip -= 1

        obser, r, done, info = env.step(a)
        observations.append(obser)
        rewards.append(r)
        rgb = env.render('rgb_array')
        upscaled = repeat_upsample(rgb, 2, 2)
        viewer.imshow(upscaled)
        if done: break
        if human_wants_restart: break
        while human_sets_pause:
            rgb = env.render('rgb_array')
            upscaled = repeat_upsample(rgb, 2, 2)
            viewer.imshow(upscaled)
            time.sleep(0.1)

# print("ACTIONS={}".format(ACTIONS))
print("Press awsd to move and space bar to shoot! Don't forget to collect a diver before trying to get oxygen again!")
# print("No keys pressed is taking action 0")

rollout(env)

timestr = time.strftime("%Y%m%d-%H%M%S")
reward = np.sum(rewards)

with open("%s-human-demo-reward-%s.pickle" % (timestr, reward), "wb") as output_file:
    pickle.dump(dict(observations=observations), output_file)
