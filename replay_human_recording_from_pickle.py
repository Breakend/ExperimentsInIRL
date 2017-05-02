#!/usr/bin/env python
from __future__ import print_function

import sys, gym
import time
import numpy as np
from gym.envs.classic_control import rendering

import pickle
import argparse

parser = argparse.ArgumentParser()
#TODO: add multiple so we plot multiple lines
parser.add_argument("datapath")

args = parser.parse_args()

with open(args.datapath, "rb") as output_file:
    data = pickle.load(output_file)

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

for obs in data["observations"]:
    upscaled = repeat_upsample(obs, 2, 2)
    viewer.imshow(upscaled)
    # time.sleep(0.1)
