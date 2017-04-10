# -*- coding: utf-8 -*-

import numpy as np
import gym
import argparse
import gym
import keras
import pickle
from cvxopt import matrix
from cvxopt import solvers #convex optimization library
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite

from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.algos.trpo import TRPO

PRECISION = 1 # 1 decimal points for continuous states

# TODO: REFACTOR

class IRLAgent(object):

    learning_rate = 0.1
    training_epochs = 200000
    batch_size = 100
    display_step = 1000
    n_rollouts = 1000
    gamma = 0.8
    train_iter = 100

    # TODO: Original repo does Q learning for RL, use TRPO instead?
    model = Sequential()
    model.add(Dense(units=32, input_dim=4))
    model.add(Activation('relu'))
    model.add(Dense(units=32))
    model.add(Activation('relu'))
    model.add(Dense(units=2))
    model.add(Activation('softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True))

    def __init__(self, env):
        self.env = env
        print("Observation Space", env.observation_space)
        print("Action Space", env.action_space)
        # Initialize array with random value
        self.weights = [np.random.uniform()]*env.observation_space.shape[0] # TODO: Probably unsafe
        self.policy = {}
        self.q = {}
        self.T = float('Inf')
        self.minT = float('Inf')


    def act(self, observation, random = True):
        observation = tuple(observation)
        if random:
            return self.env.action_space.sample()

        if observation in self.policy.keys():
            return self.policy[observation]
        else:
            # self.policy[observation] = np.argmax(self.model.predict(
            #     np.expand_dims(observation, axis=0),
            #     batch_size = 1))
            self.policy[observation] = self.env.action_space.sample()
            return int(self.policy[observation])


    def compute_weights(self, weights):
        i = 1
        while True:
            W = self.optimization() # optimize to find new weights in the list of policies
            print ("weights ::", W )
            f.write( str(W) )
            f.write('\n')
            print ("the distances  ::", self.policiesFE.keys())
            self.currentT = self.policyListUpdater(W, i)
            print ("Current distance (t) is:: ", self.currentT )
            if self.currentT <= self.epsilon: # terminate if the point reached close enough
                break
            i += 1
        f.close()
        return W

    def optimization(self): # implement the convex optimization, posed as an SVM problem
        m = len(self.expertPolicy)
        P = matrix(2.0*np.eye(m), tc='d') # min ||w||
        q = matrix(np.zeros(m), tc='d')
        policyList = [self.expertPolicy]
        h_list = [1]
        for i in self.policiesFE.keys():
            policyList.append(self.policiesFE[i])
            h_list.append(1)
        policyMat = np.matrix(policyList)
        policyMat[0] = -1*policyMat[0]
        G = matrix(policyMat, tc='d')
        h = matrix(-np.array(h_list), tc='d')
        sol = solvers.qp(P,q,G,h)

        weights = np.squeeze(np.asarray(sol['x']))
        norm = np.linalg.norm(weights)
        weights = weights/norm
        return weights # return the normalized weights



# function modified from https://github.com/wojzaremba/trpo/blob/master/utils.py
def rollout(env, agent, max_timesteps):
    feature_expect = np.zeros(len(agent.weights))

    obs, actions, rewards = [], [], []
    terminated = False
    ob = env.reset()

    for t in range(max_timesteps):
        action = agent.act(ob)
        obs.append(ob)
        actions.append(action)
        res = env.step(action)
        ob = res[0]
        rewards.append(res[1])

        if res[2]:
            terminated = True
            env.reset()
            break
        t += 1

    path = {"obs": np.concatenate(np.expand_dims(obs, 0)),
            "rewards": np.array(rewards),
            "actions": np.array(actions),
            "terminated": terminated}
    return path


'''Unrolls the input path and computes feature expectations'''
def compute_FE(path, gamma = 0.9):
    obs = path["obs"]
    feature_expect = np.zeros(len(obs[0]))
    for i, ob in enumerate(obs):
        feature_expect += (gamma**i)*np.array(ob)
    return feature_expect


def irl(env, agent, expert_data):
    max_steps = env.spec.timestep_limit
    # 1. Randomly pick some policy π(0), compute (or approximate via Monte Carlo)
    # µ(0) = µ(π(0)), and set i = 1.
    mu = []
    # First mu value, initialize random nunbers summing to 1
    path = rollout(env, agent, max_steps)
    fe_0 = compute_FE(path)
    mu.append(np.random.dirichlet(np.ones(len(agent.weights))*1000, size=1))

    # 2. Compute t(i) = maxw:kwk2≤1 minj∈{0..(i−1)} wT(µE −µ(j)),
    # and let w(i) be the value of w that attains this maximum.


    # 3. If t(i) ≤ , then terminate.
    # 4. Using the RL algorithm, compute the optimal policy π(i) for
    # the MDP using rewards R = (w(i))T φ.
    # 5. Compute (or estimate) µ(i) = µ(π(i)).
    # 6. Set i = i + 1, and go back to step 2.


def main():
    parser = argparse.ArgumentParser()
    # TODO: Add params back in
    # parser.add_argument('expert_policy_file', type=str)
    # parser.add_argument('envname', type=str)
    # parser.add_argument('--render', action='store_true')
    # parser.add_argument('--num_rollouts', type=int, default=20,
    #                     help='Number of expert roll outs')
    args = parser.parse_args()

    env = gym.make('CartPole-v0')
    agent = IRLAgent(env)
    max_steps = env.spec.timestep_limit
    expert_data = pickle.load(open('./experts/expert_rollouts_CartPole-v0.h5.o'))
    irl(env, agent, expert_data)


if __name__ == '__main__':
    main()
