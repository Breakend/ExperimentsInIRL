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

PRECISION = 1 # 1 decimal points for continuous states

# TODO: REFACTOR

class irlAgent(object):

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
        self.weights = [env.observation_space.shape[0]] # TODO: Unsafe
        self.policy = {}
        self.q = {}
        self.prev_action = 0.0
        self.prev_obs = 0.0

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


# function from https://github.com/wojzaremba/trpo/blob/master/utils.py
def rollout(env, agent, max_pathlength, n_timesteps):
    paths = []
    timesteps_sofar = 0
    q = {}
    while timesteps_sofar < n_timesteps:
        obs, actions, rewards = [], [], []
        terminated = False
        ob = env.reset()
        ob = np.around(ob, PRECISION)
        agent.prev_action *= 0.0
        agent.prev_obs *= 0.0
        for j in range(max_pathlength):
            action = agent.act(ob)
            obs.append(ob)
            actions.append(action)
            res = env.step(action)
            ob = res[0]
            ob = np.around(ob, PRECISION)
            rewards.append(res[1])
            ob = tuple(ob)
            if ob in q and len(q[ob])>0:
                q[ob][action] = q[ob][action] + agent.gamma**j * res[1]
            else:
                q[ob] = []
                q[ob].append(0)
                q[ob].append(0)
                q[ob][action] = agent.gamma**j * res[1]

            if res[2]:
                terminated = True
                env.reset()
                break

        path = {"obs": np.concatenate(np.expand_dims(obs, 0)),
                "rewards": np.array(rewards),
                "actions": np.array(actions),
                "terminated": terminated}
        print np.mean(rewards)
        paths.append(path)
        agent.prev_action *= 0.0
        agent.prev_obs *= 0.0
        timesteps_sofar += len(path["rewards"])
    for i in q.keys():
        # TODO: Catered to cartpole
        q[i][0] = q[i][0]/len(rewards)
        if len(q[i]) > 1:
            q[i][1] = q[i][1]/len(rewards)
        else:
            q[i].append(0)
    return q


def play(agent, env):

    # game_state = carmunk.GameState(weights)
    ob = env.reset()
    ob, r, done, _ = env.step(action)

    featureExpectations = np.zeros(len(agent.weights))
    t = 0

    while True:
        # Choose action.
        action = (np.argmax(model.predict(state, batch_size=1)))
        #print ("Action ", action)

        # Take action.
        # immediateReward , state, readings = game_state.frame_step(action)
        ob, r, terminated = env.step(action)

        featureExpectations += (agent.gamma**t)*np.array(ob)
        t += 1

        if terminated:
            break

    return featureExpectations

def compute_expert_FE:

def irl(agent, env):
    # 1. Randomly pick some policy π(0), compute (or approximate via Monte Carlo)
    # µ(0) = µ(π(0)), and set i = 1.

    # 2. Compute t(i) = maxw:kwk2≤1 minj∈{0..(i−1)} wT(µE −µ(j)),
    # and let w(i) be the value of w that attains this maximum.
    # 3. If t(i) ≤ , then terminate.
    # 4. Using the RL algorithm, compute the optimal policy π(i) for
    # the MDP using rewards R = (w(i))T φ.
    # 5. Compute (or estimate) µ(i) = µ(π(i)).
    # 6. Set i = i + 1, and go back to step 2.

# def unpack_expert(data):
#     # Iterate over each stored episode
#     for i in range(len(data)):
#

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
    agent = Agent(env)
    max_steps = env.spec.timestep_limit
    expert_data = pickle.load(open('./experts/expert_rollouts_CartPole-v0.h5.o'))

    for j in range(1000):
        agent.policy = irl(expert_data, agent, env)

        if j % 50 == 0:
            done = False
            obs = env.reset()
            totalr = 0.
            steps = 0
            while not done:
                env.render()
                action = agent.act(obs)
                action = np.argmax(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
            env.close()
            print('returns', totalr)


if __name__ == '__main__':
    main()
