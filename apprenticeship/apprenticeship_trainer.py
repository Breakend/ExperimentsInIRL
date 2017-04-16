from collections import OrderedDict
import numpy as np
import sys
import tensorflow as tf
from cvxopt import matrix, solvers
from scipy.special import expit
from sklearn.preprocessing import PolynomialFeatures

from random import gauss

def make_rand_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]


class ApprenticeshipCostLearningTrainer(object):

    def __init__(self, expert_paths, input_dims, gamma):
        self.dim = input_dims
        # self.weights = np.random.uniform(low=-1.0, high=1.0, size = input_dims)
        # self.weights = make_rand_vector(input_dims)
        # self.weights = [-0.63649622, -0.17047543,  0.17078687, -0.7325589 ]
        # normalize
        self.poly_augmenter = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        temp = self.poly_augmenter.fit_transform([0] * input_dims)[0]
        self.n_features = len(temp)
        self.weights = np.random.dirichlet(np.ones(self.n_features)*1, size=1)
        self.weights = self.weights[0]
        # self.weights = self.weights / (np.linalg.norm(self.weights)+1e-8)
        self.initial_weights = self.weights
        print(self.initial_weights)
        self.gamma = gamma
        # self.agent_fe_data = {}
        self.currentT = float('Inf')
        self.epsilon = 0.0001 # TODO: Param ?
        self.expert_fe = []
        self.first_call = True
        self.novice_fe = [] # = mu
        self.novice_fe_bar = [] # = mu bar
        self.iter = 0
        self.minT = float('Inf')
        self.solved = False


    def pseudo_normalize(self, x):
        return (np.tanh(x)+1.0)/2.0


    def get_reward(self, obs):
        r = []
        for ob in obs:
            # temp_ob = self.pseudo_normalize(ob/10.0)
            temp_ob = self.poly_augmenter.fit_transform((self.pseudo_normalize(ob/10.0)).reshape(1, -1))[0]
            reward = np.dot(self.weights, temp_ob)
            # Squash the reward to fit the algorithm's assumptions
            # r.append(self.pseudo_normalize(reward))
            reward = max(1e-8, reward)
            r.append(self.pseudo_normalize(reward))
        return r


    '''Unrolls the input path and computes feature expectations'''
    def compute_FE(self, path):
        feature_expect = np.zeros(self.n_features)
        for i, ob in enumerate(path):
            temp_ob = self.poly_augmenter.fit_transform((self.pseudo_normalize(ob/10.0)).reshape(1, -1))[0]
            # temp_ob = self.pseudo_normalize(ob/10.0)
            feature_expect += (self.gamma**i)*np.array(temp_ob)
        return feature_expect


    def train_cost(self, novice_rollouts_tensor, expert_rollouts_tensor, number_epochs=2, num_frames=1):
        if self.solved:
            sys.exit()
        for path in novice_rollouts_tensor:
            # We only have one path so just set it
            novice_fe = self.compute_FE(path)

        # TODO: Change structure to avoid redundant computations some other way
        if self.first_call:
            for path in expert_rollouts_tensor:
                self.expert_fe.append(self.compute_FE(path))
            # self.expert_fe = np.mean(self.expert_fe, 0) # Compute mean along feature columns
            self.expert_fe = self.expert_fe[-1]

            print(self.expert_fe)
            self.weights = self.expert_fe - novice_fe
            self.novice_fe.append(novice_fe)
            self.novice_fe_bar.append(novice_fe)
            self.first_call = False
        else:
            self.iter += 1
            print(novice_fe)
            self.novice_fe.append(novice_fe)
            # import pdb; pdb.set_trace()
            numerator = np.dot(self.novice_fe[self.iter]-self.novice_fe_bar[self.iter-1],
                        self.expert_fe - self.novice_fe_bar[self.iter-1])
            denominator = np.dot(self.novice_fe[self.iter]-self.novice_fe_bar[self.iter-1],
                        self.novice_fe[self.iter] - self.novice_fe_bar[self.iter-1]) + 1e-8
            factor = self.novice_fe[self.iter] - self.novice_fe_bar[self.iter-1]

            self.novice_fe_bar.append(self.novice_fe_bar[self.iter-1]
                                    + factor * numerator/denominator)

            self.weights = self.expert_fe - self.novice_fe_bar[self.iter]
            self.weights = self.weights/(np.linalg.norm(self.weights) + 1e-8)
            print(self.weights)

            self.currentT = np.linalg.norm(self.expert_fe - self.novice_fe_bar[self.iter])

        # print(novice_fe)
        # hyperDistance = np.abs(np.dot(self.weights, np.asarray(self.expert_fe)-np.asarray(novice_fe)))
        # hyperDistance = np.linalg.norm(np.asarray(self.expert_fe)-np.asarray(novice_fe))
        # print(hyperDistance)
        # self.agent_fe_data[hyperDistance] = novice_fe # Add to backlog

        print("Current Distance: {}".format(self.currentT))

        if self.currentT < self.minT :
            self.minT = self.currentT
        print("min so far: {}".format(self.minT))

        # min_t = sorted(list(self.agent_fe_data.keys()))[0]
        # print(min_t, self.agent_fe_data[min_t]) # Minimum distance so far

        # TODO: This is dirty, find a better way to terminate
        if self.currentT <= self.epsilon:
            # Set a flag so we can observe the next iteration's rewards
            self.solved = True
            print(self.initial_weights)
            print(self.weights)
        # self.weights = self.optimize()


    def optimize(self): # implement the convex optimization, posed as an SVM problem
        # Set m to the number of expert paths we have
        m = len(self.expert_fe)
        P = matrix(2.0*np.eye(m), tc='d') # min ||w||
        q = matrix(np.zeros(m), tc='d')
        feature_expectations = [self.expert_fe]
        feature_expectations.extend(self.agent_fe_data.values())
        h_list = [-1.0]*len(feature_expectations)
        h_list[0] = 1.0
        # Form the matrices for the quadratic solver
        policyMat = np.matrix(feature_expectations)
        policyMat[0] = policyMat[0] # Flip features for expert
        G = matrix(policyMat, tc='d')
        h = matrix(np.array(h_list), tc='d')

        # Solve
        solvers.options['show_progress'] = False
        sol = solvers.qp(P,q,G,h)

        weights = np.squeeze(np.asarray(sol['x']))
        norm = np.linalg.norm(weights)
        weights = weights/norm
        return weights # return the normalized weights
