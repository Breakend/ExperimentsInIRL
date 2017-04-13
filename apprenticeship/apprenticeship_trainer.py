from collections import OrderedDict
import numpy as np
import tensorflow as tf
from cvxopt import matrix, solvers
from scipy.special import expit

class ApprenticeshipCostLearningTrainer(object):

    def __init__(self, expert_paths, input_dims, gamma):
        self.dim = input_dims
        # Initialize the weights as RVs summing to 1
        self.weights = np.random.dirichlet(np.ones(input_dims)*10, size=1)
        self.weights = self.weights[0]
        print(self.weights)
        self.gamma = gamma
        self.agent_fe_data = {}
        self.currentT = float('Inf')
        self.epsilon = 0.1 # TODO: Param ?
        self.expert_fe = []
        self.first_call = True


    def get_reward(self, obs):
        r = []
        for ob in obs:
            temp_ob = (np.tanh(ob)+1)/2
            r.append(np.dot(self.weights, temp_ob))
        return r


    '''Unrolls the input path and computes feature expectations'''
    def compute_FE(self, path):
        feature_expect = np.zeros(self.dim)
        for i, ob in enumerate(path):
            temp_ob = (np.tanh(ob)+1)/2
            feature_expect += (self.gamma**i)*np.array(temp_ob)
        return feature_expect


    def train_cost(self, novice_rollouts_tensor, expert_rollouts_tensor, number_epochs=2, num_frames=1):
        # TODO: Change structure to avoid redundant computations another way
        if self.first_call:
            for path in expert_rollouts_tensor:
                self.expert_fe.append(self.compute_FE(path))
            # self.expert_fe = np.mean(self.expert_fe, 0) # Compute mean along feature columns
            self.expert_fe = self.expert_fe[-1]
            print(self.expert_fe)
            self.first_call = False

        # Update values for the previous iteration
        # import pdb; pdb.set_trace()
        novice_fe = []
        # novice_fe = self.compute_FE(novice_rollouts_tensor)
        for path in novice_rollouts_tensor:
            # We only have one path so just set it
            novice_fe = self.compute_FE(path)
        print(novice_fe)
        hyperDistance = np.abs(np.dot(self.weights, np.asarray(self.expert_fe)-np.asarray(novice_fe)))
        self.agent_fe_data[hyperDistance] = novice_fe # Add to backlog

        self.weights = self.optimize()

        print("min so far:")
        min_t = sorted(list(self.agent_fe_data.keys()))[0]
        print(min_t, self.agent_fe_data[min_t]) # Minimum distance so far


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
        sol = solvers.qp(P,q,G,h)

        weights = np.squeeze(np.asarray(sol['x']))
        norm = np.linalg.norm(weights)
        weights = weights/norm
        return weights # return the normalized weights
