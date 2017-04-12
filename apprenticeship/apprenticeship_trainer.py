from collections import OrderedDict
import numpy as np
import tensorflow as tf
from cvxopt import matrix, solvers

class ApprenticeshipCostLearningTrainer(object):

    def __init__(self, expert_paths, input_dims, gamma):
        # Initialize the weights as RVs summing to 1
        # self.weights = tf.Variable(np.random.dirichlet(np.ones(input_dims)*1000, size=1), name='weights')
        self.dim = input_dims
        self.weights = [np.random.uniform()]*input_dims # TODO: Probably unsafe
        self.gamma = gamma
        self.agent_fe_data = {}
        self.currentT = float('Inf')
        self.epsilon = 0.1 # TODO: Param ?
        self.expert_fe = []
        self.first_call = True

        self.last_fe=[]
        self.curr_fe = []
        self.mu_bar_prev = 0
        self.mu_bar_curr = 0

    def get_reward(self, obs):
        r = []
        for ob in obs:
            # r.append(tf.matmul(self.weights, tf.expand_dims(tf.transpose(ob), 1)).eval())
            r.append(np.dot(self.weights, ob))
            # print(ob)
            # print(self.weights)
            # print(np.dot(self.weights,np.transpose(ob)))
        return r


    '''Unrolls the input path and computes feature expectations'''
    def compute_FE(self, path):
        feature_expect = np.zeros(self.dim)
        for i, ob in enumerate(path):
            feature_expect += (self.gamma**i)*np.array(ob)
        return feature_expect


    def train_cost(self, novice_rollouts_tensor, expert_rollouts_tensor, number_epochs=2, num_frames=1):
        # TODO: Change structure to avoid redundant computations another way
        if self.first_call:
            for path in expert_rollouts_tensor:
                self.expert_fe.append(self.compute_FE(path))
            self.expert_fe = np.mean(self.expert_fe, 0) # Compute mean along feature columns
            # self.expert_fe = self.expert_fe[0]
            print(self.expert_fe)
            self.first_call = False

        # Update values for the previous iteration
        novice_fe = []
        for path in novice_rollouts_tensor:
            novice_fe.append(self.compute_FE(path))
        novice_fe = np.mean(novice_fe, 0)
        # novice_fe = novice_fe[0]
        self.curr_fe = novice_fe
        print(novice_fe)
        hyperDistance = np.abs(np.dot(self.weights, np.asarray(self.expert_fe)-np.asarray(novice_fe)))
        self.agent_fe_data[hyperDistance] = novice_fe # Add to backlog

        self.weights = self.optimize(self.expert_fe)
        # import pdb; pdb.set_trace()

        print(sorted(list(self.agent_fe_data.keys()))[0]) # Minimum distance so far


    def optimize(self, expert_fe): # implement the convex optimization, posed as an SVM problem
        # Set m to the number of expert paths we have
        m = len(expert_fe)
        P = matrix(2.0*np.eye(m), tc='d') # min ||w||
        q = matrix(np.zeros(m), tc='d')
        feature_expectations = [expert_fe]
        h_list = [1] # = labels
        for i in self.agent_fe_data.keys():
            feature_expectations.append(self.agent_fe_data[i])
            h_list.append(1)

        # Form the matrices for the quadratic solver
        policyMat = np.matrix(feature_expectations)
        policyMat[0] = -1*policyMat[0] # Flip features for expert
        G = matrix(policyMat, tc='d')
        h = matrix(np.array(h_list), tc='d')

        # Solve
        sol = solvers.qp(P,q,G,h)

        weights = np.squeeze(np.asarray(sol['x']))
        norm = np.linalg.norm(weights)
        weights = weights/norm
        return weights # return the normalized weights
