from sampling_utils import *
from rllab.sampler.base import BaseSampler

class Trainer(object):

    def __init__(self, sess, cost_approximator, cost_trainer, novice_policy, novice_policy_optimizer):
        """
        sess : tensorflow session
        cost_approximator : the NN or whatever cost function that can take in your observations/states and then give you your reward
        cost_trainer : this is the trainer for optimizing the cost (i.e. runs tensorflow training ops, etc.)
        novice_policy : the policy of your novice agent
        novice_policy_optimizer : the optimizer which runs a policy optimization step (or constrained number of iterations)
        much of this can be found in https://github.com/bstadie/third_person_im/blob/master/sandbox/bradly/third_person/algos/cyberpunk_trainer.py#L164
        """
        self.sess = sess
        self.cost_approximator = cost_approximator
        self.cost_trainer = cost_trainer
        self.iteration = 0
        self.novice_policy = novice_policy
        self.novice_policy_optimizer = novice_policy_optimizer
        self.sampler = BaseSampler(self.novice_policy_training_algo)

    def step(self, expert_rollouts, expert_horizon=200):

        # collect samples for novice policy, TODO: use cost to get rewards based on current cost
        novice_rollouts = sample_policy_trajectories(self.novice_policy, len(expert_rollouts), expert_horizon)

        self.cost_trainer.train_cost(novice_rollouts, expert_rollouts, number_epochs=2)

        # This does things like calculate advantages and entropy, etc.
        policy_training_samples = self.sampler.process_samples(itr=self.iteration, paths=novice_rollouts)

        #optimize the novice policy by one step
        self.novice_policy_optimizer.optimize_policy(itr=self.iteration, samples_data=policy_training_samples)

        self.iteration += 1
        print("Training Iteration (Full Novice Rollouts): %d" % self.iteration)
