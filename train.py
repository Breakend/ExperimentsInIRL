from sampling_utils import *
from rllab.sampler.base import BaseSampler
from rllab.misc import tensor_utils

class Trainer(object):

    def __init__(self, sess, env, cost_approximator, cost_trainer, novice_policy, novice_policy_optimizer):
        """
        sess : tensorflow session
        cost_approximator : the NN or whatever cost function that can take in your observations/states and then give you your reward
        cost_trainer : this is the trainer for optimizing the cost (i.e. runs tensorflow training ops, etc.)
        novice_policy : the policy of your novice agent
        novice_policy_optimizer : the optimizer which runs a policy optimization step (or constrained number of iterations)
        much of this can be found in https://github.com/bstadie/third_person_im/blob/master/sandbox/bradly/third_person/algos/cyberpunk_trainer.py#L164
        """
        self.sess = sess
        self.env = env
        self.cost_approximator = cost_approximator
        self.cost_trainer = cost_trainer
        self.iteration = 0
        self.novice_policy = novice_policy
        self.novice_policy_optimizer = novice_policy_optimizer
        self.sampler = BaseSampler(self.novice_policy_optimizer)

    def step(self, expert_rollouts, expert_horizon=200):

        # collect samples for novice policy
        # TODO: use cost to get rewards based on current cost, that is the rewards returned as part of the Rollouts
        #       will be from the cost function
        novice_rollouts = sample_policy_trajectories(policy=self.novice_policy, number_of_trajectories=len(expert_rollouts), env=self.env, horizon=expert_horizon, reward_extractor=self.cost_approximator)

        # import pdb; pdb.set_trace()
        print("True Reward: %f" % np.mean([np.sum(p['true_rewards']) for p in novice_rollouts]))
        print("Actual Reward: %f" % np.mean([np.sum(p['rewards']) for p in novice_rollouts]))

        # if we're using
        if self.cost_trainer:

            # Novice rollouts gets all the rewards, etc. used for policy optimization, for the cost function we just want to use the observations.
            # use "observations for the observations/states provided by the env, use "im_observations" to use the pixels (if available)
            novice_rollouts_tensor = tensor_utils.stack_tensor_list([p['observations'] for p in novice_rollouts])
            expert_rollouts_tensor = tensor_utils.stack_tensor_list([p['observations'] for p in expert_rollouts])

            self.cost_trainer.train_cost(novice_rollouts_tensor, expert_rollouts_tensor, number_epochs=2)

        # This does things like calculate advantages and entropy, etc.
        # if we use the cost function when acquiring the novice rollouts, this will use our cost function
        # for optimizing the trajectories
        policy_training_samples = self.sampler.process_samples(itr=self.iteration, paths=novice_rollouts)

        # optimize the novice policy by one step
        self.novice_policy_optimizer.optimize_policy(itr=self.iteration, samples_data=policy_training_samples)

        self.iteration += 1
        print("Training Iteration (Full Novice Rollouts): %d" % self.iteration)
