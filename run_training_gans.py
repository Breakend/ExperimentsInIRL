from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sampling_utils import load_expert_rollouts
import numpy as np
import matplotlib.pyplot as plt


from train import Trainer
from guided_cost_search.cost_ioc_tf import GuidedCostLearningTrainer
from gan.gan_trainer import GANCostTrainer
from gan.gan_trainer_with_options import GANCostTrainerWithRewardOptions, GANCostTrainerWithRewardMixtures

from experiment import *
import tensorflow as tf
import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("expert_rollout_pickle_path")
parser.add_argument("trained_policy_pickle_path")
parser.add_argument("--num_frames", default=4, type=int)
parser.add_argument("--num_experiments", default=5, type=int)
parser.add_argument("--importance_weights", default=0.0, type=float)
parser.add_argument("--algorithm", default="rlgan")
parser.add_argument("--env", default="CartPole-v0")
# parser.add_argument("--number_expert_rollouts", default=10, type=int)
# parser.add_argument("--number_novice_rollouts", default=10, type=int)
args = parser.parse_args()

# TODO: clean this up
arg_to_cost_trainer_map = {"rlgan" : GANCostTrainer, "optiongan" : GANCostTrainerWithRewardOptions, "mixgan" : GANCostTrainerWithRewardMixtures}

if args.algorithm not in arg_to_cost_trainer_map.keys():
    raise Exception("Algorithm not supported must be one of " + arg_to_cost_trainer_map.keys())

# Need to wrap in a tf environment and force_reset to true
# see https://github.com/openai/rllab/issues/87#issuecomment-282519288
env = TfEnv(normalize(GymEnv(args.env, force_reset=True)))

#TODO: move everything into the config
config = {}
config["importance_weights"] = args.importance_weights

# average results over 10 experiments
true_rewards = []
for i in range(args.num_experiments):
    print("Running Experiment %d" % i)
    with tf.variable_scope('sess_%d'%i):
        true_rewards_exp, actual_rewards_exp = run_experiment(args.expert_rollout_pickle_path, args.trained_policy_pickle_path, env, arg_to_cost_trainer_map[args.algorithm], num_frames=args.num_frames, config=config)
        true_rewards.append(true_rewards_exp)

avg_true_rewards = np.mean(true_rewards, axis=0)
true_rewards_variance = np.var(true_rewards, axis=0)

with open("%s_rewards_data.pickle" % args.algorithm, "wb") as output_file:
    pickle.dump(dict(avg=avg_true_rewards, var=true_rewards_variance), output_file)

#TODO: add variance

fig = plt.figure()
plt.plot(avg_true_rewards)
plt.xlabel('Training iterations', fontsize=18)
plt.fill_between(np.arange(len(avg_true_rewards)), avg_true_reward-true_rewards_variance, avg_true_reward+true_rewards_variance,
    alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
    linewidth=4, linestyle='dashdot', antialiased=True)

plt.ylabel('Average True Reward', fontsize=16)
# plt.legend()
fig.suptitle('True Reward over Training Iterations')
fig.savefig('true_reward_option_%s.png' % args.algorithm)
plt.clf()
