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
from gan.gan_trainer import GANCostTrainer

from experiment import *
import tensorflow as tf
import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("expert_rollout_pickle_path")
parser.add_argument("trained_policy_pickle_path")
args = parser.parse_args()

# Need to wrap in a tf environment and force_reset to true
# see https://github.com/openai/rllab/issues/87#issuecomment-282519288
env = TfEnv(normalize(GymEnv("CartPole-v0", force_reset=True)))

# average results over 10 experiments
true_rewards = []
for i in range(2):
    print("Running Experiment %d" % i)
    with tf.variable_scope('sess_%d'%i):
        true_rewards_exp, actual_rewards_exp = run_experiment(args.expert_rollout_pickle_path, args.trained_policy_pickle_path, env, GANCostTrainer)
        true_rewards.append(true_rewards_exp)

avg_true_rewards = np.mean(true_rewards, axis=0)

true_rewards_variance = np.var(true_rewards, axis=0)

with open("rlgan_rewards_data.pickle", "wb") as output_file:
    pickle.dump(dict(avg=avg_true_rewards, var=true_rewards_variance), output_file)

fig = plt.figure()
plt.plot(avg_true_rewards)
plt.xlabel('Training iterations', fontsize=18)
plt.ylabel('Average True Reward', fontsize=16)
# plt.legend()
fig.suptitle('True Reward over Training Iterations')
fig.savefig('true_reward_gan.png')
plt.clf()
