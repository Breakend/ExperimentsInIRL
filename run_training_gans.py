import matplotlib as mpl
mpl.use('Agg')

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

from rllab.misc import ext
ext.set_seed(1)


from train import Trainer
from gan.gan_trainer import GANCostTrainer
from gan.wgan_trainer import WGANCostTrainer
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
parser.add_argument("--importance_weights", default=0.5, type=float)
parser.add_argument("--algorithm", default="rlgan")
parser.add_argument("--env", default="CartPole-v0")
parser.add_argument("--iterations", default=30, type=int)
parser.add_argument("--num_expert_rollouts", default=20, type=int)
parser.add_argument("--num_novice_rollouts", default=None, type=int, help="Default none means that it\'ll match the number of expert rollouts.")
parser.add_argument("--policy_opt_steps_per_global_step", default=1, type=int)
parser.add_argument("--policy_opt_learning_schedule", action="store_true")
parser.add_argument("--max_path_length", default=200, type=int)
args = parser.parse_args()

# TODO: clean this up
arg_to_cost_trainer_map = {"rlgan" : GANCostTrainer,
                           "optiongan" : GANCostTrainerWithRewardOptions,
                           "mixgan" : GANCostTrainerWithRewardMixtures,
                           "wgan": WGANCostTrainer}

if args.algorithm not in arg_to_cost_trainer_map.keys():
    raise Exception("Algorithm not supported must be one of " + arg_to_cost_trainer_map.keys())

# Need to wrap in a tf environment and force_reset to true
# see https://github.com/openai/rllab/issues/87#issuecomment-282519288
gymenv = GymEnv(args.env, force_reset=True)
gymenv.env.seed(1)
env = TfEnv(normalize(gymenv))

#TODO: don't do this, should just eat args into config
config = {}
config["importance_weights"] = args.importance_weights
config["num_expert_rollouts"] = args.num_expert_rollouts
config["num_novice_rollouts"] = args.num_novice_rollouts
config["policy_opt_steps_per_global_step"] = args.policy_opt_steps_per_global_step
config["policy_opt_learning_schedule"] = args.policy_opt_learning_schedule
# config["replay_old_samples"] = args.replay_old_samples

# average results over 10 experiments
true_rewards = []
for i in range(args.num_experiments):
    print("Running Experiment %d" % i)
    with tf.variable_scope('sess_%d'%i):
        true_rewards_exp, actual_rewards_exp = run_experiment(args.expert_rollout_pickle_path,
                                                              args.trained_policy_pickle_path,
                                                              env,
                                                              arg_to_cost_trainer_map[args.algorithm],
                                                              iterations=args.iterations,
                                                              num_frames=args.num_frames,
                                                              config=config,
                                                              traj_len=args.max_path_length)
        true_rewards.append(true_rewards_exp)

    avg_true_rewards = np.mean(true_rewards, axis=0)
    true_rewards_variance = np.var(true_rewards, axis=0)
    true_rewards_std = np.sqrt(true_rewards_variance)

    lr_flag = "lrschedule" if args.policy_opt_learning_schedule else "nolrschedule"

    with open("%s_%s_i%f_e%d_f%d_er%d_nr%d_%s_rewards_data.pickle" % (args.algorithm, args.env, args.importance_weights, args.num_experiments, args.num_frames, args.num_expert_rollouts, args.num_novice_rollouts, lr_flag), "wb") as output_file:
        pickle.dump(dict(avg=avg_true_rewards, var=true_rewards_variance), output_file)

    #TODO: add variance

    fig = plt.figure()
    plt.plot(avg_true_rewards)
    plt.xlabel('Training iterations', fontsize=18)
    plt.fill_between(np.arange(len(avg_true_rewards)), avg_true_rewards-true_rewards_std, avg_true_rewards+true_rewards_std,
        alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
        linewidth=4, linestyle='dashdot', antialiased=True)

    plt.ylabel('Average True Reward', fontsize=16)
    # plt.legend()
    fig.suptitle('True Reward over Training Iterations')
    fig.savefig('true_reward_option_%s_%s_i%f_e%d_f%d_er%d_nr%d_%s.png' % (args.algorithm, args.env, args.importance_weights, args.num_experiments, args.num_frames, args.num_expert_rollouts, args.num_novice_rollouts, lr_flag))
    plt.clf()
