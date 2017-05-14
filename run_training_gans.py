import matplotlib as mpl
mpl.use('Agg')

from sandbox.rocky.tf.algos.trpo import TRPO
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
from gan.gan_trainer_with_options import GANCostTrainerWithRewardOptions, GANCostTrainerWithRewardMixtures
from apprenticeship.apprenticeship_trainer import ApprenticeshipCostLearningTrainer

from envs.observation_transform_wrapper import ObservationTransformWrapper
from envs.transformers import ResizeImageTransformer, SimpleNormalizePixelIntensitiesTransformer, RandomSensorMaskTransformer
from envs.tf_transformers import InceptionTransformer

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
# parser.add_argument("--policy_opt_steps_per_global_step", default=1, type=int)
# parser.add_argument("--policy_opt_learning_schedule", action="store_true")
parser.add_argument("--max_path_length", default=-1, type=int)
parser.add_argument("--record_video_sample_for_rollout", action="store_true")
parser.add_argument("--regularize_observation_space", action="store_true")
parser.add_argument("--oversample_expert", action="store_true")
parser.add_argument("--entropy_penalty", default=0.0, type=float)
parser.add_argument("--use_cv_penalty", action="store_true")
parser.add_argument("--use_mutual_info_penalty_nn_paper", action="store_true")
parser.add_argument("--use_mutual_info_penalty_infogan", action="store_true")
parser.add_argument("--img_input", action="store_true", help="The observation space of the environment is images.")
parser.add_argument("--policy_opt_batch_size", default=2000, type=int, help="Batch size of the features to feed into the policy optimization step.")
parser.add_argument("--inception_transformer_checkpoint_path", help="If you want to use the inception transformer provide a checkpoint path.")
parser.add_argument("--generate_option_graphs", action="store_true")
parser.add_argument("--add_sensor_occlusion_to_experts", action="store_true")
parser.add_argument("--second_env", default=None)
parser.add_argument("--use_prev_options_relearn_mixing_func", action="store_true")
parser.add_argument("--use_gaussian_noise_on_eval", action="store_true")
parser.add_argument("--num_extra_options_on_transfer", default=0, type=int)

args = parser.parse_args()

# TODO: clean this up
arg_to_cost_trainer_map = {"rlgan" : GANCostTrainer,
                           "optiongan" : GANCostTrainerWithRewardOptions,
                           "mixgan" : GANCostTrainerWithRewardMixtures,
                           "apprenticeship": ApprenticeshipCostLearningTrainer}

if args.algorithm not in arg_to_cost_trainer_map.keys():
    raise Exception("Algorithm not supported must be one of " + arg_to_cost_trainer_map.keys())

# Need to wrap in a tf environment and force_reset to true
# see https://github.com/openai/rllab/issues/87#issuecomment-282519288

gymenv = GymEnv(args.env, force_reset=True)

gymenv.env.seed(1)

gym_env_max_length = gymenv.horizon

max_path_length = args.max_path_length

if args.max_path_length <= 0:
    max_path_length = gym_env_max_length

print("Using a Maximum Path Length of %d" % max_path_length)

config = {}
config["img_input"] = args.img_input # TODO: also force any classic envs to use image inputs as well

def _transform_env(gymenv):
    if args.img_input:
        if args.inception_transformer_checkpoint_path:
            inception_t = InceptionTransformer(gymenv.spec.observation_space.low.shape, args.inception_transformer_checkpoint_path)
            transformers = [SimpleNormalizePixelIntensitiesTransformer(), inception_t]
            # hack, right now inception outputs (1,N) so we want to treat this as simply a large state space
            config["img_input"] = False
        else:
            transformers = [SimpleNormalizePixelIntensitiesTransformer(), ResizeImageTransformer(fraction_of_current_size=.35)]
        config["transformers"] = transformers
        transformed_env = ObservationTransformWrapper(gymenv, transformers)
    else:
        reg_obs = True if args.regularize_observation_space else False #is this necessary?
        transformed_env = normalize(gymenv, normalize_obs=reg_obs)

    if args.add_sensor_occlusion_to_experts and not args.img_input and not args.inception_transformer_checkpoint_path:
        transformers = [RandomSensorMaskTransformer(gymenv)]
        config["transformers"] = transformers

    env = TfEnv(transformed_env)
    return env

env = _transform_env(gymenv)

#TODO: don't do this, should just eat args into config
config["algorithm"] = args.algorithm
config["importance_weights"] = args.importance_weights
config["num_expert_rollouts"] = args.num_expert_rollouts
config["num_novice_rollouts"] = args.num_novice_rollouts
# config["policy_opt_steps_per_global_step"] = args.policy_opt_steps_per_global_step
# config["policy_opt_learning_schedule"] = args.policy_opt_learning_schedule
config["oversample"] = args.oversample_expert
config["entropy_penalty"] = args.entropy_penalty
config["use_mutual_info_penalty_nn_paper"] = args.use_mutual_info_penalty_nn_paper
config["use_mutual_info_penalty_infogan"] = args.use_mutual_info_penalty_infogan
config["use_cv_penalty"] = args.use_cv_penalty
config["policy_opt_batch_size"] = args.policy_opt_batch_size
config["generate_option_graphs"] = args.generate_option_graphs
config["use_gaussian_noise_on_eval"] = args.use_gaussian_noise_on_eval
config["num_extra_options_on_transfer"] = args.num_extra_options_on_transfer

## Transfer learning params
if args.second_env:
    gymenv2 = GymEnv(args.second_env, force_reset=True)
    gymenv2.env.seed(1)
    config["second_env"] = _transform_env(gymenv2)
else:
    config["second_env"] = None

config["use_prev_options_relearn_mixing_func"] = args.use_prev_options_relearn_mixing_func

if args.record_video_sample_for_rollout:
    config["recording_env"] = GymEnv(args.env, force_reset=True, record_video=True, log_dir="./data/")

# We need to know if this env has bad short runs or not. i.e. mountain car ending early is good, but cartpole ending early is bad
# We don't make this an arg so people don't accidentally forget.
bad_short_runs_mapping = {"MountainCar-v0" : False, "CartPole-v0": True,
                          "Seaquest-v0": True, "InvertedPendulum-v1":True,
                          "Hopper-v1":True, "Humanoid-v1":True,
                          "HalfCheetah-v1":True, "Ant-v1":True,
                          "Reacher-v1":False, "Walker2d-v1":True}

if args.env not in bad_short_runs_mapping.keys():
    raise Exception("Env %s not supported. Supported envs: %s" % (args.env, ", ".join(bad_short_runs_mapping.keys())))

config['short_run_is_bad'] = bad_short_runs_mapping[args.env]

# config["replay_old_samples"] = args.replay_old_samples

# average results over 10 experiments
true_rewards = []
for i in range(args.num_experiments):
    print("Running Experiment %d" % i)
    with tf.variable_scope('sess_%d'%i):
        true_rewards_exp, actual_rewards_exp, _, _ = run_experiment(args.expert_rollout_pickle_path,
                                                              args.trained_policy_pickle_path,
                                                              env,
                                                              arg_to_cost_trainer_map[args.algorithm],
                                                              iterations=args.iterations,
                                                              num_frames=args.num_frames,
                                                              config=config,
                                                              traj_len=max_path_length)
        true_rewards.append(true_rewards_exp)

    avg_true_rewards = np.mean(true_rewards, axis=0)
    true_rewards_variance = np.var(true_rewards, axis=0)
    true_rewards_std = np.sqrt(true_rewards_variance)

    lr_flag = "nolrschedule"

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
