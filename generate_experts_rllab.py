from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite

from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp

from sampling_utils import sample_policy_trajectories
import pickle
import tensorflow as tf
from envs.transfer.register_envs import *
import numpy as np
import argparse

from rllab.misc import ext
# ext.set_seed(124)

parser = argparse.ArgumentParser()
parser.add_argument("env")
parser.add_argument("expert_rollout_pickle_path")
parser.add_argument("num_iters", type=int)
args = parser.parse_args()

# stub(globals())
#
# supported_envs = ["MountainCar-v0", "CartPole-v0"]
#
# if args.env not in supported_envs:
#     raise Exception("Env not supported! Try it out though?")

# Need to wrap in a tf environment and force_reset to true
# see https://github.com/openai/rllab/issues/87#issuecomment-282519288

register_custom_envs()

gymenv = GymEnv(args.env, force_reset=True)
# gymenv.env.seed(124)
env = TfEnv(normalize(gymenv, normalize_obs=True))

if env.spec.action_space == 'Discrete':
    policy = CategoricalMLPPolicy(
    name="policy",
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32)
    )
else:
    policy = GaussianMLPPolicy(
    name="policy",
    env_spec=env.spec,
    hidden_sizes=(100, 50, 25)
    )

baseline = LinearFeatureBaseline(env_spec=env.spec)

iters = args.num_iters

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=50000, # Mujoco tasks need 20000-50000
    max_path_length=env.horizon, # And 500
    n_itr=iters,
    discount=0.99,
    step_size=0.01,
    optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
)

# run_experiment_lite(
#     ,
#     n_parallel=1,
#     snapshot_mode="last",
#     seed=1
# )
with tf.Session() as sess:

    algo.train(sess=sess)

    rollouts = algo.obtain_samples(iters+1)
    print("Average reward for expert rollouts: %f" % np.mean([np.sum(p['rewards']) for p in rollouts]))

# import pdb; pdb.set_trace()

with open(args.expert_rollout_pickle_path, "wb") as output_file:
    pickle.dump(rollouts, output_file)
