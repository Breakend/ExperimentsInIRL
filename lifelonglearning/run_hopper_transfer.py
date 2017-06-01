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
from sandbox.rocky.tf.spaces.discrete import Discrete
from policies.gaussian_decomposed_policy import GaussianDecomposedPolicy

from sampling_utils import sample_policy_trajectories
import pickle
import tensorflow as tf
from envs.transfer.register_envs import *
from gan.nn_utils import custom_train
import numpy as np
import argparse
from policies.categorical_decomposed_policy import CategoricalDecomposedPolicy

from rllab.misc import ext
# ext.set_seed(124)

parser = argparse.ArgumentParser()
# parser.add_argument("env")
# parser.add_argument("expert_rollout_pickle_path")
parser.add_argument("num_iters", type=int)
parser.add_argument("--run_baseline", action="store_true")
parser.add_argument("--use_ec2", action="store_true")
parser.add_argument("--data_dir", default="./data")
parser.add_argument("--dont_terminate_machine", action="store_false", help="Whether to terminate your spot instance or not. Be careful.")
parser.add_argument("--batch_size", type=int, default=50000)
# parser.add_argument("num_variations", type=int)
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

gymenv =  GymEnv("Hopper-v1", force_reset=True)
env = TfEnv(normalize(gymenv, normalize_obs=False))


if args.run_baseline:
    if type(env.spec.action_space) is Discrete:
        policy = CategoricalMLPPolicy(
            name="policy",
            env_spec=env.spec,
            # The neural network policy should have two hidden layers, each with 32 hidden units.
            hidden_sizes=(32, 32),
        )
    else:
        policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(100, 50, 25),
        hidden_nonlinearity=tf.nn.relu,
        )
else:
    if type(env.spec.action_space) is Discrete:
        policy = CategoricalDecomposedPolicy(
        name="policy",
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(8, 8),
        )
    else:
        policy = GaussianDecomposedPolicy(
        name="policy",
        env_spec=env.spec,
        hidden_sizes=(50, 25, 10),
        hidden_nonlinearity=tf.nn.relu,
        num_options = 4
        )

baseline = LinearFeatureBaseline(env_spec=env.spec)

variations = [GymEnv("Hopper-v1", force_reset=True), GymEnv("HopperWall-v0", force_reset=True)]

with tf.Session() as sess:

    for i in range(0, len(variations)):
        print("Variation %d" % i)

        env = TfEnv(normalize(variations[i], normalize_obs=False))

        iters = args.num_iters

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=args.batch_size, # Mujoco tasks need 20000-50000
            max_path_length=env.horizon, # And 500
            n_itr=iters,
            discount=0.99,
            step_size=0.01,
            optimizer=ConjugateGradientOptimizer(reg_coeff=0.1, hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
        )

        custom_train(algo, sess=sess)

        rollouts = algo.obtain_samples(iters+1)
        from sampling_utils import rollout_policy

        for i in range(20):
            animated = rollout_policy(policy, env, max_path_length=env.horizon, reward_extractor=None, get_image_observations=True, animated=True)
        print("Average reward for expert rollouts: %f" % np.mean([np.sum(p['rewards']) for p in rollouts]))

# import pdb; pdb.set_trace()

# with open(args.expert_rollout_pickle_path, "wb") as output_file:
#     pickle.dump(rollouts, output_file)
