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
import time
from rllab.sampler.utils import rollout
import rllab.misc.logger as logger

from gan.nn_utils import initialize_uninitialized

from rllab.misc import ext


def __replace_rollout_in_cache(cache, rollout):
    for i, x in enumerate(cache):
        if np.sum(x['rewards']) < np.sum(rollout['rewards']):
            cache[i] = rollout
            return

def train_with_accumulation(algo, expert_rollout_pickle_path, sess=None, reward_threshold=0, num_rollouts_to_store=50):
    created_session = True if (sess is None) else False
    if sess is None:
        sess = tf.Session()
        sess.__enter__()

    rollout_cache = []
    initialize_uninitialized(sess)
    algo.start_worker()
    start_time = time.time()
    for itr in range(algo.start_itr, algo.n_itr):
        itr_start_time = time.time()
        with logger.prefix('itr #%d | ' % itr):
            logger.log("Obtaining samples...")
            paths = algo.obtain_samples(itr)
            logger.log("Processing samples...")
            samples_data = algo.process_samples(itr, paths)
            logger.log("Logging diagnostics...")
            algo.log_diagnostics(paths)
            logger.log("Optimizing policy...")
            algo.optimize_policy(itr, samples_data)
            logger.log("Saving snapshot...")
            params = algo.get_itr_snapshot(itr, samples_data)  # , **kwargs)
            if algo.store_paths:
                params["paths"] = samples_data["paths"]
            logger.save_itr_params(itr, params)
            logger.log("Saved")
            logger.record_tabular('Time', time.time() - start_time)
            logger.record_tabular('ItrTime', time.time() - itr_start_time)

            # rollouts = algo.rollout(algo.env, algo.policy, animated=False, max_path_length=algo.max_path_length)
            for x in paths:
                reward_sum = np.sum(x['rewards'])
                if reward_sum > reward_threshold:
                    if len(rollout_cache) < num_rollouts_to_store:
                        rollout_cache.append(x)
                    else:
                        __replace_rollout_in_cache(rollout_cache, x)

            logger.record_tabular("AveRewardInCache", np.mean([np.sum(x['rewards']) for x in rollout_cache]))
            logger.dump_tabular(with_prefix=False)
            with open(expert_rollout_pickle_path, "wb") as output_file:
                pickle.dump(rollout_cache, output_file)


    algo.shutdown_worker()
    if created_session:
        sess.close()

# ext.set_seed(124)

parser = argparse.ArgumentParser()
parser.add_argument("env")
parser.add_argument("expert_rollout_pickle_path")
parser.add_argument("num_iters", type=int)
parser.add_argument("--num_rollouts_to_store", default=50, type=int)
parser.add_argument("--reward_threshold", default=0, type=int)
parser.add_argument("--batch_size", default=5000, type=int)
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
env = TfEnv(normalize(gymenv, normalize_obs=False))

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
    batch_size=args.batch_size, # Mujoco tasks need 20000-50000
    max_path_length=env.horizon, # And 500
    n_itr=iters,
    discount=0.99,
    step_size=0.01,
    optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
)

with tf.Session() as sess:
    train_with_accumulation(algo,
                            args.expert_rollout_pickle_path,
                            sess=sess,
                            reward_threshold=args.reward_threshold,
                            num_rollouts_to_store=args.num_rollouts_to_store)
