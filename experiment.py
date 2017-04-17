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

from train import Trainer
from guided_cost_search.cost_ioc_tf import GuidedCostLearningTrainer
from gan.gan_trainer import GANCostTrainer
from gan.gan_trainer_with_options import GANCostTrainerWithRewardOptions

import tensorflow as tf
import pickle
import argparse

def run_experiment(expert_rollout_pickle_path, trained_policy_pickle_path, env, cost_trainer_type, iterations=30, num_frames=1, config={}):

    policy = CategoricalMLPPolicy(
    name="policy",
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=100,
        n_itr=40,
        discount=0.99,
        step_size=0.01,
        optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))

    )

    expert_rollouts = load_expert_rollouts(expert_rollout_pickle_path)

    # Sanity check, TODO: should prune any "expert" rollouts with suboptimal reward?
    print("Average reward for expert rollouts: %f" % np.mean([np.sum(p['true_rewards']) for p in expert_rollouts]))
    # import pdb; pdb.set_trace()

    # TODO: hack to generically load dimensions of structuresx
    # import pdb; pdb.set_trace()
    obs_dims = len(expert_rollouts[0]['observations'][0])
    traj_len = len(expert_rollouts[0]['observations'])

    # import pdb; pdb.set_trace()
    true_rewards = []
    actual_rewards = []

    with tf.Session() as sess:

        cost_trainer = cost_trainer_type([num_frames, obs_dims], config=config)

        trainer = Trainer(env=env, sess=sess, cost_approximator=cost_trainer, cost_trainer=cost_trainer, novice_policy=policy, novice_policy_optimizer=algo, num_frames=num_frames)
        sess.run(tf.global_variables_initializer())


        for iter_step in range(0, iterations):
            dump_data = (iter_step == (iterations-1)) # is last iteration
            true_reward, actual_reward = trainer.step(expert_rollouts=expert_rollouts, dump_datapoints=dump_data)
            true_rewards.append(true_reward)
            actual_rewards.append(actual_reward)

        # TODO: should we overwrite the policy
        with open(trained_policy_pickle_path, "wb") as output_file:
            pickle.dump(policy, output_file)

    return true_rewards, actual_rewards
