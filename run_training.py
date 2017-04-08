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

from train import Trainer

import tensorflow as tf

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("expert_rollout_pickle_path")
args = parser.parse_args()

# Need to wrap in a tf environment and force_reset to true
# see https://github.com/openai/rllab/issues/87#issuecomment-282519288
env = TfEnv(normalize(GymEnv("CartPole-v0", force_reset=True)))

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

expert_rollouts = load_expert_rollouts(args.expert_rollout_pickle_path)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    trainer = Trainer(env=env, sess=sess, cost_approximator=None, cost_trainer=None, novice_policy=policy, novice_policy_optimizer=algo)

    iterations = 10

    for iter_step in range(0, iterations):
        trainer.step(expert_rollouts=expert_rollouts)
