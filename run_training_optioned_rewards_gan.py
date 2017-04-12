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
from guided_cost_search.cost_ioc_tf import GuidedCostLearningTrainer
from gan.gan_trainer import GANCostTrainer
from gan.gan_trainer_with_options import GANCostTrainerWithRewardOptions

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

# TODO: hack to generically load dimensions of structuresx
obs_dims = len(expert_rollouts[0]['observations'][0])
traj_len = len(expert_rollouts[0]['observations'])

# import pdb; pdb.set_trace()

with tf.Session() as sess:

    num_frames = 1

    cost_trainer = GANCostTrainerWithRewardOptions([num_frames, obs_dims])

    trainer = Trainer(env=env, sess=sess, cost_approximator=cost_trainer, cost_trainer=cost_trainer, novice_policy=policy, novice_policy_optimizer=algo, num_frames=num_frames)
    sess.run(tf.initialize_all_variables())

    iterations = 100

    for iter_step in range(0, iterations):
        trainer.step(expert_rollouts=expert_rollouts, dump_datapoints=True)

with open(args.trained_policy_pickle_path, "wb") as output_file:
    pickle.dump(policy, output_file)
