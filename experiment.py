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
from rllab.misc import tensor_utils


from train import Trainer
# from gan.gan_trainer import GANCostTrainer
# from gan.gan_trainer_with_options import GANCostTrainerWithRewardOptions

import tensorflow as tf
import pickle
import argparse

def run_experiment(expert_rollout_pickle_path, trained_policy_pickle_path, env, cost_trainer_type, iterations=30, num_frames=1, traj_len=200, config={}):

    expert_rollouts = load_expert_rollouts(expert_rollout_pickle_path)
    # import pdb; pdb.set_trace()

    # TODO: hack to generically load dimensions of structuresx
    # import pdb; pdb.set_trace()
    obs_dims = len(expert_rollouts[0]['observations'][0])
    # traj_len = len(expert_rollouts[0]['observations'])

    if "num_novice_rollouts" in config:
        number_of_sample_trajectories = config["num_novice_rollouts"]
    else:
        number_of_sample_trajectories = len(expert_rollouts)
    print(number_of_sample_trajectories)
    policy = CategoricalMLPPolicy(
    name="policy",
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 100 hidden units each (see RLGAN paper)
    hidden_sizes=(100, 100)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=number_of_sample_trajectories*traj_len, # This is actually used internally by the sampler. We make use of this sampler to generate our samples, hence we pass it here
        max_path_length=traj_len, # same with this value. A cleaner way may be to create our own sampler, but for now doing it this way..
        n_itr=40,
        discount=0.99,
        step_size=0.01,
        optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
    )


    if "num_expert_rollouts" in config:
        rollouts_to_use = min(config["num_expert_rollouts"], len(expert_rollouts))
        expert_rollouts = expert_rollouts[:rollouts_to_use]
        print("Only using %d expert rollouts" % rollouts_to_use)



    # Sanity check, TODO: should prune any "expert" rollouts with suboptimal reward?
    print("Average reward for expert rollouts: %f" % np.mean([np.sum(p['rewards']) for p in expert_rollouts]))
    # import pdb; pdb.set_trace()




    # import pdb; pdb.set_trace()
    true_rewards = []
    actual_rewards = []

    oversample = True
    expert_rollouts_tensor = [path["observations"] for path in expert_rollouts]
    expert_rollouts_tensor = np.asarray([tensor_utils.pad_tensor(a, traj_len, mode='last') for a in expert_rollouts_tensor])
    # expert_rollouts_tensor = tensor_utils.pad_tensor_n(expert_rollouts_tensor, traj_len)

    if oversample:
        oversample_rate = int(number_of_sample_trajectories / len(expert_rollouts_tensor))
        expert_rollouts_tensor = expert_rollouts_tensor.repeat(oversample_rate, axis=0)
        print("oversampling %d times to %d" % (oversample_rate, len(expert_rollouts_tensor)))


    with tf.Session() as sess:
        algo.start_worker()

        cost_trainer = cost_trainer_type([num_frames, obs_dims], config=config)

        trainer = Trainer(env=env, sess=sess, cost_approximator=cost_trainer, cost_trainer=cost_trainer, novice_policy=policy, novice_policy_optimizer=algo, num_frames=num_frames)
        sess.run(tf.global_variables_initializer())


        for iter_step in range(0, iterations):
            dump_data = (iter_step == (iterations-1)) # is last iteration
            true_reward, actual_reward = trainer.step(expert_rollouts_tensor=expert_rollouts_tensor, dump_datapoints=dump_data, config=config, expert_horizon=traj_len, number_of_sample_trajectories=number_of_sample_trajectories)
            true_rewards.append(true_reward)
            actual_rewards.append(actual_reward)

        novice_rollouts = algo.obtain_samples(iter_step)

        rollout_rewards = [np.sum(x['rewards']) for x in novice_rollouts]


        print("Reward stats for final policy: %f +/- %f " % (np.mean(rollout_rewards), np.std(rollout_rewards)))


        algo.shutdown_worker()
        # TODO: should we overwrite the policy
        with open(trained_policy_pickle_path, "wb") as output_file:
            pickle.dump(policy, output_file)


    return true_rewards, actual_rewards
