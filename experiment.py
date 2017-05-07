from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
# from baselines.tf_linear_feature_baseline import LinearFeatureBaseline

from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.policies.categorical_conv_policy import CategoricalConvPolicy
from sampling_utils import load_expert_rollouts, rollout_policy, shorten_tensor_dict
import numpy as np
from rllab.misc import tensor_utils
import gym.wrappers
from tqdm import tqdm
from train import Trainer
# from gan.gan_trainer import GANCostTrainer
# from gan.gan_trainer_with_options import GANCostTrainerWithRewardOptions

import tensorflow as tf
import pickle
import argparse

def run_experiment(expert_rollout_pickle_path, trained_policy_pickle_path, env, cost_trainer_type, iterations=30, num_frames=1, traj_len=200, config={}):

    # Load the expert rollouts into memory
    expert_rollouts = load_expert_rollouts(expert_rollout_pickle_path)

    # In the case that we only have one expert rollout in the file
    if type(expert_rollouts) is dict:
        expert_rollouts = [expert_rollouts]

    #TODO: make this configurable
    expert_rollouts = [shorten_tensor_dict(x, traj_len) for x in expert_rollouts]

    # import pdb; pdb.set_trace()

    # Sanity check, TODO: should prune any "expert" rollouts with suboptimal reward?
    print("Average reward for expert rollouts: %f" % np.mean([np.sum(p['rewards']) for p in expert_rollouts]))


    if "transformers" in config and len(config["transformers"]) > 0:
        print("Transforming expert rollouts...")
        for rollout in tqdm(expert_rollouts):
            transformed_observations = []
            for ob in tqdm(rollout["observations"]):
                for transformer in config["transformers"]:
                    ob = transformer.transform(ob)
                transformed_observations.append(ob)
            rollout["observations"] = np.array(transformed_observations)

    # Handle both flattened state input and image input
    # TODO: this could be done better by looking at just the shape and determining from that
    if config["img_input"]:
        obs_dims = expert_rollouts[0]['observations'][0].shape
    else:
        # import pdb; pdb.set_trace()
        obs_dims = len(expert_rollouts[0]['observations'][0])


    if "num_novice_rollouts" in config:
        number_of_sample_trajectories = config["num_novice_rollouts"]
    else:
        number_of_sample_trajectories = len(expert_rollouts)

    print(number_of_sample_trajectories)

    # Choose a policy (Conv based on images, mlp based on states)
    # TODO: may also have to switch out categorical for something else in continuous state spaces??
    # Let's just avoid that for now?
    if config["img_input"]: # TODO: unclear right now if this even works ok. get poor results early on.
        policy = CategoricalConvPolicy(
            name="policy",
            env_spec=env.spec,
            conv_filters=[32, 64, 64],
            conv_filter_sizes=[3, 3, 3],
            conv_strides=[1, 1, 1],
            conv_pads=['SAME', 'SAME', 'SAME'],
            # The neural network policy should have two hidden layers, each with 100 hidden units each (see RLGAN paper)
            hidden_sizes=[200, 200]
        )
    elif env.spec.action_space == 'Discrete':
        policy = CategoricalMLPPolicy(
            name="policy",
            env_spec=env.spec,
            # The neural network policy should have two hidden layers, each with 100 hidden units each (see RLGAN paper)
            hidden_sizes=(400, 300)
        )
    else:
        policy = GaussianMLPPolicy(
            name="policy",
            env_spec=env.spec,
            hidden_sizes=(100, 50, 25)
        )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=number_of_sample_trajectories*traj_len, # This is actually used internally by the sampler. We make use of this sampler to generate our samples, hence we pass it here
        max_path_length=traj_len, # same with this value. A cleaner way may be to create our own sampler, but for now doing it this way..
        n_itr=40,
        discount=0.995,
        step_size=0.01,
        optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5), max_backtracks=40)
    )


    # Prune the number of rollouts if that option is enabled
    if "num_expert_rollouts" in config:
        rollouts_to_use = min(config["num_expert_rollouts"], len(expert_rollouts))
        expert_rollouts = expert_rollouts[:rollouts_to_use]
        print("Only using %d expert rollouts" % rollouts_to_use)

    true_rewards = []
    actual_rewards = []

    # Extract observations to a tensor
    expert_rollouts_tensor = tensor_utils.stack_tensor_list([path["observations"] for path in expert_rollouts])

    if "oversample" in config and config["oversample"]:
        oversample_rate = max(int(number_of_sample_trajectories / len(expert_rollouts_tensor)), 1.)
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

            # run a rollout for the video
            if "recording_env" in config:
                novice_rollouts = rollout_policy(policy, config["recording_env"], get_image_observations=False, max_path_length=200)

        novice_rollouts = algo.obtain_samples(iter_step)

        rollout_rewards = [np.sum(x['rewards']) for x in novice_rollouts]

        print("Reward stats for final policy: %f +/- %f " % (np.mean(rollout_rewards), np.std(rollout_rewards)))

        algo.shutdown_worker()

        # save the novice policy learned
        with open(trained_policy_pickle_path, "wb") as output_file:
            pickle.dump(policy, output_file)

        # TODO: also save the reward function?


    return true_rewards, actual_rewards
