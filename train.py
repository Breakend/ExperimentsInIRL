from sampling_utils import *
from rllab.sampler.base import BaseSampler
from rllab.misc import tensor_utils
from rllab.policies.uniform_control_policy import UniformControlPolicy
from rllab.algos.nop import NOP
from rllab.baselines.zero_baseline import ZeroBaseline
import numpy as np

class Trainer(object):

    def __init__(self, sess, env, cost_approximator, cost_trainer, novice_policy, novice_policy_optimizer, num_frames=4, concat_timesteps=True):
        """
        sess : tensorflow session
        cost_approximator : the NN or whatever cost function that can take in your observations/states and then give you your reward
        cost_trainer : this is the trainer for optimizing the cost (i.e. runs tensorflow training ops, etc.)
        novice_policy : the policy of your novice agent
        novice_policy_optimizer : the optimizer which runs a policy optimization step (or constrained number of iterations)
        much of this can be found in https://github.com/bstadie/third_person_im/blob/master/sandbox/bradly/third_person/algos/cyberpunk_trainer.py#L164
        """
        self.sess = sess
        self.env = env
        self.cost_approximator = cost_approximator
        self.cost_trainer = cost_trainer
        self.iteration = 0
        self.novice_policy = novice_policy
        self.novice_policy_optimizer = novice_policy_optimizer
        # self.sampler = BaseSampler(self.novice_policy_optimizer)
        self.concat_timesteps = concat_timesteps
        self.num_frames = num_frames
        self.replay_buffer = {}
        self.max_replays = 3
        self.replay_index = 0
        self.replay_times = 40
        self.train_cost_per_iters = 10

        # as in traditional GANs, we add failure noise
        self.noise_fail_policy = UniformControlPolicy(env.spec)
        self.zero_baseline = ZeroBaseline(env_spec=env.spec)
        self.rand_algo = NOP(
            env=env,
            policy=self.noise_fail_policy,
            baseline=self.zero_baseline,
            batch_size=4000,
            max_path_length=500,
            n_itr=5,
            discount=0.995,
            step_size=0.01,
        )

    def step(self, expert_rollouts_tensor, expert_horizon=200, dump_datapoints=False, number_of_sample_trajectories=None, config={}):

        if number_of_sample_trajectories is None:
            number_of_sample_trajectories = len(expert_rollouts_tensor)


        # This does things like calculate advantages and entropy, etc.
        # if we use the cost function when acquiring the novice rollouts, this will use our cost function
        # for optimizing the trajectories
        # import pdb; pdb.set_trace()
        novice_rollouts = self.novice_policy_optimizer.obtain_samples(self.iteration)
        novice_rollouts = process_samples_with_reward_extractor(novice_rollouts, self.cost_approximator, self.concat_timesteps, self.num_frames)

        policy_training_samples = self.novice_policy_optimizer.process_samples(itr=self.iteration, paths=novice_rollouts)

        self.rand_algo.start_worker() # TODO: Call this in constructor instead ?
        self.rand_algo.init_opt()
        random_rollouts = self.rand_algo.sampler.obtain_samples(itr = self.iteration)
        # random_rollouts = self.rand_algo.sampler.process_samples(itr=self.iteration, paths = random_rollouts)
        # novice_rollouts = sample_policy_trajectories(policy=self.novice_policy, number_of_trajectories=number_of_sample_trajectories, env=self.env, horizon=expert_horizon, reward_extractor=self.cost_approximator, num_frames=self.num_frames, concat_timesteps=self.concat_timesteps)

        oversample = True

        print("True Reward: %f" % np.mean([np.sum(p['true_rewards']) for p in novice_rollouts]))
        print("Discriminator Reward: %f" % np.mean([np.sum(p['rewards']) for p in novice_rollouts]))

        # if we're using a cost trainer train it?
        if self.cost_trainer and self.iteration % self.train_cost_per_iters == 0:

            novice_rollouts_tensor = [path["observations"] for path in novice_rollouts]
            novice_rollouts_tensor = tensor_utils.pad_tensor_n(novice_rollouts_tensor, expert_horizon)
            random_rollouts_tensor = [path["observations"] for path in random_rollouts]
            random_rollouts_tensor = tensor_utils.pad_tensor_n(random_rollouts_tensor, expert_horizon)

            train_novice_data = novice_rollouts_tensor
            # Replace first 5 novice rollouts by random policy trajectory
            train_novice_data[:5] = random_rollouts_tensor[:5]

            # novice_rollouts_tensor = tensor_utils.pad_tensor_n(novice_rollouts_tensor, expert_horizon)
            novice_rollouts_tensor = np.asarray([tensor_utils.pad_tensor(a, expert_horizon, mode='last') for a in novice_rollouts_tensor])

            # if self.iteration % self.replay_times == 0:
            #     print("Appending replay buffer")
            #     # import pdb; pdb.set_trace()
            #     novice_rollouts_tensor = np.concatenate([novice_rollouts_tensor] + list(self.replay_buffer.values()), axis=0)
            #     # import pdb; pdb.set_trace()
            #     self.replay_buffer[self.replay_index] = novice_rollouts_tensor
            #     self.replay_index += 1
            #     self.replay_index %= self.max_replays

            self.cost_trainer.train_cost(train_novice_data, expert_rollouts_tensor, number_epochs=6, num_frames=self.num_frames)

        # optimize the novice policy by one step
        # TODO: put this in a config provider or something?
        if "policy_opt_steps_per_global_step" in config:
            policy_opt_epochs = config["policy_opt_steps_per_global_step"]
            learning_schedule = False
        else:
            learning_schedule = False
            policy_opt_epochs = 5

        if "policy_opt_learning_schedule" in config:
            learning_schedule = config["policy_opt_learning_schedule"]

        print("Number of policy opt epochs %d" % policy_opt_epochs)

        if learning_schedule:
            # override the policy epochs for larger number of iterations
            policy_opt_epochs *= 2**int(self.iteration/100)
            policy_opt_epochs = min(policy_opt_epochs, 5)
            print("increasing policy opt epochs to %d" % policy_opt_epochs )

        for i in range(policy_opt_epochs):
            # import pdb; pdb.set_trace()
            if i >= 1:
                # Resample so TRPO doesn't just reject all the steps
                novice_rollouts = self.novice_policy_optimizer.obtain_samples(self.iteration)
                novice_rollouts = process_samples_with_reward_extractor(novice_rollouts, self.cost_approximator, self.concat_timesteps, self.num_frames)
                policy_training_samples = self.novice_policy_optimizer.process_samples(itr=self.iteration, paths=novice_rollouts)

            self.novice_policy_optimizer.optimize_policy(itr=self.iteration, samples_data=policy_training_samples)
            self.iteration += 1

        if dump_datapoints:
            self.cost_trainer.dump_datapoints(self.num_frames)

        print("Training Iteration (Full Novice Rollouts): %d" % self.iteration)
        return np.mean([np.sum(p['true_rewards']) for p in novice_rollouts]), np.mean([np.sum(p['rewards']) for p in novice_rollouts])
