from sampling_utils import *
from rllab.sampler.base import BaseSampler
from rllab.misc import tensor_utils
from rllab.policies.uniform_control_policy import UniformControlPolicy
from rllab.algos.nop import NOP
from rllab.baselines.zero_baseline import ZeroBaseline
import numpy as np
from scipy.stats import norm
import tensorflow as tf
import gc
import time
from gan.nn_utils import initialize_uninitialized

class Trainer(object):

    def __init__(self, sess, env, cost_approximator, cost_trainer, novice_policy, novice_policy_optimizer, num_frames=4, concat_timesteps=True, train_disc=True):
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
        self.should_train_cost = True
        self.prev_reward_dist = None
        self.is_first_disc_update = True
        self.gc_time = time.time()
        self.gc_time_threshold = 60 # seconds between garbage collection
        # as in traditional GANs, we add failure noise
        self.noise_fail_policy = UniformControlPolicy(env.spec)
        self.train_disc = train_disc
        self.zero_baseline = ZeroBaseline(env_spec=env.spec)
        self.rand_algo = NOP(
            env=env,
            policy=self.noise_fail_policy,
            baseline=self.zero_baseline,
            batch_size=1*self.env.horizon,
            max_path_length=self.env.horizon,
            n_itr=1,
            discount=0.995,
            step_size=0.01,
        )
        self.rand_algo.start_worker() # TODO: Call this in constructor instead ?
        self.rand_algo.init_opt()
        self.should_do_policy_step = True
        self.should_do_exploration = True
        self.num_steps_since_last_trpo = 0

    def __replace_rollout_in_cache(self, cache, rollout):
        for i, x in enumerate(cache):
            if np.sum(x['rewards']) < np.sum(rollout['rewards']):
                cache[i] = rollout
                return

    def train_with_accumulation(self, algo, expert_rollout_pickle_path=None, sess=None, reward_threshold=None, num_rollouts_to_store=5, n_itr=50, start_worker=False):
        created_session = True if (sess is None) else False
        if sess is None:
            sess = tf.Session()
            sess.__enter__()

        rollout_cache = []
        initialize_uninitialized(sess)
        if start_worker:
            algo.start_worker()
        start_time = time.time()
        for itr in range(n_itr):
            itr_start_time = time.time()
            itr = self.iteration
            self.iteration += 1

            with logger.prefix('itr #%d | ' % self.iteration):
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
                    if reward_threshold is None or reward_sum > reward_threshold:
                        if len(rollout_cache) < num_rollouts_to_store:
                            rollout_cache.append(x)
                        else:
                            self.__replace_rollout_in_cache(rollout_cache, x)

                logger.record_tabular("AveRewardInCache", np.mean([np.sum(x['rewards']) for x in rollout_cache]))
                logger.dump_tabular(with_prefix=False)

                if expert_rollout_pickle_path:
                    with open(expert_rollout_pickle_path, "wb") as output_file:
                        pickle.dump(rollout_cache, output_file)


        if start_worker:
            algo.shutdown_worker()
        if created_session:
            sess.close()
        return rollout_cache,  np.mean([np.sum(x['rewards']) for x in rollout_cache])

    def run_exploration_rollouts(self, es, env, num_rollouts_to_store):
        rollout_cache = []
        iters = 0
        max_iters = 10000
        while iters < max_iters and (len(rollout_cache) <= num_rollouts_to_store or np.mean([np.sum(x['rewards']) for x in rollout_cache]) <= np.max([np.sum(x['rewards']) for x in rollout_cache])):
            rollout = rollout_policy(es, env, env.horizon, reward_extractor=None, speedup=1, get_image_observations=False, num_frames=1, concat_timesteps=True)
            reward_sum = np.sum(rollout['rewards'])
            if len(rollout_cache) < num_rollouts_to_store:
                rollout_cache.append(rollout)
            else:
                self.__replace_rollout_in_cache(rollout_cache, rollout)
            iters += 1
        print("Exploration average reward: %f" % np.mean([np.sum(x['rewards']) for x in rollout_cache]))
        return rollout_cache

    def step(self, expert_horizon=200, dump_datapoints=False, number_of_sample_trajectories=None, config={}):
        # if number_of_sample_trajectories is None:
        #     number_of_sample_trajectories = len(expert_rollouts_tensor)

        # 50 steps of TRPO
        # if self.should_do_exploration:
        #     # from rllab.exploration_strategies.gaussian_strategy import GaussianStrategy
        #     # es = GaussianStrategy(env_spec=self.env.spec)
        #     from exploration_strategies.epsilon_greedy import GaussianEpsilonGreedyStrategy
        #     self.run_exploration_rollouts(GaussianEpsilonGreedyStrategy(self.env.spec, self.novice_policy), self.env, 10)


        if self.should_do_policy_step:
            self.expert_rollouts, self.cache_reward = self.train_with_accumulation(self.novice_policy_optimizer,
                                    expert_rollout_pickle_path=None,
                                    sess=self.sess,
                                    num_rollouts_to_store=5,
                                    # n_itr=300 if self.should_do_exploration else 10)
                                    n_itr=20)
            self.should_do_policy_step = False

            # Extract observations to a tensor
            self.expert_rollouts_tensor = tensor_utils.stack_tensor_list([path["observations"] for path in self.expert_rollouts])
            # import pdb; pdb.set_trace()

            if "oversample" in config and config["oversample"]:
                oversample_rate = max(int(number_of_sample_trajectories / len(self.expert_rollouts_tensor)), 1.)
                self.expert_rollouts_tensor = self.expert_rollouts_tensor.repeat(oversample_rate, axis=0)
                print("oversampling %d times to %d" % (oversample_rate, len(self.expert_rollouts_tensor)))

            with tf.variable_scope("trainer%d" % self.iteration):
                self.cost_trainer = type(self.cost_trainer)(self.cost_trainer.input_dims, config=config)
                self.cost_approximator = self.cost_trainer
                self.is_first_disc_update = True

        # This does things like calculate advantages and entropy, etc.
        # if we use the cost function when acquiring the novice rollouts, this will use our cost function
        # for optimizing the trajectories
        orig_novice_rollouts = self.novice_policy_optimizer.obtain_samples(self.iteration)

        ave_true_reward = np.mean([np.sum(p['rewards']) for p in orig_novice_rollouts])

        print("*******************************************")
        print("True Reward: %f" % ave_true_reward)
        print("*******************************************")
        if ave_true_reward >= self.cache_reward*.85 or self.num_steps_since_last_trpo > 25:
            # if our average reward is in some threshold of the max TRPO reward, let TRPO go again
            self.should_do_policy_step = True
        else:
            self.num_steps_since_last_trpo += 1

        novice_rollouts = process_samples_with_reward_extractor(orig_novice_rollouts, self.cost_approximator, self.concat_timesteps, self.num_frames,  batch_size=config["policy_opt_batch_size"])

        mu, std = norm.fit(np.concatenate([np.array(p['rewards']).reshape(-1) for p in novice_rollouts]))
        dist = tf.contrib.distributions.Normal(loc=mu, scale=std)

        # if we're using a cost trainer train it?
        if self.cost_trainer and self.should_train_cost and self.train_disc:
            random_rollouts = self.rand_algo.sampler.obtain_samples(itr = self.iteration)

            train_novice_data = novice_rollouts_tensor = [path["observations"] for path in novice_rollouts]
            random_rollouts_tensor = [path["observations"] for path in random_rollouts]
            prev_cost_dist = dist

            # TODO: replace random noise with experience replay
            # Replace first 5 novice rollouts by random policy trajectory
            if config["algorithm"] != "apprenticeship":# ew hack
                # append the two lists
                assert type(novice_rollouts_tensor) is list
                assert type(random_rollouts_tensor) is list
                train_novice_data = novice_rollouts_tensor + random_rollouts_tensor

            train = True
            cost_t_steps = 0
            while train:
                if self.is_first_disc_update:
                    num_epochs = 10
                    self.is_first_disc_update = False
                else:
                    num_epochs = 2
                ave_loss, ave_acc = self.cost_trainer.train_cost(train_novice_data, self.expert_rollouts_tensor, number_epochs=num_epochs, num_frames=self.num_frames)
                # import pdb; pdb.set_trace()
                novice_rollouts = process_samples_with_reward_extractor(orig_novice_rollouts, self.cost_approximator, self.concat_timesteps, self.num_frames,  batch_size=config["policy_opt_batch_size"])
                mu, std = norm.fit(np.concatenate([np.array(p['rewards']).reshape(-1) for p in novice_rollouts]))
                dist = tf.contrib.distributions.Normal(loc=mu, scale=std)
                kl_divergence = tf.contrib.distributions.kl(dist, prev_cost_dist)
                kl = self.sess.run(kl_divergence)
                print("Cost training reward KL divergence: %f" % kl)
                cost_t_steps += 1
                if kl >= .01 or cost_t_steps >= 6:
                    train = False
                # if kl >= .1:
                # Probably need to lower the learning rate so we don't diverge from the distribution so much?


            self.should_train_cost = False
            self.prev_reward_dist = dist


        policy_training_samples = self.novice_policy_optimizer.process_samples(itr=self.iteration, paths=novice_rollouts)

        # import pdb; pdb.set_trace()
        # print("Number of policy opt epochs %d" % policy_opt_epochs)


        # TODO: make this not a tf function, seems like too much overhead
        if self.prev_reward_dist:
            # import pdb; pdb.set_trace()
            kl_divergence = tf.contrib.distributions.kl(dist, self.prev_reward_dist)
            kl = self.sess.run(kl_divergence)

            print("Reward distribution KL divergence since last cost update %f"% kl)
            kl_with_decay = .1 * (.96 ** self.iteration/20)
            if kl >= kl_with_decay:
                self.should_train_cost = True

        total_len = len(policy_training_samples['observations'])
        for policy_training_sample_batch in batchify_dict(policy_training_samples, batch_size=config["policy_opt_batch_size"], total_len=total_len): #TODO: configurable batch_size
            self.novice_policy_optimizer.optimize_policy(itr=self.iteration, samples_data=policy_training_sample_batch)
        self.iteration += 1

        if dump_datapoints:
            self.cost_trainer.dump_datapoints(self.num_frames)

        if time.time() - self.gc_time > self.gc_time_threshold:
            gc.collect()
            self.gc_time = time.time()
        print("*******************************************")
        print("Discriminator Reward: %f" % np.mean([np.sum(p['rewards']) for p in novice_rollouts]))
        print("Training Iteration (Full Novice Rollouts): %d" % self.iteration)
        print("*******************************************")

        return np.mean([np.sum(p['true_rewards']) for p in novice_rollouts]), np.mean([np.sum(p['rewards']) for p in novice_rollouts])
