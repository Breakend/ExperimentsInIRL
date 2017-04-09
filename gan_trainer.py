from standard_discriminators import MLPDiscriminator
import numpy as np

class GANCostTrainer(object):

    def __init__(self, input_dims):
        # input_dims is the size of the feature vectors
        self.disc = MLPDiscriminator(input_dims)

    def get_reward(self, samples):
        return self.disc.eval(samples)

    def train_cost(self, data, epochs, classes, batch_size=20, horizon=200):
        for iter_step in range(0, n_epochs):
            batch_losses = []
            lab_acc = []
            for batch_step in range(0, data_one.shape[0], batch_size):

                data_batch_zero = data_one[batch_step: batch_step+batch_size]
                data_batch_one = data_two[batch_step: batch_step+batch_size]
                data_batch = [data_batch_zero, data_batch_one]

                classes_batch = classes[batch_step: batch_step+batch_size]
                domains_batch = domains[batch_step: batch_step+batch_size]
                targets = dict(classes=classes_batch, domains=domains_batch)

                batch_losses.append(self.disc.train(data_batch, targets))
                lab_acc.append(self.disc.get_lab_accuracy(data_batch, targets['classes']))
            print('loss is ' + str(np.mean(np.array(batch_losses))))
            print('acc is ' + str(np.mean(np.array(lab_acc))))

    def shuffle_to_training_data(self, expert_data, on_policy_data, horizon=200):
        """
        Takes in expert_data as, on_policy_data, expert_fail_data as a stacked tensor of just the observation data
        #TODO: removed faildata? is this necessary?
        """
        import pdb; pdb.set_trace()
        n_trajs = len(expert_data)
        # TODO: for images should be nxmxc
        feature_space = len(expert_date[0][0]) # number of features in the observation data


        data = np.vstack([expert_data, on_policy_data])
        e_10 = np.zeros((2,))
        e_10[0] = 1
        e_01 = np.zeros((2,))
        e_01[0] = 1
        expert_classes = np.tile(e_10, (n_trajs, horizon, 1))
        novice_classes = np.tile(e_01, (n_trajs, horizon, 1))
        classes = np.vstack([expert_classes, novice_classes])
        # domains = np.vstack([expert_data['domains'], on_policy_data['domains'], expert_fail_data['domains']])

        sample_range = data.shape[0]*data.shape[1]
        all_idxs = np.random.permutation(sample_range)

        t_steps = data.shape[1]

        data_matrix = np.zeros(shape=(sample_range, feature_space))
        data_matrix_two = np.zeros(shape=(sample_range, feature_space))
        class_matrix = np.zeros(shape=(sample_range, 2))
        dom_matrix = np.zeros(shape=(sample_range, 2))
        for one_idx, iter_step in zip(all_idxs, range(0, sample_range)):
            traj_key = np.floor(one_idx/t_steps)
            time_key = one_idx % t_steps
            time_key_plus_one = min(time_key + 3, t_steps-1)
            data_matrix[iter_step, :, :, :] = data[traj_key, time_key, :]
            data_matrix_two[iter_step, :, :, :] = data[traj_key, time_key_plus_one, :]
            class_matrix[iter_step, :] = classes[traj_key, time_key, :]
            # dom_matrix[iter_step, :] = domains[traj_key, time_key, :]
        return data_matrix, data_matrix_two, class_matrix
