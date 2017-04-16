import numpy as np

def shuffle_to_training_data(expert_data, on_policy_data, num_frames=4, horizon=200):
    """
    Takes in expert_data as, on_policy_data, expert_fail_data as a stacked tensor of just the observation data
    #TODO: removed faildata? is this necessary?
    """
    # import pdb; pdb.set_trace()
    n_trajs = len(expert_data)
    # TODO: for images should be nxmxc
    feature_space = len(expert_data[0][0]) # number of features in the observation data


    data = np.vstack([expert_data, on_policy_data])
    e_10 = np.zeros((2,))
    e_10[0] = 1
    e_01 = np.zeros((2,))
    e_01[1] = 1
    expert_classes = np.tile(e_10, (n_trajs, horizon, 1))
    novice_classes = np.tile(e_01, (n_trajs, horizon, 1))
    # import pdb; pdb.set_trace()
    classes = np.vstack([expert_classes, novice_classes])
    # domains = np.vstack([expert_data['domains'], on_policy_data['domains'], expert_fail_data['domains']])

    sample_range = data.shape[0]*data.shape[1]
    all_idxs = np.random.permutation(sample_range)

    t_steps = data.shape[1]
    # import pdb; pdb.set_trace()

    data_matrix = np.zeros(shape=(sample_range, num_frames, feature_space))
    # data_matrix_two = np.zeros(shape=(sample_range, num_frames, feature_space))
    class_matrix = np.zeros(shape=(sample_range, 2))
    # dom_matrix = np.zeros(shape=(sample_range, 2))
    # generate random samples of size num_frames
    for one_idx, iter_step in zip(all_idxs, range(0, sample_range)):
        traj_key = int(np.floor(one_idx/t_steps))
        time_key = time_key_plus_one = one_idx % t_steps
        for t in range(0, num_frames):
            time_key_plus_one = min(time_key + t, t_steps-1)
            data_matrix[iter_step, t, :] = data[traj_key, time_key_plus_one, :]
        # take the class of the last frame
        class_matrix[iter_step, :] = classes[traj_key, time_key_plus_one, :]
    return data_matrix, class_matrix
