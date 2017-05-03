import numpy as np
import tensorflow as tf
import sklearn #utils.shuffle


def shuffle_to_training_data(expert_data, on_policy_data, num_frames=4, horizon=200): #TODO: make sure to pass horizon in
    """
    Takes in expert_data as, on_policy_data, expert_fail_data as a stacked tensor of just the observation data
    #TODO: removed faildata? is this necessary?
    """
    # import pdb; pdb.set_trace()
    n_trajs_exp = len(expert_data)
    n_trajs_nov = len(on_policy_data)

    # TODO: for images should be nxmxc
    if len(expert_data[0][0].shape) > 1:
        # assume image
        feature_space = np.product(expert_data[0][0].shape)
    else:
        feature_space = len(expert_data[0][0]) # number of features in the observation data

    expert_class = np.ones((1,))
    novice_class = np.zeros((1,))

    all_datas = []
    all_classes = []

    for rollout in expert_data:
        for time_key in range(len(rollout)):
            data_matrix = np.zeros(shape=(1, num_frames, feature_space))
            # we want the first thing in the sequence to be repeated until we have enough to form a sequence
            # TODO: replicate this in the extract rewards function
            for t in range(0, num_frames):
                time_key_plus_one = max(time_key - t, 0)
                data_matrix[0, num_frames-t-1, :] = np.reshape(rollout[time_key_plus_one, :], (1, -1))
            all_datas.append(data_matrix)
            all_classes.append(expert_class)

    # TODO: remove the copypasta
    for rollout in on_policy_data:
        for time_key in range(len(rollout)):
            data_matrix = np.zeros(shape=(1, num_frames, feature_space))
            # we want the first thing in the sequence to be repeated until we have enough to form a sequence
            # TODO: replicate this in the extract rewards function
            for t in range(0, num_frames):
                time_key_plus_one = max(time_key - t, 0)
                data_matrix[0, num_frames-t-1, :] = np.reshape(rollout[time_key_plus_one, :], (1, -1))
            all_datas.append(data_matrix)
            all_classes.append(novice_class)

    # import pdb; pdb.set_trace()

    data_matrix = np.vstack(all_datas)
    class_matrix = np.vstack(all_classes)

    data_matrix, class_matrix = sklearn.utils.shuffle(data_matrix, class_matrix)

    return data_matrix, class_matrix

def shuffle_to_training_data_single(data, novice=True, num_frames=4):
    """
    Takes in expert_data as, on_policy_data, expert_fail_data as a stacked tensor of just the observation data
    #TODO: removed faildata? is this necessary?
    """
    # import pdb; pdb.set_trace()
    n_trajs = len(data)
    # n_trajs_nov = len(on_policy_data)

    # TODO: for images should be nxmxc
    feature_space = len(data[0][0]) # number of features in the observation data
    # import pdb; pdb.set_trace()
    horizon = data.shape[1] # length of trajectory


    # data = np.vstack([expert_data, on_policy_data])
    onee = np.ones((1,))
    # e_10[0] = 1
    zeroo = np.zeros((1,))
    # e_01[1] = 1
    if novice:
        classes = np.tile(zeroo, (n_trajs, horizon, 1))
    else:
        classes = np.tile(onee, (n_trajs, horizon, 1))
    # import pdb; pdb.set_trace()
    # classes = np.vstack([expert_classes, novice_classes])
    # domains = np.vstack([expert_data['domains'], on_policy_data['domains'], expert_fail_data['domains']])

    sample_range = data.shape[0]*data.shape[1]
    all_idxs = np.random.permutation(sample_range)

    t_steps = data.shape[1]
    # import pdb; pdb.set_trace()

    data_matrix = np.zeros(shape=(sample_range, num_frames, feature_space))
    # data_matrix_two = np.zeros(shape=(sample_range, num_frames, feature_space))
    class_matrix = np.zeros(shape=(sample_range, 1))
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
