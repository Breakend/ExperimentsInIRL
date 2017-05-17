from .standard_discriminators import ConvStateBasedDiscriminatorWithOptions
import numpy as np
from .discriminators.optioned_mlp_disc import MLPMixingDiscriminator
from .utils import *

class GANCostTrainerWithRewardOptions(object):

    def __init__(self, input_dims, mixtures=False, config=None):
        """
        TODO: this is a hack for now, but right now just treat the mixtures param as a flag and have a super class set it for the mixtures model
        """
        # input_dims is the size of the feature vectors
        self.config = config
        self.input_dims = input_dims
        self.disc = MLPMixingDiscriminator(input_dims, num_options=config["num_options"], mixtures=mixtures, config=config)

    def get_reward(self, samples):
        return self.disc.eval(samples)[:, 0]

    def dump_datapoints(self, num_frames=4):
        if num_frames != 1:
            print("only support graphing internal things with 1 frame concated for now")
            return
        self.disc.output_termination_activations(num_frames)
        return

    def train_cost(self, novice_rollouts_tensor, expert_rollouts_tensor, number_epochs=2, num_frames=4):
        data_matrix, class_matrix = shuffle_to_training_data(expert_rollouts_tensor, novice_rollouts_tensor, num_frames=num_frames)
        return self._train_cost(data_matrix, number_epochs, class_matrix)

    def _train_cost(self, data, epochs, classes, batch_size=20, horizon=200):
        for iter_step in range(0, epochs):
            batch_losses = []
            lab_acc = []
            lab_recall = []
            lab_precision = []
            lab_error = []
            # import pdb; pdb.set_trace()
            for batch_step in range(0, data.shape[0], batch_size):

                data_batch = data[batch_step: batch_step+batch_size]
                # data_batch_one = data_two[batch_step: batch_step+batch_size]

                classes_batch = classes[batch_step: batch_step+batch_size]

                batch_losses.append(self.disc.train(data_batch, classes_batch))
                lab_acc.append(self.disc.get_lab_accuracy(data_batch, classes_batch))

                if self.config["output_enhanced_stats"]:
                    lab_error.append(self.disc.get_mse(data_batch, classes_batch))
                    lab_precision.append(self.disc.get_lab_precision(data_batch, classes_batch))
                    lab_recall.append(self.disc.get_lab_recall(data_batch, classes_batch))
            print('-----------')
            print('loss is ' + str(np.mean(np.array(batch_losses))))
            print('acc is ' + str(np.mean(np.array(lab_acc))))
            if self.config["output_enhanced_stats"]:
                print('precision is ' + str(np.mean(np.array(lab_precision))))
                print('recall is ' + str(np.mean(np.array(lab_recall))))
                print('error is ' + str(np.mean(np.array(lab_error))))
            print('-----------')
        self.disc.actual_train_step += 1
        return np.mean(np.array(batch_losses)), np.mean(np.array(lab_acc))


class GANCostTrainerWithRewardMixtures(GANCostTrainerWithRewardOptions):

    def __init__(self, input_dims, config = None):
        super(GANCostTrainerWithRewardMixtures, self).__init__(input_dims, mixtures=True, config=config)
