from .standard_discriminators import ConvStateBasedDiscriminator
from .utils import *
import numpy as np

class GANCostTrainer(object):

    def __init__(self, input_dims, config=None):
        # input_dims is the size of the feature vector
        self.config = config
        self.disc = ConvStateBasedDiscriminator(input_dims, config=config)

    def get_reward(self, samples):
        # import pdb; pdb.set_trace()
        r = self.disc.eval(samples)[:, 0]
        # print(np.sum(r))
        # print(r)
        return r

    def dump_datapoints(self, num_frames=4):
        if num_frames != 1:
            print("only support graphing internal things with 1 frame concated for now")
            return
        return

    def train_cost(self, novice_rollouts_tensor, expert_rollouts_tensor, number_epochs=2, num_frames=4):
        data_matrix, class_matrix = shuffle_to_training_data(expert_rollouts_tensor, novice_rollouts_tensor, num_frames=num_frames)
        return self._train_cost(data_matrix, number_epochs, class_matrix)

    def _train_cost(self, data, epochs, classes, batch_size=20, horizon=200, num_frames=4):
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
        return np.mean(np.array(batch_losses)), np.mean(np.array(lab_acc))
