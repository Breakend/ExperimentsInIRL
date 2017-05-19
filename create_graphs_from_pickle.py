import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import pickle
import argparse
import numpy as np

parser = argparse.ArgumentParser()
#TODO: add multiple so we plot multiple lines
parser.add_argument("datapath", nargs='+')

args = parser.parse_args()

data = {}

rewards_for_experiments = []

for path in args.datapath:
    with open(path, "rb") as output_file:
        data = pickle.load(output_file)

    avg_true_rewards = data["avg"]
    true_rewards_std = np.sqrt(data["var"])
    rewards_for_experiments.append(avg_true_rewards)

avg_true_rewards = np.mean(rewards_for_experiments, axis=0)
true_rewards_std = np.std(rewards_for_experiments, axis=0)

fig = plt.figure()
plt.plot(avg_true_rewards)
plt.xlabel('Training iterations', fontsize=18)
plt.fill_between(np.arange(len(avg_true_rewards)), avg_true_rewards-true_rewards_std, avg_true_rewards+true_rewards_std,
    alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
    linewidth=4, linestyle='dashdot', antialiased=True)

plt.ylabel('Average True Reward', fontsize=16)
# plt.legend()
fig.suptitle('True Reward over Training Iterations')
fig.savefig('all_experiments_graph.png')
plt.clf()
