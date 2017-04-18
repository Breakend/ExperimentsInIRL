#!/usr/bin/env python

import pickle
import tensorflow as tf
import numpy as np
import gym

learning_rate = 0.01
training_epochs = 20000
batch_size = 100
display_step = 1000

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with tanh activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.tanh(layer_1)
    # # Hidden layer with tanh activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.tanh(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_rollouts_path', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading expert data')
    expert_rollouts_file = open(args.expert_rollouts_path, "rb")
    expert_data = pickle.load(expert_rollouts_file)
    print('loaded expert data')
    # import pdb; pdb.set_trace()

    train_X = np.asarray([path['observations'] for path in expert_data[:args.num_rollouts]])
    train_Y = np.asarray([path['actions'] for path in expert_data[:args.num_rollouts]])

    train_X = np.reshape(train_X, (train_X.shape[0]*train_X.shape[1], train_X.shape[2]))
    train_Y = np.reshape(train_Y, (train_Y.shape[0]*train_Y.shape[1], train_Y.shape[2]))

    n_hidden_1 = 100 # 1st layer number of features
    n_hidden_2 = 100 # 2nd layer number of features

    n_inputs = train_X.shape[1] # Observation vector input
    n_outputs = train_Y.shape[1] # Actions vector

    # Build the model
    X = tf.placeholder(tf.float32, shape=[None, n_inputs])
    Y = tf.placeholder(tf.float32, shape=[None, n_outputs])

    weights = {
        'h1': tf.Variable(tf.random_normal([n_inputs, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_outputs]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_outputs]))
    }

    n_samples = train_X.shape[0]
    pred = multilayer_perceptron(X, weights, biases)
    loss = tf.reduce_mean(tf.squared_difference(pred, Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # Fit all training data
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(n_samples/batch_size)
            batch_start_idx = (epoch * batch_size) % (n_samples - batch_size)
            batch_end_idx = batch_start_idx + batch_size
            batch_xs = train_X[batch_start_idx:batch_end_idx]
            batch_ys = train_Y[batch_start_idx:batch_end_idx]
            _, c = sess.run([optimizer, loss], feed_dict={X: batch_xs,
                                                          Y: batch_ys})

            # Compute average loss
            avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            print(obs)
            done = False
            totalr = 0.
            steps = 0

            while not done:
                action = sess.run(multilayer_perceptron(
                        tf.cast(np.reshape(obs, (1, len(obs))), tf.float32),
                        sess.run(weights), sess.run(biases)))
                action = np.argmax(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

if __name__ == '__main__':
    main()
