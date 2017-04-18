#!/bin/bash

# This is a script to launch our experiment runs

# Cartpole

# 4 frames, importance 0

cmd="python -u run_training_gans.py experts/expert_rollouts_CartPole-v0.pickle novices/novice_policy_CartPole_gan_mix_f4.o --num_frames 4 --num_experiments 5 --importance_weights 0.0 --algorithm mixgan --iterations 60 --env CartPole-v0 --num_expert_rollouts 10 --num_novice_rollouts 50"
nohup $cmd &> mix_log_1 &

cmd="python -u run_training_gans.py experts/expert_rollouts_CartPole-v0.pickle novices/novice_policy_CartPole_gan_opt_f4.o --num_frames 4 --num_experiments 5 --importance_weights 0.0 --algorithm optiongan --iterations 60 --env CartPole-v0 --num_expert_rollouts 10 --num_novice_rollouts 50"
nohup $cmd &> op_log_1 &

cmd="python -u run_training_gans.py experts/expert_rollouts_CartPole-v0.pickle novices/novice_policy_CartPole_gan_rlgan_f4.o --num_frames 4 --num_experiments 5 --importance_weights 0.0 --algorithm rlgan --iterations 60 --env CartPole-v0 --num_expert_rollouts 10 --num_novice_rollouts 50"
eval $cmd &> ganlog_1
