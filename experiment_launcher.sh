#!/bin/bash

# This is a script to launch our experiment runs

# Cartpole

# 4 frames, importance 0

cmd="python -u run_training_gans.py experts/expert_rollouts_CartPole-v0.pickle novices/novice_policy_CartPole_gan_mix_f4.o --num_frames 4 --num_experiments 5 --importance_weights 0.0 --algorithm mixgan --iterations 40 --env CartPole-v0 > mix_log 2> mix_error_log"
disown -h $cmd &

cmd="python -u run_training_gans.py experts/expert_rollouts_CartPole-v0.pickle novices/novice_policy_CartPole_gan_opt_f4.o --num_frames 4 --num_experiments 5 --importance_weights 0.0 --algorithm optiongan --iterations 40 --env CartPole-v0 > opt_log 2> opt_error_log"
disown -h $cmd &

cmd="python -u run_training_gans.py experts/expert_rollouts_CartPole-v0.pickle novices/novice_policy_CartPole_gan_rlgan_f4.o --num_frames 4 --num_experiments 5 --importance_weights 0.0 --algorithm rlgan --iterations 40 --env CartPole-v0 > gan_log 2> gan_error_log"

$cmd

# 4 frames importance .1

cmd="python -u run_training_gans.py experts/expert_rollouts_CartPole-v0.pickle novices/novice_policy_CartPole_gan_mix_f4.o --num_frames 4 --num_experiments 5 --importance_weights 0.1 --algorithm mixgan --iterations 40 --env CartPole-v0 > mix_log 2> mix_error_log"
disown -h $cmd &

cmd="python -u run_training_gans.py experts/expert_rollouts_CartPole-v0.pickle novices/novice_policy_CartPole_gan_opt_f4.o --num_frames 4 --num_experiments 5 --importance_weights 0.1 --algorithm optiongan --iterations 40 --env CartPole-v0 > opt_log 2> opt_error_log"
$cmd

# 4 frames importance .5
cmd="python -u run_training_gans.py experts/expert_rollouts_CartPole-v0.pickle novices/novice_policy_CartPole_gan_opt_f4.o --num_frames 4 --num_experiments 5 --importance_weights 0.5 --algorithm optiongan --iterations 40 --env CartPole-v0 > opt_log 2> opt_error_log"
disown -h $cmd &

cmd="python -u run_training_gans.py experts/expert_rollouts_CartPole-v0.pickle novices/novice_policy_CartPole_gan_mix_f4.o --num_frames 4 --num_experiments 5 --importance_weights 0.5 --algorithm mixgan --iterations 40 --env CartPole-v0 > mix_log 2> mix_error_log"
$cmd

# 4 frames importance 1.0
cmd="python -u run_training_gans.py experts/expert_rollouts_CartPole-v0.pickle novices/novice_policy_CartPole_gan_opt_f4.o --num_frames 4 --num_experiments 5 --importance_weights 1.0 --algorithm optiongan --iterations 40 --env CartPole-v0 > opt_log 2> opt_error_log"
disown -h $cmd &

cmd="python -u run_training_gans.py experts/expert_rollouts_CartPole-v0.pickle novices/novice_policy_CartPole_gan_mix_f4.o --num_frames 4 --num_experiments 5 --importance_weights 1.0 --algorithm mixgan --iterations 40 --env CartPole-v0 > mix_log 2> mix_error_log"
$cmd
