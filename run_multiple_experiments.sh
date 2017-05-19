#!/bin/bash


die() { echo "$@" 1>&2 ; exit 1; }

# TODO: make this proper usage script

if [  $# -le 2 ]
then
  die "Usage: bash $0 num_experiments num_experiments_in_parallel [--all --other --args --to --run_training_gans.py]"
fi

# check whether user had supplied -h or --help . If yes display usage
if [[ ( $# == "--help") ||  $# == "-h" ]]
then
  die "Usage: bash $0 num_experiments num_experiments_in_parallel [--all --other --args --to --run_training_gans.py]"
fi

num_experiments=$1
parallel_exps=$2

pickle_files=()

trap 'jobs -p | xargs kill' EXIT

for (( c=1; c<=$num_experiments; c++ ))
do
  for (( j=1; j<=$parallel_exps; j++ ))
  do
    echo "Launching experiment $c"
    python run_training_gans.py "${@:3}" --experiment_data_pickle_name "exp_$c.pickle" &> exp_$c.log &
    pickle_files=("${pickle_files[@]}" "exp_$c.pickle")
    c=$((c+1))
  done
  wait
done

python create_graphs_from_pickle.py "${pickle_files[@]}"
