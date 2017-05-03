ENV=Seaquest-v0; python run_a3c.py --load ./a3c_models/"$ENV".tfmodel --env "$ENV" --episode 10 --output output_dir


I had some trouble installing tensorpack. To get this working needed to do this outside of the rllab3 environment and just use python2.
