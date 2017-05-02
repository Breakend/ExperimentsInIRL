ENV=Seaquest-v0; python run_a3c.py --load ./a3c_models/"$ENV".tfmodel --env "$ENV" --episode 10 --output output_dir
