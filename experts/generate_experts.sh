# To generate experts, we use the original TRPO code https://github.com/joschu/modular_rl

# TODO: check if we have this repository
git clone https://github.com/joschu/modular_rl.git

# TODO: install dependencies if needed

# To generate experts
KERAS_BACKEND=theano python run_pg.py --gamma=0.995 --lam=0.97 --agent=modular_rl.agentzoo.TrpoAgent --use_hdf=1 --max_kl=0.01 --cg_damping=0.1 --activation=tanh --n_iter=250 --seed=0 --timesteps_per_batch=5000 --env=CartPole-v0 --outfile=./CartPole-v0.h5

# To get expert rollouts
