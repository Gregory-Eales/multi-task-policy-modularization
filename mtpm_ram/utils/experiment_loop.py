import gym
from matplotlib import pyplot as plt


from .dir import *
from .seed import *
from .train_loop import *
from .graph import *
from .data import *

def run_experiment(
	agent_class,
	experiment_name,
	env_names,
	log,
	graph,
	random_seeds,
	n_episodes,
	n_steps,
	n_envs,
	epsilon,
	batch_sz,
	lr,
	gamma,
	critic_epochs,
	n_latent_var = 64,
	betas = (0.9, 0.999),            
	k_epochs = 4,              
):

	# 1. create experiment directory
	# 2. run experiments
	# 3. save csv file with data for each run
	# 4. save graphs for experiments
	# 5. 
	
	# create the path for the experiment
	exp_path = create_exp_dir(experiment_name)

	rewards = []
	# run a training loop for each seed

	for env_name in env_names:
		for seed in random_seeds:
	
			env = gym.make(env_name)
			state_dim = env.observation_space.shape[0]
			action_dim = env.action_space.n

			print(state_dim)
			print(action_dim)

			set_seed(env, seed)

			agent = agent_class(
				state_dim,
				action_dim,
				n_latent_var,
				lr,
				betas,
				gamma,
				k_epochs,
				epsilon
				)

			rewards.append(train(agent, env))

	plot_rewards(rewards, path=exp_path)
			
		