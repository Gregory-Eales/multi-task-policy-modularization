import gym
from matplotlib import pyplot as plt


from .dir import *
from .seed import *
from .train_loop import *
from .graph import *
from .data import *

def run_experiment(
	experiment_name,
	agent_class,
	train_fn,
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
	n_latent_var,           
	k_epochs,
	max_episodes,               
	update_episodes,
	args,              
):

	# 1. create experiment directory
	# 2. run experiments
	# 3. save csv file with data for each run
	# 4. save graphs for experiments
	# 5. 
	
	# create the path for the experiment
	exp_path = create_exp_dir(experiment_name)

	#rewards = []
	# run a training loop for each seed

	for env_name in env_names:
		rewards = []
		for seed in random_seeds:
	
			env = gym.make(env_name)

			state_dim = env.observation_space.shape[0]
			action_dim = env.action_space.n

			set_seed(env, seed)

			agent = agent_class(
				state_dim,
				action_dim,
				n_latent_var=n_latent_var,
				lr=lr,
				gamma=gamma,
				k_epochs=k_epochs,
				epsilon=epsilon
				)

			r = train(
				agent=agent,
				env=env,          
				max_episodes = max_episodes,        
				update_episodes = update_episodes,    
				)

			rewards.append(r)

		plot_rewards(
			rewards,
			path=exp_path,
			env_name=env_name
			)

		save_results(
			env_name=env_name,
			seeds=random_seeds,
			rewards=rewards,
			update_episodes=update_episodes,
			path=exp_path
			)

		save_arguments(path=exp_path, args=args)

def main():
	pass

if __name__ == "__main__":
	main()
		