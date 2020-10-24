import gym
from matplotlib import pyplot as plt
import gym3

from .dir import *
from .seed import *
from .train_loop import *
from .graph import *
from .data import *

def run_experiment(Agent, hparams):

	
	# create the path for the experiment
	exp_path = create_exp_dir(hparams.experiment_name)


	for env_name in hparams.env_names:
		rewards = []
		for seed in hparams.random_seeds:
	
			env = gym3.vectorize_gym(
				num=hparams.n_envs,
				env_kwargs={"id": "CartPole-v0"},
				seed=seed
				)

			set_seed(seed)

			agent = Agent(hparams)

			r = train(
				agent=agent,
				env=env,          
				n_steps=hparams.n_steps,        
				update_step=hparams.update_step,    
				)

			rewards.append(r)

		plot_rewards(
			rewards,
			hparams.update_step,
			path=exp_path,
			env_name=env_name
			)

		save_results(
			env_name=env_name,
			rewards=rewards,
			path=exp_path,
			update_steps=hparams.update_step,
			seeds=hparams.random_seeds
			)


		save_arguments(path=exp_path, args=hparams)

def main():
	pass

if __name__ == "__main__":
	main()
		