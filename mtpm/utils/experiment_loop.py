from matplotlib import pyplot as plt
from procgen import ProcgenGym3Env
import gym3
import gym

from .train_loop import *
from .graph import *
from .data import *
from .seed import *
from .dir import *


def run_experiment(Agent, hparams):

	exp_path = create_exp_dir(hparams.experiment_name)

	if hparams.is_multi_task:
		run_multi_task(Agent, hparams, exp_path)

	else:
		run_single_task(Agent, hparams, exp_path)


def run_multi_task(Agent, hparams, exp_path):

	"""
	runs experiment using multiple environments inside of the
	multi-task wrapper
	"""

	rewards = []

	for seed in hparams.random_seeds:

		set_seed(seed)

		agent = Agent(hparams)
		
		r = train_multi_task(
			agent=agent,
			env_names=hparams.env_names,
			seed=seed,
			n_envs=hparams.n_envs,
			n_steps=hparams.n_steps,
			update_step=hparams.update_step,
			)

		rewards.append(r)

	save(rewards, exp_path, hparams, agent=None)

	save_arguments(path=exp_path, args=hparams)


def run_single_task(Agent, hparams, exp_path):

	"""
	runs a experiment on a single task using gym3 parallelization.
	agent is an agent object and should conform to the way it is used
	inside of the training function
	"""

	for env_name in hparams.env_names:
			
			rewards = []
			for seed in hparams.random_seeds:

				set_seed(seed)

				agent = Agent(hparams)

				if hparams.is_procgen:

					r = train_procgen(
						agent=agent,
						env_name=env_name,
						seed=seed, 
						n_envs=hparams.n_envs,
						n_steps=hparams.n_steps,        
						update_step=hparams.update_step,    
						)

				else:
					r = train(
						agent=agent,
						env_name=env_name,
						seed=seed, 
						n_envs=hparams.n_envs,
						n_steps=hparams.n_steps,        
						update_step=hparams.update_step,    
						)

				rewards.append(r)
			
			save(rewards, exp_path, hparams, env_name, agent=None)

	save_arguments(path=exp_path, args=hparams)


def save(rewards, exp_path, hparams, env_name, agent=None):

	"""
	saves the rewards and parameters of an
	experiment and saves them in the corresponding experiment
	folder. the reward graph should contain a confidence band
	between all the different random seeds
	"""

	plot_rewards(
			rewards,
			hparams.update_step,
			path=exp_path,
			env_name=env_name,
			)

	save_results(
		env_name=env_name,
		rewards=rewards,
		path=exp_path,
		update_steps=hparams.update_step,
		seeds=hparams.random_seeds,
		)

	if agent != None:
		save_model(agent, env_name, exp_path)


	


def main():
	pass


if __name__ == "__main__":
	main()
