from matplotlib import pyplot as plt
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import random
import torch
import time
import gym
import os

from modules import *
from utils import *


"""
########################################
TO DO:
	- create encoder-decoder pair
	- policy head
	- action-value head
	- make sure ppo solo works
	- implement parallelization?
	- how to train routing network? (distance clustering?)

########################################
"""
		
def run_experiment(
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
	critic_lr,
	actor_lr,
	gamma,
	critic_epochs,
):
	
	# create the path for the experiment
	exp_path = create_exp_dir(experiment_name)


	# run a training loop for each seed

	for env_name in env_names:
		for seed in random_seeds:
	
			env = gym.make(env_name)

			set_seed(env, seed)

			agent = PPO(
				state_dim,
				action_dim,
				n_latent_var,
				lr, betas,
				gamma,
				K_epochs,
				eps_clip
				)

			train(agent, env)

	

if __name__ == '__main__':

	parser = ArgumentParser(add_help=False)

	# experiment and  environment
	parser.add_argument('--experiment_name', default="default", type=str)
	parser.add_argument('--env_names', default=["LunarLander-v2",])

	# saving options
	parser.add_argument('--log', default=True, type=bool)
	parser.add_argument('--graph', default=True, type=bool)

	# training params
	parser.add_argument('--random_seeds', default=list(range(10)), type=list)
	parser.add_argument('--n_episodes', default=20, type=int)
	parser.add_argument('--n_steps', default=100000, type=int)
	parser.add_argument('--batch_sz', default=64, type=int)
	parser.add_argument('--gamma', default=0.999, type=float)
	parser.add_argument('--critic_epochs', default=20, type=int)
	parser.add_argument('--n_envs', default=1, type=int)

	# model params
	parser.add_argument('--actor_lr', default=2e-1, type=float)
	parser.add_argument('--critic_lr', default=2e-1, type=float)
	parser.add_argument('--epsilon', default=0.3, type=float)

	params = parser.parse_args()

	run_experiment(
		experiment_name=params.experiment_name,
		environment_name=params.environment_name,
		log=params.log,
		graph=params.graph,
		random_seeds=params.random_seeds,
		n_episodes=params.n_episodes,
		n_steps=params.n_steps,
		n_envs=params.n_envs,
		epsilon=params.epsilon,
		batch_sz=params.batch_sz,
		critic_lr=params.critic_lr,
		actor_lr=params.actor_lr,
		gamma=params.gamma,
		critic_epochs=params.critic_epochs,
	)
