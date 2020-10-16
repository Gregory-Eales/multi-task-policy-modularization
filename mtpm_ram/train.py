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

if __name__ == '__main__':

	parser = ArgumentParser(add_help=False)

	# experiment and  environment
	parser.add_argument('--experiment_name', default="default", type=str)
	parser.add_argument('--env_names', default=["LunarLander-v2",])

	# saving options
	parser.add_argument('--log', default=True, type=bool)
	parser.add_argument('--graph', default=True, type=bool)

	# training params
	parser.add_argument('--random_seeds', default=list(range(5)), type=list)
	parser.add_argument('--n_episodes', default=20, type=int)
	parser.add_argument('--n_steps', default=100000, type=int)
	parser.add_argument('--batch_sz', default=64, type=int)
	parser.add_argument('--gamma', default=0.999, type=float)
	parser.add_argument('--critic_epochs', default=20, type=int)
	parser.add_argument('--n_envs', default=1, type=int)

	# model params
	parser.add_argument('--lr', default=0.002, type=float)
	parser.add_argument('--epsilon', default=0.3, type=float)

	params = parser.parse_args()

	run_experiment(
		agent_class=PPO,
		experiment_name=params.experiment_name,
		env_names=params.env_names,
		log=params.log,
		graph=params.graph,
		random_seeds=params.random_seeds,
		n_episodes=params.n_episodes,
		n_steps=params.n_steps,
		n_envs=params.n_envs,
		epsilon=params.epsilon,
		batch_sz=params.batch_sz,
		lr=params.lr,
		gamma=params.gamma,
		critic_epochs=params.critic_epochs,
		n_latent_var = 64,
		betas = (0.9, 0.999),           
		k_epochs = 4,
	)
