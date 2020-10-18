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
	parser.add_argument(
	'--env_names',
	default=["Breakout-ramNoFrameskip-v4"]#,"Atlantis-ramNoFrameskip-v4"]
	)

	# saving options
	parser.add_argument('--log', default=True, type=bool)
	parser.add_argument('--graph', default=True, type=bool)

	# training args
	parser.add_argument('--random_seeds', default=list(range(1)), type=list)
	parser.add_argument('--n_episodes', default=20, type=int)
	parser.add_argument('--n_steps', default=100000, type=int)
	parser.add_argument('--batch_sz', default=64, type=int)
	parser.add_argument('--gamma', default=0.999, type=float)
	parser.add_argument('--critic_epochs', default=20, type=int)
	parser.add_argument('--n_envs', default=1, type=int)

	# model args
	parser.add_argument('--lr', default=0.0005, type=float)
	parser.add_argument('--epsilon', default=0.4, type=float)
	parser.add_argument('--n_latent_var', default=256, type=int)
	parser.add_argument('--k_epochs', default=2, type=int)
	parser.add_argument('--max_episodes', default=4000, type=int)
	parser.add_argument('--update_episodes', default=100, type=int)

	args = parser.parse_args()

	run_experiment(
		agent_class=PPO,
		experiment_name=args.experiment_name,
		env_names=args.env_names,
		log=args.log,
		graph=args.graph,
		random_seeds=args.random_seeds,
		n_episodes=args.n_episodes,
		n_steps=args.n_steps,
		n_envs=args.n_envs,
		epsilon=args.epsilon,
		batch_sz=args.batch_sz,
		lr=args.lr,
		gamma=args.gamma,
		critic_epochs=args.critic_epochs,
		n_latent_var = args.n_latent_var,          
		k_epochs = args.k_epochs,
		max_episodes = args.max_episodes,              
		update_episodes = args.update_episodes,
		args = args 
	)
