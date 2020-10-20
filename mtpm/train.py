from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import os
import random
import gym3
from procgen import ProcgenGym3Env
import time
from matplotlib import pyplot as plt
import copy

from modules import *
from utils import *

"""
########################################
TO DO:

	- make sure ppo solo works
	- how to train routing network? (distance clustering?)

	1. train cluster algorithm on latent space
	2. train agent using clustered feature modularizer

########################################
"""

def train(agent, env, n_steps, update_step):


	_, prev_state, prev_first = env.observe()

	for step in tqdm(range(n_steps)):

		
		action = agent.act(prev_state['rgb'])
		env.act(action)

		reward, state, first = env.observe()

		agent.store(prev_state['rgb'], reward, prev_first)	

		prev_state = state
		prev_first = first

		if step % update_step == 0 and step!=0:
			agent.update()


def run_experiment(
	experiment_name,
	environment_name,
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
	k_epochs,
	update_steps,
):

	exp_path = create_exp_dir(experiment_name)

	agent = PPO(
		actor_lr=actor_lr,
		critic_lr=critic_lr,
		batch_sz=batch_sz,
		gamma=gamma,
		epsilon=epsilon,
		k_epochs=k_epochs,
	)

	env = ProcgenGym3Env(
			num=n_envs,
			env_name="coinrun",
			render_mode="rgb_array",
			center_agent=False,
			num_levels=1,
			start_level=2,
			)

	train(agent, env, n_steps, update_steps)
	#generate_graphs(agent, exp_path)

	plt.plot(agent.buffer.mean_reward)
	plt.show()


if __name__ == '__main__':

	parser = ArgumentParser(add_help=False)

	# experiment and  environment
	parser.add_argument('--experiment_name', default="default", type=str)
	parser.add_argument('--environment_name', default="couinrun")

	# saving options
	parser.add_argument('--log', default=True, type=bool)
	parser.add_argument('--graph', default=True, type=bool)

	# training params
	parser.add_argument('--random_seeds', default=list(range(10)), type=list)
	parser.add_argument('--n_episodes', default=2, type=int)
	parser.add_argument('--n_steps', default=10000, type=int)
	parser.add_argument('--batch_sz', default=64, type=int)
	parser.add_argument('--gamma', default=0.99, type=float)
	parser.add_argument('--k_epochs', default=5, type=int)
	parser.add_argument('--n_envs', default=8, type=int)
	parser.add_argument('--update_steps', default=1000, type=int)

	# model params
	parser.add_argument('--actor_lr', default=5e-4, type=float)
	parser.add_argument('--critic_lr', default=5e-4, type=float)
	parser.add_argument('--epsilon', default=0.2, type=float)

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
		k_epochs=params.k_epochs,
		update_steps=params.update_steps
	)
