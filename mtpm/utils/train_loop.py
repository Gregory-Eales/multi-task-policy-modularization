import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import gym
from tqdm import tqdm
from procgen import ProcgenGym3Env
import gym3

from .multi_task_wrapper import *


def train_procgen(agent, env_name, n_envs, seed, n_steps, update_step):

	env = ProcgenGym3Env(
				num=n_envs,
				env_name=env_name,
				start_level=seed,
				distribution_mode="easy",
				use_backgrounds="False",
				)

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

def train_multi_task(agent, env_name, n_envs, seed, n_steps, update_step):

	"""
	env = gym3.vectorize_gym(
            num=n_envs,
            env_fn=make_multi_task,
            env_kwargs={
            "env_name":env_name,
            "ob":8,
            "ac":4,
            "r_scale":1,
            }
            )
	"""
	env_names = ["Acrobot-v1", "MountainCar-v0", "CartPole-v0", "LunarLander-v2"]
	env = vectorize_multi_task(env_names, num=4, ob=8, ac=4, seed=seed)
	print(len(env_names[0:1]))

	_, prev_state, prev_first = env.observe()

	for step in tqdm(range(n_steps)):

		action = agent.act(prev_state)
		env.act(action)

		reward, state, first = env.observe()

		agent.store(prev_state, reward, prev_first)	

		prev_state = state
		prev_first = first

		if step % update_step == 0 and step!=0:
			agent.update()

	return agent.get_rewards()



def train(agent, env_name, n_envs, seed, n_steps, update_step):


	env = gym3.vectorize_gym(
		num=n_envs,
		env_kwargs={"id": env_name},
		seed=seed
		)

	_, prev_state, prev_first = env.observe()

	for step in tqdm(range(n_steps)):

		
		action = agent.act(prev_state)
		env.act(action)

		reward, state, first = env.observe()

		agent.store(prev_state, reward, prev_first)	

		prev_state = state
		prev_first = first

		if step % update_step == 0 and step!=0:
			agent.update()

	return agent.get_rewards()
