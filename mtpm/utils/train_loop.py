import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import gym
from tqdm import tqdm


def train_procgen(agent, env, n_steps, update_step):

	env = ProcgenGym3Env(num=64, env_name="coinrun", start_level=0, num_levels=1)

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


def train(agent, env, n_steps, update_step):

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
