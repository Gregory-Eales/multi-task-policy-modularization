import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import gym
from tqdm import tqdm
from procgen import ProcgenGym3Env


def test_procgen(agent, env, n_steps, update_step):

	rewards = []
	firsts = []

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


def test(agent, env, n_steps, update_step):

	rewards = []
	firsts = []

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