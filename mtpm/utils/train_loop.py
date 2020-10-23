import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import gym
from tqdm import tqdm

def train(
	agent,
	env,        
	max_episodes = 100,     
	update_episodes = 100,    
	):
	#############################################

	# logging variables
	rewards = []
	reward_per_epoch = []
	running_reward = 0
	avg_length = 0
	
	# training loop
	for i_episode in tqdm(range(1, max_episodes+1)):
		state = env.reset()
		running = True
		count = 0
		while running:

			count += 1
			
			action = agent.act(state)
			state, reward, done, _ = env.step(action)
			agent.store(state, reward, done)
			running_reward += reward
			
			if done or count > 200:
				running = False
				reward_per_epoch.append(running_reward)
				running_reward = 0

		if i_episode % update_episodes == 0 and i_episode != 0:
			agent.update()
			rewards.append(np.sum(reward_per_epoch)/update_episodes)
			reward_per_epoch = []
			

	return rewards