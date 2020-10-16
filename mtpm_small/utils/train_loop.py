import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
from tqdm import tqdm

def train(
	agent,
	env,        
	log_interval = 20,          
	max_episodes = 5000,       
	max_timesteps = 300,        
	update_timestep = 500,    
	):
	#############################################
	
	
	# logging variables
	rewards = []
	running_reward = 0
	avg_length = 0
	timestep = 0
	log_interval = 0
	
	# training loop
	for i_episode in tqdm(range(1, max_episodes+1)):
		state = env.reset()
		running = True
		while running:

			timestep+=1
			
			action = agent.act(state)
			state, reward, done, _ = env.step(action)
			#env.render()
			
			# Saving reward and is_terminal:
			agent.store(state, reward, done)
			running_reward += reward

			
			if done:
				running = False

		if timestep % update_timestep == 0:
				agent.update()
				rewards.append(running_reward/timestep)
				print(rewards)
				timestep = 0
				running_reward = 0
				
		"""
		# stop training if avg_reward > solved_reward
		if running_reward > (log_interval*solved_reward):
			print("########## Solved! ##########")
			torch.save(ppo.policy.state_dict(), './PPO_{}.pt'.format(env_name))
			break
			
		# logging
		if i_episode % log_interval == 0:
			avg_length = int(avg_length/log_interval)
			running_reward = int((running_reward/log_interval))
			
			print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
			running_reward = 0
			avg_length = 0
			rewards.append(running_reward)

		"""
	return rewards