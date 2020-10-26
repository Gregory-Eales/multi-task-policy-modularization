import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

class Buffer(object):

	def __init__(self):

		self.actions = []
		self.log_probs = []
		self.k_log_probs = []
		self.states = []
		self.prev_states = []
		self.rewards = []
		self.disc_rewards = []
		self.advantages = None
		self.values = []
		self.firsts = []

		self.mean_reward = []
		self.mean_episode_length = []


		plt.ion()
		
		self.figure = plt.figure()
		self.plot = self.figure.add_subplot(1,1,1)
		self.plot.title.set_text("Reward per Epoch")
		self.plot.set_xlabel("Epoch")
		self.plot.set_ylabel("Reward")
		self.plot.legend(loc="upper left")
		

	def plot_reward(self):
		num_steps = len(self.mean_reward)
		steps = np.linspace(0, len(self.firsts)*num_steps, num=num_steps)
		plt.clf()
		plt.title("Mean Reward")
		plt.xlabel("Steps")
		plt.ylabel("Reward")
		plt.plot(steps, self.mean_reward, label="reward")
		#plt.plot(steps, self.mean_episode_length, label="mean_episode_length")
		plt.legend()
		plt.pause(0.01)

	def store(self, state, reward, first):
		self.store_state(state)
		self.store_reward(reward)
		self.store_firsts(first)

	def store_act(self, actions, log_probs):
		self.store_actions(actions)
		self.store_k_log_probs(log_probs)

	def store_prev_states(self, prev_state):
		self.prev_states.append(torch.Tensor(prev_state).type(torch.int8))

	def store_state(self, state):
		self.states.append(torch.Tensor(state).type(torch.int8))

	def store_actions(self, actions):
		self.actions.append(actions)

	def store_values(self, values):
		self.values.append(values)

	def store_reward(self, reward):
		self.rewards.append(torch.Tensor(reward))

	def store_firsts(self, first):
		self.firsts.append(torch.Tensor(first).type(torch.int8))
		
	def store_k_log_probs(self, k_log_prob):
		self.k_log_probs.append(k_log_prob)

	def store_advantage(self, advantages):
		self.advantages = advantages

	def clear(self):

		sum_reward = torch.stack(self.rewards).sum()

		firsts = torch.stack(self.firsts, dim=1).reshape(-1, 1).float()
		sum_firsts = torch.sum(firsts)

		self.mean_episode_length.append(len(self.firsts)/(sum_firsts))
		self.mean_reward.append(sum_reward/sum_firsts)

		self.plot_reward()

		self.actions = []
		self.log_probs = []
		self.k_log_probs = []
		self.states = []
		self.prev_states = []
		self.rewards = []
		self.disc_rewards = []
		self.advantages = None
		self.values = []
		self.firsts = []

	def get(self):
	
		if len(self.states[0].shape) > 3:
			states = torch.stack(self.states, dim=1).reshape(-1, 3, 64, 64)
			del self.states
		
		else:
			states = torch.cat(self.states, dim=1).reshape(-1, 8)
			del self.states
		
		actions = torch.stack(self.actions, dim=1).reshape(-1, 1)

		del self.actions

		k_log_probs = torch.stack(self.k_log_probs, dim=1).reshape(-1, 1)

		del self.k_log_probs
	
		"""
		states = torch.stack(self.states).reshape(-1, 8)
		actions = torch.stack(self.actions).reshape(-1, 1)
		k_log_probs = torch.stack(self.k_log_probs).reshape(-1, 1)
		"""

		disc_rewards = self.disc_rewards

		del self.disc_rewards

		return states, actions, k_log_probs, disc_rewards
