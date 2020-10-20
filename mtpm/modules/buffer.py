import torch
import numpy as np
from tqdm import tqdm

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

	def store(self, state, reward, first):
		self.store_state(state)
		self.store_reward(reward)
		self.store_firsts(first)

	def store_prev_states(self, prev_state):
		self.prev_states.append(prev_state)

	def store_state(self, state):
		self.states.append(state)

	def store_actions(self, actions):
		self.actions.append(actions)

	def store_values(self, values):
		self.values.append(values)

	def store_reward(self, reward):
		self.rewards.append(reward)

	def store_firsts(self, first):
		self.firsts.append(first)
		
	def store_k_log_probs(self, k_log_prob):
		self.k_log_probs.append(k_log_prob)

	def store_advantage(self, advantages):
		self.advantages = advantages

	def clear(self):

		self.mean_reward.append(
			torch.sum(self.disc_rewards)/torch.sum(torch.Tensor(self.firsts).float()))

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

		states = torch.Tensor(self.states).reshape(-1, 3, 64, 64)
		actions = torch.cat(self.actions).reshape(-1, 1)
		k_log_probs = torch.cat(self.k_log_probs).reshape(-1, 1)

		disc_rewards = self.disc_rewards
		adv = self.advantages

		return states, actions, k_log_probs, disc_rewards, adv
