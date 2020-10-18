import torch
from tqdm import tqdm
import numpy as np
import random
import gym3

from actor_critic import ActorCritic
from .buffer import Buffer


class RandomAgent():

	def __init__(self, n_envs):

		self.n_envs = n_envs
		self.reward = []

	def act(self, state):
		return np.random.randint(0, high=15, size=[self.n_envs, ])

	def store(self, action, state, reward, prev_state):
		self.reward.append(reward)

	def update(self):
		pass


class PPO(object):

	def __init__(
			self,
			actor_lr,
			critic_lr,
			batch_sz,
			gamma,
			epsilon,
			k_epochs,
	):

		self.actor_lr = actor_lr
		self.critic_lr = critic_lr
		self.batch_sz = batch_sz
		self.gamma = gamma
		self.epsilon = epsilon
		self.k_epochs = k_epochs

		self.actor =  ActorCritic()
		self.k_actor = ActorCritic()
		self.transfer_weights()

		self.buffer = Buffer()

	def act(self, s):

		s = torch.tensor(s).reshape(-1, 3, 64, 64).float()

		prediction = self.actor.forward(s)
		action_probabilities = torch.distributions.Categorical(prediction)
		actions = action_probabilities.sample()
		log_prob = action_probabilities.log_prob(actions)

		k_p = self.k_actor(s)
		k_ap = torch.distributions.Categorical(k_p)
		k_log_prob = k_ap.log_prob(actions.detach())

		self.buffer.store_log_probs(log_prob, k_log_prob.detach())

		return actions.detach().numpy()

	def discount_rewards(self):
		firsts = np.array(self.buffer.firsts).reshape([-1, 1]).astype('int32')
		rewards = np.array(self.buffer.rewards).reshape([-1, 1])
		for i in tqdm(reversed(range(rewards.shape[0]-1))):
			rewards[i] += rewards[i+1]*self.gamma*(1-firsts[i])

		self.buffer.disc_rewards = rewards

	def transfer_weights(self):
		state_dict = self.actor.state_dict()
		self.k_actor.load_state_dict(state_dict)

	def store(self, state, reward, prev_state, first):
		self.buffer.store(state, reward, prev_state, first)

	def calculate_advantages(self, states, prev_states):

		v = self.critic(p_s).detach()
		q = self.critic(s).detach()
		a = (q - v + 1)
		
		return a

	def update(self):

		self.discount_rewards()
	
		s, lp, p_s, k_lp, d_r = self.buffer.get()

		adv = self.calculate_advantages(s, p_s)

		self.transfer_weights()

		for _ in tqdm(range(self.k_epochs)):

			for b in range(num_batches):

				# get 
				
				loss = self.actor_critic.loss(log_probs, k_log_probs, advantages)
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

		self.buffer.clear()

	def get_rewards(self):
		return self.buffer.mean_reward

def main():
	pass

if __name__ == "__main__":
	main()
