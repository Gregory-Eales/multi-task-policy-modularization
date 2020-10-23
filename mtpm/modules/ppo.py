import torch
from torch.distributions import Categorical
from tqdm import tqdm
import numpy as np
import random
import gym3
from matplotlib import pyplot as plt
import time


from .actor_critic_small import ActorCritic
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

		self.actor =  ActorCritic(actor_lr=actor_lr, epsilon=epsilon)
		self.k_actor = ActorCritic(actor_lr=actor_lr, epsilon=epsilon)
		self.transfer_weights()

		self.optimizer = torch.optim.Adam(
			params=self.actor.parameters(),
			lr=actor_lr
			)

		self.buffer = Buffer()

		self.value_loss = torch.nn.MSELoss()

	def act(self, s):



		with torch.no_grad():

			if len(s.shape) > 3:
				s = torch.tensor(s).reshape(-1, 3, 64, 64).float()

			else:
				#s = torch.tensor(s).float()
				pass

			pi, v = self.k_actor.forward(s)
			a_p = torch.distributions.Categorical(pi)
			a = a_p.sample()
			l_p = a_p.log_prob(a.detach())

			self.buffer.store_act(a, l_p)

			return a.detach().numpy()

	def normalize(self, tensor):
		return (tensor - tensor.mean()) / ((torch.std(tensor))+1e-5)

	def pi_loss(self, pi, actions, k_log_probs, reward, values):

		e = self.epsilon
		adv = reward - values.detach()
		dist = Categorical(pi)
		log_probs = dist.log_prob(actions.reshape(-1))
		r_theta = torch.exp(log_probs.reshape(-1, 1) - k_log_probs.detach())
		s1 = r_theta * adv
		s2 = torch.clamp(r_theta, 1-e, 1+e)*adv
		pi_loss = -torch.min(s1, s2) - 0.01*dist.entropy().reshape(-1, 1)

		return pi_loss

	def discount_rewards(self):

		f = torch.stack(self.buffer.firsts, dim=1).reshape(-1, 1).float()
		r = torch.stack(self.buffer.rewards, dim=1).reshape(-1, 1).float()

		for i in reversed(range(r.shape[0]-1)):

			r[i] = r[i] + (r[i+1]*self.gamma)*(1-f[i+1])

		self.buffer.disc_rewards = self.normalize(r)
		

	def transfer_weights(self):
		state_dict = self.actor.state_dict()
		self.k_actor.load_state_dict(state_dict)

	def store(self, state, reward, first):
		self.buffer.store(state, reward, first)

	def shuffle(self, s, a, k_lp, d_r):	
		p = np.random.permutation(s.shape[0])
		return s[p], a[p], k_lp[p], d_r[p]

	def update(self):

		self.discount_rewards()
	
		s, a, k_lp, d_r = self.buffer.get()
		s, a, k_lp, d_r= self.shuffle(s, a, k_lp, d_r)

		num_batches = s.shape[0]//self.batch_sz
		sz = self.batch_sz

		for k in range(self.k_epochs):
			
			for b in range(num_batches):
				# calculate values and policy
				pi, v = self.actor.forward(s[b*sz:(b+1)*sz])

				v_loss = self.value_loss(
					v,
					d_r[b*sz:(b+1)*sz]
					)

				pi_loss = self.pi_loss(
					pi,
					a[b*sz:(b+1)*sz],
					k_lp[b*sz:(b+1)*sz],
					d_r[b*sz:(b+1)*sz],
					v.detach()
					)

				loss = (pi_loss + 0.5*v_loss).mean()

				loss.backward()
				self.optimizer.step()
				self.optimizer.zero_grad()

		self.transfer_weights()
		self.buffer.clear()

	def get_rewards(self):
		return self.buffer.mean_reward
		

def main():
	pass

if __name__ == "__main__":
	main()
