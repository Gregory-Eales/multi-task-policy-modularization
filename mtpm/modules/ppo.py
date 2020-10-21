import torch
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
			lr=actor_lr,
			betas=(0.9, 0.999)
			)

		self.buffer = Buffer()

		self.value_loss = torch.nn.MSELoss()

	def act(self, s):



		with torch.no_grad():

			if len(s.shape) > 3:
				s = torch.tensor(s).reshape(-1, 3, 64, 64).float()

			else:
				s = torch.tensor(s).float()

			pi, v = self.k_actor.forward(s)
			self.buffer.store_values(v.detach())
			a_p = torch.distributions.Categorical(pi)
			
			actions = a_p.sample()

			self.buffer.store_actions(actions.float())

			
			k_log_prob = a_p.log_prob(actions.detach())

			self.buffer.store_k_log_probs(k_log_prob.detach())

			return actions.detach().numpy()

	def normalize(self, tensor):
		return (tensor - tensor.mean()/(torch.std(tensor))+1e-3)

	def pi_loss(self, pi, actions, k_log_probs, reward, values, adv):

		e = self.epsilon

		advantages = reward - values.detach()

		a_p = torch.distributions.Categorical(pi)
		log_probs = a_p.log_prob(actions)

		r_theta = torch.exp(log_probs-k_log_probs.detach())

		s1 = r_theta * advantages
		s2 = torch.clamp(r_theta, 1-e, 1+e)*advantages

		pi_loss = -torch.min(s1, s2) - 0.01*a_p.entropy()

		return pi_loss.mean()

	def discount_rewards(self):

		v = torch.stack(self.buffer.values, dim=1).reshape(-1, 1)
		adv = torch.clone(v)

		f = torch.stack(self.buffer.firsts, dim=1).reshape(-1, 1).float()
		r = torch.stack(self.buffer.rewards, dim=1).reshape(-1, 1).float()

		

		"""
		plt.clf()
		plt.plot(r, label="rewards")
		
		#print(f.shape)
		"""
		for i in reversed(range(r.shape[0]-1)):

			

			r[i] = r[i] + (r[i+1]*self.gamma)*(1-f[i+1])

			adv[i] = r[i] + v[i+1] - v[i]

			"""
			delta = r[i] + self.gamma*values[i+1] - values[i]
			adv[i] = (delta + adv[i+1])*(1-firsts[i+1]) + (firsts[i+1]*adv[i])
			"""

		
		
		#plt.plot(f*100)
		#plt.plot(adv, label="advantage")
		#plt.plot(r, label="discounted reward")
		"""
		plt.legend()
		plt.show()
		plt.pause(5)
		"""
		r = self.normalize(r)
		

		self.buffer.disc_rewards = r
		self.buffer.advantages = adv

	def transfer_weights(self):
		state_dict = self.actor.state_dict()
		self.k_actor.load_state_dict(state_dict)

	def store(self, state, reward, first):
		self.buffer.store(state, reward, first)

	def shuffle(self, s, a, k_lp, d_r, adv):	
		p = np.random.permutation(s.shape[0])
		return s[p], a[p], k_lp[p], d_r[p], adv[p]

	def update(self):

		self.discount_rewards()
	
		s, a, k_lp, d_r, adv = self.buffer.get()
		#s, a, k_lp, d_r, adv = self.shuffle(s, a, k_lp, d_r, adv)


		#s = self.normalize(s)

		

		num_batches = s.shape[0]//self.batch_sz

		for k in range(self.k_epochs):


			"""

			for b in range(num_batches):


				# calculate values and policy
				pi, v = self.actor.forward(s[b*self.batch_sz:(b+1)*self.batch_sz])
				
				pi_loss = self.pi_loss(
					pi,
					a[b*self.batch_sz:(b+1)*self.batch_sz],
					k_lp[b*self.batch_sz:(b+1)*self.batch_sz],
					d_r[b*self.batch_sz:(b+1)*self.batch_sz],
					v.detach(),
					adv[b*self.batch_sz:(b+1)*self.batch_sz]
					)


				v_loss = self.value_loss(
					v,
					d_r[b*self.batch_sz:(b+1)*self.batch_sz]
					)

				loss = pi_loss + 0.5*v_loss

				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

			"""

			pi, v = self.actor.forward(s)
				
			pi_loss = self.pi_loss(pi, a, k_lp, d_r, v.detach(), adv)

			v_loss = self.value_loss(v, d_r)

			loss = pi_loss + 0.5*v_loss

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

		self.transfer_weights()
		self.buffer.clear()

		del s
		del a
		del k_lp
		del d_r
		del adv

	def get_rewards(self):
		return self.buffer.mean_reward
		

def main():

	x1 = torch.ones(1, 10)
	x2 = torch.ones(1, 1)

	y = torch.clone(x1)

	y[0][0] = 0

	print(x1)

	#print(torch.cat([x1, x2], dim=1).shape)

if __name__ == "__main__":
	main()
