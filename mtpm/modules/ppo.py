import torch
from tqdm import tqdm
import numpy as np
import random
import gym3


from .actor_critic import ActorCritic
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

		self.buffer = Buffer()

	def act(self, s):

		s = torch.tensor(s).reshape(-1, 3, 64, 64).float()

		pi, v = self.actor.forward(s)

		self.buffer.store_values(v.detach())

		action_probabilities = torch.distributions.Categorical(pi)
		actions = action_probabilities.sample()
		self.buffer.store_actions(actions)

		k_p, k_v = self.k_actor(s)
		k_ap = torch.distributions.Categorical(k_p)
		k_log_prob = k_ap.log_prob(actions.detach())

		self.buffer.store_k_log_probs(k_log_prob.detach())

		return actions.detach().numpy()

	def pi_loss(self, pi, actions, k_log_probs, advantages):

		action_probabilities = torch.distributions.Categorical(pi)
		log_probs = action_probabilities.log_prob(actions)

		r_theta = torch.exp(log_probs-k_log_probs)

		clipped_r = torch.clamp(
			r_theta,
			1.0 - self.epsilon,
			1.0 + self.epsilon
			)

		return torch.mean(torch.min(r_theta*advantages, clipped_r*advantages))

	def discount_rewards(self):

		values = torch.cat(self.buffer.values, dim=1).reshape(-1, 1)
		adv = torch.clone(values)

		firsts = torch.Tensor(self.buffer.firsts).reshape(-1, 1).float()
		r = torch.Tensor(self.buffer.rewards).reshape(-1, 1)


		for i in tqdm(reversed(range(r.shape[0]-1))):
			r[i] = r[i+1]*self.gamma*(1-firsts[i+1]) + (firsts[i+1]*r[i])

			delta = r[i] + self.gamma*adv[i+1] - adv[i]
			adv[i] = (delta + adv[i+1])*(1-firsts[i+1]) + (firsts[i+1]*adv[i])


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
		s, a, k_lp, d_r, adv = self.shuffle(s, a, k_lp, d_r, adv)

		self.transfer_weights()

		num_batches = s.shape[0]//self.batch_sz

		for k in tqdm(range(self.k_epochs)):

			for b in range(num_batches):


				# calculate values and policy
				pi, v = self.actor.forward(s)
				
				pi_loss = self.pi_loss(self,
					pi[b*self.batch_sz:(b+1)*self.batch_sz],
					a[b*self.batch_sz:(b+1)*self.batch_sz],
					k_lp[b*self.batch_sz:(b+1)*self.batch_sz],
					adv[b*self.batch_sz:(b+1)*self.batch_sz]
					)

				v_loss = torch.nn.MSELoss(
					v[b*self.batch_sz:(b+1)*self.batch_sz],
					d_r[b*self.batch_sz:(b+1)*self.batch_sz]
					)

				loss = pi_loss + v_loss

				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

		self.buffer.clear()

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
