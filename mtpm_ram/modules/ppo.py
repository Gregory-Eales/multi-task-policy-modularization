import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np
from .memory import Memory
from .actor_critic import ActorCritic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPO:
	def __init__(self, state_dim, action_dim, n_latent_var, lr, gamma, k_epochs, epsilon):
		self.lr = lr
		self.gamma = gamma
		self.eps_clip = epsilon
		self.k_epochs = k_epochs

		self.memory = Memory()

		self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
		self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
		self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
		self.policy_old.load_state_dict(self.policy.state_dict())
		
		self.loss = nn.MSELoss()


	def act(self, state):
		state = torch.from_numpy(state).float().to(device) 
		action_probs = self.policy.action_layer(state)
		dist = Categorical(action_probs)
		action = dist.sample()
		
		self.memory.actions.append(action)
		self.memory.logprobs.append(dist.log_prob(action))
		
		return action.item()


	def store(self, state, reward, done):
		state = torch.from_numpy(state).float().to(device) 
		self.memory.states.append(state)
		self.memory.rewards.append(reward)
		self.memory.is_terminals.append(done)


	def actor_loss(self, logprobs, old_logprobs, rewards, state_values, dist_entropy):

		# Finding the ratio (pi_theta / pi_theta__old):
		ratios = torch.exp(logprobs - old_logprobs.detach())
			
		# Finding Surrogate Loss:
		advantages = rewards - state_values.detach()
		surr1 = ratios * advantages
		surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
		loss = -torch.min(surr1, surr2) - 0.01*dist_entropy

		return loss


	def shuffle(self, rewards, old_states, old_actions, old_logprobs):	
		p = np.random.permutation(rewards.shape[0])
		return rewards[p], old_states[p], old_actions[p], old_logprobs[p]

	
	def update(self):   
		# Monte Carlo estimate of state rewards:
		rewards = []
		discounted_reward = 0
		for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
			if is_terminal:
				discounted_reward = 0
			discounted_reward = reward + (self.gamma * discounted_reward)
			rewards.insert(0, discounted_reward)
		
		# Normalizing the rewards:
		rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
		rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
		
		# convert list to tensor
		old_states = torch.stack(self.memory.states).to(device).detach()
		old_actions = torch.stack(self.memory.actions).to(device).detach()
		old_logprobs = torch.stack(self.memory.logprobs).to(device).detach()


		rewards, old_states, old_actions, old_logprobs = self.shuffle(
			rewards,
			old_states,
			old_actions,
			old_logprobs
			)
		
		btch = 128
		# Optimize policy for K epochs:
		for k in range(self.k_epochs):
			for e in range(old_states.shape[0]//btch):
			
				state_values = self.policy.evaluate_critic(old_states[(e)*btch:(e+1)*btch])
				loss = 0.5*self.loss(state_values, rewards[(e)*btch:(e+1)*btch])
				# take gradient step
				self.optimizer.zero_grad()
				loss.mean().backward()
				self.optimizer.step()

						# Evaluating old actions and values :
				logprobs, dist_entropy = self.policy.evaluate_actor(
					old_states[(e)*btch:(e+1)*btch],
					old_actions[(e)*btch:(e+1)*btch]
					)

				#state_values= self.policy.evaluate_critic(old_states)

				loss = self.actor_loss(
					logprobs,
					old_logprobs[(e)*btch:(e+1)*btch],
					rewards[(e)*btch:(e+1)*btch],
					state_values,
					dist_entropy
					)
				
				self.optimizer.zero_grad()
				loss.mean().backward()
				self.optimizer.step()
		
		# Copy new weights into old policy:
		self.policy_old.load_state_dict(self.policy.state_dict())

		self.memory.clear_memory()


if __name__ == "__main__":

	agent = PPO(3, 3, 32, 0.02, 0.5, 0.99, 100, 0.3)