import torch
from torch import nn
from torch.distributions import Categorical
from tqdm import tqdm
import numpy as np

class ActorCritic(nn.Module):
	def __init__(self, state_dim, action_dim, n_latent_var):
		super(ActorCritic, self).__init__()

		# actor
		self.action_layer = nn.Sequential(
				nn.Linear(state_dim, n_latent_var),
				nn.Tanh(),
				nn.Linear(n_latent_var, n_latent_var),
				nn.Tanh(),
				nn.Linear(n_latent_var, action_dim),
				nn.Softmax(dim=-1)
				)
		
		# critic
		self.value_layer = nn.Sequential(
				nn.Linear(state_dim, n_latent_var),
				nn.Tanh(),
				nn.Linear(n_latent_var, n_latent_var),
				nn.Tanh(),
				nn.Linear(n_latent_var, 1)
				)
		
	def forward(self):
		raise NotImplementedError
		
	
	def evaluate(self, state, action):
		action_probs = self.action_layer(state)
		dist = Categorical(action_probs)
		
		action_logprobs = dist.log_prob(action)
		dist_entropy = dist.entropy()
		
		state_value = self.value_layer(state)
		
		return action_logprobs, torch.squeeze(state_value), dist_entropy