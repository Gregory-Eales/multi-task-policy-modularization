import torch

from .actor_critic import ActorCritic
from .router import Router


class MTPM(torch.nn.Module):

	def __init__(self):
		
		self.actor_critic = ActorCritic()
		self.critic = Critic() # needs to be a q function
		self.router = Router()
		self.encoder = Encoder()
		self.decoder = Decoder()

	def forward(self, x):

		"""
		z - latent space
		z_r - latent routes
		p - policy
		"""

		out = torch.Tensor(x).to(self.device)

		z = self.encoder(out)

		z_r = self.router(z)

		p = self.actor(z, r)